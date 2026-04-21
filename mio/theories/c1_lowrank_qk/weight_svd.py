"""SVD of W_Q / W_K weight matrices.

If the weight matrices themselves have low rank, MLA-style factorization
(W = A @ B with rank r) gives a speedup that composes cleanly with RoPE
— the rank reduction happens BEFORE the rotation, so RoPE applies to the
low-dim intermediate.

Reports per-layer per-head rank distribution for W_Q (slicing its output
dim into heads, dropping the gate half) and W_K.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import mlx.core as mx


def _svd_energy_rank(mat: np.ndarray, thresholds=(0.95, 0.98, 0.99)) -> dict:
    s = np.linalg.svd(mat, compute_uv=False)
    energy = s ** 2
    total = energy.sum()
    if total <= 0:
        return {f"r{int(t*100)}": -1 for t in thresholds}
    cum = np.cumsum(energy) / total
    out = {}
    for t in thresholds:
        r = int(np.searchsorted(cum, t)) + 1
        out[f"r{int(t*100)}"] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="large-moe")
    p.add_argument("--out",
                   default="experiments/c1_calibration/weight_svd.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import _target_text_model

    cfg = MioConfig.default()
    tc = cfg.tiers[args.tier]
    print(f"[weight-svd] loading {args.tier} ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[weight-svd] loaded.", flush=True)

    text = _target_text_model(engine._target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]
    print(f"[weight-svd] {len(attn_layers)} attention layers", flush=True)

    first_attn = attn_layers[0][1].self_attn
    d_head = int(first_attn.head_dim)
    n_q = int(first_attn.num_attention_heads)
    n_kv = int(first_attn.num_key_value_heads)
    print(f"[weight-svd] d_head={d_head} n_q={n_q} n_kv={n_kv}", flush=True)

    q_ranks_all: list[dict] = []
    k_ranks_all: list[dict] = []

    for i, layer in attn_layers:
        attn = layer.self_attn
        # W_Q shape: (n_q * d_head * 2, d_model)
        W_Q = attn.q_proj.weight
        W_K = attn.k_proj.weight
        # Dequantize if needed.
        # nn.QuantizedLinear stores weight as uint32 with scales/biases.
        # mlx-lm's standard layers expose `.weight` as an mx.array of
        # the original dtype in float. QuantizedLinear.weight is the
        # packed uint8/uint32 — dequant needed for SVD.
        W_Q_f = _dequantize(attn.q_proj).astype(mx.float32)
        W_K_f = _dequantize(attn.k_proj).astype(mx.float32)
        mx.eval(W_Q_f, W_K_f)
        W_Q_np = np.array(W_Q_f, copy=True)
        W_K_np = np.array(W_K_f, copy=True)

        # W_Q: rows are output dims = (n_q heads × d_head × 2).
        # Slice out the query half per head (skip gate).
        W_Q_reshaped = W_Q_np.reshape(n_q, 2, d_head, -1)[:, 0, :, :]
        # Shape now: (n_q, d_head, d_model). Per head: (d_head, d_model).

        W_K_reshaped = W_K_np.reshape(n_kv, d_head, -1)

        for h in range(n_q):
            ranks = _svd_energy_rank(W_Q_reshaped[h])
            ranks.update({"layer": i, "head": h, "kind": "Q"})
            q_ranks_all.append(ranks)
        for h in range(n_kv):
            ranks = _svd_energy_rank(W_K_reshaped[h])
            ranks.update({"layer": i, "head": h, "kind": "K"})
            k_ranks_all.append(ranks)

        if i == attn_layers[0][0]:
            print(f"[weight-svd] layer {i}: "
                  f"W_Q shape={W_Q_np.shape}  W_K shape={W_K_np.shape}",
                  flush=True)

    def _report(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        r95 = [r["r95"] for r in rows]
        r98 = [r["r98"] for r in rows]
        r99 = [r["r99"] for r in rows]
        def _s(xs):
            return f"min={min(xs)} median={int(statistics.median(xs))} p90={sorted(xs)[int(len(xs)*0.9)]} max={max(xs)}"
        print(f"\n  {name} weight SVD across {len(rows)} heads (d_head={d_head}):")
        print(f"    @95%: {_s(r95)}  savings={d_head/max(int(statistics.median(r95)),1):.2f}x")
        print(f"    @98%: {_s(r98)}  savings={d_head/max(int(statistics.median(r98)),1):.2f}x")
        print(f"    @99%: {_s(r99)}  savings={d_head/max(int(statistics.median(r99)),1):.2f}x")

    _report("W_Q (per head, queries only, skip gate)", q_ranks_all)
    _report("W_K (per head)", k_ranks_all)

    Path(args.out).write_text(json.dumps({
        "tier": args.tier,
        "d_head": d_head,
        "n_q": n_q,
        "n_kv": n_kv,
        "q_ranks": q_ranks_all,
        "k_ranks": k_ranks_all,
    }, indent=2))
    print(f"[weight-svd] wrote {args.out}")


def _dequantize(linear_module) -> mx.array:
    """Return the layer's weight as a dense mx.array.

    For QuantizedLinear modules, mlx provides mx.dequantize or the module
    itself has .weight / .scales / .biases. We use mx.dequantize for
    simplicity.
    """
    from mlx.nn.layers.quantized import QuantizedLinear
    if isinstance(linear_module, QuantizedLinear):
        w = mx.dequantize(
            linear_module.weight,
            scales=linear_module.scales,
            biases=linear_module.biases,
            group_size=linear_module.group_size,
            bits=linear_module.bits,
        )
        return w
    return linear_module.weight


if __name__ == "__main__":
    main()
