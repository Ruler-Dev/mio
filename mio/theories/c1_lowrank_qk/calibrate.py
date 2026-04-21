"""Capture Q/K activations from attention layers during a calibration prefill.

Hooks Qwen3NextAttention.q_proj and k_proj output tensors, stashes them
keyed by (layer_idx, sample_idx). Then runs per-head SVD and reports the
minimum rank at which each head's Frobenius energy crosses a threshold.

Output format (saved via mx.save_safetensors):
  arrays dict:
    f"layer{L}/head{H}/sv"  — singular values, shape (min(seq, d_head),)
    f"layer{L}/head{H}/U"   — left singular vectors, shape (seq, r_full)
    f"layer{L}/head{H}/Vt"  — right singular vectors, shape (r_full, d_head)

Default capture runs one prefill at ctx=4096 on a prompt constructed from
the same shim as the speed harness. Longer / multi-sample captures are
configurable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HeadRank:
    layer: int
    head: int
    d_head: int
    rank_95: int
    rank_98: int
    rank_99: int


@dataclass
class CalibReport:
    sample_count: int
    seq_len: int
    num_attention_layers: int
    num_heads_per_layer: int
    head_ranks: list[HeadRank] = field(default_factory=list)


# --- activation capture ------------------------------------------------------


def _install_capture(target_model: Any, storage: dict) -> Any:
    """Wrap the attention class's __call__ to capture Q/K inputs.

    Python dispatches `attn(x, ...)` through `type(attn).__call__`, so
    instance-level `attn.q_proj.__call__ = ...` has no effect. Instead
    we patch the attention class itself, reading Q and K from
    attn.q_proj(x) and attn.k_proj(x) before delegating to the original
    forward. Extra compute: two linear projections run twice per layer
    per sample — acceptable for a one-shot calibration.

    Returns cleanup callable.
    """
    import mlx.core as mx
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]
    id_to_slot: dict[int, int] = {id(l.self_attn): i for i, l in attn_layers}
    distinct: dict[type, Any] = {}
    for _, l in attn_layers:
        cls = type(l.self_attn)
        if cls not in distinct:
            distinct[cls] = cls.__call__

    def _build_wrapper(original_call):
        def wrapper(self, x, mask=None, cache=None):
            slot_id = id_to_slot.get(id(self), None)
            if slot_id is not None:
                # Capture q_proj/k_proj outputs pre-RoPE/pre-norm. Keep a
                # float32 copy on CPU so we can concatenate across samples
                # without holding the bf16 MLX arrays in GPU memory.
                q_out = self.q_proj(x).astype(mx.float32)
                k_out = self.k_proj(x).astype(mx.float32)
                mx.eval(q_out, k_out)
                import numpy as _np
                slot = storage.setdefault(slot_id, {"Q_list": [], "K_list": []})
                slot["Q_list"].append(_np.array(q_out[0], copy=True))
                slot["K_list"].append(_np.array(k_out[0], copy=True))
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct.items():
        cls.__call__ = _build_wrapper(orig)

    def cleanup() -> None:
        for cls, orig in distinct.items():
            cls.__call__ = orig

    return cleanup


# --- calibration prompt ------------------------------------------------------


_SHIM = """\
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Row:
    key: str
    value: object
    weight: float = 1.0

class Store:
    def __init__(self, cap: int):
        self.cap = cap
        self._data: dict = {}
        self._order: list = []

    def add(self, key: str, value: object) -> None:
        if key in self._data:
            self._data[key].value = value
            self._order.remove(key); self._order.append(key); return
        self._data[key] = Row(key=key, value=value)
        self._order.append(key)
        while len(self._data) > self.cap:
            self._data.pop(self._order.pop(0))

    def query(self, prefix: str) -> list:
        return [r for k, r in self._data.items() if k.startswith(prefix)]
"""


def _calibration_prompt(target_tokens: int, tokenizer) -> str:
    s = _SHIM
    while len(tokenizer.encode(s)) < target_tokens:
        s = s + "\n\n" + _SHIM
    while len(tokenizer.encode(s)) > target_tokens + 200:
        s = s[: int(len(s) * 0.95)]
    return s


# --- SVD + rank analysis -----------------------------------------------------


def _analyze_rank(
    activations: dict[int, dict[str, Any]],
    d_head: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> tuple[list[HeadRank], list[HeadRank]]:
    """Run SVD per (layer, head) on Q and K separately.

    q_proj outputs (L, n_q_heads * d_head * 2) — queries and gate
    interleaved per head; we take only the queries half for Q rank.
    k_proj outputs (L, n_kv_heads * d_head).

    Returns (q_ranks, k_ranks).
    """
    import numpy as np

    q_results: list[HeadRank] = []
    k_results: list[HeadRank] = []

    for i in sorted(activations.keys()):
        slot = activations[i]
        q_list = slot.get("Q_list", [])
        k_list = slot.get("K_list", [])
        if not q_list or not k_list:
            continue

        # Concatenate across samples along the L axis.
        Q_np = np.concatenate(q_list, axis=0)
        K_np = np.concatenate(k_list, axis=0)
        L = Q_np.shape[0]

        # Q: reshape to (L, n_q_heads, d_head*2), take first d_head (queries).
        Q_np = Q_np.reshape(L, n_q_heads, d_head * 2)[..., :d_head]
        K_np = K_np.reshape(L, n_kv_heads, d_head)

        for h in range(n_q_heads):
            s = np.linalg.svd(Q_np[:, h, :], compute_uv=False)
            energy = s ** 2
            total = energy.sum()
            if total <= 0:
                continue
            cum = np.cumsum(energy) / total
            q_results.append(HeadRank(
                layer=i, head=h, d_head=d_head,
                rank_95=int(np.searchsorted(cum, 0.95)) + 1,
                rank_98=int(np.searchsorted(cum, 0.98)) + 1,
                rank_99=int(np.searchsorted(cum, 0.99)) + 1,
            ))
        for h in range(n_kv_heads):
            s = np.linalg.svd(K_np[:, h, :], compute_uv=False)
            energy = s ** 2
            total = energy.sum()
            if total <= 0:
                continue
            cum = np.cumsum(energy) / total
            k_results.append(HeadRank(
                layer=i, head=h, d_head=d_head,
                rank_95=int(np.searchsorted(cum, 0.95)) + 1,
                rank_98=int(np.searchsorted(cum, 0.98)) + 1,
                rank_99=int(np.searchsorted(cum, 0.99)) + 1,
            ))

    return q_results, k_results


# --- main --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="large-moe")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--samples", type=int, default=1,
                   help="Number of independent calibration prompts to run.")
    p.add_argument("--out",
                   default="experiments/c1_calibration/report.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers[args.tier]
    print(f"[c1] loading {args.tier} ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer
    print(f"[c1] loaded.", flush=True)

    storage: dict[int, dict[str, Any]] = {}
    cleanup = _install_capture(engine._target_model, storage)

    try:
        from mio.dflash.runtime import generate_dflash_once, _target_text_model
        text = _target_text_model(engine._target_model)
        # Read d_head from any attention layer.
        attn_layer = next(
            l for l in text.layers if not bool(getattr(l, "is_linear", False))
        )
        d_head = int(attn_layer.self_attn.head_dim)
        n_heads = int(attn_layer.self_attn.num_attention_heads)
        n_kv_heads = int(attn_layer.self_attn.num_key_value_heads)

        for s in range(args.samples):
            # Vary prompt slightly per sample so activations aren't identical.
            base = _calibration_prompt(args.ctx, tok)
            # Prepend sample index to force distinct tokens; adds negligible len.
            prompt = f"# sample {s}\n\n" + base
            messages = [{"role": "user", "content": prompt}]
            prompt_tokens = engine._apply_chat_template(messages)
            print(f"[c1] sample {s}: {len(prompt_tokens)} tokens", flush=True)
            generate_dflash_once(
                target_model=engine._target_model,
                tokenizer=tok,
                draft_model=engine._draft_model,
                prompt="",
                max_new_tokens=0,
                prompt_tokens_override=prompt_tokens,
                tq_bits=engine._resolved_tq_bits(),
                pq_bits=engine._resolved_pq_bits(),
                return_final_state=False,
                prefill_only=True,
            )
    finally:
        cleanup()

    print(f"[c1] captured {len(storage)} attention layers", flush=True)
    q_ranks, k_ranks = _analyze_rank(
        storage, d_head=d_head, n_q_heads=n_heads, n_kv_heads=n_kv_heads,
    )
    print(
        f"[c1] SVD complete; Q={len(q_ranks)} heads, K={len(k_ranks)} heads",
        flush=True,
    )

    # Distributions.
    import statistics as st
    def _dist(label: str, data: list[HeadRank]) -> None:
        if not data:
            print(f"  {label}: (empty)")
            return
        r95 = [h.rank_95 for h in data]
        r98 = [h.rank_98 for h in data]
        r99 = [h.rank_99 for h in data]
        def stats(xs):
            return f"min={min(xs)} median={int(st.median(xs))} p90={sorted(xs)[int(len(xs)*0.9)]} max={max(xs)}"
        print(
            f"  {label}  @95%: {stats(r95)}  @98%: {stats(r98)}  @99%: {stats(r99)}"
            f"   savings(median@98%)={d_head/max(int(st.median(r98)),1):.1f}x"
        )
    print(f"\n-- Rank distribution (d_head={d_head}, {args.samples} sample, ctx~{args.ctx}) --")
    _dist("Q", q_ranks)
    _dist("K", k_ranks)

    out_obj = {
        "tier": args.tier,
        "ctx": args.ctx,
        "samples": args.samples,
        "num_attention_layers": len(storage),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "q_ranks": [
            {"layer": h.layer, "head": h.head, "r95": h.rank_95,
             "r98": h.rank_98, "r99": h.rank_99}
            for h in q_ranks
        ],
        "k_ranks": [
            {"layer": h.layer, "head": h.head, "r95": h.rank_95,
             "r98": h.rank_98, "r99": h.rank_99}
            for h in k_ranks
        ],
    }
    Path(args.out).write_text(json.dumps(out_obj, indent=2))
    print(f"[c1] wrote {args.out}")


if __name__ == "__main__":
    main()
