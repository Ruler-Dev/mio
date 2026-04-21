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
    """Hook attention layers' q_proj / k_proj forward to capture outputs.

    Stores per (layer_idx) the concatenated Q (1, L, n_heads * d_head)
    and K (1, L, n_kv_heads * d_head) for the most recent forward.
    Returns cleanup callable.
    """
    import mlx.core as mx
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]

    originals: dict[int, tuple[Any, Any]] = {}

    def make_hooked(orig_fn, slot: dict, key: str):
        def call(x):
            y = orig_fn(x)
            mx.eval(y)
            # Copy out reference; mx arrays are immutable so this is cheap.
            slot[key] = y
            return y
        return call

    # attn.q_proj and attn.k_proj are nn.Linear instances; replacing their
    # __call__ cleanly requires patching the linear's method.
    for (i, layer) in attn_layers:
        attn = layer.self_attn
        qp = attn.q_proj
        kp = attn.k_proj
        slot = storage.setdefault(i, {})
        originals[i] = (qp.__call__, kp.__call__)
        qp.__call__ = make_hooked(qp.__call__, slot, "Q")
        kp.__call__ = make_hooked(kp.__call__, slot, "K")

    def cleanup() -> None:
        for (i, layer) in attn_layers:
            orig_q, orig_k = originals[i]
            layer.self_attn.q_proj.__call__ = orig_q
            layer.self_attn.k_proj.__call__ = orig_k

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


def _analyze_rank(activations: dict[int, dict[str, Any]], d_head: int) -> list[HeadRank]:
    """Run SVD per head. Report rank at 95/98/99% Frobenius energy retention."""
    import mlx.core as mx
    import numpy as np

    results: list[HeadRank] = []
    layers = sorted(activations.keys())
    for i in layers:
        slot = activations[i]
        Q = slot.get("Q")  # (1, L, n_heads * d_head)
        K = slot.get("K")
        if Q is None or K is None:
            continue
        _, L, qdim = Q.shape
        n_heads = qdim // d_head
        Q_np = np.asarray(Q[0], dtype=np.float32).reshape(L, n_heads, d_head)
        for h in range(n_heads):
            Qh = Q_np[:, h, :]  # (L, d_head)
            # SVD
            s = np.linalg.svd(Qh, compute_uv=False)
            # Cumulative Frobenius² energy.
            energy = (s ** 2)
            total = energy.sum()
            if total <= 0:
                continue
            cum = np.cumsum(energy) / total
            r95 = int(np.searchsorted(cum, 0.95)) + 1
            r98 = int(np.searchsorted(cum, 0.98)) + 1
            r99 = int(np.searchsorted(cum, 0.99)) + 1
            results.append(HeadRank(
                layer=i, head=h, d_head=d_head,
                rank_95=r95, rank_98=r98, rank_99=r99,
            ))
    return results


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
    head_ranks = _analyze_rank(storage, d_head=d_head)
    print(f"[c1] SVD complete; {len(head_ranks)} (layer, head) pairs", flush=True)

    # Summarize.
    report = CalibReport(
        sample_count=args.samples,
        seq_len=args.ctx,
        num_attention_layers=len(storage),
        num_heads_per_layer=n_heads,
        head_ranks=head_ranks,
    )

    # Distributions.
    import statistics as st
    r95s = [h.rank_95 for h in head_ranks]
    r98s = [h.rank_98 for h in head_ranks]
    r99s = [h.rank_99 for h in head_ranks]
    print(f"\n-- Rank distribution across {len(head_ranks)} heads --")
    for name, data in (("95%", r95s), ("98%", r98s), ("99%", r99s)):
        if data:
            print(
                f"  energy@{name}: min={min(data)} median={int(st.median(data))} "
                f"p90={sorted(data)[int(len(data)*0.9)]} max={max(data)}"
                f"   d_head={d_head}  saving@median_98={d_head/max(int(st.median(r98s)),1):.1f}x",
            )

    # Write JSON
    out_obj = {
        "tier": args.tier,
        "ctx": args.ctx,
        "samples": args.samples,
        "num_attention_layers": len(storage),
        "n_heads": n_heads,
        "d_head": d_head,
        "head_ranks": [
            {"layer": h.layer, "head": h.head, "r95": h.rank_95,
             "r98": h.rank_98, "r99": h.rank_99}
            for h in head_ranks
        ],
    }
    Path(args.out).write_text(json.dumps(out_obj, indent=2))
    print(f"[c1] wrote {args.out}")


if __name__ == "__main__":
    main()
