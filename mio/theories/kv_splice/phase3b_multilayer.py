"""Phase 3b — multi-layer splice K-sweep.

Same E2E test as Phase 3 but sweeps the set of spliced layers:
  K=[3]           single-layer (already shown to work)
  K=[3,7]         two-layer
  K=[3,7,11]      three-layer
  K=[3,7,11,15]   four-layer

At each K, generate and measure semantic-match lcp vs fresh baseline.
Find the largest K where output quality is acceptable (lcp >= 0.3).

This tells us how many attention layers we can splice before errors
compound past the model's tolerance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mio.theories.kv_splice.phase3_end_to_end import (
    _CHUNK, _WRAPPER_A, _WRAPPER_B, _SUFFIX, _QUESTION,
    _sha, _lcp, _tokenize_full,
)


def _capture_multi_kv(
    target_model: Any,
    chunk_start: int,
    chunk_end: int,
    attn_layer_indices: list[int],
) -> tuple[Any, dict[int, dict]]:
    """Capture K_base + V at chunk positions for the given attention layer indices.

    attn_layer_indices: ABSOLUTE decoder layer indices (e.g., 3, 7, 11, 15 for
    Qwen3.6-A3B where attention happens every 4 layers starting at L3).

    Returns (cleanup, storage) where storage[layer_idx] = {"k_base": mx.array,
    "v": mx.array}.
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_instances = {
        i: text.layers[i].self_attn for i in attn_layer_indices
        if i < len(text.layers) and not bool(getattr(text.layers[i], "is_linear", False))
    }
    id_to_layer_idx = {id(attn): i for i, attn in attn_instances.items()}

    distinct_cls = {}
    for attn in attn_instances.values():
        cls = type(attn)
        if cls not in distinct_cls:
            distinct_cls[cls] = cls.__call__

    storage: dict[int, dict] = {}
    for i in attn_instances:
        storage[i] = {"k_base": None, "v": None}

    def _wrap(original_call):
        def wrapper(self, x, mask=None, cache=None):
            i = id_to_layer_idx.get(id(self))
            if i is not None:
                B, L, _ = x.shape
                n_kv = int(self.num_key_value_heads)
                d_h = int(self.head_dim)
                k = self.k_proj(x).reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
                v = self.v_proj(x).reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
                k_chunk = k[:, :, chunk_start:chunk_end, :]
                v_chunk = v[:, :, chunk_start:chunk_end, :]
                mx.eval(k_chunk, v_chunk)
                storage[i]["k_base"] = mx.array(k_chunk[0])
                storage[i]["v"] = mx.array(v_chunk[0])
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct_cls.items():
        cls.__call__ = _wrap(orig)

    def cleanup() -> None:
        for cls, orig in distinct_cls.items():
            cls.__call__ = orig

    return cleanup, storage


def _install_multi_splice(
    target_model: Any,
    chunk_start: int,
    chunk_end: int,
    spliced_data: dict[int, dict],
) -> Any:
    """For each layer in spliced_data, replace k_proj/v_proj output at
    chunk positions with the spliced values.
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_instances = {
        i: text.layers[i].self_attn for i in spliced_data
        if i < len(text.layers) and not bool(getattr(text.layers[i], "is_linear", False))
    }
    id_to_layer_idx = {id(attn): i for i, attn in attn_instances.items()}

    distinct_attn_cls = {}
    for attn in attn_instances.values():
        cls = type(attn)
        if cls not in distinct_attn_cls:
            distinct_attn_cls[cls] = cls.__call__

    # k_proj / v_proj share a class, but we want to trigger splicing only
    # when called from one of our target attention layers. Use a flag keyed
    # by current active attention layer.
    active_idx = {"i": None}

    # Capture original k_proj and v_proj __call__ by class (they're all nn.Linear).
    linear_cls = type(list(attn_instances.values())[0].k_proj)
    orig_linear_call = linear_cls.__call__

    k_proj_ids = {id(attn.k_proj): i for i, attn in attn_instances.items()}
    v_proj_ids = {id(attn.v_proj): i for i, attn in attn_instances.items()}

    def linear_wrap(self, x):
        y = orig_linear_call(self, x)
        # Only splice if we're inside an active target attention layer's call
        # AND this Linear is the k_proj or v_proj of that attention.
        if active_idx["i"] is None:
            return y
        current_layer = active_idx["i"]
        if id(self) == id(attn_instances[current_layer].k_proj):
            field = "k_base"
        elif id(self) == id(attn_instances[current_layer].v_proj):
            field = "v"
        else:
            return y
        B, L, total = y.shape
        if L < chunk_end:
            return y
        attn = attn_instances[current_layer]
        n_kv = int(attn.num_key_value_heads)
        d_h = int(attn.head_dim)
        y_r = y.reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
        chunk_len = chunk_end - chunk_start
        spliced = mx.broadcast_to(
            spliced_data[current_layer][field][None, :, :, :],
            (B, n_kv, chunk_len, d_h),
        ).astype(y_r.dtype)
        pre = y_r[:, :, :chunk_start, :]
        post = y_r[:, :, chunk_end:, :]
        y_r_new = mx.concatenate([pre, spliced, post], axis=2)
        return y_r_new.transpose(0, 2, 1, 3).reshape(B, L, total)

    def attn_wrap(original_call):
        def wrapper(self, x, mask=None, cache=None):
            i = id_to_layer_idx.get(id(self))
            if i is not None:
                active_idx["i"] = i
                try:
                    return original_call(self, x, mask=mask, cache=cache)
                finally:
                    active_idx["i"] = None
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct_attn_cls.items():
        cls.__call__ = attn_wrap(orig)
    linear_cls.__call__ = linear_wrap

    def cleanup() -> None:
        for cls, orig in distinct_attn_cls.items():
            cls.__call__ = orig
        linear_cls.__call__ = orig_linear_call

    return cleanup


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument(
        "--layer-sets", nargs="+",
        default=["3", "3,7", "3,7,11", "3,7,11,15", "3,7,11,15,19"],
        help="comma-separated layer-index lists to sweep",
    )
    p.add_argument("--out", default="experiments/kv_splice/phase3b_multilayer.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import generate_dflash_once

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[phase3b] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer

    src_full, src_cs, src_ce = _tokenize_full(engine, _WRAPPER_A, _CHUNK, "")
    tgt_full, tgt_cs, tgt_ce = _tokenize_full(engine, _WRAPPER_B, _CHUNK, _QUESTION)
    L_chunk = min(src_ce - src_cs, tgt_ce - tgt_cs)
    src_ce = src_cs + L_chunk
    tgt_ce = tgt_cs + L_chunk
    print(f"[phase3b] source chunk [{src_cs}..{src_ce}] target [{tgt_cs}..{tgt_ce}] "
          f"len={L_chunk}", flush=True)

    # Capture source KVs for the UNION of all layers we'll test.
    all_layers = sorted({
        int(li) for spec in args.layer_sets for li in spec.split(",")
    })
    print(f"[phase3b] source-capture layers: {all_layers}", flush=True)
    cleanup_src, src_storage = _capture_multi_kv(
        engine._target_model, src_cs, src_ce, all_layers,
    )
    try:
        generate_dflash_once(
            target_model=engine._target_model,
            tokenizer=tok, draft_model=engine._draft_model,
            prompt="", max_new_tokens=0,
            prompt_tokens_override=src_full,
            tq_bits=engine._resolved_tq_bits(), pq_bits=engine._resolved_pq_bits(),
            return_final_state=False, prefill_only=True,
        )
    finally:
        cleanup_src()

    # Target FRESH baseline.
    print(f"\n[phase3b] === target FRESH ===", flush=True)
    full_text = _WRAPPER_B + _SUFFIX + _CHUNK + _QUESTION
    engine._prefix_cache_invalidate()
    text_fresh, m_fresh = engine.generate(
        messages=[{"role": "user", "content": full_text}],
        max_tokens=args.gen_tokens,
    )
    fresh_sha = _sha(text_fresh)
    print(f"  sha={fresh_sha}  prefill={m_fresh.prompt_tps:.0f}t/s  gen={m_fresh.generation_tps:.1f}t/s",
          flush=True)

    # Per-layer-set sweep.
    results: dict[str, dict] = {}
    for spec in args.layer_sets:
        layer_set = [int(li) for li in spec.split(",")]
        print(f"\n[phase3b] === splice layers {layer_set} ===", flush=True)
        spliced_data = {li: src_storage[li] for li in layer_set}
        cleanup = _install_multi_splice(
            engine._target_model, tgt_cs, tgt_ce, spliced_data,
        )
        try:
            engine._prefix_cache_invalidate()
            text_splice, m_splice = engine.generate(
                messages=[{"role": "user", "content": full_text}],
                max_tokens=args.gen_tokens,
            )
        finally:
            cleanup()
        sha = _sha(text_splice)
        lcp = _lcp(text_splice, text_fresh)
        lcp_frac = lcp / max(len(text_fresh), 1)
        verdict = (
            "SHA MATCH" if sha == fresh_sha
            else ("NEAR" if lcp_frac >= 0.8
                  else ("PARTIAL" if lcp_frac >= 0.2 else "BROKEN"))
        )
        results[spec] = {
            "layers": layer_set,
            "sha": sha,
            "prefill_tps": m_splice.prompt_tps,
            "gen_tps": m_splice.generation_tps,
            "lcp": lcp,
            "lcp_fraction": lcp_frac,
            "verdict": verdict,
            "text_head": text_splice[:300],
        }
        print(f"  sha={sha}  lcp={lcp}/{len(text_fresh)}  frac={lcp_frac:.3f}  "
              f"verdict={verdict}",
              flush=True)
        print(f"  prefill={m_splice.prompt_tps:.0f}t/s  gen={m_splice.generation_tps:.1f}t/s",
              flush=True)
        print(f"  head: {text_splice[:200]!r}", flush=True)

    # Summary
    print(f"\n[phase3b] === SUMMARY ===", flush=True)
    print(f"  fresh:   prefill={m_fresh.prompt_tps:.0f}t/s  gen={m_fresh.generation_tps:.1f}t/s  "
          f"sha={fresh_sha}",
          flush=True)
    print(f"  {'layers':>20}  {'lcp_frac':>8}  {'verdict':>10}  "
          f"{'prefill':>9}  {'gen':>6}")
    for spec, r in results.items():
        print(f"  {spec:>20}  {r['lcp_fraction']:>8.3f}  {r['verdict']:>10}  "
              f"{r['prefill_tps']:>9.0f}  {r['gen_tps']:>6.1f}",
              flush=True)

    Path(args.out).write_text(json.dumps({
        "fresh": {
            "sha": fresh_sha, "text_head": text_fresh[:400],
            "prefill_tps": m_fresh.prompt_tps, "gen_tps": m_fresh.generation_tps,
        },
        "splice_results": results,
    }, indent=2))
    print(f"\n[phase3b] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
