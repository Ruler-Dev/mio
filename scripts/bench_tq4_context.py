"""TQ4 vs baseline KV-cache benchmark across tiers at native context.

Runs prefill + fixed decode for each tier under two conditions:
  - baseline: mlx_lm.cache.KVCache (fp16/bf16)
  - tq4: TurboQuantKVCacheV2(bits=4, group_size=64)

Target model only (no draft) to isolate KV-cache effects. Uses native
context window per tier (large-moe 128K, large 32K, medium 16K, small 8K).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from mio.config import MioConfig
from mio.dflash.runtime import (
    configure_full_attention_split,
    load_target_bundle,
    _target_text_model,
)
from mio.turboquant import patch as tq_patch
from mio.turboquant.cache_v2 import TurboQuantKVCacheV2


@dataclass
class BenchResult:
    tier: str
    mode: str
    prompt_tokens: int
    decode_tokens: int
    prefill_s: float
    decode_s: float
    prompt_tps: float
    decode_tps: float
    peak_mem_gb: float
    kv_bytes: int

    def row(self) -> str:
        return (
            f"{self.tier:>10s} {self.mode:>8s} "
            f"prompt={self.prompt_tokens:>6d} prefill={self.prefill_s:>6.2f}s "
            f"prompt_tps={self.prompt_tps:>8.1f} "
            f"decode_tps={self.decode_tps:>7.2f} "
            f"peak={self.peak_mem_gb:>5.1f}GB "
            f"kv={self.kv_bytes/1e9:>5.2f}GB"
        )


def _detect_head_dim(target_model) -> int:
    text = _target_text_model(target_model)
    for layer in text.layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for attr in ("head_dim", "scale"):
                v = getattr(attn, attr, None)
                if attr == "head_dim" and isinstance(v, int):
                    return v
            q_proj = getattr(attn, "q_proj", None)
            n_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_heads", None)
            if q_proj is not None and n_heads:
                out_features = q_proj.weight.shape[0]
                return out_features // n_heads
    raise RuntimeError("Could not detect head_dim")


def _make_caches(target_model, mode: str, head_dim: int) -> list:
    """Build per-layer cache list. Replace full-attention KVCache with TQ4 when requested."""
    text = _target_text_model(target_model)
    caches: list = []
    for layer in text.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            caches.append(cache_mod.ArraysCache(size=2))
        else:
            if mode == "tq4":
                caches.append(
                    TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=4, group_size=64, use_qjl=False,
                    )
                )
            else:
                caches.append(cache_mod.KVCache())
    return caches


def _kv_bytes(caches) -> int:
    total = 0
    for c in caches:
        if isinstance(c, TurboQuantKVCacheV2):
            total += int(c.nbytes)
        else:
            keys = getattr(c, "keys", None)
            values = getattr(c, "values", None)
            if keys is not None:
                total += int(keys.nbytes)
            if values is not None:
                total += int(values.nbytes)
    return total


def _make_prompt_tokens(tokenizer, target_len: int) -> list[int]:
    """Repeat-tile a fixed passage to reach target_len tokens (truncate)."""
    seed_text = (
        "In distributed systems, consensus algorithms like Raft and Paxos "
        "guarantee safety under asynchronous networks with bounded failures. "
        "Cache coherence, replication factor, and leader election interact in "
        "subtle ways that determine both latency and correctness. "
    )
    ids = list(tokenizer.encode(seed_text))
    if not ids:
        ids = [1]
    out: list[int] = []
    while len(out) < target_len:
        out.extend(ids)
    return out[:target_len]


def bench_condition(
    tier_name: str,
    target_model,
    tokenizer,
    mode: str,
    prompt_len: int,
    decode_tokens: int,
) -> BenchResult:
    """Run prefill + decode once and return metrics."""
    head_dim = _detect_head_dim(target_model)

    if mode == "tq4":
        tq_patch.apply()
        # Disable dflash split-SDPA for TQ path (falls back to patched SDPA)
        try:
            configure_full_attention_split(target_model, enabled=False)
        except Exception:
            pass
    else:
        tq_patch.revert()

    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

    caches = _make_caches(target_model, mode, head_dim)
    prompt_ids = _make_prompt_tokens(tokenizer, prompt_len)
    prompt_array = mx.array(prompt_ids, dtype=mx.uint32)[None]

    # Chunked prefill — avoids Metal single-buffer cap (~30 GB) on full-attn intermediate.
    # Chunk size picked so chunk × prompt_len × n_heads × 4B stays below 8 GB.
    CHUNK = 2048
    t0 = time.perf_counter()
    last_logits = None
    for start in range(0, prompt_len, CHUNK):
        end = min(start + CHUNK, prompt_len)
        piece = prompt_array[:, start:end]
        last_logits = target_model(piece, cache=caches)
        mx.eval(last_logits)
    prefill_s = time.perf_counter() - t0
    logits = last_logits
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

    # Decode N tokens
    t1 = time.perf_counter()
    generated = [next_token]
    for _ in range(decode_tokens - 1):
        tok = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(tok, cache=caches)
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_token)
    # Final sync
    mx.eval(mx.array(generated))
    decode_s = time.perf_counter() - t1

    peak_gb = (
        float(mx.get_peak_memory()) / 1e9
        if hasattr(mx, "get_peak_memory") else 0.0
    )
    kv = _kv_bytes(caches)

    result = BenchResult(
        tier=tier_name,
        mode=mode,
        prompt_tokens=prompt_len,
        decode_tokens=len(generated),
        prefill_s=prefill_s,
        decode_s=decode_s,
        prompt_tps=prompt_len / max(prefill_s, 1e-9),
        decode_tps=len(generated) / max(decode_s, 1e-9),
        peak_mem_gb=peak_gb,
        kv_bytes=kv,
    )
    del caches, prompt_array
    gc.collect()
    return result


def run_tier(tier_name: str, prompt_len: int, decode_tokens: int) -> list[BenchResult]:
    cfg = MioConfig.default()
    tier = cfg.tiers[tier_name]
    print(f"\n[bench] tier={tier_name} model={tier.target_model} prompt_len={prompt_len}", flush=True)

    t0 = time.perf_counter()
    model, tokenizer, _meta = load_target_bundle(
        tier.target_model, lazy=True, split_full_attention_sdpa=False, quantize_kv_cache=False,
    )
    mx.eval(model.parameters())
    print(f"[bench] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    results: list[BenchResult] = []
    # Baseline first, then TQ4 (so tq patch doesn't leak into baseline)
    for mode in ("baseline", "tq4"):
        print(f"[bench] running {mode}...", flush=True)
        try:
            r = bench_condition(tier_name, model, tokenizer, mode, prompt_len, decode_tokens)
            print("  " + r.row(), flush=True)
            results.append(r)
        except Exception as e:
            print(f"  FAILED {mode}: {e}", flush=True)
    tq_patch.revert()

    # Unload
    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
        except Exception:
            pass
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiers", default="small,medium,large,large-moe",
        help="Comma-separated tiers to bench",
    )
    parser.add_argument("--decode", type=int, default=64,
                        help="Decode tokens per run (default 64 to keep wall time bounded)")
    parser.add_argument("--headroom", type=int, default=512,
                        help="Tokens subtracted from native context window")
    parser.add_argument("--cap", type=int, default=0,
                        help="If >0, cap prompt length across all tiers (for quick smoke test)")
    args = parser.parse_args()

    cfg = MioConfig.default()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    all_results: list[BenchResult] = []
    for tier_name in tiers:
        if tier_name not in cfg.tiers:
            print(f"[bench] unknown tier: {tier_name}", flush=True)
            continue
        ctx = cfg.tiers[tier_name].context_window - args.headroom
        if args.cap:
            ctx = min(ctx, args.cap)
        try:
            all_results.extend(run_tier(tier_name, ctx, args.decode))
        except Exception as e:
            print(f"[bench] tier {tier_name} failed: {e}", flush=True)

    print("\n" + "=" * 80)
    print(f"{'tier':>10s} {'mode':>8s} {'prompt':>12s} {'prefill':>13s} "
          f"{'prompt_tps':>12s} {'decode_tps':>11s} {'peak':>7s} {'kv':>8s}")
    print("-" * 80)
    for r in all_results:
        print(r.row())
    print("=" * 80)

    # Pair and show ratios
    print("\nDelta TQ4 vs baseline (decode_tps ratio, kv bytes ratio):")
    grouped: dict[str, dict[str, BenchResult]] = {}
    for r in all_results:
        grouped.setdefault(r.tier, {})[r.mode] = r
    for tier_name, modes in grouped.items():
        b = modes.get("baseline")
        t = modes.get("tq4")
        if b and t:
            dec_ratio = t.decode_tps / b.decode_tps if b.decode_tps else 0
            kv_ratio = t.kv_bytes / b.kv_bytes if b.kv_bytes else 0
            print(f"  {tier_name:>10s}  decode {dec_ratio:5.2f}x  kv {kv_ratio:5.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
