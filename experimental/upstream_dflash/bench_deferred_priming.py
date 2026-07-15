"""Short paired pilot for deferred DFlash prompt-context projection.

The script prints JSON to stdout and never mutates production configuration.
It intentionally uses one loaded upstream target/draft bundle for both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from typing import Any

from experimental.upstream_dflash.deferred_priming import (
    DeferredPrimingStats,
    stream_with_deferred_drafter_priming,
)


PROMPTS = (
    (
        "python-refactor",
        "Refactor this Python function to be clear, deterministic, and linear-time. "
        "Explain the invariants, then return the complete implementation:\n\n"
        "def unique(items):\n    out = []\n    for item in items:\n"
        "        if item not in out:\n            out.append(item)\n    return out",
    ),
    (
        "debugging",
        "A service occasionally writes duplicate ledger entries after a timeout. "
        "Design a debugging plan and a minimal idempotency fix. Include the race "
        "condition, observability signals, tests, and rollback strategy.",
    ),
    (
        "structured-json",
        "Return JSON only. Build a five-step release checklist. Each item must have "
        "the keys id, owner, precondition, action, verification, and rollback.",
    ),
    (
        "copy-heavy",
        "Continue the following repeating sequence for sixteen more rows and then "
        "state the rule:\nalpha,beta,gamma,delta\nalpha,beta,gamma,delta\n"
        "alpha,beta,gamma,delta\nalpha,beta,gamma,delta",
    ),
)


def _config_hash(model_ref: str) -> str | None:
    path = Path(model_ref) / "config.json"
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _encode(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return [
            int(token)
            for token in tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
    return [int(token) for token in tokenizer.encode(prompt)]


def _collect(stream: Any, *, token_event_type: type, summary_event_type: type) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    observed: list[int] = []
    summary = None
    try:
        for event in stream:
            if isinstance(event, token_event_type):
                if first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
                observed.append(int(event.token_id))
            elif isinstance(event, summary_event_type):
                summary = event
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    finished_ns = time.perf_counter_ns()
    if first_token_ns is None or summary is None:
        raise RuntimeError("stream did not emit both a token and a summary")
    token_ids = [int(token) for token in summary.generated_token_ids]
    if observed != token_ids[: len(observed)]:
        raise RuntimeError("stream token events disagree with summary")
    return {
        "token_ids": token_ids,
        "ttft_seconds": (first_token_ns - started_ns) / 1e9,
        "wall_seconds": (finished_ns - started_ns) / 1e9,
        "engine_elapsed_us": float(summary.elapsed_us),
        "prefill_us": float(summary.phase_timings_us.get("prefill", 0.0)),
        "phase_timings_us": dict(summary.phase_timings_us),
        "cycles_completed": int(summary.cycles_completed),
        "acceptance_ratio": float(summary.acceptance_ratio),
    }


def _cluster_bootstrap_interval(
    pairs: list[dict[str, Any]],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | str]:
    clusters: dict[str, list[float]] = {}
    for pair in pairs:
        clusters.setdefault(str(pair["prompt_id"]), []).append(float(pair[metric]))
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        values: list[float] = []
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            values.extend(clusters[cluster_id])
        estimates.append(statistics.median(values))
    estimates.sort()
    lower_index = max(0, int(0.025 * samples))
    upper_index = min(samples - 1, int(0.975 * samples) - 1)
    return {
        "estimand": "median_paired_speedup_cluster_bootstrap_by_prompt",
        "point_estimate": statistics.median(float(pair[metric]) for pair in pairs),
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "confidence_level": 0.95,
        "bootstrap_samples": samples,
        "clusters": len(cluster_ids),
        "pairs": len(pairs),
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    context = build_offline_runtime_context(
        verify_len_cap=args.verify_len_cap,
        verify_mode="dflash",
    )
    bundle = load_runtime_bundle(
        model_ref=args.model,
        draft_ref=args.draft,
        draft_quant=args.draft_quant,
        verify_config=context.verify,
        lazy=True,
    )
    mx.eval(bundle.target_model.parameters(), bundle.draft_model.parameters())
    stop_ids = get_stop_token_ids(bundle.tokenizer)
    encoded = [
        (prompt_id, _encode(bundle.tokenizer, prompt, use_chat_template=args.chat_template))
        for prompt_id, prompt in PROMPTS
    ]

    def factory(prompt_ids: list[int], max_tokens: int) -> Any:
        return stream_dflash_generate(
            target_model=bundle.target_model,
            target_ops=bundle.target_ops,
            tokenizer=bundle.tokenizer,
            draft_model=bundle.draft_model,
            draft_backend=bundle.draft_backend,
            prompt="",
            max_new_tokens=max_tokens,
            block_tokens=args.block_tokens,
            stop_token_ids=stop_ids,
            prompt_tokens_override=prompt_ids,
            publish_generation_snapshot=False,
            runtime_context=context,
        )

    # Compile/warm both arms before any measurement.
    _collect(
        factory(encoded[0][1], min(8, args.max_tokens)),
        token_event_type=TokenEvent,
        summary_event_type=SummaryEvent,
    )
    warm_stats = DeferredPrimingStats()
    _collect(
        stream_with_deferred_drafter_priming(
            lambda: factory(encoded[0][1], min(8, args.max_tokens)),
            stats=warm_stats,
            max_prompt_tokens=args.max_deferred_prompt_tokens,
            fuse_cold_prefill=args.fuse_cold_prefill,
        ),
        token_event_type=TokenEvent,
        summary_event_type=SummaryEvent,
    )

    runs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for prompt_index, (prompt_id, prompt_ids) in enumerate(encoded):
            order = ("standard", "deferred") if (repetition + prompt_index) % 2 else ("deferred", "standard")
            by_mode: dict[str, dict[str, Any]] = {}
            for mode in order:
                mx.random.seed(args.seed + repetition * 100 + prompt_index)
                stats = DeferredPrimingStats()
                raw = factory(prompt_ids, args.max_tokens)
                stream = (
                    stream_with_deferred_drafter_priming(
                        lambda raw=raw: raw,
                        stats=stats,
                        max_prompt_tokens=args.max_deferred_prompt_tokens,
                        fuse_cold_prefill=args.fuse_cold_prefill,
                    )
                    if mode == "deferred"
                    else raw
                )
                result = _collect(
                    stream,
                    token_event_type=TokenEvent,
                    summary_event_type=SummaryEvent,
                )
                result.update(
                    {
                        "mode": mode,
                        "prompt_id": prompt_id,
                        "prompt_tokens": len(prompt_ids),
                        "repetition": repetition,
                        "deferred_stats": vars(stats) if mode == "deferred" else None,
                    }
                )
                runs.append(result)
                by_mode[mode] = result
            standard = by_mode["standard"]
            deferred = by_mode["deferred"]
            pairs.append(
                {
                    "prompt_id": prompt_id,
                    "repetition": repetition,
                    "token_parity": standard["token_ids"] == deferred["token_ids"],
                    "ttft_speedup": standard["ttft_seconds"] / deferred["ttft_seconds"],
                    "end_to_end_speedup": standard["wall_seconds"] / deferred["wall_seconds"],
                }
            )

    return {
        "schema": {"name": "mio.deferred-drafter-priming-pilot", "version": 1},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git": {
                "revision": _git_value("rev-parse", "HEAD"),
                "dirty": bool(_git_value("status", "--porcelain")),
            },
            "software": {
                name: importlib.metadata.version(name)
                for name in ("mlx", "mlx-lm", "dflash-mlx")
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "models": {
                "target": args.model,
                "target_config_sha256": _config_hash(args.model),
                "draft": args.draft,
                "draft_config_sha256": _config_hash(args.draft),
            },
        },
        "configuration": {
            "max_tokens": args.max_tokens,
            "repetitions": args.repetitions,
            "block_tokens": args.block_tokens,
            "verify_len_cap": args.verify_len_cap,
            "draft_quant": args.draft_quant,
            "max_deferred_prompt_tokens": args.max_deferred_prompt_tokens,
            "balanced_order": True,
            "warmup_each_arm": True,
            "chat_template": args.chat_template,
            "fuse_cold_prefill": args.fuse_cold_prefill,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "runs": runs,
        "pairs": pairs,
        "summary": {
            "pairs": len(pairs),
            "parity_rate": sum(pair["token_parity"] for pair in pairs) / len(pairs),
            "median_ttft_speedup": statistics.median(pair["ttft_speedup"] for pair in pairs),
            "median_end_to_end_speedup": statistics.median(
                pair["end_to_end_speedup"] for pair in pairs
            ),
            "ttft_speedup_ci95": _cluster_bootstrap_interval(
                pairs,
                "ttft_speedup",
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "end_to_end_speedup_ci95": _cluster_bootstrap_interval(
                pairs,
                "end_to_end_speedup",
                samples=args.bootstrap_samples,
                seed=args.seed + 1,
            ),
        },
        "claim_scope": "short_single_model_single_machine_pilot_not_promotion_evidence",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3.6-27B-UD-Q4_K_XL-mlx")
    parser.add_argument("--draft", default="spd/Qwen3.6-27B-DFlash")
    parser.add_argument("--draft-quant", default="w4:gs64")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--verify-len-cap", type=int, default=16)
    parser.add_argument("--max-deferred-prompt-tokens", type=int, default=512)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-cold-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_pilot(parse_args()), indent=2))
