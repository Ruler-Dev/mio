"""Read-only comparison of the matched upstream and vendored artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark root must be an object: {resolved}")
    return payload


def _point(comparison: dict[str, Any], metric: str) -> float:
    return float(comparison["metrics"][metric]["point_estimate"])


def compare_benchmark_artifacts(
    upstream_path: str | Path,
    vendored_path: str | Path,
) -> dict[str, Any]:
    """Extract comparable diagnostics without pretending the corpora are paired."""

    upstream_file = Path(upstream_path)
    vendored_file = Path(vendored_path)
    upstream = _load(upstream_file)
    vendored = _load(vendored_file)

    comparison = upstream["paired_comparisons"]["dflash-mlx"]
    upstream_runs = [
        run
        for run in upstream["runs"]
        if run.get("status") == "ok" and run.get("mode") == "dflash-mlx"
    ]
    verify_per_cycle = [
        float(run["diagnostics"]["phase_timings_us"]["verify"])
        / int(run["diagnostics"]["cycles_completed"])
        for run in upstream_runs
        if int(run["diagnostics"].get("cycles_completed", 0)) > 0
    ]

    vendored_dflash = vendored["results"]["dflash"]
    vendored_baseline = vendored["results"]["baseline"]
    dflash_aggregate = vendored_dflash["aggregate"]
    baseline_aggregate = vendored_baseline["aggregate"]
    generation_tokens = int(vendored_dflash["repetitions"][0]["generation_tokens"])
    vendored_cycles = generation_tokens / float(dflash_aggregate["median_tokens_per_cycle"])
    vendored_verify_per_cycle = (
        float(dflash_aggregate["median_phase_timings_us"]["verify"]) / vendored_cycles
    )
    upstream_verify_per_cycle = statistics.median(verify_per_cycle)

    upstream_provenance = upstream["provenance"]
    return {
        "schema": {"name": "mio.upstream-dflash-gap", "version": 1},
        "sources": {
            "upstream_matched": {
                "path": str(upstream_file),
                "created_at": upstream["created_at"],
                "git": upstream_provenance["git"],
                "software": upstream_provenance["software"],
                "hardware": upstream_provenance["hardware"],
                "models": upstream_provenance["models"],
                "workload": upstream["configuration"],
            },
            "vendored_matrix": {
                "path": str(vendored_file),
                "created_at": vendored["created_at"],
                "git": {
                    "revision": vendored["git_revision"],
                    "dirty": vendored["git_dirty"],
                },
                "software": vendored["software"],
                "hardware": vendored["hardware"],
                "models": vendored["models"],
                "workload": vendored["parameters"],
            },
        },
        "upstream_matched_result": {
            "eligible_pairs": int(comparison["eligible_pairs"]),
            "distinct_prompts": len(
                {pair["prompt_id"] for pair in comparison["pairs"] if pair["eligible"]}
            ),
            "parity_rate": float(comparison["parity_rate"]),
            "fallback_count": int(comparison["fallback_count"]),
            "strict_passed": bool(upstream["checks"]["strict_passed"]),
            "ttft_speedup": _point(comparison, "ttft_speedup"),
            "decode_speedup": _point(comparison, "decode_speedup"),
            "end_to_end_speedup": _point(comparison, "end_to_end_speedup"),
        },
        "vendored_single_run_result": {
            "parity": bool(vendored_dflash["all_match_baseline"]),
            "cache_commit_modes": list(dflash_aggregate["cache_commit_modes"]),
            "decode_speedup": (
                float(baseline_aggregate["median_decode_us"])
                / float(dflash_aggregate["median_decode_us"])
            ),
            "end_to_end_speedup": (
                float(baseline_aggregate["median_elapsed_us"])
                / float(dflash_aggregate["median_elapsed_us"])
            ),
        },
        "verification_diagnostic": {
            "upstream_median_verify_us_per_cycle": upstream_verify_per_cycle,
            "upstream_runs": len(verify_per_cycle),
            "vendored_verify_us_per_cycle": vendored_verify_per_cycle,
            "vendored_over_upstream_ratio": (
                vendored_verify_per_cycle / upstream_verify_per_cycle
            ),
        },
        "interpretation": {
            "architectural_difference": (
                "Upstream verifies the Qwen hybrid block in batch with TargetOps rollback/tape; "
                "vendored Mio reports timewise_exact_tape and serial exact component paths."
            ),
            "claim_limit": (
                "The two source artifacts use different prompt corpora and schedules.  The "
                "per-cycle ratio is a diagnostic, not a paired speedup or global claim."
            ),
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream")
    parser.add_argument("vendored")
    args = parser.parse_args()
    print(json.dumps(compare_benchmark_artifacts(args.upstream, args.vendored), indent=2))


if __name__ == "__main__":
    main()
