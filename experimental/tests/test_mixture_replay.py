from __future__ import annotations

from pathlib import Path

import pytest

from experimental.mixture.replay import load_cases, replay_benchmark


ROOT = Path(__file__).resolve().parents[2]
CURRENT_MATCHED_BENCHMARK = ROOT / "benchmarks" / "results" / "speculative-matched-qwen3-4b-20260715-v041.json"
QWEN36_27B_MATCHED_BENCHMARK = ROOT / "benchmarks" / "results" / "speculative-matched-qwen36-27b-20260715.json"


@pytest.mark.skipif(
    not CURRENT_MATCHED_BENCHMARK.exists(),
    reason="matched DSpark/DFlash benchmark is not present",
)
def test_current_replay_converges_to_dspark_without_claiming_a_gain() -> None:
    report = replay_benchmark(CURRENT_MATCHED_BENCHMARK)

    assert report["source"]["cases"] == 12
    assert report["online_protocol"]["observations_consumed"] == 12
    assert report["online_protocol"]["counterfactual_arms_used_for_online_updates"] == 0
    assert report["routing"]["selection_counts"] == {
        "dflash-mlx": 1,
        "mlx-dspark": 11,
    }
    assert report["routing"]["converged_arm"] == "mlx-dspark"
    assert report["routing"]["final_online_best_static"] == "mlx-dspark"
    assert report["offline_comparators"]["best_arm"] == "mlx-dspark"
    assert report["offline_comparators"]["per_request_wins"] == {"mlx-dspark": 12}
    assert report["offline_comparators"]["best_speedup_over_other"] == pytest.approx(1.4257028369647835)
    assert report["measurements"]["router_speedup_vs_best_static"] < 1.0
    assert report["measurements"]["selected_parity_rate"] < 1.0
    assert report["claim"]["same_corpus_speed_candidate"] is False
    assert report["claim"]["workload_candidate"] is False
    assert report["claim"]["global_breakthrough"] is False


@pytest.mark.skipif(
    not CURRENT_MATCHED_BENCHMARK.exists(),
    reason="matched DSpark/DFlash benchmark is not present",
)
def test_replay_loader_exposes_complete_matched_cases() -> None:
    _, cases = load_cases(CURRENT_MATCHED_BENCHMARK)

    assert len(cases) == 12
    assert all(set(case.runs) == {"dflash-mlx", "mlx-dspark"} for case in cases)
    assert all(case.baseline is not None for case in cases)


@pytest.mark.skipif(
    not QWEN36_27B_MATCHED_BENCHMARK.exists(),
    reason="matched Qwen3.6 27B DSpark/DFlash benchmark is not present",
)
def test_27b_replay_converges_to_dflash_without_claiming_a_gain() -> None:
    report = replay_benchmark(QWEN36_27B_MATCHED_BENCHMARK)

    assert report["source"]["cases"] == 12
    assert report["online_protocol"]["observations_consumed"] == 12
    assert report["online_protocol"]["counterfactual_arms_used_for_online_updates"] == 0
    assert report["routing"]["selection_counts"] == {
        "dflash-mlx": 11,
        "mlx-dspark": 1,
    }
    assert report["routing"]["converged_arm"] == "dflash-mlx"
    assert report["routing"]["final_online_best_static"] == "dflash-mlx"
    assert report["offline_comparators"]["best_arm"] == "dflash-mlx"
    assert report["offline_comparators"]["per_request_wins"] == {"dflash-mlx": 12}
    totals = report["offline_comparators"]["total_wall_seconds"]
    assert totals["dflash-mlx"] == pytest.approx(26.0635, abs=0.0001)
    assert totals["mlx-dspark"] == pytest.approx(45.0937, abs=0.0001)
    assert report["offline_comparators"]["best_speedup_over_other"] == pytest.approx(
        1.7301,
        abs=0.0001,
    )
    assert report["measurements"]["router_total_wall_seconds"] == pytest.approx(
        27.1818,
        abs=0.0001,
    )
    assert report["measurements"]["router_speedup_vs_best_static"] == pytest.approx(
        0.9589,
        abs=0.0001,
    )
    assert report["measurements"]["selected_parity_rate"] == 1.0
    assert report["measurements"]["selected_fallbacks"] == 0
    assert report["claim"]["same_corpus_speed_candidate"] is False
    assert report["claim"]["workload_candidate"] is False
    assert report["claim"]["global_breakthrough"] is False
