"""Tests for the reproducible Qwen 3.6 benchmark artifact schema."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import scripts.bench_qwen36_matrix as benchmark


def test_run_once_preserves_dflash_phase_and_commit_metrics(monkeypatch):
    raw_result = {
        "elapsed_us": 1_000.0,
        "prompt_token_count": 4,
        "generated_token_ids": [7, 8],
        "generation_tokens": 2,
        "acceptance_ratio": 0.5,
        "tokens_per_cycle": 2.0,
        "phase_timings_us": {
            "prefill": 100.0,
            "verify": 250.0,
            "rebuild": 40.0,
        },
        "cache_commit_mode": "restore_rebuild_singleton_exact",
        "rebuilt_target_tokens": 2,
        "exact_acceptance_corrections": 1,
    }
    monkeypatch.setattr(
        benchmark,
        "generate_dflash_once",
        lambda **_kwargs: raw_result,
    )

    row = benchmark._run_once(
        benchmark.MODES["dflash"],
        target_model=object(),
        draft_model=object(),
        tokenizer=object(),
        prompt_tokens=[1, 2, 3, 4],
        max_new_tokens=2,
    )

    assert tuple(row["phase_timings_us"]) == benchmark.PHASE_TIMING_NAMES
    assert row["phase_timings_us"]["prefill"] == 100.0
    assert row["phase_timings_us"]["verify"] == 250.0
    assert row["phase_timings_us"]["rebuild"] == 40.0
    assert row["phase_timings_us"]["replay"] == 0.0
    assert row["cache_commit_mode"] == "restore_rebuild_singleton_exact"
    assert row["rebuilt_target_tokens"] == 2
    assert row["exact_acceptance_corrections"] == 1


def test_run_once_normalizes_baseline_metrics(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "generate_baseline_once",
        lambda **_kwargs: {
            "elapsed_us": 500.0,
            "prefill_us": 125.0,
            "generated_token_ids": [9],
            "generation_tokens": 1,
        },
    )

    row = benchmark._run_once(
        benchmark.MODES["baseline"],
        target_model=object(),
        draft_model=object(),
        tokenizer=object(),
        prompt_tokens=[1, 2],
        max_new_tokens=1,
    )

    assert row["phase_timings_us"] == {
        name: 125.0 if name == "prefill" else 0.0
        for name in benchmark.PHASE_TIMING_NAMES
    }
    assert row["cache_commit_mode"] is None
    assert row["rebuilt_target_tokens"] == 0
    assert row["exact_acceptance_corrections"] == 0


def test_aggregate_includes_phase_counter_and_mode_summaries():
    def row(
        value: float,
        *,
        mode: str | None,
        rebuilt: int,
        corrections: int,
    ) -> dict:
        numeric = {
            "elapsed_us": value,
            "prefill_us": value,
            "decode_us": value,
            "prefill_tps": value,
            "decode_tps": value,
            "end_to_end_tps": value,
            "acceptance_ratio": value,
            "tokens_per_cycle": value,
            "peak_memory_gb": value,
        }
        return {
            **numeric,
            "phase_timings_us": {
                name: value * (index + 1)
                for index, name in enumerate(benchmark.PHASE_TIMING_NAMES)
            },
            "cache_commit_mode": mode,
            "rebuilt_target_tokens": rebuilt,
            "exact_acceptance_corrections": corrections,
            "paired_speedup_vs_baseline": {
                metric: value / 10.0 for metric in benchmark.SPEEDUP_METRICS
            },
        }

    aggregate = benchmark._aggregate(
        [
            row(
                10.0,
                mode="timewise_exact_tape",
                rebuilt=0,
                corrections=0,
            ),
            row(
                30.0,
                mode="restore_rebuild_singleton_exact",
                rebuilt=8,
                corrections=2,
            ),
        ]
    )

    assert benchmark.SCHEMA_VERSION == 2
    assert aggregate["median_phase_timings_us"]["verify"] == 100.0
    assert aggregate["median_phase_timings_us"]["rebuild"] == 140.0
    assert aggregate["median_rebuilt_target_tokens"] == 4.0
    assert aggregate["median_exact_acceptance_corrections"] == 1.0
    assert aggregate["cache_commit_modes"] == [
        "restore_rebuild_singleton_exact",
        "timewise_exact_tape",
    ]
    assert aggregate["median_paired_speedup_vs_baseline"] == {
        metric: 2.0 for metric in benchmark.SPEEDUP_METRICS
    }


def test_balanced_mode_schedule_is_seeded_and_position_balanced():
    modes = ["baseline", "dflash", "pq4"]
    schedule = benchmark._balanced_mode_schedule(modes, 6, seed=17)

    assert schedule == benchmark._balanced_mode_schedule(modes, 6, seed=17)
    assert all(sorted(order) == sorted(modes) for order in schedule)
    for latin_block_start in range(0, 6, len(modes)):
        latin_block = schedule[latin_block_start : latin_block_start + len(modes)]
        for position in range(len(modes)):
            assert sorted(order[position] for order in latin_block) == sorted(modes)


def _paired_row(token_ids: list[int], value: float) -> dict:
    return {
        "token_ids": token_ids,
        "prefill_tps": value,
        "decode_tps": value * 2,
        "end_to_end_tps": value * 3,
    }


def test_repetition_organization_pairs_modes_and_detects_baseline_drift():
    blocks = [
        {
            "baseline": _paired_row([1, 2], 10.0),
            "dflash": _paired_row([1, 2], 15.0),
            "pq4": _paired_row([9, 9], 20.0),
        },
        {
            "baseline": _paired_row([1, 3], 20.0),
            "dflash": _paired_row([1, 3], 30.0),
            "pq4": _paired_row([8, 8], 40.0),
        },
    ]

    grouped, baseline_deterministic = benchmark._organize_repetitions(
        ["baseline", "dflash", "pq4"],
        blocks,
    )

    assert baseline_deterministic is False
    assert [row["matches_baseline"] for row in grouped["baseline"]] == [True, False]
    assert [row["matches_baseline"] for row in grouped["dflash"]] == [True, True]
    assert [row["matches_baseline"] for row in grouped["pq4"]] == [False, False]
    assert [
        row["paired_speedup_vs_baseline"]["decode_tps"]
        for row in grouped["dflash"]
    ] == [1.5, 1.5]


def test_strict_parity_only_fails_modes_explicitly_marked_exact():
    rows = {
        "baseline": {"all_match_baseline": True},
        "dflash": {"all_match_baseline": False},
        "pq4": {"all_match_baseline": False},
        "tq4": {"all_match_baseline": False},
    }

    assert benchmark.MODES["baseline"].exact is True
    assert benchmark.MODES["dflash"].exact is True
    assert benchmark.MODES["pq4"].exact is False
    assert benchmark.MODES["tq4"].exact is False
    assert benchmark._strict_parity_failures(rows, list(rows)) == ["dflash"]

    rows["baseline"]["all_match_baseline"] = False
    assert benchmark._strict_parity_failures(rows, list(rows)) == [
        "baseline",
        "dflash",
    ]


def test_main_persists_seed_execution_order_and_paired_ratios(monkeypatch, tmp_path):
    output = tmp_path / "matrix.json"
    tier = SimpleNamespace(target_model="target", draft_model="draft")
    monkeypatch.setattr(
        benchmark.MioConfig,
        "default",
        staticmethod(lambda: SimpleNamespace(tiers={"large": tier})),
    )

    model = SimpleNamespace(parameters=lambda: [])
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        benchmark,
        "load_target_bundle",
        lambda *_args, **_kwargs: (model, tokenizer, {"config": {}}),
    )
    monkeypatch.setattr(
        benchmark,
        "load_draft_bundle",
        lambda *_args, **_kwargs: (model, {"config": {}}),
    )
    monkeypatch.setattr(benchmark, "validate_draft_target_compatibility", lambda *_args: {})
    monkeypatch.setattr(benchmark, "bind_draft_target_model", lambda *_args: None)
    monkeypatch.setattr(benchmark.mx, "eval", lambda *_args: None)

    def fake_run(mode, **_kwargs):
        multiplier = {"baseline": 1.0, "dflash": 1.5, "pq4": 2.0}[mode.name]
        return {
            "elapsed_us": 100.0 / multiplier,
            "prefill_us": 20.0 / multiplier,
            "decode_us": 80.0 / multiplier,
            "prefill_tps": 10.0 * multiplier,
            "decode_tps": 20.0 * multiplier,
            "end_to_end_tps": 5.0 * multiplier,
            "acceptance_ratio": 0.0,
            "tokens_per_cycle": 0.0,
            "phase_timings_us": {
                name: 0.0 for name in benchmark.PHASE_TIMING_NAMES
            },
            "cache_commit_mode": None,
            "rebuilt_target_tokens": 0,
            "exact_acceptance_corrections": 0,
            "peak_memory_gb": 1.0,
            "fallback_ar": False,
            "token_ids": [1, 2] if mode.exact else [9, 9],
            "token_hash": mode.name,
        }

    monkeypatch.setattr(benchmark, "_run_once", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen36_matrix.py",
            "--modes",
            "dflash,pq4",
            "--warmup",
            "0",
            "--reps",
            "3",
            "--seed",
            "17",
            "--output",
            str(output),
        ],
    )

    assert benchmark.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["parameters"]["seed"] == 17
    assert [row["modes"] for row in payload["execution_order"]] == (
        benchmark._balanced_mode_schedule(
            ["baseline", "dflash", "pq4"],
            3,
            seed=17,
        )
    )
    assert payload["checks"] == {
        "baseline_deterministic": True,
        "exact_mode_parity": {"baseline": True, "dflash": True},
        "strict_parity_failures": [],
    }
    assert payload["results"]["pq4"]["all_match_baseline"] is False
    assert payload["results"]["dflash"]["aggregate"][
        "median_paired_speedup_vs_baseline"
    ]["decode_tps"] == 1.5
    assert sorted(
        row["position_in_repetition"]
        for row in payload["results"]["baseline"]["repetitions"]
    ) == [1, 2, 3]
