"""Unit tests for the matched speculative R&D harness (no model loading)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

import scripts.bench_speculative_matched as benchmark


def test_module_import_does_not_load_model_packages():
    assert "mlx" not in benchmark.__dict__
    assert "mlx_dspark" not in benchmark.__dict__
    assert "dflash_mlx" not in benchmark.__dict__


def test_balanced_schedule_is_seeded_and_latin_balanced():
    schedule = benchmark._balanced_mode_schedule(benchmark.MODES, 6, seed=17)

    assert schedule == benchmark._balanced_mode_schedule(benchmark.MODES, 6, seed=17)
    assert schedule != benchmark._balanced_mode_schedule(benchmark.MODES, 6, seed=18)
    for start in range(0, 6, len(benchmark.MODES)):
        block = schedule[start : start + len(benchmark.MODES)]
        for position in range(len(benchmark.MODES)):
            assert sorted(order[position] for order in block) == sorted(benchmark.MODES)


def test_lookup_opt_in_uses_the_same_latin_balance_with_a_distinct_mode():
    modes = (*benchmark.MODES, benchmark.DSPARK_LOOKUP_MODE)
    schedule = benchmark._balanced_mode_schedule(modes, 8, seed=29)

    for start in range(0, 8, len(modes)):
        block = schedule[start : start + len(modes)]
        for position in range(len(modes)):
            assert sorted(order[position] for order in block) == sorted(modes)


@pytest.mark.parametrize("blocks", [-1])
def test_balanced_schedule_rejects_invalid_inputs(blocks):
    with pytest.raises(ValueError, match="non-negative"):
        benchmark._balanced_mode_schedule(benchmark.MODES, blocks, seed=1)
    with pytest.raises(ValueError, match="unique"):
        benchmark._balanced_mode_schedule(["same", "same"], 1, seed=1)


def test_load_corpus_supports_builtin_json_jsonl_and_text(tmp_path: Path):
    builtin = benchmark._load_corpus(None)
    assert len(builtin) == len(benchmark.BUILTIN_CORPUS)

    json_path = tmp_path / "corpus.json"
    json_path.write_text(
        json.dumps({"prompts": [{"id": "one", "prompt": "First"}, "Second"]}),
        encoding="utf-8",
    )
    assert [case.prompt for case in benchmark._load_corpus(json_path)] == ["First", "Second"]

    jsonl_path = tmp_path / "corpus.jsonl"
    jsonl_path.write_text(
        '{"id":"a","prompt":"Alpha"}\n"Beta"\n',
        encoding="utf-8",
    )
    assert [case.id for case in benchmark._load_corpus(jsonl_path)] == ["a", "beta"]

    text_path = tmp_path / "corpus.txt"
    text_path.write_text("First block\n---\nSecond block\n", encoding="utf-8")
    assert [case.prompt for case in benchmark._load_corpus(text_path)] == [
        "First block",
        "Second block",
    ]


def test_load_corpus_rejects_empty_prompt(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('[{"id":"empty","prompt":""}]', encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        benchmark._load_corpus(path)


def _fake_runtime(raw: benchmark.RawGeneration) -> benchmark.BenchmarkRuntime:
    def generate(_prompt_ids, _max_tokens, _seed, on_first_output):
        on_first_output()
        return raw

    return benchmark.BenchmarkRuntime(
        encode_prompt=lambda _prompt, _chat: [1, 2],
        generators={mode: generate for mode in benchmark.MODES},
        reset_peak_memory=lambda: None,
        get_peak_memory=lambda: 2_000_000_000,
    )


def test_run_once_normalizes_upstream_overshoot_and_separates_decode():
    runtime = _fake_runtime(
        benchmark.RawGeneration(
            token_ids=[10, 11, 12, 13, 14, 15],
            diagnostics={"engine": "fake"},
        )
    )
    ticks = iter([10.0, 10.2, 11.0])

    row = benchmark._run_once(
        runtime,
        "mlx-dspark",
        prompt_ids=[1, 2, 3],
        max_tokens=4,
        seed=7,
        clock=lambda: next(ticks),
    )

    assert row["token_ids"] == [10, 11, 12, 13]
    assert row["raw_token_count"] == 6
    assert row["normalized_token_count"] == 4
    assert row["overshoot_tokens"] == 2
    assert row["timing_includes_upstream_overshoot"] is True
    assert row["exact_normalized_length"] is True
    assert row["ttft_seconds"] == pytest.approx(0.2)
    assert row["decode_seconds"] == pytest.approx(0.8)
    assert row["decode_tps"] == pytest.approx(3.75)
    assert row["peak_memory_gb"] == pytest.approx(2.0)


def test_run_once_marks_unobserved_ttft_and_short_output():
    def generate(_prompt_ids, _max_tokens, _seed, _on_first_output):
        return benchmark.RawGeneration(token_ids=[1])

    runtime = benchmark.BenchmarkRuntime(
        encode_prompt=lambda _prompt, _chat: [1],
        generators={mode: generate for mode in benchmark.MODES},
        reset_peak_memory=lambda: None,
        get_peak_memory=lambda: 0,
    )
    ticks = iter([1.0, 2.0])

    row = benchmark._run_once(
        runtime,
        "baseline",
        prompt_ids=[3],
        max_tokens=4,
        seed=1,
        clock=lambda: next(ticks),
    )

    assert row["ttft_observed"] is False
    assert row["ttft_seconds"] is None
    assert row["decode_tps"] is None
    assert row["shortfall_tokens"] == 3
    assert row["exact_normalized_length"] is False


def test_native_event_collector_preserves_fallback_and_summary_metrics():
    class Token:
        def __init__(self, token_id, *, fallback=False):
            self.token_id = token_id
            self.fallback_ar = fallback
            self.fallback_reason = "unsupported-context" if fallback else None

    class Summary:
        generated_token_ids = (7, 8)
        elapsed_us = 1_000.0
        generation_tokens = 2
        acceptance_ratio = 0.5
        cycles_completed = 1
        tokens_per_cycle = 2.0
        phase_timings_us = {"prefill": 250.0, "verify": 100.0}
        fallback_ar = True
        fallback_reason = "unsupported-context"

    first_output = []
    result = benchmark._collect_native_events(
        [Token(7), Token(8, fallback=True), Summary()],
        token_event_type=Token,
        summary_event_type=Summary,
        on_first_output=lambda: first_output.append(True),
        baseline=False,
    )

    assert first_output == [True]
    assert result.token_ids == [7, 8]
    assert result.fallback is True
    assert result.fallback_reason == "unsupported-context"
    assert result.diagnostics["phase_timings_us"]["prefill"] == 250.0


def test_dspark_generators_share_objects_and_differ_only_by_lookup_flag():
    target = object()
    tokenizer = object()
    drafter = object()
    calls = []

    def speculative_generate(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["on_text"]("first")
        return SimpleNamespace(
            token_ids=[7, 8],
            seconds=0.25,
            num_tokens=2,
            num_rounds=1,
            target_forwards=1,
            mean_accept_len=2.0,
            accept_lengths=[2],
            lookup_rounds=int(kwargs["lookup_drafts"]),
        )

    pure = benchmark._build_dspark_generator(
        speculative_generate_fn=speculative_generate,
        target=target,
        tokenizer=tokenizer,
        drafter=drafter,
        max_draft_tokens=2,
        lookup_drafts=False,
        mode_label="mlx-dspark",
    )
    lookup = benchmark._build_dspark_generator(
        speculative_generate_fn=speculative_generate,
        target=target,
        tokenizer=tokenizer,
        drafter=drafter,
        max_draft_tokens=2,
        lookup_drafts=True,
        mode_label=benchmark.DSPARK_LOOKUP_MODE,
    )

    first_outputs = []
    pure_result = pure([1, 2], 2, 71, lambda: first_outputs.append("pure"))
    lookup_result = lookup([1, 2], 2, 71, lambda: first_outputs.append("lookup"))

    assert first_outputs == ["pure", "lookup"]
    assert [call[0] for call in calls] == [
        (target, tokenizer, drafter),
        (target, tokenizer, drafter),
    ]
    assert [call[1]["lookup_drafts"] for call in calls] == [False, True]
    assert [call[1]["seed"] for call in calls] == [71, 71]
    assert pure_result.diagnostics["candidate_mode"] == "mlx-dspark"
    assert pure_result.diagnostics["lookup_drafts"] is False
    assert lookup_result.diagnostics["candidate_mode"] == benchmark.DSPARK_LOOKUP_MODE
    assert lookup_result.diagnostics["lookup_drafts"] is True


def test_paired_bootstrap_is_seeded_and_clusters_repetitions_by_prompt():
    pairs = [
        {
            "prompt_id": f"p{prompt}",
            "repetition": repetition,
            "eligible": True,
            "speedup": 1.1 + prompt / 10,
        }
        for prompt in range(4)
        for repetition in range(1, 4)
    ]
    first = benchmark._paired_bootstrap_ci(
        pairs,
        metric_key="speedup",
        samples=500,
        confidence=0.95,
        seed=23,
    )
    second = benchmark._paired_bootstrap_ci(
        pairs,
        metric_key="speedup",
        samples=500,
        confidence=0.95,
        seed=23,
    )

    assert first == second
    assert first["estimand"] == "median_paired_speedup_cluster_bootstrap_by_prompt"
    assert first["n_pairs"] == 12
    assert first["n_clusters"] == 4
    assert first["point_estimate"] == pytest.approx(1.25)
    assert first["lower"] <= first["point_estimate"] <= first["upper"]


def _measured_run(
    run_id: str,
    mode: str,
    prompt_id: str,
    repetition: int,
    *,
    tokens: list[int],
    ttft: float,
    decode_tps: float,
    wall: float,
    peak_memory_bytes: int = 1_000,
    fallback: bool = False,
) -> dict:
    return {
        "run_id": run_id,
        "mode": mode,
        "prompt_id": prompt_id,
        "repetition": repetition,
        "status": "ok",
        "exact_normalized_length": True,
        "token_ids": tokens,
        "ttft_observed": True,
        "ttft_seconds": ttft,
        "decode_tps": decode_tps,
        "decode_seconds": wall - ttft,
        "wall_seconds": wall,
        "peak_memory_bytes": peak_memory_bytes,
        "fallback": fallback,
    }


def test_pairing_is_within_prompt_and_repetition():
    runs = []
    counter = 0
    for prompt in ("a", "b"):
        for repetition in (1, 2):
            counter += 1
            runs.extend(
                [
                    _measured_run(
                        f"b-{counter}",
                        "baseline",
                        prompt,
                        repetition,
                        tokens=[1, 2],
                        ttft=2.0,
                        decode_tps=10.0,
                        wall=4.0,
                    ),
                    _measured_run(
                        f"d-{counter}",
                        "mlx-dspark",
                        prompt,
                        repetition,
                        tokens=[1, 2],
                        ttft=1.0,
                        decode_tps=15.0,
                        wall=2.0,
                    ),
                    _measured_run(
                        f"f-{counter}",
                        "dflash-mlx",
                        prompt,
                        repetition,
                        tokens=[1, 9] if prompt == "b" else [1, 2],
                        ttft=1.0,
                        decode_tps=20.0,
                        wall=2.0,
                    ),
                ]
            )

    paired = benchmark._pair_runs(runs, prompt_count=2, repetitions=2)

    assert len(paired["mlx-dspark"]) == 4
    assert {pair["ttft_speedup"] for pair in paired["mlx-dspark"]} == {2.0}
    assert {pair["decode_speedup"] for pair in paired["mlx-dspark"]} == {1.5}
    assert {pair["baseline_decode_seconds"] for pair in paired["mlx-dspark"]} == {2.0}
    assert {pair["candidate_peak_memory_bytes"] for pair in paired["mlx-dspark"]} == {1_000}
    assert sum(pair["token_parity"] for pair in paired["dflash-mlx"]) == 2


def _candidate_pairs(
    *,
    parity: bool = True,
    fallback: bool = False,
    memory_multiplier: float | None = 1.0,
) -> list[dict]:
    pairs: list[dict] = []
    for prompt_index in range(4):
        for repetition in range(1, 4):
            index = prompt_index * 3 + repetition - 1
            pairs.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "repetition": repetition,
                    "eligible": True,
                    "token_parity": parity,
                    "ttft_speedup": 1.10 + index / 1000,
                    "decode_speedup": 1.20 + index / 1000,
                    "end_to_end_speedup": 1.15,
                    "baseline_ttft_seconds": 1.10 + index / 1000,
                    "candidate_ttft_seconds": 1.0,
                    "baseline_decode_seconds": 1.20 + index / 1000,
                    "candidate_decode_seconds": 1.0,
                    "baseline_peak_memory_bytes": 1_000 if memory_multiplier is not None else None,
                    "candidate_peak_memory_bytes": (
                        int(1_000 * memory_multiplier) if memory_multiplier is not None else None
                    ),
                    "candidate_fallback": fallback,
                }
            )
    return pairs


def test_workload_candidate_gate_requires_speed_tail_parity_and_no_fallback():
    analysis = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(),
        bootstrap_samples=500,
        confidence=0.95,
        seed=9,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
    )

    assert analysis["candidate_workload_speedup"] is True
    assert analysis["global_breakthrough"] is False
    assert analysis["breakthrough"] is False
    assert all(analysis["conditions"].values())
    assert all(value is True for value in analysis["conditional_conditions"].values())

    no_parity = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(parity=False),
        bootstrap_samples=100,
        confidence=0.95,
        seed=9,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
    )
    with_fallback = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(fallback=True),
        bootstrap_samples=100,
        confidence=0.95,
        seed=9,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
    )
    nondeterministic_baseline = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(),
        bootstrap_samples=100,
        confidence=0.95,
        seed=9,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
        baseline_deterministic=False,
    )
    assert no_parity["candidate_workload_speedup"] is False
    assert with_fallback["candidate_workload_speedup"] is False
    assert nondeterministic_baseline["candidate_workload_speedup"] is False


def test_cluster_bootstrap_rejects_three_good_prompts_and_one_regression():
    pairs = _candidate_pairs()
    for pair in pairs:
        speedup = 0.95 if pair["prompt_id"] == "p0" else 1.08
        pair.update(
            {
                "ttft_speedup": speedup,
                "decode_speedup": speedup,
                "baseline_ttft_seconds": speedup,
                "candidate_ttft_seconds": 1.0,
                "baseline_decode_seconds": speedup,
                "candidate_decode_seconds": 1.0,
            }
        )

    analysis = benchmark._candidate_analysis(
        "mlx-dspark",
        pairs,
        bootstrap_samples=2_000,
        confidence=0.95,
        seed=77,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
        min_distinct_prompts=4,
    )

    assert analysis["metrics"]["ttft_speedup"]["point_estimate"] == pytest.approx(1.08)
    assert analysis["metrics"]["ttft_speedup"]["n_clusters"] == 4
    assert analysis["metrics"]["ttft_speedup"]["lower"] <= 0.95
    assert analysis["conditions"]["ttft_ci_lower_bound"] is False
    assert analysis["candidate_workload_speedup"] is False


def test_timing_eligibility_and_strict_checks_cannot_produce_candidate():
    runs = []
    for prompt_index in range(4):
        prompt_id = f"p{prompt_index}"
        for repetition in range(1, 4):
            for mode in benchmark.MODES:
                run = _measured_run(
                    f"{mode}-{prompt_id}-{repetition}",
                    mode,
                    prompt_id,
                    repetition,
                    tokens=[1, 2],
                    ttft=1.0,
                    decode_tps=2.0,
                    wall=2.0,
                )
                if mode == "mlx-dspark" and prompt_index == 0 and repetition == 1:
                    run["ttft_observed"] = False
                    run["ttft_seconds"] = None
                runs.append(run)

    pairs = benchmark._pair_runs(runs, prompt_count=4, repetitions=3)["mlx-dspark"]
    assert pairs[0]["eligible"] is False
    assert pairs[0]["candidate_timing_valid"] is False
    missing_timing = benchmark._candidate_analysis(
        "mlx-dspark",
        pairs,
        bootstrap_samples=100,
        confidence=0.95,
        seed=13,
        min_ttft_speedup=0.5,
        min_decode_speedup=0.5,
        ci_floor=0.1,
        required_parity=0.9,
        min_pairs=9,
        min_distinct_prompts=4,
    )
    strict_failure = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(),
        bootstrap_samples=100,
        confidence=0.95,
        seed=13,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
        min_distinct_prompts=4,
        strict_passed=False,
    )

    assert missing_timing["conditions"]["all_runs_eligible"] is False
    assert missing_timing["candidate_workload_speedup"] is False
    assert strict_failure["conditions"]["strict_run_checks"] is False
    assert strict_failure["candidate_workload_speedup"] is False


def test_p95_tail_gate_rejects_regression_hidden_by_median_speedup():
    pairs = _candidate_pairs()
    pairs[-1].update(
        {
            "ttft_speedup": 0.25,
            "decode_speedup": 0.25,
            "baseline_ttft_seconds": 1.0,
            "candidate_ttft_seconds": 4.0,
            "baseline_decode_seconds": 1.0,
            "candidate_decode_seconds": 4.0,
        }
    )

    analysis = benchmark._candidate_analysis(
        "dflash-mlx",
        pairs,
        bootstrap_samples=500,
        confidence=0.95,
        seed=21,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
    )

    assert analysis["metrics"]["ttft_speedup"]["point_estimate"] > 1.05
    assert analysis["conditions"]["ttft_p95_point_non_regression"] is False
    assert analysis["conditions"]["decode_p95_point_non_regression"] is False
    assert analysis["candidate_workload_speedup"] is False


def test_peak_memory_gate_is_conditional_and_never_fabricates_missing_data():
    regressed = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(memory_multiplier=1.20),
        bootstrap_samples=100,
        confidence=0.95,
        seed=31,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
        max_peak_memory_regression=0.05,
    )
    unavailable = benchmark._candidate_analysis(
        "mlx-dspark",
        _candidate_pairs(memory_multiplier=None),
        bootstrap_samples=100,
        confidence=0.95,
        seed=31,
        min_ttft_speedup=1.05,
        min_decode_speedup=1.05,
        ci_floor=1.0,
        required_parity=1.0,
        min_pairs=9,
        max_peak_memory_regression=0.05,
    )

    assert regressed["evidence_completeness"]["peak_memory_gate_applied"] is True
    assert regressed["conditional_conditions"]["peak_memory_p95_point_non_regression"] is False
    assert regressed["candidate_workload_speedup"] is False
    assert unavailable["metrics"]["peak_memory_p95_efficiency"]["measurement_status"] == "unavailable"
    assert unavailable["metrics"]["peak_memory_p95_efficiency"]["point_estimate"] is None
    assert unavailable["evidence_completeness"]["peak_memory_gate_applied"] is False
    assert all(value is None for value in unavailable["conditional_conditions"].values())
    assert unavailable["candidate_workload_speedup"] is True


def test_parser_exposes_matched_local_defaults_and_conservative_gate():
    args = benchmark._build_parser().parse_args([])

    assert args.model == benchmark.DEFAULT_TARGET
    assert args.dspark_draft == benchmark.DEFAULT_DSPARK_DRAFT
    assert args.dflash_draft == benchmark.DEFAULT_DFLASH_DRAFT
    assert args.min_ttft_speedup == 1.05
    assert args.min_decode_speedup == 1.05
    assert args.ci_floor == 1.0
    assert args.required_parity == 1.0
    assert args.min_pairs == 9
    assert args.min_distinct_prompts == 4
    assert args.max_p95_latency_regression == 0.0
    assert args.max_peak_memory_regression == 0.05
    assert args.dspark_lookup is False
    assert benchmark._enabled_modes(args) == benchmark.MODES

    lookup_args = benchmark._build_parser().parse_args(["--dspark-lookup"])
    assert lookup_args.dspark_lookup is True
    assert benchmark._enabled_modes(lookup_args) == (
        *benchmark.MODES,
        benchmark.DSPARK_LOOKUP_MODE,
    )


def test_main_writes_versioned_paired_artifact_without_loading_models(monkeypatch, tmp_path: Path):
    output = tmp_path / "matched.json"

    def generate(_prompt_ids, max_tokens, _seed, on_first_output):
        on_first_output()
        return benchmark.RawGeneration(token_ids=list(range(max_tokens)))

    runtime = benchmark.BenchmarkRuntime(
        encode_prompt=lambda _prompt, _chat: [1, 2, 3],
        generators={mode: generate for mode in benchmark.MODES},
        reset_peak_memory=lambda: None,
        get_peak_memory=lambda: 1024,
        metadata={"test": True},
    )
    monkeypatch.setattr(benchmark, "_load_runtime", lambda _args: runtime)
    monkeypatch.setattr(
        benchmark,
        "_provenance",
        lambda _args, _runtime: {"software": {"mlx-dspark": "test"}},
    )

    exit_code = benchmark.main(
        [
            "--warmup",
            "0",
            "--reps",
            "3",
            "--max-tokens",
            "2",
            "--bootstrap-samples",
            "25",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema"] == {
        "name": benchmark.SCHEMA_NAME,
        "version": benchmark.SCHEMA_VERSION,
    }
    assert payload["checks"]["strict_passed"] is True
    assert payload["research_claim"]["global_breakthrough_evaluable"] is False
    assert payload["research_claim"]["global_breakthrough"] is False
    assert payload["breakthrough"]["any_candidate"] is False
    assert payload["breakthrough"]["candidate_results"] == {candidate: False for candidate in benchmark.CANDIDATES}
    assert len(payload["runs"]) == len(benchmark.BUILTIN_CORPUS) * 3 * len(benchmark.MODES)
    assert payload["configuration"]["enabled_modes"] == list(benchmark.MODES)
    assert payload["configuration"]["dspark_lookup"] is False
    assert payload["configuration"]["dspark_lookup_candidate_enabled"] is False
    assert all(
        len(analysis["pairs"]) == len(benchmark.BUILTIN_CORPUS) * 3
        for analysis in payload["paired_comparisons"].values()
    )
    assert all(sorted(order["modes"]) == sorted(benchmark.MODES) for order in payload["schedule"]["measurements"])


def test_main_opt_in_gates_lookup_as_a_distinct_matched_candidate(monkeypatch, tmp_path: Path):
    output = tmp_path / "matched-lookup.json"

    def fake_load_runtime(args):
        modes = benchmark._enabled_modes(args)

        def make_generator(mode):
            def generate(_prompt_ids, max_tokens, _seed, on_first_output):
                on_first_output()
                # Keep the fake decode interval positive on fast clocks so the
                # strict timing gate is deterministic.
                time.sleep(0.0001)
                return benchmark.RawGeneration(
                    token_ids=list(range(max_tokens)),
                    diagnostics={
                        "candidate_mode": mode,
                        "lookup_drafts": mode == benchmark.DSPARK_LOOKUP_MODE,
                    },
                )

            return generate

        return benchmark.BenchmarkRuntime(
            encode_prompt=lambda _prompt, _chat: [1, 2, 3],
            generators={mode: make_generator(mode) for mode in modes},
            reset_peak_memory=lambda: None,
            get_peak_memory=lambda: 1024,
            metadata={
                "dspark_lookup_candidate_enabled": args.dspark_lookup,
                "dspark_candidates_share_target_and_drafter": True,
            },
        )

    monkeypatch.setattr(benchmark, "_load_runtime", fake_load_runtime)
    monkeypatch.setattr(
        benchmark,
        "_provenance",
        lambda _args, runtime: {"runtime": runtime.metadata},
    )

    exit_code = benchmark.main(
        [
            "--dspark-lookup",
            "--warmup",
            "0",
            "--reps",
            "3",
            "--max-tokens",
            "2",
            "--bootstrap-samples",
            "25",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    modes = (*benchmark.MODES, benchmark.DSPARK_LOOKUP_MODE)
    candidates = modes[1:]
    assert exit_code == 0
    assert payload["configuration"]["enabled_modes"] == list(modes)
    assert payload["configuration"]["dspark_lookup"] is True
    assert payload["configuration"]["dspark_lookup_candidate_enabled"] is True
    assert set(payload["paired_comparisons"]) == set(candidates)
    assert set(payload["research_claim"]["workload_candidate_results"]) == set(candidates)
    assert payload["research_claim"]["global_breakthrough_evaluable"] is False
    assert payload["research_claim"]["global_breakthrough"] is False
    assert payload["breakthrough"]["candidate_results"] == {candidate: False for candidate in candidates}
    assert payload["breakthrough"]["any_candidate"] is False
    assert (
        payload["paired_comparisons"][benchmark.DSPARK_LOOKUP_MODE]["criterion"]
        == payload["paired_comparisons"]["mlx-dspark"]["criterion"]
    )
    assert payload["paired_comparisons"][benchmark.DSPARK_LOOKUP_MODE]["expected_pairs"] == 12
    assert len(payload["runs"]) == len(benchmark.BUILTIN_CORPUS) * 3 * len(modes)
    assert all(sorted(order["modes"]) == sorted(modes) for order in payload["schedule"]["measurements"])
    grouped = {}
    for run in payload["runs"]:
        grouped.setdefault((run["prompt_id"], run["repetition"]), []).append(run)
    assert all({run["mode"] for run in block} == set(modes) for block in grouped.values())
    assert all(len({run["call_seed"] for run in block}) == 1 for block in grouped.values())
    assert payload["provenance"]["runtime"] == {
        "dspark_lookup_candidate_enabled": True,
        "dspark_candidates_share_target_and_drafter": True,
    }
    lookup_runs = [run for run in payload["runs"] if run["mode"] == benchmark.DSPARK_LOOKUP_MODE]
    pure_runs = [run for run in payload["runs"] if run["mode"] == "mlx-dspark"]
    assert all(run["diagnostics"]["lookup_drafts"] is True for run in lookup_runs)
    assert all(run["diagnostics"]["lookup_drafts"] is False for run in pure_runs)
