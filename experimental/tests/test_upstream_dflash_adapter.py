from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from experimental.upstream_dflash.adapter import (
    MioEventAdapter,
    UpstreamGenerationRequest,
    adapt_upstream_stream,
)
from experimental.upstream_dflash.compatibility import (
    ParityCertificate,
    PromotionRequest,
    assess_promotion,
)
from experimental.upstream_dflash.comparison import compare_benchmark_artifacts
from experimental.upstream_dflash.deferred_priming import deferred_drafter_priming


@dataclass(frozen=True)
class PrefillCompleteEvent:
    prefill_us: float
    prompt_token_count: int
    prefill_tokens_restored: int


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    generated_tokens: int
    acceptance_ratio: float
    cycles_completed: int


@dataclass(frozen=True)
class SummaryEvent:
    elapsed_us: float
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    generation_tokens: int
    accepted_from_draft: int
    acceptance_ratio: float
    cycles_completed: int
    phase_timings_us: dict[str, float]


def test_adapter_preserves_mio_stream_contract_and_prefill_state() -> None:
    adapter = MioEventAdapter()
    events = [
        adapter.adapt(PrefillCompleteEvent(125.0, 8, 3)),
        adapter.adapt(TokenEvent(42, 1, 0.0, 0)),
        adapter.adapt(SummaryEvent(500.0, 8, (42,), 1, 0, 0.0, 1, {"prefill": 125.0})),
    ]

    assert [event["event"] for event in events] == ["prefill", "token", "summary"]
    assert events[0]["warm_offset"] == 3
    assert events[2]["prefill_us"] == 125.0
    assert events[2]["warm_offset"] == 3
    assert events[2]["cache_commit_mode"] == "upstream_target_ops_rollback"


def test_adapter_closes_the_upstream_generator_when_consumer_cancels() -> None:
    closed = False

    def source():
        nonlocal closed
        try:
            yield TokenEvent(7, 1, 0.0, 0)
            yield TokenEvent(8, 2, 0.5, 1)
        finally:
            closed = True

    stream = adapt_upstream_stream(source())
    assert next(stream)["token_id"] == 7
    stream.close()
    assert closed is True


def test_request_rejects_prefix_cache_without_upstream_snapshot_service() -> None:
    with pytest.raises(ValueError, match="snapshot_service"):
        UpstreamGenerationRequest(max_new_tokens=8, prefix_cache_active=True)


def test_fused_cold_prefill_patch_is_explicit_and_restores_upstream_globals() -> None:
    import dflash_mlx.engine.spec_epoch as spec_epoch

    original_store = spec_epoch.TargetFeatureStore
    original_boundary = spec_epoch.compute_snapshot_boundary
    with deferred_drafter_priming(fuse_cold_prefill=True) as stats:
        assert spec_epoch.TargetFeatureStore is not original_store
        assert spec_epoch.compute_snapshot_boundary(64, None) == 0
        assert spec_epoch.compute_snapshot_boundary(64, 32) == 32
        assert stats.fused_cold_prefill is True

    assert spec_epoch.TargetFeatureStore is original_store
    assert spec_epoch.compute_snapshot_boundary is original_boundary


def _matched_payload() -> dict:
    pairs = [
        {"eligible": True, "prompt_id": f"p{index % 4}", "token_parity": True}
        for index in range(12)
    ]
    return {
        "created_at": "2026-07-15T00:00:00Z",
        "provenance": {
            "git": {"revision": "abc", "dirty": False},
            "software": {"mlx": "0.32.0", "mlx-lm": "0.31.3", "dflash-mlx": "0.1.8"},
            "models": {
                "target": {"reference": "target", "config_sha256": "target-hash"},
                "dflash-mlx_draft": {"reference": "draft", "config_sha256": "draft-hash"},
            },
            "hardware": {},
        },
        "configuration": {
            "dflash_verify_mode": "dflash",
            "dflash_draft_quant": "w4:gs64",
            "dflash_block_tokens": None,
            "dflash_verify_cap": None,
        },
        "paired_comparisons": {
            "dflash-mlx": {
                "eligible_pairs": 12,
                "parity_rate": 1.0,
                "fallback_count": 0,
                "pairs": pairs,
            }
        },
        "checks": {"strict_passed": True},
    }


def _promotion(certificate: ParityCertificate, **overrides) -> PromotionRequest:
    values = {
        "target_reference": "target",
        "draft_reference": "draft",
        "target_config_sha256": "target-hash",
        "draft_config_sha256": "draft-hash",
        "mlx_version": "0.32.0",
        "mlx_lm_version": "0.31.3",
        "dflash_mlx_version": "0.1.8",
        "parity": certificate,
    }
    values.update(overrides)
    return PromotionRequest(**values)


def test_default_promotion_requires_exact_certificate_identity() -> None:
    certificate = ParityCertificate.from_matched_benchmark(_matched_payload())
    report = assess_promotion(_promotion(certificate))
    mismatch = assess_promotion(_promotion(certificate, mlx_version="0.33.0"))

    assert report.ready_for_default is True
    assert mismatch.prototype_eligible is False
    assert "mlx_version" in mismatch.blockers[-1].detail


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"sampling": True}, "greedy_sampling"),
        ({"dynamic_suppress_after": True}, "dynamic_suppression"),
        ({"tool_mode": "required_dynamic_eos"}, "tools"),
        ({"pq_bits": 4}, "pq_tq"),
        ({"tq_bits": 4}, "pq_tq"),
        ({"prefix_cache_mode": "mio_warm_state"}, "prefix_cache"),
        ({"require_logprobs": True}, "logprobs"),
    ],
)
def test_unsupported_mio_semantics_block_promotion(overrides: dict, blocker: str) -> None:
    certificate = ParityCertificate.from_matched_benchmark(_matched_payload())
    report = assess_promotion(_promotion(certificate, **overrides))

    assert blocker in {gate.name for gate in report.blockers}


def test_comparison_marks_cross_corpus_verify_ratio_as_diagnostic(tmp_path) -> None:
    upstream = _matched_payload()
    upstream["configuration"].update({"max_tokens": 64})
    upstream["paired_comparisons"]["dflash-mlx"]["metrics"] = {
        name: {"point_estimate": value}
        for name, value in {
            "ttft_speedup": 0.9,
            "decode_speedup": 2.3,
            "end_to_end_speedup": 2.0,
        }.items()
    }
    upstream["runs"] = [
        {
            "status": "ok",
            "mode": "dflash-mlx",
            "diagnostics": {
                "cycles_completed": 10,
                "phase_timings_us": {"verify": 40_000.0},
            },
        }
    ]
    vendored = {
        "created_at": "2026-07-15T00:00:01Z",
        "git_revision": "def",
        "git_dirty": True,
        "software": {},
        "hardware": {},
        "models": {},
        "parameters": {},
        "results": {
            "baseline": {"aggregate": {"median_decode_us": 100.0, "median_elapsed_us": 120.0}},
            "dflash": {
                "all_match_baseline": True,
                "aggregate": {
                    "median_decode_us": 200.0,
                    "median_elapsed_us": 240.0,
                    "median_tokens_per_cycle": 4.0,
                    "median_phase_timings_us": {"verify": 80_000.0},
                    "cache_commit_modes": ["timewise_exact_tape"],
                },
                "repetitions": [{"generation_tokens": 40}],
            },
        },
    }
    upstream_path = tmp_path / "upstream.json"
    vendored_path = tmp_path / "vendored.json"
    upstream_path.write_text(json.dumps(upstream))
    vendored_path.write_text(json.dumps(vendored))

    result = compare_benchmark_artifacts(upstream_path, vendored_path)

    assert result["verification_diagnostic"]["vendored_over_upstream_ratio"] == 2.0
    assert "not a paired speedup" in result["interpretation"]["claim_limit"]
