"""Drafter classification, planning, and fallback safety tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import mio.models.registry as registry
from mio.drafter_selection import (
    DrafterKind,
    classify_drafter_config,
    inspect_drafter,
    plan_drafter,
)


def _checkpoint(path, config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def test_classifies_native_dflash_and_hybrid_metadata():
    native = {
        "architectures": ["Qwen3DSparkModel"],
        "block_size": 7,
        "target_layer_ids": [1, 9],
        "markov_rank": 256,
    }
    pure = {
        "architectures": ["DFlashDraftModel"],
        "block_size": 16,
        "dflash_config": {"target_layer_ids": [1, 9]},
    }
    hybrid = {
        "architectures": ["DFlashDraftModel"],
        "block_size": 7,
        "target_layer_ids": [1, 9],
        "markov_rank": 256,
        "dflash_config": {"target_layer_ids": [1, 9], "markov_rank": 256},
    }

    assert classify_drafter_config(native) is DrafterKind.DSPARK
    assert classify_drafter_config(pure) is DrafterKind.DFLASH
    assert classify_drafter_config(hybrid) is DrafterKind.HYBRID_DFLASH_MARKOV
    assert classify_drafter_config({}) is DrafterKind.UNKNOWN


def test_inspection_reads_local_metadata_without_hf_network(tmp_path, monkeypatch):
    ref = _checkpoint(
        tmp_path / "native",
        {
            "architectures": ["Qwen3DSparkModel"],
            "block_size": 7,
            "target_layer_ids": [1],
            "markov_rank": 16,
        },
    )
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda *_args, **_kwargs: pytest.fail("local inspection must not contact HF"),
    )

    descriptor = inspect_drafter(ref)

    assert descriptor.kind is DrafterKind.DSPARK
    assert descriptor.config_path == f"{ref}/config.json"


def test_auto_hybrid_selects_dspark_and_distinct_pure_dflash(tmp_path):
    hybrid = _checkpoint(
        tmp_path / "hybrid",
        {
            "architectures": ["DFlashDraftModel"],
            "block_size": 7,
            "target_layer_ids": [1],
            "markov_rank": 16,
            "dflash_config": {"target_layer_ids": [1], "markov_rank": 16},
        },
    )
    fallback = _checkpoint(
        tmp_path / "fallback",
        {
            "architectures": ["DFlashDraftModel"],
            "block_size": 16,
            "dflash_config": {"target_layer_ids": [1]},
        },
    )
    tier = SimpleNamespace(
        target_model="unregistered-target",
        draft_model=hybrid,
        draft_fallback_model=fallback,
        drafter_backend="auto",
        drafter_strict=False,
    )

    plan = plan_drafter(tier, {})

    assert plan.detected is DrafterKind.HYBRID_DFLASH_MARKOV
    assert plan.primary_backend == "dspark"
    assert plan.primary_ref == hybrid
    assert plan.fallback_ref == fallback
    assert plan.reason == "auto_detected_hybrid_dflash_markov"


def test_hybrid_checkpoint_cannot_be_its_own_dflash_fallback(tmp_path):
    hybrid = _checkpoint(
        tmp_path / "hybrid",
        {
            "architectures": ["DFlashDraftModel"],
            "markov_rank": 16,
            "dflash_config": {"markov_rank": 16},
        },
    )
    tier = SimpleNamespace(
        target_model="unregistered-target",
        draft_model=hybrid,
        draft_fallback_model=hybrid,
        drafter_backend="auto",
        drafter_strict=False,
    )

    with pytest.raises(ValueError, match="must not reuse"):
        plan_drafter(tier, {})


def test_explicit_dflash_replaces_hybrid_with_pure_checkpoint(tmp_path):
    hybrid = _checkpoint(
        tmp_path / "hybrid",
        {
            "architectures": ["DFlashDraftModel"],
            "markov_rank": 16,
            "dflash_config": {"markov_rank": 16},
        },
    )
    fallback = _checkpoint(
        tmp_path / "fallback",
        {"architectures": ["DFlashDraftModel"], "dflash_config": {}},
    )
    tier = SimpleNamespace(
        target_model="unregistered-target",
        draft_model=hybrid,
        draft_fallback_model=fallback,
        drafter_backend="dflash",
        drafter_strict=False,
    )

    plan = plan_drafter(tier, {})

    assert plan.primary_backend == "dflash"
    assert plan.primary_ref == fallback
    assert plan.fallback_ref is None
    assert plan.reason.endswith("using_compatible_fallback")


def test_strict_mode_can_be_forced_for_benchmarks(monkeypatch, tmp_path):
    pure = _checkpoint(
        tmp_path / "pure",
        {"architectures": ["DFlashDraftModel"], "dflash_config": {}},
    )
    tier = SimpleNamespace(
        target_model="unregistered-target",
        draft_model=pure,
        draft_fallback_model=None,
        drafter_backend="auto",
        drafter_strict=False,
    )
    monkeypatch.setenv("MIO_DRAFTER_STRICT", "1")

    assert plan_drafter(tier, {}).strict is True


def test_local_dspark_without_configured_fallback_never_rebuilds_remote_repo(
    monkeypatch,
    tmp_path,
):
    """The --no-fallback pull shape must not plan ``org/dflash`` remotely."""
    dspark = _checkpoint(
        tmp_path / "dspark",
        {"architectures": ["Qwen3DSparkModel"], "dspark_config": {}},
    )
    models = tmp_path / "models"
    drafts = tmp_path / "spd"
    drafts.mkdir()
    entry = registry.ModelEntry(
        target_repo="org/target",
        target_local="target",
        draft_repo="org/dflash",
        draft_local="dflash",
        dspark_repo="org/dspark",
        dspark_local="dspark",
        adapter="qwen3_5",
        default_tier="large",
        context_window=4096,
        max_output_tokens=128,
    )
    monkeypatch.setattr(registry, "KNOWN_MODELS", {"stack": entry})
    monkeypatch.setattr(registry, "models_dir", lambda: models)
    monkeypatch.setattr(registry, "spd_dir", lambda: drafts)
    monkeypatch.setattr(
        registry,
        "_model_path_is_complete",
        lambda _path, **_kwargs: False,
    )
    tier = SimpleNamespace(
        target_model="org/target",
        draft_model=dspark,
        draft_fallback_model=None,
        drafter_backend="auto",
        drafter_strict=False,
    )

    plan = plan_drafter(tier, {})

    assert plan.primary_backend == "dspark"
    assert plan.fallback_ref is None


def test_explicit_remote_dflash_fallback_remains_allowed(tmp_path, monkeypatch):
    dspark = _checkpoint(
        tmp_path / "dspark",
        {"architectures": ["Qwen3DSparkModel"], "dspark_config": {}},
    )
    monkeypatch.setattr(registry, "KNOWN_MODELS", {})
    monkeypatch.setattr(registry, "spd_dir", lambda: tmp_path / "missing-spd")
    tier = SimpleNamespace(
        target_model="org/target",
        draft_model=dspark,
        draft_fallback_model="org/pure-dflash",
        drafter_backend="auto",
        drafter_strict=False,
    )

    plan = plan_drafter(tier, {})

    assert plan.fallback_ref == "org/pure-dflash"
