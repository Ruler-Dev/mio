"""Tests for the Mio engine."""

from __future__ import annotations

import json

import mio.models.registry as registry
from mio.config import TierConfig, MioConfig
from mio.models.registry import (
    DEFAULT_TIERS,
    KNOWN_MODELS,
    SUPPORTED_ADAPTERS,
    _model_path_is_complete,
    get_supported_models,
    models_dir,
    spd_dir,
    resolve_model_path,
)


def test_default_tiers_exist():
    """Default tiers should be defined."""
    assert "large" in DEFAULT_TIERS
    assert "medium" in DEFAULT_TIERS
    assert "small" in DEFAULT_TIERS


def test_default_tiers_use_supported_adapters():
    """All default tiers should use adapters that are implemented."""
    for name, tier in DEFAULT_TIERS.items():
        # Find matching model entry by repo or local path
        matching = [
            e
            for e in KNOWN_MODELS.values()
            if e.target_repo == tier.target_model or e.resolve_target() == tier.target_model
        ]
        assert matching, f"No known model for tier {name}: {tier.target_model}"
        assert matching[0].adapter in SUPPORTED_ADAPTERS, f"Tier {name} uses unsupported adapter: {matching[0].adapter}"


def test_tier_config_defaults():
    """TierConfig should have sensible TurboQuant defaults."""
    tier = TierConfig(
        name="test",
        target_model="test/model",
        draft_model="test/draft",
        context_window=8192,
        max_output_tokens=2048,
    )
    assert tier.tq_bits == 16  # TQ off by default; opt-in via --tq4
    assert tier.tq_group_size == 64
    assert tier.tq_use_rotation is True
    assert tier.tq_use_normalization is True
    assert tier.tq_use_qjl is False
    assert tier.temperature == 0.0  # preserves the exact DFlash fast path


def test_config_default():
    """MioConfig.default() should create valid config."""
    config = MioConfig.default()
    assert config.active_tiers == ["large-moe"]
    assert config.tandem is False
    assert config.port == 9090
    assert "large" in config.tiers
    assert "medium" in config.tiers
    assert "small" in config.tiers


def test_supported_models_filter():
    """get_supported_models should only return models with working adapters."""
    supported = get_supported_models()
    for key, entry in supported.items():
        assert entry.adapter in SUPPORTED_ADAPTERS, f"{key} has unsupported adapter"


def test_known_models_have_draft():
    """All known models should have a draft model defined."""
    for key, entry in KNOWN_MODELS.items():
        assert entry.draft_repo, f"{key} missing draft_repo"
        assert entry.draft_local, f"{key} missing draft_local"


def test_default_large_moe_is_35b_a3b():
    """Default large-moe tier should use the 35B-A3B PARO model."""
    large_moe = DEFAULT_TIERS["large-moe"]
    assert "35B-A3B" in large_moe.target_model or "PARO" in large_moe.target_model


def test_model_entry_has_local_names():
    """All entries should have target_local and draft_local set."""
    for key, entry in KNOWN_MODELS.items():
        assert entry.target_local, f"{key} missing target_local"
        assert entry.draft_local, f"{key} missing draft_local"


def test_resolve_model_path_local_vs_hf():
    """resolve_model_path should return local path if dir exists, else name as-is."""
    # A name that definitely doesn't exist locally → returns as-is (HF fallback)
    result = resolve_model_path("nonexistent-model-xyz", kind="target")
    assert result == "nonexistent-model-xyz"


def test_model_path_requires_weight_files(tmp_path):
    """A config-only or interrupted Hugging Face download is not loadable."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not _model_path_is_complete(tmp_path)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert _model_path_is_complete(tmp_path)


def test_target_completeness_requires_tokenizer_and_chat_template(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    # Drafts are valid with weights/config only, targets are not.
    assert _model_path_is_complete(tmp_path)
    assert not _model_path_is_complete(tmp_path, require_tokenizer=True)

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"processor_class": "TextProcessor"}),
        encoding="utf-8",
    )
    assert not _model_path_is_complete(tmp_path, require_tokenizer=True)

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}),
        encoding="utf-8",
    )
    assert _model_path_is_complete(tmp_path, require_tokenizer=True)


def test_target_completeness_accepts_bpe_and_processor_chat_template(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"text_processor": {"chat_template": "{{ messages }}"}}),
        encoding="utf-8",
    )

    assert _model_path_is_complete(tmp_path, require_tokenizer=True)


def test_model_path_requires_every_indexed_shard(tmp_path):
    """Indexed checkpoints become complete only after every named shard exists."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    index = {
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    assert not _model_path_is_complete(tmp_path)

    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    assert _model_path_is_complete(tmp_path)


def test_qwen36_dense_dflash_pair_is_registered():
    entry = KNOWN_MODELS["qwen3.6-27b-unsloth"]
    assert entry.target_repo == "Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx"
    assert entry.draft_repo == "z-lab/Qwen3.6-27B-DFlash"
    assert entry.dspark_repo == "Avesed/Qwen3.6-27B-DSpark"
    assert entry.dspark_local == "Qwen3.6-27B-DSpark"
    assert entry.dspark_max_draft_tokens == 3
    assert entry.dspark_lookup_drafts is False
    assert entry.context_window == 262144


def test_registry_selects_local_dspark_without_remote_dflash(monkeypatch, tmp_path):
    """A target+DSpark-only pull is immediately usable without remote fallback."""
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
    models = tmp_path / "models"
    drafts = tmp_path / "spd"
    complete = {models / "target", drafts / "dspark"}
    monkeypatch.setattr(registry, "models_dir", lambda: models)
    monkeypatch.setattr(registry, "spd_dir", lambda: drafts)
    monkeypatch.setattr(
        registry,
        "_model_path_is_complete",
        lambda path, **_kwargs: path in complete,
    )
    monkeypatch.setattr(registry, "KNOWN_MODELS", {"stack": entry})

    tier = registry._make_tier("large", "stack")

    assert entry.resolve_target() == str(models / "target")
    assert entry.resolve_draft() == "org/dflash"
    assert entry.resolve_dspark() == str(drafts / "dspark")
    assert registry._entry_is_local("stack") is True
    assert tier.draft_model == str(drafts / "dspark")
    assert tier.draft_fallback_model is None


def test_registry_selects_local_dflash_when_dspark_was_skipped(monkeypatch, tmp_path):
    """A --no-dspark pull uses local DFlash and never fetches DSpark at startup."""
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
    models = tmp_path / "models"
    drafts = tmp_path / "spd"
    complete = {models / "target", drafts / "dflash"}
    monkeypatch.setattr(registry, "models_dir", lambda: models)
    monkeypatch.setattr(registry, "spd_dir", lambda: drafts)
    monkeypatch.setattr(
        registry,
        "_model_path_is_complete",
        lambda path, **_kwargs: path in complete,
    )
    monkeypatch.setattr(registry, "KNOWN_MODELS", {"stack": entry})

    tier = registry._make_tier("large", "stack")

    assert entry.resolve_target() == str(models / "target")
    assert entry.resolve_draft() == str(drafts / "dflash")
    assert entry.resolve_dspark() is None
    assert registry._entry_is_local("stack") is True
    assert tier.draft_model == str(drafts / "dflash")
    assert tier.draft_fallback_model is None


def test_model_entry_uses_real_repo_ids_when_local_copies_are_incomplete(
    monkeypatch,
    tmp_path,
):
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
    monkeypatch.setattr(registry, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(registry, "spd_dir", lambda: tmp_path / "spd")
    monkeypatch.setattr(
        registry,
        "_model_path_is_complete",
        lambda _path, **_kwargs: False,
    )

    assert entry.resolve_target() == "org/target"
    assert entry.resolve_draft() == "org/dflash"
    assert entry.resolve_dspark() is None


def test_models_dir_and_spd_dir():
    """models_dir and spd_dir should point inside mio project."""
    assert models_dir().name == "models"
    assert spd_dir().name == "spd"
    assert models_dir().parent == spd_dir().parent  # Both under same root


def test_tier_context_windows_ordered():
    """Large > medium > small context windows."""
    large = DEFAULT_TIERS["large"]
    medium = DEFAULT_TIERS["medium"]
    small = DEFAULT_TIERS["small"]
    assert large.context_window > medium.context_window > small.context_window
    assert large.max_output_tokens > medium.max_output_tokens > small.max_output_tokens
