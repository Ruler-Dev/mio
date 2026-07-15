"""Tests for the Mio engine."""

from __future__ import annotations

import json

from mio.config import TierConfig, MioConfig
from mio.models.registry import (
    DEFAULT_TIERS, KNOWN_MODELS, SUPPORTED_ADAPTERS,
    _model_path_is_complete, get_supported_models, models_dir, spd_dir, resolve_model_path,
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
            e for e in KNOWN_MODELS.values()
            if e.target_repo == tier.target_model
            or e.resolve_target() == tier.target_model
        ]
        assert matching, f"No known model for tier {name}: {tier.target_model}"
        assert matching[0].adapter in SUPPORTED_ADAPTERS, (
            f"Tier {name} uses unsupported adapter: {matching[0].adapter}"
        )


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


def test_model_path_requires_every_indexed_shard(tmp_path):
    """Indexed checkpoints become complete only after every named shard exists."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    index = {
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    assert not _model_path_is_complete(tmp_path)

    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    assert _model_path_is_complete(tmp_path)


def test_qwen36_dense_dflash_pair_is_registered():
    entry = KNOWN_MODELS["qwen3.6-27b-unsloth"]
    assert entry.target_repo == "Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx"
    assert entry.draft_repo == "z-lab/Qwen3.6-27B-DFlash"
    assert entry.context_window == 262144


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
