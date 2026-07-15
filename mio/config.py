"""Configuration types, persistence, and defaults for Mio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path
from typing import Any


@dataclass
class TierConfig:
    """Configuration for a single model tier."""

    name: str
    target_model: str
    draft_model: str
    context_window: int
    max_output_tokens: int
    tq_bits: int = 16  # 16 = TQ off (baseline KVCache). {2, 3, 4} enable TurboQuant.
    pq_bits: int = 4   # 4 = PolarQuant 4-bit (default, ~3.8x KV compression, zero speed loss). 16 = off.
    bmp_paths: int = 1  # 1 = vanilla DFlash. K>=2 enables BMP-DFlash K-path verify.
    # DDTree (Diffusion Draft Tree) node budget. 0 = off (vanilla DFlash).
    # >0 = verify N candidate tree nodes per cycle via tree attention + parent-
    # indexed GatedDelta Metal kernels. Hybrid_gdn models only (Qwen3.5-27B,
    # Qwen3.5/3.6-35B-A3B). Incompatible with PolarQuant/TurboQuant — when
    # enabled, the engine automatically swaps PQ/TQ for mlx_lm QuantizedKVCache
    # (8-bit) so some KV compression is retained. Default 0 (opt-in) because
    # the gain is content-dependent (+10-15% on code, ~0% on creative prose).
    ddtree_budget: int = 0
    tq_group_size: int = 64
    tq_use_rotation: bool = True
    tq_use_normalization: bool = True
    tq_use_qjl: bool = False
    pq_group_size: int = 64
    # Qwen3.5/3.6 recommended: thinking-mode coding = 0.6, general = 1.0
    # Never use greedy (0.0) — causes degradation and infinite repetitions.
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class MioConfig:
    """Top-level Mio configuration."""

    tiers: dict[str, TierConfig] = field(default_factory=dict)
    active_tiers: list[str] = field(default_factory=lambda: ["large-moe"])
    tandem: bool = False
    port: int = 9090
    host: str = "0.0.0.0"
    config_dir: Path = field(default_factory=lambda: Path.home() / ".mio")

    @classmethod
    def default(cls) -> MioConfig:
        """Create an independent config from the current model registry tiers."""
        from mio.models.registry import DEFAULT_TIERS

        # ``dict(DEFAULT_TIERS)`` only copies the mapping.  The mutable
        # TierConfig instances would still be shared with the registry, so a
        # one-shot CLI override (for example ``--tq4``) could change defaults
        # for every config created later in the same process.
        return cls(tiers={name: replace(tier) for name, tier in DEFAULT_TIERS.items()})


def default_config_path() -> Path:
    """Return the canonical per-user configuration path."""
    return Path.home() / ".mio" / "config.json"


_TIER_FIELDS = {item.name for item in fields(TierConfig)}


def _normalise_cache_mode(tier: TierConfig) -> TierConfig:
    """Make legacy configs with both PQ and TQ active deterministic.

    The runtime gives PolarQuant precedence, which made an old wizard-created
    ``tq_bits=4, pq_bits=4`` configuration silently run PQ instead of TQ.
    A configured TQ bit-width is explicit, so it wins during migration.
    """
    if tier.tq_bits in (2, 3, 4) and tier.pq_bits in (2, 3, 4):
        tier.pq_bits = 16
    return tier


def _tier_from_data(name: str, data: Any, fallback: TierConfig | None) -> TierConfig | None:
    """Deserialize one tier while tolerating older/newer config schemas."""
    if not isinstance(data, dict):
        return None

    values = asdict(fallback) if fallback is not None else {}
    values.update({key: value for key, value in data.items() if key in _TIER_FIELDS})
    values["name"] = name

    try:
        return _normalise_cache_mode(TierConfig(**values))
    except (TypeError, ValueError):
        # A malformed custom tier must not make every Mio command unusable.
        return None


def load_config(path: Path | None = None) -> MioConfig:
    """Load persisted config, falling back safely to registry defaults.

    With no explicit path this reads ``~/.mio/config.json``.  Missing,
    unreadable, malformed, and legacy top-level-only files are all supported.
    Persisted tiers overlay current registry defaults so newly introduced
    fields receive their modern default values.
    """
    config = MioConfig.default()
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    config.config_dir = config_path.parent

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return config

    if not isinstance(data, dict):
        return config

    tiers_data = data.get("tiers")
    if isinstance(tiers_data, dict):
        for name, tier_data in tiers_data.items():
            if not isinstance(name, str) or not name:
                continue
            tier = _tier_from_data(name, tier_data, config.tiers.get(name))
            if tier is not None:
                config.tiers[name] = tier

    active_tiers = data.get("active_tiers")
    if isinstance(active_tiers, list) and active_tiers and all(
        isinstance(name, str) for name in active_tiers
    ):
        valid_active_tiers = [name for name in active_tiers if name in config.tiers]
        if valid_active_tiers:
            config.active_tiers = valid_active_tiers
    if isinstance(data.get("tandem"), bool):
        config.tandem = data["tandem"]
    if isinstance(data.get("port"), int) and not isinstance(data["port"], bool):
        config.port = data["port"]
    if isinstance(data.get("host"), str) and data["host"]:
        config.host = data["host"]
    return config


def save_config(config: MioConfig, path: Path) -> None:
    """Persist the complete user configuration in a forward-compatible form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "active_tiers": config.active_tiers,
        "tandem": config.tandem,
        "port": config.port,
        "host": config.host,
        "tiers": {},
    }
    for name, tier in config.tiers.items():
        serializable_tier = replace(tier)
        _normalise_cache_mode(serializable_tier)
        tier_data = asdict(serializable_tier)
        tier_data["target_model"] = str(tier_data["target_model"])
        tier_data["draft_model"] = str(tier_data["draft_model"])
        data["tiers"][name] = tier_data

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
