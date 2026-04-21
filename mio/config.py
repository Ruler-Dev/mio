"""Configuration types and defaults for Mio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
        """Create config with default Qwen 3.5 tiers."""
        from mio.models.registry import DEFAULT_TIERS

        return cls(tiers=dict(DEFAULT_TIERS))


def load_config(path: Path | None = None) -> MioConfig:
    """Load config from file, falling back to defaults."""
    config = MioConfig.default()
    if path and path.exists():
        import json

        data = json.loads(path.read_text())
        if "active_tiers" in data:
            config.active_tiers = data["active_tiers"]
        if "tandem" in data:
            config.tandem = data["tandem"]
        if "port" in data:
            config.port = data["port"]
        if "host" in data:
            config.host = data["host"]
    return config


def save_config(config: MioConfig, path: Path) -> None:
    """Save config to file."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "active_tiers": config.active_tiers,
        "tandem": config.tandem,
        "port": config.port,
        "host": config.host,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
