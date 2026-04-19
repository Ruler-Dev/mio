"""Model loading, unloading, and tier management."""

from __future__ import annotations

import threading
from typing import Any

from mio.config import MioConfig, TierConfig
from mio.engine import GenerationMetrics, MioEngine


class ModelManager:
    """Manages multiple engine tiers and their lifecycle."""

    def __init__(self, config: MioConfig) -> None:
        self.config = config
        self._engines: dict[str, MioEngine] = {}
        self._lock = threading.Lock()

    def load_tier(self, tier_name: str) -> None:
        """Load a tier's models into VRAM."""
        with self._lock:
            if tier_name in self._engines and self._engines[tier_name].is_loaded:
                return
            self._load_tier_locked(tier_name)

    def _load_tier_locked(self, tier_name: str) -> None:
        """Internal: load tier while holding lock."""

        tier_config = self.config.tiers.get(tier_name)
        if not tier_config:
            raise ValueError(
                f"Unknown tier '{tier_name}'. Available: {list(self.config.tiers.keys())}"
            )

        engine = MioEngine(tier_config=tier_config)
        engine.load()
        self._engines[tier_name] = engine

    def unload_tier(self, tier_name: str) -> None:
        """Release VRAM for a tier."""
        engine = self._engines.pop(tier_name, None)
        if engine:
            engine.unload()

    def get_engine(self, tier_name: str) -> MioEngine:
        """Get loaded engine for a tier. Raises if not loaded."""
        engine = self._engines.get(tier_name)
        if not engine or not engine.is_loaded:
            raise RuntimeError(
                f"Tier '{tier_name}' not loaded. Call load_tier() first."
            )
        return engine

    def loaded_tiers(self) -> list[str]:
        """Which tiers are currently in VRAM."""
        return [name for name, eng in self._engines.items() if eng.is_loaded]

    def load_active_tiers(self) -> None:
        """Load all tiers specified in config.active_tiers."""
        for tier_name in self.config.active_tiers:
            self.load_tier(tier_name)

    def unload_all(self) -> None:
        """Unload all tiers."""
        for name in list(self._engines.keys()):
            self.unload_tier(name)

    def vram_usage(self) -> dict[str, float]:
        """Estimated VRAM usage per loaded tier in GB."""
        try:
            import mlx.core as mx

            total = mx.metal.get_peak_memory() / (1024**3)
            # Approximate per-tier split based on model size
            loaded = self.loaded_tiers()
            if not loaded:
                return {}
            return {name: total / len(loaded) for name in loaded}
        except Exception:
            return {name: 0.0 for name in self.loaded_tiers()}

    def total_vram_gb(self) -> float:
        """Total VRAM currently used."""
        try:
            import mlx.core as mx

            return mx.metal.get_peak_memory() / (1024**3)
        except Exception:
            return 0.0

    def status(self) -> dict[str, Any]:
        """Full status report."""
        loaded = self.loaded_tiers()
        return {
            "loaded_tiers": loaded,
            "available_tiers": list(self.config.tiers.keys()),
            "tandem": self.config.tandem,
            "vram_gb": self.total_vram_gb(),
            "engines": {
                name: {
                    "model": self._engines[name].tier_config.target_model,
                    "draft": self._engines[name].tier_config.draft_model,
                    "context_window": self._engines[name].tier_config.context_window,
                    "last_gen_tps": self._engines[name].last_metrics.generation_tps,
                }
                for name in loaded
            },
        }

    def get_model_names(self) -> list[str]:
        """Return API model names for loaded tiers."""
        names = []
        for tier_name in self.loaded_tiers():
            names.append(f"mio-{tier_name}")
        if self.config.tandem and len(self.loaded_tiers()) > 1:
            names.append("mio-auto")
        return names
