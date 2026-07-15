"""Model loading, unloading, and tier management."""

from __future__ import annotations

import threading
from typing import Any

from mio.config import MioConfig
from mio.engine import MioEngine


class ModelManager:
    """Manages multiple engine tiers and their lifecycle."""

    def __init__(self, config: MioConfig) -> None:
        self.config = config
        self._engines: dict[str, MioEngine] = {}
        # Lifecycle operations can call other snapshot helpers (for example
        # ``get_model_names`` needs the loaded-tier snapshot).  An RLock keeps
        # those compositions atomic without deadlocking on a nested acquire.
        self._lock = threading.RLock()

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
        with self._lock:
            engine = self._engines.pop(tier_name, None)
            if engine:
                # Keep the lifecycle lock held until teardown completes so a
                # concurrent reader cannot receive a half-unloaded engine.
                engine.unload()

    def get_engine(self, tier_name: str) -> MioEngine:
        """Get loaded engine for a tier. Raises if not loaded."""
        with self._lock:
            engine = self._engines.get(tier_name)
            if not engine or not engine.is_loaded:
                raise RuntimeError(
                    f"Tier '{tier_name}' not loaded. Call load_tier() first."
                )
            return engine

    def loaded_tiers(self) -> list[str]:
        """Which tiers are currently in VRAM."""
        with self._lock:
            return [name for name, eng in self._engines.items() if eng.is_loaded]

    def load_active_tiers(self) -> None:
        """Load all tiers specified in config.active_tiers."""
        for tier_name in self.config.active_tiers:
            self.load_tier(tier_name)

    def unload_all(self) -> None:
        """Unload all tiers."""
        with self._lock:
            for name in list(self._engines):
                self.unload_tier(name)

    def vram_usage(self) -> dict[str, float]:
        """Estimated VRAM usage per loaded tier in GB."""
        with self._lock:
            loaded = self.loaded_tiers()
            try:
                import mlx.core as mx

                total = self._peak_memory_bytes(mx) / (1024**3)
                # Approximate per-tier split based on model size
                if not loaded:
                    return {}
                return {name: total / len(loaded) for name in loaded}
            except Exception:
                return {name: 0.0 for name in loaded}

    def total_vram_gb(self) -> float:
        """Total VRAM currently used."""
        try:
            import mlx.core as mx

            return self._peak_memory_bytes(mx) / (1024**3)
        except Exception:
            return 0.0

    @staticmethod
    def _peak_memory_bytes(mx: Any) -> float:
        """Read MLX peak memory without relying on the deprecated Metal alias."""
        getter = getattr(mx, "get_peak_memory", None)
        if getter is None:  # Compatibility with MLX releases predating the top-level API.
            getter = mx.metal.get_peak_memory
        return float(getter())

    def status(self) -> dict[str, Any]:
        """Full status report."""
        with self._lock:
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
                }
            }

    def get_model_names(self) -> list[str]:
        """Return API model names for loaded tiers."""
        with self._lock:
            loaded = self.loaded_tiers()
            names = [f"mio-{tier_name}" for tier_name in loaded]
            if self.config.tandem and len(loaded) > 1:
                names.append("mio-auto")
            return names
