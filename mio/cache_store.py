"""KV cache persistence: save/restore cache state to disk."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    key: str
    path: Path
    token_count: int
    created: float
    size_bytes: int


class CacheStore:
    """Persistent KV cache storage with LRU eviction.

    Saves quantized KV cache arrays to disk so sessions can be resumed
    without re-processing the full context.
    """

    def __init__(self, cache_dir: Path | None = None, max_size_gb: float = 10.0) -> None:
        self.cache_dir = cache_dir or Path.home() / ".mio" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _total_size(self) -> int:
        return sum(e.get("size_bytes", 0) for e in self._index.values())

    def _evict_lru(self) -> None:
        """Remove oldest entries until under max size."""
        while self._total_size() > self.max_size_bytes and self._index:
            oldest_key = min(self._index, key=lambda k: (self._index[k].get("created", 0), k))
            entry_dir = self.cache_dir / oldest_key
            if entry_dir.exists():
                shutil.rmtree(entry_dir)
            del self._index[oldest_key]
        self._save_index()

    @staticmethod
    def make_key(cache_key: str) -> str:
        """Hash a user-provided cache key to a safe directory name."""
        return hashlib.sha256(cache_key.encode()).hexdigest()[:16]

    def has(self, cache_key: str) -> bool:
        """Check if a cache entry exists."""
        key = self.make_key(cache_key)
        return key in self._index and (self.cache_dir / key).exists()

    def save(self, cache_key: str, kv_arrays: list[Any], token_count: int) -> None:
        """Save KV cache arrays to disk.

        Args:
            cache_key: User-provided session identifier.
            kv_arrays: List of per-layer cache objects (must support mx.save).
            token_count: Number of tokens in the cached state.
        """
        try:
            import mlx.core as mx
        except ImportError:
            return

        key = self.make_key(cache_key)
        entry_dir = self.cache_dir / key
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Save each layer's cache
        try:
            for i, layer_cache in enumerate(kv_arrays):
                layer_path = entry_dir / f"layer_{i:03d}.npz"
                if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                    mx.savez(str(layer_path), keys=layer_cache.keys, values=layer_cache.values)
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            return

        # Calculate size
        size = sum(f.stat().st_size for f in entry_dir.rglob("*") if f.is_file())

        self._index[key] = {
            "cache_key": cache_key,
            "token_count": token_count,
            "created": time.time(),
            "size_bytes": size,
        }
        self._save_index()
        self._evict_lru()

    def load(self, cache_key: str) -> tuple[list[Any] | None, int]:
        """Load KV cache arrays from disk.

        Returns (kv_arrays, token_count) or (None, 0) if not found.
        """
        key = self.make_key(cache_key)
        if key not in self._index:
            return None, 0

        entry_dir = self.cache_dir / key
        if not entry_dir.exists():
            del self._index[key]
            self._save_index()
            return None, 0

        try:
            import mlx.core as mx
        except ImportError:
            return None, 0

        layer_files = sorted(entry_dir.glob("layer_*.npz"))
        if not layer_files:
            return None, 0

        kv_arrays = []
        for layer_path in layer_files:
            data = mx.load(str(layer_path))
            kv_arrays.append(data)

        token_count = self._index[key].get("token_count", 0)
        # Update access time for LRU
        self._index[key]["created"] = time.time()
        self._save_index()

        return kv_arrays, token_count

    def delete(self, cache_key: str) -> None:
        key = self.make_key(cache_key)
        entry_dir = self.cache_dir / key
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        self._index.pop(key, None)
        self._save_index()

    def list_entries(self) -> list[dict]:
        return [
            {"key": v["cache_key"], "tokens": v["token_count"],
             "size_mb": v["size_bytes"] / (1024**2)}
            for v in self._index.values()
        ]

    def stats(self) -> dict:
        return {
            "entries": len(self._index),
            "total_size_gb": self._total_size() / (1024**3),
            "max_size_gb": self.max_size_bytes / (1024**3),
        }
