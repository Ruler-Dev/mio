"""Crash-safe persistence for MLX KV cache state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mio.paths import mio_home
from mio.persistence import atomic_write_json


_CACHE_FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"


@dataclass
class CacheEntry:
    key: str
    path: Path
    model_id: str
    token_count: int
    layer_count: int
    created: float
    size_bytes: int


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class CacheStore:
    """Persistent KV cache storage with transactional replacement and LRU.

    Each save is written to a new generation directory.  Only after every
    layer and its manifest are durable does one atomic index replacement make
    that generation visible.  A failed shorter overwrite therefore preserves
    the previous complete cache and can never expose stale trailing layers.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_size_gb: float = 10.0,
        model_id: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir or mio_home() / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.model_id = str(model_id) if model_id else None
        self._index_path = self.cache_dir / "index.json"
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, Any]] = self._load_index()
        self._reconcile_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self._index_path.exists():
            return {}
        try:
            value = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(key): record
            for key, record in value.items()
            if isinstance(record, dict)
        }

    def _save_index(self) -> None:
        atomic_write_json(self._index_path, self._index, sort_keys=True)

    def _total_size(self) -> int:
        return sum(
            int(entry.get("size_bytes", 0) or 0)
            for entry in self._index.values()
        )

    @staticmethod
    def make_key(cache_key: str) -> str:
        """Hash a user-provided cache key to a safe directory prefix."""
        return hashlib.sha256(cache_key.encode()).hexdigest()[:16]

    def _entry_dir(self, key: str, record: dict[str, Any]) -> Path | None:
        generation = record.get("generation")
        if (
            not isinstance(generation, str)
            or Path(generation).name != generation
            or not generation.startswith(f"{key}-")
        ):
            return None
        return self.cache_dir / generation

    @staticmethod
    def _remove_tree(path: Path | None) -> None:
        if path is None:
            return
        if path.is_symlink():
            path.unlink(missing_ok=True)
        elif not path.exists():
            return
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    def _read_manifest(
        self,
        key: str,
        record: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]] | None:
        entry_dir = self._entry_dir(key, record)
        if entry_dir is None or entry_dir.is_symlink() or not entry_dir.is_dir():
            return None
        manifest_path = entry_dir / _MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(manifest, dict):
            return None

        layer_count = manifest.get("layer_count")
        token_count = manifest.get("token_count")
        model_id = manifest.get("model")
        layers = manifest.get("layers")
        if (
            manifest.get("version") != _CACHE_FORMAT_VERSION
            or manifest.get("cache_key_hash") != key
            or isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count <= 0
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
            or not isinstance(model_id, str)
            or not model_id
            or not isinstance(layers, list)
            or len(layers) != layer_count
        ):
            return None

        expected_layers = [f"layer_{index:03d}.npz" for index in range(layer_count)]
        if layers != expected_layers:
            return None
        if (
            record.get("model") != model_id
            or record.get("token_count") != token_count
            or record.get("layer_count") != layer_count
        ):
            return None

        expected_files = {_MANIFEST_NAME, *expected_layers}
        try:
            children = list(entry_dir.iterdir())
        except OSError:
            return None
        if {child.name for child in children} != expected_files:
            return None
        if any(child.is_symlink() or not child.is_file() for child in children):
            return None
        return entry_dir, manifest

    def _reconcile_index(self) -> None:
        """Drop incomplete/legacy index pointers; orphan data stays invisible."""
        with self._lock:
            valid = {
                key: record
                for key, record in self._index.items()
                if self._read_manifest(key, record) is not None
            }
            if len(valid) != len(self._index):
                self._index = valid
                self._save_index()

    def _cleanup_generations(self, key: str, *, keep: str | None = None) -> None:
        for candidate in self.cache_dir.glob(f"{key}-*"):
            if candidate.name != keep:
                self._remove_tree(candidate)
        # Remove the pre-manifest layout after a successful publication.
        self._remove_tree(self.cache_dir / key)

    def has(self, cache_key: str, *, model_id: str | None = None) -> bool:
        """Return whether a complete, compatible cache generation exists."""
        key = self.make_key(cache_key)
        expected_model = model_id or self.model_id
        with self._lock:
            record = self._index.get(key)
            if record is None:
                return False
            loaded = self._read_manifest(key, record)
            if loaded is None:
                return False
            return expected_model is None or loaded[1]["model"] == expected_model

    def save(
        self,
        cache_key: str,
        kv_arrays: list[Any],
        token_count: int,
        *,
        model_id: str | None = None,
    ) -> None:
        """Transactionally replace a cache generation.

        ``model_id`` is persisted in both the manifest and index.  Callers that
        know the concrete checkpoint should pass it (or set it on the store)
        so a cache can never be restored into a different model.
        """
        try:
            import mlx.core as mx
        except ImportError:
            return

        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            return
        if not kv_arrays:
            return
        resolved_model = model_id or self.model_id or "unknown"
        key = self.make_key(cache_key)
        generation = f"{key}-{uuid.uuid4().hex}"
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{key}-", suffix=".tmp", dir=self.cache_dir)
        )
        published = self.cache_dir / generation
        created = time.time()
        layer_names = [f"layer_{index:03d}.npz" for index in range(len(kv_arrays))]
        index_committed = False

        try:
            for layer_cache, layer_name in zip(kv_arrays, layer_names, strict=True):
                if not hasattr(layer_cache, "keys") or not hasattr(layer_cache, "values"):
                    raise TypeError("cache layer does not expose keys and values")
                layer_path = temporary / layer_name
                mx.savez(
                    str(layer_path),
                    keys=layer_cache.keys,
                    values=layer_cache.values,
                )
                if layer_path.is_symlink() or not layer_path.is_file():
                    raise OSError(f"MLX did not create {layer_name}")
                with layer_path.open("rb") as handle:
                    os.fsync(handle.fileno())

            manifest = {
                "version": _CACHE_FORMAT_VERSION,
                "cache_key_hash": key,
                "model": resolved_model,
                "token_count": token_count,
                "layer_count": len(layer_names),
                "layers": layer_names,
                "created": created,
            }
            atomic_write_json(temporary / _MANIFEST_NAME, manifest, sort_keys=True)
            _fsync_directory(temporary)
            size = sum(
                child.stat().st_size
                for child in temporary.iterdir()
                if child.is_file() and not child.is_symlink()
            )
            new_record = {
                "cache_key": cache_key,
                "generation": generation,
                "model": resolved_model,
                "token_count": token_count,
                "layer_count": len(layer_names),
                "created": created,
                "last_access": created,
                "size_bytes": size,
            }

            with self._lock:
                # Publication, index swap, and stale-generation cleanup are one
                # in-process critical section. Concurrent saves may prepare
                # independent temporary trees, but cannot delete a generation
                # another writer is about to publish.
                os.replace(temporary, published)
                _fsync_directory(self.cache_dir)
                previous_index = self._index
                previous_record = previous_index.get(key)
                next_index = dict(previous_index)
                next_index[key] = new_record
                evicted: list[tuple[str, dict[str, Any]]] = []
                while (
                    sum(int(item.get("size_bytes", 0) or 0) for item in next_index.values())
                    > self.max_size_bytes
                    and next_index
                ):
                    oldest_key = min(
                        next_index,
                        key=lambda item_key: (
                            float(next_index[item_key].get("last_access", 0) or 0),
                            item_key,
                        ),
                    )
                    evicted.append((oldest_key, next_index.pop(oldest_key)))

                self._index = next_index
                try:
                    self._save_index()
                except Exception:
                    self._index = previous_index
                    self._remove_tree(published)
                    return
                index_committed = True

                keep = generation if key in self._index else None
                self._cleanup_generations(key, keep=keep)
                if previous_record is not None:
                    old_dir = self._entry_dir(key, previous_record)
                    if old_dir != published:
                        self._remove_tree(old_dir)
                for victim_key, victim in evicted:
                    if victim_key != key:
                        self._cleanup_generations(victim_key)
                        self._remove_tree(self._entry_dir(victim_key, victim))
        except Exception:
            if not index_committed:
                self._remove_tree(published)
        finally:
            self._remove_tree(temporary)

    def load(
        self,
        cache_key: str,
        *,
        model_id: str | None = None,
    ) -> tuple[list[Any] | None, int]:
        """Load a complete compatible generation or return ``(None, 0)``."""
        try:
            import mlx.core as mx
        except ImportError:
            return None, 0

        key = self.make_key(cache_key)
        expected_model = model_id or self.model_id
        with self._lock:
            record = self._index.get(key)
            if record is None:
                return None, 0
            loaded = self._read_manifest(key, record)
            if loaded is None:
                return None, 0
            entry_dir, manifest = loaded
            if expected_model is not None and manifest["model"] != expected_model:
                return None, 0

            try:
                kv_arrays = [
                    mx.load(str(entry_dir / layer_name))
                    for layer_name in manifest["layers"]
                ]
            except Exception:
                return None, 0

            previous_access = record.get("last_access")
            record["last_access"] = time.time()
            try:
                self._save_index()
            except Exception:
                record["last_access"] = previous_access
            return kv_arrays, int(manifest["token_count"])

    def delete(self, cache_key: str) -> None:
        key = self.make_key(cache_key)
        with self._lock:
            previous_index = self._index
            record = previous_index.get(key)
            if record is None:
                self._cleanup_generations(key)
                return
            self._index = dict(previous_index)
            self._index.pop(key, None)
            try:
                self._save_index()
            except Exception:
                self._index = previous_index
                return
            self._cleanup_generations(key)
            self._remove_tree(self._entry_dir(key, record))

    def list_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "key": value.get("cache_key", ""),
                    "model": value.get("model", "unknown"),
                    "tokens": int(value.get("token_count", 0) or 0),
                    "layers": int(value.get("layer_count", 0) or 0),
                    "size_mb": int(value.get("size_bytes", 0) or 0) / (1024**2),
                }
                for value in self._index.values()
            ]

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "entries": len(self._index),
                "total_size_gb": self._total_size() / (1024**3),
                "max_size_gb": self.max_size_bytes / (1024**3),
            }
