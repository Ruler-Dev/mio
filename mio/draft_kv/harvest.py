"""Harvest (intermediate_hidden, target_KV) pairs for projector training.

Usage pattern: during a normal target-model prefill, capture the hidden
state at a chosen "early exit" layer and the K/V produced at each of the
downstream layers. Store them to disk. A training script later pairs them
per-layer and fits a projector to approximate
    projector(hidden) ≈ (K_l, V_l) for l in late_layers.

This file provides ONLY the recorder; training is external and out of
scope. The recorder is a dumb buffer + safetensors writer with shape
checks and a deterministic filename.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_flatten


DEFAULT_HARVEST_DIR = Path.home() / ".mio" / "draft-kv-harvest"


class HarvestRecorder:
    """Collect per-call (hidden, K_list, V_list) and flush to disk on demand.

    One recorder per prompt sample. Call `record(...)` once per prefill.
    Call `flush()` to write all collected samples to a single safetensors
    shard keyed by recorder id. Thread-unsafe (intended for single-process
    harvesting).
    """

    def __init__(
        self,
        *,
        early_layer: int,
        target_layers: list[int],
        out_dir: Optional[Path] = None,
        shard_name: Optional[str] = None,
    ) -> None:
        if early_layer < 0:
            raise ValueError("early_layer must be >= 0")
        if not target_layers:
            raise ValueError("target_layers must be non-empty")
        if min(target_layers) <= early_layer:
            raise ValueError(
                f"target_layers must all be > early_layer={early_layer}; "
                f"got min target={min(target_layers)}"
            )
        self.early_layer = int(early_layer)
        self.target_layers = list(target_layers)
        self.out_dir = Path(out_dir) if out_dir is not None else DEFAULT_HARVEST_DIR
        self.shard_name = shard_name or self._fresh_shard_name()
        self._hiddens: list[mx.array] = []
        self._kv_pairs: list[list[tuple[mx.array, mx.array]]] = []
        self._sample_ids: list[str] = []

    @staticmethod
    def _fresh_shard_name() -> str:
        h = hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:12]
        return f"shard-{h}"

    def record(
        self,
        *,
        sample_id: str,
        hidden_at_early: mx.array,
        kvs_per_layer: list[tuple[mx.array, mx.array]],
    ) -> None:
        """Record one prefill's intermediate hidden + downstream KV list."""
        if hidden_at_early.ndim != 3:
            raise ValueError(
                f"hidden_at_early must be (B, L, D), got {hidden_at_early.shape}"
            )
        if len(kvs_per_layer) != len(self.target_layers):
            raise ValueError(
                f"expected {len(self.target_layers)} KV pairs, got {len(kvs_per_layer)}"
            )
        B, L, _ = hidden_at_early.shape
        for i, (k, v) in enumerate(kvs_per_layer):
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError(
                    f"KV[{i}] must be (B, H, L, D), got K={k.shape} V={v.shape}"
                )
            if k.shape[0] != B or v.shape[0] != B:
                raise ValueError(f"KV[{i}] batch dim mismatch with hidden")
            if k.shape[2] != L or v.shape[2] != L:
                raise ValueError(f"KV[{i}] seq dim mismatch with hidden")
        self._hiddens.append(hidden_at_early)
        self._kv_pairs.append(kvs_per_layer)
        self._sample_ids.append(str(sample_id))

    def sample_count(self) -> int:
        return len(self._hiddens)

    def flush(self) -> Optional[Path]:
        """Write collected samples to a single safetensors shard.

        Returns the path written, or None if no samples were recorded.
        Clears the in-memory buffer on success.
        """
        if not self._hiddens:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.shard_name}.safetensors"

        arrays: dict[str, mx.array] = {}
        for i, (hid, kvs, sid) in enumerate(
            zip(self._hiddens, self._kv_pairs, self._sample_ids, strict=True)
        ):
            arrays[f"s{i}/hidden"] = hid
            for li, (k, v) in enumerate(kvs):
                arrays[f"s{i}/layer{self.target_layers[li]}/K"] = k
                arrays[f"s{i}/layer{self.target_layers[li]}/V"] = v

        metadata = {
            "early_layer": str(self.early_layer),
            "target_layers": ",".join(str(x) for x in self.target_layers),
            "sample_count": str(len(self._hiddens)),
            "sample_ids": ",".join(self._sample_ids),
            "shard_name": self.shard_name,
            "created_at": str(int(time.time())),
        }

        tmp = path.with_name(f".{path.stem}.tmp")
        mx.save_safetensors(str(tmp), arrays, metadata)
        produced = tmp.with_suffix(tmp.suffix + ".safetensors")
        import os as _os
        _os.replace(produced, path)

        self._hiddens.clear()
        self._kv_pairs.clear()
        self._sample_ids.clear()
        return path


def load_shard(
    path: Path,
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Read a harvest shard back. Returns (flat arrays dict, metadata dict)."""
    arrays, meta = mx.load(str(path), return_metadata=True)
    if not isinstance(meta, dict):
        raise ValueError(f"harvest shard at {path} has no metadata")
    return dict(arrays), dict(meta)
