"""Content-addressable store for chunk K_base + V.

Each chunk maps to a `.safetensors` file containing:
    arrays:
      layer{L}_k_base: (n_kv_heads, chunk_len, d_head) float16 (pre-RoPE)
      layer{L}_v:      (n_kv_heads, chunk_len, d_head) float16
    metadata:
      mio_kv_splice_version: "1"
      chunk_hash: sha256 of tokens
      chunk_len: int
      model_id: str
      layers: comma-separated layer indices stored
      n_kv_heads: int
      d_head: int
      created_at: int
      hit_count: int

Lookup is by chunk_hash (sha256 of token sequence). No semantic clustering
here — that's Path A's territory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


KV_SPLICE_VERSION = 1
DEFAULT_DIR = Path.home() / ".mio" / "kv-splice"


def chunk_hash(tokens: list[int]) -> str:
    """Deterministic sha256 of a token sequence, truncated to 16 hex chars."""
    h = hashlib.sha256()
    h.update(b"mio-kv-splice-chunk-v1:")
    for t in tokens:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()[:16]


@dataclass
class ChunkEntry:
    chunk_id: str
    tokens: list[int]
    model_id: str
    layers: list[int]
    n_kv_heads: int
    d_head: int
    chunk_len: int
    path: str
    created_at: int
    hit_count: int = 0
    # Canonical text representation for text-level detection. Stored
    # because BPE tokenization is context-dependent — the "same" chunk
    # can tokenize differently depending on preceding text.
    chunk_text: str = ""


class ChunkStore:
    """On-disk store of chunk K_base + V keyed by token-sequence hash.

    Layout:
        ~/.mio/kv-splice/
            index.json              — ChunkEntry metadata list (no arrays)
            chunks/{chunk_id}.safetensors
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        *,
        max_entries: int = 500,
    ):
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "chunks").mkdir(exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self.max_entries = int(max_entries)
        self._entries: dict[str, ChunkEntry] = self._load_index()

    def _load_index(self) -> dict[str, ChunkEntry]:
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text())
            return {r["chunk_id"]: ChunkEntry(**r) for r in raw}
        except Exception:
            return {}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(
            [e.__dict__ for e in self._entries.values()], indent=2,
        ))

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[ChunkEntry]:
        return list(self._entries.values())

    def get(self, chunk_id: str) -> Optional[ChunkEntry]:
        return self._entries.get(chunk_id)

    def get_by_tokens(self, tokens: list[int]) -> Optional[ChunkEntry]:
        return self.get(chunk_hash(tokens))

    # ---- save / load ----

    def save(
        self,
        *,
        tokens: list[int],
        model_id: str,
        layers: list[int],
        n_kv_heads: int,
        d_head: int,
        kv_per_layer: dict[int, dict[str, mx.array]],
        chunk_text: str = "",
    ) -> ChunkEntry:
        cid = chunk_hash(tokens)
        chunk_path = self.base_dir / "chunks" / f"{cid}.safetensors"

        arrays: dict[str, mx.array] = {}
        for li, kv in kv_per_layer.items():
            arrays[f"layer{li}_k_base"] = kv["k_base"]
            arrays[f"layer{li}_v"] = kv["v"]

        chunk_len = int(next(iter(kv_per_layer.values()))["k_base"].shape[1])
        metadata = {
            "mio_kv_splice_version": str(KV_SPLICE_VERSION),
            "chunk_id": cid,
            "model_id": model_id,
            "layers": ",".join(str(l) for l in layers),
            "n_kv_heads": str(int(n_kv_heads)),
            "d_head": str(int(d_head)),
            "chunk_len": str(chunk_len),
            "created_at": str(int(time.time())),
        }

        # Atomic write via tmp path.
        tmp = chunk_path.with_name(f".{chunk_path.stem}.tmp")
        mx.save_safetensors(str(tmp), arrays, metadata)
        # mx.save_safetensors auto-appends .safetensors; account for it.
        written = tmp.with_suffix(tmp.suffix + ".safetensors")
        os.replace(written, chunk_path)

        entry = ChunkEntry(
            chunk_id=cid, tokens=list(tokens), model_id=model_id,
            layers=list(layers), n_kv_heads=int(n_kv_heads),
            d_head=int(d_head), chunk_len=chunk_len, path=str(chunk_path),
            created_at=int(time.time()), hit_count=0,
            chunk_text=chunk_text,
        )
        self._entries[cid] = entry
        self._evict_if_needed()
        self._save_index()
        return entry

    def load_kv(
        self, chunk_id: str,
    ) -> Optional[dict[int, dict[str, mx.array]]]:
        """Load K_base + V arrays for a chunk. None on miss/corruption."""
        entry = self.get(chunk_id)
        if entry is None:
            return None
        try:
            arrs, meta = mx.load(entry.path, return_metadata=True)
        except Exception:
            return None
        if not isinstance(meta, dict):
            return None
        if meta.get("mio_kv_splice_version") != str(KV_SPLICE_VERSION):
            return None
        result: dict[int, dict[str, mx.array]] = {}
        for li in entry.layers:
            k_key = f"layer{li}_k_base"
            v_key = f"layer{li}_v"
            if k_key not in arrs or v_key not in arrs:
                return None
            result[li] = {"k_base": arrs[k_key], "v": arrs[v_key]}
        return result

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        # Evict by oldest-first with low hit count.
        by_score = sorted(
            self._entries.values(),
            key=lambda e: (e.hit_count, e.created_at),
        )
        while len(self._entries) > self.max_entries:
            victim = by_score.pop(0)
            try:
                Path(victim.path).unlink()
            except FileNotFoundError:
                pass
            del self._entries[victim.chunk_id]

    def bump_hit(self, chunk_id: str) -> None:
        entry = self._entries.get(chunk_id)
        if entry is not None:
            entry.hit_count += 1
            self._save_index()
