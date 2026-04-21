"""Prototype store: embeddings + pointers to frozen_kv snapshots.

Disk layout:
    ~/.mio/prefill-autolearn/
        index.json        — metadata for all prototypes (list of dicts)
        emb/{id}.npy      — per-prototype d_model-dim unit-norm embedding
        (KV lives in existing ~/.mio/frozen-kv/ via mio.frozen_kv)

Each prototype record has:
    id: sha256 hex digest of prompt_tokens (stable identifier)
    tokens: list[int] — the stored prompt tokens
    emb_path: path to the embedding .npy
    frozen_kv_path: path to the safetensors snapshot (from mio.frozen_kv)
    model_id: which model the snapshot was captured for
    pq_bits, tq_bits, ctx_window: config-key fields
    created_at: epoch timestamp
    hit_count: how many times this prototype has been reused

Index is loaded at startup, updated on add/remove, flushed on shutdown.
All operations on the index are O(N) where N is number of prototypes;
we cap at 100 so this is always fast.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


DEFAULT_DIR = Path.home() / ".mio" / "prefill-autolearn"


@dataclass
class Prototype:
    id: str
    tokens: list[int]
    emb_path: str
    frozen_kv_path: str
    model_id: str
    pq_bits: int
    tq_bits: int
    ctx_window: int
    created_at: int
    hit_count: int = 0


def _prompt_id(tokens: list[int]) -> str:
    h = hashlib.sha256()
    h.update(b"prefill-autolearn:")
    for t in tokens[:512]:  # first 512 tokens is enough for identity
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()[:16]


class PrototypeStore:
    """Bounded LRU-like store of (embedding, frozen_kv) prototypes."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        *,
        max_entries: int = 100,
    ):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "emb").mkdir(exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self.max_entries = int(max_entries)
        self._prototypes: list[Prototype] = self._load_index()

    # ---- persistence ----

    def _load_index(self) -> list[Prototype]:
        if not self.index_path.exists():
            return []
        try:
            raw = json.loads(self.index_path.read_text())
            return [Prototype(**r) for r in raw]
        except Exception:
            return []

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps([asdict(p) for p in self._prototypes], indent=2)
        )

    # ---- accessors ----

    def __len__(self) -> int:
        return len(self._prototypes)

    def all(self) -> list[Prototype]:
        return list(self._prototypes)

    def get_by_id(self, proto_id: str) -> Optional[Prototype]:
        for p in self._prototypes:
            if p.id == proto_id:
                return p
        return None

    def load_embedding(self, proto: Prototype) -> mx.array:
        arr = np.load(proto.emb_path)
        return mx.array(arr, dtype=mx.float32)

    def load_all_embeddings(self) -> mx.array:
        """Stack embeddings into (N, d) matrix for batch cosine search."""
        if not self._prototypes:
            return mx.zeros((0, 1), dtype=mx.float32)
        vecs = [np.load(p.emb_path) for p in self._prototypes]
        return mx.array(np.stack(vecs, axis=0), dtype=mx.float32)

    # ---- mutation ----

    def add(
        self,
        *,
        tokens: list[int],
        embedding: mx.array,
        frozen_kv_path: str,
        model_id: str,
        pq_bits: int,
        tq_bits: int,
        ctx_window: int,
    ) -> Prototype:
        proto_id = _prompt_id(tokens)
        # If already present, just bump hit count.
        existing = self.get_by_id(proto_id)
        if existing is not None:
            existing.hit_count += 1
            self._save_index()
            return existing

        # Save embedding.
        emb_path = self.base_dir / "emb" / f"{proto_id}.npy"
        np.save(emb_path, np.array(embedding, copy=True))

        proto = Prototype(
            id=proto_id, tokens=list(tokens), emb_path=str(emb_path),
            frozen_kv_path=frozen_kv_path, model_id=model_id,
            pq_bits=int(pq_bits), tq_bits=int(tq_bits),
            ctx_window=int(ctx_window), created_at=int(time.time()),
            hit_count=0,
        )
        self._prototypes.append(proto)
        self._evict_if_needed()
        self._save_index()
        return proto

    def _evict_if_needed(self) -> None:
        """LRU-ish eviction by hit_count desc, age asc as tiebreaker."""
        if len(self._prototypes) <= self.max_entries:
            return
        # Sort: keep highest-hit-count first; among ties, newest first.
        self._prototypes.sort(
            key=lambda p: (-p.hit_count, -p.created_at),
        )
        for victim in self._prototypes[self.max_entries:]:
            try:
                Path(victim.emb_path).unlink()
            except FileNotFoundError:
                pass
            # Note: we do NOT delete the frozen_kv file; mio.frozen_kv owns
            # that lifecycle. A separate prune_cache_dir() call handles it.
        self._prototypes = self._prototypes[:self.max_entries]

    def increment_hit(self, proto_id: str) -> None:
        p = self.get_by_id(proto_id)
        if p is not None:
            p.hit_count += 1
            self._save_index()

    # ---- search ----

    def nearest(
        self,
        query: mx.array,
        *,
        k: int = 5,
        min_similarity: float = 0.85,
        model_id: Optional[str] = None,
        pq_bits: Optional[int] = None,
        tq_bits: Optional[int] = None,
    ) -> list[tuple[Prototype, float]]:
        """Return up to k prototypes whose embedding cosine with query >= min_similarity.

        Filters by matching config fields when provided. Returns list of
        (prototype, similarity_score) sorted desc by similarity.
        """
        if not self._prototypes:
            return []
        # Filter by config
        candidates: list[tuple[int, Prototype]] = []
        for i, p in enumerate(self._prototypes):
            if model_id is not None and p.model_id != model_id:
                continue
            if pq_bits is not None and p.pq_bits != pq_bits:
                continue
            if tq_bits is not None and p.tq_bits != tq_bits:
                continue
            candidates.append((i, p))
        if not candidates:
            return []

        # Load embeddings for candidates only.
        matrix = mx.stack(
            [mx.array(np.load(p.emb_path), dtype=mx.float32) for _, p in candidates],
            axis=0,
        )
        query_f32 = query.astype(mx.float32)
        scores = mx.matmul(matrix, query_f32)
        mx.eval(scores)
        s = scores.tolist()
        ranked = sorted(
            zip(candidates, s), key=lambda x: -x[1]
        )
        out: list[tuple[Prototype, float]] = []
        for ((_, proto), score) in ranked[:k]:
            if score < min_similarity:
                break
            out.append((proto, score))
        return out
