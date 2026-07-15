"""Prompt prefix caching: reuse KV cache for shared system prompt prefixes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class CachedPrefix:
    """A cached KV state for a token prefix."""

    token_hash: str
    token_count: int
    kv_cache: list[Any]  # List of per-layer cache objects (snapshots)
    hit_count: int = 0


class PrefixCache:
    """Cache KV states for common prompt prefixes.

    Particularly effective for coding agents that send the same system prompt
    (caveman ultra + directives) on every turn — saves ~1s prefill per request.
    """

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max_entries
        self._cache: dict[str, CachedPrefix] = {}

    @staticmethod
    def _hash_tokens(tokens: list[int]) -> str:
        raw = ",".join(str(t) for t in tokens)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def lookup(self, tokens: list[int]) -> CachedPrefix | None:
        """Find the longest cached prefix matching the start of tokens.

        Returns the cached entry if found, or None.
        """
        # Try progressively shorter prefixes
        for length in range(len(tokens), 0, -1):
            prefix = tokens[:length]
            h = self._hash_tokens(prefix)
            if h in self._cache:
                entry = self._cache[h]
                if entry.token_count == length:
                    entry.hit_count += 1
                    return entry
        return None

    def store(self, tokens: list[int], kv_cache: list[Any]) -> None:
        """Store a KV cache snapshot for a token prefix."""
        h = self._hash_tokens(tokens)
        if h in self._cache:
            return  # Already cached

        # Evict LRU if full
        if len(self._cache) >= self.max_entries:
            lru_key = min(self._cache, key=lambda k: self._cache[k].hit_count)
            del self._cache[lru_key]

        self._cache[h] = CachedPrefix(
            token_hash=h,
            token_count=len(tokens),
            kv_cache=kv_cache,
        )

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_hits": sum(e.hit_count for e in self._cache.values()),
        }

    def clear(self) -> None:
        self._cache.clear()
