"""Chunk detection.

Two strategies:
  - **detect_chunks_text** (production): scan the prompt text for stored
    chunks' text, map char offsets to token positions via the tokenizer's
    offset_mapping. Works around BPE context-sensitivity.
  - **detect_chunks** (token-level, for tests): exact sublist match on
    token sequences. Works only when the same text tokenizes identically
    in ingest and detection contexts — rare in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mio.kv_splice.store import ChunkStore


@dataclass
class SpliceSite:
    chunk_id: str
    start: int  # token index (inclusive)
    end: int    # token index (exclusive)
    chunk_len: int  # == end - start


def detect_chunks_text(
    prompt_text: str,
    tokenizer: Any,
    store: ChunkStore,
    *,
    model_id: Optional[str] = None,
    min_chunk_len: int = 32,
) -> list[SpliceSite]:
    """Find stored chunks in `prompt_text`, map to token positions.

    Args:
        prompt_text: the full text passed to the tokenizer (including
            chat template output if applicable).
        tokenizer: HuggingFace-compatible tokenizer exposing
            return_offsets_mapping=True.
        store: ChunkStore; entries need their `chunk_text` populated.

    Returns SpliceSites ordered by start, non-overlapping.
    """
    candidates = [
        e for e in store.all()
        if (model_id is None or e.model_id == model_id)
        and e.chunk_text
        and e.chunk_len >= min_chunk_len
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda e: -len(e.chunk_text))

    # Some mlx_lm tokenizers are wrapped (TokenizerWrapper not callable);
    # fall through to the underlying HF tokenizer for the offset-mapping
    # call.
    callee = tokenizer if callable(tokenizer) else getattr(tokenizer, "_tokenizer", None)
    if callee is None:
        return []
    try:
        enc = callee(
            prompt_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except Exception:
        return []
    offsets = enc.get("offset_mapping") if isinstance(enc, dict) else getattr(enc, "offset_mapping", None)
    if offsets is None:
        return []
    # Build char→token maps allowing boundary shrinkage.
    # tok_at_cs_gte[c] = smallest i such that offsets[i].cs >= c
    # tok_at_ce_lte[c] = largest i such that offsets[i].ce <= c
    # This lets us shrink a text-match span to the largest enclosed token
    # span even when the chunk's edge chars merge with neighboring chars.

    sites: list[SpliceSite] = []
    taken: list[tuple[int, int]] = []

    for entry in candidates:
        text = entry.chunk_text
        search_from = 0
        while True:
            idx = prompt_text.find(text, search_from)
            if idx < 0:
                break
            end_idx = idx + len(text)
            if any(not (end_idx <= s or idx >= e) for (s, e) in taken):
                search_from = idx + 1
                continue
            # Find first token whose cs >= idx.
            tok_start = None
            for i, (cs, ce) in enumerate(offsets):
                if cs == ce:
                    continue
                if cs >= idx:
                    tok_start = i
                    break
            # Find last token whose ce <= end_idx.
            tok_end = None
            for i in range(len(offsets) - 1, -1, -1):
                cs, ce = offsets[i]
                if cs == ce:
                    continue
                if ce <= end_idx:
                    tok_end = i + 1
                    break
            if tok_start is None or tok_end is None or tok_end <= tok_start:
                search_from = idx + 1
                continue
            n_tokens = tok_end - tok_start
            # Allow shrinkage up to 4 tokens (2 on each edge) for BPE merges.
            if abs(n_tokens - entry.chunk_len) > 4:
                search_from = idx + 1
                continue
            sites.append(SpliceSite(
                chunk_id=entry.chunk_id,
                start=tok_start, end=tok_end, chunk_len=n_tokens,
            ))
            taken.append((idx, end_idx))
            search_from = end_idx

    sites.sort(key=lambda s: s.start)
    return sites


# ---- Token-level detector (for simple cases / tests) ----

def detect_chunks(
    prompt_tokens: list[int],
    store: ChunkStore,
    *,
    model_id: Optional[str] = None,
    min_chunk_len: int = 32,
) -> list[SpliceSite]:
    """Exact-sublist match on tokens. Limited by BPE context-sensitivity."""
    candidates = [
        e for e in store.all()
        if model_id is None or e.model_id == model_id
    ]
    candidates = [e for e in candidates if e.chunk_len >= min_chunk_len]
    candidates.sort(key=lambda e: -e.chunk_len)
    by_first_token: dict[int, list] = {}
    for e in candidates:
        if e.tokens:
            by_first_token.setdefault(e.tokens[0], []).append(e)
    sites: list[SpliceSite] = []
    i = 0
    N = len(prompt_tokens)
    while i < N:
        matched = None
        for e in by_first_token.get(prompt_tokens[i], []):
            L = e.chunk_len
            if i + L > N:
                continue
            if prompt_tokens[i:i + L] == e.tokens:
                matched = e
                break
        if matched is not None:
            sites.append(SpliceSite(
                chunk_id=matched.chunk_id, start=i,
                end=i + matched.chunk_len, chunk_len=matched.chunk_len,
            ))
            i += matched.chunk_len
        else:
            i += 1
    return sites
