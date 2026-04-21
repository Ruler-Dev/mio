"""Frozen KV-cache snapshots for repeated system-prompt prefixes.

Agent workflows (Cline, Kilo, Roo, Caveman) wrap every request in a fixed
10-40 K token system prompt. Default mio config has PolarQuant on, which
disables the existing prefix cache, so that prefix is re-prefilled from
scratch on every request — 5-20 s of wasted compute per turn.

This module caches the full KV-cache state for the prefix to disk, keyed by
(prompt_hash, model_id, pq_bits, tq_bits, ctx_window, version). On load, a
fresh cache list is built from the live model and populated from the disk
snapshot's arrays. This keeps the frozen snapshot compatible with every
cache type mio uses (plain KVCache, QuantizedKVCache, RecurrentRollbackCache,
ArraysCache, PolarQuant/TurboQuant variants — anything exposing .state).

Safety properties:
- **Deterministic fingerprint**: identical inputs → identical hash across runs.
- **Config-keyed**: changing model, pq_bits, tq_bits, or ctx_window invalidates.
- **Type-checked**: class list on disk must match live cache types layer-for-layer.
- **Corruption-safe**: any load failure returns None, never raises.
- **Version-gated**: bump FROZEN_KV_VERSION to invalidate the on-disk format.

Public surface:
    fingerprint(prompt_tokens, ...) -> str
    snapshot_path(fp) -> Path
    freeze(caches, prompt_tokens, ...) -> Path
    try_load(template_caches, prompt_tokens, ...) -> (caches, meta) | None
    prune_cache_dir(base_dir, max_entries, max_bytes) -> int
    common_prefix_fingerprints(prompt_tokens, candidate_lens, ...) -> list[str]
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


FROZEN_KV_VERSION = 1
DEFAULT_DIR = Path.home() / ".mio" / "frozen-kv"


def fingerprint(
    prompt_tokens: Iterable[int],
    *,
    prefix_len: int,
    model_id: str,
    pq_bits: int,
    tq_bits: int,
    ctx_window: int,
) -> str:
    """Deterministic SHA-256 hex digest over (config, first prefix_len tokens).

    Raises ValueError if prompt has fewer than `prefix_len` tokens or if
    prefix_len <= 0. Separators (`|`) are included so that concatenation
    boundaries can't be confused across fields.
    """
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    if model_id == "":
        raise ValueError("model_id must be non-empty")

    tokens = list(prompt_tokens)
    if len(tokens) < prefix_len:
        raise ValueError(
            f"prompt has {len(tokens)} tokens, need at least {prefix_len}"
        )

    h = hashlib.sha256()
    header = (
        f"mio-frozen-kv:{FROZEN_KV_VERSION}|{model_id}|"
        f"pq={int(pq_bits)}|tq={int(tq_bits)}|ctx={int(ctx_window)}|"
        f"len={int(prefix_len)}"
    )
    h.update(header.encode("utf-8"))
    h.update(b"|tokens:")
    for tok in tokens[:prefix_len]:
        h.update(int(tok).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def snapshot_path(fp: str, base_dir: Optional[Path] = None) -> Path:
    """Return path for a fingerprint; does not create the directory."""
    base = Path(base_dir) if base_dir is not None else DEFAULT_DIR
    return base / f"{fp}.safetensors"


def freeze(
    caches: list[Any],
    *,
    prompt_tokens: list[int],
    prefix_len: int,
    model_id: str,
    pq_bits: int,
    tq_bits: int,
    ctx_window: int,
    base_dir: Optional[Path] = None,
) -> Path:
    """Serialize `caches` to a safetensors snapshot keyed by the prompt prefix.

    Overwrites an existing snapshot with the same fingerprint atomically
    (write to `.tmp` then rename). The caller is responsible for ensuring
    the cache offset is >= prefix_len; this function does not trim.
    """
    fp = fingerprint(
        prompt_tokens,
        prefix_len=prefix_len,
        model_id=model_id,
        pq_bits=pq_bits,
        tq_bits=tq_bits,
        ctx_window=ctx_window,
    )
    target = snapshot_path(fp, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Two sides to the snapshot:
    #   arrays side: only real mx.arrays (from cache.state trees).
    #   metadata side: header fields + per-layer meta_state + class list,
    #                  all string-typed (safetensors requires Dict[str,str]).
    state_list = [c.state for c in caches]
    flat_pairs = tree_flatten(state_list)
    flat_arrays: dict[str, mx.array] = {}
    for name, value in flat_pairs:
        if not isinstance(value, mx.array):
            raise ValueError(
                f"non-array encountered in cache state at {name}: {type(value)}"
            )
        flat_arrays[name] = value

    meta_states = [_serialize_meta_state(c) for c in caches]

    metadata = {
        "mio_frozen_kv_version": str(FROZEN_KV_VERSION),
        "model_id": model_id,
        "pq_bits": str(int(pq_bits)),
        "tq_bits": str(int(tq_bits)),
        "ctx_window": str(int(ctx_window)),
        "prefix_len": str(int(prefix_len)),
        "token_hash": fp,
        "created_at": str(int(time.time())),
        "cache_classes": json.dumps([type(c).__name__ for c in caches]),
        "cache_meta_states": json.dumps(meta_states),
    }

    # mx.save_safetensors silently appends ".safetensors" if the path doesn't
    # end in it, so we write to a sibling name and move to the real target.
    tmp_stem = target.with_name(f".{target.stem}.tmp")
    mx.save_safetensors(str(tmp_stem), flat_arrays, metadata)
    written = tmp_stem.with_suffix(tmp_stem.suffix + ".safetensors")
    os.replace(written, target)
    return target


def try_load(
    template_caches: list[Any],
    *,
    prompt_tokens: list[int],
    prefix_len: int,
    model_id: str,
    pq_bits: int,
    tq_bits: int,
    ctx_window: int,
    base_dir: Optional[Path] = None,
) -> Optional[tuple[list[Any], dict[str, str]]]:
    """Load a snapshot and install its state into `template_caches`.

    `template_caches` should be freshly built via `make_target_cache(model)`
    so the cache types and structure match the live model. On any mismatch,
    corruption, or I/O error, returns None without raising.

    On success, returns `(template_caches, metadata)`. The caches are
    mutated in-place; the caller should treat them as owned.
    """
    try:
        fp = fingerprint(
            prompt_tokens,
            prefix_len=prefix_len,
            model_id=model_id,
            pq_bits=pq_bits,
            tq_bits=tq_bits,
            ctx_window=ctx_window,
        )
    except (ValueError, TypeError):
        return None

    target = snapshot_path(fp, base_dir)
    if not target.exists():
        return None

    try:
        arrays, metadata = mx.load(str(target), return_metadata=True)
    except Exception:
        return None

    if not isinstance(metadata, dict):
        return None
    if metadata.get("mio_frozen_kv_version") != str(FROZEN_KV_VERSION):
        return None
    if metadata.get("token_hash") != fp:
        return None
    if metadata.get("prefix_len") != str(int(prefix_len)):
        return None
    if metadata.get("model_id") != model_id:
        return None
    if metadata.get("pq_bits") != str(int(pq_bits)):
        return None
    if metadata.get("tq_bits") != str(int(tq_bits)):
        return None
    if metadata.get("ctx_window") != str(int(ctx_window)):
        return None

    try:
        saved_classes = json.loads(metadata.get("cache_classes", "[]"))
    except json.JSONDecodeError:
        return None
    live_classes = [type(c).__name__ for c in template_caches]
    if saved_classes != live_classes:
        return None

    try:
        state_list = tree_unflatten(list(arrays.items()))
        if not isinstance(state_list, list):
            return None
        if len(state_list) != len(template_caches):
            return None
        meta_states = json.loads(metadata.get("cache_meta_states", "[]"))
        if not isinstance(meta_states, list):
            return None
        if len(meta_states) != len(template_caches):
            return None
    except Exception:
        return None

    try:
        for cache, state, serialized_meta in zip(
            template_caches, state_list, meta_states, strict=True
        ):
            _install_state(cache, state, serialized_meta)
    except Exception:
        return None

    return template_caches, dict(metadata)


def _serialize_meta_state(cache: Any) -> Any:
    """Return a JSON-safe representation of cache.meta_state.

    Covers the shapes mlx_lm / mio caches actually produce:
    - "" (default from _BaseCache)
    - tuple/list of stringified ints (QuantizedKVCache → ("512", "64", "8"))
    - tuple/list of ints (ChunkedKVCache / RotatingKVCache variants)
    - mx.array (rare — we store a list of ints).

    Returns: None (empty), list[int], or {"kind": "raw", "value": str} for
    opaque string blobs that round-trip via the setter.
    """
    try:
        meta = cache.meta_state
    except Exception:
        return None
    if meta is None or meta == "" or meta == ():
        return None
    if isinstance(meta, mx.array):
        return [int(x) for x in meta.tolist()]
    if isinstance(meta, (tuple, list)):
        out: list[int] = []
        for part in meta:
            if isinstance(part, (int, float)):
                out.append(int(part))
            else:
                out.append(int(str(part)))
        return out
    if isinstance(meta, str):
        return {"kind": "raw", "value": meta}
    if isinstance(meta, (int, float)):
        return [int(meta)]
    raise TypeError(f"unsupported meta_state type: {type(meta)}")


def _install_state(cache: Any, state: Any, serialized_meta: Any) -> None:
    """Apply `(state, meta_state)` back to a freshly-built cache.

    Sets `cache.state` first (for classes like KVCache that derive offset
    from keys shape), then `cache.meta_state` if the cache carries one
    (e.g., QuantizedKVCache stores offset in meta_state).
    """
    cache.state = state
    if serialized_meta is None:
        return
    if isinstance(serialized_meta, dict):
        if serialized_meta.get("kind") == "raw":
            cache.meta_state = serialized_meta.get("value", "")
        return
    if isinstance(serialized_meta, list):
        if not serialized_meta:
            return
        cache.meta_state = tuple(str(int(x)) for x in serialized_meta)
        return
    raise TypeError(
        f"unsupported serialized meta_state shape: {type(serialized_meta)}"
    )


def common_prefix_fingerprints(
    prompt_tokens: list[int],
    *,
    candidate_lens: list[int],
    model_id: str,
    pq_bits: int,
    tq_bits: int,
    ctx_window: int,
) -> list[tuple[int, str]]:
    """Compute (length, fingerprint) for each candidate prefix length.

    Caller iterates longest-first, looks each fingerprint up on disk, and
    picks the first match. Candidate lengths beyond the prompt length are
    silently skipped.
    """
    seen: set[int] = set()
    results: list[tuple[int, str]] = []
    for length in candidate_lens:
        length = int(length)
        if length <= 0 or length > len(prompt_tokens) or length in seen:
            continue
        seen.add(length)
        fp = fingerprint(
            prompt_tokens,
            prefix_len=length,
            model_id=model_id,
            pq_bits=pq_bits,
            tq_bits=tq_bits,
            ctx_window=ctx_window,
        )
        results.append((length, fp))
    return results


def prune_cache_dir(
    base_dir: Optional[Path] = None,
    *,
    max_entries: int = 16,
    max_bytes: int = 20 * 1024**3,
) -> int:
    """Evict oldest snapshots until both limits hold. Returns #deleted.

    Call manually or from a tier-load path; never invoked automatically
    during generation. Walks the directory once, sorts by mtime ascending,
    and unlinks until under the caps.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_DIR
    if not base.exists():
        return 0
    files = [p for p in base.iterdir() if p.suffix == ".safetensors"]
    files.sort(key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    n_deleted = 0
    while files and (len(files) > max_entries or total > max_bytes):
        victim = files.pop(0)
        try:
            total -= victim.stat().st_size
            victim.unlink()
            n_deleted += 1
        except FileNotFoundError:
            pass
    return n_deleted
