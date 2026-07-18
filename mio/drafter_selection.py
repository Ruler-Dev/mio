"""Metadata-driven drafter selection for Mio.

The selector is deliberately read-only: inspecting a Hugging Face reference
consults the local cache only and never downloads a checkpoint.  Actual model
loading remains owned by :class:`mio.engine.MioEngine`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any

from mio.config import normalise_drafter_backend


class DrafterKind(str, Enum):
    DSPARK = "dspark"
    DFLASH = "dflash"
    HYBRID_DFLASH_MARKOV = "hybrid_dflash_markov"
    NOT_INSPECTED = "not_inspected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DrafterDescriptor:
    ref: str
    kind: DrafterKind
    config: dict[str, Any]
    config_path: str | None = None


@dataclass(frozen=True)
class DrafterPlan:
    requested: str
    detected: DrafterKind
    primary_backend: str
    primary_ref: str | None
    fallback_ref: str | None
    strict: bool
    reason: str


def classify_drafter_config(config: dict[str, Any]) -> DrafterKind:
    """Classify native DSpark, pure DFlash, and DFlash+Markov metadata."""

    if not isinstance(config, dict) or not config:
        return DrafterKind.UNKNOWN
    architectures = " ".join(str(value).lower() for value in config.get("architectures", []))
    dflash_config = config.get("dflash_config")
    has_dflash = isinstance(dflash_config, dict) or "dflash" in architectures
    has_markov = bool(
        config.get("markov_rank")
        or config.get("markov_head_type")
        or (isinstance(dflash_config, dict) and dflash_config.get("markov_rank"))
    )
    has_dspark = bool(
        "dspark" in architectures
        or isinstance(config.get("dspark_config"), dict)
        or (has_markov and config.get("block_size") and config.get("target_layer_ids"))
    )
    if has_dflash and has_markov:
        return DrafterKind.HYBRID_DFLASH_MARKOV
    if has_dspark:
        return DrafterKind.DSPARK
    if has_dflash:
        return DrafterKind.DFLASH
    return DrafterKind.UNKNOWN


def _cached_config_path(ref: str) -> Path | None:
    direct = Path(ref).expanduser()
    if (direct / "config.json").is_file():
        return direct / "config.json"

    from mio.models.registry import spd_dir

    project_local = spd_dir() / ref
    if (project_local / "config.json").is_file():
        return project_local / "config.json"

    if "/" not in ref:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(ref, "config.json")
    except Exception:
        return None
    if isinstance(cached, str):
        path = Path(cached)
        return path if path.is_file() else None
    return None


def inspect_drafter(ref: str | os.PathLike[str]) -> DrafterDescriptor:
    """Read cached/local metadata and classify a drafter without network I/O."""

    rendered = str(ref)
    path = _cached_config_path(rendered)
    config: dict[str, Any] = {}
    if path is not None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            config = {}
    kind = classify_drafter_config(config)
    if kind is DrafterKind.UNKNOWN:
        lowered = rendered.lower()
        if "dspark" in lowered:
            kind = DrafterKind.DSPARK
        elif "dflash" in lowered:
            kind = DrafterKind.DFLASH
    return DrafterDescriptor(
        ref=rendered,
        kind=kind,
        config=config,
        config_path=str(path) if path is not None else None,
    )


def _same_ref(left: str, right: str) -> bool:
    if left == right:
        return True
    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    try:
        if left_path.exists() and right_path.exists():
            return left_path.resolve() == right_path.resolve()
    except OSError:
        pass
    return False


def _compatible_dflash(target_config: dict[str, Any], descriptor: DrafterDescriptor) -> bool:
    if descriptor.kind is not DrafterKind.DFLASH:
        return False
    if not target_config or not descriptor.config:
        return True
    try:
        from mio.dflash.runtime import validate_draft_target_compatibility

        validate_draft_target_compatibility(target_config, descriptor.config)
    except (TypeError, ValueError):
        return False
    return True


def _registry_fallbacks(target_ref: str) -> list[str]:
    from mio.models.registry import (
        KNOWN_MODELS,
        _model_path_is_complete,
        spd_dir,
    )

    candidates: list[str] = []
    target_path = Path(target_ref).expanduser()
    for entry in KNOWN_MODELS.values():
        references = {entry.target_repo, entry.target_local, entry.resolve_target()}
        matched = target_ref in references
        if not matched and target_path.exists():
            for reference in references:
                path = Path(reference).expanduser()
                try:
                    if path.exists() and path.resolve() == target_path.resolve():
                        matched = True
                        break
                except OSError:
                    continue
        if matched:
            # Registry metadata describes remote repositories too, but an
            # absent ``draft_fallback_model`` is an explicit no-remote-
            # fallback boundary (notably after ``mio pull --no-fallback``).
            # Automatic discovery may therefore reuse only a complete local
            # DFlash checkpoint. Explicit config below may still name a remote
            # repository intentionally.
            local_dflash = spd_dir() / entry.draft_local
            if _model_path_is_complete(local_dflash):
                candidates.append(str(local_dflash))
    return candidates


def _local_dflash_fallbacks(target_ref: str) -> list[str]:
    from mio.models.registry import _model_path_is_complete, spd_dir

    root = spd_dir()
    if not root.is_dir():
        return []
    target_name = Path(target_ref).name.lower()
    paths = [path for path in root.iterdir() if path.is_dir() and _model_path_is_complete(path)]
    paths.sort(
        key=lambda path: SequenceMatcher(
            None,
            target_name,
            path.name.lower().replace("dflash", ""),
        ).ratio(),
        reverse=True,
    )
    return [str(path) for path in paths]


def find_compatible_dflash(
    *,
    target_ref: str,
    target_config: dict[str, Any],
    requested_ref: str,
    explicit_ref: str | None = None,
) -> str | None:
    """Find a pure compatible DFlash fallback, never the DSpark/hybrid input."""

    if explicit_ref:
        descriptor = inspect_drafter(explicit_ref)
        if _same_ref(descriptor.ref, requested_ref):
            if descriptor.kind is not DrafterKind.DFLASH:
                raise ValueError("DFlash fallback must not reuse the DSpark/hybrid checkpoint")
        if descriptor.config and not _compatible_dflash(target_config, descriptor):
            raise ValueError(f"configured DFlash fallback is incompatible: {explicit_ref}")
        if descriptor.kind not in {DrafterKind.DFLASH, DrafterKind.UNKNOWN}:
            raise ValueError(f"configured fallback is not pure DFlash: {explicit_ref}")
        return descriptor.ref

    seen: set[str] = set()
    candidates = [
        *_registry_fallbacks(target_ref),
        *_local_dflash_fallbacks(target_ref),
    ]
    for candidate in candidates:
        if candidate in seen or _same_ref(candidate, requested_ref):
            continue
        seen.add(candidate)
        descriptor = inspect_drafter(candidate)
        if _compatible_dflash(target_config, descriptor):
            return descriptor.ref
    return None


def strict_drafter_mode(configured: bool) -> bool:
    raw = os.environ.get("MIO_DRAFTER_STRICT", "").strip().lower()
    return bool(configured or raw in {"1", "true", "yes", "on"})


def plan_drafter(tier: Any, target_config: dict[str, Any]) -> DrafterPlan:
    """Resolve the requested backend and its independently compatible fallback."""

    requested = normalise_drafter_backend(getattr(tier, "drafter_backend", "auto"))
    if requested == "target_ar":
        # This must remain ahead of every draft-model field access, metadata
        # inspection, cache scan, compatibility check, and strict-mode lookup.
        # It is the reproducible target-only control arm for matched benchmarks.
        return DrafterPlan(
            requested=requested,
            detected=DrafterKind.NOT_INSPECTED,
            primary_backend="target_ar",
            primary_ref=None,
            fallback_ref=None,
            strict=False,
            reason="explicit_target_ar",
        )

    primary_ref = str(tier.draft_model)
    descriptor = inspect_drafter(primary_ref)
    strict = strict_drafter_mode(bool(getattr(tier, "drafter_strict", False)))
    explicit_fallback = getattr(tier, "draft_fallback_model", None)

    if requested == "auto":
        primary_backend = (
            "dspark" if descriptor.kind in {DrafterKind.DSPARK, DrafterKind.HYBRID_DFLASH_MARKOV} else "dflash"
        )
        reason = f"auto_detected_{descriptor.kind.value}"
    else:
        primary_backend = requested
        reason = f"explicit_{requested}_detected_{descriptor.kind.value}"

    fallback_ref: str | None = None
    if primary_backend == "dspark":
        if descriptor.kind is DrafterKind.DFLASH:
            fallback_ref = primary_ref
        else:
            fallback_ref = find_compatible_dflash(
                target_ref=str(tier.target_model),
                target_config=target_config,
                requested_ref=primary_ref,
                explicit_ref=str(explicit_fallback) if explicit_fallback else None,
            )
    elif descriptor.kind in {DrafterKind.DSPARK, DrafterKind.HYBRID_DFLASH_MARKOV}:
        replacement = find_compatible_dflash(
            target_ref=str(tier.target_model),
            target_config=target_config,
            requested_ref=primary_ref,
            explicit_ref=str(explicit_fallback) if explicit_fallback else None,
        )
        if replacement is None:
            raise ValueError("explicit DFlash mode requires a compatible pure DFlash checkpoint")
        primary_ref = replacement
        reason += "_using_compatible_fallback"

    return DrafterPlan(
        requested=requested,
        detected=descriptor.kind,
        primary_backend=primary_backend,
        primary_ref=primary_ref,
        fallback_ref=fallback_ref,
        strict=strict,
        reason=reason,
    )
