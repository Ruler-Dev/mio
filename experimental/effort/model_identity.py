"""Content-bound model identities for reproducible MLX effort experiments.

MLX-LM deliberately treats an existing local path as authoritative and ignores
the ``revision`` argument.  A caller-supplied label therefore cannot establish
which local weights or tokenizer were measured.  This module fingerprints a
local MLX bundle once, before model loading, and requires that exact digest on
the benchmark command line.  Remote Hugging Face models are restricted to an
immutable 40-character commit revision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal, Sequence


LOCAL_MODEL_FINGERPRINT_SCHEMA = "mio.local-mlx-model-fingerprint.v1"
LOCAL_MODEL_REVISION_PREFIX = "local-sha256-v1:"
_MAX_MODEL_FILES = 10_000
_MODEL_FILE_SUFFIXES = frozenset(
    {
        ".jinja",
        ".json",
        ".jsonl",
        ".model",
        ".py",
        ".safetensors",
        ".tiktoken",
        ".txt",
    }
)
_IGNORED_DIRECTORY_NAMES = frozenset({".cache", ".git", "__pycache__"})


class ModelIdentityError(ValueError):
    """Raised when a model reference cannot be bound to immutable content."""


@dataclass(frozen=True)
class ModelFileIdentity:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LocalModelFingerprint:
    schema: str
    digest: str
    files: tuple[ModelFileIdentity, ...]
    total_bytes: int

    @property
    def revision(self) -> str:
        return f"{LOCAL_MODEL_REVISION_PREFIX}{self.digest}"


@dataclass(frozen=True)
class ResolvedModelReference:
    """One model resolution shared by identity, loading, and run provenance."""

    source_kind: Literal["local", "huggingface"]
    canonical_model_id: str
    load_model_id: str
    load_revision: str | None
    requested_model: str
    requested_revision: str


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_huggingface_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_model_file(path: Path) -> bool:
    return path.suffix.lower() in _MODEL_FILE_SUFFIXES


def _candidate_model_files(root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(name for name in directory_names if name not in _IGNORED_DIRECTORY_NAMES)
        directory_path = Path(directory)
        for name in directory_names:
            candidate_directory = directory_path / name
            if candidate_directory.is_symlink():
                raise ModelIdentityError("local model contains a symlinked directory")
        for name in sorted(file_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ModelIdentityError("local model contains a symlinked file")
            if _is_model_file(candidate):
                candidates.append(candidate)
                if len(candidates) > _MAX_MODEL_FILES:
                    raise ModelIdentityError("local model contains too many identity files")
    return tuple(sorted(candidates, key=lambda path: path.relative_to(root).as_posix()))


def _hash_regular_file(path: Path) -> tuple[int, str]:
    descriptor = -1
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ModelIdentityError("local model identity input must be a regular single-link file")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_lstat = os.lstat(path)
        named_stat = os.stat(path, follow_symlinks=True)
    except ModelIdentityError:
        raise
    except OSError as exc:
        raise ModelIdentityError("cannot hash local model identity input") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    before_identity = identity(before)
    if (
        before_identity != identity(after)
        or before_identity != identity(named_lstat)
        or before_identity != identity(named_stat)
        or before.st_nlink != 1
    ):
        raise ModelIdentityError("local model file changed while it was fingerprinted")
    return before.st_size, digest.hexdigest()


def fingerprint_local_model(path: Path | str) -> LocalModelFingerprint:
    """Hash all files that can affect MLX model or tokenizer loading."""

    root = Path(path).expanduser()
    if root.is_symlink():
        raise ModelIdentityError("local model root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelIdentityError("local model directory does not exist") from exc
    if not root.is_dir():
        raise ModelIdentityError("local model reference must be a directory")

    paths = _candidate_model_files(root)
    relative_paths = {candidate.relative_to(root).as_posix() for candidate in paths}
    if "config.json" not in relative_paths:
        raise ModelIdentityError("local model is missing config.json")
    if not any(
        Path(relative).parent == Path(".")
        and Path(relative).name.startswith("model")
        and relative.endswith(".safetensors")
        for relative in relative_paths
    ):
        raise ModelIdentityError("local model has no model*.safetensors weights")

    rows: list[ModelFileIdentity] = []
    for candidate in paths:
        size_bytes, file_sha256 = _hash_regular_file(candidate)
        rows.append(
            ModelFileIdentity(
                relative_path=candidate.relative_to(root).as_posix(),
                size_bytes=size_bytes,
                sha256=file_sha256,
            )
        )
    manifest = {
        "schema": LOCAL_MODEL_FINGERPRINT_SCHEMA,
        "files": [
            {
                "path": row.relative_path,
                "size_bytes": row.size_bytes,
                "sha256": row.sha256,
            }
            for row in rows
        ],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return LocalModelFingerprint(
        schema=LOCAL_MODEL_FINGERPRINT_SCHEMA,
        digest=hashlib.sha256(canonical).hexdigest(),
        files=tuple(rows),
        total_bytes=sum(row.size_bytes for row in rows),
    )


def resolve_model_reference(
    model: str,
    revision: str,
) -> ResolvedModelReference:
    """Resolve a local digest or immutable Hugging Face commit exactly once."""

    if not isinstance(model, str) or not model.strip():
        raise ModelIdentityError("model must be a non-empty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ModelIdentityError("model revision must be a non-empty string")

    expanded = Path(model).expanduser()
    looks_local = expanded.exists() or expanded.is_absolute() or model.startswith(("./", "../", "~"))
    if looks_local:
        fingerprint = fingerprint_local_model(expanded)
        if not revision.startswith(LOCAL_MODEL_REVISION_PREFIX):
            raise ModelIdentityError("local model revision must use local-sha256-v1:<digest>")
        requested_digest = revision.removeprefix(LOCAL_MODEL_REVISION_PREFIX)
        if not _is_sha256(requested_digest):
            raise ModelIdentityError("local model revision digest is malformed")
        if requested_digest != fingerprint.digest:
            raise ModelIdentityError("local model fingerprint does not match --model-revision")
        resolved_path = expanded.resolve(strict=True)
        return ResolvedModelReference(
            source_kind="local",
            canonical_model_id=f"local-mlx@{fingerprint.revision}",
            load_model_id=os.fspath(resolved_path),
            load_revision=None,
            requested_model=model,
            requested_revision=revision,
        )

    if not _is_huggingface_commit(revision):
        raise ModelIdentityError("remote model revision must be an immutable 40-character Hugging Face commit")
    if model.count("/") != 1 or any(part in {"", ".", ".."} for part in model.split("/")):
        raise ModelIdentityError("remote model must use the Hugging Face org/repository form")
    return ResolvedModelReference(
        source_kind="huggingface",
        canonical_model_id=f"hf://{model}@{revision}",
        load_model_id=model,
        load_revision=revision,
        requested_model=model,
        requested_revision=revision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the content-bound revision required for a local MLX model.",
    )
    parser.add_argument("--model", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    fingerprint = fingerprint_local_model(arguments.model)
    print(fingerprint.revision)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
