"""Fail-closed filesystem helpers for Mio's user-controlled local paths.

The Web UI writes trusted state below ``~/.mio`` and user deliverables below
``~/Downloads``, while Obsidian/RAG can traverse a user-selected directory.
These helpers keep lexical paths and symlink checks explicit so a symlinked
entry cannot silently redirect a listing, read, or write outside its root.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterator

from mio.paths import mio_home


class UnsafePathError(ValueError):
    """Raised when a path is absolute, traversing, or contains a symlink."""


def _relative_parts(value: str | os.PathLike[str], *, allow_nested: bool) -> tuple[str, ...]:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise UnsafePathError("path must be a non-empty string")
    if "\\" in raw:
        # Treat Windows separators as separators even when Mio runs on POSIX.
        raise UnsafePathError("backslash path separators are not allowed")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise UnsafePathError("absolute paths are not allowed")
    parts = tuple(posix.parts)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafePathError("path traversal is not allowed")
    if not allow_nested and len(parts) != 1:
        raise UnsafePathError("output filename must not contain directories")
    if len(raw.encode("utf-8")) > 4096 or any(
        len(part.encode("utf-8")) > 255 for part in parts
    ):
        raise UnsafePathError("path is too long")
    return parts


def relative_path_parts(
    value: str | os.PathLike[str],
    *,
    allow_nested: bool = True,
) -> tuple[str, ...]:
    """Public validation for archive members and other untrusted relative paths."""
    return _relative_parts(value, allow_nested=allow_nested)


def validate_directory(root: Path | str) -> Path:
    """Return a resolved directory after rejecting a symlinked root."""
    lexical = Path(root).expanduser()
    if lexical.is_symlink():
        raise UnsafePathError(f"symlinked directory is not allowed: {lexical}")
    try:
        mode = lexical.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise UnsafePathError(f"directory does not exist: {lexical}") from exc
    if not stat.S_ISDIR(mode):
        raise UnsafePathError(f"not a directory: {lexical}")
    return lexical.resolve(strict=True)


def ensure_directory_chain(
    base: Path | str,
    relative: Path | str,
    *,
    create: bool = False,
    mode: int = 0o700,
) -> Path:
    """Walk a relative directory chain without accepting symlink components."""
    current = Path(base).expanduser().resolve(strict=True)
    if not current.is_dir():
        raise UnsafePathError(f"not a directory: {current}")
    for part in _relative_parts(relative, allow_nested=True):
        candidate = current / part
        if candidate.is_symlink():
            raise UnsafePathError(f"symlinked directory is not allowed: {candidate}")
        try:
            candidate_mode = candidate.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            if not create:
                raise UnsafePathError(f"directory does not exist: {candidate}")
            candidate.mkdir(mode=mode)
            candidate_mode = candidate.stat(follow_symlinks=False).st_mode
        if not stat.S_ISDIR(candidate_mode):
            raise UnsafePathError(f"not a directory: {candidate}")
        current = candidate
    return current.resolve(strict=True)


def confined_path(
    root: Path | str,
    relative: Path | str,
    *,
    must_exist: bool = False,
    allow_nested: bool = True,
) -> Path:
    """Resolve ``relative`` below ``root`` without following any symlink."""
    root_path = validate_directory(root)
    current = root_path
    parts = _relative_parts(relative, allow_nested=allow_nested)
    for index, part in enumerate(parts):
        candidate = current / part
        if candidate.is_symlink():
            raise UnsafePathError(f"symlinked path is not allowed: {candidate}")
        exists = candidate.exists()
        if not exists:
            if must_exist or index < len(parts) - 1:
                raise UnsafePathError(f"path does not exist: {candidate}")
            current = candidate
            continue
        mode = candidate.stat(follow_symlinks=False).st_mode
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise UnsafePathError(f"path component is not a directory: {candidate}")
        current = candidate

    # Existing paths are resolved only after every lexical component passed
    # lstat-style checks. Missing output leaves keep their lexical path.
    resolved = current.resolve(strict=must_exist or current.exists())
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise UnsafePathError("path escapes its configured root") from exc
    return resolved


def downloads_output_path(filename: str, extension: str) -> Path:
    """Return a single-file output path confined to a real ``~/Downloads``."""
    if not extension.startswith(".") or "/" in extension or "\\" in extension:
        raise UnsafePathError("invalid output extension")
    name = filename if filename.endswith(extension) else filename + extension
    downloads = ensure_directory_chain(Path.home(), "Downloads", create=True)
    return confined_path(downloads, name, allow_nested=False)


def downloads_input_path(path: str | os.PathLike[str]) -> Path:
    """Return an existing regular input confined to a real ``~/Downloads``."""
    downloads = ensure_directory_chain(Path.home(), "Downloads", create=True)
    raw = Path(path).expanduser()
    if raw.is_absolute():
        try:
            relative = raw.relative_to(downloads).as_posix()
        except ValueError as exc:
            raise UnsafePathError("input must stay inside Downloads") from exc
    else:
        relative = os.fspath(path)
    source = confined_path(downloads, relative, must_exist=True)
    try:
        mode = source.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise UnsafePathError(f"cannot inspect input: {source}") from exc
    if not stat.S_ISREG(mode):
        raise UnsafePathError(f"input is not a regular file: {source}")
    return source


def mio_state_root(*, create: bool = False) -> Path:
    """Return Mio's canonical state root without following a root symlink."""
    state_root = mio_home()
    if state_root.is_symlink():
        raise UnsafePathError(f"symlinked directory is not allowed: {state_root}")
    if not state_root.exists():
        if not create:
            raise UnsafePathError(f"directory does not exist: {state_root}")
        state_root.mkdir(mode=0o700, parents=True)
    return validate_directory(state_root)


def mio_state_directory(relative: str, *, create: bool = False) -> Path:
    """Return a symlink-free directory beneath Mio's canonical state root."""
    return ensure_directory_chain(
        mio_state_root(create=create),
        relative,
        create=create,
    )


def iter_confined_regular_files(
    root: Path | str,
    *,
    suffixes: set[str] | None = None,
    recursive: bool = True,
    skip_directories: set[str] | None = None,
    skip_hidden_directories: bool = False,
    max_bytes: int | None = None,
) -> Iterator[Path]:
    """Yield regular files without following symlinks at any path level."""
    root_path = validate_directory(root)
    normalized_suffixes = {suffix.lower() for suffix in suffixes} if suffixes else None
    skipped = skip_directories or set()
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        directory = Path(dirpath)
        if recursive:
            dirnames[:] = [
                name
                for name in dirnames
                if name not in skipped
                and not (skip_hidden_directories and name.startswith("."))
                and not (directory / name).is_symlink()
            ]
        else:
            dirnames[:] = []
        for filename in filenames:
            candidate = directory / filename
            if normalized_suffixes and candidate.suffix.lower() not in normalized_suffixes:
                continue
            try:
                info = candidate.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                if max_bytes is not None and info.st_size > max_bytes:
                    continue
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root_path)
            except (FileNotFoundError, OSError, ValueError):
                continue
            yield resolved


def read_text_no_follow(path: Path | str, *, max_bytes: int | None = None) -> str:
    """Read a regular file while refusing a symlink at the final component."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("path is not a regular file")
        if max_bytes is not None and info.st_size > max_bytes:
            raise UnsafePathError(f"file exceeds {max_bytes} bytes")
        with os.fdopen(descriptor, "r", encoding="utf-8", errors="replace") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def open_binary_no_follow(
    path: Path | str,
    *,
    max_bytes: int | None = None,
) -> Iterator[BinaryIO]:
    """Open an existing regular file without following its final component."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    handle: BinaryIO | None = None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("path is not a regular file")
        if max_bytes is not None and info.st_size > max_bytes:
            raise UnsafePathError(f"file exceeds {max_bytes} bytes")
        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        yield handle
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def write_confined_text(
    root: Path | str,
    relative: str,
    content: str,
    *,
    create_parents: bool = False,
) -> Path:
    """Write a text file below ``root`` without following a symlink leaf."""
    root_path = validate_directory(root)
    parts = _relative_parts(relative, allow_nested=True)
    parent = root_path
    for part in parts[:-1]:
        candidate = parent / part
        if candidate.is_symlink():
            raise UnsafePathError(f"symlinked directory is not allowed: {candidate}")
        try:
            mode = candidate.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            if not create_parents:
                raise UnsafePathError(f"directory does not exist: {candidate}")
            candidate.mkdir(mode=0o700)
            mode = candidate.stat(follow_symlinks=False).st_mode
        if not stat.S_ISDIR(mode):
            raise UnsafePathError(f"path component is not a directory: {candidate}")
        parent = candidate

    output = confined_path(root_path, "/".join(parts), allow_nested=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(output), flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return output


def _confined_binary_output(
    root: Path | str,
    relative: str,
    *,
    create_parents: bool,
) -> tuple[Path, int]:
    root_path = validate_directory(root)
    parts = _relative_parts(relative, allow_nested=True)
    if parts[:-1]:
        ensure_directory_chain(
            root_path,
            "/".join(parts[:-1]),
            create=create_parents,
        )
    output = confined_path(root_path, "/".join(parts), allow_nested=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    return output, os.open(os.fspath(output), flags, 0o600)


@contextmanager
def open_confined_binary_writer(
    root: Path | str,
    relative: str,
    *,
    create_parents: bool = False,
) -> Iterator[tuple[Path, BinaryIO]]:
    """Open one confined binary output without following its final component."""
    output, descriptor = _confined_binary_output(
        root,
        relative,
        create_parents=create_parents,
    )
    handle: BinaryIO | None = None
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        yield output, handle
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def write_confined_bytes(
    root: Path | str,
    relative: str,
    content: bytes,
    *,
    create_parents: bool = False,
) -> Path:
    """Write bytes to a confined, no-follow output."""
    with open_confined_binary_writer(
        root,
        relative,
        create_parents=create_parents,
    ) as (output, handle):
        handle.write(content)
    return output


def confined_markdown_tree(
    root: Path | str,
    *,
    max_depth: int = 12,
    skip_names: set[str] | None = None,
) -> list[dict]:
    """Build the Obsidian tree shape without following directory/file links."""
    root_path = validate_directory(root)
    skipped = skip_names or {"node_modules"}

    def walk(directory: Path, depth: int) -> list[dict]:
        if depth > max_depth:
            return []
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name.lower())
        except (OSError, PermissionError):
            return []
        items: list[dict] = []
        for entry in entries:
            if entry.name.startswith(".") or entry.name in skipped or entry.is_symlink():
                continue
            path = Path(entry.path)
            relative = path.relative_to(root_path).as_posix()
            try:
                if entry.is_dir(follow_symlinks=False):
                    items.append({
                        "type": "folder",
                        "name": entry.name,
                        "path": relative,
                        "children": walk(path, depth + 1),
                    })
                elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in {
                    ".md", ".markdown",
                }:
                    info = entry.stat(follow_symlinks=False)
                    items.append({
                        "type": "note",
                        "name": entry.name,
                        "path": relative,
                        "size": info.st_size,
                        "mtime": info.st_mtime,
                    })
            except OSError:
                continue
        return items

    return walk(root_path, 0)
