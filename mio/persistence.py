"""Crash-safe persistence primitives for Mio-owned JSON state.

State files are written beside their final destination, flushed to stable
storage, and published with one atomic ``os.replace``.  A process or machine
failure can therefore leave either the previous complete document or the new
complete document, never a partially-written JSON file.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Iterator


_path_locks_guard = threading.Lock()
_path_locks: dict[Path, threading.RLock] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    """Return the in-process half of a path's transaction lock."""
    canonical = path.absolute()
    with _path_locks_guard:
        lock = _path_locks.get(canonical)
        if lock is None:
            lock = threading.RLock()
            _path_locks[canonical] = lock
        return lock


@contextlib.contextmanager
def _exclusive_path_lock(path: Path, *, mode: int = 0o600) -> Iterator[None]:
    """Serialize one transaction across both threads and POSIX processes.

    ``flock`` is advisory, so all Mio writers must use this primitive.  The
    per-path ``RLock`` also gives deterministic thread semantics on platforms
    where process-lock ownership is broader than an individual file handle.
    """
    import fcntl

    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    with _thread_lock_for(destination):
        descriptor = os.open(lock_path, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry after an atomic replace when supported."""
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
        # Some filesystems do not support fsync on directory descriptors.  The
        # file itself has already been fsynced before publication.
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace ``path`` with already-serialized ``payload``.

    The temporary file is created with ``mkstemp`` in the destination
    directory, which both prevents symlink races and keeps ``os.replace`` on
    the same filesystem.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    mode: int = 0o600,
) -> None:
    """Serialize and atomically publish a UTF-8 JSON document."""
    # Serialize before touching the filesystem: non-JSON input must leave an
    # existing valid state document completely unchanged.
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload, mode=mode)


def atomic_update_json(
    path: Path,
    update: Callable[[Any], Any],
    *,
    default_factory: Callable[[], Any] = dict,
    indent: int | None = 2,
    sort_keys: bool = False,
    mode: int = 0o600,
) -> Any:
    """Atomically apply a read-modify-write transaction to a JSON document.

    ``os.replace`` makes publication crash-safe, while the adjacent lock file
    prevents two threads or processes from both reading the same old document
    and silently losing one update.  Invalid existing JSON is deliberately not
    overwritten: callers see the decode error and can recover explicitly.
    """
    destination = Path(path).absolute()
    with _exclusive_path_lock(destination, mode=mode):
        try:
            current = json.loads(destination.read_text(encoding="utf-8"))
        except FileNotFoundError:
            current = default_factory()
        replacement = update(current)
        atomic_write_json(
            destination,
            replacement,
            indent=indent,
            sort_keys=sort_keys,
            mode=mode,
        )
    return replacement


__all__ = ["atomic_update_json", "atomic_write_bytes", "atomic_write_json"]
