"""Security policy and confined filesystem primitives for the native agent.

The model never supplies this policy.  A trusted caller constructs it and the
agent dispatcher injects it into tool calls, keeping authorization separate
from model-generated arguments.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import platform
import secrets
import stat
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Any

_AUDIT_LOG = logging.getLogger("mio.agent.audit")
_MAX_OUTPUT_LIMIT_CHARS = 1_000_000
_MAX_FILE_LIMIT_CHARS = 16_000_000
_MAX_COMMAND_TIMEOUT_S = 600.0


class AgentToolPermission(str, Enum):
    """Independent native-agent capabilities granted by a trusted caller."""

    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"


class AgentPolicyError(ValueError):
    """Base class for denied agent-tool operations."""


class AgentPermissionDenied(AgentPolicyError):
    """Raised when a caller did not explicitly grant a capability."""


class AgentPathViolation(AgentPolicyError):
    """Raised when a path escapes or aliases outside the workspace boundary."""


@dataclass(frozen=True)
class AgentAuditEvent:
    """One bounded, content-free audit record for an agent tool operation."""

    timestamp: float
    operation: str
    permission: str
    target: str
    allowed: bool
    outcome: str
    detail: str = ""


AuditSink = Callable[[AgentAuditEvent], None]


def is_broad_workspace_root(path: str | os.PathLike[str]) -> bool:
    """Return whether a root is too broad for an implicit agent grant."""

    candidate = Path(path).expanduser().resolve()
    filesystem_root = Path("/")
    home_roots = {Path.home().resolve()}
    try:
        import pwd

        home_roots.add(Path(pwd.getpwuid(os.getuid()).pw_dir).resolve())
    except (ImportError, KeyError, OSError):  # pragma: no cover - non-POSIX fallback
        pass

    # A top-level tree is never a sensible implicit coding workspace. Resolve
    # every comparison so macOS aliases such as /tmp -> /private/tmp cannot
    # bypass the guard.
    if (
        candidate == filesystem_root
        or candidate in home_roots
        or candidate.parent == filesystem_root
    ):
        return True

    broad_roots = {
        Path("/Users").resolve(),
        Path("/Users/Shared").resolve(),
        Path("/Volumes").resolve(),
        Path("/Network").resolve(),
        Path("/private").resolve(),
        Path("/private/etc").resolve(),
        Path("/private/tmp").resolve(),
        Path("/private/var").resolve(),
        Path("/usr/local").resolve(),
        Path("/opt/homebrew").resolve(),
    }
    if candidate in broad_roots:
        return True

    # Mount roots grant an entire disk/share. A directory below the mount root
    # is narrow enough to be an ordinary project workspace.
    for mount_root in (Path("/Volumes").resolve(), Path("/Network").resolve()):
        try:
            relative = candidate.relative_to(mount_root)
        except ValueError:
            continue
        if len(relative.parts) <= 1:
            return True
    return False


def _default_audit_sink(event: AgentAuditEvent) -> None:
    _AUDIT_LOG.info("agent_tool_audit %s", json.dumps(asdict(event), sort_keys=True))


def _normalize_permissions(
    permissions: Iterable[AgentToolPermission | str],
) -> frozenset[AgentToolPermission]:
    return frozenset(AgentToolPermission(permission) for permission in permissions)


@dataclass(frozen=True)
class AgentToolPolicy:
    """Explicit authority and resource limits for native agent tools.

    Workspace roots must already exist.  They are canonicalized once when the
    policy is created, and every later file operation is performed relative to
    a no-follow directory descriptor rooted in one of these directories.
    """

    workspace_roots: tuple[Path, ...]
    permissions: frozenset[AgentToolPermission]
    output_limit_chars: int = 10_000
    file_limit_chars: int = 4_000_000
    command_timeout_s: float = 30.0
    audit_sink: AuditSink = _default_audit_sink

    def __post_init__(self) -> None:
        roots: list[Path] = []
        for raw_root in self.workspace_roots:
            try:
                root = Path(raw_root).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"workspace root does not exist: {raw_root}") from exc
            if not root.is_dir():
                raise ValueError(f"workspace root is not a directory: {raw_root}")
            if root not in roots:
                roots.append(root)
        if not roots:
            raise ValueError("at least one workspace root is required")
        if (
            isinstance(self.output_limit_chars, bool)
            or not isinstance(self.output_limit_chars, int)
            or not 32 <= self.output_limit_chars <= _MAX_OUTPUT_LIMIT_CHARS
        ):
            raise ValueError(
                f"output_limit_chars must be an integer in [32, {_MAX_OUTPUT_LIMIT_CHARS}]"
            )
        if (
            isinstance(self.file_limit_chars, bool)
            or not isinstance(self.file_limit_chars, int)
            or not 1_024 <= self.file_limit_chars <= _MAX_FILE_LIMIT_CHARS
        ):
            raise ValueError(
                f"file_limit_chars must be an integer in [1024, {_MAX_FILE_LIMIT_CHARS}]"
            )
        if (
            isinstance(self.command_timeout_s, bool)
            or not isinstance(self.command_timeout_s, (int, float))
            or not math.isfinite(float(self.command_timeout_s))
            or not 0 < float(self.command_timeout_s) <= _MAX_COMMAND_TIMEOUT_S
        ):
            raise ValueError(
                f"command_timeout_s must be finite and in (0, {_MAX_COMMAND_TIMEOUT_S}]"
            )
        if not callable(self.audit_sink):
            raise TypeError("audit_sink must be callable")
        object.__setattr__(self, "workspace_roots", tuple(roots))
        object.__setattr__(self, "permissions", _normalize_permissions(self.permissions))

    @property
    def primary_workspace(self) -> Path:
        return self.workspace_roots[0]

    @classmethod
    def read_only(
        cls,
        workspace_root: str | os.PathLike[str],
        *,
        additional_roots: Iterable[str | os.PathLike[str]] = (),
        output_limit_chars: int = 10_000,
        file_limit_chars: int = 4_000_000,
        audit_sink: AuditSink = _default_audit_sink,
    ) -> AgentToolPolicy:
        """Build a policy that can only inspect explicitly listed workspaces."""

        roots = (Path(workspace_root), *(Path(root) for root in additional_roots))
        return cls(
            workspace_roots=roots,
            permissions=frozenset({AgentToolPermission.READ}),
            output_limit_chars=output_limit_chars,
            file_limit_chars=file_limit_chars,
            audit_sink=audit_sink,
        )

    @classmethod
    def coding_workspace(
        cls,
        workspace_root: str | os.PathLike[str],
        *,
        additional_roots: Iterable[str | os.PathLike[str]] = (),
        output_limit_chars: int = 10_000,
        file_limit_chars: int = 4_000_000,
        command_timeout_s: float = 30.0,
        allow_network: bool = False,
        audit_sink: AuditSink = _default_audit_sink,
    ) -> AgentToolPolicy:
        """Build the declared compatibility policy for the coding-agent CLI."""

        roots = (Path(workspace_root), *(Path(root) for root in additional_roots))
        permissions = {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
        }
        if allow_network:
            permissions.add(AgentToolPermission.NETWORK)
        return cls(
            workspace_roots=roots,
            permissions=frozenset(permissions),
            output_limit_chars=output_limit_chars,
            file_limit_chars=file_limit_chars,
            command_timeout_s=command_timeout_s,
            audit_sink=audit_sink,
        )

    def require(self, permission: AgentToolPermission) -> None:
        if permission not in self.permissions:
            raise AgentPermissionDenied(
                f"{permission.value} requires an explicit AgentToolPolicy grant"
            )

    def audit(
        self,
        *,
        operation: str,
        permission: AgentToolPermission,
        target: str,
        allowed: bool,
        outcome: str,
        detail: str = "",
    ) -> None:
        """Emit an audit event without allowing a broken sink to break a tool."""

        event = AgentAuditEvent(
            timestamp=time.time(),
            operation=operation[:64],
            permission=permission.value,
            target=target[:512],
            allowed=allowed,
            outcome=outcome[:64],
            detail=detail[:512],
        )
        try:
            self.audit_sink(event)
        except Exception:  # pragma: no cover - logging must never change tool behavior
            _AUDIT_LOG.exception("agent audit sink failed")


@dataclass(frozen=True)
class ResolvedWorkspacePath:
    root: Path
    absolute: Path
    relative: Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace_path(
    raw_path: str | os.PathLike[str],
    policy: AgentToolPolicy,
) -> ResolvedWorkspacePath:
    """Resolve a path lexically inside the allowlist and reject all symlinks.

    The no-symlink rule is deliberately stronger than merely checking the
    resolved destination: an in-workspace symlink cannot later be swapped to
    escape the root between validation and use.
    """

    if not isinstance(raw_path, (str, os.PathLike)):
        raise AgentPathViolation("path must be a string or path-like value")
    rendered = os.fspath(raw_path)
    if not rendered or "\x00" in rendered:
        raise AgentPathViolation("path must be non-empty and contain no NUL bytes")

    expanded = Path(rendered).expanduser()
    if ".." in expanded.parts:
        raise AgentPathViolation("parent traversal is not allowed")
    if expanded.is_absolute():
        candidate = Path(os.path.abspath(os.fspath(expanded)))
    else:
        candidate = Path(os.path.abspath(policy.primary_workspace / expanded))

    matching_roots = [root for root in policy.workspace_roots if _is_relative_to(candidate, root)]
    if not matching_roots:
        raise AgentPathViolation("path is outside the workspace allowlist")
    # Prefer the narrowest explicitly listed root when roots are nested.
    root = max(matching_roots, key=lambda item: len(item.parts))
    relative = candidate.relative_to(root)
    if not relative.parts:
        raise AgentPathViolation("workspace root is a directory, not a file")

    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise AgentPathViolation(f"cannot validate workspace path: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise AgentPathViolation("symlink paths are not allowed")

    return ResolvedWorkspacePath(root=root, absolute=candidate, relative=relative)


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _translate_nofollow_error(exc: OSError) -> OSError:
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return AgentPathViolation("symlink or non-directory path component is not allowed")
    return exc


@contextmanager
def _open_parent(
    resolved: ResolvedWorkspacePath,
    *,
    create: bool,
) -> Any:
    """Yield ``(parent_dir_fd, final_name)`` using an openat no-follow walk."""

    flags = _directory_open_flags()
    root_fd = os.open(resolved.root, flags)
    current_fd = root_fd
    try:
        for part in resolved.relative.parts[:-1]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(part, flags, dir_fd=current_fd)
                except OSError as exc:
                    raise _translate_nofollow_error(exc) from exc
            except OSError as exc:
                raise _translate_nofollow_error(exc) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        yield current_fd, resolved.relative.parts[-1]
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def read_workspace_text(
    resolved: ResolvedWorkspacePath,
    *,
    max_chars: int | None,
) -> tuple[str, bool]:
    """Read through a no-follow workspace descriptor, optionally bounded."""

    with _open_parent(resolved, create=False) as (parent_fd, name):
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise _translate_nofollow_error(exc) from exc
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise AgentPathViolation("path is not a regular file")
            if file_stat.st_nlink > 1:
                raise AgentPathViolation("hard-linked files are not allowed")
            stream: IO[str] = os.fdopen(fd, "r", encoding="utf-8", errors="replace")
            fd = -1
            with stream:
                content = stream.read() if max_chars is None else stream.read(max_chars + 1)
        finally:
            if fd >= 0:
                os.close(fd)
    if max_chars is None:
        return content, False
    return content[:max_chars], len(content) > max_chars


def write_workspace_text(
    resolved: ResolvedWorkspacePath,
    content: str,
) -> None:
    """Atomically replace a workspace file without following its inode.

    Atomic replacement also prevents an existing hard link from turning a
    confined write into a mutation of another pathname's inode.
    """

    if not isinstance(content, str):
        raise TypeError("content must be a string")
    with _open_parent(resolved, create=True) as (parent_fd, name):
        temp_name = f".mio-agent-{secrets.token_hex(12)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        mode = 0o666
        preserve_mode = False
        try:
            existing = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(existing.st_mode):
                mode = stat.S_IMODE(existing.st_mode)
                preserve_mode = True
        fd = os.open(temp_name, flags, mode, dir_fd=parent_fd)
        try:
            if preserve_mode:
                os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                fd = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        except Exception:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise
        finally:
            if fd >= 0:
                os.close(fd)


def cap_tool_output(text: str, limit: int) -> str:
    """Return a response whose complete length never exceeds ``limit``."""

    if len(text) <= limit:
        return text
    suffix = "\n... (output truncated)"
    if len(suffix) >= limit:
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def _sandbox_string(value: str | os.PathLike[str]) -> str:
    """Quote a literal for Apple's Scheme-like sandbox profile language."""

    # SBPL accepts UTF-8 string literals but does not interpret JSON surrogate
    # pairs as a filesystem path. Keep non-ASCII characters literal while
    # retaining JSON's safe quote/backslash/control escaping.
    return json.dumps(os.fspath(value), ensure_ascii=False)


def reject_hardlinked_workspace_files(policy: AgentToolPolicy) -> None:
    """Reject path aliases that an inherited pathname sandbox cannot express.

    Apple's inherited sandbox authorizes pathnames, not inode provenance. A
    regular file hard-linked into a workspace could otherwise expose or mutate
    an outside pathname through its allowed alias. The model cannot create an
    outside hard link from inside the default-deny profile, so a trusted
    preflight before every child launch closes that model-controlled route.
    Concurrent mutation by another already-authorized host process remains an
    operating-system trust boundary rather than an agent capability.
    """

    _reject_hardlinked_roots(policy.workspace_roots)


def _reject_hardlinked_roots(roots: Iterable[Path]) -> None:
    """Reject regular-file inode aliases below trusted sandbox roots."""

    def fail_closed(exc: OSError) -> None:
        raise AgentPathViolation(
            f"cannot audit sandbox links: {type(exc).__name__}"
        ) from exc

    for root in roots:
        for directory, child_directories, filenames in os.walk(
            root,
            followlinks=False,
            onerror=fail_closed,
        ):
            # os.walk lists symlinked directories separately; remove them so a
            # host-created link cannot redirect this trusted preflight.
            retained_directories: list[str] = []
            for name in child_directories:
                path = Path(directory, name)
                try:
                    mode = path.lstat().st_mode
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise AgentPathViolation(
                        f"cannot audit workspace links: {type(exc).__name__}"
                    ) from exc
                if not stat.S_ISLNK(mode):
                    retained_directories.append(name)
            child_directories[:] = retained_directories

            for name in filenames:
                path = Path(directory, name)
                try:
                    file_stat = path.lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise AgentPathViolation(
                        f"cannot audit workspace links: {type(exc).__name__}"
                    ) from exc
                if stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink > 1:
                    try:
                        relative = path.relative_to(root)
                    except ValueError:  # pragma: no cover - os.walk stays rooted
                        relative = Path(name)
                    rendered = os.fspath(relative)
                    if len(rendered) > 240:
                        rendered = "..." + rendered[-237:]
                    raise AgentPathViolation(
                        f"hard-linked workspace files are not allowed for shell execution: {rendered}"
                    )


def sandboxed_command(
    argv: list[str],
    policy: AgentToolPolicy,
    *,
    read_only_roots: Iterable[str | os.PathLike[str]] = (),
    allow_process_fork: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Wrap one command in a workspace-confined inherited process sandbox.

    Mio's inference stack targets Apple Silicon, so the native coding agent
    fails closed when the macOS sandbox launcher is unavailable.  The profile
    preserves normal system/library reads, denies sensitive/user locations,
    denies every out-of-workspace write, and denies networking unless a trusted
    caller separately granted ``NETWORK``.  Read/write workspace authority is
    generated independently from the shell grant.
    """

    if not argv:
        raise ValueError("argv must not be empty")
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if platform.system() != "Darwin" or not sandbox_exec.is_file():
        raise AgentPermissionDenied("a supported inherited process sandbox is unavailable")

    canonical_read_only_roots: list[Path] = []
    for raw_root in read_only_roots:
        try:
            root = Path(raw_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise AgentPathViolation(
                f"sandbox read-only root does not exist: {raw_root}"
            ) from exc
        if not root.is_dir():
            raise AgentPathViolation(
                f"sandbox read-only root is not a directory: {raw_root}"
            )
        if root not in canonical_read_only_roots:
            canonical_read_only_roots.append(root)

    reject_hardlinked_workspace_files(policy)
    _reject_hardlinked_roots(canonical_read_only_roots)

    allow_roots = " ".join(
        f"(subpath {_sandbox_string(root)})" for root in policy.workspace_roots
    )
    read_only_allow_roots = " ".join(
        f"(subpath {_sandbox_string(root)})" for root in canonical_read_only_roots
    )
    ancestor_roots: list[Path] = []
    for sandbox_root in (*policy.workspace_roots, *canonical_read_only_roots):
        for ancestor in reversed(sandbox_root.parents):
            if ancestor not in ancestor_roots:
                ancestor_roots.append(ancestor)
    ancestor_metadata = " ".join(
        f"(literal {_sandbox_string(ancestor)})" for ancestor in ancestor_roots
    )
    process_rule = (
        "(allow process-fork process-exec)"
        if allow_process_fork
        else "(allow process-exec)"
    )
    rules = [
        "(version 1)",
        "(deny default)",
        process_rule,
        "(allow process-info* signal (target self) (target children))",
        "(allow sysctl-read)",
        # Prevent descendants from escaping the trusted supervisor's process
        # group before it performs timeout/completion cleanup.
        "(deny syscall-unix (syscall-number SYS_setsid SYS_setpgid))",
        # macOS command-line runtimes read many public system paths while
        # loading frameworks/locales.  Default-deny still blocks Mach/XPC,
        # Apple Events, IOKit, process inspection and every write.  Sensitive
        # and user-owned trees are closed again below, then only declared
        # workspaces are reopened according to READ/WRITE grants.
        "(allow file-read*)",
        '(deny file-read* (subpath "/Users") (subpath "/Volumes") '
        '(subpath "/Network") (subpath "/private/var/folders") '
        '(subpath "/private/var/db") (subpath "/private/var/tmp") '
        '(subpath "/private/tmp") (subpath "/private/etc") '
        '(subpath "/Library/Keychains") (subpath "/Library/Preferences") '
        '(subpath "/Library/Application Support") (subpath "/Library/Caches") '
        '(subpath "/Library/Logs"))',
        '(allow file-read* (subpath "/private/var/db/timezone") '
        '(literal "/private/etc/localtime"))',
        '(allow file-read* file-write* (literal "/dev/null") '
        '(literal "/dev/zero") (subpath "/dev/fd"))',
    ]
    if AgentToolPermission.READ in policy.permissions:
        rules.append(f"(allow file-read* {allow_roots})")
    else:
        rules.append(f"(deny file-read* {allow_roots})")
    if ancestor_metadata:
        # Runtimes such as Node realpath every path component before opening an
        # allowed executable. Permit stat/lstat only on already-declared
        # ancestors; directory listings and file data remain denied.
        rules.append(f"(allow file-read-metadata {ancestor_metadata})")
    if read_only_allow_roots:
        rules.append(f"(allow file-read* {read_only_allow_roots})")
    if AgentToolPermission.WRITE in policy.permissions:
        rules.append(f"(allow file-write* {allow_roots})")
    if read_only_allow_roots:
        # A runtime executable/package root is not provider data. This explicit
        # deny also wins when an operator accidentally declares overlapping
        # roots, so adding runtime compatibility can never widen write access.
        rules.append(f"(deny file-write* {read_only_allow_roots})")
    if AgentToolPermission.NETWORK in policy.permissions:
        rules.extend(
            [
                "(allow network*)",
                '(allow file-read* (subpath "/private/etc/ssl") '
                '(literal "/private/etc/resolv.conf") '
                '(literal "/private/etc/hosts"))',
                '(allow mach-lookup '
                '(global-name "com.apple.SystemConfiguration.configd") '
                '(global-name "com.apple.system.opendirectoryd.libinfo") '
                '(global-name "com.apple.system.DirectoryService.libinfo_v1") '
                '(global-name "com.apple.mDNSResponder"))',
            ]
        )
    profile = "\n".join(rules)

    # Do not expose API keys, cloud credentials, SSH agents, or caller-specific
    # Python injection variables to a model-selected child process.
    path_candidates = [
        "/Library/Frameworks/Python.framework/Versions/Current/bin",
        str(Path(os.sys.executable).resolve().parent),
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/System/Cryptexes/App/usr/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    for root in policy.workspace_roots:
        path_candidates.extend(
            [str(root / ".venv" / "bin"), str(root / "node_modules" / ".bin")]
        )
    safe_path = ":".join(dict.fromkeys(path_candidates))
    environment = {
        "PATH": safe_path,
        "HOME": str(policy.primary_workspace),
        "TMPDIR": str(policy.primary_workspace),
        "ZDOTDIR": "/var/empty",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "PAGER": "cat",
    }
    for name in ("LANG", "LC_ALL", "LC_CTYPE", "TERM"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return [str(sandbox_exec), "-p", profile, *argv], environment
