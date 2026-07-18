"""Deterministic, content-free quality evidence for Mio coding-agent turns.

The gate lives beside the trusted tool dispatcher.  It never treats assistant
prose or shell output as evidence: mutations come from audit events or bounded
workspace snapshots, and validations come from the dedicated direct-argv tool
with its real exit metadata.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from mio.agent_policy import (
    AgentAuditEvent,
    AgentPathViolation,
    AgentPermissionDenied,
    AgentToolPolicy,
    sandboxed_command,
)


class CodingEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    ULTRA = "ultra"


class RequestIntent(str, Enum):
    GENERAL = "general"
    INSPECT = "inspect"
    CODE_CHANGE_REQUESTED = "code_change_requested"


class ValidationKind(str, Enum):
    TEST = "test"
    BUILD = "build"
    STATIC = "static"
    DIFF = "diff"
    REVIEW = "review"


class GateStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    INCOMPLETE = "incomplete"
    PASS = "pass"


_CHANGE_PATTERN = re.compile(
    r"\b(?:add|build|change|create|delete|edit|fix|implement|migrate|modify|patch|"
    r"refactor|remove|rename|repair|replace|update|write|aggiung\w*|aggiorn\w*|"
    r"complet\w*|configur\w*|corregg\w*|crea\w*|implement\w*|install\w*|"
    r"miglior\w*|modific\w*|ottimizz\w*|programm\w*|rimuov\w*|scriv\w*|"
    r"sostitui\w*)\b",
    re.IGNORECASE,
)
_NEGATED_CHANGE_PATTERN = re.compile(
    r"\b(?:do\s+not|don't|without)\s+(?:change|edit|modify|write)\b|"
    r"\b(?:non|senza)\s+(?:cambiare|modificare|scrivere)\b",
    re.IGNORECASE,
)
_INSPECT_PATTERN = re.compile(
    r"\b(?:analy[sz]e|audit|diagnos\w*|explain|inspect|review|understand|"
    r"analizz\w*|controll\w*|diagnostic\w*|ispezion\w*|spieg\w*)\b",
    re.IGNORECASE,
)
_DOC_SUFFIXES = frozenset({".adoc", ".md", ".mdx", ".rst", ".txt"})
_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".m",
        ".metal",
        ".mm",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".vue",
        ".proto",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_SOURCE_NAMES = frozenset(
    {
        "dockerfile",
        "build",
        "build.bazel",
        "cmakelists.txt",
        "gemfile",
        "justfile",
        "makefile",
        "meson.build",
        "package-lock.json",
        "package.json",
        "pom.xml",
        "pyproject.toml",
        "requirements.txt",
        "workspace",
    }
)
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "models",
        "node_modules",
        "spd",
    }
)
_MAX_GIT_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_FILES = 20_000
_MAX_MANIFEST_BYTES = 256 * 1024 * 1024
_GIT_TIMEOUT_S = 8.0


def classify_request_intent(text: str) -> RequestIntent:
    """Advisory request classification; workspace evidence remains authoritative."""

    if not isinstance(text, str):
        raise TypeError("request text must be a string")
    if _NEGATED_CHANGE_PATTERN.search(text):
        return RequestIntent.INSPECT
    if _CHANGE_PATTERN.search(text):
        return RequestIntent.CODE_CHANGE_REQUESTED
    if _INSPECT_PATTERN.search(text):
        return RequestIntent.INSPECT
    return RequestIntent.GENERAL


def _argv_digest(argv: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(argv).encode("utf-8", errors="replace")).hexdigest()


def _semantic_argv_digest(argv: Sequence[str]) -> str:
    """Hash validation scope while ignoring presentation-only verbosity."""

    values = list(argv)
    if values:
        executable = Path(values[0]).name.lower()
        if executable.startswith("python") or executable in {"pypy", "pypy3"}:
            parsed = _python_module(values[1:])
            if parsed is not None and parsed[0] in {"pytest", "unittest"}:
                values = [parsed[0], *parsed[1]]
        elif executable == "py.test":
            values[0] = "pytest"

    presentation_flags = {
        "--disable-warnings",
        "--no-header",
        "--no-summary",
        "--quiet",
        "--silent",
        "--verbose",
    }
    presentation_prefixes = (
        "--color=",
        "--durations=",
        "--durations-min=",
        "--show-capture=",
        "--tb=",
        "--verbosity=",
    )
    presentation_value_options = {
        "--color",
        "--durations",
        "--durations-min",
        "--show-capture",
        "--tb",
        "--verbosity",
        "-r",
    }
    normalized: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        lowered = value.lower()
        if lowered in presentation_value_options:
            index += 2
            continue
        if (
            lowered in presentation_flags
            or re.fullmatch(r"-[qv]+", lowered)
            or re.fullmatch(r"-r[a-zA-Z]*", value)
            or lowered.startswith(presentation_prefixes)
        ):
            index += 1
            continue
        normalized.append(value)
        index += 1
    if normalized and normalized[0].lower() == "pytest":
        normalized = [value for index, value in enumerate(normalized) if index == 0 or value not in {".", "./"}]
    return _argv_digest(normalized)


def _contains_shell_grammar(argv: Sequence[str]) -> bool:
    operators = {"&&", "||", ";", "|", "&", ">", ">>", "<", "2>", "2>&1"}
    return any(value in operators or "\n" in value or "\r" in value for value in argv)


def _requests_non_execution(args: Sequence[str]) -> bool:
    blocked = {
        "--collect-only",
        "--co",
        "--dry-run",
        "--dry",
        "--exit-zero",
        "--env-info",
        "--createstub",
        "--clean",
        "--fixtures",
        "--fixtures-per-test",
        "--fix-only",
        "--help",
        "--help-command",
        "--help-full",
        "--if-present",
        "--install-only",
        "--init",
        "--list",
        "--list-tests",
        "--listfilesonly",
        "--listenvs",
        "--list-sessions",
        "--listtests",
        "--markers",
        "--no-run",
        "--notest",
        "--setup-only",
        "--setup-plan",
        "--show-bin-path",
        "--show-only",
        "--show-files",
        "--show-settings",
        "--showconfig",
        "--print-config",
        "--print-labels",
        "--passwithnotests",
        "--stdin",
        "--version",
        "-",
        "-h",
    }
    blocked_prefixes = (
        "--co=",
        "--collect-only=",
        "--help=",
        "--help-command=",
        "--help-",
        "-list=",
        "--list-tests=",
        "--listtests=",
        "--show-only=",
    )
    if any(value in blocked or value.startswith(blocked_prefixes) for value in args):
        return True
    if any(value.startswith(("-dskiptests", "-dmaven.test.skip", "--exclude-task=test")) for value in args):
        return True
    return any(left == "-x" and right == "test" for left, right in zip(args, args[1:]))


def _python_module(args: Sequence[str]) -> tuple[str, list[str]] | None:
    """Return ``(module, module_args)`` after harmless interpreter flags."""

    index = 0
    flag_without_value = {
        "-b",
        "-B",
        "-d",
        "-E",
        "-I",
        "-O",
        "-OO",
        "-P",
        "-q",
        "-R",
        "-s",
        "-S",
        "-u",
        "-v",
        "-x",
    }
    while index < len(args):
        value = args[index]
        if value in {"-c", "-V", "-VV", "--check-hash-based-pycs", "--version"}:
            return None
        if value == "-m":
            if index + 1 >= len(args):
                return None
            return args[index + 1].lower(), list(args[index + 2 :])
        if value in flag_without_value:
            index += 1
            continue
        if value in {"-W", "-X"}:
            if index + 1 >= len(args):
                return None
            index += 2
            continue
        if value.startswith(("-W", "-X")) and len(value) > 2:
            index += 1
            continue
        return None
    return None


def infer_validation_kind(argv: Sequence[str]) -> ValidationKind | None:
    """Infer a validation category from conservative direct argv.

    Shells, wrappers, inline interpreters, and success-masking grammar are
    rejected.  Unknown commands execute through ``bash`` if authorized, but
    they cannot certify a coding-quality obligation.
    """

    if (
        isinstance(argv, (str, bytes))
        or not isinstance(argv, Sequence)
        or not argv
        or any(not isinstance(value, str) or not value or "\x00" in value for value in argv)
        or _contains_shell_grammar(argv)
    ):
        return None
    executable = Path(argv[0]).name.lower()
    raw_args = list(argv[1:])
    args = [value.lower() for value in raw_args]
    if executable != argv[0].lower() or "/" in argv[0] or "\\" in argv[0]:
        return None
    if _requests_non_execution(args):
        return None
    shells = {
        "bash",
        "cmd",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "pwsh",
        "sh",
        "zsh",
    }
    wrappers = {"env", "nice", "pipenv", "poetry", "sudo", "timeout", "uv", "xargs"}
    if executable in shells or executable in wrappers:
        return None

    if executable.startswith("python") or executable in {"pypy", "pypy3"}:
        parsed_module = _python_module(raw_args)
        if parsed_module is None:
            return None
        module, module_args = parsed_module
        if _requests_non_execution(module_args):
            return None
        if module == "pytest" and any(re.fullmatch(r"-V+", value) for value in module_args):
            return None
        if module in {"pytest", "unittest"}:
            return ValidationKind.TEST
        if module in {"compileall", "mypy", "pyright"}:
            if module == "compileall" and any(re.fullmatch(r"-q+", value) for value in module_args):
                return None
            if module == "mypy" and "-V" in module_args:
                return None
            return ValidationKind.STATIC
        if module == "ruff":
            lowered_module_args = [value.lower() for value in module_args]
            if lowered_module_args and lowered_module_args[0] == "check":
                return ValidationKind.STATIC
            if lowered_module_args and lowered_module_args[0] == "format" and "--check" in lowered_module_args[1:]:
                return ValidationKind.STATIC
            return None
        if module == "build":
            return ValidationKind.BUILD
        return None

    if executable in {"pytest", "py.test", "tox", "nox", "ctest"}:
        if executable in {"pytest", "py.test"} and any(re.fullmatch(r"-V+", value) for value in raw_args):
            return None
        if executable in {"tox", "nox"} and any(
            value == "-l" or (executable == "tox" and re.fullmatch(r"-[a-z]*a[a-z]*", value)) for value in args
        ):
            return None
        if executable == "tox" and any(value in {"config", "devenv", "exec", "list", "quickstart"} for value in args):
            return None
        if executable == "ctest" and any(value in {"-n", "-n=on"} for value in args):
            return None
        if executable == "ctest" and any(value in {"-d", "-s", "-sp", "-t"} for value in args):
            return None
        return ValidationKind.TEST
    if executable in {"mypy", "pyright", "eslint", "tsc", "stylelint"}:
        if (executable == "mypy" and "-V" in raw_args) or (
            executable in {"eslint", "stylelint", "tsc"} and "-v" in args
        ):
            return None
        return ValidationKind.STATIC
    if executable == "ruff":
        if args and args[0] == "check":
            return ValidationKind.STATIC
        if args and args[0] == "format" and "--check" in args[1:]:
            return ValidationKind.STATIC
        return None
    if executable == "biome":
        return ValidationKind.STATIC if args and args[0] == "check" else None
    if executable in {"node", "deno"}:
        if any(value in {"-e", "--eval", "-p", "--print"} for value in args):
            return None
        command = "--check" if executable == "node" else "check"
        return ValidationKind.STATIC if len(args) >= 2 and args[0] == command and not args[1].startswith("-") else None

    if executable in {"npm", "pnpm", "yarn", "bun"}:
        if not args or args[0].startswith("-"):
            return None
        if args[0] == "run" and len(args) > 1 and not args[1].startswith("-"):
            script = args[1]
        else:
            script = args[0]
        if script == "test" or script.startswith("test:"):
            return ValidationKind.TEST
        if script in {"check", "lint", "typecheck", "type-check"} or script.startswith("lint:"):
            return ValidationKind.STATIC
        if script == "build" or script.startswith("build:"):
            return ValidationKind.BUILD
        return None

    if executable == "cargo" and args:
        if "--no-run" in args or "--build-plan" in args:
            return None
        return {
            "test": ValidationKind.TEST,
            "check": ValidationKind.STATIC,
            "clippy": ValidationKind.STATIC,
            "build": ValidationKind.BUILD,
        }.get(args[0])
    if executable == "go" and args:
        if args[0] == "build" and "-n" in args[1:]:
            return None
        if args[0] == "test" and any(value == "-list" or value.startswith("-list=") for value in args[1:]):
            return None
        if args[0] == "test" and any(value in {"-c", "-exec"} or value.startswith("-exec=") for value in args[1:]):
            return None
        return {
            "test": ValidationKind.TEST,
            "vet": ValidationKind.STATIC,
            "build": ValidationKind.BUILD,
        }.get(args[0])
    if executable in {"dotnet", "swift"} and args:
        if executable == "dotnet" and args[0] == "test" and any(value in {"-t", "--list-tests"} for value in args[1:]):
            return None
        if executable == "swift" and args[0] == "test" and any(value in {"-l", "--list-tests"} for value in args[1:]):
            return None
        if executable == "swift" and args[:2] == ["test", "list"]:
            return None
        return {
            "test": ValidationKind.TEST,
            "build": ValidationKind.BUILD,
        }.get(args[0])
    if executable in {"mvn", "gradle"}:
        if executable == "gradle" and "-m" in args:
            return None
        value_options = {
            "--file",
            "--projects",
            "--settings",
            "--threads",
            "-f",
            "-p",
            "-s",
            "-t",
        }
        if any(value in value_options for value in args):
            return None
        tasks = {value.lstrip("./") for value in args if not value.startswith("-")}
        if tasks & {"test", "verify", "check"}:
            return ValidationKind.TEST
        if tasks & {"build", "package", "assemble"}:
            return ValidationKind.BUILD
        return None
    if executable == "make":
        if any(
            value in {"--just-print", "--question", "--touch"} or bool(re.fullmatch(r"-[A-Za-z]*[nqt][A-Za-z]*", value))
            for value in args
        ):
            return None
        targets = {value for value in args if not value.startswith("-") and "=" not in value}
        if any(target == "test" or target.startswith("test-") for target in targets):
            return ValidationKind.TEST
        if targets & {"check", "lint", "typecheck", "type-check"}:
            return ValidationKind.STATIC
        if targets & {"build", "all"}:
            return ValidationKind.BUILD
        return None
    if executable == "git" and args == ["diff", "--check"]:
        return ValidationKind.DIFF
    return None


@dataclass(frozen=True)
class RevisionEntry:
    path_sha256: str
    suffix: str
    state_sha256: str


@dataclass(frozen=True)
class WorkspaceSnapshot:
    revision_sha256: str
    entries: tuple[RevisionEntry, ...]
    complete: bool
    root_count: int
    method: str
    error_codes: tuple[str, ...] = ()

    def entry_map(self) -> dict[str, RevisionEntry]:
        return {entry.path_sha256: entry for entry in self.entries}


def _hash_file(path: Path, *, root: Path, byte_budget: list[int]) -> str:
    """Hash one path through an openat no-follow walk rooted in ``root``."""

    digest = hashlib.sha256()
    relative = path.relative_to(root)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, directory_flags)
    directory_fd = root_fd
    descriptor = -1
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        before_read = os.fstat(descriptor)
        if not stat.S_ISREG(before_read.st_mode):
            raise OSError("manifest_non_regular_file")
        if before_read.st_nlink > 1:
            raise OSError("manifest_hardlink_file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_budget[0] += len(chunk)
            if byte_budget[0] > _MAX_MANIFEST_BYTES:
                raise OverflowError("manifest_byte_limit")
            digest.update(chunk)
        after_read = os.fstat(descriptor)
        if (
            before_read.st_dev,
            before_read.st_ino,
            before_read.st_size,
            before_read.st_mtime_ns,
            before_read.st_mode,
            before_read.st_nlink,
        ) != (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            after_read.st_mtime_ns,
            after_read.st_mode,
            after_read.st_nlink,
        ):
            raise OSError("manifest_file_changed_during_read")
        digest.update(f"\0mode:{before_read.st_mode & 0o777}".encode())
        return digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd != root_fd:
            os.close(directory_fd)
        os.close(root_fd)


def _safe_suffix(path: Path) -> str:
    if path.name.lower() in _SOURCE_NAMES:
        return ".source"
    suffix = path.suffix.lower()
    if suffix and re.fullmatch(r"\.[a-z0-9_+-]{1,16}", suffix):
        return suffix
    return ""


def _entry(
    root_index: int,
    relative: str,
    absolute: Path,
    byte_budget: list[int],
    *,
    root: Path,
) -> RevisionEntry:
    normalized = f"{root_index}:{relative}"
    path_digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
    if not absolute.exists() and not absolute.is_symlink():
        state_digest = hashlib.sha256(b"deleted").hexdigest()
    else:
        state_digest = _hash_file(absolute, root=root, byte_budget=byte_budget)
    return RevisionEntry(
        path_sha256=path_digest,
        suffix=_safe_suffix(Path(relative)),
        state_sha256=state_digest,
    )


def _prepare_git_probe(root: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    """Build one read-only, no-child-process Git probe for ``root``.

    Repository-local Git configuration is attacker-controlled.  In particular,
    ``core.fsmonitor`` and external diff/filter helpers can otherwise execute
    during an apparently read-only status probe.  The inherited macOS sandbox
    prevents writes, network access, and child processes; command-line config
    and a minimal environment disable the corresponding Git extension points.
    If that sandbox is unavailable, callers fall back to the subprocess-free
    bounded manifest snapshot.
    """

    try:
        policy = AgentToolPolicy.read_only(root)
        wrapped, environment = sandboxed_command(
            ["git"],
            policy,
            allow_process_fork=False,
        )
    except (AgentPathViolation, AgentPermissionDenied, ValueError) as exc:
        raise OSError("git_sandbox_unavailable") from exc
    sanitized = dict(environment)
    sanitized.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
            "PAGER": "cat",
        }
    )
    sanitized.pop("GIT_EXTERNAL_DIFF", None)
    return tuple(wrapped), sanitized


def _run_git(
    root: Path,
    *args: str,
    probe: tuple[tuple[str, ...], dict[str, str]] | None = None,
) -> bytes:
    wrapped, environment = probe or _prepare_git_probe(root)
    completed = subprocess.run(
        [
            *wrapped,
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "diff.external=",
            "-c",
            "submodule.recurse=false",
            "-C",
            str(root),
            *args,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
        timeout=_GIT_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError("git_command_failed")
    if len(completed.stdout) > _MAX_GIT_OUTPUT_BYTES:
        raise OverflowError("git_output_limit")
    return completed.stdout


def _git_entries(root: Path, root_index: int, byte_budget: list[int]) -> tuple[list[RevisionEntry], bytes]:
    probe = _prepare_git_probe(root)
    inside = _run_git(root, "rev-parse", "--is-inside-work-tree", probe=probe).strip()
    if inside != b"true":
        raise OSError("not_git_worktree")
    try:
        head = _run_git(root, "rev-parse", "--verify", "HEAD", probe=probe).strip()
    except OSError:
        head = b"unborn"
    # Hash every tracked worktree file. ``git diff`` and status deliberately
    # trust index hints such as assume-unchanged/skip-worktree, so using only
    # their reported dirty paths would let repository state hide a mutation.
    # ``ls-files --cached`` enumerates those paths even when the hints are set;
    # duplicate modified/deleted entries are collapsed below.
    tracked = _run_git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--modified",
        "--deleted",
        "--",
        ".",
        probe=probe,
    )
    untracked = _run_git(
        root,
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
        "--",
        ".",
        probe=probe,
    )
    ignored = _run_git(
        root,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        ".",
        probe=probe,
    )
    raw_paths = set((tracked + untracked).split(b"\0"))
    for raw_path in ignored.split(b"\0"):
        if not raw_path:
            continue
        decoded = os.fsdecode(raw_path)
        candidate = Path(decoded)
        if not any(part in _SKIP_DIRS for part in candidate.parts):
            raw_paths.add(raw_path)
    raw_paths.discard(b"")
    if len(raw_paths) > _MAX_MANIFEST_FILES:
        raise OverflowError("manifest_file_limit")

    entries: list[RevisionEntry] = []
    root_resolved = root.resolve()
    for raw_path in sorted(raw_paths):
        relative = os.fsdecode(raw_path)
        absolute = root / relative
        try:
            absolute.resolve(strict=False).relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise OSError("git_path_escape") from exc
        if absolute.is_symlink():
            raise OSError("git_symlink_path")
        entries.append(_entry(root_index, relative, absolute, byte_budget, root=root_resolved))
    metadata = hashlib.sha256(head).digest()
    return entries, metadata


def _fallback_entries(root: Path, root_index: int, byte_budget: list[int]) -> list[RevisionEntry]:
    entries: list[RevisionEntry] = []
    root_resolved = root.resolve()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirnames):
            raise OSError("manifest_symlink_directory")
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
        for filename in sorted(filenames):
            absolute = Path(directory) / filename
            if absolute.is_symlink():
                raise OSError("manifest_symlink_file")
            relative = absolute.relative_to(root_resolved).as_posix()
            entries.append(_entry(root_index, relative, absolute, byte_budget, root=root_resolved))
            if len(entries) > _MAX_MANIFEST_FILES:
                raise OverflowError("manifest_file_limit")
    return entries


def snapshot_workspaces(roots: Iterable[str | os.PathLike[str]]) -> WorkspaceSnapshot:
    """Create a bounded, content-free revision fingerprint for allowed roots."""

    normalized: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"workspace root is not a directory: {raw_root}")
        if root not in normalized:
            normalized.append(root)
    if not normalized:
        raise ValueError("at least one workspace root is required")

    entries: list[RevisionEntry] = []
    metadata: list[bytes] = []
    error_codes: list[str] = []
    methods: list[str] = []
    byte_budget = [0]
    complete = True
    for root_index, root in enumerate(normalized):
        try:
            root_entries, root_metadata = _git_entries(root, root_index, byte_budget)
            methods.append("git")
            metadata.append(root_metadata)
            entries.extend(root_entries)
        except (OSError, subprocess.TimeoutExpired):
            try:
                entries.extend(_fallback_entries(root, root_index, byte_budget))
                methods.append("manifest")
                metadata.append(b"manifest")
            except (OSError, OverflowError):
                complete = False
                methods.append("incomplete")
                metadata.append(b"incomplete")
                error_codes.append("snapshot_incomplete")
        except OverflowError:
            complete = False
            methods.append("incomplete")
            metadata.append(b"incomplete")
            error_codes.append("snapshot_limit")

    ordered = tuple(sorted(entries, key=lambda item: item.path_sha256))
    digest = hashlib.sha256()
    for item in ordered:
        digest.update(item.path_sha256.encode())
        digest.update(item.suffix.encode())
        digest.update(item.state_sha256.encode())
    for item in metadata:
        digest.update(item)
    digest.update(str(len(normalized)).encode())
    digest.update(b"complete" if complete else b"incomplete")
    return WorkspaceSnapshot(
        revision_sha256=digest.hexdigest(),
        entries=ordered,
        complete=complete,
        root_count=len(normalized),
        method="+".join(methods),
        error_codes=tuple(sorted(set(error_codes))),
    )


def _snapshot_delta(
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
) -> tuple[bool, frozenset[str]]:
    if before.revision_sha256 == after.revision_sha256:
        return False, frozenset()
    before_entries = before.entry_map()
    after_entries = after.entry_map()
    suffixes: set[str] = set()
    for path_digest in before_entries.keys() | after_entries.keys():
        left = before_entries.get(path_digest)
        right = after_entries.get(path_digest)
        if left != right:
            suffixes.add((right or left).suffix if (right or left) is not None else "")
    return True, frozenset(suffixes)


@dataclass(frozen=True)
class ValidationEvidence:
    kind: ValidationKind
    epoch: int
    revision_sha256: str
    command_sha256: str
    allowed: bool
    outcome: str


@dataclass(frozen=True)
class GateDecision:
    status: GateStatus
    activated: bool
    satisfied: bool
    phase: str
    required: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass
class CodingQualityGate:
    roots: tuple[Path, ...]
    effort: CodingEffort = CodingEffort.HIGH
    enabled: bool = True
    intent: RequestIntent = RequestIntent.GENERAL
    request_sha256: str = field(default_factory=lambda: hashlib.sha256(b"").hexdigest())
    initial_snapshot: WorkspaceSnapshot | None = None
    current_snapshot: WorkspaceSnapshot | None = None
    mutation_epoch: int = 0
    changed_kinds: set[str] = field(default_factory=set)
    validations: list[ValidationEvidence] = field(default_factory=list)
    successful_reads: int = 0
    snapshot_failed_closed: bool = False

    def __post_init__(self) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in self.roots)
        self.effort = CodingEffort(self.effort)
        self.intent = RequestIntent(self.intent)
        if self.initial_snapshot is None:
            self.initial_snapshot = snapshot_workspaces(self.roots)
        if self.current_snapshot is None:
            self.current_snapshot = self.initial_snapshot
        if not self.initial_snapshot.complete:
            self.snapshot_failed_closed = True

    @classmethod
    def start(
        cls,
        roots: Iterable[str | os.PathLike[str]],
        request: str = "",
        *,
        effort: CodingEffort | str = CodingEffort.HIGH,
        enabled: bool = True,
    ) -> CodingQualityGate:
        normalized = tuple(Path(root).expanduser().resolve() for root in roots)
        return cls(
            roots=normalized,
            effort=CodingEffort(effort),
            enabled=bool(enabled),
            intent=classify_request_intent(request),
            request_sha256=hashlib.sha256(request.encode("utf-8", errors="replace")).hexdigest(),
        )

    @classmethod
    def from_request(
        cls,
        roots: Iterable[str | os.PathLike[str]],
        request: str,
        *,
        effort: CodingEffort | str = CodingEffort.HIGH,
        enabled: bool = True,
    ) -> CodingQualityGate:
        return cls.start(roots, request, effort=effort, enabled=enabled)

    @property
    def activated(self) -> bool:
        # The preregistered intervention begins only after an observed
        # workspace mutation. Intent classification remains advisory metadata;
        # it can never force a needless edit or create an unresolvable pending
        # obligation for an inspection-only response.
        return self.mutation_epoch > 0

    def _adopt_snapshot(
        self,
        fresh: WorkspaceSnapshot,
        *,
        conservative: bool,
    ) -> WorkspaceSnapshot:
        previous = self.current_snapshot
        recovered_from_gap = previous is not None and not previous.complete and fresh.complete
        if previous is not None:
            changed, suffixes = _snapshot_delta(previous, fresh)
            if changed or recovered_from_gap:
                self._record_mutation(
                    self._classify_suffixes(
                        suffixes,
                        conservative=conservative or recovered_from_gap,
                    )
                )
        self.current_snapshot = fresh
        self.snapshot_failed_closed = not fresh.complete
        return fresh

    def refresh(self) -> WorkspaceSnapshot:
        """Reconcile late/background mutations before a terminal decision."""

        return self._adopt_snapshot(
            snapshot_workspaces(self.roots),
            conservative=True,
        )

    def before_tool(self, name: str, args: dict | None = None) -> WorkspaceSnapshot:
        del args
        if name not in {"bash", "call_mcp_tool", "validate", "write", "edit"}:
            if self.current_snapshot is None:
                self.current_snapshot = snapshot_workspaces(self.roots)
            return self.current_snapshot
        return self._adopt_snapshot(
            snapshot_workspaces(self.roots),
            conservative=True,
        )

    def _classify_suffixes(self, suffixes: Iterable[str], *, conservative: bool) -> set[str]:
        if conservative:
            return {"code"}
        materialized = set(suffixes)
        if materialized and all(suffix in _DOC_SUFFIXES for suffix in materialized):
            return {"docs"}
        return {"code"}

    def _record_mutation(self, kinds: Iterable[str]) -> None:
        self.mutation_epoch += 1
        self.changed_kinds.update(kinds or {"code"})

    def record_validation(
        self,
        kind: ValidationKind | str,
        *,
        argv: Sequence[str] | None = None,
        command_sha256: str | None = None,
        allowed: bool,
        outcome: str,
        snapshot: WorkspaceSnapshot | None = None,
    ) -> None:
        active_snapshot = snapshot or self.current_snapshot
        if active_snapshot is None:
            active_snapshot = snapshot_workspaces(self.roots)
            self.current_snapshot = active_snapshot
        if command_sha256 is None:
            command_sha256 = _semantic_argv_digest(tuple(argv or ()))
        self.validations.append(
            ValidationEvidence(
                kind=ValidationKind(kind),
                epoch=self.mutation_epoch,
                revision_sha256=active_snapshot.revision_sha256,
                command_sha256=command_sha256,
                allowed=bool(allowed),
                outcome=str(outcome)[:64],
            )
        )

    def record_audit_event(
        self,
        event: AgentAuditEvent,
        *,
        tool_name: str | None = None,
        args: dict | None = None,
        snapshot: WorkspaceSnapshot | None = None,
    ) -> bool:
        """Record one event; return whether it created a mutation epoch."""

        operation = event.operation
        if operation == "read" and event.allowed and event.outcome == "ok":
            self.successful_reads += 1
        if operation in {"write", "edit"} and event.allowed and event.outcome == "ok":
            suffix = _safe_suffix(Path(event.target))
            self._record_mutation(self._classify_suffixes({suffix}, conservative=False))
            if snapshot is not None:
                self.current_snapshot = snapshot
            return True
        if operation == "validate":
            raw_argv = (args or {}).get("argv", ())
            kind = infer_validation_kind(raw_argv) if isinstance(raw_argv, (list, tuple)) else None
            if kind is not None:
                self.record_validation(
                    kind,
                    argv=raw_argv,
                    allowed=event.allowed,
                    outcome=event.outcome,
                    snapshot=snapshot,
                )
        return False

    def after_tool(
        self,
        name: str,
        args: dict | None = None,
        *,
        before: WorkspaceSnapshot | None = None,
        audit_events: Sequence[AgentAuditEvent] = (),
    ) -> WorkspaceSnapshot:
        before_snapshot = before or self.current_snapshot or snapshot_workspaces(self.roots)
        after_snapshot = snapshot_workspaces(self.roots)
        unsafe = name in {"bash", "call_mcp_tool", "validate", "write", "edit"}
        changed, suffixes = _snapshot_delta(before_snapshot, after_snapshot)
        direct_mutation = False
        for event in audit_events:
            direct_mutation = (
                self.record_audit_event(
                    event,
                    tool_name=name,
                    args=args,
                    snapshot=after_snapshot,
                )
                or direct_mutation
            )
        if changed and not direct_mutation:
            self._record_mutation(
                self._classify_suffixes(
                    suffixes,
                    conservative=name in {"bash", "call_mcp_tool", "validate"},
                )
            )
        if not before_snapshot.complete and after_snapshot.complete and not direct_mutation:
            # A complete snapshot after an unobservable interval recovers
            # liveness, but the unknown interval is a new conservative code
            # epoch and therefore invalidates every earlier certificate.
            self._record_mutation({"code"})
        elif (
            unsafe
            and name in {"bash", "call_mcp_tool", "validate"}
            and (not before_snapshot.complete or not after_snapshot.complete)
            and not direct_mutation
        ):
            # The tool may mutate paths hidden by an incomplete snapshot. Once
            # an unsafe capability ran, fail closed with a conservative epoch
            # instead of treating the turn as observation-only.
            self._record_mutation({"code"})
        if unsafe:
            self.snapshot_failed_closed = not after_snapshot.complete
        self.current_snapshot = after_snapshot
        return after_snapshot

    def _current_successes(self) -> list[ValidationEvidence]:
        if self.current_snapshot is None:
            return []
        revision = self.current_snapshot.revision_sha256
        return [
            evidence
            for evidence in self.validations
            if evidence.epoch == self.mutation_epoch
            and evidence.revision_sha256 == revision
            and evidence.allowed
            and evidence.outcome == "ok"
        ]

    def _requirements(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        successes = self._current_successes()
        kinds = {item.kind for item in successes}
        docs_only = self.changed_kinds == {"docs"}
        required: list[str] = []
        missing: list[str] = []

        if self.effort is CodingEffort.LOW:
            required.append("any_validation")
            if not successes:
                missing.append("any_validation")
        elif self.effort is CodingEffort.MEDIUM:
            if docs_only:
                required.append("diff")
                if ValidationKind.DIFF not in kinds:
                    missing.append("diff")
            else:
                required.append("test_or_build")
                if not kinds & {ValidationKind.TEST, ValidationKind.BUILD}:
                    missing.append("test_or_build")
        elif self.effort is CodingEffort.HIGH:
            required.extend(("test", "static_or_diff"))
            if ValidationKind.TEST not in kinds:
                missing.append("test")
            if not kinds & {ValidationKind.STATIC, ValidationKind.DIFF}:
                missing.append("static_or_diff")
        else:
            required.extend(("test", "static", "diff"))
            for kind in (ValidationKind.TEST, ValidationKind.STATIC, ValidationKind.DIFF):
                if kind not in kinds:
                    missing.append(kind.value)
            if self.effort is CodingEffort.ULTRA:
                required.append("review_or_second_distinct_test")
                distinct_tests = {item.command_sha256 for item in successes if item.kind is ValidationKind.TEST}
                if ValidationKind.REVIEW not in kinds and len(distinct_tests) < 2:
                    missing.append("review_or_second_distinct_test")
        return tuple(required), tuple(missing)

    def decision(self) -> GateDecision:
        if not self.enabled:
            return GateDecision(
                status=GateStatus.NOT_APPLICABLE,
                activated=False,
                satisfied=True,
                phase="disabled",
                required=(),
                missing=(),
            )
        if not self.activated:
            return GateDecision(
                status=GateStatus.NOT_APPLICABLE,
                activated=False,
                satisfied=True,
                phase="observing",
                required=(),
                missing=(),
            )
        required, missing = self._requirements()
        if self.snapshot_failed_closed:
            missing = tuple(dict.fromkeys((*missing, "complete_workspace_snapshot")))
        if missing:
            failed = any(
                evidence.epoch == self.mutation_epoch and evidence.outcome != "ok" for evidence in self.validations
            )
            return GateDecision(
                status=GateStatus.INCOMPLETE,
                activated=True,
                satisfied=False,
                phase="validation_failed" if failed else "dirty",
                required=required,
                missing=missing,
            )
        return GateDecision(
            status=GateStatus.PASS,
            activated=True,
            satisfied=True,
            phase="passed",
            required=required,
            missing=(),
        )

    def feedback(self) -> str:
        verdict = self.decision()
        missing = ", ".join(verdict.missing) or "current-revision validation"
        return (
            "Coding-quality gate incomplete. Do not claim completion. "
            f"Missing: {missing}. Inspect or repair the current workspace, then use the "
            "validate tool with direct argv. Bash output and assistant prose do not count."
        )

    def system_instructions(self) -> str:
        return (
            "\nMANDATORY CODING-QUALITY GATE: after every workspace mutation, use the "
            "dedicated validate tool with direct argv. Evidence is bound to the latest "
            f"revision and effort={self.effort.value}; a later edit invalidates it. "
            "Ordinary bash output cannot satisfy the gate."
        )

    def report(self) -> dict[str, object]:
        verdict = self.decision()
        current = self.current_snapshot or self.initial_snapshot
        successes = self._current_successes()
        validation_counts = {kind.value: sum(item.kind is kind for item in successes) for kind in ValidationKind}
        return {
            "schema": "mio.coding-quality-gate.v1",
            "enabled": self.enabled,
            "effort": self.effort.value,
            "intent": self.intent.value,
            "request_sha256": self.request_sha256,
            "decision": verdict.status.value,
            "phase": verdict.phase,
            "activated": verdict.activated,
            "satisfied": verdict.satisfied,
            "mutation_epoch": self.mutation_epoch,
            "changed_kinds": sorted(self.changed_kinds),
            "snapshot_complete": not self.snapshot_failed_closed,
            "initial_revision_sha256": (self.initial_snapshot.revision_sha256 if self.initial_snapshot else ""),
            "current_revision_sha256": current.revision_sha256 if current else "",
            "required": list(verdict.required),
            "missing": list(verdict.missing),
            "validation_counts": validation_counts,
            "validation_attempts": len(self.validations),
            "successful_reads": self.successful_reads,
        }
