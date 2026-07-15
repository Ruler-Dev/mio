"""Managed, local skill catalog for Mio.

The catalog deliberately separates *instructions* from executable tools.  A
``SKILL.md`` is always discoverable and readable, even when it has no runner.
Code execution is a second capability which requires both an explicit
``script`` field in the frontmatter and two independent opt-ins at runtime.

Installed third-party skills live below ``$MIO_HOME/skills`` (``~/.mio`` by
default).  They are not Codex/Claude plugins and never depend on either
application's home directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from mio.paths import mio_home


MANIFEST_NAME = ".mio-skills.json"
MANIFEST_SCHEMA = 1
MAX_SKILL_MD_BYTES = 2 * 1024 * 1024
MAX_READ_CHARS = 200_000
MAX_SCRIPT_INPUT_BYTES = 64 * 1024
MAX_SCRIPT_OUTPUT_BYTES = 1024 * 1024
# Managed Agent Skills use kebab-case, but Mio's original user-skill loader
# accepted Python-style snake_case.  Keep that local compatibility while
# still rejecting path separators, dots, whitespace, and traversal tokens.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<yaml>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)


class SkillCatalogError(RuntimeError):
    """Base error for catalog, installation, and policy failures."""


class SkillValidationError(SkillCatalogError):
    """A skill does not satisfy Mio's on-disk contract."""


class SkillNotFoundError(SkillCatalogError):
    """No installed skill has the requested name or alias."""


class SkillExecutionDisabled(SkillCatalogError):
    """Execution was requested without every required opt-in."""


@dataclass(frozen=True)
class SkillSource:
    """Immutable source selection used by the bundled installer."""

    source_id: str
    repository: str
    revision: str
    include_prefixes: tuple[str, ...]
    exclude_prefixes: tuple[str, ...] = ()
    alias_prefix: str = "external"
    expected_skills: int | None = None


# Revisions are intentionally full commit hashes.  Updating the bundle is a
# reviewable source change; an installation never silently follows a branch.
PINNED_SOURCES: tuple[SkillSource, ...] = (
    SkillSource(
        source_id="hallmark",
        repository="https://github.com/Nutlope/hallmark.git",
        revision="aeb42fb354ff4efa36ab475773a082315a3af2ce",
        include_prefixes=("skills/",),
        alias_prefix="hallmark",
        expected_skills=1,
    ),
    SkillSource(
        source_id="mattpocock-skills",
        repository="https://github.com/mattpocock/skills.git",
        revision="e9fcdf95b402d360f90f1db8d776d5dd450f9234",
        include_prefixes=("skills/",),
        exclude_prefixes=("skills/deprecated/", "skills/in-progress/", "skills/personal/"),
        alias_prefix="matt",
        expected_skills=26,
    ),
    SkillSource(
        source_id="anthropic-cybersecurity-skills",
        repository="https://github.com/Ruler-Dev/Anthropic-Cybersecurity-Skills.git",
        revision="673da1f3b0b7be34ffc9624ef3858fe45f1c3bed",
        include_prefixes=("skills/",),
        alias_prefix="cyber",
        expected_skills=817,
    ),
    SkillSource(
        source_id="claude-code-game-studios",
        repository="https://github.com/Ruler-Dev/Claude-Code-Game-Studios.git",
        revision="666e0fcb5ad3f5f0f56e1219e8cf03d44e62a49a",
        include_prefixes=(".claude/skills/",),
        alias_prefix="game",
        expected_skills=72,
    ),
)


@dataclass(frozen=True)
class SkillMetadata:
    canonical_name: str
    description: str
    tags: tuple[str, ...]
    script: str | None
    raw: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class SkillRecord:
    installed_name: str
    canonical_name: str
    description: str
    tags: tuple[str, ...]
    source_id: str = "local"
    source_url: str = ""
    source_revision: str = ""
    source_path: str = ""
    digest: str = ""
    script: str | None = None
    execution_enabled: bool = False

    @property
    def kind(self) -> str:
        return "executable" if self.script else "instruction"

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["kind"] = self.kind
        return data


@dataclass(frozen=True)
class SkillDocument:
    record: SkillRecord
    content: str
    truncated: bool


@dataclass(frozen=True)
class InstallReport:
    destination: str
    installed: int
    preserved: int
    aliases: Mapping[str, str]
    sources: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "installed": self.installed,
            "preserved": self.preserved,
            "aliases": dict(self.aliases),
            "sources": dict(self.sources),
        }


def skills_root(home: str | os.PathLike[str] | None = None) -> Path:
    return mio_home(home) / "skills"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _normalise_tags(meta: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    tags = meta.get("tags", ())
    if isinstance(tags, str):
        values.extend(part.strip() for part in tags.split(","))
    elif isinstance(tags, Sequence) and not isinstance(tags, (bytes, bytearray)):
        values.extend(str(part).strip() for part in tags)
    for key in ("domain", "subdomain"):
        value = meta.get(key)
        if isinstance(value, str):
            values.append(value.strip())
    return tuple(sorted({tag.casefold() for tag in values if tag}))


def parse_skill(skill_path: Path) -> tuple[SkillMetadata, str]:
    """Validate and parse a directory or its ``SKILL.md`` file."""

    md_path = skill_path / "SKILL.md" if skill_path.is_dir() else skill_path
    try:
        if md_path.is_symlink() or not md_path.is_file():
            raise SkillValidationError(f"missing regular SKILL.md: {md_path}")
        size = md_path.stat().st_size
        if size > MAX_SKILL_MD_BYTES:
            raise SkillValidationError(f"SKILL.md exceeds {MAX_SKILL_MD_BYTES} bytes: {md_path}")
        raw = md_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillValidationError(f"SKILL.md is not UTF-8: {md_path}") from exc
    except OSError as exc:
        if isinstance(exc, SkillValidationError):
            raise
        raise SkillValidationError(f"cannot read {md_path}: {exc}") from exc

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise SkillValidationError(f"missing YAML frontmatter: {md_path}")
    try:
        meta = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as exc:
        raise SkillValidationError(f"invalid YAML frontmatter in {md_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillValidationError(f"frontmatter must be a mapping: {md_path}")

    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise SkillValidationError(f"invalid skill name {name!r} in {md_path}")
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError(f"missing skill description in {md_path}")
    if not raw[match.end() :].strip():
        raise SkillValidationError(f"skill instructions are empty: {md_path}")

    script: str | None = None
    declared_script = meta.get("script")
    if declared_script is not None:
        if not isinstance(declared_script, str) or not declared_script.strip():
            raise SkillValidationError(f"script must be a relative Python path: {md_path}")
        script_path = PurePosixPath(declared_script)
        if (
            "\\" in declared_script
            or script_path.is_absolute()
            or ".." in script_path.parts
            or script_path.suffix != ".py"
        ):
            raise SkillValidationError(f"unsafe script path {declared_script!r} in {md_path}")
        base = md_path.parent.resolve()
        lexical_candidate = md_path.parent / Path(*script_path.parts)
        candidate = lexical_candidate.resolve()
        if (
            lexical_candidate.is_symlink()
            or not _is_relative_to(candidate, base)
            or not candidate.is_file()
        ):
            raise SkillValidationError(f"declared script is missing or escapes its skill: {declared_script!r}")
        script = script_path.as_posix()

    return (
        SkillMetadata(
            canonical_name=name,
            description=description.strip(),
            tags=_normalise_tags(meta),
            script=script,
            raw=meta,
        ),
        raw,
    )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise SkillValidationError(f"symlinks are not allowed in managed skills: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_managed_tree(root: Path) -> None:
    file_count = 0
    total_size = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SkillValidationError(f"symlinks are not allowed in managed skills: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SkillValidationError(f"unsupported file type in skill: {path}")
        file_count += 1
        size = path.stat().st_size
        total_size += size
        if size > 16 * 1024 * 1024:
            raise SkillValidationError(f"skill file exceeds 16 MiB: {path}")
        if file_count > 4096 or total_size > 128 * 1024 * 1024:
            raise SkillValidationError(f"skill tree is unreasonably large: {root}")


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _read_manifest(root: Path, *, strict: bool = False) -> dict[str, Any]:
    path = _manifest_path(root)
    if not path.is_file():
        return {"schema_version": MANIFEST_SCHEMA, "sources": [], "skills": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("unsupported schema")
        if not isinstance(data.get("sources", []), list) or not isinstance(data.get("skills", []), list):
            raise ValueError("sources/skills must be lists")
        return data
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if strict:
            raise SkillCatalogError(f"invalid skill manifest {path}: {exc}") from exc
        return {"schema_version": MANIFEST_SCHEMA, "sources": [], "skills": []}


def _atomic_json_write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _catalog_lock(root: Path, timeout_s: float = 30.0) -> Iterator[None]:
    """Advisory process lock kept outside the replaceable skill tree."""

    import fcntl

    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = root.parent / f".{root.name}.lock"
    with lock_path.open("a+") as handle:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SkillCatalogError(f"timed out waiting for {lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class MioSkillCatalog:
    """Discover, search, and read Mio-local skills."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root).expanduser().absolute() if root is not None else skills_root()
        self.diagnostics: list[str] = []

    def records(self) -> list[SkillRecord]:
        self.diagnostics = []
        manifest = _read_manifest(self.root)
        manifest_records = {
            item.get("installed_name"): item
            for item in manifest.get("skills", [])
            if isinstance(item, dict) and isinstance(item.get("installed_name"), str)
        }
        records: list[SkillRecord] = []
        if not self.root.is_dir() or self.root.is_symlink():
            return records
        for skill_dir in sorted(self.root.iterdir(), key=lambda item: item.name):
            if skill_dir.name.startswith(".") or not skill_dir.is_dir() or skill_dir.is_symlink():
                continue
            try:
                metadata, _ = parse_skill(skill_dir)
                stored = manifest_records.get(skill_dir.name, {})
                enabled = bool(stored.get("execution_enabled", False)) and bool(metadata.script)
                digest = str(stored.get("digest", ""))
                # An enabled runner is fail-closed if any installed byte changed.
                if enabled and (not digest or _tree_digest(skill_dir) != digest):
                    enabled = False
                    self.diagnostics.append(f"execution disabled after local modification: {skill_dir.name}")
                records.append(
                    SkillRecord(
                        installed_name=skill_dir.name,
                        canonical_name=metadata.canonical_name,
                        description=metadata.description,
                        tags=metadata.tags,
                        source_id=str(stored.get("source_id", "local")),
                        source_url=str(stored.get("source_url", "")),
                        source_revision=str(stored.get("source_revision", "")),
                        source_path=str(stored.get("source_path", skill_dir.name)),
                        digest=digest,
                        script=metadata.script,
                        execution_enabled=enabled,
                    )
                )
            except SkillValidationError as exc:
                self.diagnostics.append(str(exc))
        return records

    def search(
        self,
        query: str = "",
        *,
        tag: str = "",
        source: str = "",
        limit: int | None = 50,
    ) -> list[SkillRecord]:
        terms = tuple(part.casefold() for part in query.split() if part)
        wanted_tag = tag.strip().casefold()
        wanted_source = source.strip().casefold()
        matches: list[SkillRecord] = []
        for record in self.records():
            haystack = " ".join(
                (record.installed_name, record.canonical_name, record.description, *record.tags)
            ).casefold()
            if terms and not all(term in haystack for term in terms):
                continue
            if wanted_tag and wanted_tag not in record.tags:
                continue
            if wanted_source and record.source_id.casefold() != wanted_source:
                continue
            matches.append(record)
        matches.sort(key=lambda item: (item.installed_name != query, item.installed_name))
        if limit is None:
            return matches
        return matches[: max(1, min(int(limit), 200))]

    def resolve(self, name: str) -> SkillRecord:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise SkillNotFoundError(f"invalid or unknown skill name: {name!r}")
        records = self.records()
        exact = [record for record in records if record.installed_name == name]
        if exact:
            return exact[0]
        canonical = [record for record in records if record.canonical_name == name]
        if len(canonical) == 1:
            return canonical[0]
        if len(canonical) > 1:
            aliases = ", ".join(record.installed_name for record in canonical)
            raise SkillNotFoundError(f"ambiguous canonical name {name!r}; use one of: {aliases}")
        raise SkillNotFoundError(f"skill not found: {name}")

    def read(self, name: str, *, max_chars: int = 32_000) -> SkillDocument:
        record = self.resolve(name)
        root = self.root.resolve()
        path = (self.root / record.installed_name / "SKILL.md").resolve()
        if not _is_relative_to(path, root) or path.is_symlink() or not path.is_file():
            raise SkillCatalogError("skill path escaped the Mio skill root")
        limit = max(1, min(int(max_chars), MAX_READ_CHARS))
        content = path.read_text(encoding="utf-8")
        truncated = len(content) > limit
        return SkillDocument(record=record, content=content[:limit], truncated=truncated)

    def set_execution_enabled(self, name: str, enabled: bool) -> SkillRecord:
        """Persist the first execution opt-in; callers still opt in per call."""

        with _catalog_lock(self.root):
            record = self.resolve(name)
            if enabled and not record.script:
                raise SkillExecutionDisabled(f"{record.installed_name} declares no executable script")
            skill_dir = self.root / record.installed_name
            digest = _tree_digest(skill_dir)
            manifest = _read_manifest(self.root, strict=True)
            items = [item for item in manifest.get("skills", []) if isinstance(item, dict)]
            replacement = SkillRecord(
                installed_name=record.installed_name,
                canonical_name=record.canonical_name,
                description=record.description,
                tags=record.tags,
                source_id=record.source_id,
                source_url=record.source_url,
                source_revision=record.source_revision,
                source_path=record.source_path,
                digest=digest,
                script=record.script,
                execution_enabled=bool(enabled),
            ).to_manifest()
            for index, item in enumerate(items):
                if item.get("installed_name") == record.installed_name:
                    items[index] = replacement
                    break
            else:
                items.append(replacement)
            manifest["skills"] = sorted(items, key=lambda item: str(item.get("installed_name", "")))
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json_write(_manifest_path(self.root), manifest)
        return self.resolve(record.installed_name)

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        allow_execution: bool = False,
        timeout_s: float = 15.0,
        max_output_bytes: int = 64 * 1024,
    ) -> dict[str, Any]:
        """Run an explicitly declared Python runner under bounded resources.

        This is not exposed as a default Mio tool.  It requires both a
        persisted per-skill policy flag and ``allow_execution=True`` at the
        call site.  The subprocess boundary limits time, output, files, and
        address space; it is not advertised as a network sandbox.
        """

        record = self.resolve(name)
        if not allow_execution or not record.execution_enabled:
            raise SkillExecutionDisabled(
                "skill execution needs catalog enablement and allow_execution=True"
            )
        if not record.script:
            raise SkillExecutionDisabled(f"{record.installed_name} declares no script")
        encoded = json.dumps(dict(arguments), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_SCRIPT_INPUT_BYTES:
            raise SkillCatalogError(f"skill input exceeds {MAX_SCRIPT_INPUT_BYTES} bytes")

        skill_dir = (self.root / record.installed_name).resolve()
        lexical_script = skill_dir / Path(*PurePosixPath(record.script).parts)
        script = lexical_script.resolve()
        if lexical_script.is_symlink() or not _is_relative_to(script, skill_dir) or not script.is_file():
            raise SkillCatalogError("declared runner escaped its skill directory")
        timeout = max(0.1, min(float(timeout_s), 30.0))
        output_limit = max(1024, min(int(max_output_bytes), MAX_SCRIPT_OUTPUT_BYTES))

        def _resource_limits() -> None:
            try:
                import resource

                cpu = max(1, min(int(timeout) + 1, 31))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit + 1, output_limit + 1))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                if hasattr(resource, "RLIMIT_AS"):
                    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
            except (ImportError, OSError, ValueError):
                pass

        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"}
        }
        environment["MIO_SKILL_NAME"] = record.installed_name
        with tempfile.TemporaryDirectory(prefix="mio-skill-") as temporary_dir:
            environment["MIO_SKILL_TMPDIR"] = temporary_dir
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-B", str(script)],
                    cwd=skill_dir,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    preexec_fn=_resource_limits if os.name == "posix" else None,
                )
                try:
                    process.communicate(input=encoded, timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    process.kill()
                    process.wait()
                    raise SkillCatalogError(f"skill timed out after {timeout:.1f}s") from exc
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read(output_limit + 1)
                stderr = stderr_file.read(output_limit + 1)

        out_truncated = len(stdout) > output_limit
        err_truncated = len(stderr) > output_limit
        out_text = stdout[:output_limit].decode("utf-8", errors="replace")
        err_text = stderr[:output_limit].decode("utf-8", errors="replace")
        try:
            result: Any = json.loads(out_text) if out_text else {}
        except json.JSONDecodeError:
            result = {"stdout": out_text}
        if not isinstance(result, dict):
            result = {"result": result}
        result.setdefault("returncode", process.returncode)
        if err_text:
            result.setdefault("stderr", err_text)
        if out_truncated or err_truncated:
            result["truncated"] = True
        return result


def _record_summary(record: SkillRecord) -> dict[str, Any]:
    return {
        "name": record.installed_name,
        "canonical_name": record.canonical_name,
        "description": record.description,
        "tags": list(record.tags),
        "source": record.source_id,
        "revision": record.source_revision,
        "kind": record.kind,
        "execution_enabled": record.execution_enabled,
    }


def list_mio_skills(
    query: str = "",
    tag: str = "",
    source: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Tool-friendly catalog search; never executes skill code."""

    catalog = MioSkillCatalog()
    all_matches = catalog.search(query, tag=tag, source=source, limit=None)
    bounded_limit = max(1, min(int(limit), 200))
    return {
        "skills": [_record_summary(record) for record in all_matches[:bounded_limit]],
        "matched": len(all_matches),
        "returned": min(len(all_matches), bounded_limit),
        "diagnostics": catalog.diagnostics[:10],
    }


def read_mio_skill(name: str, max_chars: int = 32_000) -> dict[str, Any]:
    """Tool-friendly, confined read of one skill's instructions."""

    try:
        document = MioSkillCatalog().read(name, max_chars=max_chars)
    except SkillCatalogError as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}
    return {
        "skill": _record_summary(document.record),
        "content": document.content,
        "truncated": document.truncated,
    }


def _source_skill_paths(checkout: Path, source: SkillSource) -> list[Path]:
    paths: list[Path] = []
    for md_path in checkout.rglob("SKILL.md"):
        relative = md_path.relative_to(checkout).as_posix()
        if not any(relative.startswith(prefix) for prefix in source.include_prefixes):
            continue
        if any(relative.startswith(prefix) for prefix in source.exclude_prefixes):
            continue
        paths.append(md_path.parent)
    return sorted(paths, key=lambda path: path.relative_to(checkout).as_posix())


def _short_alias(base: str, salt: str) -> str:
    if len(base) <= 64:
        return base
    suffix = hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8]
    return f"{base[:55].rstrip('-')}-{suffix}"


def _allocate_name(canonical: str, source: SkillSource, source_path: str, occupied: set[str]) -> str:
    if canonical not in occupied:
        return canonical
    base = _short_alias(f"{source.alias_prefix}-{canonical}", f"{source.source_id}:{source_path}")
    if base not in occupied:
        return base
    for attempt in range(1000):
        salt = f"{source.source_id}:{source_path}:{attempt}"
        suffix = hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8]
        candidate = _short_alias(f"{source.alias_prefix}-{canonical}-{suffix}", salt)
        if candidate not in occupied:
            return candidate
    raise SkillCatalogError(f"could not allocate a unique name for {source.source_id}:{source_path}")


def _copy_managed_skill(source: Path, destination: Path) -> None:
    _validate_managed_tree(source)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _replace_skill_tree(stage: Path, destination: Path) -> None:
    """Publish a complete snapshot with rollback; readers never see a partial tree."""

    def atomic_exchange(left: Path, right: Path) -> bool:
        # Mio's production platform is macOS. renamex_np(RENAME_SWAP) makes
        # replacement of two non-empty directories one atomic filesystem op.
        # Other platforms use the rollback path below.
        if sys.platform != "darwin":
            return False
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:
            return False
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        if renamex(os.fsencode(left), os.fsencode(right), 0x00000002) == 0:  # RENAME_SWAP
            return True
        error_number = ctypes.get_errno()
        if error_number in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS, errno.EXDEV}:
            return False
        raise OSError(error_number, os.strerror(error_number))

    backup = destination.parent / f".{destination.name}-backup-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if destination.is_symlink():
            raise SkillCatalogError(f"refusing to replace symlinked skill root: {destination}")
        if destination.exists() and not destination.is_dir():
            raise SkillCatalogError(f"skill root is not a directory: {destination}")
        if destination.exists():
            if atomic_exchange(stage, destination):
                shutil.rmtree(stage)
                return
            os.replace(destination, backup)
            moved_old = True
        os.replace(stage, destination)
    except Exception:
        if moved_old and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if moved_old:
            shutil.rmtree(backup)


def install_skill_sources_from_checkouts(
    checkouts: Mapping[str, Path],
    *,
    root: str | os.PathLike[str] | None = None,
    sources: Sequence[SkillSource] = PINNED_SOURCES,
    preserve_unmanaged: bool = True,
) -> InstallReport:
    """Build and publish a validated snapshot from already checked-out repos."""

    destination = Path(root).expanduser().absolute() if root is not None else skills_root()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    selected_ids = {source.source_id for source in sources}
    missing = selected_ids.difference(checkouts)
    if missing:
        raise SkillCatalogError(f"missing source checkouts: {', '.join(sorted(missing))}")

    with _catalog_lock(destination):
        old_manifest = _read_manifest(destination)
        old_records = [item for item in old_manifest.get("skills", []) if isinstance(item, dict)]
        old_by_name = {
            str(item.get("installed_name")): item
            for item in old_records
            if isinstance(item.get("installed_name"), str)
        }
        stage = destination.parent / f".{destination.name}-stage-{uuid.uuid4().hex}"
        stage.mkdir(mode=0o700)
        records: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        preserved = 0
        occupied: set[str] = set()
        aliases: dict[str, str] = {}
        source_counts: dict[str, int] = {}
        try:
            # Keep local/unmanaged directories and managed sources not selected
            # for this update.  --replace-all disables both behaviours.
            if preserve_unmanaged and destination.is_dir() and not destination.is_symlink():
                for child in sorted(destination.iterdir(), key=lambda item: item.name):
                    if child.name == MANIFEST_NAME:
                        continue
                    previous = old_by_name.get(child.name)
                    should_keep = previous is None or previous.get("source_id") not in selected_ids
                    if not should_keep:
                        continue
                    target = stage / child.name
                    if child.is_symlink():
                        target.symlink_to(os.readlink(child), target_is_directory=child.is_dir())
                    elif child.is_dir():
                        shutil.copytree(child, target, symlinks=True)
                    elif child.is_file():
                        shutil.copy2(child, target, follow_symlinks=False)
                    else:
                        raise SkillCatalogError(f"unsupported existing entry in skill root: {child}")
                    if _NAME_RE.fullmatch(child.name):
                        occupied.add(child.name)
                    preserved += 1
                    if previous is not None:
                        records.append(previous)
                kept_source_ids = {
                    str(item.get("source_id")) for item in records if item.get("source_id")
                }
                source_rows.extend(
                    row
                    for row in old_manifest.get("sources", [])
                    if isinstance(row, dict) and row.get("source_id") in kept_source_ids
                )

            for source in sources:
                checkout = Path(checkouts[source.source_id]).resolve()
                if not checkout.is_dir():
                    raise SkillCatalogError(f"source checkout is not a directory: {checkout}")
                source_skill_paths = _source_skill_paths(checkout, source)
                if source.expected_skills is not None and len(source_skill_paths) != source.expected_skills:
                    raise SkillValidationError(
                        f"{source.source_id} expected {source.expected_skills} skills at the pinned "
                        f"revision, found {len(source_skill_paths)}"
                    )
                seen_canonical: set[str] = set()
                installed_for_source = 0
                for skill_dir in source_skill_paths:
                    metadata, _ = parse_skill(skill_dir)
                    if metadata.canonical_name in seen_canonical:
                        raise SkillValidationError(
                            f"duplicate {metadata.canonical_name!r} in {source.source_id}"
                        )
                    seen_canonical.add(metadata.canonical_name)
                    source_path = skill_dir.relative_to(checkout).as_posix()
                    installed_name = _allocate_name(
                        metadata.canonical_name, source, source_path, occupied
                    )
                    if installed_name in occupied:
                        raise SkillCatalogError(f"could not allocate unique alias for {source_path}")
                    occupied.add(installed_name)
                    if installed_name != metadata.canonical_name:
                        aliases[f"{source.source_id}:{metadata.canonical_name}"] = installed_name
                    target = stage / installed_name
                    _copy_managed_skill(skill_dir, target)
                    digest = _tree_digest(target)
                    previous = old_by_name.get(installed_name, {})
                    unchanged_runner = (
                        previous.get("source_id") == source.source_id
                        and previous.get("source_revision") == source.revision
                        and previous.get("digest") == digest
                        and previous.get("script") == metadata.script
                    )
                    execution_enabled = bool(previous.get("execution_enabled")) and unchanged_runner
                    record = SkillRecord(
                        installed_name=installed_name,
                        canonical_name=metadata.canonical_name,
                        description=metadata.description,
                        tags=metadata.tags,
                        source_id=source.source_id,
                        source_url=source.repository,
                        source_revision=source.revision,
                        source_path=source_path,
                        digest=digest,
                        script=metadata.script,
                        execution_enabled=execution_enabled,
                    )
                    records.append(record.to_manifest())
                    installed_for_source += 1
                source_counts[source.source_id] = installed_for_source
                source_rows.append(
                    {
                        "source_id": source.source_id,
                        "repository": source.repository,
                        "revision": source.revision,
                        "include_prefixes": list(source.include_prefixes),
                        "exclude_prefixes": list(source.exclude_prefixes),
                        "expected_skills": source.expected_skills,
                        "installed": installed_for_source,
                    }
                )

            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sources": sorted(source_rows, key=lambda item: str(item.get("source_id", ""))),
                "skills": sorted(records, key=lambda item: str(item.get("installed_name", ""))),
            }
            _atomic_json_write(_manifest_path(stage), manifest)
            _replace_skill_tree(stage, destination)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise

    return InstallReport(
        destination=str(destination),
        installed=sum(source_counts.values()),
        preserved=preserved,
        aliases=aliases,
        sources=source_counts,
    )


def _run_git(arguments: Sequence[str], *, cwd: Path | None = None, timeout_s: float = 600) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SkillCatalogError(f"git failed: {exc}") from exc
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        raise SkillCatalogError(f"git {' '.join(arguments[:2])} failed: {message}")
    return completed.stdout.strip()


def _checkout_pinned_source(source: SkillSource, parent: Path) -> Path:
    checkout = parent / source.source_id
    _run_git(
        ["clone", "--quiet", "--filter=blob:none", "--no-checkout", source.repository, str(checkout)]
    )
    _run_git(["fetch", "--quiet", "--depth=1", "origin", source.revision], cwd=checkout)
    _run_git(["checkout", "--quiet", "--detach", source.revision], cwd=checkout)
    actual = _run_git(["rev-parse", "HEAD"], cwd=checkout)
    if actual != source.revision:
        raise SkillCatalogError(
            f"revision mismatch for {source.source_id}: expected {source.revision}, got {actual}"
        )
    return checkout


def install_pinned_sources(
    *,
    root: str | os.PathLike[str] | None = None,
    source_ids: Sequence[str] | None = None,
    preserve_unmanaged: bool = True,
    progress: Callable[[str], None] | None = None,
) -> InstallReport:
    """Clone the reviewed revisions and install them into Mio's home."""

    known = {source.source_id: source for source in PINNED_SOURCES}
    requested = set(source_ids or known)
    unknown = requested.difference(known)
    if unknown:
        raise SkillCatalogError(f"unknown sources: {', '.join(sorted(unknown))}")
    selected = tuple(source for source in PINNED_SOURCES if source.source_id in requested)
    if not selected:
        raise SkillCatalogError("at least one source is required")
    with tempfile.TemporaryDirectory(prefix="mio-skill-sources-") as temporary:
        parent = Path(temporary)
        checkouts: dict[str, Path] = {}
        for source in selected:
            if progress:
                progress(f"fetch {source.source_id}@{source.revision[:12]}")
            checkouts[source.source_id] = _checkout_pinned_source(source, parent)
        if progress:
            progress("validate and publish Mio skill snapshot")
        return install_skill_sources_from_checkouts(
            checkouts,
            root=root,
            sources=selected,
            preserve_unmanaged=preserve_unmanaged,
        )


__all__ = [
    "InstallReport",
    "MioSkillCatalog",
    "PINNED_SOURCES",
    "SkillCatalogError",
    "SkillExecutionDisabled",
    "SkillNotFoundError",
    "SkillRecord",
    "SkillSource",
    "SkillValidationError",
    "install_pinned_sources",
    "install_skill_sources_from_checkouts",
    "list_mio_skills",
    "mio_home",
    "parse_skill",
    "read_mio_skill",
    "skills_root",
]
