"""Install pinned Headroom and Ponytail MCP tools entirely inside Mio.

The installer never invokes third-party agent installers.  It builds an
isolated release below ``$MIO_HOME/tools/mcp-releases`` and switches the
``mcp-current`` symlink only after both tools pass their local checks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

from mio.paths import mio_home


SCHEMA_VERSION = 3
HEADROOM_VERSION = "0.31.0"
HEADROOM_REQUIREMENT = f"headroom-ai[all]=={HEADROOM_VERSION}"
HEADROOM_LOCK_SHA256 = "0d8be19498726fb690f5d52e0a0b06ed2d8de3cc9989c354d59e7b02bc037d41"
HEADROOM_DIGEST_ALGORITHM = "sha256-full-environment-tree-v1"
PONYTAIL_REPOSITORY = "https://github.com/DietrichGebert/ponytail.git"
PONYTAIL_REVISION = "14a0d79548d4de8fc2de95c1b94bb0de63a739d3"
PONYTAIL_LOCK_SHA256 = "84a159c69ee9c7289164d799dc458e03521bfb3814b3cd30e804aaa14d205342"
MIN_NODE_VERSION = (18, 14, 1)
LOCK_ASSET = Path(__file__).with_name("assets") / "ponytail-mcp-package-lock.json"
HEADROOM_LOCK_ASSET = (
    Path(__file__).with_name("assets") / "headroom-py312-darwin-arm64.lock.json"
)


class InstallerError(RuntimeError):
    """A reproducibility, toolchain, or publication check failed."""


Runner = Callable[..., str]
Which = Callable[[str], str | None]


@dataclass(frozen=True)
class Toolchain:
    python: str
    git: str
    node: str
    npm: str
    python_version: str
    git_version: str
    node_version: str
    npm_version: str


@dataclass(frozen=True)
class InstallPaths:
    home: Path
    tools: Path
    releases: Path
    current: Path
    manifest_link: Path
    headroom_entrypoint: Path
    headroom_environment: Path
    ponytail_entrypoint: Path

    @classmethod
    def for_home(cls, home: Path) -> "InstallPaths":
        tools = home / "tools"
        return cls(
            home=home,
            tools=tools,
            releases=tools / "mcp-releases",
            current=tools / "mcp-current",
            manifest_link=tools / "mcp-tools.json",
            headroom_entrypoint=home / "bin" / "headroom",
            headroom_environment=tools / "headroom-ai",
            ponytail_entrypoint=tools / "sources" / "ponytail",
        )


@dataclass
class _LinkChange:
    path: Path
    existed: bool
    old_symlink: str | None = None
    backup: Path | None = None


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 900.0,
) -> str:
    """Run one argv-only command and return stdout; never invokes a shell."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InstallerError(f"cannot run {Path(command[0]).name}: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise InstallerError(
            f"{Path(command[0]).name} exited {completed.returncode}: {detail or '(no output)'}"
        )
    return completed.stdout.strip()


def _validated_home(value: str | os.PathLike[str] | None = None) -> Path:
    home = mio_home(value).resolve()
    user_home = Path.home().resolve()
    forbidden = (user_home / ".codex", user_home / ".claude")
    if any(home == root or root in home.parents for root in forbidden):
        raise InstallerError(f"refusing third-party agent home as MIO_HOME: {home}")
    return home


def _safe_env(home: Path) -> dict[str, str]:
    """Return a minimal environment with no ambient API/package credentials."""

    allowed = ("PATH", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE")
    env = {name: os.environ[name] for name in allowed if os.environ.get(name)}
    cache = home / "cache"
    env.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(cache),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PIP_INDEX_URL": "https://pypi.org/simple",
            "NPM_CONFIG_CACHE": str(cache / "npm"),
            "NPM_CONFIG_USERCONFIG": os.devnull,
            "NPM_CONFIG_REGISTRY": "https://registry.npmjs.org/",
            "NPM_CONFIG_IGNORE_SCRIPTS": "true",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _prepare_layout(paths: InstallPaths, *, create: bool) -> None:
    """Reject symlinked state parents so writes cannot escape ``MIO_HOME``."""

    directories = (
        paths.home,
        paths.tools,
        paths.releases,
        paths.home / "bin",
        paths.tools / "sources",
        paths.home / "cache",
    )
    root = paths.home.resolve()
    for directory in directories:
        if directory.is_symlink():
            raise InstallerError(f"refusing symlinked Mio state directory: {directory}")
        if directory.exists() and not directory.is_dir():
            raise InstallerError(f"Mio state path is not a directory: {directory}")
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.exists():
            resolved = directory.resolve()
            if resolved != root and root not in resolved.parents:
                raise InstallerError(f"Mio state directory escapes MIO_HOME: {directory}")


def _parse_node_version(raw: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", raw.strip())
    if not match:
        raise InstallerError(f"cannot parse Node version: {raw!r}")
    return tuple(int(part) for part in match.groups())


def _preflight(
    home: Path,
    *,
    runner: Runner = _run,
    which: Which = shutil.which,
) -> Toolchain:
    env = _safe_env(home)
    resolved: dict[str, str] = {}
    for name in ("git", "node", "npm"):
        executable = which(name)
        if not executable:
            raise InstallerError(f"required executable not found: {name}")
        resolved[name] = str(Path(executable).resolve())
    python = str(Path(sys.executable).resolve())
    python_version = runner([python, "--version"], env=env, timeout_s=30)
    git_version = runner([resolved["git"], "--version"], env=env, timeout_s=30)
    node_version = runner([resolved["node"], "--version"], env=env, timeout_s=30)
    npm_version = runner([resolved["npm"], "--version"], env=env, timeout_s=30)
    if _parse_node_version(node_version) < MIN_NODE_VERSION:
        required = ".".join(str(part) for part in MIN_NODE_VERSION)
        raise InstallerError(f"Node {required}+ is required, found {node_version}")
    return Toolchain(
        python=python,
        git=resolved["git"],
        node=resolved["node"],
        npm=resolved["npm"],
        python_version=python_version,
        git_version=git_version,
        node_version=node_version,
        npm_version=npm_version,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _validated_lock_entry(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise InstallerError("Headroom lock entry is not an object")
    name = value.get("name")
    version = value.get("version")
    url = value.get("url")
    digest = value.get("sha256")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise InstallerError("Headroom lock contains an invalid distribution name")
    if (
        not isinstance(version, str)
        or not version
        or len(version) > 128
        or any(character.isspace() for character in version)
    ):
        raise InstallerError(f"Headroom lock contains an invalid version for {name!r}")
    if not isinstance(url, str):
        raise InstallerError(f"Headroom lock contains an invalid URL for {name!r}")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise InstallerError(f"Headroom lock contains an untrusted artifact URL for {name!r}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise InstallerError(f"Headroom lock contains an invalid SHA-256 for {name!r}")
    return {"name": name, "version": version, "url": url, "sha256": digest}


def _load_headroom_lock() -> dict[str, Any]:
    if (
        not HEADROOM_LOCK_ASSET.is_file()
        or _sha256_file(HEADROOM_LOCK_ASSET) != HEADROOM_LOCK_SHA256
    ):
        raise InstallerError("bundled Headroom dependency lock digest does not match")
    try:
        raw = json.loads(HEADROOM_LOCK_ASSET.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallerError(f"cannot read bundled Headroom dependency lock: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise InstallerError("unsupported Headroom dependency lock schema")

    expected_environment = {
        "implementation": platform.python_implementation(),
        "python": ".".join(platform.python_version_tuple()[:2]),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    if raw.get("environment") != expected_environment:
        expected = ", ".join(f"{key}={value}" for key, value in expected_environment.items())
        raise InstallerError(f"Headroom lock is not compatible with this runtime ({expected})")

    build_raw = raw.get("build")
    packages_raw = raw.get("packages")
    if not isinstance(build_raw, list) or not isinstance(packages_raw, list):
        raise InstallerError("Headroom dependency lock is missing build or runtime entries")
    build = [_validated_lock_entry(entry) for entry in build_raw]
    packages = [_validated_lock_entry(entry) for entry in packages_raw]
    if not build or not packages:
        raise InstallerError("Headroom dependency lock must not be empty")

    seen: set[str] = set()
    for entry in [*build, *packages]:
        normalized = _normalized_distribution_name(entry["name"])
        if normalized in seen:
            raise InstallerError(f"duplicate distribution in Headroom lock: {entry['name']!r}")
        seen.add(normalized)
    headroom = next(
        (
            entry
            for entry in packages
            if _normalized_distribution_name(entry["name"]) == "headroom-ai"
        ),
        None,
    )
    if headroom is None or headroom["version"] != HEADROOM_VERSION:
        raise InstallerError("Headroom dependency lock does not contain the pinned package")
    return {
        "schema_version": 1,
        "environment": expected_environment,
        "build": build,
        "packages": packages,
    }


def _requirements_text(entries: Sequence[Mapping[str, str]]) -> str:
    return "".join(
        f"{entry['name']} @ {entry['url']} --hash=sha256:{entry['sha256']}\n"
        for entry in entries
    )


def _lock_inventory(lock: Mapping[str, Any]) -> dict[str, str]:
    return {
        _normalized_distribution_name(entry["name"]): entry["version"]
        for entry in [*lock["build"], *lock["packages"]]
    }


def _pip_report_inventory(path: Path) -> set[tuple[str, str, str, str]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallerError(f"cannot read Headroom pip report: {exc}") from exc
    rows = report.get("install") if isinstance(report, dict) else None
    if not isinstance(rows, list):
        raise InstallerError("Headroom pip report has no installation inventory")
    inventory: set[tuple[str, str, str, str]] = set()
    for row in rows:
        try:
            name = _normalized_distribution_name(row["metadata"]["name"])
            version = row["metadata"]["version"]
            url = row["download_info"]["url"]
            raw_hash = row["download_info"]["archive_info"]["hash"]
            algorithm, digest = raw_hash.split("=", 1)
        except (KeyError, TypeError, ValueError) as exc:
            raise InstallerError("Headroom pip report contains an incomplete entry") from exc
        if algorithm != "sha256":
            raise InstallerError("Headroom pip report contains a non-SHA-256 artifact")
        inventory.add((name, version, url, digest))
    return inventory


def _tree_digest(root: Path, *, ignored_names: frozenset[str] = frozenset()) -> str:
    """Hash names, contents, and symlink targets without following symlinks."""

    if not root.is_dir() or root.is_symlink():
        raise InstallerError(f"managed component is not a regular directory: {root}")
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative_path = path.relative_to(root)
        if any(part in ignored_names for part in relative_path.parts):
            continue
        # Interpreter-created caches are not installation artifacts and can
        # appear after a harmless ``--version`` check. A sourceless top-level
        # .pyc *is* executable package content, so only __pycache__ is ignored.
        if "__pycache__" in relative_path.parts:
            continue
        relative = relative_path.as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"L")
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_dir():
            digest.update(b"D")
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise InstallerError(f"unsupported file type in managed component: {path}")
    return digest.hexdigest()


def _headroom_digest(environment: Path) -> str:
    site_packages = sorted(environment.glob("lib/python*/site-packages"))
    if len(site_packages) != 1:
        raise InstallerError("Headroom installation has an unexpected site-packages layout")
    site = site_packages[0]
    package_dirs = [path / "headroom" for path in site_packages if (path / "headroom").is_dir()]
    metadata_dirs = [
        path
        for site in site_packages
        for path in site.glob(f"headroom_ai-{HEADROOM_VERSION}.dist-info")
        if path.is_dir()
    ]
    records = sorted(path for path in site.glob("*.dist-info/RECORD") if path.is_file())
    required = [
        environment / "bin" / "headroom",
        environment / ".mio-headroom-lock.json",
        environment / ".mio-build-requirements.txt",
        environment / ".mio-requirements.txt",
        environment / ".mio-dependencies.json",
        environment / ".mio-pip-report.json",
    ]
    if (
        len(package_dirs) != 1
        or len(metadata_dirs) != 1
        or not records
        or any(not path.is_file() for path in required)
    ):
        raise InstallerError("Headroom installation layout is incomplete")

    # Hash the complete isolated environment, not merely wheel RECORD files.
    # RECORD proves what pip intended to install; this tree digest also detects
    # a changed, removed, or injected transitive module before MCP activation.
    return _tree_digest(environment)


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_headroom(
    release: Path,
    toolchain: Toolchain,
    *,
    env: Mapping[str, str],
    runner: Runner = _run,
) -> dict[str, Any]:
    lock = _load_headroom_lock()
    destination = release / "headroom"
    runner([toolchain.python, "-m", "venv", str(destination)], env=env)
    python = destination / "bin" / "python"
    headroom = destination / "bin" / "headroom"
    if not python.is_file():
        raise InstallerError("Python venv did not create its interpreter")
    install_env = dict(env)
    install_env["VIRTUAL_ENV"] = str(destination)
    install_env["PATH"] = os.pathsep.join(
        part for part in (str(destination / "bin"), env.get("PATH", "")) if part
    )

    lock_copy = destination / ".mio-headroom-lock.json"
    build_requirements = destination / ".mio-build-requirements.txt"
    requirements = destination / ".mio-requirements.txt"
    shutil.copyfile(HEADROOM_LOCK_ASSET, lock_copy)
    build_requirements.write_text(_requirements_text(lock["build"]), encoding="utf-8")
    requirements.write_text(_requirements_text(lock["packages"]), encoding="utf-8")
    for managed_file in (lock_copy, build_requirements, requirements):
        managed_file.chmod(0o600)

    runner(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-compile",
            "--no-deps",
            "--require-hashes",
            "-r",
            str(build_requirements),
        ],
        env=install_env,
    )
    runner(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-compile",
            "--no-deps",
            "--no-build-isolation",
            "--require-hashes",
            "--report",
            str(destination / ".mio-pip-report.json"),
            "-r",
            str(requirements),
        ],
        env=install_env,
    )
    runner([str(python), "-m", "pip", "check"], env=install_env, timeout_s=120)
    version_output = runner([str(headroom), "--version"], env=install_env, timeout_s=60)
    if not re.search(rf"\b{re.escape(HEADROOM_VERSION)}\b", version_output):
        raise InstallerError(f"unexpected Headroom version: {version_output!r}")

    inventory_script = (
        "import json; from importlib.metadata import distributions; from sysconfig import get_path; "
        "rows=sorted((d.metadata.get('Name',''),d.version) "
        "for d in distributions(path=[get_path('purelib')])); "
        "print(json.dumps([{'name':n,'version':v} for n,v in rows], separators=(',',':')))"
    )
    inventory_raw = runner(
        [str(python), "-c", inventory_script],
        env=install_env,
        timeout_s=120,
    )
    try:
        inventory = json.loads(inventory_raw)
    except json.JSONDecodeError as exc:
        raise InstallerError("cannot inventory Headroom dependencies") from exc
    if not isinstance(inventory, list):
        raise InstallerError("Headroom dependency inventory is not a list")
    actual_inventory: dict[str, str] = {}
    for row in inventory:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("version"), str)
        ):
            raise InstallerError("Headroom dependency inventory contains an invalid row")
        name = _normalized_distribution_name(row["name"])
        if name in actual_inventory:
            raise InstallerError(f"duplicate installed Headroom distribution: {name}")
        actual_inventory[name] = row["version"]
    expected_inventory = _lock_inventory(lock)
    if actual_inventory != expected_inventory:
        missing = sorted(set(expected_inventory) - set(actual_inventory))
        unexpected = sorted(set(actual_inventory) - set(expected_inventory))
        mismatched = sorted(
            name
            for name in set(actual_inventory) & set(expected_inventory)
            if actual_inventory[name] != expected_inventory[name]
        )
        raise InstallerError(
            "Headroom installed inventory does not match lock "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]}, mismatched={mismatched[:5]})"
        )

    expected_report = {
        (
            _normalized_distribution_name(entry["name"]),
            entry["version"],
            entry["url"],
            entry["sha256"],
        )
        for entry in lock["packages"]
    }
    if _pip_report_inventory(destination / ".mio-pip-report.json") != expected_report:
        raise InstallerError("Headroom pip report does not match the bundled artifact lock")
    _atomic_json_write(destination / ".mio-dependencies.json", {"distributions": inventory})
    return {
        "version": HEADROOM_VERSION,
        "requirement": HEADROOM_REQUIREMENT,
        "digest": _headroom_digest(destination),
        "digest_algorithm": HEADROOM_DIGEST_ALGORITHM,
        "lock_sha256": HEADROOM_LOCK_SHA256,
        "build_requirements_sha256": _sha256_file(build_requirements),
        "requirements_sha256": _sha256_file(requirements),
        "dependencies_sha256": _sha256_file(destination / ".mio-dependencies.json"),
        "pip_report_sha256": _sha256_file(destination / ".mio-pip-report.json"),
    }


def _install_ponytail(
    release: Path,
    toolchain: Toolchain,
    *,
    env: Mapping[str, str],
    runner: Runner = _run,
) -> dict[str, Any]:
    if not LOCK_ASSET.is_file() or _sha256_file(LOCK_ASSET) != PONYTAIL_LOCK_SHA256:
        raise InstallerError("bundled Ponytail npm lock digest does not match the validated lock")
    destination = release / "ponytail"
    destination.mkdir(mode=0o700)
    runner([toolchain.git, "init", "--quiet", str(destination)], env=env, timeout_s=120)
    runner(
        [toolchain.git, "-C", str(destination), "remote", "add", "origin", PONYTAIL_REPOSITORY],
        env=env,
        timeout_s=30,
    )
    runner(
        [
            toolchain.git,
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--depth=1",
            "origin",
            PONYTAIL_REVISION,
        ],
        env=env,
    )
    runner(
        [toolchain.git, "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        env=env,
        timeout_s=120,
    )
    actual_revision = runner(
        [toolchain.git, "-C", str(destination), "rev-parse", "HEAD"],
        env=env,
        timeout_s=30,
    )
    if actual_revision != PONYTAIL_REVISION:
        raise InstallerError(f"unexpected Ponytail revision: {actual_revision!r}")

    mcp_root = destination / "ponytail-mcp"
    index = mcp_root / "index.js"
    package = mcp_root / "package.json"
    if not index.is_file() or not package.is_file():
        raise InstallerError("pinned Ponytail checkout does not contain ponytail-mcp")
    shutil.copyfile(LOCK_ASSET, mcp_root / "package-lock.json")
    runner(
        [toolchain.npm, "ci", "--omit=dev", "--ignore-scripts", "--no-audit", "--no-fund"],
        cwd=mcp_root,
        env=env,
    )
    if _sha256_file(mcp_root / "package-lock.json") != PONYTAIL_LOCK_SHA256:
        raise InstallerError("npm changed the validated Ponytail package lock")
    runner([toolchain.node, "--check", str(index)], env=env, timeout_s=60)
    test_file = mcp_root / "test" / "instructions.test.js"
    if test_file.is_file():
        runner([toolchain.node, "--test", str(test_file)], cwd=mcp_root, env=env, timeout_s=120)
    runner(
        [toolchain.git, "-C", str(destination), "diff", "--quiet", "HEAD", "--"],
        env=env,
        timeout_s=30,
    )
    return {
        "repository": PONYTAIL_REPOSITORY,
        "revision": PONYTAIL_REVISION,
        "npm_lock_sha256": PONYTAIL_LOCK_SHA256,
        "digest": _tree_digest(destination, ignored_names=frozenset({".git"})),
    }


def _manifest(
    paths: InstallPaths,
    release: Path,
    toolchain: Toolchain,
    headroom: Mapping[str, Any],
    ponytail: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release": release.relative_to(paths.home).as_posix(),
        "components": {"headroom": dict(headroom), "ponytail": dict(ponytail)},
        "entrypoints": {
            "headroom": paths.headroom_entrypoint.relative_to(paths.home).as_posix(),
            "headroom_environment": paths.headroom_environment.relative_to(paths.home).as_posix(),
            "ponytail": paths.ponytail_entrypoint.relative_to(paths.home).as_posix(),
        },
        "toolchain": {
            "python": toolchain.python_version,
            "git": toolchain.git_version,
            "node": toolchain.node_version,
            "npm": toolchain.npm_version,
        },
        "security": {
            "credential_environment": "sanitized",
            "python_artifacts": "sha256-locked",
            "python_dependency_resolver": False,
            "python_build_isolation": False,
            "npm_lifecycle_scripts": "disabled",
            "third_party_agent_installers": False,
        },
    }


def _temporary_symlink(path: Path, target: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.symlink_to(target)
    return temporary


def _apply_link(path: Path, target: str) -> _LinkChange:
    temporary = _temporary_symlink(path, target)
    existed = path.exists() or path.is_symlink()
    change = _LinkChange(path=path, existed=existed)
    try:
        if path.is_symlink():
            change.old_symlink = os.readlink(path)
            os.replace(temporary, path)
        elif path.exists():
            change.backup = path.with_name(f".{path.name}.backup-{uuid.uuid4().hex}")
            os.replace(path, change.backup)
            try:
                os.replace(temporary, path)
            except Exception:
                os.replace(change.backup, path)
                change.backup = None
                raise
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return change


def _rollback_link(change: _LinkChange) -> None:
    if change.old_symlink is not None:
        temporary = _temporary_symlink(change.path, change.old_symlink)
        os.replace(temporary, change.path)
        return
    if change.backup is not None and change.backup.exists():
        if change.path.is_symlink() or change.path.is_file():
            change.path.unlink(missing_ok=True)
        elif change.path.exists():
            shutil.rmtree(change.path)
        os.replace(change.backup, change.path)
        return
    if not change.existed:
        if change.path.is_symlink() or change.path.is_file():
            change.path.unlink(missing_ok=True)


def _finish_link(change: _LinkChange) -> None:
    backup = change.backup
    if backup is None or not (backup.exists() or backup.is_symlink()):
        return
    if backup.is_symlink() or backup.is_file():
        backup.unlink()
    else:
        shutil.rmtree(backup)


def _publish(paths: InstallPaths, release: Path) -> list[_LinkChange]:
    """Switch the release and compatibility entrypoints with rollback."""

    relative_release = release.relative_to(paths.tools).as_posix()
    links = (
        (paths.current, relative_release),
        (paths.headroom_entrypoint, "../tools/mcp-current/headroom/bin/headroom"),
        (paths.manifest_link, "mcp-current/manifest.json"),
        (paths.headroom_environment, "mcp-current/headroom"),
        (paths.ponytail_entrypoint, "../mcp-current/ponytail"),
    )
    changes: list[_LinkChange] = []
    try:
        for path, target in links:
            changes.append(_apply_link(path, target))
    except Exception:
        for change in reversed(changes):
            with contextlib.suppress(Exception):
                _rollback_link(change)
        raise
    return changes


@contextlib.contextmanager
def _install_lock(paths: InstallPaths) -> Iterator[None]:
    import fcntl

    paths.tools.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = paths.tools / ".mcp-tools-install.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _resolved_inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise InstallerError(f"managed path escapes {root}: {path}")
    return resolved


def check_installation(
    home: str | os.PathLike[str] | None = None,
    *,
    runner: Runner = _run,
    which: Which = shutil.which,
) -> dict[str, Any]:
    """Verify pins, entrypoints, runtime checks, and component digests offline."""

    errors: list[str] = []
    try:
        root = _validated_home(home)
    except InstallerError as exc:
        return {"ok": False, "mode": "check", "errors": [str(exc)]}
    paths = InstallPaths.for_home(root)
    try:
        _prepare_layout(paths, create=False)
    except InstallerError as exc:
        return {"ok": False, "mode": "check", "mio_home": str(root), "errors": [str(exc)]}
    toolchain: Toolchain | None = None
    try:
        toolchain = _preflight(root, runner=runner, which=which)
    except InstallerError as exc:
        errors.append(str(exc))

    if not paths.current.is_symlink():
        errors.append(f"missing managed current release symlink: {paths.current}")
        return {"ok": False, "mode": "check", "mio_home": str(root), "errors": errors}
    try:
        release = _resolved_inside(paths.current, paths.releases)
    except (InstallerError, OSError) as exc:
        errors.append(str(exc))
        return {"ok": False, "mode": "check", "mio_home": str(root), "errors": errors}
    manifest_path = release / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read managed MCP manifest: {exc}")
        return {"ok": False, "mode": "check", "mio_home": str(root), "errors": errors}
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported managed MCP manifest schema")
        manifest = {}
    expected_release = release.relative_to(root).as_posix()
    if manifest.get("release") != expected_release:
        errors.append("manifest release path does not match the active release")

    expected_links = {
        paths.manifest_link: manifest_path,
        paths.headroom_entrypoint: release / "headroom" / "bin" / "headroom",
        paths.headroom_environment: release / "headroom",
        paths.ponytail_entrypoint: release / "ponytail",
    }
    for link, expected in expected_links.items():
        try:
            if not link.is_symlink() or link.resolve() != expected.resolve():
                errors.append(f"managed entrypoint does not target the active release: {link}")
        except OSError as exc:
            errors.append(f"cannot resolve managed entrypoint {link}: {exc}")

    components = manifest.get("components", {}) if isinstance(manifest, dict) else {}
    headroom_meta = components.get("headroom", {}) if isinstance(components, dict) else {}
    ponytail_meta = components.get("ponytail", {}) if isinstance(components, dict) else {}
    if headroom_meta.get("version") != HEADROOM_VERSION:
        errors.append("manifest Headroom pin does not match installer")
    if headroom_meta.get("requirement") != HEADROOM_REQUIREMENT:
        errors.append("manifest Headroom requirement does not match installer")
    if headroom_meta.get("digest_algorithm") != HEADROOM_DIGEST_ALGORITHM:
        errors.append("manifest Headroom digest algorithm does not match installer")
    if headroom_meta.get("lock_sha256") != HEADROOM_LOCK_SHA256:
        errors.append("manifest Headroom artifact lock does not match installer")
    if ponytail_meta.get("revision") != PONYTAIL_REVISION:
        errors.append("manifest Ponytail pin does not match installer")
    if ponytail_meta.get("npm_lock_sha256") != PONYTAIL_LOCK_SHA256:
        errors.append("manifest Ponytail npm lock does not match installer")

    headroom_root = release / "headroom"
    ponytail_root = release / "ponytail"
    try:
        if headroom_meta.get("digest") != _headroom_digest(headroom_root):
            errors.append("Headroom component digest mismatch")
    except InstallerError as exc:
        errors.append(str(exc))
    dependencies_path = headroom_root / ".mio-dependencies.json"
    pip_report_path = headroom_root / ".mio-pip-report.json"
    lock_path = headroom_root / ".mio-headroom-lock.json"
    build_requirements_path = headroom_root / ".mio-build-requirements.txt"
    requirements_path = headroom_root / ".mio-requirements.txt"
    if not lock_path.is_file() or _sha256_file(lock_path) != HEADROOM_LOCK_SHA256:
        errors.append("Headroom installed artifact lock digest mismatch")
    if (
        not build_requirements_path.is_file()
        or headroom_meta.get("build_requirements_sha256")
        != _sha256_file(build_requirements_path)
    ):
        errors.append("Headroom build requirements digest mismatch")
    if (
        not requirements_path.is_file()
        or headroom_meta.get("requirements_sha256") != _sha256_file(requirements_path)
    ):
        errors.append("Headroom runtime requirements digest mismatch")
    if (
        not dependencies_path.is_file()
        or headroom_meta.get("dependencies_sha256") != _sha256_file(dependencies_path)
    ):
        errors.append("Headroom dependency inventory digest mismatch")
    if not pip_report_path.is_file() or headroom_meta.get("pip_report_sha256") != _sha256_file(
        pip_report_path
    ):
        errors.append("Headroom pip report digest mismatch")
    try:
        lock = _load_headroom_lock()
        expected_inventory = _lock_inventory(lock)
        inventory_raw = json.loads(dependencies_path.read_text(encoding="utf-8"))
        rows = inventory_raw.get("distributions") if isinstance(inventory_raw, dict) else None
        if not isinstance(rows, list):
            raise InstallerError("Headroom dependency inventory is not a list")
        installed_inventory = {
            _normalized_distribution_name(row["name"]): row["version"]
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and isinstance(row.get("version"), str)
        }
        if len(installed_inventory) != len(rows) or installed_inventory != expected_inventory:
            errors.append("Headroom installed inventory does not match artifact lock")
        expected_report = {
            (
                _normalized_distribution_name(entry["name"]),
                entry["version"],
                entry["url"],
                entry["sha256"],
            )
            for entry in lock["packages"]
        }
        if _pip_report_inventory(pip_report_path) != expected_report:
            errors.append("Headroom pip report does not match artifact lock")
    except (InstallerError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    try:
        digest = _tree_digest(ponytail_root, ignored_names=frozenset({".git"}))
        if ponytail_meta.get("digest") != digest:
            errors.append("Ponytail component digest mismatch")
    except InstallerError as exc:
        errors.append(str(exc))

    lock_path = ponytail_root / "ponytail-mcp" / "package-lock.json"
    if not lock_path.is_file() or _sha256_file(lock_path) != PONYTAIL_LOCK_SHA256:
        errors.append("Ponytail package lock digest mismatch")
    if toolchain is not None:
        env = _safe_env(root)
        try:
            version = runner(
                [str(headroom_root / "bin" / "headroom"), "--version"],
                env=env,
                timeout_s=60,
            )
            if not re.search(rf"\b{re.escape(HEADROOM_VERSION)}\b", version):
                errors.append(f"unexpected Headroom runtime version: {version!r}")
        except InstallerError as exc:
            errors.append(str(exc))
        try:
            runner(
                [toolchain.node, "--check", str(ponytail_root / "ponytail-mcp" / "index.js")],
                env=env,
                timeout_s=60,
            )
            revision = runner(
                [toolchain.git, "-C", str(ponytail_root), "rev-parse", "HEAD"],
                env=env,
                timeout_s=30,
            )
            if revision != PONYTAIL_REVISION:
                errors.append(f"unexpected Ponytail runtime revision: {revision!r}")
            runner(
                [toolchain.git, "-C", str(ponytail_root), "diff", "--quiet", "HEAD", "--"],
                env=env,
                timeout_s=30,
            )
        except InstallerError as exc:
            errors.append(str(exc))

    return {
        "ok": not errors,
        "mode": "check",
        "mio_home": str(root),
        "manifest": str(manifest_path),
        "release": str(release),
        "components": {
            "headroom": HEADROOM_VERSION,
            "ponytail": PONYTAIL_REVISION,
        },
        "errors": errors,
    }


def install_mcp_tools(
    home: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    runner: Runner = _run,
    which: Which = shutil.which,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build and atomically activate the pinned Mio MCP tool release."""

    root = _validated_home(home)
    paths = InstallPaths.for_home(root)
    _prepare_layout(paths, create=True)
    with _install_lock(paths):
        if not force:
            current = check_installation(root, runner=runner, which=which)
            if current.get("ok"):
                return {**current, "mode": "install", "status": "unchanged"}
        toolchain = _preflight(root, runner=runner, which=which)
        env = _safe_env(root)
        (root / "cache" / "pip").mkdir(mode=0o700, exist_ok=True)
        (root / "cache" / "npm").mkdir(mode=0o700, exist_ok=True)
        release_id = (
            f"headroom-{HEADROOM_VERSION}_ponytail-{PONYTAIL_REVISION[:12]}-{uuid.uuid4().hex[:8]}"
        )
        release = paths.releases / release_id
        release.mkdir(mode=0o700)
        published = False
        try:
            if progress:
                progress(f"install {HEADROOM_REQUIREMENT}")
            headroom = _install_headroom(release, toolchain, env=env, runner=runner)
            if progress:
                progress(f"install ponytail@{PONYTAIL_REVISION[:12]}")
            ponytail = _install_ponytail(release, toolchain, env=env, runner=runner)
            manifest = _manifest(paths, release, toolchain, headroom, ponytail)
            _atomic_json_write(release / "manifest.json", manifest)
            if progress:
                progress(f"activate {release_id}")
            changes = _publish(paths, release)
            published = True
            for change in changes:
                # Publication already succeeded. A legacy backup that cannot
                # be removed is harmless and must never invalidate the release.
                with contextlib.suppress(OSError):
                    _finish_link(change)
        finally:
            if not published and release.exists():
                shutil.rmtree(release)

    report = check_installation(root, runner=runner, which=which)
    if not report.get("ok"):
        raise InstallerError("installed release failed verification: " + "; ".join(report["errors"]))
    return {**report, "mode": "install", "status": "installed"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install pinned Headroom and Ponytail MCP runtimes inside Mio only; "
            "no Codex/Claude integration is invoked."
        )
    )
    parser.add_argument(
        "--mio-home",
        type=Path,
        help="Mio application home (default: $MIO_HOME or ~/.mio)",
    )
    parser.add_argument("--check", action="store_true", help="Verify the active release without network access")
    parser.add_argument("--force", action="store_true", help="Build a fresh release even if the active one is valid")
    parser.add_argument("--json", action="store_true", help="Print only the final JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def progress(message: str) -> None:
        if not args.json:
            print(f"[mio-mcp-tools] {message}", file=sys.stderr)

    try:
        if args.check:
            report = check_installation(args.mio_home)
        else:
            report = install_mcp_tools(args.mio_home, force=args.force, progress=progress)
    except InstallerError as exc:
        report = {"ok": False, "mode": "check" if args.check else "install", "errors": [str(exc)]}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
