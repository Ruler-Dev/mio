from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mio.mcp import tool_installer as installer


def _toolchain() -> installer.Toolchain:
    return installer.Toolchain(
        python="/usr/bin/python3",
        git="/usr/bin/git",
        node="/usr/bin/node",
        npm="/usr/bin/npm",
        python_version="Python 3.12.0",
        git_version="git version 2.50.0",
        node_version="v20.19.0",
        npm_version="10.8.0",
    )


def _fake_headroom_tree(release: Path) -> dict:
    lock = installer._load_headroom_lock()
    environment = release / "headroom"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "headroom").write_text("#!/bin/sh\n", encoding="utf-8")
    site = environment / "lib" / "python3.12" / "site-packages"
    (site / "headroom").mkdir(parents=True)
    (site / "headroom" / "__init__.py").write_text("VERSION = '0.31.0'\n", encoding="utf-8")
    metadata = site / "headroom_ai-0.31.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: headroom-ai\nVersion: 0.31.0\n", encoding="utf-8")
    (metadata / "RECORD").write_text("headroom/__init__.py,,\n", encoding="utf-8")
    lock_copy = environment / ".mio-headroom-lock.json"
    lock_copy.write_bytes(installer.HEADROOM_LOCK_ASSET.read_bytes())
    build_requirements = environment / ".mio-build-requirements.txt"
    build_requirements.write_text(installer._requirements_text(lock["build"]), encoding="utf-8")
    requirements = environment / ".mio-requirements.txt"
    requirements.write_text(installer._requirements_text(lock["packages"]), encoding="utf-8")
    pip_report = {
        "install": [
            {
                "metadata": {"name": row["name"], "version": row["version"]},
                "download_info": {
                    "url": row["url"],
                    "archive_info": {"hash": f"sha256={row['sha256']}"},
                },
            }
            for row in lock["packages"]
        ]
    }
    (environment / ".mio-pip-report.json").write_text(
        json.dumps(pip_report) + "\n",
        encoding="utf-8",
    )
    inventory = [
        {"name": row["name"], "version": row["version"]}
        for row in [*lock["build"], *lock["packages"]]
    ]
    installer._atomic_json_write(
        environment / ".mio-dependencies.json",
        {"distributions": inventory},
    )
    return {
        "version": installer.HEADROOM_VERSION,
        "requirement": installer.HEADROOM_REQUIREMENT,
        "digest": installer._headroom_digest(environment),
        "digest_algorithm": installer.HEADROOM_DIGEST_ALGORITHM,
        "lock_sha256": installer.HEADROOM_LOCK_SHA256,
        "build_requirements_sha256": installer._sha256_file(build_requirements),
        "requirements_sha256": installer._sha256_file(requirements),
        "dependencies_sha256": installer._sha256_file(environment / ".mio-dependencies.json"),
        "pip_report_sha256": installer._sha256_file(environment / ".mio-pip-report.json"),
    }


def _fake_ponytail_tree(release: Path) -> dict:
    root = release / "ponytail"
    mcp = root / "ponytail-mcp"
    (mcp / "node_modules" / "example").mkdir(parents=True)
    (root / ".git").mkdir()
    (mcp / "index.js").write_text("export const ok = true;\n", encoding="utf-8")
    (mcp / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
    (mcp / "node_modules" / "example" / "index.js").write_text("export {};\n", encoding="utf-8")
    (mcp / "package-lock.json").write_bytes(installer.LOCK_ASSET.read_bytes())
    return {
        "repository": installer.PONYTAIL_REPOSITORY,
        "revision": installer.PONYTAIL_REVISION,
        "npm_lock_sha256": installer.PONYTAIL_LOCK_SHA256,
        "digest": installer._tree_digest(root, ignored_names=frozenset({".git"})),
    }


def test_safe_environment_drops_secrets_and_rejects_agent_homes(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("NPM_TOKEN", "secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:password@example.invalid/simple")
    home = tmp_path / "mio-home"

    env = installer._safe_env(home)

    assert "HF_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "NPM_TOKEN" not in env
    assert env["PIP_INDEX_URL"] == "https://pypi.org/simple"
    assert env["NPM_CONFIG_USERCONFIG"] == os.devnull
    with pytest.raises(installer.InstallerError, match="third-party agent home"):
        installer._validated_home(Path.home() / ".codex" / "mio")


def test_headroom_lock_covers_every_runtime_artifact_and_build_tool():
    lock = installer._load_headroom_lock()

    assert installer._sha256_file(installer.HEADROOM_LOCK_ASSET) == installer.HEADROOM_LOCK_SHA256
    assert len(lock["packages"]) == 130
    assert {"pip", "maturin", "puccinialin"} <= {
        row["name"].casefold() for row in lock["build"]
    }
    assert any(
        row["name"].casefold() == "headroom-ai"
        and row["version"] == installer.HEADROOM_VERSION
        for row in lock["packages"]
    )
    assert all(row["url"].startswith("https://files.pythonhosted.org/") for row in lock["packages"])
    assert all(len(row["sha256"]) == 64 for row in [*lock["build"], *lock["packages"]])


def test_headroom_digest_covers_transitive_files_and_injected_modules(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    _fake_headroom_tree(release)
    environment = release / "headroom"
    site = next(environment.glob("lib/python*/site-packages"))
    dependency = site / "dependency" / "__init__.py"
    dependency.parent.mkdir()
    dependency.write_text("SAFE = True\n", encoding="utf-8")

    original = installer._headroom_digest(environment)
    dependency.write_text("TAMPERED = True\n", encoding="utf-8")
    modified = installer._headroom_digest(environment)
    assert modified != original

    injected = site / "injected.py"
    injected.write_text("VALUE = 'unexpected'\n", encoding="utf-8")
    assert installer._headroom_digest(environment) != modified


def test_layout_rejects_symlinked_parent_that_escapes_mio_home(tmp_path):
    home = tmp_path / "mio-home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / "tools").symlink_to(outside, target_is_directory=True)

    with pytest.raises(installer.InstallerError, match="symlinked Mio state"):
        installer._prepare_layout(installer.InstallPaths.for_home(home), create=True)
    assert list(outside.iterdir()) == []


def test_preflight_requires_supported_node(monkeypatch, tmp_path):
    monkeypatch.setattr(installer.sys, "executable", "/usr/bin/python3")

    def runner(command, **kwargs):
        executable = Path(command[0]).name
        return {
            "python3": "Python 3.12.0",
            "git": "git version 2.50.0",
            "node": "v16.20.0",
            "npm": "10.8.0",
        }[executable]

    with pytest.raises(installer.InstallerError, match="Node 18.14.1"):
        installer._preflight(
            tmp_path,
            runner=runner,
            which=lambda name: f"/usr/bin/{name}",
        )


def test_headroom_install_is_exact_isolated_and_never_calls_agent_installer(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    lock = installer._load_headroom_lock()
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        command = list(command)
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[-1])
            (environment / "bin").mkdir(parents=True)
            (environment / "bin" / "python").write_text("python\n", encoding="utf-8")
            (environment / "bin" / "headroom").write_text("#!/bin/sh\n", encoding="utf-8")
            site = environment / "lib" / "python3.12" / "site-packages"
            (site / "headroom").mkdir(parents=True)
            (site / "headroom" / "__init__.py").write_text("", encoding="utf-8")
            metadata = site / "headroom_ai-0.31.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text("Version: 0.31.0\n", encoding="utf-8")
            (metadata / "RECORD").write_text("headroom/__init__.py,,\n", encoding="utf-8")
        if "--report" in command:
            report = {
                "install": [
                    {
                        "metadata": {"name": row["name"], "version": row["version"]},
                        "download_info": {
                            "url": row["url"],
                            "archive_info": {"hash": f"sha256={row['sha256']}"},
                        },
                    }
                    for row in lock["packages"]
                ]
            }
            Path(command[command.index("--report") + 1]).write_text(
                json.dumps(report) + "\n",
                encoding="utf-8",
            )
        if command[-1] == "--version" and Path(command[0]).name == "headroom":
            return "headroom, version 0.31.0"
        if "-c" in command:
            inventory = [
                {"name": row["name"], "version": row["version"]}
                for row in [*lock["build"], *lock["packages"]]
            ]
            return json.dumps(inventory)
        return ""

    result = installer._install_headroom(
        release,
        _toolchain(),
        env=installer._safe_env(tmp_path / "home"),
        runner=runner,
    )

    assert result["version"] == "0.31.0"
    pip_commands = [command for command in commands if "install" in command]
    assert len(pip_commands) == 2
    assert all("--no-deps" in command and "--require-hashes" in command for command in pip_commands)
    runtime_command = next(command for command in pip_commands if "--report" in command)
    assert "--no-build-isolation" in runtime_command
    assert installer.HEADROOM_REQUIREMENT not in runtime_command
    assert result["lock_sha256"] == installer.HEADROOM_LOCK_SHA256
    assert all(command[-2:] != ["mcp", "install"] for command in commands)
    assert all(".codex" not in argument for command in commands for argument in command)


def test_ponytail_install_uses_pinned_fetch_locked_npm_and_node_checks(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    commands: list[tuple[list[str], Path | None]] = []

    def runner(command, *, cwd=None, **kwargs):
        command = list(command)
        commands.append((command, cwd))
        if "checkout" in command:
            root = Path(command[command.index("-C") + 1])
            mcp = root / "ponytail-mcp"
            (mcp / "test").mkdir(parents=True)
            (mcp / "index.js").write_text("export {};\n", encoding="utf-8")
            (mcp / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
            (mcp / "test" / "instructions.test.js").write_text("export {};\n", encoding="utf-8")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return installer.PONYTAIL_REVISION
        if len(command) > 1 and command[1] == "ci":
            (cwd / "node_modules" / "example").mkdir(parents=True)
            (cwd / "node_modules" / "example" / "index.js").write_text("export {};\n", encoding="utf-8")
        return ""

    result = installer._install_ponytail(
        release,
        _toolchain(),
        env=installer._safe_env(tmp_path / "home"),
        runner=runner,
    )

    assert result["revision"] == installer.PONYTAIL_REVISION
    fetch = next(command for command, _ in commands if "fetch" in command)
    assert fetch[-1] == installer.PONYTAIL_REVISION
    npm = next(command for command, _ in commands if len(command) > 1 and command[1] == "ci")
    assert {"--ignore-scripts", "--no-audit", "--no-fund"} <= set(npm)
    assert any(command[1] == "--check" for command, _ in commands if command[0].endswith("node"))
    assert installer._sha256_file(
        release / "ponytail" / "ponytail-mcp" / "package-lock.json"
    ) == installer.PONYTAIL_LOCK_SHA256


def test_install_publishes_release_manifest_and_check_detects_tampering(monkeypatch, tmp_path):
    home = tmp_path / "mio-home"
    legacy = home / "tools" / "sources" / "ponytail"
    legacy.mkdir(parents=True)
    (legacy / "legacy.txt").write_text("old\n", encoding="utf-8")
    legacy_headroom = home / "tools" / "headroom-ai"
    legacy_headroom.mkdir(parents=True)
    (legacy_headroom / "legacy.txt").write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(installer, "_preflight", lambda *args, **kwargs: _toolchain())
    monkeypatch.setattr(
        installer,
        "_install_headroom",
        lambda release, *args, **kwargs: _fake_headroom_tree(release),
    )
    monkeypatch.setattr(
        installer,
        "_install_ponytail",
        lambda release, *args, **kwargs: _fake_ponytail_tree(release),
    )

    def runner(command, **kwargs):
        if command[-1] == "--version" and Path(command[0]).name == "headroom":
            return "headroom, version 0.31.0"
        if command[-2:] == ["rev-parse", "HEAD"]:
            return installer.PONYTAIL_REVISION
        return ""

    report = installer.install_mcp_tools(home, force=True, runner=runner)

    assert report["ok"] is True
    assert report["status"] == "installed"
    paths = installer.InstallPaths.for_home(home.resolve())
    assert paths.current.is_symlink()
    assert paths.headroom_entrypoint.resolve() == paths.current.resolve() / "headroom" / "bin" / "headroom"
    assert paths.headroom_environment.resolve() == paths.current.resolve() / "headroom"
    assert paths.ponytail_entrypoint.resolve() == paths.current.resolve() / "ponytail"
    manifest = json.loads(paths.manifest_link.read_text(encoding="utf-8"))
    assert manifest["components"]["headroom"]["version"] == "0.31.0"
    assert (
        manifest["components"]["headroom"]["digest_algorithm"]
        == installer.HEADROOM_DIGEST_ALGORITHM
    )
    assert manifest["components"]["ponytail"]["revision"] == installer.PONYTAIL_REVISION
    assert paths.manifest_link.resolve().stat().st_mode & 0o777 == 0o600
    assert not legacy.joinpath("legacy.txt").exists()
    assert not legacy_headroom.joinpath("legacy.txt").exists()

    record = next(paths.headroom_environment.glob("lib/python*/site-packages/*.dist-info/RECORD"))
    original_record = record.read_text(encoding="utf-8")
    record.write_text(original_record + "tampered.py,,\n", encoding="utf-8")
    checked = installer.check_installation(home, runner=runner)
    assert checked["ok"] is False
    assert "Headroom component digest mismatch" in checked["errors"]
    record.write_text(original_record, encoding="utf-8")

    index = paths.ponytail_entrypoint / "ponytail-mcp" / "index.js"
    index.write_text("tampered\n", encoding="utf-8")
    checked = installer.check_installation(home, runner=runner)
    assert checked["ok"] is False
    assert "Ponytail component digest mismatch" in checked["errors"]


def test_publish_rolls_back_prior_links_on_failure(monkeypatch, tmp_path):
    home = tmp_path / "mio-home"
    paths = installer.InstallPaths.for_home(home)
    old_release = paths.releases / "old"
    new_release = paths.releases / "new"
    old_release.mkdir(parents=True)
    new_release.mkdir()
    paths.current.symlink_to("mcp-releases/old")
    paths.headroom_entrypoint.parent.mkdir(parents=True)
    paths.headroom_entrypoint.symlink_to("old-headroom")
    real_apply = installer._apply_link

    def failing_apply(path, target):
        if path == paths.ponytail_entrypoint:
            raise OSError("simulated publication failure")
        return real_apply(path, target)

    monkeypatch.setattr(installer, "_apply_link", failing_apply)
    with pytest.raises(OSError, match="simulated"):
        installer._publish(paths, new_release)

    assert os.readlink(paths.current) == "mcp-releases/old"
    assert os.readlink(paths.headroom_entrypoint) == "old-headroom"
    assert not paths.manifest_link.exists()


def test_check_json_cli_is_read_only(monkeypatch, tmp_path, capsys):
    expected = {"ok": True, "mode": "check", "errors": [], "mio_home": str(tmp_path)}
    monkeypatch.setattr(installer, "check_installation", lambda home: expected)
    monkeypatch.setattr(
        installer,
        "install_mcp_tools",
        lambda *args, **kwargs: pytest.fail("--check must not install"),
    )

    assert installer.main(["--mio-home", str(tmp_path), "--check", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
