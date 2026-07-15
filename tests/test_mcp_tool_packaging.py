"""Source/wheel resource regressions for the Mio-owned MCP installer."""

from __future__ import annotations

import importlib.util
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

from mio import main as mio_main
from mio.mcp import tool_installer


def test_pinned_ponytail_lock_is_a_package_resource():
    lock = resources.files("mio.mcp").joinpath(
        "assets", "ponytail-mcp-package-lock.json"
    )
    assert lock.is_file()
    assert tool_installer._sha256_file(Path(str(lock))) == tool_installer.PONYTAIL_LOCK_SHA256


def test_source_script_remains_a_thin_compatibility_wrapper():
    wrapper = Path(__file__).parents[1] / "scripts" / "install_mio_mcp_tools.py"
    spec = importlib.util.spec_from_file_location("mio_installer_wrapper", wrapper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is tool_installer.main


def test_mio_mcp_check_dispatches_to_packaged_installer(monkeypatch, capsys, tmp_path):
    expected = {
        "ok": True,
        "mode": "check",
        "release": str(tmp_path / "release"),
        "errors": [],
    }
    monkeypatch.setattr(tool_installer, "check_installation", lambda home: expected)
    args = SimpleNamespace(
        mcp_action="check",
        mio_home=str(tmp_path),
        json=True,
        force=False,
        name=None,
    )

    mio_main._cmd_mcp(args)

    assert '"ok": true' in capsys.readouterr().out
