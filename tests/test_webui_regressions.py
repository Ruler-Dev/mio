"""Focused regressions for WebUI lifecycle, imports, hooks, and scheduling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from mio.webui import router, scheduler, webhooks


def test_sovereignty_onboarding_opens_the_real_network_monitor():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "onboarding_sovereignty.js"
    source = asset.read_text(encoding="utf-8")

    assert 'data-act="noop"' not in source
    assert 'data-act="network"' in source
    assert 'document.querySelector(".mio-sovereignty .mio-sov-net")' in source
    assert "networkMonitor.click();" in source


def test_compare_requires_two_distinct_real_models():
    page = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "compare.html"
    source = page.read_text(encoding="utf-8")

    assert "TIERS.length === 1" in source
    assert "left.value === right.value" in source
    assert "both sides must be distinct" in source
    assert "runButton.disabled = true" in source
    assert "['small', 'medium', 'large', 'large-moe']" not in source
    assert "if (!prompt || !validateComparison()) return;" in source


def test_settings_mcp_health_panel_is_retryable_and_uses_safe_dom_rendering():
    root = Path(__file__).parents[1]
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    css = (root / "mio" / "webui" / "assets" / "main.css").read_text(encoding="utf-8")
    health_ui = shell.split("function renderMcpHealth(payload)", 1)[1].split(
        "function openSettings()", 1
    )[0]

    assert 'id="mcpHealthRetry"' in shell
    assert "fetch('/v1/mcp/health'" in health_ui
    assert "method: 'POST'" in health_ui
    assert "list.replaceChildren();" in health_ui
    assert "name.textContent" in health_ui
    assert ".innerHTML" not in health_ui
    assert "retry.disabled = true" in health_ui
    assert "refreshMcpHealth();" in shell.split("function openSettings()", 1)[1]
    assert ".mcp-health-card" in css
    assert '.mcp-health-state[data-state="degraded"]' in css


def test_emoji_button_uses_parent_prepend_instead_of_descendant_insert_before():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "emoji.js"
    source = asset.read_text(encoding="utf-8")
    add_button = source.split("function addButton()", 1)[1].split(
        "function injectCSS()", 1
    )[0]

    assert "actions.prepend(btn);" in add_button
    assert "actions.insertBefore(" not in add_button
    assert "actions.querySelector('button')" not in add_button


def test_mobile_layout_does_not_create_an_implicit_metrics_column():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "main.css"
    source = asset.read_text(encoding="utf-8")
    responsive = source.split("/* ===== Responsive ===== */", 1)[1].split(
        "@media (prefers-reduced-motion", 1
    )[0]

    assert "grid-template-columns: 48px minmax(0, 1fr)" in responsive
    assert ".metrics-bar" in responsive
    assert "grid-column: 2" in responsive


def test_flow_has_a_canvas_first_mobile_layout():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "main.css"
    source = asset.read_text(encoding="utf-8")
    flow_mobile = source.split("@keyframes pulse", 1)[1].split(
        "/* ===== Journal view", 1
    )[0]

    assert "@media (max-width: 768px)" in flow_mobile
    assert "grid-template-columns: minmax(0, 1fr)" in flow_mobile
    assert "grid-template-rows: 140px minmax(320px, 1fr)" in flow_mobile
    assert ".flow-nodes" in flow_mobile
    assert "flex-direction: row" in flow_mobile


@pytest.mark.asyncio
async def test_tier_switch_loads_new_tier_before_retiring_old(monkeypatch):
    lock = threading.Lock()
    events: list[str] = []

    class Manager:
        config = SimpleNamespace(tiers={"old": object(), "new": object()})

        def __init__(self):
            self.loaded = ["old"]

        def loaded_tiers(self):
            assert lock.locked()
            events.append("snapshot")
            return list(self.loaded)

        def load_tier(self, tier):
            assert lock.locked()
            events.append(f"load:{tier}")
            self.loaded.append(tier)

        def unload_tier(self, tier):
            assert lock.locked()
            events.append(f"unload:{tier}")
            self.loaded.remove(tier)

    manager = Manager()
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)

    result = await router.switch_tier({"tier": "new"})

    assert result == {"ok": True, "tier": "new", "already_loaded": False}
    assert events == ["snapshot", "load:new", "unload:old"]
    assert manager.loaded == ["new"]


@pytest.mark.asyncio
async def test_tier_switch_load_failure_keeps_serving_tier(monkeypatch):
    lock = threading.Lock()

    class Manager:
        config = SimpleNamespace(tiers={"old": object(), "new": object()})

        def __init__(self):
            self.loaded = ["old"]
            self.unloaded: list[str] = []

        def loaded_tiers(self):
            assert lock.locked()
            return list(self.loaded)

        def load_tier(self, _tier):
            assert lock.locked()
            raise MemoryError("insufficient unified memory")

        def unload_tier(self, tier):
            self.unloaded.append(tier)

    manager = Manager()
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)

    with pytest.raises(HTTPException) as raised:
        await router.switch_tier({"tier": "new"})

    assert raised.value.status_code == 409
    assert manager.loaded == ["old"]
    assert manager.unloaded == []


@pytest.mark.asyncio
async def test_tier_switch_retires_other_tiers_when_target_is_already_loaded(
    monkeypatch,
):
    lock = threading.Lock()

    class Manager:
        config = SimpleNamespace(tiers={"old": object(), "new": object()})

        def __init__(self):
            self.loaded = ["old", "new"]
            self.load_calls: list[str] = []

        def loaded_tiers(self):
            assert lock.locked()
            return list(self.loaded)

        def load_tier(self, tier):
            self.load_calls.append(tier)

        def unload_tier(self, tier):
            assert lock.locked()
            self.loaded.remove(tier)

    manager = Manager()
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)

    result = await router.switch_tier({"tier": "new"})

    assert result == {"ok": True, "tier": "new", "already_loaded": True}
    assert manager.loaded == ["new"]
    assert manager.load_calls == []


def _chatgpt_node(
    parent: str | None,
    role: str,
    text: str,
    *,
    created: float,
) -> dict:
    return {
        "parent": parent,
        "children": [],
        "message": {
            "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "create_time": created,
        },
    }


def test_chatgpt_import_uses_selected_branch_instead_of_flattening_siblings():
    mapping = {
        "root": _chatgpt_node(None, "user", "question", created=1),
        "old": _chatgpt_node("root", "assistant", "old answer", created=2),
        "active": _chatgpt_node("root", "assistant", "active answer", created=3),
    }
    mapping["root"]["children"] = ["old", "active"]

    messages = router._normalize_imported_chat(
        {"mapping": mapping, "current_node": "active"}
    )

    assert messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "active answer"},
    ]


def test_chatgpt_import_is_iterative_and_falls_back_to_newest_leaf():
    mapping: dict[str, dict] = {}
    parent = None
    for index in range(2_000):
        node_id = f"n{index}"
        mapping[node_id] = _chatgpt_node(
            parent,
            "user" if index % 2 == 0 else "assistant",
            str(index),
            created=index,
        )
        parent = node_id

    messages = router._normalize_imported_chat({"mapping": mapping})

    assert len(messages) == 2_000
    assert messages[-1]["content"] == "1999"


def test_chat_import_validates_full_batch_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "_sessions_dir", tmp_path)
    too_large = "x" * (router._MAX_IMPORT_MESSAGE_BYTES + 1)
    body = {
        "source": "mio",
        "data": [
            {"messages": [{"role": "user", "content": "valid"}]},
            {"messages": [{"role": "user", "content": too_large}]},
        ],
    }

    with pytest.raises(HTTPException, match="2 MiB"):
        asyncio.run(router.import_chats(body))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "invalid",
    [True, False, "16", 16.0, None, 0, -1, 32_769],
)
def test_webui_config_rejects_non_integer_or_unbounded_max_tokens(
    invalid,
    monkeypatch,
):
    monkeypatch.setattr(router, "_max_tokens", 123)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(router.update_config({"max_tokens": invalid}))

    assert raised.value.status_code == 400
    assert router._max_tokens == 123


@pytest.mark.parametrize("valid", [1, 32_768])
def test_webui_config_accepts_bounded_max_tokens(valid, monkeypatch):
    monkeypatch.setattr(router, "_max_tokens", 123)

    result = asyncio.run(router.update_config({"max_tokens": valid}))

    assert result["max_tokens"] == valid
    assert router._max_tokens == valid


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [True, "8", 8.0, None, 0, 32_769])
async def test_webui_socket_rejects_invalid_max_tokens_before_side_effects(
    invalid,
    monkeypatch,
):
    class Socket:
        def __init__(self):
            self.events: list[dict] = []

        async def send_json(self, value):
            self.events.append(value)

    socket = Socket()
    monkeypatch.setattr(router, "_manager", None)

    await router._handle_chat(
        socket,
        {
            "max_tokens": invalid,
            "messages": [{"role": "user", "content": "/remember must-not-save"}],
        },
    )

    assert socket.events == [
        {
            "type": "error",
            "message": "max_tokens must be an integer between 1 and 32768",
        }
    ]


def test_webhook_requires_secret_and_never_returns_auth_material(tmp_path, monkeypatch):
    hooks_file = tmp_path / "webhooks.json"
    monkeypatch.setattr(webhooks, "_WEBHOOKS_FILE", hooks_file)

    with pytest.raises(ValueError, match="secret is required"):
        webhooks.create_webhook("build", "Build {{target}}")

    created = webhooks.create_webhook(
        "build",
        "Build {{target}}",
        secret="correct horse battery staple",
    )
    [stored] = json.loads(hooks_file.read_text())
    [public] = webhooks.public_webhooks()

    assert created["ok"] is True
    assert "secret" not in stored
    assert stored["secret_hash"].startswith("sha256:")
    assert webhooks.verify_secret(stored, "correct horse battery staple")
    assert not webhooks.verify_secret(stored, "wrong secret value")
    assert public["has_secret"] is True
    assert "secret" not in public and "secret_hash" not in public


def test_webhook_log_redacts_bounds_and_rotates(tmp_path, monkeypatch):
    log = tmp_path / "webhooks-log.jsonl"
    monkeypatch.setattr(webhooks, "_WEBHOOKS_LOG", log)
    monkeypatch.setattr(webhooks, "_LOG_MAX_BYTES", 700)
    monkeypatch.setattr(webhooks, "_LOG_BACKUPS", 2)

    for index in range(12):
        webhooks.append_log(
            "build",
            {
                "secret": "never-write-this-secret",
                "Authorization": "Bearer never-write-this-token",
                "safe": "x" * 180,
                "index": index,
            },
            {"ok": True, "output": "y" * 180},
        )

    paths = [log, log.with_name(log.name + ".1"), log.with_name(log.name + ".2")]
    content = b"".join(path.read_bytes() for path in paths if path.exists())
    assert b"never-write-this" not in content
    assert b"<redacted>" in content
    assert all(path.stat().st_size <= 700 for path in paths if path.exists())
    assert log.with_name(log.name + ".1").exists()
    assert len(webhooks.recent_runs(10_000)) <= 100


def test_once_schedule_runs_exactly_once_and_normalizes_timestamp():
    cadence = scheduler.validate_cadence(
        {"kind": "once", "at": "2026-07-15T12:30:00"}
    )
    schedule = {"enabled": True, "cadence": cadence}

    assert cadence == {"kind": "once", "at": "2026-07-15T12:30:00"}
    assert not scheduler._should_run_now(
        schedule,
        scheduler._dt.datetime(2026, 7, 15, 12, 29),
        None,
    )
    assert scheduler._should_run_now(
        schedule,
        scheduler._dt.datetime(2026, 7, 15, 12, 30),
        None,
    )
    assert not scheduler._should_run_now(
        schedule,
        scheduler._dt.datetime(2026, 7, 15, 12, 31),
        "2026-07-15T12:30:01",
    )


def test_natural_language_schedule_emits_scheduler_kind_schema():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "nl_schedule.js"
    harness = f"""
global.window = {{ Mio: {{}} }};
global.document = {{ readyState: 'loading', addEventListener() {{}} }};
eval(require('fs').readFileSync({json.dumps(str(asset))}, 'utf8'));
const api = window.Mio.nlSchedule;
console.log(JSON.stringify({{
  daily: api.toCadence(api.parse('every day at 8:30am stretch')),
  weekly: api.toCadence(api.parse('every Sunday at 9am plan')),
  once: api.toCadence(api.parse('in 30 minutes stretch')),
  unsupported: api.parse('every weekday at 8am plan'),
}}));
"""

    completed = subprocess.run(
        [node, "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["daily"] == {"kind": "daily", "hour": 8, "minute": 30}
    assert result["weekly"] == {
        "kind": "weekly",
        "weekday": 6,
        "hour": 9,
        "minute": 0,
    }
    assert result["once"]["kind"] == "once"
    assert set(result["once"]) == {"kind", "at"}
    assert result["unsupported"] is None
