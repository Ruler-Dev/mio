from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from mio import server
from mio.mcp import MCPRegistry, MCPServerConfig, MCPTransport
from mio.mcp import hub as hub_module
from mio.webui import scheduler


@pytest.mark.asyncio
async def test_scheduler_start_and_shutdown_are_idempotent(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def run_loop() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(scheduler, "_task", None)
    monkeypatch.setattr(scheduler, "_run_loop", run_loop)

    manager = object()
    first = scheduler.init(manager)
    second = scheduler.init(manager)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert first is second
    assert scheduler.is_running()

    await scheduler.shutdown()
    await scheduler.shutdown()

    assert stopped.is_set()
    assert not scheduler.is_running()


@pytest.mark.asyncio
async def test_scheduler_does_not_resurrect_a_schedule_deleted_during_fire(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(scheduler, "_SCHED_FILE", tmp_path / "schedules.json")
    monkeypatch.setattr(scheduler, "_SCHED_LOG", tmp_path / "schedules.jsonl")
    entry = scheduler.create_schedule(
        "slow",
        "prompt",
        {"kind": "interval", "every_seconds": 60},
    )["schedule"]
    started = asyncio.Event()
    release = asyncio.Event()

    async def fire(_schedule):
        started.set()
        await release.wait()
        return {"ok": True}

    monkeypatch.setattr(scheduler, "_fire", fire)
    task = asyncio.create_task(scheduler._run_due_schedules(scheduler._dt.datetime.now()))
    await started.wait()
    scheduler.delete_schedule(entry["id"])
    release.set()
    await task

    assert scheduler.load_schedules() == []


@pytest.mark.asyncio
async def test_scheduler_merges_result_without_overwriting_concurrent_edits(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(scheduler, "_SCHED_FILE", tmp_path / "schedules.json")
    monkeypatch.setattr(scheduler, "_SCHED_LOG", tmp_path / "schedules.jsonl")
    entry = scheduler.create_schedule(
        "before",
        "prompt",
        {"kind": "interval", "every_seconds": 60},
    )["schedule"]
    started = asyncio.Event()
    release = asyncio.Event()

    async def fire(_schedule):
        started.set()
        await release.wait()
        return {"ok": True, "output": "done"}

    monkeypatch.setattr(scheduler, "_fire", fire)
    task = asyncio.create_task(scheduler._run_due_schedules(scheduler._dt.datetime.now()))
    await started.wait()
    scheduler.update_schedule(entry["id"], {"name": "after", "enabled": False})
    release.set()
    await task

    [saved] = scheduler.load_schedules()
    assert saved["name"] == "after"
    assert saved["enabled"] is False
    assert saved["last_result"] == {"ok": True, "output": "done"}
    assert saved["last_run"]


@pytest.mark.parametrize(
    "cadence",
    [
        {"kind": "daily", "hour": 24, "minute": 0},
        {"kind": "weekly", "weekday": 7, "hour": 9, "minute": 0},
        {"kind": "interval", "every_seconds": "not-a-number"},
        {"kind": "unknown"},
        "daily",
    ],
)
def test_scheduler_rejects_invalid_cadence_without_persisting(
    cadence,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(scheduler, "_SCHED_FILE", tmp_path / "schedules.json")

    with pytest.raises(ValueError, match="cadence"):
        scheduler.create_schedule("invalid", "prompt", cadence)

    assert scheduler.load_schedules() == []


def test_scheduler_rejects_invalid_cadence_update_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_SCHED_FILE", tmp_path / "schedules.json")
    entry = scheduler.create_schedule(
        "valid",
        "prompt",
        {"kind": "daily", "hour": 9, "minute": 0},
    )["schedule"]

    with pytest.raises(ValueError, match="cadence hour"):
        scheduler.update_schedule(
            entry["id"],
            {"name": "must-not-apply", "cadence": {"kind": "daily", "hour": 99}},
        )

    [saved] = scheduler.load_schedules()
    assert saved["name"] == "valid"
    assert saved["cadence"] == {"kind": "daily", "hour": 9, "minute": 0}


@pytest.mark.asyncio
async def test_scheduler_isolates_malformed_persisted_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_SCHED_FILE", tmp_path / "schedules.json")
    monkeypatch.setattr(scheduler, "_SCHED_LOG", tmp_path / "schedules.jsonl")
    scheduler.save_schedules(
        [
            {
                "id": "bad",
                "name": "bad",
                "prompt": "bad",
                "enabled": True,
                "cadence": {"kind": "daily", "hour": 99},
                "last_run": None,
            },
            {
                "id": "good",
                "name": "good",
                "prompt": "good",
                "enabled": True,
                "cadence": {"kind": "interval", "every_seconds": 60},
                "last_run": None,
            },
        ]
    )
    fired: list[str] = []

    async def fire(item):
        fired.append(item["id"])
        return {"ok": True, "output": item["prompt"]}

    monkeypatch.setattr(scheduler, "_fire", fire)
    now = scheduler._dt.datetime(2026, 7, 15, 12, 0)
    await scheduler._run_due_schedules(now)

    assert fired == ["good"]
    saved = {item["id"]: item for item in scheduler.load_schedules()}
    assert saved["bad"]["last_run"] is None
    assert saved["good"]["last_result"] == {"ok": True, "output": "good"}
    assert any(run["id"] == "bad" and "error" in run["result"] for run in scheduler.recent_runs())


@pytest.mark.asyncio
async def test_scheduler_resolves_engine_after_acquiring_gpu_lock(monkeypatch):
    class Lock:
        active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, *_args):
            self.active = False

    lock = Lock()

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            assert lock.active
            yield "done", None

    class Manager:
        def loaded_tiers(self):
            assert lock.active
            return ["current"]

        def get_engine(self, tier):
            assert lock.active
            assert tier == "current"
            return Engine()

    monkeypatch.setattr(scheduler, "_manager_ref", Manager())
    monkeypatch.setattr(scheduler, "_gpu_lock_ref", lock)

    result = await scheduler._fire({"prompt": "scheduled", "tier": "stale"})

    assert result == {"ok": True, "tier": "current", "output": "done"}


def test_global_app_lifespan_restarts_services_for_each_testclient(monkeypatch):
    events: list[str] = []
    registry = MCPRegistry()
    manager = object()

    monkeypatch.setattr(server, "_manager", manager)
    monkeypatch.setattr(server, "_mcp_registry", registry)
    monkeypatch.setattr(server, "_webui_enabled", True)
    monkeypatch.setattr(
        "mio.mcp.configure_default_hub",
        lambda selected: events.append("mcp-start") or selected,
    )
    monkeypatch.setattr(
        "mio.mcp.close_default_hub",
        lambda: events.append("mcp-stop"),
    )
    monkeypatch.setattr(
        scheduler,
        "init",
        lambda selected, **kwargs: events.append("scheduler-start"),
    )

    async def stop_scheduler() -> None:
        events.append("scheduler-stop")

    monkeypatch.setattr(scheduler, "shutdown", stop_scheduler)

    with TestClient(server.app):
        pass
    with TestClient(server.app):
        pass

    assert events == [
        "mcp-start",
        "scheduler-start",
        "scheduler-stop",
        "mcp-stop",
        "mcp-start",
        "scheduler-start",
        "scheduler-stop",
        "mcp-stop",
    ]


def test_default_mcp_hub_closes_provider_and_can_be_recreated(monkeypatch):
    closed = 0

    class Provider:
        async def initialize(self):
            return {}

        async def list_tools(self):
            return {"tools": []}

        async def call_tool(self, _name, _arguments):
            return {}

        async def close(self):
            nonlocal closed
            closed += 1

    config = MCPServerConfig(
        name="local",
        transport=MCPTransport.STDIO,
        command=("unused",),
    )
    registry = MCPRegistry([config])
    created = hub_module.MCPHub(
        registry,
        provider_factory=lambda _name, _config, _grants: Provider(),
    )
    created.list_tools("local")
    monkeypatch.setattr(hub_module, "_default_hub", created)

    hub_module.close_default_hub()
    hub_module.close_default_hub()

    assert created.closed
    assert closed == 1
    assert hub_module._default_hub is None

    replacement = hub_module.configure_default_hub(registry)
    assert replacement is hub_module.configure_default_hub(registry)
    assert replacement is not created
    hub_module.close_default_hub()


def test_start_server_mounts_webui_router_and_cors_only_once(monkeypatch):
    class Manager:
        def loaded_tiers(self):
            return []

        def get_model_names(self):
            return []

    original_routes = list(server.app.router.routes)
    original_middleware = list(server.app.user_middleware)
    original_stack = server.app.middleware_stack
    state_names = (
        "_manager",
        "_router",
        "_tandem_enabled",
        "_mcp_registry",
        "_webui_enabled",
        "_webui_router_mounted",
        "_webui_cors_middleware_added",
    )
    original_state = {name: getattr(server, name) for name in state_names}

    monkeypatch.setattr(server, "_probe_server_bind", lambda _host, _port: None)
    monkeypatch.setattr(server, "_install_plain_line_printer", lambda: None)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mio.webui.router.mount_webui", lambda *_args, **_kwargs: None)
    server._webui_router_mounted = False
    server._webui_cors_middleware_added = False

    try:
        server.start_server(Manager(), port=19092, webui=True, live_panel=False)
        route_count = len(server.app.router.routes)
        middleware_count = len(server.app.user_middleware)

        server.start_server(Manager(), port=19092, webui=True, live_panel=False)

        assert len(server.app.router.routes) == route_count
        assert len(server.app.user_middleware) == middleware_count

        server.start_server(Manager(), port=19092, webui=False, live_panel=False)
        client = TestClient(server.app, base_url="http://127.0.0.1:19092")
        response = client.get("/ui")
        assert response.status_code == 404
        assert response.json()["detail"] == "Mio UI is disabled"
    finally:
        hub_module.close_default_hub()
        server.app.router.routes[:] = original_routes
        server.app.user_middleware[:] = original_middleware
        server.app.middleware_stack = original_stack
        for name, value in original_state.items():
            setattr(server, name, value)


def test_start_server_mounts_explicit_api_cors_without_webui(monkeypatch):
    class Manager:
        def loaded_tiers(self):
            return []

        def get_model_names(self):
            return []

    original_middleware = list(server.app.user_middleware)
    original_stack = server.app.middleware_stack
    state_names = (
        "_manager",
        "_router",
        "_tandem_enabled",
        "_mcp_registry",
        "_webui_enabled",
        "_webui_cors_middleware_added",
    )
    original_state = {name: getattr(server, name) for name in state_names}
    monkeypatch.setenv("MIO_CORS_ORIGINS", "https://allowed.example")
    monkeypatch.setattr(server, "_probe_server_bind", lambda _host, _port: None)
    monkeypatch.setattr(server, "_install_plain_line_printer", lambda: None)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    server._webui_cors_middleware_added = False

    try:
        server.start_server(
            Manager(),
            port=19094,
            webui=False,
            live_panel=False,
            mcp_registry=MCPRegistry([]),
        )
        client = TestClient(server.app, base_url="http://127.0.0.1:19094")
        preflight = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://allowed.example"
        assert server._webui_enabled is False
    finally:
        hub_module.close_default_hub()
        server.app.user_middleware[:] = original_middleware
        server.app.middleware_stack = original_stack
        for name, value in original_state.items():
            setattr(server, name, value)
