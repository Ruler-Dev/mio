"""Live dashboard metric bridge and WebSocket lifecycle regressions."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from mio import dashboard, server
from mio.engine import GenerationMetrics
from mio import web_security


def _metrics() -> GenerationMetrics:
    return GenerationMetrics(
        prompt_tokens=128,
        completion_tokens=32,
        generation_tps=41.5,
        avg_acceptance_length=2.75,
    )


def test_serve_stats_bridge_populates_non_zero_dashboard_snapshot(monkeypatch):
    live_collector = dashboard.MetricsCollector()
    monkeypatch.setattr(dashboard, "collector", live_collector)

    server._ServeStats().record(_metrics(), 1.25, "large", "answer")

    snapshot = live_collector.get_full_state({"vram_gb": 12.5})
    assert snapshot["schema_version"] == dashboard.DASHBOARD_SCHEMA_VERSION
    assert snapshot["type"] == "metrics.snapshot"
    assert snapshot["total_requests"] == 1
    assert snapshot["avg_tps"] == pytest.approx(41.5)
    assert snapshot["avg_acceptance"] == pytest.approx(2.75)
    assert snapshot["server"] == {"vram_gb": 12.5}
    assert snapshot["recent"] == [
        {
            "time": snapshot["recent"][0]["time"],
            "tier": "large",
            "model": "large",
            "tps": 41.5,
            "tokens": 32,
            "prompt_tokens": 128,
            "completion_tokens": 32,
            "accept": 2.8,
            "time_s": 1.25,
            "validated": False,
            "retries": 0,
        }
    ]


def test_dashboard_renders_metric_fields_without_html_injection():
    shell = dashboard.DASHBOARD_HTML
    assert "tbody.replaceChildren()" in shell
    assert "td.textContent = value" in shell
    assert "tr.innerHTML" not in shell
    assert "d.schema_version !== 1" in shell


@pytest.mark.asyncio
async def test_collector_fans_out_worker_thread_events_on_owner_loop():
    live_collector = dashboard.MetricsCollector()
    websocket = object()
    updates = live_collector.subscribe(websocket)  # type: ignore[arg-type]

    await asyncio.to_thread(
        live_collector.record_generation,
        _metrics(),
        wall_s=0.75,
        tier="large",
    )
    event = await asyncio.wait_for(updates.get(), timeout=1.0)
    live_collector.unsubscribe(websocket)  # type: ignore[arg-type]

    assert event["sequence"] == 1
    assert event["total_requests"] == 1
    assert event["recent"][0]["tps"] == 41.5


class _DashboardSocket:
    def __init__(self, *, disconnect_immediately: bool = False) -> None:
        self.accepted = asyncio.Event()
        self.disconnect = asyncio.Event()
        self.second_message = asyncio.Event()
        self.messages: list[dict] = []
        if disconnect_immediately:
            self.disconnect.set()

    async def accept(self) -> None:
        self.accepted.set()

    async def send_text(self, value: str) -> None:
        self.messages.append(json.loads(value))
        if len(self.messages) >= 2:
            self.second_message.set()

    async def receive_text(self) -> str:
        await self.disconnect.wait()
        raise WebSocketDisconnect(code=1000)


@pytest.mark.asyncio
async def test_websocket_streams_real_snapshot_and_reconnects_with_history(
    monkeypatch,
):
    live_collector = dashboard.MetricsCollector()
    monkeypatch.setattr(dashboard, "collector", live_collector)

    async def allow(_websocket) -> bool:
        return False

    monkeypatch.setattr(web_security, "reject_untrusted_websocket", allow)
    first = _DashboardSocket()
    first_task = asyncio.create_task(
        dashboard.websocket_metrics(
            first,  # type: ignore[arg-type]
            manager_status_provider=lambda: {"vram_gb": 9.5},
        )
    )
    await asyncio.wait_for(first.accepted.wait(), timeout=1.0)

    await asyncio.to_thread(
        live_collector.record_generation,
        _metrics(),
        wall_s=0.5,
        tier="large",
    )
    await asyncio.wait_for(first.second_message.wait(), timeout=1.0)
    first.disconnect.set()
    await asyncio.wait_for(first_task, timeout=1.0)

    assert first.messages[0]["total_requests"] == 0
    assert first.messages[0]["server"] == {"vram_gb": 9.5}
    assert first.messages[1]["total_requests"] == 1
    assert first.messages[1]["recent"][0]["completion_tokens"] == 32

    second = _DashboardSocket(disconnect_immediately=True)
    await asyncio.wait_for(
        dashboard.websocket_metrics(second),  # type: ignore[arg-type]
        timeout=1.0,
    )

    assert second.messages[0]["total_requests"] == 1
    assert second.messages[0]["sequence"] == 1
    assert second.messages[0]["recent"][0]["tps"] == 41.5
