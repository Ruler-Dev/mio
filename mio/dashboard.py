"""Live web dashboard with WebSocket metrics streaming."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

DASHBOARD_SCHEMA_VERSION = 1
_SUBSCRIBER_QUEUE_SIZE = 8


@dataclass
class RequestMetric:
    timestamp: float
    tier: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    generation_tps: float
    acceptance_length: float
    response_time_s: float
    validated: bool = False
    retries: int = 0


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[dict[str, Any]]


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


class MetricsCollector:
    """Collects and stores inference metrics for the dashboard."""

    def __init__(self, max_history: int = 100) -> None:
        self.history: deque[RequestMetric] = deque(maxlen=max_history)
        self._subscribers: dict[WebSocket, _Subscriber] = {}
        self._lock = threading.RLock()
        self._total_requests = 0
        self._sequence = 0

    def record(self, metric: RequestMetric) -> None:
        """Store and fan out one metric from any application thread."""
        with self._lock:
            self.history.append(metric)
            self._total_requests += 1
            self._sequence += 1
            data = self._snapshot_locked()
            subscribers = list(self._subscribers.items())
        for websocket, subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(
                    self._enqueue_if_subscribed,
                    websocket,
                    subscriber,
                    data,
                )
            except RuntimeError:
                self.unsubscribe(websocket)

    def record_generation(
        self,
        generation_metrics: Any,
        *,
        wall_s: float,
        tier: str,
        model: str | None = None,
        timestamp: float | None = None,
        validated: bool = False,
        retries: int = 0,
    ) -> RequestMetric:
        """Translate engine telemetry into the stable dashboard event schema."""
        metric = RequestMetric(
            timestamp=_finite_float(timestamp if timestamp is not None else time.time()),
            tier=str(tier or "unknown"),
            model=str(model or tier or "unknown"),
            prompt_tokens=_non_negative_int(
                getattr(generation_metrics, "prompt_tokens", 0)
            ),
            completion_tokens=_non_negative_int(
                getattr(generation_metrics, "completion_tokens", 0)
            ),
            generation_tps=_finite_float(
                getattr(generation_metrics, "generation_tps", 0.0)
            ),
            acceptance_length=_finite_float(
                getattr(generation_metrics, "avg_acceptance_length", 0.0)
            ),
            response_time_s=max(0.0, _finite_float(wall_s)),
            validated=bool(validated),
            retries=_non_negative_int(retries),
        )
        self.record(metric)
        return metric

    def _enqueue_if_subscribed(
        self,
        websocket: WebSocket,
        expected: _Subscriber,
        data: dict[str, Any],
    ) -> None:
        with self._lock:
            subscriber = self._subscribers.get(websocket)
        if subscriber is not expected:
            return
        if subscriber.queue.full():
            try:
                subscriber.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        subscriber.queue.put_nowait(data)

    def subscribe(self, ws: WebSocket) -> asyncio.Queue[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        subscriber = _Subscriber(
            loop=loop,
            queue=asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE),
        )
        with self._lock:
            self._subscribers[ws] = subscriber
        return subscriber.queue

    def unsubscribe(self, ws: WebSocket) -> None:
        with self._lock:
            self._subscribers.pop(ws, None)

    def _snapshot_locked(self) -> dict[str, Any]:
        recent = list(self.history)[-20:]
        return {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "type": "metrics.snapshot",
            "sequence": self._sequence,
            "total_requests": self._total_requests,
            "avg_tps": sum(m.generation_tps for m in recent) / max(len(recent), 1),
            "avg_acceptance": sum(m.acceptance_length for m in recent) / max(len(recent), 1),
            "recent": [
                {
                    "time": m.timestamp,
                    "tier": m.tier,
                    "model": m.model,
                    "tps": round(m.generation_tps, 1),
                    "tokens": m.completion_tokens,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "accept": round(m.acceptance_length, 1),
                    "time_s": round(m.response_time_s, 2),
                    "validated": m.validated,
                    "retries": m.retries,
                }
                for m in recent
            ],
        }

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def get_full_state(self, manager_status: dict | None = None) -> dict:
        snapshot = self._snapshot()
        if manager_status is not None:
            snapshot["server"] = manager_status
        return snapshot


# Global collector instance
collector = MetricsCollector()


def record_generation(
    generation_metrics: Any,
    *,
    wall_s: float,
    tier: str,
    model: str | None = None,
    timestamp: float | None = None,
    validated: bool = False,
    retries: int = 0,
) -> RequestMetric:
    """Bridge one completed generation into the process-wide collector."""
    return collector.record_generation(
        generation_metrics,
        wall_s=wall_s,
        tier=tier,
        model=model,
        timestamp=timestamp,
        validated=validated,
        retries=retries,
    )


async def websocket_metrics(
    websocket: WebSocket,
    *,
    manager_status: dict | None = None,
    manager_status_provider: Callable[[], dict | None] | None = None,
) -> None:
    """WebSocket handler for live metrics streaming."""
    from mio.web_security import reject_untrusted_websocket

    if await reject_untrusted_websocket(websocket):
        return
    await websocket.accept()
    updates = collector.subscribe(websocket)
    receive_task: asyncio.Task[str] | None = None
    update_task: asyncio.Task[dict[str, Any]] | None = None
    try:
        status = manager_status
        if manager_status_provider is not None:
            try:
                status = manager_status_provider()
            except Exception:
                status = None
        await websocket.send_text(json.dumps(collector.get_full_state(status)))
        receive_task = asyncio.create_task(websocket.receive_text())
        update_task = asyncio.create_task(updates.get())
        while True:
            done, _pending = await asyncio.wait(
                {receive_task, update_task},
                timeout=30,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await websocket.send_text(
                    json.dumps(
                        {
                            "schema_version": DASHBOARD_SCHEMA_VERSION,
                            "type": "metrics.heartbeat",
                            "heartbeat": True,
                        }
                    )
                )
                continue
            if update_task in done:
                await websocket.send_text(json.dumps(update_task.result()))
                update_task = asyncio.create_task(updates.get())
            if receive_task in done:
                receive_task.result()
                receive_task = asyncio.create_task(websocket.receive_text())
    except WebSocketDisconnect:
        pass
    finally:
        for task in (receive_task, update_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (receive_task, update_task):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        collector.unsubscribe(websocket)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mio Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0c0c14; color: #e0e0f0; font-family: -apple-system, system-ui, sans-serif; padding: 20px; }
  h1 { color: #00c8ff; font-size: 24px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1a1d2e; border-radius: 8px; padding: 16px; }
  .card .value { font-size: 32px; font-weight: bold; color: #00c8ff; }
  .card .label { font-size: 12px; color: #8890a5; margin-top: 4px; }
  .card.green .value { color: #00dc82; }
  .card.orange .value { color: #ffa028; }
  .card.purple .value { color: #825aff; }
  table { width: 100%; border-collapse: collapse; background: #1a1d2e; border-radius: 8px; overflow: hidden; }
  th { background: #1e3a5f; color: #00c8ff; padding: 10px 12px; text-align: left; font-size: 12px; }
  td { padding: 8px 12px; border-bottom: 1px solid #252840; font-size: 13px; }
  tr:nth-child(even) { background: #16192a; }
  .bar { height: 6px; background: #00c8ff; border-radius: 3px; transition: width 0.3s; }
  .bar-bg { height: 6px; background: #252840; border-radius: 3px; width: 100%; }
  #status { color: #00dc82; font-size: 12px; margin-bottom: 16px; }
  #status.offline { color: #ff4b4b; }
</style>
</head>
<body>
<h1>Mio Dashboard</h1>
<div id="status">Connecting...</div>

<div class="grid">
  <div class="card"><div class="value" id="total-req">0</div><div class="label">Total Requests</div></div>
  <div class="card green"><div class="value" id="avg-tps">0</div><div class="label">Avg tok/s</div></div>
  <div class="card orange"><div class="value" id="avg-accept">0</div><div class="label">Avg Acceptance</div></div>
  <div class="card purple"><div class="value" id="vram">0</div><div class="label">VRAM (GB)</div></div>
</div>

<h2 style="color:#8890a5;font-size:14px;margin-bottom:12px;">Recent Requests</h2>
<table>
  <thead><tr><th>Time</th><th>Tier</th><th>tok/s</th><th>Tokens</th><th>Accept</th><th>Response</th><th>Speed</th></tr></thead>
  <tbody id="history"></tbody>
</table>

<script>
let ws;
function csrfToken() {
  const prefix = 'mio_csrf=';
  for (const part of document.cookie.split(';')) {
    const item = part.trim();
    if (item.startsWith(prefix)) return decodeURIComponent(item.slice(prefix.length));
  }
  return '';
}
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const token = csrfToken();
  const protocols = token ? ['mio-ui', 'mio-csrf.' + token] : ['mio-ui'];
  ws = new WebSocket(`${protocol}//${location.host}/ws/metrics`, protocols);
  ws.onopen = () => { document.getElementById('status').textContent = 'Connected'; document.getElementById('status').className = ''; };
  ws.onclose = () => { document.getElementById('status').textContent = 'Disconnected'; document.getElementById('status').className = 'offline'; setTimeout(connect, 3000); };
  ws.onmessage = (e) => {
    let d;
    try {
      d = JSON.parse(e.data);
    } catch (_error) {
      document.getElementById('status').textContent = 'Invalid metrics event';
      document.getElementById('status').className = 'offline';
      return;
    }
    if (d.heartbeat) return;
    if (!d || d.schema_version !== 1 || d.type !== 'metrics.snapshot') return;
    document.getElementById('status').textContent = 'Connected';
    document.getElementById('status').className = '';
    document.getElementById('total-req').textContent = d.total_requests || 0;
    document.getElementById('avg-tps').textContent = (d.avg_tps || 0).toFixed(1);
    document.getElementById('avg-accept').textContent = (d.avg_acceptance || 0).toFixed(1);
    if (d.server) document.getElementById('vram').textContent = (d.server.vram_gb || 0).toFixed(1);
    const tbody = document.getElementById('history');
    tbody.replaceChildren();
    Array.from(d.recent || []).reverse().forEach(r => {
      const tr = document.createElement('tr');
      const maxTps = 120;
      const numericTps = Number.isFinite(Number(r.tps)) ? Number(r.tps) : 0;
      const pct = Math.max(0, Math.min(numericTps / maxTps * 100, 100));
      const values = [
        new Date(Number(r.time || 0) * 1000).toLocaleTimeString(),
        String(r.tier || 'unknown'),
        String(numericTps),
        String(Number(r.tokens || 0)),
        String(Number(r.accept || 0)),
        `${Number(r.time_s || 0)}s`,
      ];
      values.forEach(value => {
        const td = document.createElement('td');
        td.textContent = value;
        tr.appendChild(td);
      });
      const speedCell = document.createElement('td');
      const barBackground = document.createElement('div');
      barBackground.className = 'bar-bg';
      const bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.width = `${pct}%`;
      barBackground.appendChild(bar);
      speedCell.appendChild(barBackground);
      tr.appendChild(speedCell);
      tbody.appendChild(tr);
    });
  };
}
connect();
</script>
</body>
</html>"""


def get_dashboard_html() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD_HTML)
