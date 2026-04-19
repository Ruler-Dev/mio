"""Live web dashboard with WebSocket metrics streaming."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


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


class MetricsCollector:
    """Collects and stores inference metrics for the dashboard."""

    def __init__(self, max_history: int = 100) -> None:
        self.history: deque[RequestMetric] = deque(maxlen=max_history)
        self._subscribers: list[WebSocket] = []

    def record(self, metric: RequestMetric) -> None:
        self.history.append(metric)
        # Notify subscribers asynchronously
        data = json.dumps(self._snapshot())
        subscribers_snapshot = list(self._subscribers)
        for ws in subscribers_snapshot:
            try:
                asyncio.create_task(ws.send_text(data))
            except Exception:
                if ws in self._subscribers:
                    self._subscribers.remove(ws)

    def subscribe(self, ws: WebSocket) -> None:
        self._subscribers.append(ws)

    def unsubscribe(self, ws: WebSocket) -> None:
        if ws in self._subscribers:
            self._subscribers.remove(ws)

    def _snapshot(self) -> dict:
        recent = list(self.history)[-20:]
        return {
            "total_requests": len(self.history),
            "avg_tps": sum(m.generation_tps for m in recent) / max(len(recent), 1),
            "avg_acceptance": sum(m.acceptance_length for m in recent) / max(len(recent), 1),
            "recent": [
                {
                    "time": m.timestamp,
                    "tier": m.tier,
                    "tps": round(m.generation_tps, 1),
                    "tokens": m.completion_tokens,
                    "accept": round(m.acceptance_length, 1),
                    "time_s": round(m.response_time_s, 2),
                }
                for m in recent
            ],
        }

    def get_full_state(self, manager_status: dict | None = None) -> dict:
        snapshot = self._snapshot()
        if manager_status:
            snapshot["server"] = manager_status
        return snapshot


# Global collector instance
collector = MetricsCollector()


async def websocket_metrics(websocket: WebSocket) -> None:
    """WebSocket handler for live metrics streaming."""
    await websocket.accept()
    collector.subscribe(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps(collector.get_full_state()))
        # Keep alive and listen for pings
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"heartbeat": True}))
    except WebSocketDisconnect:
        pass
    finally:
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
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws/metrics`);
  ws.onopen = () => { document.getElementById('status').textContent = 'Connected'; document.getElementById('status').className = ''; };
  ws.onclose = () => { document.getElementById('status').textContent = 'Disconnected'; document.getElementById('status').className = 'offline'; setTimeout(connect, 3000); };
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.heartbeat) return;
    document.getElementById('total-req').textContent = d.total_requests || 0;
    document.getElementById('avg-tps').textContent = (d.avg_tps || 0).toFixed(1);
    document.getElementById('avg-accept').textContent = (d.avg_acceptance || 0).toFixed(1);
    if (d.server) document.getElementById('vram').textContent = (d.server.vram_gb || 0).toFixed(1);
    const tbody = document.getElementById('history');
    tbody.innerHTML = '';
    (d.recent || []).reverse().forEach(r => {
      const tr = document.createElement('tr');
      const maxTps = 120;
      const pct = Math.min(r.tps / maxTps * 100, 100);
      tr.innerHTML = `<td>${new Date(r.time*1000).toLocaleTimeString()}</td><td>${r.tier}</td><td>${r.tps}</td><td>${r.tokens}</td><td>${r.accept}</td><td>${r.time_s}s</td><td><div class="bar-bg"><div class="bar" style="width:${pct}%"></div></div></td>`;
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
