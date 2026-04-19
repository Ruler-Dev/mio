"""Scheduled prompts — simple in-process scheduler that fires registered
prompts at configured cadences.

Schedules live in ~/.mio/schedules.json. Each entry:

  {
    "id": "uuid",
    "name": "Morning brief",
    "prompt": "What happened in AI yesterday?",
    "tier": "large-moe",
    "cadence": {"kind": "interval", "every_seconds": 3600}
                 | {"kind": "daily", "hour": 9, "minute": 0}
                 | {"kind": "weekly", "weekday": 0, "hour": 9, "minute": 0},
    "enabled": true,
    "created": "…",
    "last_run": "…" | null,
    "last_result": {...} | null,
  }

The scheduler runs a lightweight asyncio loop started by mount_webui().
Runs are persisted to ~/.mio/schedules-log.jsonl for history.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import time
import uuid
from pathlib import Path

PIMIO_DIR = Path.home() / ".mio"
PIMIO_DIR.mkdir(parents=True, exist_ok=True)
_SCHED_FILE = PIMIO_DIR / "schedules.json"
_SCHED_LOG = PIMIO_DIR / "schedules-log.jsonl"

_manager_ref = None  # set by init()
_task = None


def init(manager):
    """Inject the ModelManager and start the scheduler loop."""
    global _manager_ref, _task
    _manager_ref = manager
    if _task is None or _task.done():
        try:
            loop = asyncio.get_event_loop()
            _task = loop.create_task(_run_loop())
        except RuntimeError:
            pass  # no event loop yet — will be started by the server


def load_schedules() -> list[dict]:
    if not _SCHED_FILE.exists():
        return []
    try:
        return json.loads(_SCHED_FILE.read_text()) or []
    except Exception:
        return []


def save_schedules(items: list[dict]) -> None:
    _SCHED_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2))


def create_schedule(name: str, prompt: str, cadence: dict,
                    tier: str | None = None, enabled: bool = True) -> dict:
    items = load_schedules()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": name or "schedule",
        "prompt": prompt or "",
        "tier": tier,
        "cadence": cadence or {"kind": "daily", "hour": 9, "minute": 0},
        "enabled": bool(enabled),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "last_run": None,
        "last_result": None,
    }
    items.append(entry)
    save_schedules(items)
    return {"ok": True, "id": entry["id"], "schedule": entry}


def delete_schedule(sched_id: str) -> dict:
    items = [s for s in load_schedules() if s["id"] != sched_id]
    save_schedules(items)
    return {"ok": True}


def update_schedule(sched_id: str, updates: dict) -> dict:
    items = load_schedules()
    for s in items:
        if s["id"] == sched_id:
            s.update({k: v for k, v in (updates or {}).items() if k in
                      ("name", "prompt", "tier", "cadence", "enabled")})
            save_schedules(items)
            return {"ok": True, "schedule": s}
    return {"error": "not found"}


def recent_runs(limit: int = 50) -> list[dict]:
    if not _SCHED_LOG.exists():
        return []
    out = []
    try:
        with _SCHED_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out[-limit:][::-1]


def _append_log(entry: dict) -> None:
    try:
        with _SCHED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _should_run_now(schedule: dict, now: _dt.datetime, last_run_iso: str | None) -> bool:
    """Cadence evaluator. Returns True iff this schedule should fire now."""
    if not schedule.get("enabled"):
        return False
    cad = schedule.get("cadence") or {}
    kind = cad.get("kind")
    last_run = None
    if last_run_iso:
        try:
            last_run = _dt.datetime.fromisoformat(last_run_iso)
        except Exception:
            last_run = None
    if kind == "interval":
        every = int(cad.get("every_seconds") or 3600)
        if last_run is None:
            return True
        return (now - last_run).total_seconds() >= every
    if kind == "daily":
        hr = int(cad.get("hour", 9))
        mn = int(cad.get("minute", 0))
        target = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and last_run.date() == now.date():
            return False
        return now >= target
    if kind == "weekly":
        wd = int(cad.get("weekday", 0))  # Monday=0
        hr = int(cad.get("hour", 9))
        mn = int(cad.get("minute", 0))
        if now.weekday() != wd:
            return False
        target = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and (now - last_run).total_seconds() < 3600:
            return False
        return now >= target
    return False


async def _fire(schedule: dict) -> dict:
    """Run the prompt against the current model."""
    if _manager_ref is None:
        return {"error": "manager unset"}
    loaded = _manager_ref.loaded_tiers()
    if not loaded:
        return {"error": "no tiers loaded"}
    tier = schedule.get("tier") if schedule.get("tier") in loaded else loaded[0]
    engine = _manager_ref.get_engine(tier)
    parts: list[str] = []
    try:
        for chunk_text, _m in engine.generate_stream(
            [{"role": "user", "content": schedule["prompt"]}],
            max_tokens=1500,
        ):
            parts.append(chunk_text)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "tier": tier, "output": "".join(parts)}


async def _run_loop():
    """Main scheduler loop. Ticks every 30 seconds."""
    while True:
        try:
            await asyncio.sleep(30)
            now = _dt.datetime.now()
            items = load_schedules()
            dirty = False
            for s in items:
                if _should_run_now(s, now, s.get("last_run")):
                    result = await _fire(s)
                    s["last_run"] = now.isoformat(timespec="seconds")
                    s["last_result"] = result
                    _append_log({
                        "id": s["id"],
                        "name": s["name"],
                        "at": s["last_run"],
                        "result": result,
                    })
                    dirty = True
            if dirty:
                save_schedules(items)
        except Exception:
            # Loop must never die; swallow and continue
            await asyncio.sleep(5)
