"""Scheduled prompts — simple in-process scheduler that fires registered
prompts at configured cadences.

Schedules live in ~/.mio/schedules.json. Each entry:

  {
    "id": "uuid",
    "name": "Morning brief",
    "prompt": "What happened in AI yesterday?",
    "tier": "large-moe",
    "cadence": {"kind": "once", "at": "2026-07-15T09:00:00"}
                 | {"kind": "interval", "every_seconds": 3600}
                 | {"kind": "daily", "hour": 9, "minute": 0}
                 | {"kind": "weekly", "weekday": 0, "hour": 9, "minute": 0},
    "enabled": true,
    "created": "…",
    "last_run": "…" | null,
    "last_result": {...} | null,
  }

The scheduler runs a lightweight asyncio loop owned by the FastAPI lifespan.
``init`` remains a compatibility entry point for UI callers, but task startup
and shutdown are both idempotent.
Runs are persisted to ~/.mio/schedules-log.jsonl for history.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import re
import uuid
from typing import Any

from mio.paths import mio_home
from mio.persistence import atomic_update_json

PIMIO_DIR = mio_home()
PIMIO_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
_SCHED_FILE = PIMIO_DIR / "schedules.json"
_SCHED_LOG = PIMIO_DIR / "schedules-log.jsonl"

_manager_ref = None  # set by configure()/init()
_gpu_lock_ref = None
_task: asyncio.Task[None] | None = None


class ScheduleStoreError(ValueError):
    """The persisted schedule envelope is structurally invalid."""


def _cadence_int(value, *, field: str, minimum: int, maximum: int) -> int:
    """Parse one cadence integer without silently truncating floats/bools."""
    if isinstance(value, bool):
        raise ValueError(f"cadence {field} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        parsed = int(value)
    else:
        raise ValueError(f"cadence {field} must be an integer")
    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"cadence {field} must be between {minimum} and {maximum}"
        )
    return parsed


def validate_cadence(cadence: dict | None) -> dict:
    """Return a canonical, bounded cadence or raise ``ValueError``.

    Persisting only canonical values keeps the background loop safe even when
    callers bypass the Web UI and use the scheduler API directly.
    """
    if cadence is None or cadence == {}:
        cadence = {"kind": "daily", "hour": 9, "minute": 0}
    if not isinstance(cadence, dict):
        raise ValueError("cadence must be an object")

    kind = cadence.get("kind")
    if kind == "once":
        at = cadence.get("at")
        if not isinstance(at, str) or not at or len(at) > 64:
            raise ValueError("cadence at must be an ISO-8601 timestamp")
        try:
            parsed = _dt.datetime.fromisoformat(at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("cadence at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return {"kind": "once", "at": parsed.isoformat(timespec="seconds")}

    if kind == "interval":
        every_seconds = _cadence_int(
            cadence.get("every_seconds", 3600),
            field="every_seconds",
            minimum=60,
            maximum=366 * 24 * 60 * 60,
        )
        return {"kind": "interval", "every_seconds": every_seconds}

    if kind == "daily":
        hour = _cadence_int(cadence.get("hour", 9), field="hour", minimum=0, maximum=23)
        minute = _cadence_int(
            cadence.get("minute", 0), field="minute", minimum=0, maximum=59
        )
        return {"kind": "daily", "hour": hour, "minute": minute}

    if kind == "weekly":
        weekday = _cadence_int(
            cadence.get("weekday", 0), field="weekday", minimum=0, maximum=6
        )
        hour = _cadence_int(cadence.get("hour", 9), field="hour", minimum=0, maximum=23)
        minute = _cadence_int(
            cadence.get("minute", 0), field="minute", minimum=0, maximum=59
        )
        return {
            "kind": "weekly",
            "weekday": weekday,
            "hour": hour,
            "minute": minute,
        }

    raise ValueError("cadence kind must be one of: once, interval, daily, weekly")


def configure(manager, gpu_lock=None) -> None:
    """Inject runtime dependencies without requiring an active event loop."""
    global _manager_ref, _gpu_lock_ref
    _manager_ref = manager
    if gpu_lock is not None:
        _gpu_lock_ref = gpu_lock


def start() -> asyncio.Task[None] | None:
    """Start the scheduler on the current loop, once.

    Returning ``None`` outside an async context lets synchronous UI mounting
    configure the runtime safely; FastAPI startup will perform the real start.
    """
    global _task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    if _task is not None and not _task.done():
        return _task
    _task = loop.create_task(_run_loop(), name="mio-scheduler")
    return _task


def init(manager, gpu_lock=None) -> asyncio.Task[None] | None:
    """Compatibility wrapper: configure dependencies and start if possible."""
    configure(manager, gpu_lock=gpu_lock)
    return start()


def is_running() -> bool:
    """Return whether the owned scheduler task is currently live."""
    return _task is not None and not _task.done()


async def _cancel_task(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def shutdown() -> None:
    """Cancel and await the scheduler task, safely and idempotently."""
    global _task
    task = _task
    _task = None
    if task is None or task.done():
        return

    task_loop = task.get_loop()
    current_loop = asyncio.get_running_loop()
    if task_loop is current_loop:
        await _cancel_task(task)
        return

    # This is mainly defensive for overlapping TestClient/event-loop owners.
    # Normal FastAPI shutdown executes on the same loop that started the task.
    if task_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_cancel_task(task), task_loop)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wrap_future(future)
    elif not task_loop.is_closed():
        # A dormant loop cannot drive the cancellation to completion, but
        # marking the task cancelled prevents it from being resumed later.
        with contextlib.suppress(RuntimeError):
            task.cancel()


def _validate_schedule_store(value: Any) -> list[dict]:
    """Validate the collection envelope while isolating bad cadences later."""
    if not isinstance(value, list):
        raise ValueError("schedules store must contain a JSON array")
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"schedule record {index} must be a JSON object")
        schedule_id = item.get("id")
        if not isinstance(schedule_id, str) or not schedule_id:
            raise ValueError(f"schedule record {index} must have a non-empty id")
        if schedule_id in seen:
            raise ValueError(f"schedules store contains duplicate id {schedule_id!r}")
        seen.add(schedule_id)
    return value


def load_schedules() -> list[dict]:
    try:
        value = json.loads(_SCHED_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    try:
        return _validate_schedule_store(value)
    except ValueError as exc:
        raise ScheduleStoreError(str(exc)) from exc


def _update_schedules(update) -> list[dict]:
    def transaction(current: Any) -> list[dict]:
        try:
            items = _validate_schedule_store(current)
        except ValueError as exc:
            raise ScheduleStoreError(str(exc)) from exc
        replacement = update([dict(item) for item in items])
        return _validate_schedule_store(replacement)

    return atomic_update_json(_SCHED_FILE, transaction, default_factory=list)


def save_schedules(items: list[dict]) -> None:
    replacement = _validate_schedule_store(items)
    _update_schedules(lambda _current: replacement)


def create_schedule(name: str, prompt: str, cadence: dict | None,
                    tier: str | None = None, enabled: bool = True) -> dict:
    cadence = validate_cadence(cadence)
    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": name or "schedule",
        "prompt": prompt or "",
        "tier": tier,
        "cadence": cadence,
        "enabled": bool(enabled),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "last_run": None,
        "last_result": None,
    }
    _update_schedules(lambda items: [*items, entry])
    return {"ok": True, "id": entry["id"], "schedule": entry}


def delete_schedule(sched_id: str) -> dict:
    _update_schedules(
        lambda items: [item for item in items if item.get("id") != sched_id]
    )
    return {"ok": True}


def update_schedule(sched_id: str, updates: dict) -> dict:
    updates = dict(updates or {})
    if "cadence" in updates:
        updates["cadence"] = validate_cadence(updates["cadence"])
    updated: dict[str, Any] = {}

    def apply(items: list[dict]) -> list[dict]:
        for schedule in items:
            if schedule.get("id") == sched_id:
                schedule.update(
                    {
                        key: value
                        for key, value in updates.items()
                        if key in ("name", "prompt", "tier", "cadence", "enabled")
                    }
                )
                updated["schedule"] = dict(schedule)
                break
        return items

    _update_schedules(apply)
    if "schedule" in updated:
        return {"ok": True, "schedule": updated["schedule"]}
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
    cad = validate_cadence(schedule.get("cadence"))
    kind = cad.get("kind")
    last_run = None
    if last_run_iso:
        try:
            last_run = _dt.datetime.fromisoformat(last_run_iso)
        except Exception:
            last_run = None
    if kind == "interval":
        every = cad["every_seconds"]
        if last_run is None:
            return True
        return (now - last_run).total_seconds() >= every
    if kind == "once":
        if last_run is not None:
            return False
        at = _dt.datetime.fromisoformat(cad["at"])
        return now >= at
    if kind == "daily":
        hr = cad["hour"]
        mn = cad["minute"]
        target = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and last_run.date() == now.date():
            return False
        return now >= target
    if kind == "weekly":
        wd = cad["weekday"]  # Monday=0
        hr = cad["hour"]
        mn = cad["minute"]
        if now.weekday() != wd:
            return False
        target = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
        if last_run and last_run.date() == now.date():
            return False
        return now >= target
    return False


async def _fire(schedule: dict) -> dict:
    """Run the prompt against the current model."""
    if _manager_ref is None:
        return {"error": "manager unset"}
    def generate() -> dict:
        parts: list[str] = []
        lock = _gpu_lock_ref
        with lock if lock is not None else contextlib.nullcontext():
            # Model load/unload uses this same lock. Resolve the live tier and
            # engine only after acquiring it so a concurrent switch cannot leave
            # the scheduler holding an engine that has already been unloaded.
            loaded = _manager_ref.loaded_tiers()
            if not loaded:
                return {"error": "no tiers loaded"}
            tier = schedule.get("tier") if schedule.get("tier") in loaded else loaded[0]
            engine = _manager_ref.get_engine(tier)
            for chunk_text, _m in engine.generate_stream(
                [{"role": "user", "content": schedule["prompt"]}],
                max_tokens=1500,
            ):
                parts.append(chunk_text)
        return {"ok": True, "tier": tier, "output": "".join(parts)}

    try:
        return await asyncio.to_thread(generate)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _run_due_schedules(now: _dt.datetime) -> None:
    """Fire one due snapshot, then merge results into current state by id.

    Generation yields to the event loop for a potentially long time. Reloading
    after each fire preserves edits/deletes made from the UI while the model
    was running instead of overwriting them with the stale pre-fire snapshot.
    """
    snapshot = load_schedules()
    for scheduled in snapshot:
        try:
            if not _should_run_now(scheduled, now, scheduled.get("last_run")):
                continue
            result = await _fire(dict(scheduled))
        except Exception as exc:
            # One corrupt persisted record or one unexpected worker failure must
            # never prevent independent schedules later in the snapshot firing.
            _append_log(
                {
                    "id": scheduled.get("id"),
                    "name": scheduled.get("name", "schedule"),
                    "at": now.isoformat(timespec="seconds"),
                    "result": {"error": f"{type(exc).__name__}: {exc}"},
                }
            )
            continue
        completed_at = now.isoformat(timespec="seconds")
        completed: dict[str, Any] = {}

        def merge_result(current: list[dict]) -> list[dict]:
            matched = next(
                (item for item in current if item.get("id") == scheduled.get("id")),
                None,
            )
            if matched is not None:
                matched["last_run"] = completed_at
                matched["last_result"] = result
                completed["schedule"] = dict(matched)
            return current

        _update_schedules(merge_result)
        matched = completed.get("schedule")
        if matched is None:
            # Deleted while running: record no new state and never resurrect.
            continue
        _append_log(
            {
                "id": matched["id"],
                "name": matched.get("name", scheduled.get("name", "schedule")),
                "at": completed_at,
                "result": result,
            }
        )


async def _run_loop():
    """Main scheduler loop. Ticks every 30 seconds."""
    while True:
        try:
            await asyncio.sleep(30)
            now = _dt.datetime.now()
            await _run_due_schedules(now)
        except Exception:
            # Loop must never die; swallow and continue
            await asyncio.sleep(5)
