"""Mio UI backend — FastAPI router for the web interface.

Serves the single-page UI and provides:
  - GET  /ui              → HTML page
  - WS   /ui/ws/chat      → streaming chat via WebSocket
  - GET  /ui/api/config    → current model/tier/caveman config
  - POST /ui/api/config    → update config
  - GET  /ui/api/sessions  → list saved chat sessions
  - GET  /ui/api/sessions/{id} → load a session
  - POST /ui/api/sessions  → save a session
  - DELETE /ui/api/sessions/{id} → delete a session
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from mio.paths import mio_home
from mio.persistence import atomic_update_json, atomic_write_bytes, atomic_write_json
from mio.prompt_policy import PromptMode, PromptPolicy, apply_prompt_policy
from mio.webui.image_proxy import ImageFetchError, fetch_image
from mio.webui.safe_files import (
    UnsafePathError,
    confined_markdown_tree,
    confined_path,
    iter_confined_regular_files,
    mio_state_directory,
    mio_state_root,
    open_binary_no_follow,
    read_text_no_follow,
    validate_directory,
    write_confined_bytes,
    write_confined_text,
)

router = APIRouter(prefix="/ui")

# --- Globals set by mount_webui() ---
_manager = None
_caveman_level = "full"
_prompt_policy = PromptPolicy()
_gpu_lock = threading.Lock()
_sessions_dir: Path | None = None
_system_prompt: str | None = None
_temperature: float = 0.0  # exact greedy DFlash default; positive values are explicit sampling
_max_tokens: int = 16384

_WS_STREAM_QUEUE_MAXSIZE = 16
_WS_STREAM_JOIN_TIMEOUT_SECONDS = 1.0

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_SHARED_ARTIFACT_BYTES = 5 * 1024 * 1024
_MAX_SHARED_ARTIFACTS = 128
_MAX_LOCAL_NOTE_BYTES = 2 * 1024 * 1024
_MAX_PROJECT_FILE_BYTES = 25 * 1024 * 1024
_FILE_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_WEBUI_COMPLETION_TOKENS = 32_768


def _validate_identifier(value: Any, *, label: str = "identifier") -> str:
    """Return a storage-safe identifier or reject it.

    Session and flow IDs become filenames.  Replacing suspicious characters is
    unsafe here because two attacker-controlled values can collapse to the same
    path; strict validation keeps the mapping one-to-one.
    """
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=f"invalid {label}: use 1-64 ASCII letters, digits, '-' or '_'",
        )
    return value


def _json_storage_path(root: Path | None, identifier: Any, *, label: str) -> Path:
    if root is None:
        raise HTTPException(status_code=503, detail=f"{label} storage is not initialized")
    safe = _validate_identifier(identifier, label=f"{label} id")
    try:
        root_path = validate_directory(root)
        return confined_path(
            root_path,
            f"{safe}.json",
            allow_nested=False,
        )
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=f"invalid {label} path") from exc


def _regular_files_confined(
    root: Path | None,
    pattern: str = "*.json",
    *,
    recursive: bool = False,
    max_files: int = 10_000,
) -> list[Path]:
    """Enumerate regular files without following symlinks at any path level."""
    if root is None:
        return []
    lexical_root = Path(root).expanduser()
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        return []
    resolved_root = lexical_root.resolve()
    iterator = lexical_root.rglob(pattern) if recursive else lexical_root.glob(pattern)
    files: list[Path] = []
    scanned = 0
    for candidate in iterator:
        scanned += 1
        if scanned > max(0, int(max_files)):
            break
        try:
            relative = candidate.relative_to(lexical_root)
            cursor = lexical_root
            linked = False
            for part in relative.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    linked = True
                    break
            if linked or not stat.S_ISREG(candidate.lstat().st_mode):
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError):
            continue
        files.append(resolved)
    return files


def _safe_upload_name(filename: str | None) -> str:
    """Normalize an untrusted browser filename without creating hidden files."""
    raw = unicodedata.normalize("NFKC", filename or "upload")
    # Treat Windows separators as path separators even on macOS/Linux.
    raw = raw.replace("\\", "/").rsplit("/", 1)[-1]
    raw = re.sub(r"[\x00-\x1f\x7f<>:\"|?*]", "_", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" .")
    if not raw:
        raw = "upload"
    if raw.startswith("."):
        raw = "upload-" + raw.lstrip(".")
    # Stay below common 255-byte filesystem limits, preserving the extension.
    raw = raw[:512]
    suffix = Path(raw).suffix[:24]
    stem = raw[: -len(suffix)] if suffix else raw
    while len((stem + suffix).encode("utf-8")) > 180 and stem:
        stem = stem[:-1]
    return (stem or "upload") + suffix


def _downloads_dir() -> Path:
    return Path.home() / "Downloads"


def _validated_downloads_dir(*, create: bool = False) -> Path:
    directory = _downloads_dir().expanduser()
    if directory.is_symlink():
        raise UnsafePathError("symlinked Downloads directory is not allowed")
    if create:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return validate_directory(directory)


def _write_unique_download(name: str, data: bytes) -> Path:
    """Create, but never overwrite, a file in Downloads."""
    if _safe_upload_name(name) != name:
        raise HTTPException(status_code=400, detail="invalid download filename")
    try:
        directory = _validated_downloads_dir(create=True)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested = Path(name)
    stem, suffix = requested.stem, requested.suffix
    for index in range(1000):
        candidate_name = name if index == 0 else f"{stem} ({index}){suffix}"
        try:
            candidate = confined_path(directory, candidate_name, allow_nested=False)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(candidate, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            return candidate
        except FileExistsError:
            continue
        except UnsafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError:
            if "candidate" in locals():
                candidate.unlink(missing_ok=True)
            raise
    raise HTTPException(status_code=409, detail="too many files with the same name")


def _json_for_inline_script(value: Any) -> str:
    """Serialize JSON that cannot terminate its containing script element."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _validate_webui_max_tokens(value: Any) -> int:
    """Accept only JSON integers within Mio's bounded WebUI generation cap."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_tokens must be an integer between 1 and 32768")
    if not 1 <= value <= _MAX_WEBUI_COMPLETION_TOKENS:
        raise ValueError("max_tokens must be an integer between 1 and 32768")
    return value


def mount_webui(
    manager,
    *,
    caveman_level: str = "full",
    prompt_policy: PromptPolicy | None = None,
    gpu_lock=None,
    sessions_dir: Path | None = None,
):
    """Initialize the webui module with server state."""
    global _manager, _caveman_level, _prompt_policy, _gpu_lock, _sessions_dir
    _manager = manager
    _prompt_policy = prompt_policy or PromptPolicy.resolve(caveman=caveman_level)
    _caveman_level = (
        _prompt_policy.level.value if _prompt_policy.mode is PromptMode.CAVEMAN else "off"
    )
    if gpu_lock is not None:
        _gpu_lock = gpu_lock
    _sessions_dir = sessions_dir or mio_home() / "sessions"
    _sessions_dir.mkdir(parents=True, exist_ok=True)
    # Start the scheduler (asyncio loop); safe if no loop is running yet
    from mio.webui import scheduler as _sched
    _sched.init(manager, gpu_lock=_gpu_lock)
    from mio.webui.flow_skills import configure_runtime as _configure_flow_skills

    _configure_flow_skills(manager, _gpu_lock)


def _resolve_chat_prompt_policy(data: dict) -> PromptPolicy:
    """Resolve per-request modern flags without legacy UI defaults erasing Ponytail."""
    if "prompt_mode" in data:
        return PromptPolicy.resolve(
            prompt_mode=data.get("prompt_mode"),
            prompt_level=data.get("prompt_level"),
        )
    if "caveman" in data and _prompt_policy.mode is not PromptMode.PONYTAIL:
        return PromptPolicy.resolve(caveman=str(data["caveman"]))
    return _prompt_policy


# --- UI page ---

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    # ``mount_webui`` may run before Uvicorn has installed an event loop.
    # Retrying here guarantees the scheduler is attached to the live loop.
    from mio.webui import scheduler as _sched

    _sched.init(_manager, gpu_lock=_gpu_lock)
    html_path = Path(__file__).parent / "mio_ui.html"
    # Prevent the browser from caching the shell — asset modules are
    # referenced by path and will pick up their own Cache-Control headers.
    return HTMLResponse(
        html_path.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache",
                 "Content-Security-Policy": (
                     "default-src 'self'; "
                     "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com https://esm.sh; "
                     "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                     "font-src 'self' data:; img-src 'self' data: blob:; "
                     "connect-src 'self' ws://127.0.0.1:* ws://localhost:* ws://[::1]:* "
                     "wss://127.0.0.1:* wss://localhost:* wss://[::1]:*; worker-src 'self' blob:; "
                     "frame-src 'self' blob: https://www.youtube.com https://www.youtube-nocookie.com; "
                     "object-src 'none'; base-uri 'none'; "
                     "frame-ancestors 'none'; form-action 'self'"
                 ),
                 "Referrer-Policy": "no-referrer",
                 "X-Content-Type-Options": "nosniff"},
    )


@router.get("/playground", response_class=HTMLResponse)
async def serve_playground():
    """Skill playground — lists every registered skill with a try-it form."""
    html_path = Path(__file__).parent / "assets" / "playground.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Playground not available</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


@router.get("/compare", response_class=HTMLResponse)
async def serve_compare():
    """Side-by-side model-compare UI."""
    html_path = Path(__file__).parent / "assets" / "compare.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Compare not available</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


@router.get("/stats", response_class=HTMLResponse)
async def serve_stats():
    html_path = Path(__file__).parent / "assets" / "stats.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Stats not available</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


@router.get("/attachments", response_class=HTMLResponse)
async def serve_attachments():
    html_path = Path(__file__).parent / "assets" / "attachments.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Attachments not available</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


@router.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Workspace dashboard: schedules, webhooks, indexed folders."""
    html_path = Path(__file__).parent / "assets" / "workspace.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not available</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


@router.get("/api/skills")
async def list_skills():
    """Return the full skill registry as JSON (name + description + schema)."""
    from mio.webui.skills import SKILLS
    from mio.web_security import webui_skill_operator_granted, webui_skill_risk

    result = []
    for name, spec in sorted(SKILLS.items()):
        result.append({
            "name": name,
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}),
            "risk": webui_skill_risk(name),
            "operator_granted": webui_skill_operator_granted(name),
        })
    return {"skills": result}


@router.post("/api/skills/run")
async def run_skill(body: dict, request: Request):
    """Execute a single skill directly for the playground. Not a normal
    chat path — just runs the skill function with the given JSON args.
    """
    from mio.webui.skills import SKILLS, execute_skill
    from mio.web_security import (
        DANGEROUS_ACTION_HEADER,
        webui_skill_direct_authorized,
        webui_skill_operator_granted,
        webui_skill_risk,
    )

    name = (body or {}).get("name", "")
    args = (body or {}).get("args", (body or {}).get("arguments", {})) or {}
    if name not in SKILLS:
        return {"error": f"unknown skill: {name}"}
    risk = webui_skill_risk(name)
    if not webui_skill_direct_authorized(
        name,
        confirmed=(body or {}).get("confirm_sensitive"),
        action_header=request.headers.get(DANGEROUS_ACTION_HEADER),
    ):
        reason = (
            "operator grant required: add the exact skill name to "
            "MIO_WEBUI_SKILL_GRANTS and restart Mio"
            if not webui_skill_operator_granted(name)
            else "explicit confirmation required for this invocation"
        )
        raise HTTPException(
            status_code=403,
            detail={"error": "sensitive_skill_denied", "skill": name, "risk": risk, "reason": reason},
        )
    try:
        result = await asyncio.to_thread(execute_skill, name, args)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --- Config API ---

# --- Prompts library (user-defined reusable prompts) ---
_prompts_file: Path | None = None


def _prompts_path() -> Path:
    global _prompts_file
    if _prompts_file is None:
        _prompts_file = mio_home() / "prompts.json"
    return _prompts_file


class PersistentStoreError(ValueError):
    """A persisted Web UI store is valid JSON with an invalid schema."""


def _raise_store_conflict(store: str, exc: BaseException) -> None:
    raise HTTPException(
        status_code=409,
        detail=f"stored {store} data is invalid; no changes were written",
    ) from exc


def _validate_record_store(value: Any, *, store: str, identity: str) -> list[dict]:
    """Validate a persisted Web UI collection without repairing corruption.

    Legacy records may omit optional presentation fields, but the envelope and
    stable identity are required for every mutation.  Raising on malformed
    state is intentional: silently treating corruption as an empty collection
    would let the next write destroy the only recoverable copy.
    """
    if not isinstance(value, list):
        raise ValueError(f"{store} store must contain a JSON array")
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{store} record {index} must be a JSON object")
        item_id = item.get(identity)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(
                f"{store} record {index} must have a non-empty {identity}"
            )
        if item_id in seen:
            raise ValueError(f"{store} store contains duplicate {identity} {item_id!r}")
        seen.add(item_id)
    return value


def _load_record_store(path: Path, *, store: str, identity: str) -> list[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    try:
        return _validate_record_store(value, store=store, identity=identity)
    except ValueError as exc:
        raise PersistentStoreError(str(exc)) from exc


def _update_record_store(
    path: Path,
    update,
    *,
    store: str,
    identity: str,
) -> list[dict]:
    def transaction(current: Any) -> list[dict]:
        try:
            records = _validate_record_store(current, store=store, identity=identity)
        except ValueError as exc:
            raise PersistentStoreError(str(exc)) from exc
        replacement = update([dict(item) for item in records])
        return _validate_record_store(replacement, store=store, identity=identity)

    return atomic_update_json(path, transaction, default_factory=list)


def _load_prompts() -> list:
    return _load_record_store(_prompts_path(), store="prompts", identity="id")


def _save_prompts(prompts: list) -> None:
    replacement = _validate_record_store(prompts, store="prompts", identity="id")
    _update_record_store(
        _prompts_path(),
        lambda _current: replacement,
        store="prompts",
        identity="id",
    )


def _update_prompts(update) -> list[dict]:
    return _update_record_store(
        _prompts_path(), update, store="prompts", identity="id"
    )


@router.get("/api/prompts")
async def list_prompts():
    try:
        return {"prompts": _load_prompts()}
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("prompts", exc)


@router.post("/api/prompts")
async def save_prompt(body: dict):
    pid = str(body.get("id") or str(uuid.uuid4())[:8])
    entry = {
        "id": pid,
        "name": body.get("name", "Untitled"),
        "body": body.get("body", ""),
        "slash": body.get("slash", ""),  # optional custom /slash keyword
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    def upsert(prompts: list[dict]) -> list[dict]:
        prompts = [prompt for prompt in prompts if prompt.get("id") != pid]
        prompts.append(entry)
        return prompts

    try:
        _update_prompts(upsert)
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("prompts", exc)
    return entry


@router.delete("/api/prompts/{pid}")
async def delete_prompt(pid: str):
    try:
        _update_prompts(
            lambda prompts: [prompt for prompt in prompts if prompt.get("id") != pid]
        )
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("prompts", exc)
    return {"ok": True}


# --- Persistent memory (facts the model remembers across chats) ---
def _memory_path() -> Path:
    return mio_home() / "memory.json"


def _load_memory() -> list:
    return _load_record_store(_memory_path(), store="memory", identity="id")


def _save_memory(mem: list) -> None:
    replacement = _validate_record_store(mem, store="memory", identity="id")
    _update_record_store(
        _memory_path(),
        lambda _current: replacement,
        store="memory",
        identity="id",
    )


def _update_memory(update) -> list[dict]:
    return _update_record_store(
        _memory_path(), update, store="memory", identity="id"
    )


@router.get("/api/memory")
async def list_memory():
    try:
        return {"memory": _load_memory()}
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("memory", exc)


@router.post("/api/memory")
async def add_memory(body: dict):
    mid = str(body.get("id") or str(uuid.uuid4())[:8])
    entry = {"id": mid, "text": body.get("text", ""), "added": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def upsert(mem: list[dict]) -> list[dict]:
        mem = [item for item in mem if item.get("id") != mid]
        mem.append(entry)
        return mem

    try:
        _update_memory(upsert)
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("memory", exc)
    return entry


@router.delete("/api/memory/{mid}")
async def delete_memory(mid: str):
    try:
        _update_memory(lambda mem: [item for item in mem if item.get("id") != mid])
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("memory", exc)
    return {"ok": True}


# --- Projects (shared system prompt + knowledge base per project) ---
def _projects_path() -> Path:
    return mio_home() / "projects.json"


def _load_projects() -> list:
    return _load_record_store(_projects_path(), store="projects", identity="id")


def _save_projects(projs: list) -> None:
    replacement = _validate_record_store(projs, store="projects", identity="id")
    _update_record_store(
        _projects_path(),
        lambda _current: replacement,
        store="projects",
        identity="id",
    )


def _update_projects(update) -> list[dict]:
    return _update_record_store(
        _projects_path(), update, store="projects", identity="id"
    )


@router.get("/api/projects")
async def list_projects():
    try:
        return {"projects": _load_projects()}
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("projects", exc)


@router.post("/api/projects")
async def save_project(body: dict):
    pid = _validate_identifier(body.get("id") or str(uuid.uuid4())[:8], label="project id")
    raw_files = body.get("files", [])
    files = []
    if isinstance(raw_files, list):
        for filename in raw_files[:64]:
            if not isinstance(filename, str):
                continue
            if _safe_upload_name(filename) == filename and "/" not in filename and "\\" not in filename:
                files.append(filename)
    entry = {
        "id": pid,
        "name": body.get("name", "Untitled"),
        "description": body.get("description", ""),
        "system_prompt": body.get("system_prompt", ""),
        "color": body.get("color", "#3b82f6"),
        "icon": body.get("icon", ""),           # optional emoji/icon
        "files": files,                           # validated filenames in ~/Downloads
        # Optional per-workspace model / context overrides. Unset = use
        # global defaults. The UI reads these to pre-apply a tier before
        # sending the first message of a session in that workspace.
        "tier": body.get("tier") or None,
        "context_window": body.get("context_window") or None,
        "caveman_level": body.get("caveman_level") or None,
        "pinned_prompts": body.get("pinned_prompts", []),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    def upsert(projects: list[dict]) -> list[dict]:
        projects = [project for project in projects if project.get("id") != pid]
        projects.append(entry)
        return projects

    try:
        _update_projects(upsert)
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("projects", exc)
    return entry


@router.delete("/api/projects/{pid}")
async def delete_project(pid: str):
    try:
        _update_projects(
            lambda projects: [project for project in projects if project.get("id") != pid]
        )
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("projects", exc)
    return {"ok": True}


@router.get("/api/projects/{pid}")
async def get_project(pid: str):
    try:
        projects = _load_projects()
    except (json.JSONDecodeError, PersistentStoreError) as exc:
        _raise_store_conflict("projects", exc)
    for p in projects:
        if p.get("id") == pid:
            return p
    return {"error": "not found"}


# Cache of the last effective system prompt so the UI can verify what was
# actually sent to the model (caveman, memory, style, project, etc.)
_last_system_prompt: str | None = None


@router.get("/api/debug/last-prompt")
async def debug_last_prompt():
    return {"system_prompt": _last_system_prompt or "(no chat sent yet this session)"}


@router.get("/api/config")
async def get_config():
    tiers = _manager.loaded_tiers() if _manager else []
    all_tiers = list(_manager.config.tiers.keys()) if _manager else []
    return {
        "loaded_tiers": tiers,
        "all_tiers": all_tiers,
        "caveman": _caveman_level,
        "prompt_mode": _prompt_policy.mode.value,
        "prompt_level": _prompt_policy.level.value if _prompt_policy.mode is not PromptMode.NONE else None,
        "prompt_policy": _prompt_policy.label,
        "active_tier": tiers[0] if tiers else None,
        "system_prompt": _system_prompt or "",
        "temperature": _temperature,
        "max_tokens": _max_tokens,
    }


@router.post("/api/config")
async def update_config(body: dict):
    global _caveman_level, _prompt_policy, _system_prompt, _temperature, _max_tokens
    candidate_policy = _prompt_policy
    try:
        if "prompt_mode" in body:
            candidate_policy = PromptPolicy.resolve(
                prompt_mode=body.get("prompt_mode"),
                prompt_level=body.get("prompt_level"),
            )
        elif "caveman" in body:
            candidate_policy = PromptPolicy.resolve(caveman=str(body["caveman"]))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candidate_system_prompt = _system_prompt
    if "system_prompt" in body:
        candidate_system_prompt = body["system_prompt"] or None
    candidate_temperature = _temperature
    if "temperature" in body:
        try:
            candidate_temperature = float(body["temperature"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="temperature must be numeric") from exc
        if not math.isfinite(candidate_temperature) or not 0.0 <= candidate_temperature <= 2.0:
            raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
    candidate_max_tokens = _max_tokens
    if "max_tokens" in body:
        try:
            candidate_max_tokens = _validate_webui_max_tokens(body["max_tokens"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Publish only after every supplied setting validates, avoiding partial
    # mutation when one field in a multi-setting request is malformed.
    _prompt_policy = candidate_policy
    _caveman_level = (
        candidate_policy.level.value
        if candidate_policy.mode is PromptMode.CAVEMAN
        else "off"
    )
    _system_prompt = candidate_system_prompt
    _temperature = candidate_temperature
    _max_tokens = candidate_max_tokens
    return {
        "ok": True,
        "caveman": _caveman_level,
        "prompt_mode": _prompt_policy.mode.value,
        "prompt_level": _prompt_policy.level.value if _prompt_policy.mode is not PromptMode.NONE else None,
        "prompt_policy": _prompt_policy.label,
        "system_prompt": _system_prompt or "",
        "temperature": _temperature,
        "max_tokens": _max_tokens,
    }


# --- Model info ---

@router.get("/api/model-info")
async def get_model_info():
    if not _manager:
        return {"error": "no manager"}
    loaded = _manager.loaded_tiers()
    if not loaded:
        return {"error": "no tiers loaded"}
    active = loaded[0]
    engine = _manager.get_engine(active)
    tc = engine.tier_config
    m = engine.last_metrics
    model_name = tc.target_model.rsplit("/", 1)[-1]
    return {
        "tier": active,
        "model_name": model_name,
        "context_window": tc.context_window,
        "max_output_tokens": tc.max_output_tokens,
        "pq_bits": getattr(tc, "pq_bits", 16),
        "temperature": tc.temperature,
        "vram_gb": round(_manager.total_vram_gb(), 1) if hasattr(_manager, "total_vram_gb") else 0,
        "last_prompt_tokens": m.prompt_tokens,
        "last_gen_tps": round(m.generation_tps, 1),
    }


# --- Tier switching ---

@router.post("/api/tier")
async def switch_tier(body: dict):
    tier = body.get("tier")
    if not tier or not _manager:
        return {"error": "invalid request"}
    if tier not in _manager.config.tiers:
        return {"error": f"unknown tier: {tier}", "available": list(_manager.config.tiers.keys())}
    manager = _manager

    def _switch() -> bool:
        with _gpu_lock:
            # Snapshot only after taking the same lifecycle lock used by
            # generation and the model endpoints.  Two concurrent switch
            # requests can therefore never act on the same stale tier list.
            loaded = manager.loaded_tiers()
            already_loaded = tier in loaded

            # Loading is the transactional prepare step.  ``ModelManager``
            # publishes an engine only after ``engine.load()`` succeeds, so a
            # load failure leaves every currently serving tier untouched.
            if not already_loaded:
                manager.load_tier(tier)
            for loaded_tier in loaded:
                if loaded_tier != tier:
                    manager.unload_tier(loaded_tier)
            return already_loaded

    try:
        already_loaded = await asyncio.to_thread(_switch)
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"could not switch to tier '{tier}': {exc}",
        ) from exc
    return {"ok": True, "tier": tier, "already_loaded": already_loaded}


# --- Session persistence ---

@router.get("/api/sessions")
async def list_sessions(project_id: str | None = None):
    if not _sessions_dir or not _sessions_dir.exists():
        return {"sessions": []}
    sessions = []
    for f in sorted(
        _regular_files_confined(_sessions_dir),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            meta = json.loads(f.read_text())
            pid = meta.get("project_id")
            if project_id is not None and pid != project_id:
                # When filtering by project_id="" (no project), match sessions without one
                if not (project_id == "" and not pid):
                    continue
            sessions.append({
                "id": f.stem,
                "title": meta.get("title", "Untitled"),
                "updated": meta.get("updated", ""),
                "message_count": len(meta.get("messages", [])),
                "project_id": pid,
                "tags": meta.get("tags", []),
            })
        except Exception:
            continue
    return {"sessions": sessions}


@router.get("/api/sessions/{session_id}")
async def load_session(session_id: str):
    path = _json_storage_path(_sessions_dir, session_id, label="session")
    if not path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="session data is unreadable") from exc


@router.post("/api/sessions")
async def save_session(body: dict):
    payload = dict(body or {})
    session_id = payload.get("id") or str(uuid.uuid4())[:8]
    path = _json_storage_path(_sessions_dir, session_id, label="session")
    payload["id"] = session_id
    payload["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not payload.get("title"):
        messages = payload.get("messages", [])
        first_user = next(
            (
                str(message.get("content", ""))[:60]
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "New Chat",
        )
        payload["title"] = first_user or "New Chat"
    # project_id optional — sessions can be grouped under a project
    atomic_write_json(path, payload)
    return {"id": session_id, "title": payload["title"], "project_id": payload.get("project_id")}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = _json_storage_path(_sessions_dir, session_id, label="session")
    if path.exists():
        path.unlink()
    return {"ok": True}


# --- Generated file serving ---
# Skills like generate_pdf, generate_docx, generate_xlsx, generate_pptx,
# generate_chart save outputs to ~/Downloads. The UI needs HTTP access so
# it can embed the PDF in an iframe or offer download links. Restricted to
# plain filenames (no path traversal) from that directory only.

_ALLOWED_EXT = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".svg",
    ".webp", ".gif", ".csv", ".ics", ".sqlite", ".db", ".md", ".txt", ".html",
}


# Artifact share: store a handful of artifacts keyed by short IDs so they
# can be viewed read-only at /ui/share/<id>. Kept in-memory — deliberate
# ephemerality so private artifacts don't leak across restarts.
_shared_artifacts: dict[str, dict] = {}


@router.post("/api/share")
async def share_artifact(body: dict):
    art_id = _validate_identifier(
        body.get("identifier") or str(uuid.uuid4())[:8], label="artifact id"
    )
    content = str(body.get("content", ""))
    if len(content.encode("utf-8")) > _MAX_SHARED_ARTIFACT_BYTES:
        raise HTTPException(status_code=413, detail="artifact exceeds the 5 MiB limit")
    if art_id not in _shared_artifacts and len(_shared_artifacts) >= _MAX_SHARED_ARTIFACTS:
        _shared_artifacts.pop(next(iter(_shared_artifacts)))
    _shared_artifacts[art_id] = {
        "type": str(body.get("type", "text/html"))[:160],
        "title": str(body.get("title", "Artifact"))[:500],
        "content": content,
        "language": body.get("language", ""),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"id": art_id, "url": f"/ui/share/{art_id}"}


@router.get("/share/{art_id}", response_class=HTMLResponse)
async def view_shared_artifact(art_id: str):
    art_id = _validate_identifier(art_id, label="artifact id")
    art = _shared_artifacts.get(art_id)
    if not art:
        return HTMLResponse(status_code=404, content="Artifact not found or expired.")
    import html as _html
    j = _json_for_inline_script(art)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{_html.escape(art['title'])}</title>
<style>body{{margin:0;font-family:-apple-system,sans-serif;background:#111;color:#eee}}
.hdr{{padding:10px 16px;border-bottom:1px solid #333;font-size:13px;display:flex;gap:12px;align-items:center}}
.hdr .t{{font-weight:500;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.hdr .d{{color:#888;font-family:monospace;font-size:11px}}
iframe,pre{{width:100vw;height:calc(100vh - 42px);border:0;margin:0}}
pre{{background:#1a1a1a;color:#eee;padding:16px;box-sizing:border-box;overflow:auto}}</style></head>
<body><div class="hdr"><div class="t">{_html.escape(art['title'])}</div><div class="d">{art['created']}</div></div>
<div id="mount"></div>
<script>const art = {j};
const mount = document.getElementById('mount');
if (art.type === 'text/html' || art.type.startsWith('application/vnd.pimio.') || art.type.startsWith('application/vnd.ant.')) {{
  const f = document.createElement('iframe');
  f.sandbox = 'allow-scripts';
  f.srcdoc = art.content;
  mount.appendChild(f);
}} else if (art.type === 'image/svg+xml') {{
  const f = document.createElement('iframe');
  f.sandbox = '';
  f.srcdoc = art.content;
  mount.appendChild(f);
}} else {{
  const p = document.createElement('pre');
  p.textContent = art.content;
  mount.appendChild(p);
}}</script></body></html>"""
    return HTMLResponse(page)


@router.post("/api/upload")
async def upload_attachment(file: UploadFile = File(...)):
    """Accept an attachment, store it in ~/Downloads, extract text if PDF/txt/md,
    and return a summary the client can inject into the next message.
    """
    name = _safe_upload_name(file.filename)
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload exceeds the 25 MiB limit")

    chunks = bytearray()
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        if len(chunks) + len(chunk) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="upload exceeds the 25 MiB limit")
        chunks.extend(chunk)
    data = bytes(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    out = _write_unique_download(name, data)
    name = out.name

    ext = ("." + name.rsplit(".", 1)[-1]).lower() if "." in name else ""
    text = ""
    note = ""
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(str(out)) as pdf:
                parts = []
                for p in pdf.pages[:40]:
                    t = p.extract_text() or ""
                    if t.strip():
                        parts.append(t)
                text = "\n\n".join(parts)[:20000]
            note = f"PDF extracted, {len(pdf.pages)} pages"
        except Exception as e:
            note = f"PDF extract failed: {e}"
    elif ext in (".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".ts"):
        try:
            text = data.decode("utf-8", errors="replace")[:20000]
            note = f"Text file, {len(text)} chars"
        except Exception:
            pass
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        # Image attached — MLX Qwen 3.6 VL tiers accept images in multimodal
        # messages. Also try a best-effort OCR pass so text-only codepaths
        # still surface readable content.
        note = "Image attached — sent to VL model as inline image"
        try:
            import pytesseract  # type: ignore
            from PIL import Image as _PILImage  # type: ignore
            img = _PILImage.open(str(out))
            ocr = (pytesseract.image_to_string(img) or "").strip()
            if ocr:
                text = ocr[:8000]
                note += f"; OCR extracted {len(ocr)} chars"
        except Exception:
            pass

    result = {
        "filename": name,
        "path": str(out),
        "size": len(data),
        "ext": ext,
        "extracted_text": text,
        "note": note,
    }
    # Images return a URL the UI can render as a thumbnail
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"):
        result["url"] = "/ui/files/" + quote(name)
        result["is_image"] = True
    return result


# --- Browser-extension ingest: stash web content into Mio ---
@router.post("/api/ingest")
async def ingest_from_browser(body: dict):
    """Receive a payload from the Mio browser extension.

    Expected JSON body:
        {
          "url":       "https://…",            // required
          "title":     "Page title",           // optional
          "text":      "readable plain text",  // required (readability-extracted)
          "html":      "<article>…</article>",  // optional raw HTML
          "selection": "highlighted text only",// optional
          "tags":      ["optional", "tags"],   // optional
          "target":    "rag"|"attach"|"chat"   // how Mio should handle it
        }

    Saves the document under ~/.mio/ingest/ as a timestamped markdown file,
    adds it to the local RAG index (if SQLite FTS5 is initialised), and
    returns { id, path, summary } so the extension can link to it and the
    chat UI can @-mention it.
    """
    import datetime as _dt
    url = (body or {}).get("url") or ""
    if not url:
        return {"error": "url required"}
    title = (body.get("title") or "").strip() or url
    selection = (body.get("selection") or "").strip()
    text = (body.get("text") or "").strip()
    effective_text = selection or text
    if not effective_text:
        return {"error": "text or selection required"}

    tags = body.get("tags") or []
    target = (body.get("target") or "rag").strip().lower()

    try:
        root = mio_state_directory("ingest", create=True)
    except UnsafePathError as exc:
        return {"error": str(exc)}
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = hashlib.sha1(url.encode()).hexdigest()[:8]
    filename = f"{stamp}-{slug}.md"

    frontmatter = (
        "---\n"
        f"title: {title!r}\n"
        f"source: {url}\n"
        f"fetched: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"tags: {tags}\n"
        f"selection_only: {bool(selection)}\n"
        "---\n\n"
    )
    try:
        md_path = write_confined_text(
            root,
            filename,
            frontmatter + f"# {title}\n\n" + effective_text,
        )
    except (OSError, UnsafePathError) as exc:
        return {"error": str(exc)}

    summary = effective_text[:280].replace("\n", " ")
    doc_id = f"{stamp}-{slug}"

    # Add to RAG index if available
    indexed = False
    if target in ("rag", ""):
        try:
            from mio.webui.skills_rag import index_folder
            index_result = index_folder(str(root))
            indexed = not bool(index_result.get("error"))
        except Exception:
            pass

    return {
        "id":       doc_id,
        "path":     str(md_path),
        "url":      url,
        "title":    title,
        "summary":  summary,
        "chars":    len(effective_text),
        "tags":     tags,
        "indexed":  indexed,
        "target":   target,
    }


@router.get("/api/ingest")
async def list_ingested(limit: int = 100, tag: str | None = None):
    """List everything the browser extension has stashed, newest first.

    Each item includes parsed YAML front-matter fields (title, source,
    tags, fetched). Filter by `tag` query param to see only items with
    that tag.
    """
    import re as _re

    try:
        root = mio_state_directory("ingest")
    except UnsafePathError:
        return {"items": [], "tags": []}
    items = []
    all_tags: set[str] = set()
    paths = iter_confined_regular_files(
        root,
        suffixes={".md"},
        recursive=False,
        max_bytes=8 * 1024 * 1024,
    )
    for p in sorted(paths, reverse=True):
        try:
            head = read_text_no_follow(p, max_bytes=8 * 1024 * 1024)[:2048]
        except (OSError, UnsafePathError):
            continue
        title = url = ""
        tags: list[str] = []
        m = _re.search(r"^title:\s*['\"]?(.*?)['\"]?$", head, _re.MULTILINE)
        if m:
            title = m.group(1)
        m = _re.search(r"^source:\s*(.*?)$", head, _re.MULTILINE)
        if m:
            url = m.group(1)
        m = _re.search(r"^tags:\s*(\[.*?\])$", head, _re.MULTILINE)
        if m:
            try:
                raw = m.group(1).strip("[]")
                tags = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
            except Exception:
                tags = []
        for t in tags:
            all_tags.add(t)
        if tag and tag not in tags:
            continue
        items.append({
            "id":    p.stem,
            "path":  str(p),
            "title": title or p.stem,
            "url":   url,
            "tags":  tags,
            "size":  p.stat().st_size,
            "mtime": p.stat().st_mtime,
        })
        if len(items) >= limit:
            break
    return {"items": items, "tags": sorted(all_tags)}


# --- RAG index management ---
@router.get("/api/rag/indexes")
async def rag_list_indexes():
    """List all indexed folders with file counts + timestamps."""
    try:
        from mio.webui.skills_rag import list_indexes
        r = list_indexes()
        return {"indexes": r.get("indexes", [])}
    except Exception as e:
        return {"indexes": [], "error": str(e)}


@router.post("/api/rag/index")
async def rag_add_index(body: dict):
    """Index a folder path for full-text search. Body: {path, label?}."""
    path = (body or {}).get("path", "").strip()
    label = (body or {}).get("label") or None
    if not path:
        return {"error": "path required"}
    from pathlib import Path as _P
    if not _P(path).expanduser().exists():
        return {"error": f"path does not exist: {path}"}
    try:
        from mio.webui.skills_rag import index_folder
        return index_folder(str(_P(path).expanduser()), label=label, replace=True)
    except Exception as e:
        return {"error": str(e)}


@router.delete("/api/rag/index/{index_id}")
async def rag_drop_index(index_id: int):
    try:
        from mio.webui.skills_rag import drop_index
        return drop_index(int(index_id))
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/rag/search")
async def rag_search(q: str, limit: int = 10, label: str | None = None):
    if not q:
        return {"results": []}
    try:
        from mio.webui.skills_rag import search_local_folder
        r = search_local_folder(q, limit=limit, index_label=label)
        return {"results": r.get("results", []), "count": r.get("count", 0)}
    except Exception as e:
        return {"results": [], "error": str(e)}


# --- Obsidian vault integration ------------------------------------
#
# Not just RAG: Mio knows how to list, read, and write notes in a
# configured vault. The vault path persists in ~/.mio/obsidian.json;
# everything is confined to that path (no traversal outside).

def _obsidian_config_path(*, create_parent: bool = False) -> Path:
    root = mio_state_root(create=create_parent)
    return confined_path(root, "obsidian.json")


def _load_obsidian_config() -> dict:
    try:
        p = _obsidian_config_path()
    except UnsafePathError:
        return {}
    if not p.exists() or p.is_symlink():
        return {}
    try:
        return json.loads(read_text_no_follow(p, max_bytes=64 * 1024))
    except (OSError, UnsafePathError, ValueError):
        return {}


def _save_obsidian_config(cfg: dict) -> None:
    root = mio_state_root(create=True)
    path = confined_path(root, "obsidian.json", allow_nested=False)
    atomic_write_json(path, cfg)


def _obsidian_vault() -> Path | None:
    cfg = _load_obsidian_config()
    vp = cfg.get("vault_path")
    if not vp:
        return None
    try:
        return validate_directory(Path(vp).expanduser())
    except UnsafePathError:
        return None


def _obsidian_safe_join(vault: Path, rel: str) -> Path | None:
    """Resolve rel under vault, refusing any path that escapes it."""
    if not rel:
        return None
    try:
        return confined_path(vault, rel)
    except UnsafePathError:
        return None


@router.get("/api/obsidian/config")
async def obsidian_get_config():
    cfg = _load_obsidian_config()
    vp = cfg.get("vault_path") or ""
    try:
        vault_exists = bool(vp) and bool(validate_directory(Path(vp).expanduser()))
    except UnsafePathError:
        vault_exists = False
    return {"vault_path": vp, "vault_exists": vault_exists}


@router.post("/api/obsidian/config")
async def obsidian_set_config(body: dict):
    vp = (body or {}).get("vault_path", "").strip()
    if not vp:
        return {"error": "vault_path required"}
    try:
        expanded = validate_directory(Path(vp).expanduser())
        _save_obsidian_config({"vault_path": str(expanded)})
    except (OSError, UnsafePathError) as exc:
        return {"error": str(exc)}
    return {"ok": True, "vault_path": str(expanded)}


@router.get("/api/obsidian/tree")
async def obsidian_tree():
    vault = _obsidian_vault()
    if not vault:
        return {"error": "vault not configured"}
    return {"vault_path": str(vault), "tree": confined_markdown_tree(vault)}


@router.get("/api/obsidian/note")
async def obsidian_read_note(path: str):
    vault = _obsidian_vault()
    if not vault:
        return {"error": "vault not configured"}
    p = _obsidian_safe_join(vault, path)
    if not p or not p.exists() or p.is_symlink() or not p.is_file():
        return {"error": "note not found"}
    try:
        content = read_text_no_follow(p, max_bytes=8 * 1024 * 1024)
    except (OSError, UnsafePathError) as e:
        return {"error": str(e)}
    return {
        "path":    path,
        "name":    p.name,
        "content": content,
        "size":    p.stat().st_size,
        "mtime":   p.stat().st_mtime,
    }


@router.post("/api/obsidian/note")
async def obsidian_write_note(body: dict):
    vault = _obsidian_vault()
    if not vault:
        return {"error": "vault not configured"}
    rel = (body or {}).get("path", "").strip()
    content = (body or {}).get("content", "")
    if not rel:
        return {"error": "path required"}
    # Ensure .md suffix so Obsidian picks it up.
    if not rel.lower().endswith((".md", ".markdown")):
        rel += ".md"
    try:
        write_confined_text(
            vault,
            rel,
            content,
            create_parents=True,
        )
    except (OSError, UnsafePathError) as exc:
        return {"error": str(exc)}
    return {"ok": True, "path": rel, "size": len(content.encode("utf-8"))}


@router.post("/api/design/export")
async def design_export(body: dict):
    """Package a Design Mode session into a downloadable zip archive.

    Body: {
        title:    str,                     # display title
        platform: str,                     # web | ios | android | ipad
        versions: [{n, title, html, prompt, ts, ...}],
        active:   int,                     # index of the "final" version
        history:  [{role, text}],          # chat history
    }

    Contents of the archive:
        index.html                         — active version (ready to host)
        README.md                          — prompt + version log
        versions/v{N}-{title}.html         — every version
        notes.md                           — machine-readable manifest
    """
    import io
    import zipfile as _zip
    import re as _re

    body = body or {}
    title    = (body.get("title") or "mio-design").strip() or "mio-design"
    platform = (body.get("platform") or "web").strip()
    versions = body.get("versions") or []
    active   = max(0, min(int(body.get("active") or 0), len(versions) - 1)) if versions else 0
    history  = body.get("history") or []

    safe = _re.sub(r"[^A-Za-z0-9_\-]+", "-", title).strip("-") or "mio-design"

    buf = io.BytesIO()
    with _zip.ZipFile(buf, "w", compression=_zip.ZIP_DEFLATED) as z:
        # index.html = active version
        if versions:
            z.writestr("index.html", versions[active].get("html") or "")
        # README
        readme_lines = [
            f"# {title}",
            "",
            f"Exported from **Mio** Design Mode on {time.strftime('%Y-%m-%d %H:%M')}.",
            f"Platform: `{platform}`",
            "",
            "## How to use",
            "",
            "Open `index.html` in a browser, or drop this entire folder onto any static host",
            "(Netlify / Cloudflare Pages / GitHub Pages). `index.html` is the final version you",
            "picked; `versions/` has every iteration so you can diff or roll back.",
            "",
            "## Iteration log",
            "",
        ]
        for m in history:
            role = (m.get("role") or "").strip()
            text = (m.get("text") or "").strip()
            if not text:
                continue
            prefix = "- **You:**  " if role == "user" else "- **Mio:** "
            readme_lines.append(prefix + text.replace("\n", " "))
        readme_lines.extend([
            "",
            "## Versions",
            "",
            "| # | Title | Prompt |",
            "|---|-------|--------|",
        ])
        for i, v in enumerate(versions):
            t = (v.get("title") or f"v{i+1}").replace("|", "\\|")
            p = (v.get("prompt") or "").replace("|", "\\|").replace("\n", " ")
            active_mark = " ← exported as index.html" if i == active else ""
            readme_lines.append(f"| {i+1} | {t}{active_mark} | {p} |")
        z.writestr("README.md", "\n".join(readme_lines))

        # Each version as its own file
        for i, v in enumerate(versions):
            vt = _re.sub(r"[^A-Za-z0-9_\-]+", "-", (v.get("title") or f"v{i+1}")).strip("-")[:60]
            z.writestr(f"versions/v{i+1}-{vt}.html", v.get("html") or "")

        # Machine-readable manifest
        z.writestr("notes.md", json.dumps({
            "title":    title,
            "platform": platform,
            "active":   active,
            "versions": [
                {"n": i + 1, "title": v.get("title"), "prompt": v.get("prompt"), "ts": v.get("ts")}
                for i, v in enumerate(versions)
            ],
            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))

    buf.seek(0)
    filename = f"{safe}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- Flow Mode (visual agent graphs) ------------------------------

def _flows_dir() -> Path:
    p = mio_home() / "flows"
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/api/flows")
async def list_flows():
    flows = []
    candidates = _regular_files_confined(_flows_dir())
    for p in sorted(candidates, key=lambda x: -x.stat().st_mtime):
        try:
            data = json.loads(p.read_text())
            skill = data.get("skill") if isinstance(data.get("skill"), dict) else {}
            flows.append({
                "id":       p.stem,
                "name":     data.get("name", p.stem),
                "nodes":    len(data.get("nodes", {})),
                "updated":  p.stat().st_mtime,
                "exposed":  bool(skill.get("exposed")),
                "skill_name": skill.get("name") if skill.get("exposed") else None,
            })
        except Exception:
            continue
    return {"flows": flows}


@router.get("/api/flows/{flow_id}")
async def get_flow(flow_id: str):
    p = _json_storage_path(_flows_dir(), flow_id, label="flow")
    if not p.exists():
        raise HTTPException(status_code=404, detail="flow not found")
    return json.loads(p.read_text())


@router.post("/api/flows")
async def save_flow(body: dict):
    body = body or {}
    fid = body.get("id") or str(uuid.uuid4())[:8]
    safe = _validate_identifier(fid, label="flow id")
    p = _json_storage_path(_flows_dir(), safe, label="flow")
    nodes = body.get("nodes", {})
    if not isinstance(nodes, dict):
        raise HTTPException(status_code=400, detail="flow nodes must be an object")
    if len(nodes) > 200:
        raise HTTPException(status_code=413, detail="flow exceeds the 200-node limit")
    try:
        graph_bytes = len(json.dumps(nodes, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="flow nodes must be JSON serializable") from exc
    if graph_bytes > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="flow graph exceeds the 2 MiB limit")
    existing_skill: dict[str, Any] = {"exposed": False}
    if p.is_file():
        try:
            existing = json.loads(p.read_text())
            executable_unchanged = (
                existing.get("nodes") == nodes
                and existing.get("edges", []) == body.get("edges", [])
            )
            if executable_unchanged and isinstance(existing.get("skill"), dict):
                existing_skill = existing["skill"]
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    data = {
        "id":      safe,
        "name":    body.get("name", "Untitled flow"),
        "nodes":   nodes,
        "edges":   body.get("edges", []),
        "skill":   existing_skill,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(p, data)
    return {"ok": True, "id": safe}


@router.delete("/api/flows/{flow_id}")
async def delete_flow(flow_id: str):
    p = _json_storage_path(_flows_dir(), flow_id, label="flow")
    if p.exists() and p.is_file():
        p.unlink()
    return {"ok": True}


@router.post("/api/flows/{flow_id}/expose")
async def expose_flow(flow_id: str, body: dict | None = None):
    """Publish one graph behind Mio's bounded list/run flow-skill tools."""
    from mio.webui.flow_skills import FlowSkillError, publish_flow

    body = body or {}
    try:
        return publish_flow(
            flow_id,
            name=str(body.get("name") or ""),
            description=str(body.get("description") or ""),
            root=_flows_dir(),
        )
    except FlowSkillError as exc:
        status = 409 if "already in use" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.delete("/api/flows/{flow_id}/expose")
async def unexpose_flow(flow_id: str):
    from mio.webui.flow_skills import FlowSkillError, unpublish_flow

    try:
        return unpublish_flow(flow_id, root=_flows_dir())
    except FlowSkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/flows/{flow_id}/run")
async def start_flow_run(flow_id: str, body: dict | None = None):
    """Kick off a flow execution. Returns { run_id } which the client
    uses to subscribe to /api/flows/runs/{run_id}/events for SSE."""
    p = _json_storage_path(_flows_dir(), flow_id, label="flow")
    if not p.exists():
        raise HTTPException(status_code=404, detail="flow not found")
    flow = json.loads(p.read_text())
    from mio.webui.flow_runner import start_run
    env = (body or {}).get("env") or {}
    run_id = start_run(flow, env, manager=_manager, gpu_lock=_gpu_lock)
    return {"run_id": run_id}


@router.get("/api/flows/runs/{run_id}/events")
async def flow_run_events(run_id: str):
    """SSE stream of per-node events for a running flow."""
    from mio.webui.flow_runner import discard_run, get_run
    run = get_run(run_id)
    if not run:
        return {"error": "run not found"}

    async def gen():
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(run.queue.get(), timeout=60)
                    yield "data: " + json.dumps(evt) + "\n\n"
                    if evt.get("type") == "run_finished":
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    if run.done:
                        break
        finally:
            discard_run(run_id)
    return StreamingResponse(gen(), media_type="text/event-stream")


# --- Knowledge graph --------------------------------------------
@router.get("/api/graph")
async def knowledge_graph():
    """Walk the user's local state + emit a node/edge graph suitable
    for a Cytoscape-style visualiser.

    Nodes: chat sessions · artifacts · ingest (clipped docs) · projects
           (workspaces) · Obsidian notes (top level only).
    Edges: session→artifact (contains), session→project (belongs-to),
           session↔session (@-mention or title similarity skipped for
           v1), artifact→session (reverse).
    """
    nodes = []
    edges = []

    # Sessions + their artifacts
    if _sessions_dir and _sessions_dir.exists():
        for p in sorted(
            _regular_files_confined(_sessions_dir),
            key=lambda x: -x.stat().st_mtime,
        )[:120]:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            sid = data.get("id") or p.stem
            title = (data.get("title") or sid)[:48]
            nodes.append({"id": f"session:{sid}", "type": "session", "label": title})
            pid = data.get("project_id")
            if pid:
                edges.append({"source": f"session:{sid}", "target": f"project:{pid}", "rel": "in"})
            for art in data.get("artifacts", [])[:20]:
                aid = art.get("id") or f"art-{hash(art.get('title',''))}"
                atitle = (art.get("title") or art.get("type") or "artifact")[:42]
                nodes.append({"id": f"artifact:{aid}", "type": "artifact", "label": atitle})
                edges.append({"source": f"session:{sid}", "target": f"artifact:{aid}", "rel": "emitted"})

    # Projects / workspaces
    for pr in _load_projects():
        nodes.append({"id": f"project:{pr['id']}", "type": "project",
                      "label": pr.get("name", pr["id"])[:42]})

    # Ingested docs
    try:
        ing = mio_state_directory("ingest")
        ingest_files = sorted(
            iter_confined_regular_files(
                ing,
                suffixes={".md"},
                recursive=False,
            ),
            reverse=True,
        )[:60]
    except UnsafePathError:
        ingest_files = []
    for p in ingest_files:
        nodes.append({
            "id": f"doc:{p.stem}", "type": "doc",
            "label": p.stem.split("-", 2)[-1][:42],
        })

    # Obsidian notes (top-level files only — keeps the graph tractable)
    vault = _obsidian_vault()
    if vault is not None:
        for p in list(
            iter_confined_regular_files(
                vault,
                suffixes={".md", ".markdown"},
                recursive=False,
            )
        )[:40]:
            nodes.append({
                "id": f"note:{p.name}",
                "type": "note",
                "label": p.stem[:42],
            })

    return {"nodes": nodes, "edges": edges}


# --- Scratchpad --------------------------------------------------

@router.get("/api/scratchpad")
async def scratchpad_get():
    p = mio_home() / "scratchpad.md"
    if not p.exists():
        return {"content": ""}
    return {"content": p.read_text(), "path": str(p)}


@router.post("/api/scratchpad")
async def scratchpad_set(body: dict):
    p = mio_home() / "scratchpad.md"
    content = (body or {}).get("content", "")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="scratchpad content must be text")
    payload = content.encode("utf-8")
    if len(payload) > _MAX_LOCAL_NOTE_BYTES:
        raise HTTPException(status_code=413, detail="scratchpad exceeds the 2 MiB limit")
    atomic_write_bytes(p, payload)
    return {"ok": True, "size": len(payload)}


# --- Daily Note --------------------------------------------------

def _journal_path(date_str: str | None = None) -> Path:
    import datetime as _dt
    d = date_str or _dt.date.today().isoformat()
    p = mio_home() / "journal" / f"{d}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/api/journal/today")
async def journal_today():
    p = _journal_path()
    content = p.read_text() if p.exists() else ""
    # Pull yesterday's tail (last 600 chars) so a "today + yesterday"
    # view feels continuous.
    import datetime as _dt
    yesterday_date = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
    yp = _journal_path(yesterday_date)
    yesterday_tail = ""
    if yp.exists():
        ytext = yp.read_text()
        yesterday_tail = ytext[-600:] if len(ytext) > 600 else ytext
    return {
        "date":             _dt.date.today().isoformat(),
        "path":             str(p),
        "content":          content,
        "yesterday_date":   yesterday_date,
        "yesterday_tail":   yesterday_tail,
    }


@router.post("/api/journal/today")
async def journal_save(body: dict):
    p = _journal_path()
    content = (body or {}).get("content", "")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="journal content must be text")
    payload = content.encode("utf-8")
    if len(payload) > _MAX_LOCAL_NOTE_BYTES:
        raise HTTPException(status_code=413, detail="journal entry exceeds the 2 MiB limit")
    atomic_write_bytes(p, payload)
    return {"ok": True, "path": str(p), "size": len(payload)}


@router.post("/api/reveal")
async def reveal_in_finder(body: dict):
    """Open a local path in the user's file browser (Finder on macOS)."""
    from pathlib import Path as _P
    path_str = (body or {}).get("path", "").strip() or str(mio_home())
    p = _P(path_str).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
        return {"ok": True, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/export-workspace")
async def export_workspace():
    """Zip everything under ~/.mio into a single archive and stream it back.
    Skips large caches (image-cache/, web-cache/, files-cache/) — the user
    can re-download those. Includes sessions, projects, memory, journal,
    todos/habits SQLite dbs, ingest, obsidian config, rag.sqlite.
    """
    root = mio_home()
    if not root.exists():
        return {"error": "no ~/.mio yet"}

    SKIP_DIRS = {"image-cache", "web-cache", "files-cache", "__pycache__"}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in _regular_files_confined(root, "*", recursive=True):
            rel = path.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            try:
                z.write(path, arcname=str(rel))
            except Exception:
                continue
    buf.seek(0)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"mio-workspace-{stamp}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/obsidian/reindex")
async def obsidian_reindex():
    """Full-text index the vault into the local RAG store so `@note:`
    references and chat search can find notes."""
    vault = _obsidian_vault()
    if not vault:
        return {"error": "vault not configured"}
    try:
        from mio.webui.skills_rag import index_folder
        return index_folder(str(vault), label="obsidian", replace=True)
    except Exception as e:
        return {"error": str(e)}


@router.delete("/api/ingest/{doc_id}")
async def delete_ingested(doc_id: str):
    """Remove a stashed ingest file (and leave the RAG index to self-heal
    on next re-index — cheap)."""
    if not _IDENTIFIER_RE.fullmatch(doc_id):
        return {"deleted": False, "error": "invalid id"}
    try:
        root = mio_state_directory("ingest")
        target = confined_path(root, f"{doc_id}.md", must_exist=True)
    except UnsafePathError:
        return {"deleted": False, "error": "not found"}
    if target.is_file() and not target.is_symlink():
        target.unlink()
        return {"deleted": True, "id": doc_id}
    return {"deleted": False, "error": "not found"}


# --- Global search across saved sessions ---
@router.get("/api/search")
async def global_search(q: str, limit: int = 30):
    if not _sessions_dir or not _sessions_dir.exists() or not q:
        return {"results": []}
    ql = q.lower()
    results = []
    for f in sorted(
        _regular_files_confined(_sessions_dir),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            meta = json.loads(f.read_text())
            title = meta.get("title", "Untitled")
            hits = []
            for i, m in enumerate(meta.get("messages", [])):
                c = (m.get("content") or "")
                if ql in c.lower() or ql in title.lower():
                    idx = c.lower().find(ql)
                    start = max(0, idx - 60)
                    end = min(len(c), (idx if idx >= 0 else 0) + 180)
                    snippet = c[start:end].replace("\n", " ")
                    hits.append({"msg_index": i, "role": m.get("role"), "snippet": snippet})
                    if len(hits) >= 3:
                        break
            if hits or ql in title.lower():
                results.append({
                    "id": f.stem,
                    "title": title,
                    "updated": meta.get("updated", ""),
                    "hits": hits,
                })
                if len(results) >= limit:
                    break
        except Exception:
            continue
    return {"results": results}


# --- Image cache: on-disk persistent store for poster / thumbnail URLs ---
# The mediacard iframe is sandboxed with srcdoc — its origin is opaque,
# so MAL/Jikan/etc. may intermittently reject its img requests. We pre-
# download poster URLs at skill-run time into ~/.mio/image-cache/ and
# rewrite the artifact JSON to reference local /ui/img/<hash>.<ext>
# paths, so the artifact keeps working forever even offline.
_img_proxy_cache: dict[str, tuple[bytes, str]] = {}

PIMIO_DIR = mio_state_root(create=True)
IMAGE_CACHE_DIR = mio_state_directory("image-cache", create=True)
WEB_CACHE_DIR = mio_state_directory("web-cache", create=True)
FILES_CACHE_DIR = mio_state_directory("files-cache", create=True)

# Extensions we accept for the served cache (prevents arbitrary-read)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif"}


def web_cache_get(url: str) -> str | None:
    """Return cached page text for `url` or None."""
    if not url:
        return None
    key = hashlib.sha1(url.encode()).hexdigest()
    try:
        p = confined_path(
            WEB_CACHE_DIR,
            f"{key}.json",
            must_exist=True,
            allow_nested=False,
        )
        payload = read_text_no_follow(p, max_bytes=16 * 1024 * 1024)
        return json.loads(payload)["content"]
    except (KeyError, OSError, TypeError, ValueError, UnsafePathError):
        return None


def web_cache_put(url: str, content: str) -> None:
    """Persist page text keyed by URL."""
    if not url or content is None:
        return
    key = hashlib.sha1(url.encode()).hexdigest()
    try:
        path = confined_path(
            WEB_CACHE_DIR,
            f"{key}.json",
            allow_nested=False,
        )
        atomic_write_json(
            path,
            {
                "url": url,
                "fetched_at": int(time.time()),
                "content": content,
            },
        )
    except (OSError, TypeError, ValueError, UnsafePathError):
        pass


def _dir_stats(p: Path, suffixes: set[str] | None = None) -> dict:
    """Return {count, bytes} for files inside `p`, optionally filtered."""
    count = 0
    total = 0
    try:
        files = iter_confined_regular_files(p, recursive=False)
    except UnsafePathError:
        return {"count": 0, "bytes": 0}
    for f in files:
        if suffixes and f.suffix.lower() not in suffixes:
            continue
        try:
            total += f.stat().st_size
            count += 1
        except Exception:
            pass
    return {"count": count, "bytes": total}


def _clear_dir(p: Path) -> int:
    """Unlink every file in `p`. Returns count of files removed."""
    n = 0
    try:
        files = list(iter_confined_regular_files(p, recursive=False))
    except UnsafePathError:
        return 0
    for f in files:
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def cache_image_to_disk(url: str) -> str | None:
    """Download `url` once into IMAGE_CACHE_DIR, return a local
    `/ui/img/<hash>.<ext>` path (served by serve_cached_image). Returns
    None on any failure so callers can keep the original URL or drop it.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return None
    key = hashlib.sha1(url.encode()).hexdigest()
    # If any file with this hash stem already exists, reuse it.
    try:
        for existing in iter_confined_regular_files(
            IMAGE_CACHE_DIR,
            suffixes=_IMG_EXTS,
            recursive=False,
        ):
            if existing.stem == key:
                return f"/ui/img/{existing.name}"
        payload = fetch_image(url, allowed_hosts=None)
        path = write_confined_bytes(
            IMAGE_CACHE_DIR,
            f"{key}{payload.extension}",
            payload.data,
        )
    except (ImageFetchError, OSError, UnsafePathError):
        return None
    return f"/ui/img/{path.name}"


@router.get("/proxy-image")
async def proxy_image(url: str):
    from fastapi.responses import Response as _R
    key = hashlib.sha1(url.encode()).hexdigest()
    if key in _img_proxy_cache:
        data, mime = _img_proxy_cache[key]
        return _R(content=data, media_type=mime, headers={"Cache-Control": "max-age=3600"})
    try:
        payload = await asyncio.to_thread(fetch_image, url)
        data = payload.data
        mime = payload.media_type
    except ImageFetchError as exc:
        message = str(exc)
        if "host is not allowed" in message:
            status = 403
        elif "URL" in message or "http://" in message or "host is required" in message:
            status = 400
        elif "Content-Type" in message or "supported raster image" in message:
            status = 415
        else:
            status = 502
        return _R(status_code=status, content=message)
    except OSError as exc:
        return _R(status_code=502, content=f"fetch failed: {exc}")
    _img_proxy_cache[key] = (data, mime)
    # Cap the cache at ~200 entries
    if len(_img_proxy_cache) > 200:
        for k in list(_img_proxy_cache.keys())[:50]:
            del _img_proxy_cache[k]
    return _R(content=data, media_type=mime, headers={"Cache-Control": "max-age=3600"})


@router.get("/api/cache/stats")
async def cache_stats():
    """Return {images, webpages, files} — each with {count, bytes}."""
    return {
        "images": _dir_stats(IMAGE_CACHE_DIR, _IMG_EXTS),
        "webpages": _dir_stats(WEB_CACHE_DIR, {".json"}),
        "files": _dir_stats(FILES_CACHE_DIR),
    }


@router.post("/api/cache/clear")
async def cache_clear(payload: dict):
    """POST {"kind": "images" | "webpages" | "files" | "all"}.
    Deletes files in the matching ~/.mio/ sub-cache. Never touches
    ~/Downloads — user's own files live there and we can't reliably
    distinguish them from our generated ones.
    """
    kind = (payload or {}).get("kind", "")
    if kind not in ("images", "webpages", "files", "all"):
        return {"error": f"unknown kind: {kind}"}
    removed = 0
    if kind in ("images", "all"):
        removed += _clear_dir(IMAGE_CACHE_DIR)
    if kind in ("webpages", "all"):
        removed += _clear_dir(WEB_CACHE_DIR)
    if kind in ("files", "all"):
        removed += _clear_dir(FILES_CACHE_DIR)
    return {"ok": True, "removed": removed}


# ---- Chat import (ChatGPT / Claude exports) ----


_MAX_IMPORT_CONVERSATIONS = 1_000
_MAX_IMPORT_MAPPING_NODES = 20_000
_MAX_IMPORT_MESSAGES = 8_192
_MAX_IMPORT_MESSAGE_BYTES = 2 * 1024 * 1024
_MAX_IMPORT_SESSION_BYTES = 16 * 1024 * 1024


def _bounded_import_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        return ""
    if len(value.encode("utf-8")) > _MAX_IMPORT_MESSAGE_BYTES:
        raise ValueError(f"{label} exceeds the 2 MiB limit")
    return value


def _chatgpt_active_branch(raw: dict, mapping: dict) -> list[dict]:
    """Return one selected ChatGPT branch without recursion.

    ChatGPT exports retain alternative edits as sibling nodes.  Flattening a
    depth-first traversal merges mutually exclusive replies into one false
    transcript.  ``current_node`` identifies the selected leaf; older exports
    without it fall back deterministically to the most recently created leaf.
    """
    if len(mapping) > _MAX_IMPORT_MAPPING_NODES:
        raise ValueError("ChatGPT mapping exceeds the 20,000-node limit")
    if not mapping:
        return []

    ordered_ids: list[str] = []
    parent_ids: set[str] = set()
    for node_id, node in mapping.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            raise ValueError("ChatGPT mapping contains an invalid node")
        ordered_ids.append(node_id)
        parent = node.get("parent")
        if parent is not None:
            if not isinstance(parent, str) or parent not in mapping:
                raise ValueError("ChatGPT mapping contains an invalid parent")
            parent_ids.add(parent)

    selected = raw.get("current_node")
    if selected is not None:
        if not isinstance(selected, str) or selected not in mapping:
            raise ValueError("ChatGPT current_node is missing from the mapping")
    else:
        leaves = [node_id for node_id in ordered_ids if node_id not in parent_ids]
        if not leaves:
            raise ValueError("ChatGPT mapping has no acyclic leaf")
        positions = {node_id: index for index, node_id in enumerate(ordered_ids)}

        def leaf_rank(node_id: str) -> tuple[float, int]:
            message = mapping[node_id].get("message")
            created = message.get("create_time") if isinstance(message, dict) else None
            try:
                timestamp = float(created) if isinstance(created, (int, float)) else float("-inf")
            except (OverflowError, ValueError):
                timestamp = float("-inf")
            if not math.isfinite(timestamp):
                timestamp = float("-inf")
            return timestamp, positions[node_id]

        selected = max(leaves, key=leaf_rank)

    lineage: list[dict] = []
    seen: set[str] = set()
    node_id: str | None = selected
    while node_id is not None:
        if node_id in seen:
            raise ValueError("ChatGPT mapping contains a parent cycle")
        if len(lineage) >= _MAX_IMPORT_MESSAGES:
            raise ValueError("ChatGPT branch exceeds the 8,192-node limit")
        seen.add(node_id)
        node = mapping[node_id]
        lineage.append(node)
        node_id = node.get("parent")
    lineage.reverse()
    return lineage


def _chatgpt_message_text(message: dict) -> str:
    content = message.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return _bounded_import_text("\n".join(text_parts), label="imported message")


def _normalize_imported_chat(raw: dict) -> list[dict]:
    """Given one conversation dict from either ChatGPT (conversations.json)
    or Claude export format, return a list of Mio-shaped messages."""
    # ChatGPT format: has mapping = {"uuid": {"message": {...}, "children": [...]}}
    if "mapping" in raw:
        mapping = raw.get("mapping")
        if not isinstance(mapping, dict):
            raise ValueError("ChatGPT mapping must be an object")
        msgs: list[dict] = []
        for node in _chatgpt_active_branch(raw, mapping):
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            author = msg.get("author")
            role = author.get("role") if isinstance(author, dict) else None
            text = _chatgpt_message_text(msg)
            if text.strip() and role in ("user", "assistant"):
                msgs.append({"role": role, "content": text})
        return msgs
    # Claude export format: {"chat_messages": [{"text": "...", "sender": "human|assistant"}]}
    if "chat_messages" in raw:
        messages = raw.get("chat_messages")
        if not isinstance(messages, list) or len(messages) > _MAX_IMPORT_MESSAGES:
            raise ValueError("Claude messages must be a list of at most 8,192 items")
        out: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            sender = message.get("sender")
            if sender not in ("human", "assistant"):
                continue
            role = "user" if sender == "human" else "assistant"
            text = _bounded_import_text(message.get("text"), label="imported message")
            if text.strip():
                out.append({"role": role, "content": text})
        return out
    # Mio native: already {"messages": [...]}
    if "messages" in raw:
        messages = raw.get("messages")
        if not isinstance(messages, list) or len(messages) > _MAX_IMPORT_MESSAGES:
            raise ValueError("Mio messages must be a list of at most 8,192 items")
        out = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                continue
            text = _bounded_import_text(message.get("content"), label="imported message")
            if text.strip():
                out.append({"role": message["role"], "content": text})
        return out
    return []


@router.post("/api/chats/import")
async def import_chats(body: dict):
    """Import a batch of conversations from ChatGPT, Claude, or Mio-native
    export. Returns how many sessions were created.

    Accepts:
      { "source": "chatgpt" | "claude" | "mio" | "auto",
        "data": <the parsed JSON> }
    `data` may be a single conversation dict or a list of them.
    """
    data = (body or {}).get("data")
    if data is None:
        return {"error": "no data"}
    items = data if isinstance(data, list) else [data]
    if len(items) > _MAX_IMPORT_CONVERSATIONS:
        raise HTTPException(status_code=413, detail="import exceeds the 1,000-conversation limit")
    source = (body or {}).get("source", "auto")
    if source not in ("auto", "chatgpt", "claude", "mio"):
        raise HTTPException(status_code=400, detail="unknown chat import source")

    prepared: list[dict] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        try:
            msgs = _normalize_imported_chat(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not msgs:
            continue
        text_bytes = sum(
            len(message["content"].encode("utf-8")) + len(message["role"])
            for message in msgs
        )
        if text_bytes > _MAX_IMPORT_SESSION_BYTES:
            raise HTTPException(status_code=413, detail="imported conversation exceeds 16 MiB")
        raw_title = raw.get("title") or raw.get("name") or msgs[0]["content"][:60]
        title = _bounded_import_text(raw_title, label="imported title")[:200] or "Imported"
        sid = str(uuid.uuid4())[:8]
        payload = {
            "id": sid,
            "title": title,
            "messages": msgs,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "imported_from": source,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(encoded) > _MAX_IMPORT_SESSION_BYTES:
            raise HTTPException(status_code=413, detail="imported conversation exceeds 16 MiB")
        prepared.append(payload)

    # Validate the full batch before creating its first session so malformed
    # later conversations cannot produce a surprising partial import.
    for payload in prepared:
        path = _json_storage_path(_sessions_dir, payload["id"], label="session")
        atomic_write_json(path, payload)
    return {"ok": True, "created": len(prepared)}


# ---- Stats + attachments dashboards ----


@router.get("/api/stats")
async def stats_summary():
    """Aggregate numbers across saved sessions for the /stats dashboard."""
    import collections
    if not _sessions_dir or not _sessions_dir.exists():
        return {"error": "no sessions dir"}
    total_msgs = 0
    total_user = 0
    total_assistant = 0
    sessions = 0
    by_day: dict[str, int] = collections.defaultdict(int)
    skills_used: dict[str, int] = collections.defaultdict(int)
    artifact_types: dict[str, int] = collections.defaultdict(int)
    personas_used: dict[str, int] = collections.defaultdict(int)
    for f in _regular_files_confined(_sessions_dir):
        try:
            meta = json.loads(f.read_text())
        except Exception:
            continue
        sessions += 1
        day = (meta.get("updated") or "")[:10]
        msgs = meta.get("messages") or []
        if day:
            by_day[day] += len(msgs)
        for m in msgs:
            total_msgs += 1
            if m.get("role") == "user":
                total_user += 1
                # Detect /as <persona> style usage
                c = m.get("content") or ""
                mpers = re.match(r"^/as\s+(\w+)", c)
                if mpers:
                    personas_used[mpers.group(1)] += 1
            elif m.get("role") == "assistant":
                total_assistant += 1
                c = m.get("content") or ""
                for _m in re.finditer(r"<tool_call>\s*<function=([\w_]+)", c):
                    skills_used[_m.group(1)] += 1
        for a in meta.get("artifacts") or []:
            t = a.get("type") or "?"
            artifact_types[t] += 1
    return {
        "sessions": sessions,
        "messages": total_msgs,
        "user_messages": total_user,
        "assistant_messages": total_assistant,
        "messages_by_day": sorted(by_day.items()),
        "skills_used": sorted(skills_used.items(), key=lambda x: -x[1])[:20],
        "artifact_types": sorted(artifact_types.items(), key=lambda x: -x[1])[:20],
        "personas_used": sorted(personas_used.items(), key=lambda x: -x[1])[:10],
    }


@router.get("/api/attachments")
async def list_attachments():
    """Walk ~/Downloads for likely Mio-generated files (by extension) and
    return them as an attachment library."""
    try:
        downloads = _validated_downloads_dir()
    except UnsafePathError:
        return {"files": []}
    files = []
    for f in iter_confined_regular_files(downloads, recursive=False):
        if f.suffix.lower() not in _ALLOWED_EXT:
            continue
        try:
            st = f.stat(follow_symlinks=False)
        except OSError:
            continue
        files.append({
            "name": f.name,
            "ext": f.suffix.lower(),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "url": f"/ui/files/{f.name}",
        })
    files.sort(key=lambda x: -x["mtime"])
    return {"files": files[:500]}


# ---- Chat templates ----

_TEMPLATES_FILE = mio_home() / "chat-templates.json"


def _load_chat_templates() -> list:
    if not _TEMPLATES_FILE.exists():
        return []
    try:
        return json.loads(_TEMPLATES_FILE.read_text()) or []
    except Exception:
        return []


def _save_chat_templates(items: list) -> None:
    atomic_write_json(_TEMPLATES_FILE, items)


@router.get("/api/chat-templates")
async def chat_templates_list():
    return {"templates": _load_chat_templates()}


@router.post("/api/chat-templates")
async def chat_templates_save(body: dict):
    name = (body or {}).get("name", "").strip()
    if not name:
        return {"error": "name required"}
    messages = (body or {}).get("messages", [])
    items = _load_chat_templates()
    existing = next((t for t in items if t["name"] == name), None)
    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "messages": messages,
        "description": (body or {}).get("description", ""),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if existing:
        existing.update({k: v for k, v in entry.items() if k != "id" and k != "created"})
    else:
        items.append(entry)
    _save_chat_templates(items)
    return {"ok": True, "id": existing["id"] if existing else entry["id"]}


@router.delete("/api/chat-templates/{template_id}")
async def chat_templates_delete(template_id: str):
    items = [t for t in _load_chat_templates() if t.get("id") != template_id]
    _save_chat_templates(items)
    return {"ok": True}


# ---- Scheduler endpoints ----

@router.get("/api/schedules")
async def schedules_list():
    from mio.webui import scheduler as _sched
    try:
        return {"schedules": _sched.load_schedules(), "recent": _sched.recent_runs()}
    except (json.JSONDecodeError, _sched.ScheduleStoreError) as exc:
        _raise_store_conflict("schedules", exc)


@router.post("/api/schedules")
async def schedules_create(body: dict):
    from mio.webui import scheduler as _sched
    try:
        return _sched.create_schedule(
            name=(body or {}).get("name", ""),
            prompt=(body or {}).get("prompt", ""),
            cadence=(body or {}).get("cadence"),
            tier=(body or {}).get("tier"),
            enabled=(body or {}).get("enabled", True),
        )
    except (json.JSONDecodeError, _sched.ScheduleStoreError) as exc:
        _raise_store_conflict("schedules", exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/api/schedules/{sched_id}")
async def schedules_update(sched_id: str, body: dict):
    from mio.webui import scheduler as _sched
    try:
        return _sched.update_schedule(sched_id, body or {})
    except (json.JSONDecodeError, _sched.ScheduleStoreError) as exc:
        _raise_store_conflict("schedules", exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/schedules/{sched_id}")
async def schedules_delete(sched_id: str):
    from mio.webui import scheduler as _sched
    try:
        return _sched.delete_schedule(sched_id)
    except (json.JSONDecodeError, _sched.ScheduleStoreError) as exc:
        _raise_store_conflict("schedules", exc)


# ---- Webhook endpoints ----

@router.get("/api/webhooks")
async def webhooks_list():
    from mio.webui import webhooks as _wh
    try:
        return {"webhooks": _wh.public_webhooks(), "recent": _wh.recent_runs()}
    except (json.JSONDecodeError, _wh.WebhookStoreError) as exc:
        _raise_store_conflict("webhooks", exc)


@router.post("/api/webhooks")
async def webhooks_create(body: dict):
    from mio.webui import webhooks as _wh
    try:
        return _wh.create_webhook(
            slug=(body or {}).get("slug", ""),
            prompt=(body or {}).get("prompt", ""),
            tier=(body or {}).get("tier"),
            secret=(body or {}).get("secret"),
        )
    except (json.JSONDecodeError, _wh.WebhookStoreError) as exc:
        _raise_store_conflict("webhooks", exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/webhooks/{slug}")
async def webhooks_delete(slug: str):
    from mio.webui import webhooks as _wh
    try:
        return _wh.delete_webhook(slug)
    except (json.JSONDecodeError, _wh.WebhookStoreError) as exc:
        _raise_store_conflict("webhooks", exc)


@router.post("/api/webhook/{slug}")
async def webhook_fire(slug: str, request: Request, body: dict | None = None):
    """Fire a configured webhook. POST body is JSON; its keys substitute
    into the prompt template's {{key}} slots. Authentication always uses the
    ``X-Mio-Webhook-Secret`` header; secrets in JSON are never accepted or
    forwarded into the model prompt.
    """
    from mio.webui import webhooks as _wh
    try:
        hooks = _wh.load_webhooks()
    except (json.JSONDecodeError, _wh.WebhookStoreError) as exc:
        _raise_store_conflict("webhooks", exc)
    hook = next((h for h in hooks if h["slug"] == slug), None)
    if not hook:
        raise HTTPException(status_code=404, detail="unknown webhook")
    payload = body or {}
    supplied_secret = request.headers.get("x-mio-webhook-secret")
    if not _wh.verify_secret(hook, supplied_secret):
        raise HTTPException(status_code=401, detail="invalid webhook secret")
    # A field named ``secret`` is application data only in legacy callers.  Do
    # not let it enter substitutions or persisted run logs.
    payload = {key: value for key, value in payload.items() if key != "secret"}
    prompt = _wh.render_prompt(hook.get("prompt", ""), payload)
    if not _manager:
        return {"error": "no model loaded"}
    manager = _manager

    def _generate() -> tuple[str, str]:
        with _gpu_lock:
            loaded = manager.loaded_tiers()
            if not loaded:
                raise RuntimeError("no tiers loaded")
            tier = hook.get("tier") if hook.get("tier") in loaded else loaded[0]
            engine = manager.get_engine(tier)
            text, _metrics = engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
        return tier, text

    try:
        tier, output = await asyncio.to_thread(_generate)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
        _wh.append_log(slug, payload, result)
        return result
    result = {"ok": True, "slug": slug, "tier": tier, "output": output}
    _wh.append_log(slug, payload, result)
    return result


_ASSETS_DIR = Path(__file__).parent / "assets"
_ASSET_ALLOWED_EXT = {".js", ".mjs", ".css", ".svg", ".png", ".ico", ".woff", ".woff2", ".map", ".html", ".json"}


@router.get("/assets/{path:path}")
async def serve_asset(path: str):
    """Serve modular front-end assets (.js / .css / images) from
    mio/webui/assets/. Paths are sandboxed to the assets dir so
    ``../`` can't escape. Cached aggressively — file names should be
    content-versioned by the client if cache-busting is needed.
    """
    from fastapi.responses import FileResponse as _FR, Response as _R
    if ".." in path or path.startswith("/"):
        return _R(status_code=400, content="invalid path")
    p = (_ASSETS_DIR / path).resolve()
    try:
        p.relative_to(_ASSETS_DIR.resolve())
    except ValueError:
        return _R(status_code=400, content="path traversal")
    if p.suffix.lower() not in _ASSET_ALLOWED_EXT:
        return _R(status_code=400, content="unsupported extension")
    if not p.exists() or not p.is_file():
        return _R(status_code=404, content="not found")
    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    # Zero-cache while we're iterating fast. Switch back to a longer
    # max-age once the module set is stable.
    return _FR(str(p), media_type=mime, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    })


@router.get("/img/{filename}")
async def serve_cached_image(filename: str):
    """Serve an image from IMAGE_CACHE_DIR by its on-disk filename
    (<sha1>.<ext>). Filenames are chosen by cache_image_to_disk, never
    by user input — but still validate to be safe.
    """
    from fastapi.responses import Response as _R
    if "/" in filename or "\\" in filename or ".." in filename:
        return _R(status_code=400, content="invalid filename")
    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in _IMG_EXTS:
        return _R(status_code=400, content="unsupported extension")
    try:
        p = confined_path(
            IMAGE_CACHE_DIR,
            filename,
            must_exist=True,
            allow_nested=False,
        )
        with open_binary_no_follow(p, max_bytes=8 * 1024 * 1024) as handle:
            data = handle.read()
    except (OSError, UnsafePathError):
        return _R(status_code=404, content="not cached")
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    headers = {"Cache-Control": "max-age=31536000, immutable", "X-Content-Type-Options": "nosniff"}
    # Long cache — filenames are content-addressed by URL hash, so they
    # never change meaning.
    return _R(content=data, media_type=mime, headers=headers)


@router.get("/files/{filename}")
async def serve_generated_file(filename: str, download: int = 0):
    if _safe_upload_name(filename) != filename or ".." in filename:
        return Response(status_code=400, content="invalid filename")
    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in _ALLOWED_EXT:
        return Response(status_code=400, content="unsupported file type")
    try:
        downloads = _validated_downloads_dir()
        p = confined_path(
            downloads,
            filename,
            must_exist=True,
            allow_nested=False,
        )
        # Open once before sending headers so symlinks/special files fail as
        # a normal 404 rather than midway through the response.
        with open_binary_no_follow(p):
            pass
    except (OSError, UnsafePathError):
        return Response(status_code=404, content="not found")
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # ?download=1 → attachment (triggers Save dialog). Default is inline so
    # PDFs can render in an <iframe> rather than being downloaded.
    disposition = "attachment" if download else "inline"
    ascii_filename = filename.encode("ascii", errors="ignore").decode() or "download"
    headers = {
        "Content-Disposition": (
            f'{disposition}; filename="{ascii_filename}"; filename*=UTF-8\'\'{quote(filename)}'
        ),
        "X-Content-Type-Options": "nosniff",
    }
    if ext in {".html", ".svg"}:
        # HTML and SVG are active document formats when opened directly.
        # Keep generated/uploaded previews scriptless and origin-isolated.
        headers["Content-Security-Policy"] = (
            "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'"
        )
    def stream_file():
        with open_binary_no_follow(p) as handle:
            while chunk := handle.read(_FILE_STREAM_CHUNK_BYTES):
                yield chunk

    return StreamingResponse(
        stream_file(),
        media_type=mime,
        headers=headers,
    )


# --- WebSocket streaming chat ---

@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    from mio.web_security import reject_untrusted_websocket

    if await reject_untrusted_websocket(websocket):
        return
    offered_protocols = {
        protocol.strip()
        for protocol in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if protocol.strip()
    }
    # Browser clients offer the stable application protocol plus a separate
    # CSRF-bearing protocol.  Echo only the stable value: selecting one of the
    # offered protocols is required by stricter WebSocket implementations,
    # while the secret never appears in the response headers.
    await websocket.accept(subprotocol="mio-ui" if "mio-ui" in offered_protocols else None)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "chat")

            if action == "chat":
                await _handle_chat(websocket, data)
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
            elif action == "cancel":
                # Best-effort: disconnect will raise in _handle_chat's await
                # so the producer thread's for-loop can finish on the next
                # chunk. Future hook: wire engine.cancel_stream() when
                # available.
                await websocket.close()
                return
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def _localize_image_url(u: Any) -> Any:
    """If `u` looks like an http(s) image URL, try to download it into
    IMAGE_CACHE_DIR and return the resulting `/ui/img/<...>` path. On
    failure, return the original value unchanged."""
    if not isinstance(u, str):
        return u
    if not (u.startswith("http://") or u.startswith("https://")):
        return u
    local = cache_image_to_disk(u)
    return local or u


def _auto_artifact_from_skill(skill: str, args: dict, result: dict) -> dict | None:
    """Turn a successful media-oriented skill result directly into an
    artifact payload so the UI can render it even if the model never emits
    the <antArtifact> tag. Returns None for skills without auto-rendering.
    """
    if not result or result.get("error"):
        return None
    if skill in ("find_anime", "find_manga", "find_movie_tv", "find_game"):
        items = result.get("results") or []
        if not items:
            return None
        # Pre-download poster images so reopening the artifact later never
        # 404s against MAL / Jikan / TVmaze hotlink rules.
        for it in items:
            if not isinstance(it, dict):
                continue
            for k in ("image", "poster", "thumbnail"):
                if it.get(k):
                    it[k] = _localize_image_url(it[k])
        label = {"find_anime": "Anime", "find_manga": "Manga",
                 "find_movie_tv": "Movie / TV", "find_game": "Games"}[skill]
        q = args.get("query") or args.get("genre") or "Suggestions"
        return {
            "artifact_type": "application/vnd.pimio.mediacard",
            "identifier": f"{skill}-{int(time.time())}",
            "title": f"{label}: {q}",
            "content": json.dumps({"title": f"{label} — {q}", "items": items},
                                  ensure_ascii=False),
        }
    if skill == "search_images":
        imgs = result.get("results") or []
        if not imgs:
            return None
        # Images can be bare-string URLs or {url, thumb, source, title} dicts
        localized = []
        for it in imgs:
            if isinstance(it, str):
                localized.append(_localize_image_url(it))
            elif isinstance(it, dict):
                for k in ("url", "thumb", "src"):
                    if it.get(k):
                        it[k] = _localize_image_url(it[k])
                localized.append(it)
            else:
                localized.append(it)
        q = args.get("query") or "Images"
        return {
            "artifact_type": "application/vnd.pimio.imagegrid",
            "identifier": f"images-{int(time.time())}",
            "title": f"Images: {q}",
            "content": json.dumps({"title": q, "images": localized}, ensure_ascii=False),
        }
    if skill == "search_youtube":
        vids = result.get("results") or []
        if not vids:
            return None
        for v in vids:
            if isinstance(v, dict) and v.get("thumbnail"):
                v["thumbnail"] = _localize_image_url(v["thumbnail"])
        q = args.get("query") or "YouTube"
        return {
            "artifact_type": "application/vnd.pimio.youtubegrid",
            "identifier": f"yt-{int(time.time())}",
            "title": f"YouTube: {q}",
            "content": json.dumps({"title": q, "videos": vids}, ensure_ascii=False),
        }
    if skill == "get_weather":
        if not result.get("current"):
            return None
        loc = (result.get("location") or {}).get("name", "")
        return {
            "artifact_type": "application/vnd.pimio.weather",
            "identifier": f"weather-{int(time.time())}",
            "title": f"Weather in {loc}",
            "content": json.dumps(result, ensure_ascii=False),
        }
    return None


@dataclass(frozen=True, slots=True)
class _WebUIRoundResult:
    text: str
    metrics: dict | None
    completed: bool
    failed: bool


async def _stream_webui_round(
    websocket: WebSocket,
    *,
    manager: Any,
    tier: str,
    gpu_lock: Any,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    tools: list[dict] | None,
    is_first: bool,
) -> _WebUIRoundResult:
    """Bridge one synchronous MLX stream into a bounded WebSocket stream.

    Generation owns the lifecycle GPU lock from engine lookup through source
    closure, so a concurrent unload cannot invalidate weights mid-stream.
    The producer blocks on a small asyncio queue and observes cancellation
    while waiting for both queue capacity and the GPU lock.
    """
    from concurrent.futures import CancelledError as FutureCancelledError
    from concurrent.futures import TimeoutError as FutureTimeoutError

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_WS_STREAM_QUEUE_MAXSIZE)
    sentinel = object()
    error_tag = object()
    cancelled = threading.Event()

    def put_from_thread(item: object) -> bool:
        if cancelled.is_set():
            return False
        put_coro = queue.put(item)
        try:
            future = asyncio.run_coroutine_threadsafe(put_coro, loop)
        except RuntimeError:
            put_coro.close()
            return False
        while not cancelled.is_set():
            try:
                future.result(timeout=0.1)
                return True
            except FutureTimeoutError:
                continue
            except (FutureCancelledError, RuntimeError):
                return False
        future.cancel()
        return False

    def produce() -> None:
        acquired_gpu = False
        source = None
        try:
            while not cancelled.is_set():
                if gpu_lock.acquire(timeout=0.1):
                    acquired_gpu = True
                    break
            if not acquired_gpu:
                return

            # Model load/unload uses this same outer lock. Resolve only now,
            # never from a stale reference captured before the wait.
            active_engine = manager.get_engine(tier)
            if cancelled.is_set():
                return
            source = active_engine.generate_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            )
            iterator = iter(source)
            while not cancelled.is_set():
                try:
                    chunk_text, chunk_metrics = next(iterator)
                except StopIteration:
                    break
                event: dict[str, Any] = {"type": "token", "text": chunk_text}
                if chunk_metrics:
                    event["metrics"] = {
                        "prompt_tokens": chunk_metrics.prompt_tokens,
                        "completion_tokens": chunk_metrics.completion_tokens,
                        "prompt_tps": round(chunk_metrics.prompt_tps, 1),
                        "generation_tps": round(chunk_metrics.generation_tps, 1),
                        "acceptance_ratio": round(chunk_metrics.acceptance_ratio, 2),
                    }
                if not put_from_thread(event):
                    break
        except Exception as exc:
            if not cancelled.is_set():
                put_from_thread((error_tag, f"{type(exc).__name__}: {exc}"))
        finally:
            if source is not None:
                close = getattr(source, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
            if acquired_gpu:
                gpu_lock.release()
            if not cancelled.is_set():
                put_from_thread(sentinel)
            try:
                from mio import server as server_module

                server_module._unregister_stream_producer(threading.current_thread())
            except Exception:
                pass

    producer_thread = threading.Thread(
        target=produce,
        name=f"mio-ui-{tier}-{uuid.uuid4().hex[:8]}",
        daemon=True,
    )
    try:
        from mio import server as server_module

        server_module._register_stream_producer(producer_thread, cancelled)
    except Exception:
        pass
    try:
        producer_thread.start()
    except BaseException:
        cancelled.set()
        try:
            from mio import server as server_module

            server_module._unregister_stream_producer(producer_thread)
        except Exception:
            pass
        raise

    async def send_json(payload: dict) -> None:
        try:
            await websocket.send_json(payload)
        except BaseException:
            cancelled.set()
            raise

    chunks: list[str] = []
    metrics: dict | None = None
    completed = False
    failed = False
    try:
        if not is_first:
            await send_json({"type": "followup_start"})
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                await send_json({"type": "keepalive"})
                continue
            if item is sentinel:
                completed = True
                break
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and item[0] is error_tag
            ):
                failed = True
                await send_json({"type": "error", "message": item[1]})
                break
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(text)
                item_metrics = item.get("metrics")
                if isinstance(item_metrics, dict):
                    metrics = item_metrics
                await send_json(item)
    finally:
        cancelled.set()
        if producer_thread.is_alive():
            await asyncio.to_thread(
                producer_thread.join,
                _WS_STREAM_JOIN_TIMEOUT_SECONDS,
            )

    return _WebUIRoundResult(
        text="".join(chunks),
        metrics=metrics,
        completed=completed,
        failed=failed,
    )


async def _handle_chat(websocket: WebSocket, data: dict):
    """Handle a chat request over WebSocket with streaming and skill execution."""
    import asyncio

    try:
        max_tokens = _validate_webui_max_tokens(
            data["max_tokens"] if "max_tokens" in data else _max_tokens
        )
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        return

    messages = data.get("messages", [])
    # /remember <fact> — store in persistent memory without consulting the model
    if messages:
        last = messages[-1]
        if last.get("role") == "user":
            c = (last.get("content") or "").strip()
            m = re.match(r"^/remember\s+(.+)", c, re.IGNORECASE | re.DOTALL)
            if m:
                fact = m.group(1).strip()
                memory_entry = {
                    "id": str(uuid.uuid4())[:8],
                    "text": fact,
                    "added": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                _update_memory(lambda mem: [*mem, memory_entry])
                await websocket.send_json({"type": "start", "tier": "memory"})
                await websocket.send_json({"type": "token", "text": "✓ Saved to memory: \"" + fact + "\""})
                await websocket.send_json({"type": "done", "full_text": "✓ Saved to memory: \"" + fact + "\"",
                                           "metrics": {"completion_tokens": 0, "generation_tps": 0, "prompt_tokens": 0, "prompt_tps": 0, "acceptance_ratio": 0}})
                return
    tier = data.get("tier")
    system_prompt = data.get("system_prompt") or _system_prompt
    prompt_policy = _resolve_chat_prompt_policy(data)
    use_skills = data.get("skills", True)
    requested_skill_grants = data.get("skill_grants", [])
    if not isinstance(requested_skill_grants, list):
        requested_skill_grants = []
    requested_skill_grants = [str(name) for name in requested_skill_grants[:32]]
    temperature = data.get("temperature")
    if temperature is None:
        temperature = _temperature

    manager = _manager
    if manager is None:
        await websocket.send_json({"type": "error", "message": "No model loaded"})
        return

    # Select a tier, but do not retain an engine reference here. The stream
    # producer resolves it after acquiring the lifecycle GPU lock.
    loaded = manager.loaded_tiers()
    if not loaded:
        await websocket.send_json({"type": "error", "message": "No tiers loaded"})
        return
    tier = tier if tier in loaded else loaded[0]

    # Inject skills as tools
    tools = None
    if use_skills:
        from mio.webui.skills import SKILLS, get_tools_spec
        from mio.web_security import webui_model_skill_allowed

        tools = get_tools_spec(
            allowed_names={
                name
                for name in SKILLS
                if webui_model_skill_allowed(name, requested_skill_grants)
            }
        )

    # Build the effective system prompt. We always inject today's date and
    # (when tools are enabled) a short browsing protocol, then append any
    # user-configured custom system prompt.
    import datetime as _dt
    today_iso = _dt.date.today().isoformat()
    base_sys = f"Today's date is {today_iso}."
    if use_skills:
        base_sys += (
            "\n\nBrowsing protocol: when a user asks about current events, "
            "dates, schedules, releases, news, or any specific fact that "
            "may have changed since your training, you MUST:\n"
            "1. Call web_search with a focused query.\n"
            "2. Call fetch_url on the 2-3 most relevant results.\n"
            "3. Base the answer on the fetched content only. If every "
            "fetch_url returns an error, say so — do NOT fabricate.\n\n"
            "Artifacts — CRITICAL RULE:\n"
            "Whenever the user asks you to GENERATE, CREATE, BUILD, SHOW, "
            "or MAKE a web page, webpage, pagina web, sito web, interactive "
            "widget, animated page, dashboard, visualization, diagram, SVG, "
            "3D scene, mindmap, timeline, map, chart, or ANY visual/"
            "interactive content — you MUST wrap the output in "
            "<antArtifact> tags. Never dump raw HTML into the chat body; "
            "the chat is markdown only — raw HTML will not render.\n\n"
            "Required format:\n"
            "  <antArtifact identifier=\"short-unique-slug\" "
            "type=\"text/html\" title=\"Short descriptive title\">\n"
            "  <!doctype html>\n"
            "  <html><head>...</head><body>...</body></html>\n"
            "  </antArtifact>\n\n"
            "Outside the tag, keep a brief sentence describing what you "
            "made. Inside the tag, produce the COMPLETE document — no "
            "ellipsis, no truncation. Use the same conventions as Claude "
            "Artifacts: full-page HTML is sandboxed and runs freely "
            "(scripts, fetches to same-origin, CDN imports all work).\n\n"
            "Supported type values: text/html, image/svg+xml, text/markdown, "
            "application/vnd.ant.mermaid, application/vnd.ant.react, "
            "application/vnd.ant.code (include `language` attribute for code), "
            "application/vnd.pimio.weather (for get_weather results), "
            "application/pdf (body = generated PDF filename — UI embeds it), "
            "application/vnd.pimio.image (body = image filename), "
            "application/vnd.pimio.file (body = filename, for .docx/.xlsx/.pptx).\n"
            "Visual/interactive types (use them when the task fits):\n"
            "  - application/vnd.pimio.threejs — body: JS that adds meshes "
            "to the pre-created `scene` (scene, camera, renderer, controls, "
            "lights are already set up). Optionally define `update = (t) => "
            "{...}` for per-frame animation.\n"
            "  - application/vnd.pimio.p5 — body: standard p5.js code with "
            "setup()/draw() functions.\n"
            "  - application/vnd.pimio.chartjs — body: a Chart.js config JSON "
            "(with type, data, options). UI calls new Chart(canvas, body).\n"
            "  - application/vnd.pimio.leaflet — body: JSON "
            "{center:[lat,lon], zoom, markers:[{latlng:[...], popup, tooltip}], "
            "polylines:[...], geojson?}.\n"
            "  - application/vnd.pimio.math — body: text with $inline$ and "
            "$$display$$ LaTeX, rendered via KaTeX.\n"
            "  - application/vnd.pimio.graphviz — body: DOT source.\n"
            "  - application/vnd.pimio.mindmap — body: markdown outline "
            "(nested bullet lists or headings) → markmap radial mind map.\n"
            "  - application/vnd.pimio.revealjs — body: markdown slides "
            "separated by lines containing only '---'.\n"
            "  - application/vnd.pimio.timeline — body: JSON {items:[{id, "
            "content, start, end?}], options?} for vis-timeline.\n"
            "  - application/vnd.pimio.pyrepl — body: Python code. An "
            "in-browser Pyodide REPL runs it with numpy, pandas, matplotlib "
            "available. Use for 'show me Python that does X' with live "
            "execution the user can edit and re-run.\n"
            "  - application/vnd.pimio.tone — body: JS using Tone.js. "
            "User clicks ▶ to start audio context. Great for 'play a "
            "melody / make a synth / drum loop'.\n"
            "  - application/vnd.pimio.shader — body: GLSL fragment shader "
            "source. Uniforms u_resolution, u_time, u_mouse are provided. "
            "Use for generative visuals.\n"
            "  - application/vnd.pimio.jsonviewer — body: raw JSON string. "
            "Rendered as a collapsible tree. Use for API responses or "
            "structured data the user wants to explore.\n"
            "  - application/vnd.pimio.table — body: JSON, either an array "
            "of row-dicts or {headers:[...], rows:[[...]]}. Rendered as a "
            "sortable + filterable table.\n"
            "  - application/vnd.pimio.diff — body: JSON {name?, oldStr, "
            "newStr}. Side-by-side diff view via diff2html.\n"
            "  - application/vnd.pimio.regex — body: JSON {pattern, flags, "
            "test}. Live regex tester with match highlighting.\n"
            "  - application/vnd.pimio.piano — body: ignored; renders an "
            "interactive piano keyboard (click or A-L/W-U keys).\n"
            "  - application/vnd.pimio.flashcards — body: JSON {cards:"
            "[{front, back}]} or [{q, a}]. Click/space to flip, arrows to "
            "navigate.\n"
            "  - application/vnd.pimio.kanban — body: JSON {columns:"
            "[{name, cards:[...] }]}. Drag cards between columns.\n"
            "  - application/vnd.pimio.palette — body: JSON {title?, "
            "colors:[{name, hex}]}. Clickable color swatches that copy.\n"
            "  - application/vnd.pimio.whiteboard — body: ignored; "
            "renders an interactive drawing canvas with pen/eraser tools.\n"
            "  - application/vnd.pimio.pomodoro — body: JSON {focus, "
            "break, longBreak, rounds} minutes. Live Pomodoro timer.\n"
            "  - application/vnd.pimio.gradient — body: JSON {gradients:"
            "[{name, css}]} — clickable gradient previews that copy CSS.\n"
            "  - application/vnd.pimio.countdown — body: JSON {title, "
            "target: ISO-datetime}. Live days/hours/min/sec countdown.\n"
            "  - application/vnd.pimio.qrview — body: any text/URL string. "
            "Large QR rendered in-panel (for scanning from your phone).\n"
            "  - application/vnd.pimio.wavedrom — body: WaveDrom JSON "
            "(signal timing diagrams for digital design).\n"
            "  - application/vnd.pimio.physics — body: Matter.js JS. "
            "Pre-wired engine/render/runner/mouse drag/ground; model adds "
            "bodies via Bodies.rectangle/circle and World.add.\n"
            "  - application/vnd.pimio.graph — body: JSON {nodes:"
            "[{id, label?, r?, color?}], links:[{source, target}]}. "
            "d3-force network graph, draggable nodes.\n"
            "  - application/vnd.pimio.plantuml — body: PlantUML source. "
            "Rendered via kroki.io (needs internet).\n"
            "  - application/vnd.pimio.jscad — body: JSCAD code. Must "
            "define `function main() { return cube(...); }` returning a "
            "geom3 or array of them. Primitives/booleans/transforms are "
            "pre-imported. Rendered in 3D with OrbitControls.\n"
            "  - application/vnd.pimio.modelviewer — body: URL or JSON "
            "{src, poster?, background?}. Renders a GLB/GLTF model with "
            "orbit controls and iOS AR Quick Look support.\n"
            "  - application/vnd.pimio.excalidraw — body: ignored; embeds "
            "an Excalidraw whiteboard for hand-drawn diagrams.\n"
            "  - application/vnd.pimio.audio — body: URL or JSON "
            "{url, title}. Wavesurfer.js waveform player.\n"
            "  - application/vnd.pimio.youtube — body: YouTube URL or 11-"
            "char video ID. Embeds the player.\n"
            "  - application/vnd.pimio.terminal — body: JSON array of "
            "{prompt, cmd, output} entries. Types out a fake terminal "
            "session for demos/tutorials.\n"
            "React artifacts have lucide-react, recharts, and framer-motion "
            "available — import via `import { Heart } from 'lucide-react'` "
            "etc. Define a top-level `function App() { ... }`.\n"
            "Keep narrative text OUTSIDE the artifact tag — the tag contains "
            "ONLY the rendered content.\n\n"
            "Weather requests: after calling get_weather, emit an "
            "application/vnd.pimio.weather artifact containing the full "
            "JSON result verbatim — the UI turns it into an animated widget "
            "with location, current temp, hourly strip, and 7-day forecast.\n\n"
            "Visual / interactive explainers — when the user asks to "
            "'explain visually', 'show me visually', 'interactive "
            "explainer', 'walk me through with diagrams', 'visualize', "
            "'teach me with a playground', 'demo', 'simulator', or "
            "similar: DO NOT generate a PDF. Emit ONE inline artifact:\n"
            "  • application/vnd.ant.react for anything interactive "
            "(buttons, sliders, step-throughs, mini-simulators). Define "
            "`function App() { ... }` that manages state and renders an "
            "animated / clickable scene with Tailwind classes.\n"
            "  • text/html for a self-contained page with custom JS + "
            "animations (Canvas, SVG, CSS transitions).\n"
            "  • image/svg+xml for a single static diagram with labels.\n"
            "  • application/vnd.ant.mermaid for flow/sequence/class "
            "diagrams.\n"
            "  • application/vnd.pimio.threejs for a 3D scene.\n"
            "A visual explainer should be one RICH inline artifact, not a "
            "PDF + separate artifact. PDFs are for deliverables the user "
            "will print, email, or save — NOT for learning / exploration.\n\n"
            "Document requests — pick the skill that matches the DOCUMENT "
            "TYPE the user asked for; don't default to generate_pdf_report "
            "for everything:\n"
            "  • Report / whitepaper / analysis → generate_pdf_report\n"
            "  • Formal letter / cover letter / complaint → generate_letter\n"
            "  • Award / diploma / recognition → generate_certificate\n"
            "  • Event poster / promo / announcement → generate_flyer\n"
            "  • Restaurant menu → generate_menu\n"
            "  • Tri-fold marketing brochure → generate_brochure\n"
            "  • Company / team newsletter → generate_newsletter\n"
            "  • Business card → generate_business_card\n"
            "  • Resume / CV → generate_resume\n"
            "  • Invoice / bill → generate_invoice\n"
            "  • Word / Excel / PowerPoint → generate_docx / generate_xlsx / "
            "generate_pptx (only when the user explicitly asks for that "
            "format).\n"
            "VISUAL STYLE IS AUTOMATIC — do NOT set `preset` or `color` "
            "on the FIRST call. The skill picks an appropriate look from "
            "~60 presets based on the document type and the words in the "
            "title / body (legal letter → serif corporate; birthday flyer "
            "→ bubblegum; tech report → carbon; etc.).\n"
            "REFINEMENT rules when the user asks to change the look:\n"
            "  • 'make it {color}' (green, dark blue, red, pink, etc.) → "
            "regenerate with the SAME `preset` the skill used last time "
            "PLUS `color=\"<that name>\"`. Color override keeps the "
            "layout/decoration/fonts and only swaps the palette. Color "
            "names accept aliases: 'blue' → azure, 'dark green' → forest, "
            "'hot pink' → magenta, 'gold' → brass, 'gray' → slate, etc.\n"
            "  • 'make it {darker,warmer,more minimal,more formal, etc.}' "
            "→ switch `preset` to a matching one (keep `color` unset so "
            "that preset's own palette is used).\n"
            "  • 'more playful' / 'more corporate' → switch preset.\n"
            "  • User gives SPECIFIC color instructions ('background light "
            "green', 'text black', 'accent red', 'pagina con sfondo chiaro') "
            "→ use the surgical params `background_color` / `text_color` / "
            "`accent_color` on `generate_pdf_report`. These win over preset "
            "and palette, so they actually respect the user's exact colors "
            "instead of just tinting the existing preset. Accepts names "
            "('black', 'light green', 'mint', 'cream') or hex ('#09090b'). "
            "Pass ONLY the ones the user specified — leave the others unset "
            "so the preset keeps them.\n"
            "Don't invent preset names — pick one from the schema enum. "
            "Don't invent color names — use the palette or aliases above.\n"
            "Only fall back to generate_pdf (plain fpdf2) when the user "
            "insists on the simplest possible PDF.\n\n"
            "Media recommendations — WHEN to use find_anime / find_manga "
            "/ find_movie_tv / find_game:\n"
            "  ✓ User EXPLICITLY asks to find / recommend / suggest titles: "
            "'find me an anime like …', 'recommend a movie', 'suggest some "
            "manga', 'what game should I play'.\n"
            "  ✓ User names a title they clearly know is in that medium and "
            "wants info about THAT TITLE: 'tell me about Attack on Titan', "
            "'what is Breaking Bad about'.\n"
            "  ✗ DO NOT use these skills for:\n"
            "    - Ambiguous / proper-noun queries where it's unclear if "
            "the subject is even a show (VTubers, streamers, idols, bands, "
            "people, companies, products). For 'who is X?' or 'what is X?' "
            "where X could be a person — ALWAYS call web_search FIRST. "
            "Example: 'who is koseki bijou' is a Hololive VTuber, NOT an "
            "anime — answer it with web_search + fetch_url.\n"
            "    - User explicitly says 'search on internet' / 'search the "
            "web' / 'look it up online' — honor this verbatim with "
            "web_search, never call find_*.\n"
            "Pass query EXACTLY as the user typed it. Skill rules:\n"
            "  - Anime  → find_anime (NEVER find_movie_tv). Do NOT "
            "translate titles to Japanese — Jikan matches English fine.\n"
            "  - Manga  → find_manga.\n"
            "  - Movies / TV shows → find_movie_tv.\n"
            "  - Video games → find_game.\n"
            "These skills return real, working poster URLs. The UI auto-"
            "renders a mediacard artifact from the result.\n"
            "If the user corrects you ('no, search online' / 'I meant the "
            "VTuber' / 'that's not what I asked'), switch to web_search "
            "immediately — do NOT re-run the wrong find_* skill.\n\n"
            "CRITICAL — NEVER emit an <antArtifact> tag for any of these "
            "types (the server auto-emits them from REAL tool results):\n"
            "  • image/png, image/jpeg, image/webp, image/gif (any image/*)\n"
            "  • application/vnd.pimio.imagegrid\n"
            "  • application/vnd.pimio.mediacard\n"
            "  • application/vnd.pimio.weather\n"
            "URLs you write into these artifacts are hallucinations and "
            "will 404. If the user asks for a poster, a weather widget, "
            "or an image gallery: call the matching skill and STOP — the "
            "UI will render the artifact for you automatically.\n\n"
            "After the tool call, write 1-3 sentences in the chat body "
            "about the title (genre, vibe, why they'd enjoy it). The "
            "poster and synopsis are already visible in the side panel.\n\n"
            "Video / trailer / tutorial requests: call search_youtube (no "
            "key, scrapes YouTube). The UI auto-renders a youtubegrid "
            "artifact with clickable thumbnails that open an embedded "
            "player. Do NOT emit youtubegrid yourself.\n"
            "Anime trailers are ALREADY returned by find_anime as "
            "trailer_id — the mediacard auto-surfaces a ▶ Trailer button "
            "on each card. No extra call needed."
        )
    # Persistent memory injected as "Known facts about the user" block
    mem_entries = _load_memory()
    if mem_entries:
        base_sys += "\n\nKnown facts about the user (auto-injected memory):\n" + \
            "\n".join(f"- {m.get('text','')}" for m in mem_entries if m.get('text'))
    # Project context: system_prompt + attached files
    project_id = data.get("project_id")
    if project_id:
        for proj in _load_projects():
            if proj.get("id") == project_id:
                if proj.get("system_prompt"):
                    base_sys += "\n\nProject '" + proj.get("name", "") + "' context:\n" + proj["system_prompt"]
                # Attach project files as extracted text (PDFs) or raw
                try:
                    downloads = _validated_downloads_dir()
                except UnsafePathError:
                    downloads = None
                for fname in proj.get("files", [])[:64]:
                    if (
                        downloads is None
                        or not isinstance(fname, str)
                        or _safe_upload_name(fname) != fname
                    ):
                        continue
                    try:
                        fpath = confined_path(
                            downloads,
                            fname,
                            must_exist=True,
                            allow_nested=False,
                        )
                    except UnsafePathError:
                        continue
                    try:
                        if fname.lower().endswith(".pdf"):
                            import pdfplumber
                            with open_binary_no_follow(
                                fpath,
                                max_bytes=_MAX_PROJECT_FILE_BYTES,
                            ) as source:
                                with pdfplumber.open(source) as pdf:
                                    text = "\n\n".join(
                                        (page.extract_text() or "")
                                        for page in pdf.pages[:20]
                                    )
                        else:
                            text = read_text_no_follow(
                                fpath,
                                max_bytes=_MAX_PROJECT_FILE_BYTES,
                            )
                        if text.strip():
                            base_sys += f"\n\n=== Project file: {fname} ===\n{text[:8000]}\n=== end {fname} ==="
                    except Exception:
                        # Project attachments are optional context; malformed
                        # third-party PDF/text inputs must not abort the chat.
                        pass
                break
    # Style preset
    style = data.get("style") or ""
    STYLE_HINTS = {
        "concise": "Style: be concise. Answer in ≤3 sentences unless the user asks for more.",
        "detailed": "Style: be detailed and thorough. Prefer complete explanations.",
        "formal": "Style: use formal tone, full sentences, no colloquialisms.",
        "eli5": "Style: explain like I'm 5. Use simple words and analogies.",
        "code-only": "Style: answer with code only. Minimal prose. No explanations unless asked.",
    }
    if style and style in STYLE_HINTS:
        base_sys += "\n\n" + STYLE_HINTS[style]
    if system_prompt:
        base_sys = base_sys + "\n\n" + system_prompt
    has_system = any(m.get("role") == "system" for m in messages)
    if has_system:
        for m in messages:
            if m.get("role") == "system":
                m["content"] = base_sys + "\n\n" + (m.get("content") or "")
                break
    else:
        messages = [{"role": "system", "content": base_sys}] + messages

    messages = apply_prompt_policy(messages, prompt_policy)

    # Snapshot the final system prompt for /api/debug/last-prompt
    try:
        global _last_system_prompt
        _sys = next((m.get("content") for m in messages if m.get("role") == "system"), None)
        _last_system_prompt = (
            f"# prompt={prompt_policy.label} · style={style!r} · project={project_id!r} · "
            f"skills={use_skills}\n\n{_sys or '(no system message)'}"
        )
    except Exception:
        pass

    # Send start event
    await websocket.send_json({"type": "start", "tier": tier})

    def _summarize_tool_results(tool_results: list[dict]) -> str:
        """Render tool results as plain text the model can consume on the
        next round. Keep URLs intact so the model can fetch_url on them.
        """
        lines: list[str] = []
        # Count citation-worthy sources so we can tell the model to use
        # inline [N] refs. Each web_search result + each fetch_url becomes
        # a citation candidate.
        _citations: list[dict] = []
        for tr in tool_results:
            skill = tr.get("skill")
            if skill == "web_search":
                for r in (tr.get("result") or {}).get("results", []) or []:
                    _citations.append({"title": r.get("title", ""), "url": r.get("url", ""), "domain": r.get("domain", "")})
            elif skill == "fetch_url":
                _citations.append({"title": tr.get("args", {}).get("url", ""), "url": tr.get("args", {}).get("url", ""), "domain": ""})
        if _citations:
            lines.append("Cite sources with [N] inline where N is 1-indexed into this list:")
            for i, c in enumerate(_citations, 1):
                lines.append(f"  [{i}] {c['title']} — {c['url']}")
        for tr in tool_results:
            skill = tr["skill"]
            args = tr["args"]
            res = tr["result"]
            if skill == "web_search" and res.get("results"):
                lines.append(f"web_search(\"{args.get('query', '')}\") →")
                for r in res["results"]:
                    lines.append(f"  - {r.get('title','')}  [{r.get('url','')}]")
                    if r.get("snippet"):
                        lines.append(f"    {r['snippet']}")
            elif skill == "fetch_url":
                url = args.get("url", "")
                if res.get("error"):
                    lines.append(f"fetch_url({url}) → ERROR: {res['error']}")
                else:
                    content = (res.get("content") or "")[:3500]
                    trunc = " [truncated]" if res.get("truncated") else ""
                    lines.append(f"fetch_url({url}){trunc} →\n{content}")
            elif skill in ("generate_pdf", "generate_pdf_report",
                           "generate_docx", "generate_xlsx", "generate_pptx",
                           "generate_chart", "generate_qr_code",
                           "generate_resume", "generate_invoice",
                           "generate_csv", "generate_ical",
                           "generate_sqlite_db",
                           "generate_letter", "generate_certificate",
                           "generate_flyer", "generate_menu",
                           "generate_brochure", "generate_newsletter",
                           "generate_business_card", "generate_markdown"):
                path = res.get("path") or ""
                err = res.get("error") or ""
                if err:
                    lines.append(f"{skill} → ERROR: {err}")
                elif path:
                    fname = path.rsplit("/", 1)[-1]
                    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                    if ext == "pdf":
                        art_type = "application/pdf"
                    elif ext in ("png", "jpg", "jpeg"):
                        art_type = "application/vnd.pimio.image"
                    else:
                        art_type = "application/vnd.pimio.file"
                    lines.append(
                        f"{skill} → saved at {path}. Now emit an artifact "
                        f"so the user can view it:\n"
                        f"<antArtifact identifier=\"{fname.rsplit('.',1)[0]}\" "
                        f"type=\"{art_type}\" title=\"{fname}\">{fname}"
                        f"</antArtifact>\n"
                        "Include the filename as the artifact body. The UI "
                        "resolves it against /ui/files/ automatically."
                    )
                else:
                    lines.append(f"{skill} → {json.dumps(res)[:400]}")
            elif skill == "get_weather":
                if res.get("error"):
                    lines.append(f"get_weather → ERROR: {res['error']}")
                else:
                    loc = res.get("location", {})
                    lines.append(
                        f"get_weather → {loc.get('name','?')}, "
                        f"{loc.get('country','')} — current "
                        f"{res.get('current',{}).get('temp')}°. The widget "
                        "artifact has already been rendered to the side panel "
                        "(do NOT re-emit it). Write a brief natural-language "
                        "summary of the current conditions for the chat body."
                    )
            elif skill in ("find_anime", "find_manga", "find_movie_tv", "find_game"):
                if res.get("error"):
                    lines.append(f"{skill} → ERROR: {res['error']}")
                else:
                    items = res.get("results", [])
                    summary = "; ".join(
                        f"{i.get('title','?')}"
                        + (f" ({i.get('year') or (i.get('premiered') or '')[:4]})" if i.get('year') or i.get('premiered') else "")
                        + (f" · score {i.get('score') or i.get('rating')}" if i.get('score') or i.get('rating') else "")
                        for i in items[:5]
                    )
                    lines.append(
                        f"{skill} → {len(items)} results: {summary}. "
                        "The poster-card artifact has already been rendered to "
                        "the side panel (do NOT emit an <antArtifact> tag). "
                        "Write a concise natural-language response describing "
                        "the title(s) and why the user might enjoy them — the "
                        "user can already see posters, synopses, and scores "
                        "in the panel."
                    )
            elif skill == "search_images":
                if res.get("error"):
                    lines.append(f"search_images → ERROR: {res['error']}")
                else:
                    items = res.get("results", [])
                    lines.append(
                        f"search_images → {len(items)} images. The image-grid "
                        "artifact is already rendered in the side panel (do "
                        "NOT emit an <antArtifact>). Write a short caption "
                        "for the chat body."
                    )
            elif skill == "execute_python":
                out = (res.get("stdout") or "")[:1000]
                err = (res.get("stderr") or "")[:500]
                lines.append(f"execute_python →\nstdout: {out}\nstderr: {err}")
            else:
                lines.append(f"{skill} → {json.dumps(res)[:600]}")
        return "\n".join(lines)

    # --- Tool-use loop -----------------------------------------------------
    # Generate → parse tool calls → execute → feed results back → repeat.
    # Stop when the model emits a response with no tool calls, or when we
    # hit MAX_ROUNDS to avoid runaway loops.
    MAX_ROUNDS = 5
    current_messages = list(messages)
    parse_tool_calls = None
    execute_skill = None
    if use_skills:
        from mio.tool_calls import parse_tool_calls as _ptc
        from mio.webui.skills import execute_skill as _es
        parse_tool_calls = _ptc
        execute_skill = _es

    turn_start = time.time()
    turn_text_parts: list[str] = []
    for round_idx in range(MAX_ROUNDS):
        round_result = await _stream_webui_round(
            websocket,
            manager=manager,
            tier=tier,
            gpu_lock=_gpu_lock,
            messages=current_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            is_first=(round_idx == 0),
        )
        if round_result.failed or not round_result.completed:
            # The round already emitted one error frame when appropriate.
            # Never follow it with a misleading `done` success event.
            return
        full_text_str = round_result.text
        final_metrics = round_result.metrics
        # Strip any model-emitted artifact whose contents should only
        # come from server-side auto_artifact events (i.e. real tool
        # results). These types are:
        #   - image/png, image/jpeg, image/webp, image/gif, image/avif —
        #     URLs get hallucinated
        #   - application/vnd.pimio.imagegrid   — same
        #   - application/vnd.pimio.mediacard   — poster URLs get hallucinated
        #   - application/vnd.pimio.weather     — needs real Open-Meteo data
        # EXPLICITLY allowed (generated inline, no URL, safe):
        #   - image/svg+xml — model authors the SVG markup directly
        import re as _re_
        _BLOCKED_TYPES = (
            r'image/(?:png|jpeg|jpg|webp|gif|avif)'
            r'|application/vnd\.pimio\.imagegrid'
            r'|application/vnd\.pimio\.mediacard'
            r'|application/vnd\.pimio\.weather'
            r'|application/vnd\.pimio\.youtubegrid'
        )
        full_text_str = _re_.sub(
            r'<antArtifact[^>]*\btype="(?:' + _BLOCKED_TYPES + r')"[^>]*>[\s\S]*?</antArtifact>',
            '[Blocked fabricated media artifact — use find_anime / find_manga / '
            'find_movie_tv / find_game / search_images / get_weather so real '
            'data is fetched.]',
            full_text_str,
        )
        turn_text_parts.append(full_text_str)

        tool_calls = []
        if use_skills and "<tool_call>" in full_text_str and parse_tool_calls is not None:
            _, tool_calls = parse_tool_calls(full_text_str)

        tool_results: list[dict] = []
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                fn_args = {}
            await websocket.send_json({
                "type": "skill_start", "skill": fn_name, "args": fn_args,
            })
            from mio.web_security import webui_model_skill_allowed, webui_skill_risk

            if not webui_model_skill_allowed(fn_name, requested_skill_grants):
                result = {
                    "error": "WebUI model tool denied by skill policy",
                    "skill": fn_name,
                    "risk": webui_skill_risk(fn_name),
                }
            else:
                from mio.web_security import model_request_skill_grants

                def execute_with_request_grants():
                    with model_request_skill_grants(requested_skill_grants):
                        return execute_skill(fn_name, fn_args)

                result = await asyncio.to_thread(execute_with_request_grants)
            tool_results.append({"skill": fn_name, "args": fn_args, "result": result})
            await websocket.send_json({
                "type": "skill_result", "skill": fn_name, "result": result,
            })
            # Auto-emit artifact for skills where the model consistently fumbles
            # the JSON relay. The client injects it into the chat as an
            # <antArtifact> in the next message so version history + source
            # tab still work.
            auto_art = _auto_artifact_from_skill(fn_name, fn_args, result)
            if auto_art:
                await websocket.send_json({
                    "type": "auto_artifact", **auto_art,
                })

        done_event = {"type": "done", "full_text": full_text_str}
        if final_metrics:
            done_event["metrics"] = final_metrics
        if tool_results:
            done_event["tool_results"] = tool_results
        await websocket.send_json(done_event)

        if not tool_results:
            break

        summary = _summarize_tool_results(tool_results)
        current_messages = list(current_messages) + [
            {"role": "assistant", "content": full_text_str},
            {
                "role": "user",
                "content": (
                    "Tool results:\n\n" + summary + "\n\n"
                    "If you need more information (e.g. fetch an additional "
                    "URL), emit another <tool_call>. Otherwise, answer the "
                    "original question using ONLY the fetched content above. "
                    "Do not invent facts."
                ),
            },
        ]
    else:
        await websocket.send_json({
            "type": "system",
            "message": f"Reached tool-use round limit ({MAX_ROUNDS}).",
        })

    # Report the turn to the shared _stats collector so `mio serve`'s
    # rich Live panel reflects webui traffic (was previously only seeing
    # OpenAI-compatible /v1/chat/completions requests).
    try:
        from mio import server as _srv
        class _MetricsObj:
            def __init__(self, fm):
                self.prompt_tokens = int(fm.get("prompt_tokens", 0) or 0)
                self.completion_tokens = int(fm.get("completion_tokens", 0) or 0)
                self.prompt_tps = float(fm.get("prompt_tps", 0) or 0)
                self.generation_tps = float(fm.get("generation_tps", 0) or 0)
                self.acceptance_ratio = float(fm.get("acceptance_ratio", 0) or 0)
                self.warm_offset = 0
                self.cache_entries = 0
        if final_metrics:
            _srv._stats.record(_MetricsObj(final_metrics), time.time() - turn_start,
                               tier, "".join(turn_text_parts))
    except Exception:
        pass
