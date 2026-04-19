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

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/ui")

# --- Globals set by mount_webui() ---
_manager = None
_caveman_level = "full"
_gpu_lock = threading.Lock()
_sessions_dir: Path | None = None
_system_prompt: str | None = None
_temperature: float = 0.6  # Qwen3.5/3.6 recommended for coding
_max_tokens: int = 16384


def mount_webui(manager, *, caveman_level: str = "full", gpu_lock=None, sessions_dir: Path | None = None):
    """Initialize the webui module with server state."""
    global _manager, _caveman_level, _gpu_lock, _sessions_dir
    _manager = manager
    _caveman_level = caveman_level
    if gpu_lock is not None:
        _gpu_lock = gpu_lock
    _sessions_dir = sessions_dir or Path.home() / ".mio" / "sessions"
    _sessions_dir.mkdir(parents=True, exist_ok=True)
    # Start the scheduler (asyncio loop); safe if no loop is running yet
    from mio.webui import scheduler as _sched
    _sched.init(manager)


# --- UI page ---

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "mio_ui.html"
    # Prevent the browser from caching the shell — asset modules are
    # referenced by path and will pick up their own Cache-Control headers.
    return HTMLResponse(
        html_path.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache"},
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
    result = []
    for name, spec in sorted(SKILLS.items()):
        result.append({
            "name": name,
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}),
        })
    return {"skills": result}


@router.post("/api/skills/run")
async def run_skill(body: dict):
    """Execute a single skill directly for the playground. Not a normal
    chat path — just runs the skill function with the given JSON args.
    """
    from mio.webui.skills import SKILLS, execute_skill
    name = (body or {}).get("name", "")
    args = (body or {}).get("args", {}) or {}
    if name not in SKILLS:
        return {"error": f"unknown skill: {name}"}
    try:
        result = execute_skill(name, args)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --- Config API ---

# --- Prompts library (user-defined reusable prompts) ---
_prompts_file: Path | None = None


def _prompts_path() -> Path:
    global _prompts_file
    if _prompts_file is None:
        _prompts_file = Path.home() / ".mio" / "prompts.json"
    return _prompts_file


def _load_prompts() -> list:
    p = _prompts_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_prompts(prompts: list) -> None:
    p = _prompts_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))


@router.get("/api/prompts")
async def list_prompts():
    return {"prompts": _load_prompts()}


@router.post("/api/prompts")
async def save_prompt(body: dict):
    prompts = _load_prompts()
    pid = body.get("id") or str(uuid.uuid4())[:8]
    entry = {
        "id": pid,
        "name": body.get("name", "Untitled"),
        "body": body.get("body", ""),
        "slash": body.get("slash", ""),  # optional custom /slash keyword
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    prompts = [p for p in prompts if p.get("id") != pid]
    prompts.append(entry)
    _save_prompts(prompts)
    return entry


@router.delete("/api/prompts/{pid}")
async def delete_prompt(pid: str):
    prompts = [p for p in _load_prompts() if p.get("id") != pid]
    _save_prompts(prompts)
    return {"ok": True}


# --- Persistent memory (facts the model remembers across chats) ---
def _memory_path() -> Path:
    return Path.home() / ".mio" / "memory.json"


def _load_memory() -> list:
    p = _memory_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_memory(mem: list) -> None:
    p = _memory_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mem, indent=2, ensure_ascii=False))


@router.get("/api/memory")
async def list_memory():
    return {"memory": _load_memory()}


@router.post("/api/memory")
async def add_memory(body: dict):
    mem = _load_memory()
    mid = body.get("id") or str(uuid.uuid4())[:8]
    entry = {"id": mid, "text": body.get("text", ""), "added": time.strftime("%Y-%m-%dT%H:%M:%S")}
    mem = [m for m in mem if m.get("id") != mid]
    mem.append(entry)
    _save_memory(mem)
    return entry


@router.delete("/api/memory/{mid}")
async def delete_memory(mid: str):
    mem = [m for m in _load_memory() if m.get("id") != mid]
    _save_memory(mem)
    return {"ok": True}


# --- Projects (shared system prompt + knowledge base per project) ---
def _projects_path() -> Path:
    return Path.home() / ".mio" / "projects.json"


def _load_projects() -> list:
    p = _projects_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_projects(projs: list) -> None:
    p = _projects_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(projs, indent=2, ensure_ascii=False))


@router.get("/api/projects")
async def list_projects():
    return {"projects": _load_projects()}


@router.post("/api/projects")
async def save_project(body: dict):
    projs = _load_projects()
    pid = body.get("id") or str(uuid.uuid4())[:8]
    entry = {
        "id": pid,
        "name": body.get("name", "Untitled"),
        "description": body.get("description", ""),
        "system_prompt": body.get("system_prompt", ""),
        "color": body.get("color", "#3b82f6"),
        "icon": body.get("icon", ""),           # optional emoji/icon
        "files": body.get("files", []),          # list of filenames in ~/Downloads
        # Optional per-workspace model / context overrides. Unset = use
        # global defaults. The UI reads these to pre-apply a tier before
        # sending the first message of a session in that workspace.
        "tier": body.get("tier") or None,
        "context_window": body.get("context_window") or None,
        "caveman_level": body.get("caveman_level") or None,
        "pinned_prompts": body.get("pinned_prompts", []),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    projs = [p for p in projs if p.get("id") != pid]
    projs.append(entry)
    _save_projects(projs)
    return entry


@router.delete("/api/projects/{pid}")
async def delete_project(pid: str):
    projs = [p for p in _load_projects() if p.get("id") != pid]
    _save_projects(projs)
    return {"ok": True}


@router.get("/api/projects/{pid}")
async def get_project(pid: str):
    for p in _load_projects():
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
        "active_tier": tiers[0] if tiers else None,
        "system_prompt": _system_prompt or "",
        "temperature": _temperature,
        "max_tokens": _max_tokens,
    }


@router.post("/api/config")
async def update_config(body: dict):
    global _caveman_level, _system_prompt, _temperature, _max_tokens
    if "caveman" in body:
        _caveman_level = body["caveman"]
    if "system_prompt" in body:
        _system_prompt = body["system_prompt"] or None
    if "temperature" in body:
        _temperature = float(body["temperature"])
    if "max_tokens" in body:
        _max_tokens = int(body["max_tokens"])
    return {
        "ok": True,
        "caveman": _caveman_level,
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
    import asyncio

    tier = body.get("tier")
    if not tier or not _manager:
        return {"error": "invalid request"}
    if tier not in _manager.config.tiers:
        return {"error": f"unknown tier: {tier}", "available": list(_manager.config.tiers.keys())}
    loaded = _manager.loaded_tiers()
    if tier in loaded:
        return {"ok": True, "tier": tier, "already_loaded": True}

    loop = asyncio.get_event_loop()

    def _switch():
        with _gpu_lock:
            for t in list(loaded):
                _manager.unload_tier(t)
            _manager.load_tier(tier)

    await loop.run_in_executor(None, _switch)
    return {"ok": True, "tier": tier}


# --- Session persistence ---

@router.get("/api/sessions")
async def list_sessions(project_id: str | None = None):
    if not _sessions_dir or not _sessions_dir.exists():
        return {"sessions": []}
    sessions = []
    for f in sorted(_sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
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
    path = _sessions_dir / f"{session_id}.json"
    if not path.exists():
        return {"error": "not found"}
    return json.loads(path.read_text())


@router.post("/api/sessions")
async def save_session(body: dict):
    session_id = body.get("id") or str(uuid.uuid4())[:8]
    path = _sessions_dir / f"{session_id}.json"
    body["id"] = session_id
    body["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if "title" not in body or not body["title"]:
        msgs = body.get("messages", [])
        first_user = next((m["content"][:60] for m in msgs if m.get("role") == "user"), "New Chat")
        body["title"] = first_user
    # project_id optional — sessions can be grouped under a project
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    return {"id": session_id, "title": body["title"], "project_id": body.get("project_id")}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = _sessions_dir / f"{session_id}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


# --- Generated file serving ---
# Skills like generate_pdf, generate_docx, generate_xlsx, generate_pptx,
# generate_chart save outputs to ~/Downloads. The UI needs HTTP access so
# it can embed the PDF in an iframe or offer download links. Restricted to
# plain filenames (no path traversal) from that directory only.

import mimetypes
from fastapi.responses import FileResponse, Response

_ALLOWED_EXT = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".svg",
    ".csv", ".ics", ".sqlite", ".db",
}


# Artifact share: store a handful of artifacts keyed by short IDs so they
# can be viewed read-only at /ui/share/<id>. Kept in-memory — deliberate
# ephemerality so private artifacts don't leak across restarts.
_shared_artifacts: dict[str, dict] = {}


@router.post("/api/share")
async def share_artifact(body: dict):
    art_id = body.get("identifier") or str(uuid.uuid4())[:8]
    _shared_artifacts[art_id] = {
        "type": body.get("type", "text/html"),
        "title": body.get("title", "Artifact"),
        "content": body.get("content", ""),
        "language": body.get("language", ""),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"id": art_id, "url": f"/ui/share/{art_id}"}


@router.get("/share/{art_id}", response_class=HTMLResponse)
async def view_shared_artifact(art_id: str):
    art = _shared_artifacts.get(art_id)
    if not art:
        return HTMLResponse(status_code=404, content="Artifact not found or expired.")
    import html as _html
    j = json.dumps(art)
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
  f.sandbox = 'allow-scripts allow-same-origin';
  f.srcdoc = art.content;
  mount.appendChild(f);
}} else if (art.type === 'image/svg+xml') {{
  mount.innerHTML = art.content;
}} else {{
  const p = document.createElement('pre');
  p.textContent = art.content;
  mount.appendChild(p);
}}</script></body></html>"""
    return HTMLResponse(page)


# --- File upload (attachments) ---
from fastapi import UploadFile, File


@router.post("/api/upload")
async def upload_attachment(file: UploadFile = File(...)):
    """Accept an attachment, store it in ~/Downloads, extract text if PDF/txt/md,
    and return a summary the client can inject into the next message.
    """
    name = (file.filename or "upload").replace("/", "_").replace("\\", "_")
    from pathlib import Path as _Path
    out = _Path.home() / "Downloads" / name
    out.parent.mkdir(exist_ok=True)
    data = await file.read()
    out.write_bytes(data)

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
        from urllib.parse import quote
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
    import hashlib
    import json as _json
    from pathlib import Path as _P

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

    root = _P.home() / ".mio" / "ingest"
    root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = hashlib.sha1(url.encode()).hexdigest()[:8]
    md_path = root / f"{stamp}-{slug}.md"

    frontmatter = (
        "---\n"
        f"title: {title!r}\n"
        f"source: {url}\n"
        f"fetched: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"tags: {tags}\n"
        f"selection_only: {bool(selection)}\n"
        "---\n\n"
    )
    md_path.write_text(frontmatter + f"# {title}\n\n" + effective_text)

    summary = effective_text[:280].replace("\n", " ")
    doc_id = f"{stamp}-{slug}"

    # Add to RAG index if available
    indexed = False
    if target in ("rag", ""):
        try:
            from mio.webui.skills_rag import index_folder
            index_folder(str(root))
            indexed = True
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
    from pathlib import Path as _P
    import re as _re
    root = _P.home() / ".mio" / "ingest"
    if not root.exists():
        return {"items": [], "tags": []}
    items = []
    all_tags: set[str] = set()
    for p in sorted(root.glob("*.md"), reverse=True):
        head = p.read_text()[:2048]
        title = url = ""
        tags: list[str] = []
        m = _re.search(r"^title:\s*['\"]?(.*?)['\"]?$", head, _re.MULTILINE)
        if m: title = m.group(1)
        m = _re.search(r"^source:\s*(.*?)$", head, _re.MULTILINE)
        if m: url = m.group(1)
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

def _obsidian_config_path() -> Path:
    return Path.home() / ".mio" / "obsidian.json"


def _load_obsidian_config() -> dict:
    p = _obsidian_config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_obsidian_config(cfg: dict) -> None:
    p = _obsidian_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def _obsidian_vault() -> Path | None:
    cfg = _load_obsidian_config()
    vp = cfg.get("vault_path")
    if not vp:
        return None
    return Path(vp).expanduser()


def _obsidian_safe_join(vault: Path, rel: str) -> Path | None:
    """Resolve rel under vault, refusing any path that escapes it."""
    if not rel:
        return None
    p = (vault / rel).resolve()
    try:
        p.relative_to(vault.resolve())
    except ValueError:
        return None
    return p


@router.get("/api/obsidian/config")
async def obsidian_get_config():
    cfg = _load_obsidian_config()
    vp = cfg.get("vault_path") or ""
    vault_exists = False
    if vp:
        vault_exists = Path(vp).expanduser().exists()
    return {"vault_path": vp, "vault_exists": vault_exists}


@router.post("/api/obsidian/config")
async def obsidian_set_config(body: dict):
    vp = (body or {}).get("vault_path", "").strip()
    if not vp:
        return {"error": "vault_path required"}
    expanded = Path(vp).expanduser()
    if not expanded.exists():
        return {"error": f"path does not exist: {vp}"}
    _save_obsidian_config({"vault_path": str(expanded)})
    return {"ok": True, "vault_path": str(expanded)}


@router.get("/api/obsidian/tree")
async def obsidian_tree():
    vault = _obsidian_vault()
    if not vault:
        return {"error": "vault not configured"}
    if not vault.exists():
        return {"error": f"vault missing: {vault}"}

    def walk(d: Path, depth: int = 0) -> list:
        if depth > 12:
            return []
        items = []
        try:
            entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return []
        for entry in entries:
            name = entry.name
            if name.startswith(".") or name == "node_modules":
                continue
            rel = str(entry.relative_to(vault))
            if entry.is_dir():
                items.append({
                    "type": "folder",
                    "name": name,
                    "path": rel,
                    "children": walk(entry, depth + 1),
                })
            elif entry.suffix.lower() in (".md", ".markdown"):
                items.append({
                    "type": "note",
                    "name": name,
                    "path": rel,
                    "size": entry.stat().st_size,
                    "mtime": entry.stat().st_mtime,
                })
        return items

    return {"vault_path": str(vault), "tree": walk(vault)}


@router.get("/api/obsidian/note")
async def obsidian_read_note(path: str):
    vault = _obsidian_vault()
    if not vault:
        return {"error": "vault not configured"}
    p = _obsidian_safe_join(vault, path)
    if not p or not p.exists() or not p.is_file():
        return {"error": "note not found"}
    try:
        content = p.read_text()
    except Exception as e:
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
    p = _obsidian_safe_join(vault, rel)
    if not p:
        return {"error": "invalid path"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
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
    from fastapi.responses import StreamingResponse

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


@router.post("/api/reveal")
async def reveal_in_finder(body: dict):
    """Open a local path in the user's file browser (Finder on macOS)."""
    import subprocess, sys
    from pathlib import Path as _P
    path_str = (body or {}).get("path", "").strip() or "~/.mio"
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
    import io, zipfile as _zip
    from pathlib import Path as _P
    from fastapi.responses import StreamingResponse

    root = _P.home() / ".mio"
    if not root.exists():
        return {"error": "no ~/.mio yet"}

    SKIP_DIRS = {"image-cache", "web-cache", "files-cache", "__pycache__"}

    buf = io.BytesIO()
    with _zip.ZipFile(buf, "w", compression=_zip.ZIP_DEFLATED) as z:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
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
    from pathlib import Path as _P
    root = _P.home() / ".mio" / "ingest"
    # Prevent path traversal
    safe = doc_id.replace("/", "_").replace("..", "_")
    target = root / f"{safe}.md"
    if target.exists() and target.is_file() and target.resolve().is_relative_to(root.resolve()):
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
    for f in sorted(_sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
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
import urllib.request as _urlreq
import hashlib as _hl
import mimetypes as _mt
_img_proxy_cache: dict[str, tuple[bytes, str]] = {}

PIMIO_DIR = Path.home() / ".mio"
IMAGE_CACHE_DIR = PIMIO_DIR / "image-cache"
WEB_CACHE_DIR = PIMIO_DIR / "web-cache"
FILES_CACHE_DIR = PIMIO_DIR / "files-cache"
for _d in (IMAGE_CACHE_DIR, WEB_CACHE_DIR, FILES_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Extensions we accept for the served cache (prevents arbitrary-read)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".avif"}


def web_cache_get(url: str) -> str | None:
    """Return cached page text for `url` or None."""
    if not url:
        return None
    key = _hl.sha1(url.encode()).hexdigest()
    p = WEB_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["content"]
    except Exception:
        return None


def web_cache_put(url: str, content: str) -> None:
    """Persist page text keyed by URL."""
    if not url or content is None:
        return
    key = _hl.sha1(url.encode()).hexdigest()
    p = WEB_CACHE_DIR / f"{key}.json"
    try:
        p.write_text(json.dumps({
            "url": url,
            "fetched_at": int(time.time()),
            "content": content,
        }, ensure_ascii=False))
    except Exception:
        pass


def _dir_stats(p: Path, suffixes: set[str] | None = None) -> dict:
    """Return {count, bytes} for files inside `p`, optionally filtered."""
    count = 0
    total = 0
    if not p.exists():
        return {"count": 0, "bytes": 0}
    for f in p.iterdir():
        if not f.is_file():
            continue
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
    if not p.exists():
        return 0
    for f in p.iterdir():
        if f.is_file():
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
    return n


def _guess_ext(url: str, content_type: str | None) -> str:
    # Prefer Content-Type, then URL extension, default to .jpg
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        ext = _mt.guess_extension(ct) or ""
        if ext == ".jpe":
            ext = ".jpg"
        if ext in _IMG_EXTS:
            return ext
    tail = url.rsplit("?", 1)[0].rsplit("#", 1)[0].rsplit(".", 1)
    if len(tail) == 2:
        candidate = "." + tail[1].lower()
        if candidate in _IMG_EXTS:
            return candidate
    return ".jpg"


def cache_image_to_disk(url: str) -> str | None:
    """Download `url` once into IMAGE_CACHE_DIR, return a local
    `/ui/img/<hash>.<ext>` path (served by serve_cached_image). Returns
    None on any failure so callers can keep the original URL or drop it.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return None
    key = _hl.sha1(url.encode()).hexdigest()
    # If any file with this hash stem already exists, reuse it.
    for existing in IMAGE_CACHE_DIR.glob(key + ".*"):
        if existing.suffix.lower() in _IMG_EXTS:
            return f"/ui/img/{existing.name}"
    try:
        req = _urlreq.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        })
        import ssl as _ssl
        ctx = _ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        with _urlreq.urlopen(req, timeout=20, context=ctx) as r:
            data = r.read()
            mime = r.headers.get("Content-Type", "")
    except Exception:
        return None
    if not data:
        return None
    ext = _guess_ext(url, mime)
    path = IMAGE_CACHE_DIR / f"{key}{ext}"
    try:
        path.write_bytes(data)
    except Exception:
        return None
    return f"/ui/img/{path.name}"


@router.get("/proxy-image")
async def proxy_image(url: str):
    from fastapi.responses import Response as _R
    # Only proxy http(s), never file://, etc.
    if not (url.startswith("http://") or url.startswith("https://")):
        return _R(status_code=400, content="invalid url")
    # Short allowlist to prevent using our server as an arbitrary proxy
    allowed_hosts = (
        "myanimelist.net", "cdn.myanimelist.net",
        "i.ytimg.com", "yt3.ggpht.com",
        "commons.wikimedia.org", "upload.wikimedia.org",
        "tvmaze.com", "static.tvmaze.com",
    )
    host = url.split("/")[2] if "://" in url else ""
    if not any(host.endswith(h) for h in allowed_hosts):
        return _R(status_code=403, content="host not allowed")
    key = _hl.sha1(url.encode()).hexdigest()
    if key in _img_proxy_cache:
        data, mime = _img_proxy_cache[key]
        return _R(content=data, media_type=mime, headers={"Cache-Control": "max-age=3600"})
    try:
        req = _urlreq.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        })
        import ssl as _ssl
        ctx = _ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        with _urlreq.urlopen(req, timeout=20, context=ctx) as r:
            data = r.read()
            mime = r.headers.get("Content-Type", "image/jpeg")
    except Exception as e:
        return _R(status_code=502, content=f"fetch failed: {e}")
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


def _normalize_imported_chat(raw: dict) -> list[dict]:
    """Given one conversation dict from either ChatGPT (conversations.json)
    or Claude export format, return a list of Mio-shaped messages."""
    # ChatGPT format: has mapping = {"uuid": {"message": {...}, "children": [...]}}
    if "mapping" in raw:
        msgs: list[dict] = []
        # Walk the tree in depth-first order
        root_ids = [k for k, v in raw["mapping"].items() if not v.get("parent")]
        def walk(node_id):
            node = raw["mapping"].get(node_id)
            if not node:
                return
            msg = node.get("message")
            if msg and msg.get("content") and msg.get("author"):
                role = msg["author"]["role"]
                content = msg["content"]
                text = ""
                if content.get("content_type") == "text":
                    text = "\n".join(content.get("parts", []) or [])
                elif content.get("parts"):
                    text = "\n".join(str(p) for p in content["parts"] if p)
                if text.strip() and role in ("user", "assistant"):
                    msgs.append({"role": role, "content": text})
            for child in node.get("children", []):
                walk(child)
        for r in root_ids:
            walk(r)
        return msgs
    # Claude export format: {"chat_messages": [{"text": "...", "sender": "human|assistant"}]}
    if "chat_messages" in raw:
        out = []
        for m in raw.get("chat_messages", []):
            role = "user" if m.get("sender") == "human" else "assistant"
            out.append({"role": role, "content": m.get("text", "")})
        return out
    # Mio native: already {"messages": [...]}
    if "messages" in raw:
        return [{"role": m.get("role"), "content": m.get("content", "")} for m in raw["messages"]]
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
    created = 0
    for raw in items:
        if not isinstance(raw, dict):
            continue
        msgs = _normalize_imported_chat(raw)
        if not msgs:
            continue
        title = raw.get("title") or raw.get("name") or (msgs[0]["content"][:60] if msgs else "Imported")
        sid = str(uuid.uuid4())[:8]
        payload = {
            "id": sid,
            "title": title,
            "messages": msgs,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "imported_from": (body or {}).get("source", "auto"),
        }
        (_sessions_dir / f"{sid}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        created += 1
    return {"ok": True, "created": created}


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
    for f in _sessions_dir.glob("*.json"):
        try:
            meta = json.loads(f.read_text())
        except Exception:
            continue
        sessions += 1
        day = (meta.get("updated") or "")[:10]
        msgs = meta.get("messages") or []
        if day: by_day[day] += len(msgs)
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
    dl = Path.home() / "Downloads"
    if not dl.exists():
        return {"files": []}
    allowed = {".pdf", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg",
               ".svg", ".csv", ".ics", ".md", ".sqlite", ".db", ".html", ".txt"}
    files = []
    for f in dl.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in allowed:
            continue
        try:
            st = f.stat()
        except Exception:
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

_TEMPLATES_FILE = Path.home() / ".mio" / "chat-templates.json"


def _load_chat_templates() -> list:
    if not _TEMPLATES_FILE.exists():
        return []
    try:
        return json.loads(_TEMPLATES_FILE.read_text()) or []
    except Exception:
        return []


def _save_chat_templates(items: list) -> None:
    _TEMPLATES_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2))


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
    return {"schedules": _sched.load_schedules(), "recent": _sched.recent_runs()}


@router.post("/api/schedules")
async def schedules_create(body: dict):
    from mio.webui import scheduler as _sched
    return _sched.create_schedule(
        name=(body or {}).get("name", ""),
        prompt=(body or {}).get("prompt", ""),
        cadence=(body or {}).get("cadence"),
        tier=(body or {}).get("tier"),
        enabled=(body or {}).get("enabled", True),
    )


@router.patch("/api/schedules/{sched_id}")
async def schedules_update(sched_id: str, body: dict):
    from mio.webui import scheduler as _sched
    return _sched.update_schedule(sched_id, body or {})


@router.delete("/api/schedules/{sched_id}")
async def schedules_delete(sched_id: str):
    from mio.webui import scheduler as _sched
    return _sched.delete_schedule(sched_id)


# ---- Webhook endpoints ----

@router.get("/api/webhooks")
async def webhooks_list():
    from mio.webui import webhooks as _wh
    return {"webhooks": _wh.load_webhooks(), "recent": _wh.recent_runs()}


@router.post("/api/webhooks")
async def webhooks_create(body: dict):
    from mio.webui import webhooks as _wh
    return _wh.create_webhook(
        slug=(body or {}).get("slug", ""),
        prompt=(body or {}).get("prompt", ""),
        tier=(body or {}).get("tier"),
        secret=(body or {}).get("secret"),
    )


@router.delete("/api/webhooks/{slug}")
async def webhooks_delete(slug: str):
    from mio.webui import webhooks as _wh
    return _wh.delete_webhook(slug)


@router.post("/api/webhook/{slug}")
async def webhook_fire(slug: str, body: dict | None = None):
    """Fire a configured webhook. POST body is JSON; its keys substitute
    into the prompt template's {{key}} slots. If the webhook has a
    `secret`, expect it in the payload's `secret` field.
    """
    from mio.webui import webhooks as _wh
    hook = next((h for h in _wh.load_webhooks() if h["slug"] == slug), None)
    if not hook:
        return {"error": "unknown webhook"}
    payload = body or {}
    if hook.get("secret") and payload.get("secret") != hook["secret"]:
        return {"error": "bad secret"}
    prompt = _wh.render_prompt(hook.get("prompt", ""), payload)
    if not _manager:
        return {"error": "no model loaded"}
    loaded = _manager.loaded_tiers()
    if not loaded:
        return {"error": "no tiers loaded"}
    tier = hook.get("tier") if hook.get("tier") in loaded else loaded[0]
    engine = _manager.get_engine(tier)
    out_parts: list[str] = []
    try:
        for chunk_text, _m in engine.generate_stream(
            [{"role": "user", "content": prompt}],
            max_tokens=2048,
        ):
            out_parts.append(chunk_text)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
        _wh.append_log(slug, payload, result)
        return result
    result = {"ok": True, "slug": slug, "tier": tier, "output": "".join(out_parts)}
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
    mime = _mt.guess_type(str(p))[0] or "application/octet-stream"
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
    from fastapi.responses import FileResponse as _FR, Response as _R
    if "/" in filename or "\\" in filename or ".." in filename:
        return _R(status_code=400, content="invalid filename")
    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in _IMG_EXTS:
        return _R(status_code=400, content="unsupported extension")
    p = IMAGE_CACHE_DIR / filename
    if not p.exists() or not p.is_file():
        return _R(status_code=404, content="not cached")
    mime = _mt.guess_type(filename)[0] or "image/jpeg"
    # Long cache — filenames are content-addressed by URL hash, so they
    # never change meaning.
    return _FR(str(p), media_type=mime, headers={"Cache-Control": "max-age=31536000, immutable"})


@router.get("/files/{filename}")
async def serve_generated_file(filename: str, download: int = 0):
    if "/" in filename or "\\" in filename or ".." in filename:
        return Response(status_code=400, content="invalid filename")
    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in _ALLOWED_EXT:
        return Response(status_code=400, content="unsupported file type")
    from pathlib import Path as _Path
    p = _Path.home() / "Downloads" / filename
    if not p.exists() or not p.is_file():
        return Response(status_code=404, content="not found")
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # ?download=1 → attachment (triggers Save dialog). Default is inline so
    # PDFs can render in an <iframe> rather than being downloaded.
    disposition = "attachment" if download else "inline"
    return FileResponse(
        str(p),
        media_type=mime,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


# --- WebSocket streaming chat ---

@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
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


async def _handle_chat(websocket: WebSocket, data: dict):
    """Handle a chat request over WebSocket with streaming and skill execution."""
    import asyncio

    messages = data.get("messages", [])
    # /remember <fact> — store in persistent memory without consulting the model
    if messages:
        last = messages[-1]
        if last.get("role") == "user":
            c = (last.get("content") or "").strip()
            m = re.match(r"^/remember\s+(.+)", c, re.IGNORECASE | re.DOTALL)
            if m:
                fact = m.group(1).strip()
                mem = _load_memory()
                mem.append({"id": str(uuid.uuid4())[:8], "text": fact, "added": time.strftime("%Y-%m-%dT%H:%M:%S")})
                _save_memory(mem)
                await websocket.send_json({"type": "start", "tier": "memory"})
                await websocket.send_json({"type": "token", "text": "✓ Saved to memory: \"" + fact + "\""})
                await websocket.send_json({"type": "done", "full_text": "✓ Saved to memory: \"" + fact + "\"",
                                           "metrics": {"completion_tokens": 0, "generation_tps": 0, "prompt_tokens": 0, "prompt_tps": 0, "acceptance_ratio": 0}})
                return
    tier = data.get("tier")
    max_tokens = data.get("max_tokens") or _max_tokens
    system_prompt = data.get("system_prompt") or _system_prompt
    caveman = data.get("caveman") or _caveman_level
    use_skills = data.get("skills", True)
    temperature = data.get("temperature")
    if temperature is None:
        temperature = _temperature

    if not _manager:
        await websocket.send_json({"type": "error", "message": "No model loaded"})
        return

    # Resolve engine
    loaded = _manager.loaded_tiers()
    if not loaded:
        await websocket.send_json({"type": "error", "message": "No tiers loaded"})
        return
    tier = tier if tier in loaded else loaded[0]
    engine = _manager.get_engine(tier)

    # Inject skills as tools
    tools = None
    if use_skills:
        from mio.webui.skills import get_tools_spec
        tools = get_tools_spec()

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
                for fname in proj.get("files", []):
                    fpath = Path.home() / "Downloads" / fname
                    if not fpath.exists():
                        continue
                    try:
                        if fname.lower().endswith(".pdf"):
                            import pdfplumber
                            with pdfplumber.open(str(fpath)) as pdf:
                                text = "\n\n".join((p.extract_text() or "") for p in pdf.pages[:20])
                        else:
                            text = fpath.read_text(errors="replace")
                        if text.strip():
                            base_sys += f"\n\n=== Project file: {fname} ===\n{text[:8000]}\n=== end {fname} ==="
                    except Exception:
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

    # Inject caveman
    if caveman and caveman != "off":
        from mio.agent import CAVEMAN_LEVELS
        ctext = CAVEMAN_LEVELS.get(caveman, "")
        if ctext:
            sys_msg = next((m for m in messages if m.get("role") == "system"), None)
            if sys_msg:
                sys_msg["content"] = (sys_msg["content"] or "") + "\n\n" + ctext
            else:
                messages = [{"role": "system", "content": ctext}] + messages

    # Snapshot the final system prompt for /api/debug/last-prompt
    try:
        global _last_system_prompt
        _sys = next((m.get("content") for m in messages if m.get("role") == "system"), None)
        _last_system_prompt = (
            f"# caveman={caveman} · style={style!r} · project={project_id!r} · "
            f"skills={use_skills}\n\n{_sys or '(no system message)'}"
        )
    except Exception:
        pass

    # Send start event
    await websocket.send_json({"type": "start", "tier": tier})

    loop = asyncio.get_event_loop()
    SENTINEL = object()

    async def _stream_one_round(round_messages: list[dict], is_first: bool) -> tuple[str, dict | None]:
        """Run one model generation round, forwarding tokens to the WS.
        Returns (full_text, final_metrics).
        """
        q: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                with _gpu_lock:
                    for chunk_text, chunk_metrics in engine.generate_stream(
                        round_messages, max_tokens=max_tokens,
                        temperature=temperature, tools=tools,
                    ):
                        event = {"type": "token", "text": chunk_text}
                        if chunk_metrics:
                            event["metrics"] = {
                                "prompt_tokens": chunk_metrics.prompt_tokens,
                                "completion_tokens": chunk_metrics.completion_tokens,
                                "prompt_tps": round(chunk_metrics.prompt_tps, 1),
                                "generation_tps": round(chunk_metrics.generation_tps, 1),
                                "acceptance_ratio": round(chunk_metrics.acceptance_ratio, 2),
                            }
                        loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "message": str(e)})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, SENTINEL)

        threading.Thread(target=_produce, daemon=True).start()

        if not is_first:
            await websocket.send_json({"type": "followup_start"})

        chunks: list[str] = []
        metrics: dict | None = None
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
                continue
            if item is SENTINEL:
                break
            if isinstance(item, dict):
                if item.get("type") == "error":
                    await websocket.send_json(item)
                    break
                if item.get("text"):
                    chunks.append(item["text"])
                if item.get("metrics"):
                    metrics = item["metrics"]
                await websocket.send_json(item)
        return "".join(chunks), metrics

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
        full_text_str, final_metrics = await _stream_one_round(
            current_messages, is_first=(round_idx == 0)
        )
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
            result = execute_skill(fn_name, fn_args)
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
