"""Webhook triggers — POST /ui/api/webhook/<slug> runs a pre-registered
prompt (with substitution from the POST body) through the current model.

Webhooks are stored at ~/.mio/webhooks.json. Each entry:

  {
    "slug": "morning-brief",
    "prompt": "Summarize yesterday's {{topic}} news in 5 bullets.",
    "tier": "large-moe",
    "secret": "optional-shared-secret",
    "created": "2026-04-19T...",
  }

POST body is merged with the prompt template as {{key}} substitutions.
Result is saved to ~/.mio/webhooks-log.jsonl and returned in the
response so external callers can chain on it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PIMIO_DIR = Path.home() / ".mio"
PIMIO_DIR.mkdir(parents=True, exist_ok=True)
_WEBHOOKS_FILE = PIMIO_DIR / "webhooks.json"
_WEBHOOKS_LOG = PIMIO_DIR / "webhooks-log.jsonl"


def load_webhooks() -> list[dict]:
    if not _WEBHOOKS_FILE.exists():
        return []
    try:
        return json.loads(_WEBHOOKS_FILE.read_text()) or []
    except Exception:
        return []


def save_webhooks(hooks: list[dict]) -> None:
    _WEBHOOKS_FILE.write_text(json.dumps(hooks, ensure_ascii=False, indent=2))


def create_webhook(slug: str, prompt: str, tier: str | None = None,
                   secret: str | None = None) -> dict:
    slug = (slug or "").strip().lower().replace(" ", "-")
    if not slug or not slug.replace("-", "").isalnum():
        return {"error": "slug must be alphanumeric/hyphens"}
    hooks = load_webhooks()
    existing = next((h for h in hooks if h["slug"] == slug), None)
    entry = {
        "slug": slug,
        "prompt": prompt or "",
        "tier": tier,
        "secret": secret,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if existing:
        existing.update(entry)
    else:
        hooks.append(entry)
    save_webhooks(hooks)
    return {"ok": True, "slug": slug, "url": f"/ui/api/webhook/{slug}"}


def delete_webhook(slug: str) -> dict:
    hooks = [h for h in load_webhooks() if h["slug"] != slug]
    save_webhooks(hooks)
    return {"ok": True}


def render_prompt(template: str, payload: dict) -> str:
    """{{key}} substitution, missing keys become empty strings."""
    import re
    if not template:
        return ""
    def sub(m):
        k = m.group(1).strip()
        v = payload.get(k, "")
        return str(v) if v is not None else ""
    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", sub, template)


def append_log(slug: str, payload: dict, result: dict) -> None:
    try:
        with _WEBHOOKS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "slug": slug,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "payload": payload,
                "result": result,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def recent_runs(limit: int = 20) -> list[dict]:
    if not _WEBHOOKS_LOG.exists():
        return []
    out = []
    try:
        with _WEBHOOKS_LOG.open("r", encoding="utf-8") as f:
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
