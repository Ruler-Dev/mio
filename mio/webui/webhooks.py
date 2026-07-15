"""Webhook triggers — POST /ui/api/webhook/<slug> runs a pre-registered
prompt (with substitution from the POST body) through the current model.

Webhooks are stored at ~/.mio/webhooks.json. Each entry:

  {
    "slug": "morning-brief",
    "prompt": "Summarize yesterday's {{topic}} news in 5 bullets.",
    "tier": "large-moe",
    "secret_hash": "sha256:<digest>",
    "created": "2026-04-19T...",
  }

POST body is merged with the prompt template as {{key}} substitutions.
Result is saved to ~/.mio/webhooks-log.jsonl and returned in the
response so external callers can chain on it.
"""
from __future__ import annotations

from collections import deque
import hashlib
import hmac
import json
import os
import re
import stat
import threading
import time
from typing import Any

from mio.paths import mio_home
from mio.persistence import atomic_write_json

PIMIO_DIR = mio_home()
PIMIO_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
_WEBHOOKS_FILE = PIMIO_DIR / "webhooks.json"
_WEBHOOKS_LOG = PIMIO_DIR / "webhooks-log.jsonl"

_SECRET_MIN_CHARS = 16
_SECRET_MAX_BYTES = 1024
_PROMPT_MAX_BYTES = 128 * 1024
_LOG_MAX_BYTES = 2 * 1024 * 1024
_LOG_ENTRY_MAX_BYTES = 64 * 1024
_LOG_BACKUPS = 3
_LOG_STRING_MAX_CHARS = 8_192
_LOG_COLLECTION_MAX_ITEMS = 100
_LOG_LOCK = threading.Lock()
_SENSITIVE_KEY = re.compile(
    r"(?:secret|token|authorization|api[-_]?key|password|passwd|cookie|credential)",
    re.IGNORECASE,
)


def load_webhooks() -> list[dict]:
    if not _WEBHOOKS_FILE.exists():
        return []
    try:
        return json.loads(_WEBHOOKS_FILE.read_text()) or []
    except Exception:
        return []


def save_webhooks(hooks: list[dict]) -> None:
    atomic_write_json(_WEBHOOKS_FILE, hooks)


def _hash_secret(secret: str) -> str:
    return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()


def verify_secret(hook: dict, supplied: Any) -> bool:
    """Verify a webhook secret without exposing stored authentication data."""
    if not isinstance(supplied, str) or not supplied:
        return False
    stored_hash = hook.get("secret_hash")
    if isinstance(stored_hash, str) and stored_hash.startswith("sha256:"):
        return hmac.compare_digest(_hash_secret(supplied), stored_hash)
    # Fail-closed compatibility for pre-migration entries.  Saving the hook
    # again replaces this plaintext legacy field with ``secret_hash``.
    legacy = hook.get("secret")
    return isinstance(legacy, str) and bool(legacy) and hmac.compare_digest(
        supplied,
        legacy,
    )


def public_webhooks() -> list[dict]:
    """Return webhook metadata while never serializing a secret or its hash."""
    public: list[dict] = []
    for hook in load_webhooks():
        if not isinstance(hook, dict):
            continue
        public.append(
            {
                "slug": hook.get("slug"),
                "prompt": hook.get("prompt", ""),
                "tier": hook.get("tier"),
                "created": hook.get("created"),
                "has_secret": bool(hook.get("secret_hash") or hook.get("secret")),
            }
        )
    return public


def create_webhook(slug: str, prompt: str, tier: str | None = None,
                   secret: str | None = None) -> dict:
    slug = (slug or "").strip().lower().replace(" ", "-")
    if (
        not slug
        or len(slug) > 64
        or not slug.isascii()
        or not slug.replace("-", "").isalnum()
    ):
        raise ValueError("slug must use at most 64 ASCII letters, digits, or hyphens")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    if len(prompt.encode("utf-8")) > _PROMPT_MAX_BYTES:
        raise ValueError("prompt exceeds the 128 KiB limit")
    if not isinstance(secret, str) or len(secret) < _SECRET_MIN_CHARS:
        raise ValueError("secret is required and must contain at least 16 characters")
    if len(secret.encode("utf-8")) > _SECRET_MAX_BYTES:
        raise ValueError("secret exceeds the 1 KiB limit")
    if tier is not None and not isinstance(tier, str):
        raise ValueError("tier must be text")
    hooks = load_webhooks()
    existing = next((h for h in hooks if h["slug"] == slug), None)
    entry = {
        "slug": slug,
        "prompt": prompt,
        "tier": tier,
        "secret_hash": _hash_secret(secret),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if existing:
        entry["created"] = existing.get("created") or entry["created"]
        existing.pop("secret", None)
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


def _redact_log_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if depth >= 6:
        return "<max-depth>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= _LOG_COLLECTION_MAX_ITEMS:
                result["_truncated"] = True
                break
            normalized_key = str(item_key)[:256]
            result[normalized_key] = _redact_log_value(
                item_value,
                key=normalized_key,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _redact_log_value(item, depth=depth + 1)
            for item in value[:_LOG_COLLECTION_MAX_ITEMS]
        ]
        if len(value) > _LOG_COLLECTION_MAX_ITEMS:
            result.append("<truncated>")
        return result
    if isinstance(value, str):
        if len(value) <= _LOG_STRING_MAX_CHARS:
            return value
        return value[:_LOG_STRING_MAX_CHARS] + "…<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_LOG_STRING_MAX_CHARS]


def _backup_path(index: int):
    return _WEBHOOKS_LOG.with_name(f"{_WEBHOOKS_LOG.name}.{index}")


def _rotate_log_if_needed(incoming_bytes: int) -> None:
    if _WEBHOOKS_LOG.is_symlink():
        raise OSError("refusing to write a symlinked webhook log")
    try:
        current_size = _WEBHOOKS_LOG.stat(follow_symlinks=False).st_size
    except FileNotFoundError:
        current_size = 0
    if current_size + incoming_bytes <= _LOG_MAX_BYTES:
        return

    oldest = _backup_path(_LOG_BACKUPS)
    oldest.unlink(missing_ok=True)
    for index in range(_LOG_BACKUPS - 1, 0, -1):
        source = _backup_path(index)
        if source.is_symlink():
            source.unlink(missing_ok=True)
            continue
        if source.exists():
            os.replace(source, _backup_path(index + 1))
    if _WEBHOOKS_LOG.exists():
        os.replace(_WEBHOOKS_LOG, _backup_path(1))


def append_log(slug: str, payload: dict, result: dict) -> None:
    entry = {
        "slug": str(slug)[:64],
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "payload": _redact_log_value(payload),
        "result": _redact_log_value(result),
    }
    encoded = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > _LOG_ENTRY_MAX_BYTES:
        entry["payload"] = {"_truncated": True}
        encoded = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > _LOG_ENTRY_MAX_BYTES:
        entry["result"] = {"_truncated": True}
        encoded = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with _LOG_LOCK:
            _WEBHOOKS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _rotate_log_if_needed(len(encoded))
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(_WEBHOOKS_LOG, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("webhook log is not a regular file")
                view = memoryview(encoded)
                while view:
                    view = view[os.write(descriptor, view):]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        pass


def recent_runs(limit: int = 20) -> list[dict]:
    try:
        bounded_limit = max(0, min(int(limit), 100))
    except (TypeError, ValueError):
        bounded_limit = 20
    if bounded_limit == 0:
        return []
    out: deque[dict] = deque(maxlen=bounded_limit)
    # Read the newest rotated segment as well as the current log.  Both are
    # size-bounded, and each individual line is bounded at write time.
    for path in (_backup_path(1), _WEBHOOKS_LOG):
        if not path.exists() or path.is_symlink():
            continue
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                size = os.fstat(handle.fileno()).st_size
                offset = max(0, size - _LOG_MAX_BYTES)
                handle.seek(offset)
                if offset:
                    handle.readline(_LOG_ENTRY_MAX_BYTES + 1)
                for raw_line in handle:
                    if len(raw_line) > _LOG_ENTRY_MAX_BYTES:
                        continue
                    try:
                        value = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(value, dict):
                        out.append(value)
        except OSError:
            continue
        finally:
            if "descriptor" in locals() and descriptor >= 0:
                os.close(descriptor)
    return list(reversed(out))
