"""Flow Mode — async DAG executor.

Takes a Drawflow-exported graph and runs it server-side, emitting
SSE events per node. Supported node classes in this MVP:

  llm_call     chat completion via the OpenAI-compatible endpoint
  skill_call   dispatch to the registered Mio skills
  http_fetch   simple GET/POST
  if_else      branch on a truthy-expression against the input
  iterate      normalize/map every item in an input list
  user_input   resolve a value supplied in the run environment
  output       terminal — collects into the run's result bucket

Nodes carry a `data` payload edited in the UI. Variable interpolation
uses simple `{{...}}` templating:
  {{input}}    — the last upstream output
  {{n.out}}    — the output of node N (id)
  {{env.X}}    — environment values set at run start

This is a deliberately small executor: no parallelism inside a
single flow, no retries / backoff, no loop detection beyond max
200 node-hops. Enough to be useful; big enough to iterate.
"""

from __future__ import annotations

import asyncio
import ast
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
from typing import Any
from urllib.parse import urljoin, urlsplit

from mio.paths import mio_home
from mio.persistence import atomic_update_json, atomic_write_json

_runs: dict[str, "_Run"] = {}
_MAX_HOPS = 200
_MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_ARTIFACT_CONTENT_BYTES = 512 * 1024
_MAX_ARTIFACT_RUN_BYTES = 2 * 1024 * 1024
_MAX_ARTIFACTS_PER_RUN = 16
_MAX_ARTIFACT_TITLE_CHARS = 200
_MAX_ARTIFACT_TYPE_CHARS = 128
_MAX_RUNS = 128
_RUN_TTL_SECONDS = 60 * 60
_MAX_REGEX_PATTERN_BYTES = 4 * 1024
_MAX_REGEX_INPUT_BYTES = 256 * 1024
_REGEX_TIMEOUT_SECONDS = 0.5
_MAX_CONDITION_CHARS = 4096

_CONDITION_OPERAND = r"(?:\{\{[^{}]+\}\}|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s<>=!]+)"
_CONDITION_RE = re.compile(
    rf"\s*(?P<lhs>{_CONDITION_OPERAND})"
    rf"(?:\s*(?P<op>==|!=|<=|>=|<|>)\s*(?P<rhs>{_CONDITION_OPERAND}))?\s*"
)
_NUMBER_LITERAL_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)

# CPython's stdlib ``re`` engine has no timeout.  Execute untrusted patterns
# in a fresh isolated interpreter so a timeout can terminate the whole engine
# rather than abandoning a permanently blocked worker thread.
_REGEX_WORKER = r"""
import json
import re
import sys

payload = json.load(sys.stdin)
try:
    match = re.search(payload["pattern"], payload["text"], int(payload["flags"]))
    value = "" if match is None else (match.group(1) if match.groups() else match.group(0))
    result = {"ok": True, "value": value}
except Exception as exc:
    result = {"ok": False, "error": str(exc)[:500]}
json.dump(result, sys.stdout, ensure_ascii=False)
"""


class _Run:
    def __init__(
        self,
        run_id: str,
        flow: dict,
        env: dict | None = None,
        *,
        manager: Any = None,
        gpu_lock: Any = None,
        request_skill_grants: frozenset[str] | None = None,
    ):
        self.id = run_id
        self.flow = flow
        self.env = env or {}
        self.outputs: dict[str, Any] = {}
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.done = False
        self.started = time.time()
        self.finished: float | None = None
        self.manager = manager
        self.gpu_lock = gpu_lock
        self.request_skill_grants = request_skill_grants
        self.final_event: dict[str, Any] | None = None
        self.artifact_count = 0
        self.artifact_bytes = 0
        self.cancelled = threading.Event()
        self.task: asyncio.Task[None] | None = None

    async def emit(self, event: dict) -> None:
        if event.get("type") == "run_finished":
            self.final_event = event
        await self.queue.put(event)

    def close(self) -> None:
        self.done = True
        self.finished = time.time()

    def cancel(self) -> None:
        self.cancelled.set()


def _interpolate(tpl: str, ctx: dict) -> str:
    def stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def repl(m: re.Match) -> str:
        expr = m.group(1).strip()
        if expr == "input":
            return stringify(ctx.get("_last", ""))
        if expr == "item":
            return stringify(ctx.get("item", ctx.get("_last", "")))
        if expr == "index":
            return str(ctx.get("index", 0))
        if expr.startswith("env."):
            return stringify(ctx.get("env", {}).get(expr[4:], ""))
        # nodeid.out — use output of that node
        if "." in expr:
            nid, _ = expr.split(".", 1)
            val = ctx.get("outputs", {}).get(nid, "")
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            return str(val)
        return ""

    return re.sub(r"\{\{\s*(.*?)\s*\}\}", repl, tpl or "")


def _resolve_value(value: Any, ctx: dict) -> Any:
    """Preserve native values when a field is one complete template token."""
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\{\{\s*(.*?)\s*\}\}", value)
    if not match:
        return _interpolate(value, ctx)
    expr = match.group(1).strip()
    if expr == "input":
        return ctx.get("_last", "")
    if expr == "item":
        return ctx.get("item", ctx.get("_last", ""))
    if expr == "index":
        return ctx.get("index", 0)
    if expr.startswith("env."):
        return ctx.get("env", {}).get(expr[4:], "")
    if "." in expr:
        node_id, _ = expr.split(".", 1)
        return ctx.get("outputs", {}).get(node_id, "")
    return ""


async def _run_llm(data: dict, ctx: dict) -> str:
    prompt = _interpolate(data.get("prompt", "") or "{{input}}", ctx)
    system = _interpolate(data.get("system", ""), ctx)
    manager = ctx.get("_manager")
    if manager is None:
        raise RuntimeError("flow LLM node has no model manager")
    requested_tier = str(data.get("tier") or "")
    messages = [
        *([{"role": "system", "content": system}] if system else []),
        {"role": "user", "content": prompt},
    ]
    raw_temperature = data.get("temperature")
    temperature = 0.0 if raw_temperature is None else float(raw_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("flow LLM temperature must be between 0 and 2")
    max_tokens = max(1, min(int(data.get("max_tokens", 1024) or 1024), 32768))

    def generate() -> str:
        lock = ctx.get("_gpu_lock")
        cancelled = ctx.get("_cancelled")
        acquired = False
        context_managed = False
        try:
            if lock is not None:
                acquire = getattr(lock, "acquire", None)
                if callable(acquire):
                    while cancelled is None or not cancelled.is_set():
                        if acquire(timeout=0.1):
                            acquired = True
                            break
                else:
                    lock.__enter__()
                    acquired = True
                    context_managed = True
                if not acquired:
                    raise RuntimeError("flow run was cancelled")
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("flow run was cancelled")
            loaded = manager.loaded_tiers()
            if not loaded:
                raise RuntimeError("no model tier is loaded")
            tier = requested_tier if requested_tier in loaded else loaded[0]
            engine = manager.get_engine(tier)
            chunks: list[str] = []
            stream = engine.generate_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                for chunk, _metrics in stream:
                    if cancelled is not None and cancelled.is_set():
                        raise RuntimeError("flow run was cancelled")
                    if chunk:
                        chunks.append(chunk)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            return "".join(chunks)
        finally:
            if acquired:
                if context_managed:
                    lock.__exit__(None, None, None)
                else:
                    lock.release()

    # MLX generation is synchronous. Keep it off Uvicorn's event loop and use
    # the same Metal lock as chat/scheduler requests.
    return await asyncio.to_thread(generate)


async def _run_skill(data: dict, ctx: dict) -> Any:
    from mio.webui.skills import execute_skill
    from mio.web_security import webui_skill_operator_granted, webui_skill_risk

    skill = str(data.get("skill") or "")
    risk = webui_skill_risk(skill)
    if risk != "read" and not webui_skill_operator_granted(skill):
        raise PermissionError(
            f"flow skill {skill!r} has risk {risk!r} and needs an explicit "
            "MIO_WEBUI_SKILL_GRANTS operator grant"
        )
    request_grants = ctx.get("_request_skill_grants")
    if risk != "read" and request_grants is not None and skill not in request_grants:
        raise PermissionError(
            f"flow skill {skill!r} needs an explicit grant in this model request"
        )
    cancelled = ctx.get("_cancelled")
    if cancelled is not None and cancelled.is_set():
        raise RuntimeError("flow run was cancelled")
    args_raw = data.get("args", "{}")
    try:
        args = json.loads(_interpolate(args_raw if isinstance(args_raw, str) else json.dumps(args_raw), ctx))
    except Exception:
        args = {}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: execute_skill(skill, args))
    if cancelled is not None and cancelled.is_set():
        raise RuntimeError("flow run was cancelled")
    return result


def _validated_http_target(url: str) -> tuple[Any, str, int]:
    """Resolve once, validate every answer, and return one address to pin."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http:// and https:// URLs are allowed")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("URL host is required and credentials are not allowed")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid URL port")
    effective_port = port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, effective_port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve HTTP host: {parsed.hostname}") from exc
    if not addresses:
        raise ValueError(f"cannot resolve HTTP host: {parsed.hostname}")
    allow_private = os.environ.get("MIO_FLOW_ALLOW_PRIVATE_HTTP", "").lower() in {
        "1", "true", "yes",
    }
    resolved_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        resolved_ips.append(ip)
        if not allow_private and not ip.is_global:
            raise ValueError("private, loopback, link-local and reserved HTTP targets are blocked")
    return parsed, str(resolved_ips[0]), effective_port


def _validate_http_url(url: str) -> None:
    """Compatibility validator used by tests and callers that do not fetch."""
    _validated_http_target(url)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, pinned_ip: str, *, timeout: float):
        super().__init__(host, port=port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, pinned_ip: str, *, timeout: float):
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )
        # ``self.host`` remains the original hostname, so TLS SNI and
        # certificate verification are preserved while DNS cannot rebind.
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)


def _host_header(parsed: Any, port: int) -> str:
    hostname = parsed.hostname or ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if parsed.scheme == "https" else 80
    return host if port == default_port else f"{host}:{port}"


def _fetch_pinned_http(url: str, method: str, request_body: bytes | None) -> bytes:
    current_url = url
    current_method = method
    current_body = request_body
    for redirect_count in range(6):
        parsed, pinned_ip, port = _validated_http_target(current_url)
        connection_type = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
        connection = connection_type(parsed.hostname or "", port, pinned_ip, timeout=30.0)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Host": _host_header(parsed, port),
            "User-Agent": "Mio-Flow/0.1",
        }
        if current_body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(current_body))
        try:
            connection.request(current_method, path, body=current_body, headers=headers)
            response = connection.getresponse()
            body = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
            status = response.status
            location = response.getheader("Location")
        finally:
            connection.close()
        if len(body) > _MAX_HTTP_RESPONSE_BYTES:
            raise ValueError(f"HTTP response exceeds {_MAX_HTTP_RESPONSE_BYTES} bytes")
        if status in {301, 302, 303, 307, 308} and location:
            if redirect_count >= 5:
                raise ValueError("HTTP redirect limit exceeded")
            redirected_url = urljoin(current_url, location)
            redirected = urlsplit(redirected_url)
            if parsed.scheme == "https" and redirected.scheme != "https":
                raise ValueError("HTTPS redirect downgrade to HTTP is blocked")
            current_url = redirected_url
            # Match browser/urllib semantics without leaking a POST body to a
            # newly redirected origin unless the server explicitly uses 307/308.
            if status in {301, 302, 303} and current_method == "POST":
                current_method = "GET"
                current_body = None
            continue
        if not 200 <= status < 300:
            raise ValueError(f"HTTP server returned status {status}")
        return body
    raise ValueError("HTTP redirect limit exceeded")


async def _run_http(data: dict, ctx: dict) -> Any:
    method = (data.get("method") or "GET").upper()
    url = _interpolate(data.get("url", ""), ctx)
    if not url:
        raise ValueError("url required")
    if method not in {"GET", "POST"}:
        raise ValueError("only GET and POST are allowed")
    request_body = None
    if method == "POST" and data.get("body") is not None:
        rendered = _resolve_value(data.get("body"), ctx)
        request_body = (
            rendered.encode("utf-8")
            if isinstance(rendered, str)
            else json.dumps(rendered, ensure_ascii=False).encode("utf-8")
        )

    try:
        body = await asyncio.to_thread(_fetch_pinned_http, url, method, request_body)
        try:
            return json.loads(body)
        except Exception:
            return body.decode(errors="replace")
    except Exception as e:
        raise ValueError(str(e)) from e


async def _run_constant(data: dict, ctx: dict) -> Any:
    v = data.get("value", "")
    # Allow JSON-shaped values (user types {...} in the field)
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
        return _interpolate(v, ctx)
    return v


async def _run_template(data: dict, ctx: dict) -> str:
    return _interpolate(data.get("template", "{{input}}"), ctx)


async def _run_parse_json(data: dict, ctx: dict) -> Any:
    val = ctx.get("_last", "")
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(str(val))
    except Exception as e:
        raise ValueError(f"parse_json: {e}") from e


async def _run_to_json(data: dict, ctx: dict) -> str:
    indent = data.get("indent")
    try:
        indent = int(indent) if indent is not None else None
    except Exception:
        indent = None
    val = ctx.get("_last", "")
    try:
        return json.dumps(val, indent=indent, ensure_ascii=False)
    except Exception:
        return str(val)


def _bounded_regex_search(pattern: str, text: str, flags: int) -> Any:
    pattern_bytes = len(pattern.encode("utf-8", errors="surrogatepass"))
    input_bytes = len(text.encode("utf-8", errors="surrogatepass"))
    if pattern_bytes > _MAX_REGEX_PATTERN_BYTES:
        raise ValueError(
            f"regex pattern exceeds {_MAX_REGEX_PATTERN_BYTES} bytes"
        )
    if input_bytes > _MAX_REGEX_INPUT_BYTES:
        raise ValueError(f"regex input exceeds {_MAX_REGEX_INPUT_BYTES} bytes")

    payload = json.dumps(
        {"pattern": pattern, "text": text, "flags": flags},
        ensure_ascii=False,
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _REGEX_WORKER],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            timeout=_REGEX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        # ``subprocess.run`` kills and waits for the child before raising, so
        # neither a runaway regex process nor a blocked executor thread leaks.
        raise ValueError("regex timed out") from exc
    if completed.returncode != 0:
        raise ValueError("regex worker failed")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("regex worker returned an invalid response") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error", "unknown regex error") if isinstance(result, dict) else "unknown regex error"
        raise ValueError(f"regex error: {detail}")
    return result.get("value", "")


async def _run_regex_extract(data: dict, ctx: dict) -> Any:
    pattern = _interpolate(data.get("pattern", ""), ctx)
    flags_str = (data.get("flags") or "").lower()
    flags = 0
    if "i" in flags_str:
        flags |= re.IGNORECASE
    if "m" in flags_str:
        flags |= re.MULTILINE
    if "s" in flags_str:
        flags |= re.DOTALL
    try:
        return await asyncio.to_thread(
            _bounded_regex_search,
            pattern,
            str(ctx.get("_last", "")),
            flags,
        )
    except Exception as e:
        if isinstance(e, ValueError) and str(e).startswith("regex"):
            raise
        raise ValueError(f"regex error: {e}") from e


async def _run_split(data: dict, ctx: dict) -> list:
    delim = data.get("delim", ",")
    return str(ctx.get("_last", "")).split(delim)


async def _run_join(data: dict, ctx: dict) -> str:
    delim = data.get("delim", ", ")
    last = ctx.get("_last", [])
    if isinstance(last, str):
        last = [last]
    try:
        return delim.join(str(x) for x in last)
    except Exception:
        return str(last)


def _memory_path() -> "Any":
    p = mio_home() / "flow-memory.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _memory_load() -> dict:
    p = _memory_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _memory_save(data: dict) -> None:
    atomic_write_json(_memory_path(), data)


async def _run_mem_get(data: dict, ctx: dict) -> Any:
    key = _interpolate(data.get("key", ""), ctx)
    mem = _memory_load()
    return mem.get(key, None)


def _memory_set(key: str, value: Any) -> None:
    def merge(current: Any) -> dict:
        if not isinstance(current, dict):
            raise ValueError("flow memory must contain a JSON object")
        replacement = dict(current)
        replacement[key] = value
        return replacement

    atomic_update_json(_memory_path(), merge)


async def _run_mem_set(data: dict, ctx: dict) -> Any:
    key = _interpolate(data.get("key", ""), ctx)
    if not key:
        return ctx.get("_last")
    await asyncio.to_thread(_memory_set, key, ctx.get("_last"))
    return ctx.get("_last")  # passthrough


async def _run_delay(data: dict, ctx: dict) -> Any:
    ms = max(0.0, min(float(data.get("ms", 500) or 500), 300_000.0))
    await asyncio.sleep(ms / 1000.0)
    return ctx.get("_last")


async def _run_clock(data: dict, ctx: dict) -> str:
    import datetime as _dt

    return _dt.datetime.now().isoformat(timespec="seconds")


async def _run_random(data: dict, ctx: dict) -> Any:
    import random as _rand

    val = ctx.get("_last", [])
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            val = val.split("\n")
    if not isinstance(val, list) or not val:
        return None
    return _rand.choice(val)


async def _run_rag_search(data: dict, ctx: dict) -> dict:
    try:
        from mio.webui.skills_rag import search_local_folder

        q = _interpolate(data.get("query", "{{input}}"), ctx) or str(ctx.get("_last", ""))
        limit = int(data.get("limit", 5) or 5)
        r = search_local_folder(q, limit=limit)
        return {"count": r.get("count", 0), "results": r.get("results", [])}
    except Exception as e:
        raise ValueError(str(e)) from e


async def _run_artifact_emit(data: dict, ctx: dict) -> dict:
    """Return one bounded artifact payload for the Flow event stream.

    ``_execute_run`` emits this object as ``artifact_emitted``.  The browser
    consumes that event through ``window.Mio.artifacts`` and registers it in
    the same gallery/version store used by chat artifacts.
    """
    title = str(data.get("title") or "Flow output").strip()
    if not title:
        title = "Flow output"
    if len(title) > _MAX_ARTIFACT_TITLE_CHARS:
        raise ValueError(
            f"artifact title exceeds {_MAX_ARTIFACT_TITLE_CHARS} characters"
        )
    kind = str(data.get("type") or "text/html").strip().lower()
    if (
        not kind
        or len(kind) > _MAX_ARTIFACT_TYPE_CHARS
        or not re.fullmatch(r"[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*", kind)
    ):
        raise ValueError("artifact type must be a valid MIME type")
    body = ctx.get("_last", "")
    if not isinstance(body, str):
        try:
            body = json.dumps(body, ensure_ascii=False)
        except (TypeError, ValueError):
            body = str(body)
    content_bytes = len(body.encode("utf-8"))
    if content_bytes > _MAX_ARTIFACT_CONTENT_BYTES:
        raise ValueError(
            f"artifact content exceeds {_MAX_ARTIFACT_CONTENT_BYTES} bytes"
        )
    artifact = {
        "id": f"flow-{uuid.uuid4().hex[:12]}",
        "type": kind,
        "title": title,
        "content": body,
        "source": "flow",
    }
    return {"artifact": artifact}


def _condition_operand(token: str, ctx: dict) -> Any:
    if re.fullmatch(r"\{\{[^{}]+\}\}", token):
        return _resolve_value(token, ctx)
    if token[:1] in {"\"", "'"}:
        value = ast.literal_eval(token)
        if not isinstance(value, str):
            raise ValueError("quoted condition operands must be strings")
        return value

    lowered = token.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if token in {"[]", "{}"}:
        return json.loads(token)
    if _NUMBER_LITERAL_RE.fullmatch(token):
        value = float(token) if any(marker in token.lower() for marker in (".", "e")) else int(token)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("condition numbers must be finite")
        return value
    return token


def _condition_equal(left: Any, right: Any) -> bool:
    left_number = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_number = isinstance(right, (int, float)) and not isinstance(right, bool)
    if left_number or right_number:
        return left_number and right_number and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _condition_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _condition_equal(left[key], right[key]) for key in left
        )
    return left == right


def _condition_compare(left: Any, operator: str | None, right: Any = None) -> bool:
    if operator is None:
        return bool(left)
    if operator == "==":
        return _condition_equal(left, right)
    if operator == "!=":
        return not _condition_equal(left, right)

    both_numbers = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (left, right)
    )
    both_strings = isinstance(left, str) and isinstance(right, str)
    if not (both_numbers or both_strings):
        return False
    if operator == "<":
        return left < right
    if operator == ">":
        return left > right
    if operator == "<=":
        return left <= right
    if operator == ">=":
        return left >= right
    return False


async def _run_if_else(data: dict, ctx: dict) -> dict:
    raw_expression = data.get("expr", "")
    expression = raw_expression if isinstance(raw_expression, str) else str(raw_expression)
    branch = "false"
    if len(expression) <= _MAX_CONDITION_CHARS:
        safe = _CONDITION_RE.fullmatch(expression)
        if safe:
            try:
                left = _condition_operand(safe.group("lhs"), ctx)
                operator = safe.group("op")
                right = (
                    _condition_operand(safe.group("rhs"), ctx)
                    if operator is not None
                    else None
                )
                branch = "true" if _condition_compare(left, operator, right) else "false"
            except (SyntaxError, TypeError, ValueError):
                branch = "false"
    return {"_branch": branch, "value": ctx.get("_last", "")}


async def _run_iterate(data: dict, ctx: dict) -> list[Any]:
    """Normalize a list input and optionally map a template over each item.

    The previous implementation stringified Python lists and often produced one
    malformed item. Keeping native template values makes list propagation
    deterministic; ``template`` provides useful per-item semantics without
    pretending an arbitrary cyclic subgraph can be repeated safely.
    """
    raw = _resolve_value(data.get("list_expr", "{{input}}"), ctx)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            items: list[Any] = []
        else:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = [line for line in raw.splitlines() if line]
            items = parsed if isinstance(parsed, list) else [parsed]
    elif isinstance(raw, (tuple, set)):
        items = list(raw)
    elif isinstance(raw, list):
        items = raw
    elif raw is None:
        items = []
    else:
        items = [raw]

    template = data.get("template")
    if not isinstance(template, str) or not template:
        return items

    mapped: list[Any] = []
    for index, item in enumerate(items):
        item_ctx = {
            **ctx,
            "_last": item,
            "item": item,
            "index": index,
        }
        rendered = _interpolate(template, item_ctx)
        if data.get("parse_json"):
            try:
                mapped.append(json.loads(rendered))
                continue
            except json.JSONDecodeError:
                pass
        mapped.append(rendered)
    return mapped


async def _run_user_input(data: dict, ctx: dict, node_id: str) -> Any:
    """Resolve a user-input node from ``env.user_input`` or a default value."""
    key = str(data.get("key") or node_id)
    env = ctx.get("env", {})
    supplied = env.get("user_input", {}) if isinstance(env, dict) else {}
    if isinstance(supplied, dict):
        if key in supplied:
            return supplied[key]
        if node_id in supplied:
            return supplied[node_id]
    if isinstance(env, dict) and key in env:
        return env[key]
    if "default" in data:
        return data["default"]
    label = str(data.get("label") or key)
    raise ValueError(f"missing user input: {label} ({key})")


def required_flow_skill_grants(
    flow: dict,
    request_grants: frozenset[str] | None = None,
) -> list[dict[str, str]]:
    """Return sensitive nested skills not granted by the Mio operator.

    A published flow is a bounded capability: the model still needs the
    request-level grant for ``run_flow_skill``, while every nested sensitive
    skill must be explicitly granted by the operator. Preflighting the whole
    graph prevents earlier nodes from producing side effects before a later
    permission failure.
    """
    from mio.web_security import webui_skill_operator_granted, webui_skill_risk

    missing: dict[str, str] = {}
    nodes = flow.get("nodes") if isinstance(flow, dict) else None
    for node in (nodes or {}).values():
        if not isinstance(node, dict) or (node.get("class") or node.get("name")) != "skill_call":
            continue
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        skill = str(data.get("skill") or "")
        risk = webui_skill_risk(skill)
        if risk != "read" and (
            not webui_skill_operator_granted(skill)
            or (request_grants is not None and skill not in request_grants)
        ):
            missing[skill] = risk
    return [{"skill": skill, "risk": missing[skill]} for skill in sorted(missing)]


async def _execute_run_inner(run: _Run) -> None:
    """Topologically walk the graph in a single pass from user_input /
    first llm_call nodes to output nodes. For this MVP we do a naïve BFS
    respecting `connections_in` dependencies — no parallelism."""
    nodes = {str(node_id): node for node_id, node in (run.flow.get("nodes") or {}).items() if isinstance(node, dict)}
    if not nodes:
        await run.emit({"type": "run_finished", "ok": False, "error": "empty flow"})
        run.close()
        return

    if len(nodes) > _MAX_HOPS:
        await run.emit(
            {
                "type": "run_finished",
                "ok": False,
                "error": f"flow exceeds the {_MAX_HOPS}-node safety limit",
            }
        )
        run.close()
        return

    missing_grants = required_flow_skill_grants(
        run.flow,
        request_grants=run.request_skill_grants,
    )
    if missing_grants:
        await run.emit(
            {
                "type": "run_finished",
                "ok": False,
                "error": "flow contains sensitive skills without required grants",
                "missing_grants": missing_grants,
            }
        )
        run.close()
        return

    # Build immutable incoming/outgoing maps. The old Kahn pass mutated the
    # incoming lists, erasing every predecessor before value propagation.
    incoming: dict[str, list[str]] = {str(node_id): [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {str(node_id): [] for node_id in nodes}
    edge_ports: dict[tuple[str, str], str] = {}
    for nid, node in nodes.items():
        for port in (node.get("inputs") or {}).values():
            for conn in port.get("connections", []):
                source = str(conn.get("node"))
                target = str(nid)
                if source not in nodes or source in incoming[target]:
                    continue
                incoming[target].append(source)
                outgoing[source].append(target)
                # Drawflow exports the source output port as `input` on the
                # target-side connection object. Keep `output` as compatibility
                # for early Mio graph files written before this was corrected.
                edge_ports[(source, target)] = str(
                    conn.get("input") or conn.get("output") or ""
                )

    # Toposort
    order: list[str] = []
    indegree = {nid: len(deps) for nid, deps in incoming.items()}
    queue = [nid for nid in nodes if indegree[str(nid)] == 0]
    while queue:
        nid = str(queue.pop(0))
        order.append(nid)
        for target in outgoing[nid]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(order) != len(nodes):
        unresolved = [str(node_id) for node_id in nodes if str(node_id) not in order]
        await run.emit(
            {
                "type": "run_finished",
                "ok": False,
                "error": "flow contains a cycle or unresolved dependency",
                "unresolved": unresolved,
            }
        )
        run.close()
        return

    ctx = {
        "env": run.env,
        "outputs": run.outputs,
        "_last": "",
        "_manager": run.manager,
        "_gpu_lock": run.gpu_lock,
        "_cancelled": run.cancelled,
        "_request_skill_grants": run.request_skill_grants,
    }
    outputs_bucket: list[Any] = []
    skipped: set[str] = set()

    await run.emit({"type": "run_started", "node_order": order})
    for nid in order:
        node = nodes.get(nid)
        if not node:
            continue
        cls = node.get("class") or node.get("name") or ""
        data = node.get("data", {}) or {}

        # Gather active upstream outputs. If/else nodes activate output_1 for
        # true and output_2 for false; an entirely inactive branch is skipped.
        predecessors = incoming.get(nid) or []
        active_predecessors: list[str] = []
        for source in predecessors:
            if source in skipped:
                continue
            source_output = run.outputs.get(source)
            source_node = nodes.get(source) or {}
            source_class = source_node.get("class") or source_node.get("name") or ""
            if source_class == "if_else" and isinstance(source_output, dict):
                expected = "output_1" if source_output.get("_branch") == "true" else "output_2"
                port = edge_ports.get((source, nid), "")
                if port and port != expected:
                    continue
            active_predecessors.append(source)

        if predecessors and not active_predecessors:
            skipped.add(nid)
            await run.emit({"type": "node_skipped", "node_id": nid, "class": cls})
            continue

        upstream_values = []
        for source in active_predecessors:
            value = run.outputs.get(source, "")
            if isinstance(value, dict) and "_branch" in value and "value" in value:
                value = value["value"]
            upstream_values.append(value)
        ctx["_inputs"] = {source: value for source, value in zip(active_predecessors, upstream_values, strict=True)}
        if len(upstream_values) == 1:
            ctx["_last"] = upstream_values[0]
        elif upstream_values:
            ctx["_last"] = upstream_values
        else:
            ctx["_last"] = ""

        await run.emit({"type": "node_started", "node_id": nid, "class": cls})
        try:
            if cls == "llm_call":
                out = await _run_llm(data, ctx)
            elif cls == "skill_call":
                out = await _run_skill(data, ctx)
            elif cls == "http_fetch":
                out = await _run_http(data, ctx)
            elif cls == "if_else":
                out = await _run_if_else(data, ctx)
            elif cls == "iterate":
                out = await _run_iterate(data, ctx)
            elif cls == "user_input":
                out = await _run_user_input(data, ctx, nid)
            elif cls == "output":
                out = ctx["_last"]
                outputs_bucket.append(out)
            elif cls == "constant":
                out = await _run_constant(data, ctx)
            elif cls == "template":
                out = await _run_template(data, ctx)
            elif cls == "parse_json":
                out = await _run_parse_json(data, ctx)
            elif cls == "to_json":
                out = await _run_to_json(data, ctx)
            elif cls == "regex_extract":
                out = await _run_regex_extract(data, ctx)
            elif cls == "split":
                out = await _run_split(data, ctx)
            elif cls == "join":
                out = await _run_join(data, ctx)
            elif cls == "mem_get":
                out = await _run_mem_get(data, ctx)
            elif cls == "mem_set":
                out = await _run_mem_set(data, ctx)
            elif cls == "delay":
                out = await _run_delay(data, ctx)
            elif cls == "clock":
                out = await _run_clock(data, ctx)
            elif cls == "random":
                out = await _run_random(data, ctx)
            elif cls == "rag_search":
                out = await _run_rag_search(data, ctx)
            elif cls == "artifact_emit":
                out = await _run_artifact_emit(data, ctx)
            else:
                raise ValueError(f"unknown node class: {cls or '(empty)'}")

            if cls in {"http_fetch", "skill_call", "rag_search", "parse_json"} and isinstance(out, dict):
                error = out.get("error") or out.get("_error")
                if error or out.get("ok") is False:
                    raise RuntimeError(str(error or f"{cls} returned ok=false"))

            run.outputs[nid] = out
            if cls == "artifact_emit":
                artifact = out["artifact"]
                artifact_bytes = len(artifact["content"].encode("utf-8"))
                if run.artifact_count >= _MAX_ARTIFACTS_PER_RUN:
                    raise ValueError(
                        f"flow exceeds the {_MAX_ARTIFACTS_PER_RUN}-artifact run limit"
                    )
                if run.artifact_bytes + artifact_bytes > _MAX_ARTIFACT_RUN_BYTES:
                    raise ValueError(
                        f"flow artifact output exceeds {_MAX_ARTIFACT_RUN_BYTES} bytes"
                    )
                run.artifact_count += 1
                run.artifact_bytes += artifact_bytes
                await run.emit(
                    {
                        "type": "artifact_emitted",
                        "node_id": nid,
                        "class": cls,
                        "artifact": artifact,
                    }
                )
                output_preview: Any = {
                    "artifact": {
                        "id": artifact["id"],
                        "type": artifact["type"],
                        "title": artifact["title"],
                        "content_bytes": artifact_bytes,
                    }
                }
            else:
                output_preview = _preview(out)
            await run.emit(
                {
                    "type": "node_finished",
                    "node_id": nid,
                    "class": cls,
                    "output": output_preview,
                }
            )
        except Exception as e:
            error = str(e)
            await run.emit({"type": "node_error", "node_id": nid, "class": cls, "error": error})
            run.outputs[nid] = {"_error": error}
            # A downstream node cannot distinguish a real error-shaped value
            # from normal data. Fail fast instead of reporting a false success
            # or allowing side-effecting nodes to run with poisoned inputs.
            await run.emit(
                {
                    "type": "run_finished",
                    "ok": False,
                    "error": f"node {nid} ({cls or 'unknown'}) failed: {error}",
                    "failed_node": nid,
                    "outputs": outputs_bucket,
                    "elapsed_ms": int((time.time() - run.started) * 1000),
                }
            )
            run.close()
            return

    await run.emit(
        {
            "type": "run_finished",
            "ok": True,
            "outputs": outputs_bucket,
            "elapsed_ms": int((time.time() - run.started) * 1000),
        }
    )
    run.close()


async def _execute_run(run: _Run) -> None:
    """Execute one run and always publish a terminal state.

    Persisted graphs are untrusted state: malformed connection shapes can
    fail before a node starts. Keeping the terminal guard outside the graph
    walker prevents zombie runs from exhausting the bounded registry.
    """
    try:
        await _execute_run_inner(run)
    except asyncio.CancelledError:
        run.cancel()
        if not run.done:
            await run.emit(
                {
                    "type": "run_finished",
                    "ok": False,
                    "error": "flow run was cancelled",
                    "elapsed_ms": int((time.time() - run.started) * 1000),
                }
            )
            run.close()
        raise
    except Exception as exc:
        run.cancel()
        if not run.done:
            await run.emit(
                {
                    "type": "run_finished",
                    "ok": False,
                    "error": f"invalid flow graph: {type(exc).__name__}: {exc}",
                    "elapsed_ms": int((time.time() - run.started) * 1000),
                }
            )
            run.close()


def _preview(val: Any) -> Any:
    if isinstance(val, str):
        return val[:400]
    if isinstance(val, (dict, list)):
        try:
            s = json.dumps(val)[:400]
            return json.loads(s) if s.strip().startswith(("[", "{")) else s
        except Exception:
            return str(val)[:400]
    return val


def start_run(
    flow: dict,
    env: dict | None = None,
    *,
    manager: Any = None,
    gpu_lock: Any = None,
) -> str:
    _prune_runs()
    if len(_runs) >= _MAX_RUNS:
        raise RuntimeError(f"too many active Flow runs (limit {_MAX_RUNS})")
    run_id = str(uuid.uuid4())[:12]
    run = _Run(run_id, flow, env, manager=manager, gpu_lock=gpu_lock)
    _runs[run_id] = run
    run.task = asyncio.create_task(_execute_run(run))
    return run_id


def get_run(run_id: str) -> _Run | None:
    _prune_runs()
    return _runs.get(run_id)


def discard_run(run_id: str) -> None:
    run = _runs.pop(run_id, None)
    if run is None or run.done:
        return
    run.cancel()
    if run.task is not None and not run.task.done():
        run.task.cancel()


def _prune_runs(now: float | None = None) -> None:
    """Evict completed run state by age and enforce a hard registry bound."""
    now = time.time() if now is None else now
    for run_id, run in list(_runs.items()):
        if run.done and run.finished is not None and now - run.finished >= _RUN_TTL_SECONDS:
            _runs.pop(run_id, None)
    if len(_runs) < _MAX_RUNS:
        return
    completed = sorted(
        ((run_id, run) for run_id, run in _runs.items() if run.done),
        key=lambda item: item[1].finished or item[1].started,
    )
    for run_id, _run in completed:
        if len(_runs) < _MAX_RUNS:
            break
        _runs.pop(run_id, None)
