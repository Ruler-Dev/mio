"""FastAPI OpenAI-compatible API server."""

from __future__ import annotations

import json
import time
import uuid
import threading
from collections import deque
from typing import Any

# Serializes MLX work across concurrent HTTP requests. Without this two
# producer threads can encode Metal command buffers simultaneously and
# Apple's driver aborts with "command encoder is already encoding".
_GPU_LOCK = threading.Lock()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mio.model_manager import ModelManager
from mio.router import TandemRouter

app = FastAPI(title="Mio", version="0.1.0")

# Global state — set by start_server()
_manager: ModelManager | None = None
_router: TandemRouter | None = None
_validate_enabled: bool = False
_caveman_level: str = "full"
# Context auto-compaction thresholds (set by start_server)
_compact_threshold: float = 0.75
_compact_target: float = 0.50
_compact_enabled: bool = True
_compact_summarize: bool = True


# --- Live per-request stats for serve-mode console ---

class _ServeStats:
    """Running stats for the live serve panel.

    Tracks per-request decode tok/s, prefill tok/s, acceptance, tool-call
    detection, and exposes a rich.Table renderable for a live TUI.
    """

    def __init__(self, history: int = 20) -> None:
        self.requests = 0
        self.tool_calls = 0
        self.cache_hits = 0
        self.cache_hit_tokens = 0  # cumulative tokens saved via prefix cache
        self.compactions = 0
        self.compact_tokens_saved = 0  # cumulative tokens saved via compaction
        self.started_at = time.time()
        self._decode_tps: list[float] = []
        self._prefill_tps: list[float] = []
        # Recent request log for display
        self._recent: deque = deque(maxlen=history)

    def record(self, gen_metrics, wall_s: float, tier: str, text: str) -> None:
        self.requests += 1
        dec = float(getattr(gen_metrics, "generation_tps", 0.0) or 0.0)
        pre = float(getattr(gen_metrics, "prompt_tps", 0.0) or 0.0)
        self._decode_tps.append(dec)
        self._prefill_tps.append(pre)
        if len(self._decode_tps) > 100:
            self._decode_tps = self._decode_tps[-100:]
            self._prefill_tps = self._prefill_tps[-100:]

        # Detect tool-call formats used by Cline / Kilo Code / OpenAI / Claude.
        t = text or ""
        is_tool = (
            # Generic XML tool wrappers
            "<tool" in t or "<function" in t or "<execute" in t
            # Cline / Kilo Code per-tool tags (non-exhaustive but covers the common ones)
            or "<read_file>" in t or "<write_to_file>" in t or "<replace_in_file>" in t
            or "<execute_command>" in t or "<list_files>" in t or "<search_files>" in t
            or "<list_code_definition_names>" in t or "<use_mcp_tool>" in t
            or "<access_mcp_resource>" in t or "<ask_followup_question>" in t
            or "<attempt_completion>" in t or "<new_task>" in t or "<plan_mode_response>" in t
            # OpenAI / Anthropic JSON-style
            or '"tool_calls"' in t or '"function"' in t or '"tool_use"' in t
        )
        if is_tool:
            self.tool_calls += 1
        snippet = (text or "").strip().replace("\n", " ")[:200]

        warm_off = int(getattr(gen_metrics, "warm_offset", 0) or 0)
        cache_n = int(getattr(gen_metrics, "cache_entries", 0) or 0)
        if warm_off > 0:
            self.cache_hits += 1
            self.cache_hit_tokens += warm_off

        self._recent.appendleft({
            "id": self.requests,
            "tier": tier,
            "prompt": int(getattr(gen_metrics, "prompt_tokens", 0) or 0),
            "gen": int(getattr(gen_metrics, "completion_tokens", 0) or 0),
            "wall_ms": wall_s * 1000,
            "prefill_tps": pre,
            "decode_tps": dec,
            "accept": float(getattr(gen_metrics, "acceptance_ratio", 0.0) or 0.0),
            "tool": is_tool,
            "warm_off": warm_off,
            "cache_n": cache_n,
            "snippet": snippet,
        })

    # ---- Aggregate getters (for live panel renderer) ----
    def avg_decode_tps(self) -> float:
        return sum(self._decode_tps) / len(self._decode_tps) if self._decode_tps else 0.0

    def p1_low_decode_tps(self) -> float:
        if not self._decode_tps:
            return 0.0
        s = sorted(self._decode_tps)
        return s[max(0, int(len(s) * 0.01))]

    def max_decode_tps(self) -> float:
        return max(self._decode_tps) if self._decode_tps else 0.0

    def avg_prefill_tps(self) -> float:
        return sum(self._prefill_tps) / len(self._prefill_tps) if self._prefill_tps else 0.0

    def max_prefill_tps(self) -> float:
        return max(self._prefill_tps) if self._prefill_tps else 0.0


_stats = _ServeStats()


def _debug_log(event: str, data: dict) -> None:
    """Append a JSON line to /tmp/mio-serve-debug.log when MIO_DEBUG_LOG=1.

    Used to compare exactly what mio receives vs. what it generates against
    other local LLM servers (LM Studio, llama.cpp server).
    """
    import os
    if os.environ.get("MIO_DEBUG_LOG", "") not in ("1", "true", "yes"):
        return
    path = os.environ.get("MIO_DEBUG_LOG_PATH", "/tmp/mio-serve-debug.log")
    try:
        with open(path, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "event": event, **data,
            }, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# --- Streaming <think>...</think> stripper ---
#
# Qwen3.5 is a hybrid reasoning model. Even with enable_thinking=False in the
# chat template (which prepends an empty <think></think>), the model can still
# emit reasoning blocks mid-response. Cline/Kilo/Roo XML tool parsers don't
# recognize <think> — they stop at the first non-whitespace non-XML byte and
# report "assistant wrote plain text". We drop think blocks before they reach
# the client so the actual tool-call XML is what gets parsed.
#
# Stateful because blocks can straddle chunk boundaries. Only the content
# INSIDE the tags is dropped; tool XML outside <think> is emitted verbatim,
# including literal triple-backticks (file contents often contain them).

class _ThinkStripper:
    """Drop <think>...</think> spans from a streamed assistant response.

    Only holds chars back when a partial "<think" prefix is actually forming,
    so tool-call XML streams through with minimal latency.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if self._in_think:
                end = self._buf.find(self._CLOSE)
                if end < 0:
                    # Hold a tail big enough to catch a split "</think>".
                    keep = min(len(self._buf), len(self._CLOSE) - 1)
                    self._buf = self._buf[-keep:] if keep else ""
                    break
                self._buf = self._buf[end + len(self._CLOSE):]
                self._in_think = False
                if self._buf.startswith("\n"):
                    self._buf = self._buf[1:]
            else:
                start = self._buf.find(self._OPEN)
                if start >= 0:
                    if start > 0:
                        out.append(self._buf[:start])
                    self._buf = self._buf[start + len(self._OPEN):]
                    self._in_think = True
                    continue
                # No complete "<think>". Find the longest suffix of buf that
                # is a prefix of "<think>" — only THAT needs holding.
                hold = 0
                max_hold = min(len(self._buf), len(self._OPEN) - 1)
                for k in range(max_hold, 0, -1):
                    if self._OPEN.startswith(self._buf[-k:]):
                        hold = k
                        break
                if hold:
                    out.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out.append(self._buf)
                    self._buf = ""
                break
        return "".join(out)

    def flush(self) -> str:
        buf = self._buf
        self._buf = ""
        if self._in_think:
            return ""
        return buf


def _render_live_panel() -> "Any":
    """Build a rich renderable reflecting current serve state.

    Called by the rich.Live loop in start_server() at ~4 Hz.
    """
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Group
    from rich.text import Text

    # Top summary
    uptime_s = int(time.time() - _stats.started_at)
    mm, ss = divmod(uptime_s, 60)
    hh, mm = divmod(mm, 60)
    loaded = ", ".join(_manager.loaded_tiers()) if _manager else "—"

    summary = Text.from_markup(
        f"[bold cyan]Mio serve[/bold cyan]   "
        f"[dim]uptime[/dim] {hh:02d}:{mm:02d}:{ss:02d}   "
        f"[dim]tiers[/dim] [yellow]{loaded}[/yellow]   "
        f"[dim]caveman[/dim] [green]{_caveman_level}[/green]"
    )

    # Aggregate numbers row
    hit_rate = (_stats.cache_hits / _stats.requests * 100) if _stats.requests else 0.0
    agg = Table.grid(padding=(0, 2))
    agg.add_column(style="dim")
    agg.add_column()
    agg.add_column(style="dim")
    agg.add_column()
    agg.add_row(
        "requests", f"[bold]{_stats.requests}[/bold]",
        "tool-calls", f"[bold yellow]{_stats.tool_calls}[/bold yellow]",
    )
    agg.add_row(
        "decode avg", f"[bold green]{_stats.avg_decode_tps():.1f}[/bold green] tok/s",
        "decode max", f"[bold green]{_stats.max_decode_tps():.1f}[/bold green] tok/s",
    )
    agg.add_row(
        "decode 1% low", f"[bold red]{_stats.p1_low_decode_tps():.1f}[/bold red] tok/s",
        "prefill max", f"[bold cyan]{_stats.max_prefill_tps():.0f}[/bold cyan] tok/s",
    )
    agg.add_row(
        "prefill avg", f"[bold cyan]{_stats.avg_prefill_tps():.0f}[/bold cyan] tok/s",
        "cache hits", (
            f"[bold magenta]{_stats.cache_hits}/{_stats.requests}[/bold magenta] "
            f"({hit_rate:.0f}%, {_stats.cache_hit_tokens:,} tok saved)"
        ),
    )
    agg.add_row(
        "compactions", (
            f"[bold yellow]{_stats.compactions}[/bold yellow] "
            f"({_stats.compact_tokens_saved:,} tok reclaimed)"
        ),
        "", "",
    )

    # Recent requests table
    reqs = Table(
        title=None, show_header=True, header_style="bold",
        expand=True, box=None, padding=(0, 1),
    )
    reqs.add_column("#", style="dim", width=5, justify="right")
    reqs.add_column("tier", width=11)
    reqs.add_column("prompt", justify="right", width=7)
    reqs.add_column("gen", justify="right", width=6)
    reqs.add_column("wall", justify="right", width=8)
    reqs.add_column("pref t/s", justify="right", width=9)
    reqs.add_column("dec t/s", justify="right", width=9)
    reqs.add_column("accept", justify="right", width=7)
    reqs.add_column("cache", justify="right", width=12)
    reqs.add_column("tool", width=5)
    reqs.add_column("snippet", ratio=1, no_wrap=True)
    for r in list(_stats._recent)[:12]:
        # Cache column: HIT with tokens skipped, or MISS, or size-only if no request yet
        warm = r.get("warm_off", 0)
        cache_n = r.get("cache_n", 0)
        if warm > 0:
            cache_cell = f"[green]HIT {warm}[/green]"
        else:
            cache_cell = f"[red]MISS[/red] ({cache_n})"
        reqs.add_row(
            str(r["id"]),
            r["tier"],
            str(r["prompt"]),
            f"{r['gen']}tok",
            f"{r['wall_ms']:.0f}ms",
            f"{r['prefill_tps']:.0f}",
            f"{r['decode_tps']:.1f}",
            f"{r['accept']:.2f}",
            cache_cell,
            "[yellow]●[/yellow]" if r["tool"] else "",
            r["snippet"],
        )

    return Panel(
        Group(summary, Text(""), agg, Text(""), reqs),
        title="mio live", border_style="cyan",
    )


# Markers indicating the system prompt ITSELF specifies a Cline/Roo XML
# tool-call protocol (the protocol is described in the prompt, not passed
# via the OpenAI `tools` field). When these are present we skip caveman —
# exact tag-name matching is required and caveman's "drop articles" would
# corrupt tag names. Client name strings like "Kilo"/"Cline" are NOT
# markers: Kilo specifically uses OpenAI function tools, not XML, so a
# system prompt mentioning "Kilo" is fine for caveman.
_TOOL_CLIENT_MARKERS = (
    "<read_file>", "<write_to_file>", "<replace_in_file>",
    "<execute_command>", "<list_files>", "<search_files>",
    "<attempt_completion>", "<use_mcp_tool>", "<ask_followup_question>",
    "<apply_diff>", "<edit_file>", "<new_task>", "<plan_mode_response>",
)


# Appended to the user's LAST message when `tools` are provided but the user's
# request clearly asks for action. Counteracts Qwen3.5's built-in escape hatch
# ("answer like normal if no tool is applicable") — quantized Qwen tends to
# narrate ("Now I'll update the file:") then stop with EOS instead of emitting
# the write/execute tool call.
_TOOL_ACTION_REMINDER = (
    "\n\nReminder: respond ONLY with a single <tool_call>...</tool_call> block "
    "as specified above. Do not narrate what you will do. Do not write "
    "'Now I'll ...' or similar prose. Emit the tool call directly. To modify "
    "files use write/edit tools; to run commands use bash. Empty prose without "
    "a tool call is always incorrect when tools are available."
)


def _apply_caveman(messages: list[dict], tools: list[dict] | None = None) -> list[dict]:
    """Inject the configured Caveman system prompt + tool-action reminder.

    Caveman's "be terse, drop articles" directive is SAFE to apply alongside
    tools because Qwen's chat template renders exact tool names and the
    `Technical terms exact` rule in caveman preserves them. Caveman only
    shortens narrative prose — which Kilo ignores anyway (it parses
    tool_calls, not content).

    When tools are present we also append `_TOOL_ACTION_REMINDER` to the
    last user message so Qwen can't escape via "Now I'll update..." + EOS.
    """
    from mio.agent import CAVEMAN_LEVELS

    out = list(messages)

    # Caveman: prepend to system prompt (or insert new system message) —
    # unless the system prompt is already a Cline/Roo XML protocol spec,
    # where exact-tag-name matching would break.
    text = CAVEMAN_LEVELS.get(_caveman_level, "").strip()
    if text:
        if out and out[0].get("role") == "system":
            sys_content = out[0].get("content", "") or ""
            if not any(m in sys_content for m in _TOOL_CLIENT_MARKERS):
                merged = text + "\n\n" + sys_content
                out = [{"role": "system", "content": merged}, *out[1:]]
        else:
            out = [{"role": "system", "content": text}, *out]

    # Tool-action reminder: appended to the last user turn when tools are
    # supplied, so the most recent instruction forces a tool call.
    if tools and out and out[-1].get("role") == "user":
        last = dict(out[-1])
        last_content = last.get("content", "") or ""
        last["content"] = last_content.rstrip() + _TOOL_ACTION_REMINDER
        out = [*out[:-1], last]

    return out


# --- Request/Response Models ---


class ChatMessage(BaseModel):
    # Accept both plain strings (legacy) and OpenAI's multimodal content-parts
    # list (used by Cline, many SDKs, etc). We normalize to a plain string at
    # request time — mio is text-only for now.
    model_config = {"extra": "allow"}  # tolerate extra fields (name, tool_call_id, etc.)
    role: str
    content: str | list[dict] | None = None


def _coerce_message_content(content: str | list[dict] | None) -> str:
    """Normalize OpenAI multimodal content lists to a plain string.

    [{"type": "text", "text": "hi"}, {"type": "image_url", ...}] → "hi"
    Strings pass through unchanged. None/empty → "".
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # List of parts: concatenate text parts, drop non-text silently.
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


class ChatCompletionRequest(BaseModel):
    # Be lenient: Cline / OpenAI clients send extra fields (tools, tool_choice,
    # response_format, user, etc.) that mio ignores but shouldn't 422 on.
    model_config = {"extra": "allow"}
    model: str = "mio-large"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    stop: list[str] | None = None
    validate: bool = False  # Mio extension: auto-validate generated code


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mio"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    loaded_tiers: list[str]
    vram_gb: float
    models: list[str]


class MetricsResponse(BaseModel):
    tiers: dict[str, Any]


class TierLoadRequest(BaseModel):
    tier: str


# --- Endpoints ---


@app.get("/health")
async def health() -> HealthResponse:
    if not _manager:
        return HealthResponse(status="not_initialized", loaded_tiers=[], vram_gb=0, models=[])
    return HealthResponse(
        status="ready",
        loaded_tiers=_manager.loaded_tiers(),
        vram_gb=_manager.total_vram_gb(),
        models=_manager.get_model_names(),
    )


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    if not _manager:
        return ModelsResponse(data=[])
    return ModelsResponse(
        data=[ModelInfo(id=name) for name in _manager.get_model_names()]
    )


@app.post("/v1/models/load")
async def load_model(req: TierLoadRequest) -> dict:
    if not _manager:
        raise HTTPException(500, "Server not initialized")
    try:
        _manager.load_tier(req.tier)
        return {"status": "loaded", "tier": req.tier}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/v1/models/unload")
async def unload_model(req: TierLoadRequest) -> dict:
    if not _manager:
        raise HTTPException(500, "Server not initialized")
    _manager.unload_tier(req.tier)
    return {"status": "unloaded", "tier": req.tier}


@app.get("/metrics")
async def metrics() -> MetricsResponse:
    if not _manager:
        return MetricsResponse(tiers={})
    status = _manager.status()
    return MetricsResponse(tiers=status.get("engines", {}))


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    if not _manager:
        raise HTTPException(500, "Server not initialized")

    # Preserve tool_calls / tool_call_id / name through to the chat template
    # so Qwen can re-render prior assistant tool calls and tool results as the
    # exact same bytes it generated last turn — without this the prefix cache
    # can never hit and every turn re-prefills the full context.
    dumped_msgs = req.model_dump().get("messages", [])
    messages: list[dict] = []
    for m in dumped_msgs:
        out = {"role": m.get("role", "user"), "content": _coerce_message_content(m.get("content"))}
        tc = m.get("tool_calls")
        if tc:
            # Qwen's chat template wants `arguments` as a dict, not a JSON string.
            norm: list[dict] = []
            for call in tc:
                call = dict(call)
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args else {}
                    except (json.JSONDecodeError, ValueError):
                        fn["arguments"] = {}
                call["function"] = fn
                norm.append(call)
            out["tool_calls"] = norm
        if m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            out["name"] = m["name"]
        messages.append(out)

    extra = {k: v for k, v in req.model_dump().items()
             if k not in ("model","messages","max_tokens","temperature","stream","stop","validate")}
    tools = extra.get("tools") if isinstance(extra.get("tools"), list) else None

    _debug_log("request_raw", {
        "model": req.model, "stream": req.stream,
        "max_tokens": req.max_tokens, "temperature": req.temperature,
        "stop": req.stop,
        "messages": messages,
        "extra_fields": extra,
    })

    # Route to tier first — compaction needs the engine's tokenizer + context window.
    tier_name = _resolve_tier(req.model)
    try:
        engine = _manager.get_engine(tier_name)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    # Context auto-compaction: when prompt tokens exceed `_compact_threshold`
    # of the context window, reduce to `_compact_target`. Prevents Metal
    # OOM crashes on long agent sessions. Runs BEFORE caveman so caveman
    # injection targets the compacted message list.
    if _compact_enabled:
        from mio.compactor import compact
        messages, compact_stats = compact(
            messages, engine, tools=tools,
            threshold=_compact_threshold,
            target=_compact_target,
            enable_summarization=_compact_summarize,
            gpu_lock=_GPU_LOCK,
        )
        if compact_stats.triggered:
            _stats.compactions += 1
            _stats.compact_tokens_saved += compact_stats.tokens_saved
            print(
                f"[compact] stage={compact_stats.stage} "
                f"{compact_stats.before_tokens:,} → {compact_stats.after_tokens:,} "
                f"tokens ({compact_stats.tokens_saved:,} saved, "
                f"{compact_stats.tool_results_truncated} tool results truncated, "
                f"{compact_stats.messages_summarized} turns summarized)",
                flush=True,
            )
            _debug_log("compact", compact_stats.as_dict())

    messages = _apply_caveman(messages, tools=tools)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = f"mio-{tier_name}"

    if req.stream:
        return StreamingResponse(
            _stream_response(
                engine, messages, completion_id, created, model_name, tier_name,
                max_tokens=req.max_tokens, temperature=req.temperature, stop=req.stop,
                tools=tools,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming (with optional validation + retry). Must also hold the
    # GPU lock to stay serialized with streaming requests.
    _ns_t0 = time.perf_counter()
    def _run_generate():
        with _GPU_LOCK:
            return engine.generate(
                messages, max_tokens=req.max_tokens,
                temperature=req.temperature, stop=req.stop,
                tools=tools,
            )
    import asyncio as _asyncio
    text, gen_metrics = await _asyncio.to_thread(_run_generate)
    # Strip <think>...</think> reasoning blocks.
    _stripper = _ThinkStripper()
    text = _stripper.feed(text) + _stripper.flush()
    # Parse Qwen3.5 native <tool_call>... into OpenAI tool_calls JSON.
    from mio.tool_calls import parse_tool_calls
    leading_text, parsed_calls = parse_tool_calls(text) if tools else (text, [])
    _stats.record(gen_metrics, time.perf_counter() - _ns_t0, tier_name, text)

    if req.validate and _validate_enabled:
        from mio.validator import validate_response, build_retry_message

        for _retry in range(2):
            result = validate_response(text)
            if result.passed:
                break
            retry_msg = build_retry_message(result.errors)
            retry_messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": retry_msg},
            ]
            text, gen_metrics = engine.generate(
                retry_messages, max_tokens=req.max_tokens,
                temperature=req.temperature, stop=req.stop,
            )

    # OpenAI-compatible shape: when the model emitted tool calls, content
    # becomes null and finish_reason becomes "tool_calls". Kilo/OpenAI SDKs
    # key off both.
    if parsed_calls:
        msg_payload: dict = {
            "role": "assistant",
            "content": leading_text or None,
            "tool_calls": parsed_calls,
        }
        finish = "tool_calls"
    else:
        msg_payload = {"role": "assistant", "content": text}
        finish = "stop"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": msg_payload, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": gen_metrics.prompt_tokens,
            "completion_tokens": gen_metrics.completion_tokens,
            "total_tokens": gen_metrics.total_tokens,
        },
    }


async def _stream_response(
    engine,
    messages: list[dict],
    completion_id: str,
    created: int,
    model_name: str,
    tier_name: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    tools: list[dict] | None = None,
):
    """Yield SSE chunks in OpenAI format.

    engine.generate_stream is a synchronous generator that can block for
    minutes during long-prompt prefill. We drive it from a background
    thread and use an asyncio.Queue (populated via call_soon_threadsafe)
    so the event loop stays responsive and we can emit SSE keepalives
    while prefill is running.
    """
    import asyncio
    import threading

    # Role chunk — client sees SOMETHING the moment the stream opens.
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    stream_t0 = time.perf_counter()
    full_text: list[str] = []

    def _producer():
        def _put(x):
            loop.call_soon_threadsafe(aq.put_nowait, x)
        try:
            # Serialize MLX work — only one request at a time touches the GPU.
            # Other requests wait here; they can still receive keepalives because
            # the asyncio consumer stays alive.
            with _GPU_LOCK:
                for item in engine.generate_stream(
                    messages, max_tokens=max_tokens, temperature=temperature,
                    stop=stop, tools=tools,
                ):
                    _put(item)
        except Exception as e:
            _put(("__error__", repr(e)))
        finally:
            _put(SENTINEL)

    threading.Thread(target=_producer, daemon=True).start()

    final_metrics = None
    KEEPALIVE_EVERY = 10.0
    stripper = _ThinkStripper()
    # Parse Qwen <tool_call>...</tool_call> spans out of the stream when the
    # client passed `tools` (OpenAI function-calling). Tool calls are deferred
    # to the final delta; content outside tool calls streams normally.
    from mio.tool_calls import StreamingToolCallParser
    tool_parser = StreamingToolCallParser() if tools else None

    while True:
        try:
            item = await asyncio.wait_for(aq.get(), timeout=KEEPALIVE_EVERY)
        except asyncio.TimeoutError:
            # Producer thread still working (likely prefilling). Send a
            # SSE comment to keep the HTTP connection warm for clients
            # with a body timeout (Cline, curl, etc.).
            yield ": keepalive\n\n"
            continue

        if item is SENTINEL:
            # Flush any remaining buffered characters.
            tail = stripper.flush()
            if tail:
                full_text.append(tail)
                if tool_parser is not None:
                    tail_content = tool_parser.feed(tail)
                else:
                    tail_content = tail
                if tail_content:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": tail_content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            _debug_log("response_stream", {
                "id": completion_id, "model": model_name,
                "full_text": "".join(full_text),
                "token_count": int(getattr(final_metrics, "completion_tokens", 0)) if final_metrics else 0,
            })
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
            err_msg = item[1]
            yield f"data: {json.dumps({'error': {'message': err_msg}})}\n\n"
            break

        text_chunk, chunk_metrics = item
        if chunk_metrics:
            final_metrics = chunk_metrics

        # Strip <think> blocks first, then extract any <tool_call> spans
        # (saved for final delta). What remains is streamed as content.
        filtered = stripper.feed(text_chunk) if text_chunk else ""
        if filtered:
            full_text.append(filtered)
            if tool_parser is not None:
                filtered = tool_parser.feed(filtered)
            if filtered:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"content": filtered}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

    # Collect tool calls parsed during the stream. When present, OpenAI spec
    # says finish_reason = "tool_calls" and the calls arrive as deltas —
    # Kilo/SDK clients wait for this exact signal before invoking tools.
    parsed_calls: list[dict] = []
    if tool_parser is not None:
        _residual, parsed_calls = tool_parser.flush()
        # Any residual text outside a tool call goes out as one last content delta.
        if _residual:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": _residual}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    if parsed_calls:
        # Emit each tool_call as its own delta, as OpenAI does for streaming.
        for idx, call in enumerate(parsed_calls):
            tc_delta = {
                "index": idx,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"tool_calls": [tc_delta]}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    finish_reason = "tool_calls" if parsed_calls else "stop"
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if final_metrics:
        final_chunk["usage"] = {
            "prompt_tokens": final_metrics.prompt_tokens,
            "completion_tokens": final_metrics.completion_tokens,
            "total_tokens": final_metrics.total_tokens,
        }
        _stats.record(
            final_metrics, time.perf_counter() - stream_t0, tier_name, "".join(full_text),
        )
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _resolve_tier(model: str) -> str:
    """Resolve model name to tier."""
    if not _manager:
        raise HTTPException(500, "Server not initialized")

    if model.startswith("mio-"):
        tier = model.removeprefix("mio-")
        if tier == "auto" and _router:
            return _router.route([], model)
        if tier in _manager.loaded_tiers():
            return tier

    # Default to first loaded tier
    loaded = _manager.loaded_tiers()
    if not loaded:
        raise HTTPException(503, "No models loaded")
    return loaded[0]


def init_server(
    manager: ModelManager,
    tandem: bool = False,
    validate: bool = False,
    caveman_level: str = "full",
    compact_threshold: float = 0.75,
    compact_target: float = 0.50,
    compact_summarize: bool = True,
) -> None:
    """Initialize server with a model manager."""
    global _manager, _router, _validate_enabled, _caveman_level
    global _compact_threshold, _compact_target, _compact_enabled, _compact_summarize
    _manager = manager
    _validate_enabled = validate
    _caveman_level = caveman_level
    _compact_threshold = float(compact_threshold)
    _compact_target = float(compact_target)
    _compact_enabled = _compact_threshold < 1.0
    _compact_summarize = bool(compact_summarize)
    if tandem and len(manager.loaded_tiers()) > 1:
        _router = TandemRouter(manager.loaded_tiers())


# --- Dashboard ---

@app.get("/dashboard")
async def dashboard():
    from mio.dashboard import get_dashboard_html
    return get_dashboard_html()


@app.websocket("/ws/metrics")
async def ws_metrics(websocket):
    from mio.dashboard import websocket_metrics
    await websocket_metrics(websocket)


# --- Batch ---

class BatchRequestItem(BaseModel):
    messages: list[ChatMessage]
    model: str = "mio-large"
    max_tokens: int | None = None
    temperature: float | None = None


class BatchCompletionRequest(BaseModel):
    requests: list[BatchRequestItem]


@app.post("/v1/batch")
async def batch_completions(req: BatchCompletionRequest) -> list[dict]:
    if not _manager:
        raise HTTPException(500, "Server not initialized")

    results = []
    for i, item in enumerate(req.requests):
        tier_name = _resolve_tier(item.model)
        try:
            engine = _manager.get_engine(tier_name)
            messages = [{"role": m.role, "content": m.content} for m in item.messages]
            messages = _apply_caveman(messages)
            text, metrics = engine.generate(
                messages, max_tokens=item.max_tokens, temperature=item.temperature,
            )
            results.append({
                "index": i, "text": text,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "generation_tps": metrics.generation_tps,
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return results


def _pids_on_port(port: int) -> list[int]:
    """Return PIDs currently bound to (or listening on) `port`. macOS/Linux."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"], stderr=subprocess.DEVNULL, timeout=3.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return sorted({int(p) for p in out.split() if p.strip().isdigit()})


def _kill_port_holders(port: int) -> None:
    """Kill any process currently bound to `port`. Dev-only convenience so
    `mio serve` can re-launch cleanly without manual cleanup."""
    import os
    import signal as _sig
    import time as _t

    self_pid = os.getpid()
    pids = [p for p in _pids_on_port(port) if p != self_pid]
    if not pids:
        return
    print(f"[serve] port {port} held by PIDs {pids}; killing...")

    # First, SIGTERM each; give a short grace period; then SIGKILL survivors.
    for pid in pids:
        try:
            os.kill(pid, _sig.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue

    for _ in range(10):                 # up to 1.0s
        _t.sleep(0.1)
        survivors = [p for p in _pids_on_port(port) if p != self_pid]
        if not survivors:
            break
    else:
        survivors = [p for p in _pids_on_port(port) if p != self_pid]

    for pid in survivors:
        try:
            os.kill(pid, _sig.SIGKILL)
            print(f"[serve] SIGKILL {pid}")
        except (ProcessLookupError, PermissionError):
            pass

    # Final wait for the socket to fully release (macOS can linger briefly
    # in TIME_WAIT/CLOSE_WAIT even after the process is gone).
    for _ in range(20):                 # up to 2.0s
        if not _pids_on_port(port):
            break
        _t.sleep(0.1)


def start_server(
    manager: ModelManager,
    host: str = "0.0.0.0",
    port: int = 9090,
    tandem: bool = False,
    validate: bool = False,
    caveman_level: str = "full",
    live_panel: bool = True,
    compact_threshold: float = 0.75,
    compact_target: float = 0.50,
    compact_summarize: bool = True,
    webui: bool = False,
) -> None:
    """Start the API server (blocking).

    If the stdout is a TTY and `live_panel=True`, renders a rich.Live panel
    showing running tok/s and recent requests. Otherwise falls back to plain
    per-request stdout logging (old behaviour).

    Before binding, any process currently holding `port` is killed — so
    rapid restarts of `mio serve` don't get "address already in use".
    """
    import sys
    import threading
    import uvicorn

    # Try up to 3 times to clear the port. Handles lingering CLOSE_WAIT sockets
    # from previous uvicorn crashes that lsof sometimes doesn't attribute to any PID.
    import socket as _sock
    import time as _tm
    for attempt in range(3):
        _kill_port_holders(port)
        test = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        test.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        try:
            test.bind((host, port))
            test.close()
            break
        except OSError:
            test.close()
            if attempt == 2:
                print(f"[serve] WARN: port {port} still not free after 3 attempts")
            else:
                _tm.sleep(0.5)

    init_server(
        manager, tandem, validate, caveman_level,
        compact_threshold=compact_threshold,
        compact_target=compact_target,
        compact_summarize=compact_summarize,
    )

    # Mount Mio UI if enabled
    if webui:
        from starlette.middleware.cors import CORSMiddleware
        from mio.webui.router import router as webui_router, mount_webui
        mount_webui(manager, caveman_level=caveman_level, gpu_lock=_GPU_LOCK)
        app.include_router(webui_router)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
        )

    print(f"\nMio API server starting on http://{host}:{port}")
    print(f"  Models:    {', '.join(manager.get_model_names())}")
    print(f"  Tandem:    {'enabled' if tandem else 'disabled'}")
    print(f"  Validate:  {'enabled' if validate else 'disabled'}")
    print(f"  Caveman:   {caveman_level}")
    if compact_threshold < 1.0:
        print(
            f"  Compact:   >{int(compact_threshold*100)}% ctx → {int(compact_target*100)}%"
            f" ({'stage1+2' if compact_summarize else 'stage1 only'})"
        )
    else:
        print(f"  Compact:   disabled")
    print(f"\n  API:       http://localhost:{port}/v1")
    if webui:
        print(f"  Mio UI:    http://localhost:{port}/ui")
    print(f"  Dashboard: http://localhost:{port}/dashboard")
    print(f"  Health:    http://localhost:{port}/health")

    use_live = live_panel and sys.stdout.isatty()
    if not use_live:
        # Non-TTY (piped / redirected) — keep plain serial logging via _stats.record's print.
        # Re-enable the inline print since no TUI will paint.
        _install_plain_line_printer()
        uvicorn.run(app, host=host, port=port, log_level="warning")
        return

    # Run uvicorn in a background thread, main thread drives the live panel.
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    def _run():
        try:
            import asyncio
            asyncio.run(server.serve())
        except Exception as e:
            print(f"[serve] uvicorn stopped: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for uvicorn's startup event — server.started becomes True when ready.
    import time as _t
    for _ in range(50):
        if server.started:
            break
        _t.sleep(0.05)

    try:
        from rich.live import Live
        from rich.console import Console
        console = Console()
        with Live(_render_live_panel(), console=console, refresh_per_second=4, screen=False) as live:
            while t.is_alive():
                live.update(_render_live_panel())
                _t.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[serve] shutting down...")
    finally:
        server.should_exit = True
        t.join(timeout=5.0)


def _install_plain_line_printer() -> None:
    """Replace _stats.record behaviour with a per-request stdout print for non-TTY runs."""
    original = _stats.record

    def record_with_print(gen_metrics, wall_s, tier, text):
        original(gen_metrics, wall_s, tier, text)
        r = _stats._recent[0] if _stats._recent else None
        if r is None:
            return
        avg = _stats.avg_decode_tps()
        p1 = _stats.p1_low_decode_tps()
        tool = " [tool-call]" if r["tool"] else ""
        print(
            f"[req {r['id']:>4d}] tier={r['tier']} "
            f"prompt={r['prompt']:>5d} gen={r['gen']:>4d}tok "
            f"wall={r['wall_ms']:>6.0f}ms "
            f"prefill={r['prefill_tps']:>6.0f} tok/s "
            f"decode={r['decode_tps']:>6.1f} tok/s "
            f"(avg {avg:.1f}, 1%low {p1:.1f}) "
            f"accept={r['accept']:.2f}{tool} | {r['snippet']!r}",
            flush=True,
        )

    _stats.record = record_with_print  # type: ignore
