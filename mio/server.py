"""FastAPI OpenAI-compatible API server."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from mio.agent_policy import AgentToolPermission, AgentToolPolicy
from mio.model_manager import ModelManager
from mio.mcp import (
    MCPError,
    MCPPermission,
    MCPPermissionError,
    MCPProtocolError,
    MCPRegistry,
    MCPServerConfig,
    MCPTransport,
    load_registry,
)
from mio.paths import mio_home
from mio.prompt_policy import PromptMode, PromptPolicy, apply_prompt_policy
from mio.router import TandemRouter
from mio.web_security import (
    WebSecurityMiddleware,
    configured_cors_origins,
    configure_runtime_web_security,
    runtime_web_origins,
)

# Serializes MLX work across concurrent HTTP requests. Without this two
# producer threads can encode Metal command buffers simultaneously and
# Apple's driver aborts with "command encoder is already encoding".
_GPU_LOCK = threading.Lock()

# Global state — set by start_server()
_manager: ModelManager | None = None
_router: TandemRouter | None = None
_tandem_enabled: bool = False
_validate_enabled: bool = False
_caveman_level: str = "full"
_prompt_policy = PromptPolicy()
_mcp_registry: MCPRegistry | None = None
# Health probes are deliberately stricter than ordinary MCP calls. The UI must
# never turn an unusually large registry or a hung provider into an unbounded
# request, and health responses never echo provider configuration or errors.
_MCP_HEALTH_MAX_SERVERS = 32
_MCP_HEALTH_CONCURRENCY = 8
_MCP_HEALTH_TIMEOUT_S = 3.0
_MCP_HEALTH_CLOSE_TIMEOUT_S = 0.5
_MCP_HEALTH_MAX_OUTPUT_BYTES = 256 * 1024
# One health batch is shared by every concurrent request on the serving event
# loop.  The lock also makes sequential TestClient/event-loop lifecycles safe:
# an asyncio Task is never awaited from a loop other than the one that owns it.
_MCP_HEALTH_FLIGHTS_LOCK = threading.Lock()
_MCP_HEALTH_FLIGHTS: dict[
    asyncio.AbstractEventLoop,
    asyncio.Task[dict[str, Any]],
] = {}
# Context auto-compaction thresholds (set by start_server)
_compact_threshold: float = 0.75
_compact_target: float = 0.50
_compact_enabled: bool = True
_compact_summarize: bool = True
_webui_enabled: bool = False
_webui_router_mounted: bool = False
_webui_cors_middleware_added: bool = False

# Each SSE stream or non-streaming REST attempt owns one synchronous producer
# thread. Track them so app shutdown can signal in-flight generation and wait
# briefly for cooperative cleanup instead of abandoning queued Metal work.
_STREAM_QUEUE_MAXSIZE = 16
_STREAM_PRODUCERS_LOCK = threading.Lock()
_STREAM_PRODUCERS: dict[threading.Thread, threading.Event] = {}


async def _join_generation_worker(thread: threading.Thread) -> None:
    """Wait for a cooperatively-cancelled worker without blocking the loop."""
    if thread.is_alive():
        await asyncio.to_thread(thread.join)


async def _run_rest_generation(
    *,
    manager: ModelManager,
    tier_name: str,
    messages: list[dict],
    request: Request | None,
    worker_name: str,
    max_tokens: int | None,
    temperature: float | None,
    stop: list[str] | None,
    tools: list[dict] | None,
    tool_required: bool,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
) -> tuple[str, Any]:
    """Run one REST completion with cooperative disconnect cancellation.

    ``asyncio.to_thread`` cannot cancel work that is already queued on a
    ``threading.Lock``.  A cancelled request would therefore wake up later and
    generate an answer nobody can consume.  This worker mirrors the SSE
    lifecycle instead: it polls the GPU lock and re-resolves the engine only
    after acquiring it.  A disconnect can cancel while queued; once a backend
    non-streaming call begins, cleanup waits for that synchronous call to end.

    Non-streaming requests use the engine's non-streaming entry point so mode
    selection (including BMP) remains identical to direct engine use.  The
    streaming fallback is retained for lightweight third-party/test engines
    that do not implement ``generate``.
    """

    cancelled = threading.Event()
    finished = threading.Event()
    outcome: dict[str, Any] = {}

    def _worker() -> None:
        acquired_gpu = False
        source = None
        try:
            while not cancelled.is_set():
                if _GPU_LOCK.acquire(timeout=0.1):
                    acquired_gpu = True
                    break
            if not acquired_gpu or cancelled.is_set():
                return

            active_engine = manager.get_engine(tier_name)
            generate_method = getattr(active_engine, "generate", None)
            if callable(generate_method):
                outcome["value"] = generate_method(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    tools=tools,
                    tool_required=tool_required,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
            else:
                stream_method = getattr(active_engine, "generate_stream", None)
                if not callable(stream_method):
                    raise RuntimeError("engine implements neither generate nor generate_stream")
                source = stream_method(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    tools=tools,
                    tool_required=tool_required,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
                chunks: list[str] = []
                final_metrics = None
                iterator = iter(source)
                while not cancelled.is_set():
                    try:
                        chunk, metrics = next(iterator)
                    except StopIteration:
                        break
                    if cancelled.is_set():
                        break
                    if chunk:
                        chunks.append(chunk)
                    if metrics is not None:
                        final_metrics = metrics
                if cancelled.is_set():
                    return
                if final_metrics is None:
                    final_metrics = getattr(active_engine, "last_metrics", None)
                if final_metrics is None:
                    raise RuntimeError("generation stream completed without metrics")
                outcome["value"] = ("".join(chunks), final_metrics)
        except BaseException as exc:
            if not cancelled.is_set():
                outcome["error"] = exc
        finally:
            if source is not None:
                close = getattr(source, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            if acquired_gpu:
                _GPU_LOCK.release()
            finished.set()
            _unregister_stream_producer(threading.current_thread())

    thread = threading.Thread(
        target=_worker,
        name=f"mio-rest-{worker_name}",
        daemon=True,
    )
    _register_stream_producer(thread, cancelled)
    try:
        thread.start()
    except BaseException:
        _unregister_stream_producer(thread)
        raise

    try:
        while not finished.is_set():
            await asyncio.sleep(0.05)
            if finished.is_set():
                break
            if request is not None and await request.is_disconnected():
                cancelled.set()
                await _join_generation_worker(thread)
                raise HTTPException(status_code=499, detail="Client closed request")

        error = outcome.get("error")
        if error is not None:
            raise error
        if "value" not in outcome:
            raise RuntimeError("generation cancelled")
        return outcome["value"]
    except BaseException:
        cancelled.set()
        await _join_generation_worker(thread)
        raise


async def _run_rest_compaction(
    *,
    manager: ModelManager,
    tier_name: str,
    messages: list[dict],
    request: Request | None,
    tools: list[dict] | None,
    threshold: float,
    target: float,
    enable_summarization: bool,
) -> tuple[list[dict], Any]:
    """Compact under the lifecycle lock without leaving abandoned work."""
    from mio.compactor import compact

    cancelled = threading.Event()
    finished = threading.Event()
    outcome: dict[str, Any] = {}

    def _worker() -> None:
        acquired_gpu = False
        try:
            while not cancelled.is_set():
                if _GPU_LOCK.acquire(timeout=0.1):
                    acquired_gpu = True
                    break
            if not acquired_gpu or cancelled.is_set():
                return
            active_engine = manager.get_engine(tier_name)
            value = compact(
                messages,
                active_engine,
                tools=tools,
                threshold=threshold,
                target=target,
                enable_summarization=enable_summarization,
                gpu_lock=None,
                cancellation_event=cancelled,
            )
            if not cancelled.is_set():
                outcome["value"] = value
        except BaseException as exc:
            if not cancelled.is_set():
                outcome["error"] = exc
        finally:
            if acquired_gpu:
                _GPU_LOCK.release()
            finished.set()
            _unregister_stream_producer(threading.current_thread())

    thread = threading.Thread(
        target=_worker,
        name=f"mio-rest-compact-{uuid.uuid4().hex[:8]}",
        daemon=True,
    )
    _register_stream_producer(thread, cancelled)
    try:
        thread.start()
    except BaseException:
        _unregister_stream_producer(thread)
        raise

    try:
        while not finished.is_set():
            await asyncio.sleep(0.05)
            if finished.is_set():
                break
            if request is not None and await request.is_disconnected():
                cancelled.set()
                await _join_generation_worker(thread)
                raise HTTPException(status_code=499, detail="Client closed request")
        error = outcome.get("error")
        if error is not None:
            raise error
        if "value" not in outcome:
            raise RuntimeError("compaction cancelled")
        return outcome["value"]
    except BaseException:
        cancelled.set()
        await _join_generation_worker(thread)
        raise


def _register_stream_producer(
    thread: threading.Thread,
    cancelled: threading.Event,
) -> None:
    with _STREAM_PRODUCERS_LOCK:
        _STREAM_PRODUCERS[thread] = cancelled


def _unregister_stream_producer(thread: threading.Thread) -> None:
    with _STREAM_PRODUCERS_LOCK:
        _STREAM_PRODUCERS.pop(thread, None)


def _cancel_stream_producers(join_timeout: float = 1.0) -> None:
    """Cooperatively stop active generation workers and wait a bounded time."""
    with _STREAM_PRODUCERS_LOCK:
        active = list(_STREAM_PRODUCERS.items())
    for _thread, cancelled in active:
        cancelled.set()

    deadline = time.monotonic() + max(0.0, join_timeout)
    current = threading.current_thread()
    for thread, _cancelled in active:
        if thread is current or not thread.is_alive():
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)


@asynccontextmanager
async def _lifespan(_application: FastAPI):
    """Own background services for each FastAPI run.

    ``app`` is process-global, so both startup and shutdown deliberately remain
    repeatable (including sequential TestClient contexts).
    """
    from mio.mcp import close_default_hub, configure_default_hub
    from mio.webui import scheduler

    try:
        if _mcp_registry is not None:
            configure_default_hub(_mcp_registry)
        if _webui_enabled and _manager is not None:
            scheduler.init(_manager, gpu_lock=_GPU_LOCK)
        yield
    finally:
        # Stop the producer before tearing down tools it could otherwise call.
        try:
            await _cancel_mcp_health_flight()
            await asyncio.to_thread(_cancel_stream_producers)
            await scheduler.shutdown()
        finally:
            await asyncio.to_thread(close_default_hub)


app = FastAPI(title="Mio", version="0.1.0", lifespan=_lifespan)
app.add_middleware(WebSecurityMiddleware, webui_enabled=lambda: _webui_enabled)


def _cors_origins(port: int) -> list[str]:
    """Return explicit CORS origins; wildcard is never an implicit default."""
    configured = os.environ.get("MIO_CORS_ORIGINS")
    if configured is not None:
        return configured_cors_origins()
    origins = [
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        f"http://[::1]:{port}",
    ]
    return list(dict.fromkeys([*origins, *runtime_web_origins()]))


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
        # The standalone dashboard is fed from the same completion boundary as
        # the serve console.  The dashboard collector owns the cross-thread
        # handoff, so this remains safe when ``record`` is invoked by a worker.
        try:
            from mio.dashboard import record_generation

            record_generation(gen_metrics, wall_s=wall_s, tier=tier)
        except Exception:
            # Telemetry must never turn a successful inference into an error.
            pass

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
        f"[dim]prompt[/dim] [green]{_prompt_policy.label}[/green]"
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
    "files use write/edit tools; for validation use validate when provided, and "
    "use bash only for other shell commands. Empty prose without "
    "a tool call is always incorrect when tools are available."
)


def _apply_policy(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_requirement: str | None = None,
) -> list[dict]:
    """Apply prompt policy and only force tools for required/named choices."""
    out = apply_prompt_policy(messages, _prompt_policy, skip_system_markers=_TOOL_CLIENT_MARKERS)

    # OpenAI's default/"auto" choice must leave the model free to answer in
    # prose.  A reminder is therefore added only for explicit required/named
    # choices.  For a named choice the native template receives just that tool.
    if tools and tool_requirement and out and out[-1].get("role") == "user":
        last = dict(out[-1])
        last_content = last.get("content", "") or ""
        reminder = _TOOL_ACTION_REMINDER
        if tool_requirement != "required":
            reminder += f" You must call the function named {tool_requirement!r}."
        last["content"] = last_content.rstrip() + reminder
        out = [*out[:-1], last]

    return out


def _apply_caveman(messages: list[dict], tools: list[dict] | None = None) -> list[dict]:
    """Compatibility wrapper for integrations importing the historical helper."""
    return _apply_policy(messages, tools=tools)


# --- Request/Response Models ---


class ChatMessage(BaseModel):
    # Accept both plain strings (legacy) and OpenAI's multimodal content-parts
    # list (used by Cline, many SDKs, etc). We normalize to a plain string at
    # request time — mio is text-only for now.
    model_config = {"extra": "allow"}  # tolerate extra fields (name, tool_call_id, etc.)
    role: str
    content: str | list[dict] | None = None


class FunctionDefinition(BaseModel):
    """OpenAI-compatible function declaration passed to the chat template."""

    model_config = {"extra": "allow"}
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class ChatCompletionTool(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedToolChoiceFunction(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class NamedToolChoice(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["function"] = "function"
    function: NamedToolChoiceFunction


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
    model_config = {"extra": "allow", "populate_by_name": True}
    model: str = "mio-large"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stream: bool = False
    stop: list[str] | None = None
    tools: list[ChatCompletionTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    validate_output: bool = Field(
        default=False,
        alias="validate",
        description="Mio extension: auto-validate generated code",
    )

    @field_validator("stop", mode="before")
    @classmethod
    def _normalize_stop(cls, value):
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not 1 <= len(value) <= 4:
            raise ValueError("stop must contain between 1 and 4 strings")
        if any(not item for item in value):
            raise ValueError("stop strings must not be empty")
        return value

    @model_validator(mode="after")
    def _coalesce_completion_limit(self):
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens must match")
        if self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self


def _resolve_request_tools(
    request: ChatCompletionRequest,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return native-template tools and an optional required/named constraint."""

    tools = [tool.model_dump(exclude_none=True) for tool in (request.tools or [])]
    choice = request.tool_choice
    if not tools:
        if choice == "required" or isinstance(choice, NamedToolChoice):
            raise HTTPException(400, "tool_choice requires at least one tool")
        return None, None
    if choice == "none":
        return None, None
    if choice is None or choice == "auto":
        return tools, None
    if choice == "required":
        return tools, "required"

    selected = choice.function.name
    matching = [tool for tool in tools if tool["function"]["name"] == selected]
    if not matching:
        raise HTTPException(400, f"tool_choice names an undeclared function: {selected}")
    return matching, selected


def _completion_finish_reason(
    metrics: Any,
    max_tokens: int | None,
    engine: Any,
) -> Literal["stop", "length"]:
    """Infer the OpenAI length reason from the backend's token accounting."""

    limit = max_tokens
    if limit is None:
        limit = getattr(getattr(engine, "tier_config", None), "max_output_tokens", None)
    completion_tokens = int(getattr(metrics, "completion_tokens", 0) or 0)
    return "length" if limit is not None and completion_tokens >= int(limit) else "stop"


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
    manager = _manager

    def _load() -> None:
        with _GPU_LOCK:
            manager.load_tier(req.tier)

    try:
        await asyncio.to_thread(_load)
        _refresh_tandem_router()
        return {"status": "loaded", "tier": req.tier}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/v1/models/unload")
async def unload_model(req: TierLoadRequest) -> dict:
    if not _manager:
        raise HTTPException(500, "Server not initialized")
    manager = _manager

    def _unload() -> None:
        with _GPU_LOCK:
            manager.unload_tier(req.tier)

    await asyncio.to_thread(_unload)
    _refresh_tandem_router()
    return {"status": "unloaded", "tier": req.tier}


@app.get("/metrics")
async def metrics() -> MetricsResponse:
    if not _manager:
        return MetricsResponse(tiers={})
    status = _manager.status()
    return MetricsResponse(tiers=status.get("engines", {}))


@app.get("/v1/mcp/servers")
async def list_mcp_servers() -> dict:
    """List declarations only; this endpoint never launches or calls an MCP."""
    if _mcp_registry is None:
        return {"object": "list", "data": []}
    data = []
    for config in _mcp_registry.list():
        item = config.as_dict()
        if "environment" in item:
            item["environment"] = {key: "<redacted>" for key in item["environment"]}
        data.append(item)
    return {"object": "list", "data": data}


def _mcp_health_item(config: MCPServerConfig) -> dict[str, Any]:
    """Return the fixed, non-sensitive portion of an MCP health record."""
    return {
        "name": config.name,
        "transport": config.transport.value,
        "enabled": bool(config.enabled),
        "local": config.is_local,
        "authenticated": config.uses_auth,
    }


def _mcp_health_failure(exc: BaseException) -> tuple[str, str]:
    """Map provider failures to stable codes without returning exception text."""
    if isinstance(exc, TimeoutError):
        return "timeout", "probe_timeout"
    if isinstance(exc, MCPPermissionError):
        return "unavailable", "permission_denied"
    if isinstance(exc, MCPProtocolError):
        return "unavailable", "protocol_error"
    if isinstance(exc, (MCPError, OSError)):
        return "unavailable", "provider_unavailable"
    return "unavailable", "probe_failed"


def _mcp_health_agent_policy(config: MCPServerConfig) -> AgentToolPolicy:
    """Build a least-authority stdio policy for provider discovery only."""

    runtime = mio_home()
    try:
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        canonical_runtime = runtime.resolve(strict=True)
        health_root = runtime / "mcp-health"
        prospective_root = health_root.resolve(strict=False)
        prospective_root.relative_to(canonical_runtime)
        if prospective_root == canonical_runtime:
            raise ValueError("health workspace must be a strict Mio child")
        health_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        canonical_health_root = health_root.resolve(strict=True)
        canonical_health_root.relative_to(canonical_runtime)
        canonical_health_root.chmod(0o700)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MCPPermissionError("cannot prepare isolated MCP health workspace") from exc

    permission_map: dict[MCPPermission, AgentToolPermission | None] = {
        MCPPermission.READ: AgentToolPermission.READ,
        MCPPermission.FILESYSTEM_READ: AgentToolPermission.READ,
        MCPPermission.WRITE: AgentToolPermission.WRITE,
        MCPPermission.FILESYSTEM_WRITE: AgentToolPermission.WRITE,
        MCPPermission.PROCESS: AgentToolPermission.SHELL,
        MCPPermission.NETWORK: AgentToolPermission.NETWORK,
        MCPPermission.SECRETS: None,
    }
    permissions: set[AgentToolPermission] = set()
    for permission in config.permissions:
        mapped = permission_map[permission]
        if mapped is None:
            raise MCPPermissionError("MCP health never receives credential authority")
        permissions.add(mapped)
    return AgentToolPolicy(
        workspace_roots=(Path(canonical_health_root),),
        permissions=frozenset(permissions),
        output_limit_chars=10_000,
        file_limit_chars=1_048_576,
        command_timeout_s=min(config.timeout_s, max(0.01, _MCP_HEALTH_TIMEOUT_S)),
    )


def _observe_mcp_health_task(task: asyncio.Task[Any]) -> None:
    """Consume a detached cleanup/flight result so it cannot warn at exit."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _close_mcp_health_provider(provider: Any) -> None:
    """Close a probe provider without releasing its concurrency slot early.

    ``StdioProvider.close`` owns the bounded TERM/KILL sequence and deliberately
    delays cancellation until the child has been reaped. ``wait_for`` may thus
    exceed this nominal timeout for that bounded termination and final reap;
    keeping the await here prevents later probes from exceeding the process
    concurrency bound.
    """
    timeout = max(0.001, float(_MCP_HEALTH_CLOSE_TIMEOUT_S))
    try:
        await asyncio.wait_for(provider.close(), timeout)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


async def _probe_mcp_health(
    config: MCPServerConfig,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Initialize one safe provider briefly and return a redacted health row."""
    item = _mcp_health_item(config)
    if not config.enabled:
        return {**item, "status": "disabled"}
    if not config.is_local:
        return {**item, "status": "skipped", "reason": "remote_not_probed"}
    if config.uses_auth:
        return {**item, "status": "skipped", "reason": "credentials_not_probed"}
    if config.transport is not MCPTransport.STDIO:
        return {**item, "status": "skipped", "reason": "transport_not_isolated"}

    async with semaphore:
        started = time.monotonic()
        provider = None
        try:
            # Apply the health budget inside the transport as well. In
            # particular, this bounds urllib's socket timeout even if the
            # outer coroutine is cancelled while its worker thread is active.
            transport_timeout = min(
                config.timeout_s,
                max(0.01, _MCP_HEALTH_TIMEOUT_S / 2),
            )
            # Discovery is observational even when the provider advertises
            # mutating or networked tools. Do not hand those capabilities to
            # startup code merely because an operator requested health.
            probe_permissions = config.permissions & frozenset(
                {
                    MCPPermission.READ,
                    MCPPermission.FILESYSTEM_READ,
                }
            )
            probe_config = replace(
                config,
                timeout_s=transport_timeout,
                max_output_bytes=min(config.max_output_bytes, _MCP_HEALTH_MAX_OUTPUT_BYTES),
                permissions=probe_permissions,
            )
            probe_registry = MCPRegistry([probe_config])
            provider = probe_registry.create_provider(
                probe_config.name,
                granted_permissions=probe_config.permissions,
                agent_policy=_mcp_health_agent_policy(probe_config),
            )

            async def inspect_provider() -> Any:
                await provider.initialize()
                return await provider.list_tools()

            payload = await asyncio.wait_for(inspect_provider(), _MCP_HEALTH_TIMEOUT_S)
            tools = payload.get("tools") if isinstance(payload, dict) else None
            if not isinstance(tools, list):
                raise MCPProtocolError("MCP tools/list returned an invalid payload")
            return {
                **item,
                "status": "ready",
                "tool_count": len(tools),
                "latency_ms": max(0, round((time.monotonic() - started) * 1000)),
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Every ordinary provider failure is reduced to a stable code.
            status, reason = _mcp_health_failure(exc)
            return {
                **item,
                "status": status,
                "reason": reason,
                "latency_ms": max(0, round((time.monotonic() - started) * 1000)),
            }
        finally:
            if provider is not None:
                await _close_mcp_health_provider(provider)


async def _collect_mcp_health() -> dict[str, Any]:
    """Build one bounded health snapshot for all concurrent HTTP callers."""
    if _mcp_registry is None:
        return {
            "object": "mcp.health",
            "status": "empty",
            "summary": {"configured": 0, "reported": 0, "omitted": 0},
            "data": [],
        }

    registry = _mcp_registry
    configs = registry.list()
    selected = configs[:_MCP_HEALTH_MAX_SERVERS]
    semaphore = asyncio.Semaphore(max(1, int(_MCP_HEALTH_CONCURRENCY)))
    data = await asyncio.gather(
        *(_probe_mcp_health(config, semaphore) for config in selected)
    )
    counts = {
        status: sum(item["status"] == status for item in data)
        for status in ("ready", "unavailable", "timeout", "disabled", "skipped")
    }
    omitted = max(0, len(configs) - len(selected))
    if not data:
        overall = "empty"
    elif counts["unavailable"] or counts["timeout"] or omitted:
        overall = "degraded"
    elif counts["ready"]:
        overall = "ready"
    else:
        overall = "idle"
    return {
        "object": "mcp.health",
        "status": overall,
        "summary": {
            "configured": len(configs),
            "reported": len(data),
            "omitted": omitted,
            **counts,
        },
        "probe": {
            "timeout_ms": round(_MCP_HEALTH_TIMEOUT_S * 1000),
            "max_servers": _MCP_HEALTH_MAX_SERVERS,
        },
        "data": data,
    }


def _finish_mcp_health_flight(
    loop: asyncio.AbstractEventLoop,
    task: asyncio.Task[dict[str, Any]],
) -> None:
    """Forget a completed batch without racing a newly created replacement."""
    with _MCP_HEALTH_FLIGHTS_LOCK:
        if _MCP_HEALTH_FLIGHTS.get(loop) is task:
            _MCP_HEALTH_FLIGHTS.pop(loop, None)
    _observe_mcp_health_task(task)


def _mcp_health_flight() -> asyncio.Task[dict[str, Any]]:
    """Return the serving loop's existing probe batch or atomically start it."""
    loop = asyncio.get_running_loop()
    with _MCP_HEALTH_FLIGHTS_LOCK:
        task = _MCP_HEALTH_FLIGHTS.get(loop)
        if task is None or task.done():
            task = loop.create_task(_collect_mcp_health(), name="mio-mcp-health")
            _MCP_HEALTH_FLIGHTS[loop] = task
            task.add_done_callback(
                lambda completed, owner=loop: _finish_mcp_health_flight(owner, completed)
            )
        return task


async def _cancel_mcp_health_flight() -> None:
    """Cancel and join the current loop's batch during application shutdown."""
    loop = asyncio.get_running_loop()
    with _MCP_HEALTH_FLIGHTS_LOCK:
        task = _MCP_HEALTH_FLIGHTS.pop(loop, None)
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@app.post("/v1/mcp/health")
async def mcp_health() -> dict[str, Any]:
    """Return a shared, redacted and resource-bounded MCP health snapshot.

    Remote and credential-bearing providers are never contacted. Concurrent
    callers on the server loop share one probe batch, and cancelling one HTTP
    request cannot cancel work still awaited by other callers. The POST route
    is intentionally protected by the WebUI CSRF session because a probe can
    launch local provider processes; declarations remain available through the
    side-effect-free GET endpoint.
    """
    return await asyncio.shield(_mcp_health_flight())


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request = None,
) -> Any:
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

    tools, tool_requirement = _resolve_request_tools(req)
    supported_fields = {
        "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
        "seed", "stream", "stop", "tools", "tool_choice", "validate_output",
    }
    extra = {key: value for key, value in req.model_dump().items() if key not in supported_fields}

    _debug_log("request_raw", {
        "model": req.model, "stream": req.stream,
        "max_tokens": req.max_tokens, "temperature": req.temperature,
        "top_p": req.top_p, "top_k": req.top_k, "seed": req.seed,
        "stop": req.stop,
        "tool_choice": req.tool_choice,
        "messages": messages,
        "extra_fields": extra,
    })

    # Route to tier first — compaction needs the engine's tokenizer + context window.
    tier_name = _resolve_tier(req.model, messages)
    manager = _manager
    try:
        engine = manager.get_engine(tier_name)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    # Context auto-compaction: when prompt tokens exceed `_compact_threshold`
    # of the context window, reduce to `_compact_target`. Prevents Metal
    # OOM crashes on long agent sessions. Runs BEFORE caveman so caveman
    # injection targets the compacted message list.
    if _compact_enabled:
        # Tokenization and optional summarization are synchronous. The worker
        # polls the lifecycle lock and streams stage-2 summarization so a
        # disconnected request cannot later wake up and generate in the
        # background.
        messages, compact_stats = await _run_rest_compaction(
            manager=manager,
            tier_name=tier_name,
            messages=messages,
            request=request,
            tools=tools,
            threshold=_compact_threshold,
            target=_compact_target,
            enable_summarization=_compact_summarize,
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

    messages = _apply_policy(messages, tools=tools, tool_requirement=tool_requirement)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = f"mio-{tier_name}"

    if req.stream:
        return StreamingResponse(
            _stream_response(
                engine, messages, completion_id, created, model_name, tier_name,
                max_tokens=req.max_tokens, temperature=req.temperature, stop=req.stop,
                tools=tools, top_p=req.top_p, top_k=req.top_k, seed=req.seed,
                tool_requirement=tool_requirement,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming (with optional validation + retry). Each attempt uses the
    # same cooperative producer lifecycle as SSE so cancellation/disconnect
    # cannot leave work queued behind the GPU lock.
    _ns_t0 = time.perf_counter()
    text, gen_metrics = await _run_rest_generation(
        manager=manager,
        tier_name=tier_name,
        messages=messages,
        request=request,
        worker_name=completion_id[-12:],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=req.stop,
        tools=tools,
        tool_required=bool(tool_requirement),
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
    )
    # Strip <think>...</think> reasoning blocks.
    _stripper = _ThinkStripper()
    text = _stripper.feed(text) + _stripper.flush()
    if req.validate_output and _validate_enabled:
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
            text, gen_metrics = await _run_rest_generation(
                manager=manager,
                tier_name=tier_name,
                messages=retry_messages,
                request=request,
                worker_name=f"{completion_id[-8:]}-retry{_retry + 1}",
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=req.stop,
                tools=tools,
                tool_required=bool(tool_requirement),
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
            )
            retry_stripper = _ThinkStripper()
            text = retry_stripper.feed(text) + retry_stripper.flush()

    # Parse the final attempt, not a stale pre-validation response.
    from mio.tool_calls import parse_tool_calls
    leading_text, parsed_calls = parse_tool_calls(text) if tools else (text, [])
    if tool_requirement and not parsed_calls:
        raise HTTPException(
            502,
            f"model did not satisfy tool_choice={tool_requirement!r}",
        )
    if tool_requirement not in (None, "required") and any(
        call.get("function", {}).get("name") != tool_requirement for call in parsed_calls
    ):
        raise HTTPException(
            502,
            f"model called a function other than required {tool_requirement!r}",
        )
    _stats.record(gen_metrics, time.perf_counter() - _ns_t0, tier_name, text)

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
        finish = _completion_finish_reason(gen_metrics, req.max_tokens, engine)
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
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    tool_requirement: str | None = None,
):
    """Yield SSE chunks in OpenAI format.

    engine.generate_stream is a synchronous generator that can block for
    minutes during long-prompt prefill. We drive it from a cooperative
    background thread and bridge it through a bounded asyncio queue.  The
    producer therefore follows client backpressure instead of buffering an
    unbounded completion when the consumer is slow or disconnected.
    """
    from concurrent.futures import CancelledError as FutureCancelledError
    from concurrent.futures import TimeoutError as FutureTimeoutError

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
    aq: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
    sentinel = object()
    cancelled = threading.Event()
    stream_t0 = time.perf_counter()
    full_text: list[str] = []

    def _put_from_thread(item: object) -> bool:
        """Put with backpressure while remaining responsive to cancellation."""
        if cancelled.is_set():
            return False
        put_coro = aq.put(item)
        try:
            future = asyncio.run_coroutine_threadsafe(put_coro, loop)
        except RuntimeError:
            # The request loop closed between the cancellation check and
            # scheduling. Explicitly close the un-awaited queue coroutine.
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

    def _producer() -> None:
        acquired_gpu = False
        source = None
        try:
            # A request may disconnect while waiting behind another generation.
            # Timed acquires let it leave without later running abandoned work.
            while not cancelled.is_set():
                if _GPU_LOCK.acquire(timeout=0.1):
                    acquired_gpu = True
                    break
            if not acquired_gpu:
                return

            # Resolve again *inside* the lifecycle lock.  Model load/unload use
            # the same GPU lock, so the selected engine stays valid for the
            # complete synchronous stream.
            active_engine = (
                _manager.get_engine(tier_name) if _manager is not None else engine
            )
            source = active_engine.generate_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                tools=tools,
                tool_required=bool(tool_requirement),
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
            for item in source:
                if cancelled.is_set() or not _put_from_thread(item):
                    break
        except Exception as exc:
            if not cancelled.is_set():
                _put_from_thread(("__error__", repr(exc)))
        finally:
            if source is not None:
                close = getattr(source, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
            if acquired_gpu:
                _GPU_LOCK.release()
            if not cancelled.is_set():
                _put_from_thread(sentinel)
            _unregister_stream_producer(threading.current_thread())

    producer_thread = threading.Thread(
        target=_producer,
        name=f"mio-sse-{completion_id[-12:]}",
        daemon=True,
    )
    _register_stream_producer(producer_thread, cancelled)
    producer_thread.start()

    final_metrics = None
    KEEPALIVE_EVERY = 10.0
    stripper = _ThinkStripper()
    # Parse Qwen <tool_call>...</tool_call> spans out of the stream when the
    # client passed `tools` (OpenAI function-calling). Tool calls are deferred
    # to the final delta; content outside tool calls streams normally.
    from mio.tool_calls import StreamingToolCallParser
    tool_parser = StreamingToolCallParser() if tools else None

    completed = False
    failed = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(aq.get(), timeout=KEEPALIVE_EVERY)
            except asyncio.TimeoutError:
                # Producer thread still working (likely prefilling). Send a
                # SSE comment to keep the HTTP connection warm for clients
                # with a body timeout (Cline, curl, etc.).
                yield ": keepalive\n\n"
                continue

            if item is sentinel:
                completed = True
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
                            "choices": [{
                                "index": 0,
                                "delta": {"content": tail_content},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                _debug_log("response_stream", {
                    "id": completion_id,
                    "model": model_name,
                    "full_text": "".join(full_text),
                    "token_count": (
                        int(getattr(final_metrics, "completion_tokens", 0))
                        if final_metrics else 0
                    ),
                })
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                failed = True
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
                        "choices": [{
                            "index": 0,
                            "delta": {"content": filtered},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        # Error/cancellation paths never emit an OpenAI success trailer.
        if failed or not completed:
            return

        # Collect tool calls parsed during the stream. When present, OpenAI
        # clients require finish_reason="tool_calls" before invoking tools.
        parsed_calls: list[dict] = []
        if tool_parser is not None:
            residual, parsed_calls = tool_parser.flush()
            if residual:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": residual},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        if tool_requirement and not parsed_calls:
            error = {
                "error": {
                    "message": f"model did not satisfy tool_choice={tool_requirement!r}",
                    "type": "tool_choice_violation",
                }
            }
            yield f"data: {json.dumps(error)}\n\n"
            return
        if tool_requirement not in (None, "required") and any(
            call.get("function", {}).get("name") != tool_requirement
            for call in parsed_calls
        ):
            error = {
                "error": {
                    "message": (
                        f"model called a function other than required {tool_requirement!r}"
                    ),
                    "type": "tool_choice_violation",
                }
            }
            yield f"data: {json.dumps(error)}\n\n"
            return

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
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [tc_delta]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        finish_reason = (
            "tool_calls"
            if parsed_calls
            else _completion_finish_reason(final_metrics, max_tokens, engine)
        )
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        }
        if final_metrics:
            final_chunk["usage"] = {
                "prompt_tokens": final_metrics.prompt_tokens,
                "completion_tokens": final_metrics.completion_tokens,
                "total_tokens": final_metrics.total_tokens,
            }
            _stats.record(
                final_metrics,
                time.perf_counter() - stream_t0,
                tier_name,
                "".join(full_text),
            )
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        cancelled.set()
        if producer_thread.is_alive():
            # ``join`` itself runs off-loop; a backend stuck inside one token
            # step may outlive this bounded wait but remains a daemon and sees
            # the cancellation event before producing the next chunk.
            await asyncio.to_thread(producer_thread.join, 1.0)


def _resolve_tier(model: str, messages: list[dict] | None = None) -> str:
    """Resolve model name to tier."""
    if not _manager:
        raise HTTPException(500, "Server not initialized")

    loaded = _manager.loaded_tiers()
    if not loaded:
        raise HTTPException(503, "No models loaded")

    if model.startswith("mio-"):
        tier = model.removeprefix("mio-")
        if tier == "auto" and _router:
            # `_router` records that tandem mode is enabled, but its tier list
            # was created at server startup. Build a request-local view so
            # model load/unload changes are reflected immediately and route on
            # the actual user content rather than an empty placeholder.
            return TandemRouter(list(loaded)).route(messages or [], model)
        if tier in loaded:
            return tier

    # Default to first loaded tier
    return loaded[0]


def _refresh_tandem_router() -> None:
    """Refresh tandem routing after a dynamic tier load or unload."""

    global _router
    loaded_tiers = getattr(_manager, "loaded_tiers", None)
    loaded = list(loaded_tiers()) if callable(loaded_tiers) else []
    _router = TandemRouter(loaded) if _tandem_enabled and loaded else None


def init_server(
    manager: ModelManager,
    tandem: bool = False,
    validate: bool = False,
    caveman_level: str = "full",
    compact_threshold: float = 0.75,
    compact_target: float = 0.50,
    compact_summarize: bool = True,
    prompt_policy: PromptPolicy | None = None,
    mcp_registry: MCPRegistry | None = None,
) -> None:
    """Initialize server with a model manager."""
    global _manager, _tandem_enabled, _validate_enabled, _caveman_level, _prompt_policy, _mcp_registry
    global _compact_threshold, _compact_target, _compact_enabled, _compact_summarize
    _manager = manager
    _tandem_enabled = bool(tandem)
    _validate_enabled = validate
    _prompt_policy = prompt_policy or PromptPolicy.resolve(caveman=caveman_level)
    _caveman_level = (
        _prompt_policy.level.value if _prompt_policy.mode is PromptMode.CAVEMAN else "off"
    )
    _compact_threshold = float(compact_threshold)
    _compact_target = float(compact_target)
    _compact_enabled = _compact_threshold < 1.0
    _compact_summarize = bool(compact_summarize)
    _mcp_registry = mcp_registry or load_registry()
    from mio.mcp import configure_default_hub

    configure_default_hub(_mcp_registry)
    _refresh_tandem_router()


# --- Dashboard ---

@app.get("/dashboard")
async def dashboard():
    from mio.dashboard import get_dashboard_html
    return get_dashboard_html()


@app.websocket("/ws/metrics")
async def ws_metrics(websocket):
    from mio.dashboard import websocket_metrics

    def manager_status() -> dict | None:
        manager = _manager
        return manager.status() if manager is not None else None

    await websocket_metrics(websocket, manager_status_provider=manager_status)


# --- Batch ---

class BatchRequestItem(BaseModel):
    messages: list[ChatMessage]
    model: str = "mio-large"
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stop: list[str] | None = None

    @field_validator("stop", mode="before")
    @classmethod
    def _normalize_stop(cls, value):
        return [value] if isinstance(value, str) else value

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not 1 <= len(value) <= 4 or any(not item for item in value):
            raise ValueError("stop must contain between 1 and 4 non-empty strings")
        return value

    @model_validator(mode="after")
    def _coalesce_completion_limit(self):
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens must match")
        if self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self


class BatchCompletionRequest(BaseModel):
    requests: list[BatchRequestItem]


@app.post("/v1/batch")
async def batch_completions(req: BatchCompletionRequest) -> list[dict]:
    if not _manager:
        raise HTTPException(500, "Server not initialized")
    if not req.requests:
        raise HTTPException(400, "batch must contain at least one request")
    if len(req.requests) > 64:
        raise HTTPException(413, "batch is limited to 64 requests")

    import asyncio

    from mio.batch import BatchRequest as EngineBatchRequest
    from mio.batch import process_batch

    # Resolve routing and prompt policy on the request loop, then run all MLX
    # work in one worker while holding the same Metal lock as normal chat.
    groups: dict[str, list[tuple[int, EngineBatchRequest]]] = {}
    results: list[dict | None] = [None] * len(req.requests)
    for index, item in enumerate(req.requests):
        try:
            messages = [
                {"role": message.role, "content": _coerce_message_content(message.content)}
                for message in item.messages
            ]
            tier_name = _resolve_tier(item.model, messages)
            groups.setdefault(tier_name, []).append(
                (
                    index,
                    EngineBatchRequest(
                        messages=_apply_policy(messages),
                        model=item.model,
                        max_tokens=item.max_tokens,
                        temperature=item.temperature,
                        top_p=item.top_p,
                        top_k=item.top_k,
                        seed=item.seed,
                        stop=item.stop,
                    ),
                )
            )
        except Exception as exc:
            results[index] = {"index": index, "error": str(exc), "backend": "error"}

    def generate_groups() -> None:
        assert _manager is not None
        with _GPU_LOCK:
            for tier_name, indexed_requests in groups.items():
                try:
                    generated = process_batch(
                        [request for _, request in indexed_requests],
                        _manager,
                        tier=tier_name,
                    )
                    for (original_index, _request), result in zip(
                        indexed_requests, generated, strict=True
                    ):
                        item = {
                            "index": original_index,
                            "text": result.text,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "generation_tps": result.generation_tps,
                            "batch_generation_tps": result.batch_generation_tps,
                            "metrics_scope": result.metrics_scope,
                            "batch_size": result.batch_size,
                            "time_s": result.time_s,
                            "backend": result.backend,
                        }
                        if result.error:
                            item["error"] = result.error
                        results[original_index] = item
                except Exception as exc:
                    for original_index, _request in indexed_requests:
                        results[original_index] = {
                            "index": original_index,
                            "error": str(exc),
                            "backend": "error",
                        }

    await asyncio.to_thread(generate_groups)
    return [
        result if result is not None else {"index": index, "error": "missing batch result"}
        for index, result in enumerate(results)
    ]


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
    """Kill processes bound to ``port`` after explicit replace opt-in."""
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


def _is_loopback_bind(host: str) -> bool:
    """Accept only unambiguous loopback names/addresses by default."""
    import ipaddress

    normalized = str(host or "").strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _probe_server_bind(host: str, port: int) -> None:
    """Fail before Uvicorn startup when the requested socket is unavailable."""
    import socket

    if not 1 <= int(port) <= 65535:
        raise RuntimeError(f"invalid server port: {port}")
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror as exc:
        raise RuntimeError(f"cannot resolve server bind host {host!r}: {exc}") from exc
    if not addresses:
        raise RuntimeError(f"cannot resolve server bind host {host!r}")
    family, socktype, proto, _canonname, sockaddr = addresses[0]
    probe = socket.socket(family, socktype, proto)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(sockaddr)
    except OSError as exc:
        raise RuntimeError(
            f"cannot bind Mio to {host}:{port}: {exc}. "
            "Choose another port or pass --replace-existing to explicitly terminate its listener."
        ) from exc
    finally:
        probe.close()


def start_server(
    manager: ModelManager,
    host: str = "127.0.0.1",
    port: int = 9090,
    tandem: bool = False,
    validate: bool = False,
    caveman_level: str = "full",
    live_panel: bool = True,
    compact_threshold: float = 0.75,
    compact_target: float = 0.50,
    compact_summarize: bool = True,
    webui: bool = False,
    prompt_policy: PromptPolicy | None = None,
    mcp_registry: MCPRegistry | None = None,
    unsafe_remote_bind: bool = False,
    replace_existing: bool = False,
) -> None:
    """Start the API server (blocking).

    If the stdout is a TTY and `live_panel=True`, renders a rich.Live panel
    showing running tok/s and recent requests. Otherwise falls back to plain
    per-request stdout logging (old behaviour).

    Remote binds and replacing an existing listener both require explicit
    opt-in because Mio serves local files/tools and has no HTTP authentication.
    """
    import sys
    import threading
    import uvicorn

    import os

    global _webui_enabled, _webui_router_mounted, _webui_cors_middleware_added

    remote_env = os.environ.get("MIO_UNSAFE_REMOTE_BIND", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    remote_allowed = bool(unsafe_remote_bind or remote_env)
    if not _is_loopback_bind(host) and not remote_allowed:
        raise RuntimeError(
            f"refusing non-loopback bind {host!r}: Mio has no built-in HTTP authentication. "
            "Pass --unsafe-remote-bind (or MIO_UNSAFE_REMOTE_BIND=1) only behind a trusted "
            "firewall/proxy. Concrete binds trust that Host:port; wildcard binds trust private "
            "numeric LAN addresses on this port. Add named hosts with MIO_TRUSTED_HOSTS."
        )
    if not _is_loopback_bind(host):
        print("[serve] WARNING: unsafe remote bind enabled; Mio endpoints are unauthenticated")

    if replace_existing:
        _kill_port_holders(port)
    _probe_server_bind(host, port)
    configure_runtime_web_security(host, port, allow_remote=remote_allowed)
    cors_requested = bool(webui or os.environ.get("MIO_CORS_ORIGINS") is not None)
    try:
        cors_origins = _cors_origins(port) if cors_requested else []
    except ValueError as exc:
        raise RuntimeError(f"invalid MIO_CORS_ORIGINS: {exc}") from exc

    active_prompt_policy = prompt_policy or PromptPolicy.resolve(caveman=caveman_level)
    _webui_enabled = bool(webui)
    init_server(
        manager, tandem, validate, caveman_level,
        compact_threshold=compact_threshold,
        compact_target=compact_target,
        compact_summarize=compact_summarize,
        prompt_policy=active_prompt_policy,
        mcp_registry=mcp_registry,
    )

    # Mount Mio UI if enabled
    if webui:
        from mio.webui.router import router as webui_router, mount_webui
        mount_webui(
            manager,
            caveman_level=_caveman_level,
            prompt_policy=active_prompt_policy,
            gpu_lock=_GPU_LOCK,
        )
        if not _webui_router_mounted:
            app.include_router(webui_router)
            _webui_router_mounted = True

    # Explicit API CORS is useful without the Mio UI (for example an operator-
    # controlled browser client on another origin). Keep it opt-in and clear a
    # prior embedded server's allowlist when the capability is later disabled.
    if cors_requested or _webui_cors_middleware_added:
        from starlette.middleware.cors import CORSMiddleware

        if cors_requested and not _webui_cors_middleware_added:
            # A previous TestClient may already have built the process-global
            # middleware stack. start_server runs before the next server, so
            # invalidating that cached stack here is safe and makes mounting
            # repeatable in tests and embedded launchers.
            app.middleware_stack = None
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            _webui_cors_middleware_added = True
        else:
            # Keep origins correct if an embedded launcher reuses the global
            # application on a different port or disables CORS entirely.
            for middleware in app.user_middleware:
                if middleware.cls is CORSMiddleware:
                    middleware.kwargs["allow_origins"] = cors_origins
                    app.middleware_stack = None
                    break

    print(f"\nMio API server starting on http://{host}:{port}")
    print(f"  Models:    {', '.join(manager.get_model_names())}")
    print(f"  Tandem:    {'enabled' if tandem else 'disabled'}")
    print(f"  Validate:  {'enabled' if validate else 'disabled'}")
    print(f"  Prompt:    {active_prompt_policy.label}")
    enabled_mcp = [config.name for config in (_mcp_registry.list() if _mcp_registry else []) if config.enabled]
    print(f"  MCP:       {', '.join(enabled_mcp) if enabled_mcp else 'none'}")
    if compact_threshold < 1.0:
        print(
            f"  Compact:   >{int(compact_threshold*100)}% ctx → {int(compact_target*100)}%"
            f" ({'stage1+2' if compact_summarize else 'stage1 only'})"
        )
    else:
        print("  Compact:   disabled")
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
