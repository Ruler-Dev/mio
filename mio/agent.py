"""Standalone Mio coding agent: interactive LLM with tools and slash commands."""

from __future__ import annotations

import hashlib
import math
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mio.config import CODING_EFFORT_LEVELS, MioConfig
from mio.engine import MioEngine
from mio.model_manager import ModelManager
from mio.agent_policy import (
    AgentAuditEvent,
    AgentPathViolation,
    AgentPermissionDenied,
    AgentToolPermission,
    AgentToolPolicy,
    cap_tool_output,
    is_broad_workspace_root,
    read_workspace_text,
    resolve_workspace_path,
    sandboxed_command,
    write_workspace_text,
)
from mio.prompt_policy import PromptMode, PromptPolicy, apply_prompt_policy

console = Console()
_MAX_TOOL_CALLS_PER_TURN = 32
_MAX_TOOL_RESULT_CHARS_PER_TURN = 100_000
_MAX_AGENT_ROUNDS_PER_TURN = 12
_FINALIZE_TOOL_LOOP = (
    "Tool execution must stop now ({reason}). Do not call another tool. "
    "Give a concise, evidence-based status with files changed, latest validation "
    "or test outcome, and any unfinished work or limitation."
)


@dataclass(frozen=True)
class AgentRoundTrace:
    """Content-free generation metrics for one model round."""

    round_index: int
    prompt_tokens: int
    completion_tokens: int
    total_time_s: float
    prompt_tps: float
    generation_tps: float
    generation_backend: str
    fallback_ar: bool


@dataclass(frozen=True)
class AgentToolTrace:
    """Sanitized evidence for one dispatcher/audit event."""

    round_index: int
    tool_name: str
    operation: str
    permission: str
    allowed: bool
    outcome: str
    target_sha256: str


@dataclass(frozen=True)
class AgentTurnResult:
    """Structured turn result for benchmarks and non-console callers.

    ``assistant_text`` remains available to interactive callers, but benchmark
    artifact serializers must omit it.  All trace fields are deliberately
    content-free and contain neither tool arguments nor tool output.
    """

    assistant_text: str
    terminal_reason: str
    rounds: tuple[AgentRoundTrace, ...]
    tool_events: tuple[AgentToolTrace, ...]
    tool_calls: int
    tool_result_chars: int
    wall_time_s: float
    quality_gate: dict[str, object] | None = None


def _round_trace(round_index: int, metrics: object) -> AgentRoundTrace:
    return AgentRoundTrace(
        round_index=round_index,
        prompt_tokens=int(getattr(metrics, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(metrics, "completion_tokens", 0) or 0),
        total_time_s=float(getattr(metrics, "total_time_s", 0.0) or 0.0),
        prompt_tps=float(getattr(metrics, "prompt_tps", 0.0) or 0.0),
        generation_tps=float(getattr(metrics, "generation_tps", 0.0) or 0.0),
        generation_backend=str(getattr(metrics, "generation_backend", "unknown")),
        fallback_ar=bool(getattr(metrics, "fallback_ar", False)),
    )


def _tool_trace(
    round_index: int,
    tool_name: str,
    event: AgentAuditEvent,
) -> AgentToolTrace:
    target_digest = hashlib.sha256(
        event.target.encode("utf-8", errors="replace")
    ).hexdigest()
    return AgentToolTrace(
        round_index=round_index,
        tool_name=tool_name[:64],
        operation=event.operation,
        permission=event.permission,
        allowed=event.allowed,
        outcome=event.outcome,
        target_sha256=target_digest,
    )

# --- System Prompts ---

AGENT_SYSTEM_PROMPT = """You are Mio, a fast local coding agent running on Apple Silicon.
You have access to local coding, Mio skill-catalog, and permission-gated Mio MCP tools.
When the user asks you to write or modify code, do it directly. Be precise and concise.
If you encounter an error, fix it and retry.
Use tools without narrating obvious steps or echoing tool-call XML. Do not repeat an
identical tool call unless its inputs or relevant state changed, or the previous
result explains why a retry can succeed. After edits, run the narrowest relevant
validation. Finish with changed files, observed test/command outcome, and any real
limitation; do not paste full code or command output unless the user asks.
File tools are confined to declared workspace roots. Write, edit, and shell
tools work only when the trusted caller explicitly granted their capability."""

CAVEMAN_ULTRA = """
COMMUNICATION MODE: ULTRA TERSE.
Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging.
Abbreviate: DB/auth/config/req/res/fn/impl. Arrows for causality (X -> Y).
Pattern: [thing] [action] [reason]. [next step].
Code blocks and commits always written normally.
One word when one word enough."""

CAVEMAN_FULL = """
COMMUNICATION MODE: TERSE.
Drop articles, fragments OK, short synonyms. Technical terms exact.
Code blocks unchanged."""

CAVEMAN_LITE = """
COMMUNICATION MODE: CONCISE.
No filler or hedging. Keep articles and full sentences. Professional but tight."""


CAVEMAN_LEVELS = {
    "ultra": CAVEMAN_ULTRA,
    "full": CAVEMAN_FULL,
    "lite": CAVEMAN_LITE,
    "off": "",
}


# --- Tool Execution ---

def _default_read_policy() -> AgentToolPolicy:
    """Compatibility boundary for direct, unprivileged read-tool callers."""

    workspace = Path.cwd().resolve()
    if is_broad_workspace_root(workspace):
        return AgentToolPolicy(
            workspace_roots=(workspace,),
            permissions=frozenset(),
        )
    return AgentToolPolicy.read_only(workspace)


def _policy_or_read_only(policy: AgentToolPolicy | None) -> AgentToolPolicy:
    return policy if policy is not None else _default_read_policy()


def _audit_target_for_command(command: str, argv: list[str] | None = None) -> str:
    executable = Path(argv[0]).name if argv else "invalid"
    digest = hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()[:32]
    return f"{executable} sha256:{digest}"


def _capped(policy: AgentToolPolicy, text: str) -> str:
    return cap_tool_output(text, policy.output_limit_chars)


@dataclass(frozen=True)
class _BoundedCommandResult:
    output: str
    returncode: int
    timed_out: bool = False
    output_exceeded: bool = False


def _terminate_process_group(process: subprocess.Popen, *, grace_s: float = 0.05) -> None:
    """Terminate the complete session created for one model-selected command."""

    group_existed = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        group_existed = True
    except ProcessLookupError:
        pass
    except OSError:
        pass
    if group_existed and grace_s > 0:
        time.sleep(grace_s)
    if group_existed:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        process.wait(timeout=1.0)


def _run_bounded_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    output_limit_chars: int,
) -> _BoundedCommandResult:
    """Run a command with bounded memory, input, lifetime, and descendants."""

    max_output_bytes = max(128, output_limit_chars * 4)
    process = subprocess.Popen(
        argv,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
        start_new_session=True,
        close_fds=True,
        bufsize=0,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        _terminate_process_group(process)
        raise RuntimeError("command output pipe was not created")

    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    captured = bytearray()
    timed_out = False
    output_exceeded = False
    deadline = time.monotonic() + timeout_s

    def drain_available() -> bool:
        """Drain available bytes; return False after EOF."""

        nonlocal output_exceeded
        while True:
            try:
                chunk = os.read(descriptor, 65_536)
            except BlockingIOError:
                return True
            if not chunk:
                return False
            remaining = max_output_bytes - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > remaining:
                output_exceeded = True
                return True
            if len(chunk) < 65_536:
                return True

    try:
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                timed_out = True
                break
            events = selector.select(timeout=min(0.05, remaining_s))
            for _key, _mask in events:
                if not drain_available():
                    try:
                        selector.unregister(descriptor)
                    except KeyError:
                        pass
            if output_exceeded:
                break
            if process.poll() is not None:
                # A non-interactive shell can exit while a background job still
                # owns the pipe. Cleanup below kills that whole process group.
                drain_available()
                break
    finally:
        _terminate_process_group(process)
        # Capture any final bytes already in the pipe without waiting for more.
        drain_available()
        selector.close()
        process.stdout.close()

    output = bytes(captured).decode("utf-8", errors="replace")
    return _BoundedCommandResult(
        output=output,
        returncode=int(process.returncode if process.returncode is not None else -1),
        timed_out=timed_out,
        output_exceeded=output_exceeded,
    )


def _shell_argv(command: str, *, timeout_s: float) -> list[str]:
    """Build a no-startup-file zsh with inherited hard resource ceilings."""

    cpu_seconds = max(1, math.ceil(timeout_s) + 2)
    wrapper = (
        "ulimit -S -c 0 && ulimit -H -c 0 && "
        "ulimit -S -n 256 && ulimit -H -n 256 && "
        "ulimit -S -f 524288 && ulimit -H -f 524288 && "
        f"ulimit -S -t {cpu_seconds} && ulimit -H -t {cpu_seconds} && "
        'exec /bin/zsh -df +o BG_NICE -c "$1"'
    )
    return ["/bin/zsh", "-dfc", wrapper, "mio-agent-command", command]


def _command_response(result: _BoundedCommandResult, policy: AgentToolPolicy) -> str:
    suffix = ""
    if result.timed_out:
        suffix = f"(command timed out after {policy.command_timeout_s:g}s)"
    elif result.output_exceeded:
        suffix = "(command stopped: output limit exceeded)"
    body = result.output.strip()
    if suffix:
        if len(suffix) >= policy.output_limit_chars:
            return cap_tool_output(suffix, policy.output_limit_chars)
        available = max(0, policy.output_limit_chars - len(suffix) - (1 if body else 0))
        body = cap_tool_output(body, available) if available else ""
        return cap_tool_output(
            f"{body}\n{suffix}".strip(),
            policy.output_limit_chars,
        )
    return cap_tool_output(body or "(no output)", policy.output_limit_chars)


def tool_bash(command: str, *, policy: AgentToolPolicy | None = None) -> str:
    """Execute a real shell inside an inherited workspace sandbox."""

    active_policy = _policy_or_read_only(policy)
    target = _audit_target_for_command(str(command))
    try:
        active_policy.require(AgentToolPermission.SHELL)
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        shell_argv = _shell_argv(command, timeout_s=active_policy.command_timeout_s)
        target = _audit_target_for_command(command, shell_argv)
        sandboxed_argv, command_env = sandboxed_command(shell_argv, active_policy)
        result = _run_bounded_process(
            sandboxed_argv,
            cwd=active_policy.primary_workspace,
            env=command_env,
            timeout_s=active_policy.command_timeout_s,
            output_limit_chars=active_policy.output_limit_chars,
        )
        response = _command_response(result, active_policy)
        outcome = "ok" if result.returncode == 0 else "nonzero"
        if result.timed_out:
            outcome = "timeout"
        elif result.output_exceeded:
            outcome = "output_limit"
        active_policy.audit(
            operation="bash",
            permission=AgentToolPermission.SHELL,
            target=target,
            allowed=True,
            outcome=outcome,
            detail=f"returncode={result.returncode}; output_chars={len(response)}",
        )
        return response
    except (AgentPermissionDenied, AgentPathViolation) as exc:
        active_policy.audit(
            operation="bash",
            permission=AgentToolPermission.SHELL,
            target=target,
            allowed=False,
            outcome="denied",
            detail=str(exc),
        )
        return _capped(active_policy, f"(permission denied: {exc})")
    except Exception as e:
        active_policy.audit(
            operation="bash",
            permission=AgentToolPermission.SHELL,
            target=target,
            allowed=True,
            outcome="error",
            detail=f"{type(e).__name__}: {e}",
        )
        return cap_tool_output(f"(error: {e})", active_policy.output_limit_chars)


def _validation_argv(raw_argv: object) -> tuple[str, ...]:
    """Validate a model-supplied direct argv without invoking a shell."""

    if not isinstance(raw_argv, (list, tuple)) or not raw_argv:
        raise ValueError("argv must be a non-empty array of strings")
    if len(raw_argv) > 128:
        raise ValueError("argv exceeds 128 entries")
    argv: list[str] = []
    total_chars = 0
    for value in raw_argv:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("argv entries must be non-empty strings without NUL bytes")
        if len(value) > 4096:
            raise ValueError("an argv entry exceeds 4096 characters")
        total_chars += len(value)
        if total_chars > 32_768:
            raise ValueError("argv exceeds 32768 characters")
        argv.append(value)
    return tuple(argv)


def _audit_target_for_argv(argv: tuple[str, ...], kind: str = "unrecognized") -> str:
    rendered = "\x00".join(argv)
    digest = hashlib.sha256(rendered.encode("utf-8", errors="replace")).hexdigest()[:32]
    executable = Path(argv[0]).name if argv else "invalid"
    return f"{kind}:{executable} sha256:{digest}"


def tool_validate(
    argv: list[str] | tuple[str, ...],
    *,
    policy: AgentToolPolicy | None = None,
) -> str:
    """Run one recognized validation as direct argv and audit its true status.

    This tool intentionally supports no shell grammar, wrappers, inline code,
    pipes, redirections, or success-masking operators. Coding-quality verdicts
    consume its structured audit event rather than interpreting command text.
    """

    active_policy = _policy_or_read_only(policy)
    target = "unrecognized:invalid sha256:" + hashlib.sha256(b"").hexdigest()[:32]
    try:
        active_policy.require(AgentToolPermission.SHELL)
        normalized = _validation_argv(argv)
        from mio.coding_quality import infer_validation_kind

        kind = infer_validation_kind(normalized)
        if kind is None:
            target = _audit_target_for_argv(normalized)
            active_policy.audit(
                operation="validate",
                permission=AgentToolPermission.SHELL,
                target=target,
                allowed=False,
                outcome="unrecognized",
                detail="direct argv is not a recognized validation command",
            )
            return _capped(
                active_policy,
                "(validation rejected: use a direct supported test, static, build, or git diff --check argv)",
            )

        kind_name = str(getattr(kind, "value", kind))
        target = _audit_target_for_argv(normalized, kind_name)
        sandboxed_argv, command_env = sandboxed_command(list(normalized), active_policy)
        scratch = Path(
            tempfile.mkdtemp(
                prefix=".mio-validation-",
                dir=active_policy.primary_workspace,
            )
        )
        command_env = dict(command_env)
        command_env.update(
            {
                "HOME": str(scratch),
                "TMPDIR": str(scratch),
                "XDG_CACHE_HOME": str(scratch / "cache"),
                "PYTHONPYCACHEPREFIX": str(scratch / "pycache"),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        started = time.perf_counter()
        try:
            result = _run_bounded_process(
                sandboxed_argv,
                cwd=active_policy.primary_workspace,
                env=command_env,
                timeout_s=active_policy.command_timeout_s,
                output_limit_chars=active_policy.output_limit_chars,
            )
        finally:
            # The unique trusted scratch root prevents test caches/temp files
            # from becoming workspace mutations or leaking between checks.
            shutil.rmtree(scratch, ignore_errors=True)
        duration_s = time.perf_counter() - started
        response = _command_response(result, active_policy)
        outcome = "ok" if result.returncode == 0 else "nonzero"
        if result.timed_out:
            outcome = "timeout"
        elif result.output_exceeded:
            outcome = "output_limit"
        active_policy.audit(
            operation="validate",
            permission=AgentToolPermission.SHELL,
            target=target,
            allowed=True,
            outcome=outcome,
            detail=(
                f"kind={kind_name}; returncode={result.returncode}; "
                f"duration_ms={duration_s * 1000:.3f}; output_chars={len(response)}"
            ),
        )
        verdict = "PASS" if outcome == "ok" else "FAIL"
        return _capped(
            active_policy,
            f"(validation {kind_name}: {verdict}; returncode={result.returncode})\n{response}",
        )
    except (AgentPermissionDenied, AgentPathViolation) as exc:
        active_policy.audit(
            operation="validate",
            permission=AgentToolPermission.SHELL,
            target=target,
            allowed=False,
            outcome="denied",
            detail=str(exc),
        )
        return _capped(active_policy, f"(permission denied: {exc})")
    except Exception as exc:
        active_policy.audit(
            operation="validate",
            permission=AgentToolPermission.SHELL,
            target=target,
            allowed=True,
            outcome="error",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return _capped(active_policy, f"(validation error: {exc})")


def tool_read(path: str, *, policy: AgentToolPolicy | None = None) -> str:
    """Read a file and return its contents."""
    active_policy = _policy_or_read_only(policy)
    target = str(path)
    try:
        active_policy.require(AgentToolPermission.READ)
        resolved = resolve_workspace_path(path, active_policy)
        target = str(resolved.absolute)
        content, truncated = read_workspace_text(
            resolved,
            max_chars=active_policy.output_limit_chars,
        )
        response = cap_tool_output(
            content + ("\n... (output truncated)" if truncated else ""),
            active_policy.output_limit_chars,
        )
        active_policy.audit(
            operation="read",
            permission=AgentToolPermission.READ,
            target=target,
            allowed=True,
            outcome="ok",
            detail=f"output_chars={len(response)}; truncated={str(truncated).lower()}",
        )
        return response
    except FileNotFoundError:
        active_policy.audit(
            operation="read",
            permission=AgentToolPermission.READ,
            target=target,
            allowed=True,
            outcome="not_found",
        )
        return _capped(active_policy, f"(file not found: {path})")
    except (AgentPermissionDenied, AgentPathViolation) as exc:
        active_policy.audit(
            operation="read",
            permission=AgentToolPermission.READ,
            target=target,
            allowed=False,
            outcome="denied",
            detail=str(exc),
        )
        return _capped(active_policy, f"(permission denied reading {path}: {exc})")
    except Exception as e:
        active_policy.audit(
            operation="read",
            permission=AgentToolPermission.READ,
            target=target,
            allowed=True,
            outcome="error",
            detail=f"{type(e).__name__}: {e}",
        )
        return _capped(active_policy, f"(error reading {path}: {e})")


def tool_write(
    path: str,
    content: str,
    *,
    policy: AgentToolPolicy | None = None,
) -> str:
    """Write content to a file."""
    active_policy = _policy_or_read_only(policy)
    target = str(path)
    try:
        active_policy.require(AgentToolPermission.WRITE)
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if len(content) > active_policy.file_limit_chars:
            raise AgentPathViolation("content exceeds the file-size policy limit")
        resolved = resolve_workspace_path(path, active_policy)
        target = str(resolved.absolute)
        write_workspace_text(resolved, content)
        active_policy.audit(
            operation="write",
            permission=AgentToolPermission.WRITE,
            target=target,
            allowed=True,
            outcome="ok",
            detail=f"content_chars={len(content)}",
        )
        return _capped(active_policy, f"(written {len(content)} chars to {path})")
    except (AgentPermissionDenied, AgentPathViolation) as exc:
        active_policy.audit(
            operation="write",
            permission=AgentToolPermission.WRITE,
            target=target,
            allowed=False,
            outcome="denied",
            detail=str(exc),
        )
        return _capped(active_policy, f"(permission denied writing {path}: {exc})")
    except Exception as e:
        active_policy.audit(
            operation="write",
            permission=AgentToolPermission.WRITE,
            target=target,
            allowed=True,
            outcome="error",
            detail=f"{type(e).__name__}: {e}",
        )
        return _capped(active_policy, f"(error writing {path}: {e})")


def tool_edit(
    path: str,
    old: str,
    new: str,
    *,
    policy: AgentToolPolicy | None = None,
) -> str:
    """Replace a substring in a file."""
    active_policy = _policy_or_read_only(policy)
    target = str(path)
    required_permission = AgentToolPermission.WRITE
    try:
        active_policy.require(required_permission)
        required_permission = AgentToolPermission.READ
        active_policy.require(required_permission)
        resolved = resolve_workspace_path(path, active_policy)
        target = str(resolved.absolute)
        text, truncated = read_workspace_text(
            resolved,
            max_chars=active_policy.file_limit_chars,
        )
        if truncated:
            raise AgentPathViolation("file exceeds the editable-size policy limit")
        if old not in text:
            active_policy.audit(
                operation="edit",
                permission=AgentToolPermission.WRITE,
                target=target,
                allowed=True,
                outcome="old_string_not_found",
            )
            return _capped(active_policy, f"(old_string not found in {path})")
        replacement = text.replace(old, new, 1)
        if len(replacement) > active_policy.file_limit_chars:
            raise AgentPathViolation("edited file exceeds the file-size policy limit")
        write_workspace_text(resolved, replacement)
        active_policy.audit(
            operation="edit",
            permission=AgentToolPermission.WRITE,
            target=target,
            allowed=True,
            outcome="ok",
            detail="replacements=1",
        )
        return _capped(active_policy, f"(edited {path}: 1 replacement)")
    except FileNotFoundError:
        active_policy.audit(
            operation="edit",
            permission=AgentToolPermission.WRITE,
            target=target,
            allowed=True,
            outcome="not_found",
        )
        return _capped(active_policy, f"(file not found: {path})")
    except (AgentPermissionDenied, AgentPathViolation) as exc:
        active_policy.audit(
            operation="edit",
            permission=required_permission,
            target=target,
            allowed=False,
            outcome="denied",
            detail=str(exc),
        )
        return _capped(active_policy, f"(permission denied editing {path}: {exc})")
    except Exception as e:
        active_policy.audit(
            operation="edit",
            permission=AgentToolPermission.WRITE,
            target=target,
            allowed=True,
            outcome="error",
            detail=f"{type(e).__name__}: {e}",
        )
        return _capped(active_policy, f"(error editing {path}: {e})")


def tool_list_mio_skills(
    query: str = "",
    tag: str = "",
    source: str = "",
    limit: int = 50,
) -> str:
    """Search Mio-local instruction skills without executing them."""
    import json

    from mio.skill_catalog import list_mio_skills

    return json.dumps(
        list_mio_skills(query=query, tag=tag, source=source, limit=limit),
        ensure_ascii=False,
    )


def tool_read_mio_skill(name: str, max_chars: int = 32_000) -> str:
    """Read one Mio-local SKILL.md through the confined catalog API."""
    import json

    from mio.skill_catalog import read_mio_skill

    return json.dumps(read_mio_skill(name=name, max_chars=max_chars), ensure_ascii=False)


def tool_list_mcp_tools(
    server: str,
    *,
    policy: AgentToolPolicy | None = None,
) -> str:
    """Discover tools on one enabled local Mio MCP server."""
    import json

    from mio.mcp import list_mcp_tools

    active_policy = _policy_or_read_only(policy)
    return json.dumps(
        list_mcp_tools(server, agent_policy=active_policy),
        ensure_ascii=False,
    )


def tool_call_mcp_tool(
    server: str,
    name: str,
    arguments: dict | None = None,
    *,
    policy: AgentToolPolicy | None = None,
) -> str:
    """Call one advertised tool on an enabled local Mio MCP server."""
    import json

    from mio.mcp import call_mcp_tool

    active_policy = _policy_or_read_only(policy)
    return json.dumps(
        call_mcp_tool(
            server,
            name,
            arguments or {},
            agent_policy=active_policy,
        ),
        ensure_ascii=False,
    )


# Tool registry used by the native agent's tool-use loop.
AGENT_TOOLS = {
    "bash": {
        "fn": tool_bash,
        "args": ["command"],
        "permission": AgentToolPermission.SHELL,
    },
    "validate": {
        "fn": tool_validate,
        "args": ["argv"],
        "permission": AgentToolPermission.SHELL,
    },
    "read": {
        "fn": tool_read,
        "args": ["path"],
        "permission": AgentToolPermission.READ,
    },
    "write": {
        "fn": tool_write,
        "args": ["path", "content"],
        "permission": AgentToolPermission.WRITE,
    },
    "edit": {
        "fn": tool_edit,
        "args": ["path", "old", "new"],
        "permission": AgentToolPermission.WRITE,
    },
    "list_mio_skills": {
        "fn": tool_list_mio_skills,
        "args": ["query", "tag", "source", "limit"],
    },
    "read_mio_skill": {"fn": tool_read_mio_skill, "args": ["name", "max_chars"]},
    "list_mcp_tools": {
        "fn": tool_list_mcp_tools,
        "args": ["server"],
        "inject_policy": True,
    },
    "call_mcp_tool": {
        "fn": tool_call_mcp_tool,
        "args": ["server", "name", "arguments"],
        "inject_policy": True,
    },
}

AGENT_TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a zsh command (including pipes, redirections, and scripts) in an inherited workspace sandbox. Requires the caller's shell grant; network needs a separate caller grant; output and runtime are capped.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "validate",
        "description": (
            "Run a recognized test, static check, build, or git diff --check as direct argv. "
            "Use this instead of bash for evidence after any edit. No shell, pipes, redirections, "
            "wrappers, or inline code are accepted; the true exit status is audited."
        ),
        "parameters": {"type": "object", "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 128,
                "description": "Direct executable and arguments, e.g. [\"python3\", \"-m\", \"pytest\", \"-q\"]",
            },
        }, "required": ["argv"]},
    }},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a regular, non-symlink file inside a caller-allowed workspace. Output is capped by policy.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write",
        "description": "Atomically create or overwrite a non-symlink file inside a caller-allowed workspace. Requires the caller's write grant.",
        "parameters": {"type": "object", "properties": {
            "path":    {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace a substring in a confined regular file. Requires the caller's read and write grants.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old":  {"type": "string", "description": "Exact substring to replace"},
            "new":  {"type": "string", "description": "Replacement"},
        }, "required": ["path", "old", "new"]},
    }},
    {"type": "function", "function": {
        "name": "list_mio_skills",
        "description": (
            "Search instruction skills installed inside Mio. Filter by text, exact tag, "
            "or source. This only lists metadata and never executes skill code."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words matched across name, description, and tags"},
            "tag": {"type": "string", "description": "Optional exact tag"},
            "source": {"type": "string", "description": "Optional exact source id"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "read_mio_skill",
        "description": (
            "Read the validated SKILL.md instructions for one Mio-local skill. "
            "Call list_mio_skills first when the name is unknown. Never executes the skill."
        ),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Installed skill name or unique canonical name"},
            "max_chars": {
                "type": "integer", "minimum": 1, "maximum": 200000, "default": 32000,
            },
        }, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "list_mcp_tools",
        "description": (
            "List tools advertised by one enabled Mio-local MCP server. "
            "Known built-ins: headroom, llm-wiki, ponytail. Never reaches remote/auth MCPs."
        ),
        "parameters": {"type": "object", "properties": {
            "server": {"type": "string", "description": "Enabled Mio MCP server name"},
        }, "required": ["server"]},
    }},
    {"type": "function", "function": {
        "name": "call_mcp_tool",
        "description": (
            "Call an advertised tool on an enabled Mio-local MCP. Discover with list_mcp_tools first. "
            "Use mutating tools only when the user's request explicitly requires that change."
        ),
        "parameters": {"type": "object", "properties": {
            "server": {"type": "string", "description": "Enabled Mio MCP server name"},
            "name": {"type": "string", "description": "Advertised MCP tool name"},
            "arguments": {"type": "object", "description": "Tool arguments", "additionalProperties": True},
        }, "required": ["server", "name"]},
    }},
]


# --- Context Interactive Selection ---

_CTX_OPTIONS = [
    (8192, "8K"),
    (16384, "16K"),
    (32768, "32K"),
    (65536, "64K"),
    (131072, "128K"),
    (262144, "256K"),
]

_TQ_OPTIONS = [
    (4, "TQ 4-bit", "3.6x compression, best speed"),
    (3, "TQ 3-bit", "4.7x compression, moderate quality loss"),
    (2, "TQ 2-bit", "5.5x compression, max context"),
    (0, "OFF", "no compression, fp16 cache"),
]


def _context_interactive(tc) -> str:
    """Interactive context + TQ selection with numbered menus."""
    from rich.prompt import IntPrompt

    # Show current
    tq_label = f"TQ {tc.tq_bits}-bit" if tc.tq_bits < 16 else "OFF"
    console.print(f"\n[dim]Current: {tc.context_window:,} tokens, {tq_label}[/dim]\n")

    # Step 1: Context window
    console.print("[bold]Select context window:[/bold]")
    current_idx = 0
    for i, (tokens, label) in enumerate(_CTX_OPTIONS):
        marker = " [cyan]<-- current[/cyan]" if tokens == tc.context_window else ""
        console.print(f"  [{i + 1}] {label:>5s}  ({tokens:>7,} tokens){marker}")
        if tokens == tc.context_window:
            current_idx = i + 1

    try:
        ctx_choice = IntPrompt.ask("Context", default=current_idx or 5)
    except (EOFError, KeyboardInterrupt):
        return "Cancelled."

    ctx_idx = max(1, min(ctx_choice, len(_CTX_OPTIONS))) - 1
    new_ctx = _CTX_OPTIONS[ctx_idx][0]

    console.print()

    # Step 2: TQ mode
    console.print("[bold]Select TurboQuant cache:[/bold]")
    current_tq_idx = 0
    for i, (bits, name, desc) in enumerate(_TQ_OPTIONS):
        marker = " [cyan]<-- current[/cyan]" if bits == tc.tq_bits else ""
        console.print(f"  [{i + 1}] {name:12s} ({desc}){marker}")
        if bits == tc.tq_bits:
            current_tq_idx = i + 1

    try:
        tq_choice = IntPrompt.ask("TQ mode", default=current_tq_idx or 1)
    except (EOFError, KeyboardInterrupt):
        return "Cancelled."

    tq_idx = max(1, min(tq_choice, len(_TQ_OPTIONS))) - 1
    new_tq = _TQ_OPTIONS[tq_idx][0]

    # Apply
    tc.context_window = new_ctx
    tc.max_output_tokens = min(new_ctx // 4, 8192)
    if new_tq > 0:
        tc.tq_bits = new_tq
        tc.tq_use_rotation = True
        tc.tq_use_normalization = True
    else:
        tc.tq_bits = 16
        tc.tq_use_rotation = False
        tc.tq_use_normalization = False

    tq_display = f"TQ {new_tq}-bit" if new_tq > 0 else "OFF (fp16)"
    return (
        f"\n**Context set:** {_CTX_OPTIONS[ctx_idx][1]}, {tq_display}\n"
        f"- Window: {new_ctx:,} tokens\n"
        f"- Max output: {tc.max_output_tokens:,} tokens\n"
        f"- KV cache: {tq_display}"
    )


# --- Slash Commands ---

def handle_slash_command(
    cmd: str,
    manager: ModelManager,
    config: MioConfig,
    state: dict,
) -> str | None:
    """Handle a slash command. Returns response text or None if not a command."""
    parts = cmd.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None

    command = parts[0].lower()
    args = parts[1:]

    if command == "/help":
        return (
            "**Slash Commands:**\n"
            "- `/model` - Show current model and tier\n"
            "- `/tier [max|large-moe|large|medium|small]` - Switch tier\n"
            "- `/context [8k|16k|32k|64k|128k|256k] [tq2|tq3|tq4|off]` - Set context + TQ\n"
            "- `/caveman [off|lite|full|ultra]` - Set communication mode\n"
            "- `/ponytail [off|lite|full|ultra]` - Set engineering policy\n"
            "- `/effort [low|medium|high|xhigh|ultra]` - Set mandatory coding-quality gate\n"
            "- `/tq` - Show TurboQuant status\n"
            "- `/status` - Show engine status\n"
            "- `/models` - List available models\n"
            "- `/configure` - Run configuration wizard\n"
            "- `/clear` - Clear conversation history\n"
            "- `/help` - This message\n"
            "- `/quit` - Exit"
        )

    elif command == "/model":
        tier = state.get("tier", "large-moe")
        tc = config.tiers.get(tier)
        if tc:
            return (
                f"**Current Model:**\n"
                f"- Tier: {tier}\n"
                f"- Target: {tc.target_model}\n"
                f"- Draft: {tc.draft_model}\n"
                f"- Context: {tc.context_window:,} tokens\n"
                f"- TQ: {tc.tq_bits}-bit"
            )
        return f"Tier: {tier} (not configured)"

    elif command == "/tier":
        if args:
            new_tier = args[0].lower()
            if new_tier not in config.tiers:
                return f"Unknown tier: {new_tier}. Available: {', '.join(config.tiers.keys())}"
            old_tier = state.get("tier", "large-moe")
            if new_tier != old_tier:
                console.print(f"[yellow]Reloading model for {new_tier} tier...[/yellow]")
                # Unload old if different model
                if old_tier in manager.loaded_tiers():
                    manager.unload_tier(old_tier)
                manager.load_tier(new_tier)
                state["tier"] = new_tier
                state["reload"] = True
                return f"Switched to **{new_tier}** tier. Model reloaded."
            return f"Already on **{new_tier}** tier."
        return f"Current tier: **{state.get('tier', 'large-moe')}**. Usage: `/tier large-moe|large|medium|small`"

    elif command == "/caveman":
        if args:
            level = args[0].lower()
            if level not in CAVEMAN_LEVELS:
                return f"Unknown level: {level}. Options: off, lite, full, ultra"
            state["prompt_policy"] = PromptPolicy.resolve(caveman=level)
            return f"Prompt policy: **{state['prompt_policy'].label}**"
        policy = state.get("prompt_policy", PromptPolicy())
        return f"Prompt policy: **{policy.label}**. Usage: `/caveman off|lite|full|ultra`"

    elif command == "/ponytail":
        if args and args[0].lower() == "off":
            state["prompt_policy"] = PromptPolicy.resolve(prompt_mode=PromptMode.NONE)
            return "Prompt policy: **none**"
        if args and args[0].lower() in {"lite", "full", "ultra"}:
            state["prompt_policy"] = PromptPolicy.resolve(ponytail=args[0].lower())
            return f"Prompt policy: **{state['prompt_policy'].label}**"
        policy = state.get("prompt_policy", PromptPolicy())
        return f"Prompt policy: **{policy.label}**. Usage: `/ponytail off|lite|full|ultra`"

    elif command == "/effort":
        current = str(state.get("coding_effort", "medium"))
        if args:
            level = args[0].lower()
            if level not in CODING_EFFORT_LEVELS:
                return (
                    f"Unknown effort: {level}. Options: "
                    + ", ".join(CODING_EFFORT_LEVELS)
                )
            if state.get("quality_gate_pending"):
                return (
                    "Cannot change effort while a coding-quality obligation is pending. "
                    "Complete or explicitly report the current validation first."
                )
            state["coding_effort"] = level
            return f"Coding-quality effort: **{level}** (mandatory)"
        return (
            f"Coding-quality effort: **{current}** (mandatory). Usage: "
            "`/effort low|medium|high|xhigh|ultra`"
        )

    elif command == "/context":
        tier = state.get("tier", "large-moe")
        tc = config.tiers.get(tier)
        if not tc:
            return "No tier configured."
        # Interactive selection
        result = _context_interactive(tc)
        if result and not result.startswith("Cancelled"):
            # Reload model with new settings
            console.print("[yellow]Reloading model with new context/TQ settings...[/yellow]")
            if tier in manager.loaded_tiers():
                manager.unload_tier(tier)
            manager.load_tier(tier)
            state["reload"] = True
        return result

    elif command == "/tq":
        tier = state.get("tier", "large-moe")
        tc = config.tiers.get(tier)
        if tc:
            return (
                f"**TurboQuant V2:**\n"
                f"- Bits: {tc.tq_bits}\n"
                f"- Group size: {tc.tq_group_size}\n"
                f"- Rotation: {tc.tq_use_rotation}\n"
                f"- Normalization: {tc.tq_use_normalization}\n"
                f"- QJL: {tc.tq_use_qjl}"
            )
        return "No tier configured."

    elif command == "/status":
        status = manager.status()
        loaded = status.get("loaded_tiers", [])
        vram = status.get("vram_gb", 0)
        lines = ["**Engine Status:**", f"- Loaded tiers: {', '.join(loaded)}", f"- VRAM: {vram:.1f} GB"]
        for name, info in status.get("engines", {}).items():
            lines.append(f"- {name}: {info.get('last_gen_tps', 0):.1f} tok/s")
        return "\n".join(lines)

    elif command == "/models":
        from mio.models.registry import KNOWN_MODELS, SUPPORTED_ADAPTERS

        lines = ["**Available Models:**"]
        for key, entry in KNOWN_MODELS.items():
            supported = entry.adapter in SUPPORTED_ADAPTERS
            status = "ready" if supported else "needs adapter"
            lines.append(f"- `{key}` ({entry.description}) [{status}]")
        return "\n".join(lines)

    elif command == "/configure":
        from mio.configure import configure_interactive

        configure_interactive(config)
        return "Configuration updated. Restart to apply changes."

    elif command == "/clear":
        state["messages"] = []
        return "Conversation cleared."

    elif command in ("/quit", "/exit", "/q"):
        return "__QUIT__"

    else:
        return f"Unknown command: {command}. Type `/help` for available commands."


# --- Main Agent Loop ---

def run_agent(
    config: MioConfig,
    manager: ModelManager,
    tier: str = "large-moe",
    initial_prompt: str | None = None,
    prompt_policy: PromptPolicy | None = None,
    tool_policy: AgentToolPolicy | None = None,
    coding_effort: str = "medium",
    quality_gate_enabled: bool = True,
) -> None:
    """Run the interactive coding agent."""
    # Library callers that omit a policy are intentionally read-only. The
    # native CLI passes its named coding policy explicitly at the trust edge.
    declared_tool_policy = tool_policy or _default_read_policy()
    if coding_effort not in CODING_EFFORT_LEVELS:
        raise ValueError(
            f"coding_effort must be one of: {', '.join(CODING_EFFORT_LEVELS)}"
        )
    state = {
        "tier": tier,
        "prompt_policy": prompt_policy or PromptPolicy(),
        "tool_policy": declared_tool_policy,
        "coding_effort": coding_effort,
        "quality_gate_enabled": bool(quality_gate_enabled),
        "messages": [],
    }

    # Banner
    console.print(Panel(
        "[bold cyan]Mio Agent[/bold cyan]\n"
        f"[dim]Tier: {tier} | Prompt: {state['prompt_policy'].label} | "
        f"Quality: {coding_effort + ' (mandatory)' if quality_gate_enabled else 'off (benchmark control)'} "
        "| /help for commands[/dim]",
        border_style="cyan",
    ))
    console.print()

    engine = manager.get_engine(tier)

    # Process initial prompt if provided
    if initial_prompt:
        _process_user_input(initial_prompt, engine, manager, config, state)

    # Main loop
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue

        # Slash commands
        if user_input.strip().startswith("/"):
            result = handle_slash_command(user_input, manager, config, state)
            if result == "__QUIT__":
                console.print("[dim]Goodbye.[/dim]")
                break
            if result:
                console.print(Markdown(result))
                console.print()
            # Pick up reloaded engine after /tier or /context
            if state.pop("reload", False):
                current_tier = state.get("tier", "large-moe")
                engine = manager.get_engine(current_tier)
                console.print(f"[green]Engine ready: {current_tier}[/green]\n")
            continue

        _process_user_input(user_input, engine, manager, config, state)


def _process_user_input(
    user_input: str,
    engine: MioEngine,
    manager: ModelManager,
    config: MioConfig,
    state: dict,
) -> AgentTurnResult:
    """Process a user message: build prompt, generate, run any tool calls
    the model emits, feed the results back, and repeat until the model
    stops calling tools. The last bounded model round is reserved for a
    no-tools status synthesis when execution has not converged. Without this
    loop the model would just emit <tool_call>…</tool_call> tags as literal
    text and the file would never actually be written.
    """
    turn_started = time.perf_counter()
    current_tier = state.get("tier", "large-moe")
    if current_tier in manager.loaded_tiers():
        engine = manager.get_engine(current_tier)

    # Build system prompt (selected policy + hint that tools are real)
    prompt_policy = state.get("prompt_policy", PromptPolicy())
    tool_policy = state.get("tool_policy")
    if not isinstance(tool_policy, AgentToolPolicy):
        tool_policy = _default_read_policy()
    audit_events: list[AgentAuditEvent] = []

    def capture_audit(event: AgentAuditEvent) -> None:
        audit_events.append(event)
        tool_policy.audit_sink(event)

    # Capture the exact structured outcomes without changing the authority or
    # resource limits of the caller-declared policy. The original sink still
    # receives every event for operational audit logging.
    execution_policy = replace(tool_policy, audit_sink=capture_audit)
    system_prompt = AGENT_SYSTEM_PROMPT
    tool_registry = state.get("tool_registry", AGENT_TOOLS)
    tool_specs = state.get("tool_specs", AGENT_TOOLS_SPEC)
    if not isinstance(tool_registry, Mapping):
        raise TypeError("state.tool_registry must be a frozen dispatcher mapping")
    if not isinstance(tool_specs, (list, tuple)):
        raise TypeError("state.tool_specs must be a tool-schema sequence")
    quality_gate = None
    if bool(state.get("quality_gate_enabled", False)):
        from mio.coding_quality import CodingQualityGate

        pending_gate = state.get("_quality_gate")
        if isinstance(pending_gate, CodingQualityGate) and not pending_gate.decision().satisfied:
            quality_gate = pending_gate
        else:
            quality_gate = CodingQualityGate.start(
                execution_policy.workspace_roots,
                user_input,
                effort=str(state.get("coding_effort", "medium")),
                enabled=True,
            )
        system_prompt += quality_gate.system_instructions()

    # Initial messages
    current_messages = apply_prompt_policy(
        [{"role": "system", "content": system_prompt}],
        prompt_policy,
    )
    current_messages.extend(state.get("messages", []))
    current_messages.append({"role": "user", "content": user_input})
    # Persist the user turn early so history is consistent even if generation
    # is interrupted.
    state["messages"].append({"role": "user", "content": user_input})

    from mio.tool_calls import StreamingToolCallParser, parse_tool_calls as _parse_tc

    terminal_assistant_text = ""
    tool_calls_used = 0
    tool_result_chars = 0
    forced_finalization_reason: str | None = None
    terminal_reason = "model_final"
    round_traces: list[AgentRoundTrace] = []
    tool_event_traces: list[AgentToolTrace] = []

    for _round_idx in range(_MAX_AGENT_ROUNDS_PER_TURN):
        finalization_reason = forced_finalization_reason
        if finalization_reason is None and _round_idx == _MAX_AGENT_ROUNDS_PER_TURN - 1:
            finalization_reason = f"model round limit {_MAX_AGENT_ROUNDS_PER_TURN} reached"
        finalization_only = finalization_reason is not None
        generation_messages = current_messages
        generation_tools = tool_specs
        if finalization_only:
            generation_messages = list(current_messages) + [{
                "role": "user",
                "content": _FINALIZE_TOOL_LOOP.format(reason=finalization_reason),
            }]
            # The final round is reserved for truthful synthesis. Omitting the
            # tool schema prevents one more mutation from silently exceeding
            # the model-round, call-count, or result-size budget.
            generation_tools = None

        console.print("[bold green]Mio[/bold green]: ", end="")
        full_text = ""
        visible_parts: list[str] = []
        display_parser = StreamingToolCallParser()
        for chunk, metrics in engine.generate_stream(generation_messages, tools=generation_tools):
            full_text += chunk
            # Keep native tool XML for parsing, but never expose it as ordinary
            # assistant prose in the terminal. The incremental parser also
            # handles a tag split across model-stream chunks.
            visible_chunk = display_parser.feed(chunk)
            if visible_chunk:
                visible_parts.append(visible_chunk)
                console.print(visible_chunk, end="", highlight=False)
        visible_tail, streamed_tool_calls = display_parser.flush()
        if visible_tail:
            visible_parts.append(visible_tail)
            console.print(visible_tail, end="", highlight=False)
        console.print()

        # Metrics line
        m = engine.last_metrics
        round_traces.append(_round_trace(_round_idx, m))
        if m.generation_tps > 0:
            console.print(
                f"[dim]  {m.generation_tps:.1f} tok/s · "
                f"{m.completion_tokens} tokens · "
                f"{m.total_time_s:.2f}s[/dim]"
            )

        # Extract tool calls (OpenAI-format: {function: {name, arguments}})
        import json as _json
        _leading, parsed_tool_calls = _parse_tc(full_text)
        # Streaming and whole-response parsing have the same grammar. Prefer
        # the streamed calls so terminal filtering and dispatch share one
        # interpretation; retain the whole-response fallback for test doubles
        # and unusual generators that return a single incomplete stream event.
        tool_calls = streamed_tool_calls or parsed_tool_calls
        # Persist exactly what the terminal filter considered ordinary prose.
        # An unterminated tool block is neither shown nor smuggled into future
        # conversation history as assistant content.
        visible_text = "".join(visible_parts).strip()
        if finalization_only:
            terminal_reason = "budget_finalization"
            if tool_calls:
                # A model can still emit memorized XML after the schema is
                # removed. It is never dispatched on the reserved final round.
                notice = (
                    f"Tool loop stopped: {finalization_reason}. "
                    "The final model response requested another tool, so no "
                    "additional operation was executed."
                )
                console.print(notice, style="yellow")
            terminal_assistant_text = "\n\n".join(
                text for text in (visible_text, notice if tool_calls else "") if text
            )
            if quality_gate is not None and not quality_gate.decision().satisfied:
                quality_notice = (
                    "Coding-quality gate: INCOMPLETE. The latest workspace revision "
                    "does not have the required trusted validation evidence; no success "
                    "is certified."
                )
                console.print(quality_notice, style="yellow")
                terminal_assistant_text = "\n\n".join(
                    text for text in (terminal_assistant_text, quality_notice) if text
                )
                terminal_reason = "quality_incomplete"
            break

        if not tool_calls:
            if quality_gate is not None and not quality_gate.decision().satisfied:
                feedback = quality_gate.feedback()
                console.print(feedback, style="yellow")
                current_messages = list(current_messages) + [
                    {"role": "assistant", "content": visible_text or None},
                    {"role": "user", "content": feedback},
                ]
                terminal_reason = "quality_reprompt"
                continue
            terminal_assistant_text = visible_text
            terminal_reason = "model_final"
            break  # model stopped calling tools

        # Re-render the previous turn through the tokenizer's native structured
        # tool transcript. Qwen templates expect assistant.tool_calls followed
        # by role=tool messages; treating a tool result as a fresh user query
        # breaks multi-step state and can make the model repeat the same read.
        invocations: list[tuple[dict, str, dict]] = []
        normalized_calls: list[dict] = []
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = str(fn.get("name", ""))
            raw_args = fn.get("arguments", "{}")
            try:
                args = _json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except Exception:
                args = {}
            normalized_call = dict(tc)
            normalized_call["function"] = {**fn, "name": name, "arguments": args}
            normalized_calls.append(normalized_call)
            invocations.append((tc, name, args))

        current_messages = list(current_messages) + [{
            "role": "assistant",
            "content": _leading or None,
            "tool_calls": normalized_calls,
        }]

        for tc, name, args in invocations:
            spec = tool_registry.get(name)
            audit_start = len(audit_events)
            gate_before = (
                quality_gate.before_tool(name, args)
                if quality_gate is not None and spec is not None
                else None
            )
            tool_calls_used += 1
            if tool_calls_used > _MAX_TOOL_CALLS_PER_TURN:
                result = "(tool call budget exhausted for this turn)"
                forced_finalization_reason = (
                    f"tool call limit {_MAX_TOOL_CALLS_PER_TURN} reached"
                )
            elif tool_result_chars >= _MAX_TOOL_RESULT_CHARS_PER_TURN:
                result = "(tool result budget exhausted for this turn)"
                forced_finalization_reason = (
                    f"tool result limit {_MAX_TOOL_RESULT_CHARS_PER_TURN} characters reached"
                )
            elif not spec:
                result = f"(unknown tool: {name})"
            else:
                try:
                    kwargs = {k: args[k] for k in spec["args"] if k in args}
                    if "permission" in spec or spec.get("inject_policy"):
                        kwargs["policy"] = execution_policy
                    result = spec["fn"](**kwargs)
                except Exception as e:
                    result = f"(tool error: {type(e).__name__}: {e})"
            invocation_audits = audit_events[audit_start:]
            if quality_gate is not None and spec is not None and gate_before is not None:
                quality_gate.after_tool(
                    name,
                    args,
                    before=gate_before,
                    audit_events=invocation_audits,
                )
            tool_event_traces.extend(
                _tool_trace(_round_idx, name, event)
                for event in invocation_audits
            )
            remaining_result_chars = max(
                0,
                _MAX_TOOL_RESULT_CHARS_PER_TURN - tool_result_chars,
            )
            per_call_limit = min(tool_policy.output_limit_chars, remaining_result_chars)
            result = (
                cap_tool_output(str(result), per_call_limit)
                if per_call_limit
                else "(tool result budget exhausted for this turn)"
            )
            tool_result_chars += len(result)
            if tool_result_chars >= _MAX_TOOL_RESULT_CHARS_PER_TURN:
                forced_finalization_reason = (
                    f"tool result limit {_MAX_TOOL_RESULT_CHARS_PER_TURN} characters reached"
                )
            preview = ", ".join(k for k in (spec["args"] if spec else []) if k in args)
            console.print(
                f"  ◆ {name}({preview}) → {str(result)[:120]}",
                style="dim cyan",
                markup=False,
            )
            # The template supplies the surrounding <tool_response> element.
            # Neutralize only a result-supplied closing delimiter so ordinary
            # source code operators such as '<' remain exact for later edits.
            safe_result = str(result).replace(
                "</tool_response>",
                "&lt;/tool_response&gt;",
            )
            current_messages.append({
                "role": "tool",
                "tool_call_id": str(tc.get("id", "")),
                "name": name,
                "content": safe_result,
            })

    console.print()
    # Persist the final assistant text (joined across rounds) so multi-turn
    # history stays sensible.
    state["messages"].append({
        "role": "assistant",
        # Earlier pre-tool narration was already visible live and was replayed
        # while this turn ran. Persist only the terminal synthesis so the next
        # user turn starts from the outcome, not a duplicate execution diary.
        "content": terminal_assistant_text or "(tool-only turn)",
    })

    # Trim history (keep last 40 entries — ~20 exchanges)
    if len(state["messages"]) > 40:
        state["messages"] = state["messages"][-40:]

    quality_report = quality_gate.report() if quality_gate is not None else None
    if quality_gate is not None and not quality_gate.decision().satisfied:
        state["_quality_gate"] = quality_gate
        state["quality_gate_pending"] = True
    else:
        state.pop("_quality_gate", None)
        state["quality_gate_pending"] = False

    return AgentTurnResult(
        assistant_text=terminal_assistant_text or "(tool-only turn)",
        terminal_reason=terminal_reason,
        rounds=tuple(round_traces),
        tool_events=tuple(tool_event_traces),
        tool_calls=tool_calls_used,
        tool_result_chars=tool_result_chars,
        wall_time_s=time.perf_counter() - turn_started,
        quality_gate=quality_report,
    )
