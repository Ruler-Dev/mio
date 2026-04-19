"""Standalone Mio coding agent: interactive LLM with tools and slash commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mio.config import MioConfig
from mio.engine import MioEngine
from mio.model_manager import ModelManager

console = Console()

# --- System Prompts ---

AGENT_SYSTEM_PROMPT = """You are Mio, a fast local coding agent running on Apple Silicon.
You have access to tools: bash (run shell commands), read (read files), write (write files), edit (edit files).
When the user asks you to write or modify code, do it directly. Be precise and concise.
Always show the code you write or modify.
When running bash commands, show the command and its output.
If you encounter an error, fix it and retry."""

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

def tool_bash(command: str) -> str:
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "(command timed out after 30s)"
    except Exception as e:
        return f"(error: {e})"


def tool_read(path: str) -> str:
    """Read a file and return its contents."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"(file not found: {path})"
        content = p.read_text(errors="replace")
        if len(content) > 10000:
            return content[:10000] + f"\n... (truncated, {len(content)} chars total)"
        return content
    except Exception as e:
        return f"(error reading {path}: {e})"


def tool_write(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"(written {len(content)} chars to {path})"
    except Exception as e:
        return f"(error writing {path}: {e})"


def tool_edit(path: str, old: str, new: str) -> str:
    """Replace a substring in a file."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"(file not found: {path})"
        text = p.read_text()
        if old not in text:
            return f"(old_string not found in {path})"
        p.write_text(text.replace(old, new, 1))
        return f"(edited {path}: 1 replacement)"
    except Exception as e:
        return f"(error editing {path}: {e})"


# Tool registry used by the native agent's tool-use loop.
AGENT_TOOLS = {
    "bash":  {"fn": tool_bash,  "args": ["command"]},
    "read":  {"fn": tool_read,  "args": ["path"]},
    "write": {"fn": tool_write, "args": ["path", "content"]},
    "edit":  {"fn": tool_edit,  "args": ["path", "old", "new"]},
}

AGENT_TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command and return stdout+stderr. Use for listing files, running tests, git, curl, etc. 30s timeout.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file's contents (truncated at 10000 chars). Use absolute paths or ~.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write",
        "description": "Create or overwrite a file with `content`. Parent directories are created automatically. Use for creating new files from scratch.",
        "parameters": {"type": "object", "properties": {
            "path":    {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace a substring in an existing file. Fails if `old` isn't present. Use for surgical edits instead of rewriting the whole file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old":  {"type": "string", "description": "Exact substring to replace"},
            "new":  {"type": "string", "description": "Replacement"},
        }, "required": ["path", "old", "new"]},
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
            state["caveman"] = level
            return f"Caveman mode: **{level}**" + (" (disabled)" if level == "off" else "")
        return f"Caveman mode: **{state.get('caveman', 'lite')}**. Usage: `/caveman off|lite|full|ultra`"

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
        lines = [f"**Engine Status:**", f"- Loaded tiers: {', '.join(loaded)}", f"- VRAM: {vram:.1f} GB"]
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
) -> None:
    """Run the interactive coding agent."""
    state = {
        "tier": tier,
        "caveman": "lite",
        "messages": [],
    }

    # Banner
    console.print(Panel(
        "[bold cyan]Mio Agent[/bold cyan]\n"
        f"[dim]Tier: {tier} | Caveman: lite | /help for commands[/dim]",
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
) -> None:
    """Process a user message: build prompt, generate, run any tool calls
    the model emits, feed the results back, and repeat until the model
    stops calling tools (up to MAX_ROUNDS). Without this loop the model
    would just emit <tool_call>…</tool_call> tags as literal text and the
    file would never actually be written.
    """
    current_tier = state.get("tier", "large-moe")
    if current_tier in manager.loaded_tiers():
        engine = manager.get_engine(current_tier)

    # Build system prompt (caveman mode + hint that tools are real)
    caveman_level = state.get("caveman", "lite")
    system_prompt = AGENT_SYSTEM_PROMPT
    caveman_text = CAVEMAN_LEVELS.get(caveman_level, "")
    if caveman_text:
        system_prompt += "\n\n" + caveman_text

    # Initial messages
    current_messages = [{"role": "system", "content": system_prompt}]
    current_messages.extend(state.get("messages", []))
    current_messages.append({"role": "user", "content": user_input})
    # Persist the user turn early so history is consistent even if generation
    # is interrupted.
    state["messages"].append({"role": "user", "content": user_input})

    from mio.tool_calls import parse_tool_calls as _parse_tc

    MAX_ROUNDS = 5
    assistant_text_accum: list[str] = []

    for round_idx in range(MAX_ROUNDS):
        console.print("[bold green]Mio[/bold green]: ", end="")
        full_text = ""
        for chunk, metrics in engine.generate_stream(current_messages, tools=AGENT_TOOLS_SPEC):
            # Strip <tool_call> tags from live display so the raw XML doesn't
            # clutter the terminal while still streaming.
            console.print(chunk, end="", highlight=False)
            full_text += chunk
        console.print()

        # Metrics line
        m = engine.last_metrics
        if m.generation_tps > 0:
            console.print(
                f"[dim]  {m.generation_tps:.1f} tok/s · "
                f"{m.completion_tokens} tokens · "
                f"{m.total_time_s:.2f}s[/dim]"
            )

        # Extract tool calls (OpenAI-format: {function: {name, arguments}})
        import json as _json, re as _re
        _leading, tool_calls = _parse_tc(full_text)
        visible_text = _re.sub(r"<tool_call>[\s\S]*?</tool_call>\s*", "", full_text).strip()
        assistant_text_accum.append(visible_text)

        if not tool_calls:
            break  # model stopped calling tools

        current_messages = list(current_messages) + [
            {"role": "assistant", "content": full_text},
        ]

        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = _json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except Exception:
                args = {}
            spec = AGENT_TOOLS.get(name)
            if not spec:
                result = f"(unknown tool: {name})"
            else:
                try:
                    kwargs = {k: args[k] for k in spec["args"] if k in args}
                    result = spec["fn"](**kwargs)
                except Exception as e:
                    result = f"(tool error: {type(e).__name__}: {e})"
            preview = ", ".join(repr(args.get(k, ""))[:40] for k in (spec["args"] if spec else []))
            console.print(f"[dim cyan]  ◆ {name}({preview}) → {str(result)[:120]}[/dim cyan]")
            current_messages.append({
                "role": "user",
                "content": f"<tool_response name=\"{name}\">{result}</tool_response>",
            })

    console.print()
    # Persist the final assistant text (joined across rounds) so multi-turn
    # history stays sensible.
    state["messages"].append({
        "role": "assistant",
        "content": "\n\n".join(t for t in assistant_text_accum if t) or "(tool-only turn)",
    })

    # Trim history (keep last 40 entries — ~20 exchanges)
    if len(state["messages"]) > 40:
        state["messages"] = state["messages"][-40:]
