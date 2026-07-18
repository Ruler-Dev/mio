"""Mio CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def _add_prompt_policy_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_no_caveman: bool = False,
    dest_prefix: str = "",
) -> None:
    """Add the modern prompt-policy flags and backwards-compatible aliases."""
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--prompt-mode",
        dest=f"{dest_prefix}prompt_mode",
        choices=["none", "caveman", "ponytail"],
        default=None,
        help="System prompt policy (default: caveman/full).",
    )
    selector.add_argument(
        "--caveman",
        dest=f"{dest_prefix}caveman",
        choices=["off", "lite", "full", "ultra"],
        default=None,
        help="Legacy Caveman selector; 'off' maps to --prompt-mode none.",
    )
    selector.add_argument(
        "--ponytail",
        dest=f"{dest_prefix}ponytail",
        choices=["lite", "full", "ultra"],
        default=None,
        help="Ponytail engineering policy level.",
    )
    if include_no_caveman:
        selector.add_argument(
            "--no-caveman",
            dest=f"{dest_prefix}no_caveman",
            action="store_true",
            help="Legacy alias for --prompt-mode none.",
        )
    parser.add_argument(
        "--prompt-level",
        dest=f"{dest_prefix}prompt_level",
        choices=["lite", "full", "ultra"],
        default=None,
        help="Level for --prompt-mode caveman or ponytail (default: full).",
    )


def _resolve_prompt_policy_args(args, *, dest_prefix: str = ""):
    """Resolve CLI flags without importing policy code during shell completion."""
    from mio.prompt_policy import PromptMode, PromptPolicy

    if getattr(args, f"{dest_prefix}no_caveman", False):
        return PromptPolicy.resolve(
            prompt_mode=PromptMode.NONE,
            prompt_level=getattr(args, f"{dest_prefix}prompt_level", None),
        )
    return PromptPolicy.resolve(
        prompt_mode=getattr(args, f"{dest_prefix}prompt_mode", None),
        prompt_level=getattr(args, f"{dest_prefix}prompt_level", None),
        caveman=getattr(args, f"{dest_prefix}caveman", None),
        ponytail=getattr(args, f"{dest_prefix}ponytail", None),
    )


def _apply_tq4_flag(config, tier_names: list[str], enabled: bool) -> None:
    """Select TQ4 (and disable the mutually exclusive PQ cache)."""
    if not enabled:
        return
    for name in tier_names:
        if name in config.tiers:
            tier = config.tiers[name]
            tier.tq_bits = 4
            tier.pq_bits = 16
            tier.tq_use_rotation = True
            tier.tq_use_normalization = True


def _apply_mpath_flag(config, tier_names: list[str], K: int) -> None:
    """Set bmp_paths=K on the specified tiers when --mpath N is passed."""
    if K is None or K <= 1:
        return
    for name in tier_names:
        if name in config.tiers:
            config.tiers[name].bmp_paths = int(K)


def _parse_context(value: str | None) -> int | None:
    """Accept "32k", "128K", "131072", "128_000" — return int or None."""
    if value is None:
        return None
    s = str(value).strip().lower().replace(",", "").replace("_", "")
    if not s:
        return None
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def _apply_context_flag(config, tier_names: list[str], ctx: int | None) -> None:
    """Override context_window (and clamp max_output_tokens) for the given tiers."""
    if not ctx:
        return
    for name in tier_names:
        if name in config.tiers:
            config.tiers[name].context_window = ctx
            config.tiers[name].max_output_tokens = min(ctx // 4, 8192)


def _configured_tier_name(config, requested: str | None = None) -> str:
    """Resolve a CLI tier, otherwise use the first persisted active tier."""
    if requested:
        return requested
    for name in config.active_tiers:
        if name in config.tiers:
            return name
    if "large-moe" in config.tiers:
        return "large-moe"
    return next(iter(config.tiers), "large-moe")


def _agent_workspace_root(
    requested: str | None,
    *,
    unsafe_broad: bool,
) -> Path:
    """Resolve a narrow agent root, preferring the nearest VCS boundary."""

    from mio.agent_policy import is_broad_workspace_root

    candidate = Path(requested).expanduser() if requested else Path.cwd()
    candidate = candidate.resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError(f"agent workspace is not a directory: {candidate}")
    if requested is None:
        for directory in (candidate, *candidate.parents):
            if (directory / ".git").exists():
                candidate = directory
                break

    if is_broad_workspace_root(candidate) and not unsafe_broad:
        raise ValueError(
            f"refusing broad agent workspace {candidate}; choose --workspace PROJECT "
            "or acknowledge it with --unsafe-broad-workspace"
        )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mio",
        description="Local MLX inference and agent stack with DSpark/DFlash",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: persisted config, otherwise 9090)",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: persisted config, otherwise 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--unsafe-remote-bind",
        action="store_true",
        help=(
            "Allow a non-loopback bind without disabling Host/origin checks. A concrete bind "
            "trusts only that Host:port; a wildcard bind trusts private numeric LAN addresses "
            "on --port. Use MIO_TRUSTED_HOSTS for LAN hostnames and MIO_CORS_ORIGINS for an "
            "exact comma-separated cross-origin browser allowlist (wildcards are rejected). "
            "Mio has no built-in HTTP authentication."
        ),
    )
    serve_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Terminate processes already listening on --port before startup (never done by default).",
    )
    serve_parser.add_argument("--tandem", action="store_true", help="Load all tiers for tandem routing")
    serve_parser.add_argument("--tiers", type=str, default=None, help="Comma-separated tiers to load (default: large)")
    serve_parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Single tier to load (default: persisted active tier, otherwise large-moe)",
    )
    serve_parser.add_argument("--validate", action="store_true", help="Enable auto-validation of generated code")
    _add_prompt_policy_arguments(serve_parser)
    serve_parser.add_argument("--tq4", action="store_true", help="Enable TurboQuant 4-bit KV cache (default: off)")
    serve_parser.add_argument(
        "--mpath", type=int, default=1, help="Batched Multi-Path DFlash paths K (1 = vanilla DFlash, 2-4 typical)"
    )
    serve_parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Override context window for the loaded tier(s). "
        "Accepts '8k', '16k', '32k', '64k', '128k', '256k', or a raw integer. "
        "Larger contexts cost more memory; smaller contexts free memory for other apps.",
    )
    serve_parser.add_argument(
        "--compact-threshold",
        type=float,
        default=0.75,
        help="Compact messages when prompt > this fraction of context window (default 0.75, 1.0 to disable)",
    )
    serve_parser.add_argument(
        "--compact-target",
        type=float,
        default=0.50,
        help="Compact down to this fraction of context window (default 0.50)",
    )
    serve_parser.add_argument(
        "--no-compact-summarize",
        action="store_true",
        help="Disable stage-2 LLM summarization — use only heuristic tool-result truncation",
    )
    serve_parser.add_argument(
        "--webui", action="store_true", help="Enable Mio UI web interface at /ui (disabled by default)"
    )
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Mio MCP config (default: ~/.mio/mcp.json or MIO_MCP_CONFIG).",
    )

    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Interactive chat (no tools, no agent)")
    chat_parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Model tier (default: persisted active tier, otherwise large-moe)",
    )
    chat_parser.add_argument("--paro", action="store_true", help="Use PARO quantized models (higher quality, slower)")
    _add_prompt_policy_arguments(chat_parser, include_no_caveman=True)
    chat_parser.add_argument("--tq4", action="store_true", help="Enable TurboQuant 4-bit KV cache (default: off)")
    chat_parser.add_argument(
        "--mpath", type=int, default=1, help="Batched Multi-Path DFlash paths K (1 = vanilla DFlash)"
    )
    chat_parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Override context window: '8k', '32k', '128k', etc., or a raw integer.",
    )

    # --- download ---
    dl_parser = subparsers.add_parser("download", help="Download model weights from HuggingFace")
    dl_parser.add_argument("--tier", type=str, default=None, help="Specific tier to download (default: all)")

    # --- pull ---
    pull_parser = subparsers.add_parser(
        "pull",
        help="Download target + preferred DSpark + DFlash fallback for a tier",
    )
    pull_parser.add_argument(
        "model_key",
        nargs="?",
        default=None,
        help="Tier name (large-moe|large|medium|small) or raw model key. Run without args to list everything.",
    )
    pull_parser.add_argument(
        "--no-dspark",
        action="store_true",
        help="Skip the preferred DSpark drafter and pull only target/fallback.",
    )
    pull_parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Skip downloading the compatible DFlash fallback for this pull.",
    )

    # --- configure ---
    subparsers.add_parser("configure", help="Interactive model, speculation, and KV-cache configuration")

    # --- batch ---
    batch_parser = subparsers.add_parser("batch", help="Batch inference from JSONL file")
    batch_parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    batch_parser.add_argument("--output", type=str, default="results.jsonl", help="Output JSONL file")
    batch_parser.add_argument("--tier", type=str, default="large-moe", help="Model tier (default: large-moe)")

    # --- bench ---
    subparsers.add_parser("bench", help="Run inference benchmarks")

    # --- status ---
    subparsers.add_parser("status", help="Show engine status")

    # --- menu ---
    subparsers.add_parser("menu", help="Interactive menu")

    # --- mcp ---
    mcp_parser = subparsers.add_parser("mcp", help="Inspect or configure Mio-owned MCP servers")
    mcp_parser.add_argument(
        "mcp_action",
        nargs="?",
        choices=[
            "list",
            "enable",
            "disable",
            "tools",
            "call",
            "install-tools",
            "check",
            "doctor",
        ],
        default="list",
    )
    mcp_parser.add_argument("name", nargs="?", help="MCP server name for enable/disable")
    mcp_parser.add_argument("tool", nargs="?", help="Tool name for the call action")
    mcp_parser.add_argument("--config", type=str, default=None, help="Override ~/.mio/mcp.json")
    mcp_parser.add_argument("--args", dest="arguments_json", default="{}", help="JSON object for mcp call")
    mcp_parser.add_argument(
        "--grant",
        action="append",
        default=[],
        choices=["read", "write", "process", "network", "filesystem_read", "filesystem_write", "secrets"],
        help="Explicit permission for remote/authenticated MCPs (repeatable).",
    )
    mcp_parser.add_argument("--allow-remote", action="store_true", help="Allow a remote MCP for this command only")
    mcp_parser.add_argument(
        "--allow-auth", action="store_true", help="Allow credential injection for this command only"
    )
    mcp_parser.add_argument(
        "--mio-home",
        type=str,
        default=None,
        help="Override $MIO_HOME for install-tools/check/doctor.",
    )
    mcp_parser.add_argument(
        "--force",
        action="store_true",
        help="Build a fresh pinned MCP-tool release even when the active release is valid.",
    )
    mcp_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable report for install-tools/check/doctor.",
    )

    # --- Top-level flags for agent mode ---
    parser.add_argument("--tandem", action="store_true", help="Agent mode with tandem routing")
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Agent mode tier (default: persisted active tier, otherwise large-moe)",
    )
    parser.add_argument("--paro", action="store_true", help="Use PARO quantized models (higher quality, slower)")
    parser.add_argument("--port", type=int, default=None, help="API port for agent mode")
    parser.add_argument("--tq4", action="store_true", help="Enable TurboQuant 4-bit KV cache (default: off)")
    parser.add_argument("--mpath", type=int, default=1, help="Batched Multi-Path DFlash paths K (1 = vanilla DFlash)")
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Override context window for agent mode: '8k', '32k', '128k', etc., or a raw integer.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Agent workspace root (default: nearest Git root, otherwise current directory).",
    )
    parser.add_argument(
        "--agent-root",
        action="append",
        default=[],
        help="Additional agent workspace root (repeatable; trusted caller grant).",
    )
    parser.add_argument(
        "--agent-network",
        action="store_true",
        help="Allow network access from the native agent shell/MCP policy for this session.",
    )
    parser.add_argument(
        "--unsafe-broad-workspace",
        action="store_true",
        help="Allow /, home, or another broad system root as the agent workspace.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "ultra"],
        default=None,
        help="Mandatory coding-quality gate effort (default: persisted medium).",
    )
    _add_prompt_policy_arguments(parser, dest_prefix="agent_")
    parser.add_argument("prompt", nargs="*", default=[], help="Initial prompt for agent mode")

    args = parser.parse_args()

    if args.command in {"serve", "chat"}:
        try:
            args.prompt_policy = _resolve_prompt_policy_args(args)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.command is None:
        try:
            args.prompt_policy = _resolve_prompt_policy_args(args, dest_prefix="agent_")
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "download":
        _cmd_download(args)
    elif args.command == "pull":
        _cmd_pull(args)
    elif args.command == "configure":
        _cmd_configure(args)
    elif args.command == "batch":
        _cmd_batch(args)
    elif args.command == "bench":
        _cmd_bench(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "menu":
        _cmd_menu(args)
    elif args.command == "mcp":
        _cmd_mcp(args)
    elif args.command is None:
        # No subcommand → launch native coding agent
        _cmd_native_agent(args)


def _cmd_serve(args) -> None:
    """Start the API server."""
    from mio.config import load_config
    from mio.model_manager import ModelManager
    from mio.server import start_server

    config = load_config()
    if args.port is not None:
        config.port = args.port
    if args.host is not None:
        config.host = args.host

    if args.tandem:
        config.active_tiers = list(config.tiers.keys())
        config.tandem = True
    elif args.tiers:
        config.active_tiers = [t.strip() for t in args.tiers.split(",")]
        config.tandem = False
    elif args.tier:
        config.active_tiers = [args.tier]
        config.tandem = False

    _apply_tq4_flag(config, config.active_tiers, getattr(args, "tq4", False))
    _apply_mpath_flag(config, config.active_tiers, getattr(args, "mpath", 1))
    _apply_context_flag(config, config.active_tiers, _parse_context(getattr(args, "context", None)))

    manager = ModelManager(config)
    manager.load_active_tiers()
    validate = getattr(args, "validate", False)
    prompt_policy = getattr(args, "prompt_policy", None) or _resolve_prompt_policy_args(args)
    from mio.mcp import load_registry

    mcp_registry = load_registry(getattr(args, "mcp_config", None))
    start_server(
        manager,
        host=config.host,
        port=config.port,
        tandem=config.tandem,
        validate=validate,
        prompt_policy=prompt_policy,
        compact_threshold=float(getattr(args, "compact_threshold", 0.75)),
        compact_target=float(getattr(args, "compact_target", 0.50)),
        compact_summarize=not getattr(args, "no_compact_summarize", False),
        webui=getattr(args, "webui", False),
        mcp_registry=mcp_registry,
        unsafe_remote_bind=getattr(args, "unsafe_remote_bind", False),
        replace_existing=getattr(args, "replace_existing", False),
    )


def _cmd_mcp(args) -> None:
    """List or toggle Mio's MCP providers without starting or calling them."""
    from rich.console import Console
    from rich.table import Table

    from mio.mcp import load_registry

    console = Console()
    action = getattr(args, "mcp_action", "list")
    name = getattr(args, "name", None)
    if action in {"install-tools", "check", "doctor"}:
        import json

        from mio.mcp.tool_installer import (
            InstallerError,
            check_installation,
            install_mcp_tools,
        )

        home = getattr(args, "mio_home", None)
        try:
            if action == "install-tools":
                report = install_mcp_tools(
                    home,
                    force=bool(getattr(args, "force", False)),
                    progress=lambda message: console.print(f"[dim]{message}[/dim]"),
                )
            else:
                report = check_installation(home)
        except InstallerError as exc:
            report = {"ok": False, "mode": action, "errors": [str(exc)]}

        if getattr(args, "json", False):
            print(json.dumps(report, indent=2, sort_keys=True))
        elif report.get("ok"):
            console.print(f"[green]Mio MCP tools OK[/green] — {report.get('release', 'active release')}")
        else:
            console.print("[red]Mio MCP tool check failed[/red]")
            for error in report.get("errors", []):
                console.print(f"  - {error}")
        if not report.get("ok"):
            raise SystemExit(1)
        return

    registry = load_registry(getattr(args, "config", None))
    if action in {"enable", "disable"}:
        if not name:
            raise SystemExit(f"mio mcp {action} requires a server name")
        config = registry.set_enabled(name, action == "enable")
        path = registry.save()
        console.print(f"[green]{config.name}[/green]: {'enabled' if config.enabled else 'disabled'} ({path})")
        return

    if action in {"tools", "call"}:
        import json

        from mio.mcp import MCPError, MCPHub, MCPHubPolicy, MCPPermission

        if not name:
            raise SystemExit(f"mio mcp {action} requires a server name")
        grants = frozenset(MCPPermission(value) for value in getattr(args, "grant", []))
        policy = MCPHubPolicy(
            allow_remote=bool(getattr(args, "allow_remote", False)),
            allow_authenticated=bool(getattr(args, "allow_auth", False)),
            explicit_grants={name: grants},
        )
        hub = MCPHub(registry, policy=policy)
        try:
            if action == "tools":
                result = hub.list_tools(name)
            else:
                tool = getattr(args, "tool", None)
                if not tool:
                    raise SystemExit("mio mcp call requires SERVER and TOOL")
                try:
                    arguments = json.loads(getattr(args, "arguments_json", "{}"))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid --args JSON: {exc}") from exc
                if not isinstance(arguments, dict):
                    raise SystemExit("--args must decode to a JSON object")
                result = hub.call_tool(name, tool, arguments)
            console.print_json(data=result)
        except MCPError as exc:
            raise SystemExit(f"MCP error: {exc}") from exc
        finally:
            hub.close()
        return

    table = Table(title=f"Mio MCP — {registry.config_path}")
    table.add_column("server")
    table.add_column("transport")
    table.add_column("scope")
    table.add_column("enabled")
    table.add_column("permissions")
    for config in registry.list():
        table.add_row(
            config.name,
            config.transport.value,
            "local" if config.is_local else "remote",
            "yes" if config.enabled else "no",
            ", ".join(sorted(permission.value for permission in config.permissions)),
        )
    console.print(table)


def _cmd_chat(args) -> None:
    """Interactive chat loop."""
    from rich.console import Console
    from rich.prompt import Prompt

    from mio.config import load_config
    from mio.model_manager import ModelManager

    console = Console()
    config = load_config()
    if getattr(args, "paro", False):
        from dataclasses import replace

        from mio.models.registry import PARO_TIERS

        config.tiers = {name: replace(tier) for name, tier in PARO_TIERS.items()}
        console.print("[bold yellow]PARO mode[/bold yellow]")
    tier_name = _configured_tier_name(config, getattr(args, "tier", None))
    config.active_tiers = [tier_name]
    config.tandem = False
    _apply_tq4_flag(config, config.active_tiers, getattr(args, "tq4", False))
    _apply_mpath_flag(config, config.active_tiers, getattr(args, "mpath", 1))
    _apply_context_flag(config, config.active_tiers, _parse_context(getattr(args, "context", None)))

    manager = ModelManager(config)
    console.print(f"[bold]Loading {tier_name} tier...[/bold]")
    manager.load_active_tiers()
    engine = manager.get_engine(tier_name)

    from mio.prompt_policy import apply_prompt_policy

    prompt_policy = getattr(args, "prompt_policy", None) or _resolve_prompt_policy_args(args)
    messages: list[dict] = apply_prompt_policy([], prompt_policy)
    console.print(f"[green]Ready.[/green] Prompt policy: {prompt_policy.label}. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})

        console.print("[bold green]Assistant[/bold green]: ", end="")
        full_text = ""
        for chunk, metrics in engine.generate_stream(messages):
            console.print(chunk, end="")
            full_text += chunk
        console.print()

        messages.append({"role": "assistant", "content": full_text})

        # Show metrics
        m = engine.last_metrics
        console.print(
            f"[dim]  {m.generation_tps:.1f} tok/s | {m.completion_tokens} tokens | {m.total_time_s:.2f}s[/dim]\n"
        )

    manager.unload_all()


def _cmd_download(args) -> None:
    """Download model weights."""
    from rich.console import Console
    from rich.progress import Progress

    from mio.config import load_config
    from mio.menu import confirm_download

    console = Console()
    config = load_config()

    if args.tier:
        tiers_to_download = {args.tier: config.tiers[args.tier]}
    else:
        tiers_to_download = config.tiers

    models_to_download = []
    for name, tier in tiers_to_download.items():
        models_to_download.append(tier.target_model)
        models_to_download.append(tier.draft_model)
        if tier.draft_fallback_model:
            models_to_download.append(tier.draft_fallback_model)

    if not confirm_download(models_to_download):
        return

    from huggingface_hub import snapshot_download

    with Progress() as progress:
        task = progress.add_task("Downloading models...", total=len(models_to_download))
        for repo in models_to_download:
            console.print(f"  Downloading {repo}...")
            try:
                snapshot_download(repo)
                progress.advance(task)
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                progress.advance(task)

    console.print("[green]Download complete.[/green]")


def _cmd_batch(args) -> None:
    """Run batch inference."""
    from mio.batch import run_batch_cli

    run_batch_cli(args.input, args.output, tier=args.tier)


def _cmd_pull(args) -> None:
    """Pull a target plus its requested DSpark/DFlash stack."""
    from mio.pull import list_available, pull_model

    if args.model_key:
        success = pull_model(
            args.model_key,
            include_dspark=not getattr(args, "no_dspark", False),
            include_fallback=not getattr(args, "no_fallback", False),
        )
        if not success:
            raise SystemExit(1)
    else:
        list_available()


def _cmd_configure(args) -> None:
    """Configure model, speculative backend, and KV-cache settings."""
    from mio.config import load_config
    from mio.configure import configure_interactive

    config = load_config()
    configure_interactive(config)


def _cmd_bench(args) -> None:
    """Run benchmarks."""
    from rich.console import Console

    from mio.config import load_config
    from mio.model_manager import ModelManager

    console = Console()
    config = load_config()

    prompt = "Write a Python function to sort a list using quicksort. Include type hints and a docstring."
    messages = [{"role": "user", "content": prompt}]

    for tier_name in config.tiers:
        console.print(f"\n[bold]Benchmarking {tier_name} tier...[/bold]")
        config.active_tiers = [tier_name]
        manager = ModelManager(config)

        try:
            manager.load_active_tiers()
            engine = manager.get_engine(tier_name)

            # Warmup
            engine.generate(messages, max_tokens=32)

            # Benchmark
            text, metrics = engine.generate(messages, max_tokens=256)

            console.print(f"  Generation: [bold yellow]{metrics.generation_tps:.1f} tok/s[/bold yellow]")
            console.print(f"  Prompt:     {metrics.prompt_tps:.1f} tok/s")
            console.print(f"  E2E:        {metrics.end_to_end_tps:.1f} tok/s")
            console.print(f"  Acceptance: {metrics.avg_acceptance_length:.1f} avg tokens/step")
            console.print(f"  Memory:     {metrics.peak_memory_gb:.1f} GB")
            console.print(f"  Tokens:     {metrics.completion_tokens}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
        finally:
            manager.unload_all()


def _cmd_status(args) -> None:
    """Show status."""
    import urllib.request
    import json as json_mod

    from rich.console import Console

    from mio.menu import show_models_table, show_tier_config
    from mio.config import load_config

    console = Console()
    config = load_config()

    show_tier_config(config)
    console.print()
    show_models_table()

    # Try to reach running server
    try:
        req = urllib.request.Request(f"http://localhost:{config.port}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json_mod.loads(resp.read())
            console.print(f"\n[green]Server running[/green]: {data}")
    except Exception:
        console.print("\n[dim]No running server detected on port 9090[/dim]")


def _cmd_native_agent(args) -> None:
    """Launch the Mio coding agent."""
    from mio.agent import run_agent
    from mio.agent_policy import AgentToolPolicy
    from mio.config import load_config
    from mio.model_manager import ModelManager

    config = load_config()

    requested_tier = getattr(args, "tier", None)
    tier = _configured_tier_name(config, requested_tier)
    if args.tandem:
        config.active_tiers = list(config.tiers.keys())
        config.tandem = True
    elif requested_tier:
        config.active_tiers = [tier]
        config.tandem = False
    elif not config.active_tiers:
        config.active_tiers = [tier]

    _apply_tq4_flag(config, config.active_tiers, getattr(args, "tq4", False))
    _apply_mpath_flag(config, config.active_tiers, getattr(args, "mpath", 1))
    _apply_context_flag(config, config.active_tiers, _parse_context(getattr(args, "context", None)))

    unsafe_broad = bool(getattr(args, "unsafe_broad_workspace", False))
    workspace = _agent_workspace_root(
        getattr(args, "workspace", None),
        unsafe_broad=unsafe_broad,
    )
    additional_roots = tuple(
        _agent_workspace_root(root, unsafe_broad=unsafe_broad)
        for root in (getattr(args, "agent_root", None) or [])
    )
    tool_policy = AgentToolPolicy.coding_workspace(
        workspace,
        additional_roots=additional_roots,
        allow_network=bool(getattr(args, "agent_network", False)),
    )

    manager = ModelManager(config)
    console_import = __import__("rich.console", fromlist=["Console"])
    c = console_import.Console()
    capabilities = ", ".join(
        sorted(permission.value for permission in tool_policy.permissions)
    )
    roots = ", ".join(str(root) for root in tool_policy.workspace_roots)
    c.print(f"[dim]Agent policy: {capabilities}; roots: {roots}[/dim]")
    c.print(f"[dim]Loading {tier} tier...[/dim]")
    manager.load_active_tiers()

    initial = " ".join(args.prompt) if args.prompt else None
    run_agent(
        config,
        manager,
        tier=tier,
        initial_prompt=initial,
        prompt_policy=getattr(args, "prompt_policy", None),
        # Agent mode is an explicit coding trust boundary. Other run_agent()
        # callers remain read-only unless they declare their own policy.
        tool_policy=tool_policy,
        coding_effort=(
            getattr(args, "effort", None)
            or getattr(config, "coding_effort", "medium")
        ),
    )
    manager.unload_all()


def _cmd_menu(args) -> None:
    """Interactive menu."""
    from mio.config import load_config
    from mio.menu import interactive_menu

    config = load_config()

    while True:
        choice = interactive_menu(config)
        if not choice:
            break

        if choice == "q":
            break
        elif choice == "1":
            args.tier = "large"
            args.tandem = False
            _cmd_native_agent(args)
            break
        elif choice == "2":
            args.tandem = True
            _cmd_native_agent(args)
            break
        elif choice == "3":
            args.tier = "large"
            args.host = None
            args.tiers = None
            _cmd_serve(args)
            break
        elif choice == "4":
            args.tandem = True
            args.host = None
            args.tiers = None
            _cmd_serve(args)
            break
        elif choice == "5":
            args.tier = "large"
            _cmd_chat(args)
        elif choice == "6":
            _cmd_configure(args)
            # Reload config after configure
            config = load_config()
        elif choice == "7":
            from mio.menu import show_tier_config

            show_tier_config(config)
            input("\nPress Enter to continue...")
        elif choice == "8":
            from mio.menu import show_models_table

            show_models_table()
            input("\nPress Enter to continue...")
        elif choice == "9":
            args.tier = None
            _cmd_download(args)
        elif choice == "0":
            _cmd_bench(args)
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
