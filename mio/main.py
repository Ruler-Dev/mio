"""Mio CLI entry point."""

from __future__ import annotations

import argparse


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mio",
        description="Fast local MLX inference with DFlash + TurboQuant",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_parser.add_argument(
        "--port", type=int, default=None,
        help="Server port (default: persisted config, otherwise 9090)",
    )
    serve_parser.add_argument(
        "--host", type=str, default=None,
        help="Server host (default: persisted config, otherwise 0.0.0.0)",
    )
    serve_parser.add_argument("--tandem", action="store_true", help="Load all tiers for tandem routing")
    serve_parser.add_argument("--tiers", type=str, default=None, help="Comma-separated tiers to load (default: large)")
    serve_parser.add_argument(
        "--tier", type=str, default=None,
        help="Single tier to load (default: persisted active tier, otherwise large-moe)",
    )
    serve_parser.add_argument("--validate", action="store_true", help="Enable auto-validation of generated code")
    serve_parser.add_argument(
        "--caveman",
        choices=["off", "lite", "full", "ultra"],
        default="full",
        help="Caveman system-prompt level injected into every request (default: full). "
             "Safe alongside OpenAI function tools — Qwen's chat template emits exact "
             "tool names, and caveman only shortens narrative prose. Skipped if the "
             "system prompt declares a Cline/Roo XML protocol (exact-tag match).",
    )
    serve_parser.add_argument("--tq4", action="store_true", help="Enable TurboQuant 4-bit KV cache (default: off)")
    serve_parser.add_argument("--mpath", type=int, default=1, help="Batched Multi-Path DFlash paths K (1 = vanilla DFlash, 2-4 typical)")
    serve_parser.add_argument(
        "--context", type=str, default=None,
        help="Override context window for the loaded tier(s). "
             "Accepts '8k', '16k', '32k', '64k', '128k', '256k', or a raw integer. "
             "Larger contexts cost more memory; smaller contexts free memory for other apps.",
    )
    serve_parser.add_argument("--compact-threshold", type=float, default=0.75,
                              help="Compact messages when prompt > this fraction of context window (default 0.75, 1.0 to disable)")
    serve_parser.add_argument("--compact-target", type=float, default=0.50,
                              help="Compact down to this fraction of context window (default 0.50)")
    serve_parser.add_argument("--no-compact-summarize", action="store_true",
                              help="Disable stage-2 LLM summarization — use only heuristic tool-result truncation")
    serve_parser.add_argument("--webui", action="store_true",
                              help="Enable Mio UI web interface at /ui (disabled by default)")

    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Interactive chat (no tools, no agent)")
    chat_parser.add_argument(
        "--tier", type=str, default=None,
        help="Model tier (default: persisted active tier, otherwise large-moe)",
    )
    chat_parser.add_argument("--paro", action="store_true", help="Use PARO quantized models (higher quality, slower)")
    chat_parser.add_argument("--no-caveman", action="store_true", help="Disable the Caveman Ultra system prompt")
    chat_parser.add_argument("--tq4", action="store_true", help="Enable TurboQuant 4-bit KV cache (default: off)")
    chat_parser.add_argument("--mpath", type=int, default=1, help="Batched Multi-Path DFlash paths K (1 = vanilla DFlash)")
    chat_parser.add_argument(
        "--context", type=str, default=None,
        help="Override context window: '8k', '32k', '128k', etc., or a raw integer.",
    )

    # --- download ---
    dl_parser = subparsers.add_parser("download", help="Download model weights from HuggingFace")
    dl_parser.add_argument("--tier", type=str, default=None, help="Specific tier to download (default: all)")

    # --- pull ---
    pull_parser = subparsers.add_parser("pull", help="Download target + DFlash draft for a tier")
    pull_parser.add_argument(
        "model_key",
        nargs="?",
        default=None,
        help="Tier name (large-moe|large|medium|small) or raw model key. "
             "Run without args to list everything.",
    )

    # --- configure ---
    subparsers.add_parser("configure", help="Interactive model + DFlash + TurboQuant configuration")

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

    # --- Top-level flags for agent mode ---
    parser.add_argument("--tandem", action="store_true", help="Agent mode with tandem routing")
    parser.add_argument(
        "--tier", type=str, default=None,
        help="Agent mode tier (default: persisted active tier, otherwise large-moe)",
    )
    parser.add_argument("--paro", action="store_true", help="Use PARO quantized models (higher quality, slower)")
    parser.add_argument("--port", type=int, default=None, help="API port for agent mode")
    parser.add_argument("--tq4", action="store_true", help="Enable TurboQuant 4-bit KV cache (default: off)")
    parser.add_argument("--mpath", type=int, default=1, help="Batched Multi-Path DFlash paths K (1 = vanilla DFlash)")
    parser.add_argument(
        "--context", type=str, default=None,
        help="Override context window for agent mode: '8k', '32k', '128k', etc., or a raw integer.",
    )
    parser.add_argument("prompt", nargs="*", default=[], help="Initial prompt for agent mode")

    args = parser.parse_args()

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
    caveman_level = getattr(args, "caveman", "full")
    start_server(
        manager,
        host=config.host,
        port=config.port,
        tandem=config.tandem,
        validate=validate,
        caveman_level=caveman_level,
        compact_threshold=float(getattr(args, "compact_threshold", 0.75)),
        compact_target=float(getattr(args, "compact_target", 0.50)),
        compact_summarize=not getattr(args, "no_compact_summarize", False),
        webui=getattr(args, "webui", False),
    )


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

    use_caveman = not getattr(args, "no_caveman", False)
    messages: list[dict] = []
    if use_caveman:
        from mio.agent import CAVEMAN_LITE
        messages.append({"role": "system", "content": CAVEMAN_LITE.strip()})
    console.print(
        f"[green]Ready.[/green] Caveman: {'lite' if use_caveman else 'off'}. "
        "Type 'quit' to exit.\n"
    )
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
            f"[dim]  {m.generation_tps:.1f} tok/s | "
            f"{m.completion_tokens} tokens | "
            f"{m.total_time_s:.2f}s[/dim]\n"
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
    """Pull a model (target + DFlash draft)."""
    from mio.pull import list_available, pull_model

    if args.model_key:
        pull_model(args.model_key)
    else:
        list_available()


def _cmd_configure(args) -> None:
    """Interactive model + DFlash + TurboQuant configuration."""
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

    manager = ModelManager(config)
    console_import = __import__("rich.console", fromlist=["Console"])
    c = console_import.Console()
    c.print(f"[dim]Loading {tier} tier...[/dim]")
    manager.load_active_tiers()

    initial = " ".join(args.prompt) if args.prompt else None
    run_agent(config, manager, tier=tier, initial_prompt=initial)
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
            args.host = "0.0.0.0"
            args.tiers = None
            _cmd_serve(args)
            break
        elif choice == "4":
            args.tandem = True
            args.host = "0.0.0.0"
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
