"""Interactive CLI menu using Rich."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mio.config import MioConfig
from mio.models.registry import KNOWN_MODELS, SUPPORTED_ADAPTERS

console = Console()


def show_banner() -> None:
    """Display the Mio banner."""
    console.print(
        Panel(
            "[bold cyan]Mio v0.1.0[/bold cyan]\n"
            "[dim]DSpark + DFlash MLX Engine[/dim]\n"
            "[dim]Fast local inference for Apple Silicon[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def show_models_table() -> None:
    """Display all known models with their status."""
    table = Table(title="Known Model Registry")
    table.add_column("Model", style="cyan")
    table.add_column("Tier", style="green")
    table.add_column("Adapter", style="yellow")
    table.add_column("Speculation", style="bold")
    table.add_column("Description")

    for key, entry in KNOWN_MODELS.items():
        supported = entry.adapter in SUPPORTED_ADAPTERS
        if not supported:
            speculative_status = "[red]unsupported[/red]"
        elif entry.dspark_repo:
            speculative_status = "[green]DSpark + DFlash[/green]"
        else:
            speculative_status = "[green]DFlash[/green]"
        table.add_row(
            key,
            entry.default_tier,
            entry.adapter,
            speculative_status,
            entry.description,
        )

    console.print(table)


def show_tier_config(config: MioConfig) -> None:
    """Display current tier configuration."""
    table = Table(title="Active Tier Configuration")
    table.add_column("Tier", style="cyan")
    table.add_column("Target Model", style="green")
    table.add_column("Draft Model", style="yellow")
    table.add_column("Context", justify="right")
    table.add_column("TQ Bits", justify="right")
    table.add_column("Active", style="bold")

    for name, tier in config.tiers.items():
        active = "[green]YES[/green]" if name in config.active_tiers else "[dim]no[/dim]"
        table.add_row(
            name,
            tier.target_model.split("/")[-1],
            tier.draft_model.split("/")[-1],
            str(tier.context_window),
            str(tier.tq_bits),
            active,
        )

    console.print(table)


def show_status(status: dict) -> None:
    """Display server/engine status."""
    table = Table(title="Engine Status")
    table.add_column("Tier", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Gen tok/s", justify="right", style="bold yellow")
    table.add_column("Context", justify="right")

    for name, info in status.get("engines", {}).items():
        table.add_row(
            name,
            info["model"].split("/")[-1],
            f"{info['last_gen_tps']:.1f}",
            str(info["context_window"]),
        )

    console.print(table)
    console.print(f"VRAM: {status.get('vram_gb', 0):.1f} GB")
    console.print(f"Tandem: {'enabled' if status.get('tandem') else 'disabled'}")


def interactive_menu(config: MioConfig) -> str:
    """Show interactive menu and return user choice."""
    show_banner()
    console.print()

    options = {
        "1": "Start coding agent (large model)",
        "2": "Start coding agent (tandem mode)",
        "3": "Start API server (large model)",
        "4": "Start API server (tandem mode)",
        "5": "Interactive chat",
        "6": "Configure model + speculation + KV cache",
        "7": "Show tier config",
        "8": "Show model registry",
        "9": "Download models",
        "0": "Benchmark",
        "q": "Quit",
    }

    for key, desc in options.items():
        style = "bold" if key in ("1", "3") else ""
        console.print(f"  [{style}]{key}[/{style}]. {desc}")

    console.print()
    choice = Prompt.ask("Select", choices=list(options.keys()), default="1")
    return choice


def confirm_download(models: list[str]) -> bool:
    """Ask user to confirm model download."""
    console.print("\n[bold]Models to download:[/bold]")
    for m in models:
        console.print(f"  - {m}")
    console.print()
    return Prompt.ask("Download?", choices=["y", "n"], default="y") == "y"
