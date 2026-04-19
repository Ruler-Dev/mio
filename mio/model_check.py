"""Check which models are downloaded and ready to use."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table


def check_all_models() -> None:
    """Print status of all required models."""
    from mio.models.registry import DEFAULT_TIERS, models_dir, spd_dir

    console = Console()

    console.print(f"[dim]models/ : {models_dir()}[/dim]")
    console.print(f"[dim]spd/    : {spd_dir()}[/dim]\n")

    table = Table(title="Mio Model Status")
    table.add_column("Tier", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Resolved Path")
    table.add_column("Status", style="bold")

    all_ready = True
    for tier_name, tier_config in DEFAULT_TIERS.items():
        for label, path_str in [("target", tier_config.target_model), ("draft", tier_config.draft_model)]:
            p = Path(path_str)
            if p.exists() and (p / "config.json").exists():
                status = "[green]LOCAL[/green]"
                display = str(p)
            else:
                # Check if it's a HF repo that's cached
                available, hf_path = _check_hf_cache(path_str)
                if available:
                    status = "[yellow]HF CACHE[/yellow]"
                    display = path_str
                else:
                    status = "[red]MISSING[/red]"
                    display = path_str
                    all_ready = False

            table.add_row(tier_name, label, display, status)

    console.print(table)

    if all_ready:
        console.print("\n[green bold]All default tier models are ready.[/green bold]")
    else:
        console.print("\n[yellow]Some models are missing. Run:[/yellow]")
        console.print("  [bold]python scripts/download_all.py[/bold]")


def _check_hf_cache(repo_id: str) -> tuple[bool, str]:
    """Check if a HF repo is in the local cache."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    return True, str(rev.snapshot_path)
        return False, ""
    except Exception:
        return False, ""


if __name__ == "__main__":
    check_all_models()
