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
            status, display, ready = _model_status(path_str, label)
            all_ready = all_ready and ready

            table.add_row(tier_name, label, display, status)

    console.print(table)

    if all_ready:
        console.print("\n[green bold]All default tier models are ready.[/green bold]")
    else:
        console.print("\n[yellow]Some models are missing. Run:[/yellow]")
        console.print("  [bold]python scripts/download_all.py[/bold]")


def _find_local_candidate(path_str: str, kind: str) -> Path | None:
    """Find an existing local directory for a resolved path or HF repo ID."""
    from mio.models.registry import KNOWN_MODELS, models_dir, spd_dir

    direct = Path(path_str).expanduser()
    if direct.exists():
        return direct

    base = models_dir() if kind == "target" else spd_dir()
    repo_field = f"{kind}_repo"
    local_field = f"{kind}_local"
    for entry in KNOWN_MODELS.values():
        if path_str in (getattr(entry, repo_field), getattr(entry, local_field)):
            candidate = base / getattr(entry, local_field)
            if candidate.exists():
                return candidate

    candidate = base / Path(path_str).name
    return candidate if candidate.exists() else None


def _model_status(path_str: str, kind: str) -> tuple[str, str, bool]:
    """Return rich status, display path, and readiness for one checkpoint."""
    from mio.models.registry import _model_path_is_complete

    local_path = _find_local_candidate(path_str, kind)
    if local_path is not None and _model_path_is_complete(local_path):
        return "[green]LOCAL[/green]", str(local_path), True

    # A complete cache snapshot remains usable even if a local-dir download
    # was interrupted.  Incomplete snapshots are deliberately ignored below.
    available, hf_path = _check_hf_cache(path_str)
    if available:
        return "[yellow]HF CACHE[/yellow]", hf_path, True

    if local_path is not None:
        return "[red]INCOMPLETE[/red]", str(local_path), False
    return "[red]MISSING[/red]", path_str, False


def _check_hf_cache(repo_id: str) -> tuple[bool, str]:
    """Check if a HF repo has a complete snapshot in the local cache."""
    try:
        from huggingface_hub import scan_cache_dir
        from mio.models.registry import _model_path_is_complete

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    snapshot_path = Path(rev.snapshot_path)
                    if _model_path_is_complete(snapshot_path):
                        return True, str(snapshot_path)
        return False, ""
    except Exception:
        return False, ""


if __name__ == "__main__":
    check_all_models()
