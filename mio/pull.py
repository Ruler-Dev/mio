"""Model pull: download target + DFlash draft in one command."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from mio.models.registry import KNOWN_MODELS, _model_path_is_complete, models_dir, spd_dir

console = Console()


# Tier names users actually type (matches DEFAULT_TIERS keys). Qwen 3.6 target
# and DFlash repositories are public, so new pulls use the current generation.
TIER_TO_MODEL_KEY: dict[str, str] = {
    "large-moe": "qwen3.6-35b-a3b-unsloth",
    "large":     "qwen3.6-27b-unsloth",
    "medium":    "qwen3.5-9b-unsloth",
    "small":     "qwen3.5-4b-4bit",
}


def list_available() -> None:
    """Print all pullable tiers + raw model keys."""
    console.print("[bold]Tiers (recommended):[/bold]\n")
    for tier, key in TIER_TO_MODEL_KEY.items():
        entry = KNOWN_MODELS[key]
        console.print(f"  [cyan]{tier:12s}[/cyan] -> {key}  ({entry.description})")
    console.print()
    console.print("[bold]All model keys:[/bold]\n")
    for key, entry in KNOWN_MODELS.items():
        console.print(f"  [dim]{key:32s}[/dim] {entry.description}")
    console.print("\nUsage: [bold]mio pull <tier|model-key>[/bold]")
    console.print("Example: [bold]mio pull large-moe[/bold]")


def pull_model(key: str) -> bool:
    """Download target + DFlash draft for a tier name or raw model key.

    Returns True on success.
    """
    requested = key
    if key in TIER_TO_MODEL_KEY:
        key = TIER_TO_MODEL_KEY[key]
        console.print(f"[dim]Tier '{requested}' -> model '{key}'[/dim]")

    entry = KNOWN_MODELS.get(key)
    if not entry:
        console.print(f"[red]Unknown tier or model key: {requested}[/red]\n")
        list_available()
        return False

    target_dir = models_dir() / entry.target_local
    draft_dir = spd_dir() / entry.draft_local

    target_exists = _model_path_is_complete(target_dir)
    draft_exists = _model_path_is_complete(draft_dir)

    if target_exists and draft_exists:
        console.print(f"[green]Already downloaded:[/green] {key}")
        console.print(f"  Target: {target_dir}")
        console.print(f"  Draft:  {draft_dir}")
        return True

    from huggingface_hub import snapshot_download

    console.print(f"[bold]Pulling {key}[/bold]")
    console.print(f"  Target: {entry.target_repo}")
    console.print(f"  Draft:  {entry.draft_repo}")
    console.print()

    success = True
    success &= _snapshot_into(entry.target_repo, target_dir, "target", target_exists, snapshot_download)
    success &= _snapshot_into(entry.draft_repo, draft_dir, "draft", draft_exists, snapshot_download)

    if success:
        console.print(f"\n[green bold]Pull complete: {key}[/green bold]")
        console.print("Run [bold]mio[/bold] to launch the agent, or [bold]mio serve[/bold] for the API.")
    return success


def _snapshot_into(repo: str, dest_dir: Path, kind: str, already: bool, snapshot_download) -> bool:
    """Download a HF checkpoint directly into ``dest_dir`` with resume support."""
    if already:
        console.print(f"  {kind.capitalize()} already exists: {dest_dir}")
        return True

    console.print(f"Downloading {kind}: {repo}")
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(dest_dir))
        if not _model_path_is_complete(dest_dir):
            console.print(f"  [red]Download incomplete: missing one or more weight shards in {dest_dir}[/red]")
            return False
        console.print(f"  [green]{kind.capitalize()} saved to {dest_dir}[/green]")
        return True
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False
