"""Model pull: download target + DFlash draft in one command."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from mio.models.registry import KNOWN_MODELS, models_dir, spd_dir

console = Console()


# Tier names users actually type (matches DEFAULT_TIERS keys).
# large-moe uses Qwen 3.5 by default — Qwen 3.6's draft repo is gated and
# can't be downloaded by the CLI without manual HF approval.
TIER_TO_MODEL_KEY: dict[str, str] = {
    "large-moe": "qwen3.5-35b-a3b-unsloth",
    "large":     "qwen3.5-27b-unsloth",
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
    console.print(f"\nUsage: [bold]mio pull <tier|model-key>[/bold]")
    console.print(f"Example: [bold]mio pull large-moe[/bold]")


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

    target_exists = target_dir.exists() and (target_dir / "config.json").exists()
    draft_exists = draft_dir.exists() and (draft_dir / "config.json").exists()

    if target_exists and draft_exists:
        console.print(f"[green]Already downloaded:[/green] {key}")
        console.print(f"  Target: {target_dir}")
        console.print(f"  Draft:  {draft_dir}")
        _print_36_note(requested)
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
        console.print(f"Run [bold]mio[/bold] to launch the agent, or [bold]mio serve[/bold] for the API.")
        _print_36_note(requested)
    return success


def _snapshot_into(repo: str, dest_dir: Path, kind: str, already: bool, snapshot_download) -> bool:
    """Download a HF repo and copy resolved files into dest_dir. Returns True on success."""
    if already:
        console.print(f"  {kind.capitalize()} already exists: {dest_dir}")
        return True

    console.print(f"Downloading {kind}: {repo}")
    try:
        cached = snapshot_download(repo)
        import shutil

        dest_dir.mkdir(parents=True, exist_ok=True)
        cached_path = Path(cached)
        for item in cached_path.iterdir():
            dest = dest_dir / item.name
            if item.is_file():
                shutil.copy2(item.resolve(), dest)
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, copy_function=shutil.copy2)
        if kind == "target" and not (dest_dir / "config.json").exists():
            console.print(f"  [yellow]Warning: config.json not found in {dest_dir}[/yellow]")
        console.print(f"  [green]{kind.capitalize()} saved to {dest_dir}[/green]")
        return True
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def _print_36_note(requested: str) -> None:
    """Print Qwen 3.6 upgrade note when the user pulled the large-moe tier."""
    if requested != "large-moe":
        return
    console.print(
        "\n[dim yellow]Note:[/dim yellow] [yellow]large-moe[/yellow] pulls Qwen 3.5 35B-A3B by default.\n"
        "Qwen 3.6 (faster, longer context) is available but its DFlash draft repo is\n"
        "gated and can't be fetched via this CLI without manual HuggingFace approval.\n"
        "To use 3.6: request access on HF, then download manually:\n"
        "  hf download Brooooooklyn/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx --local-dir models/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx\n"
        "  hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir spd/Qwen3.6-35B-A3B-DFlash"
    )
