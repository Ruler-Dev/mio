"""Interactive configuration: select model + DFlash + TurboQuant + context window."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from mio.config import MioConfig, TierConfig, load_config, save_config
from mio.models.registry import (
    KNOWN_MODELS,
    SUPPORTED_ADAPTERS,
    models_dir,
    spd_dir,
)

console = Console()


# --- VRAM Estimation ---

def _resolve_tq_selection(selected_bits: int) -> tuple[int, int, bool]:
    """Return canonical (TQ bits, PQ bits, TQ enabled) wizard settings."""
    if selected_bits in (2, 3, 4):
        return selected_bits, 16, True
    return 16, 16, False

def _get_model_arch(model_path: Path) -> dict:
    """Read architecture params from config.json."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        console.print(f"[yellow]Warning: no config.json in {model_path.name}, using default estimates[/yellow]")
        return {}
    with open(config_path) as f:
        c = json.load(f)
    # Qwen 3.5 nests under text_config
    return c.get("text_config", c)


def _estimate_weights_gb(model_path: Path) -> float:
    """Estimate model weight size from safetensors files."""
    return sum(f.stat().st_size for f in model_path.rglob("*.safetensors")) / (1024**3)


def _estimate_kv_cache_gb(
    arch: dict,
    context_tokens: int,
    tq_bits: int,
) -> float:
    """Estimate KV cache VRAM in GB.

    Qwen 3.5 uses hybrid attention: every 4th layer is full attention,
    the rest are linear attention (O(1) recurrent state, not growing).
    KV cache only grows with context for full attention layers.

    KV per token per full-attn layer = 2 (K+V) * n_kv_heads * head_dim * 2 bytes (fp16)
    With TurboQuant: multiply by (tq_bits / 16)
    """
    n_layers = arch.get("num_hidden_layers", 32)
    n_kv_heads = arch.get("num_key_value_heads", 4)
    head_dim = arch.get("head_dim", 256)

    # Count full attention layers (every 4th in Qwen 3.5)
    layer_types = arch.get("layer_types", [])
    if layer_types:
        n_full_attn = sum(1 for lt in layer_types if lt == "full_attention")
    else:
        n_full_attn = n_layers  # Fallback: assume all full attention

    # KV bytes per token for all full attention layers (fp16)
    kv_per_token_fp16 = 2 * n_full_attn * n_kv_heads * head_dim * 2  # bytes

    # Apply TQ compression only for an actual TurboQuant bit-width.  ``16``
    # is the canonical persisted sentinel for an uncompressed fp16 cache.
    if tq_bits in (2, 3, 4):
        kv_per_token = kv_per_token_fp16 * (tq_bits / 16)
    else:
        kv_per_token = kv_per_token_fp16

    total_bytes = kv_per_token * context_tokens
    return total_bytes / (1024**3)


def _estimate_total_vram(
    model_path: Path,
    draft_path: Path,
    arch: dict,
    context_tokens: int,
    tq_bits: int,
) -> float:
    """Estimate total VRAM: weights + draft + KV cache."""
    weights = _estimate_weights_gb(model_path)
    draft = _estimate_weights_gb(draft_path)
    kv = _estimate_kv_cache_gb(arch, context_tokens, tq_bits)
    overhead = 0.5  # MLX runtime, activations, etc.
    return weights + draft + kv + overhead


# --- Scanning ---

def _scan_local_models() -> list[tuple[str, str, Path]]:
    """Scan models/ for locally available target models.

    Returns list of (registry_key, display_name, path).
    """
    mdir = models_dir()
    if not mdir.exists():
        return []

    results = []
    for entry in sorted(mdir.iterdir()):
        if entry.is_dir() and (entry / "config.json").exists():
            for key, model in KNOWN_MODELS.items():
                if model.target_local == entry.name:
                    results.append((key, entry.name, entry))
                    break
            else:
                results.append(("custom", entry.name, entry))
    return results


def _scan_local_drafts() -> list[tuple[str, Path]]:
    """Scan spd/ for locally available DFlash drafts."""
    sdir = spd_dir()
    if not sdir.exists():
        return []

    results = []
    for entry in sorted(sdir.iterdir()):
        if entry.is_dir() and (entry / "config.json").exists():
            results.append((entry.name, entry))
    return results


def _find_matching_drafts(target_key: str) -> list[tuple[str, Path]]:
    """Find DFlash drafts that match a target model."""
    if target_key == "custom":
        return _scan_local_drafts()

    entry = KNOWN_MODELS.get(target_key)
    if not entry:
        return _scan_local_drafts()

    drafts = _scan_local_drafts()
    matching = [(name, path) for name, path in drafts if name == entry.draft_local]
    if matching:
        return matching
    return drafts


# --- Interactive wizard ---

def configure_interactive(config: MioConfig | None = None) -> MioConfig:
    """Run the interactive configuration wizard."""
    if config is None:
        config = load_config()

    console.print(Panel("[bold cyan]Mio Configuration[/bold cyan]", border_style="cyan"))
    console.print()

    # --- Step 1: Select target model ---
    console.print("[bold]Step 1: Select target model[/bold]\n")

    local_models = _scan_local_models()
    if not local_models:
        console.print("[red]No models found in models/ directory.[/red]")
        console.print("Run: mio download")
        return config

    table = Table()
    table.add_column("#", style="bold yellow", justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("Adapter", style="green")
    table.add_column("Weights", justify="right")

    for i, (key, name, path) in enumerate(local_models, 1):
        adapter = KNOWN_MODELS[key].adapter if key in KNOWN_MODELS else "unknown"
        supported = adapter in SUPPORTED_ADAPTERS
        adapter_display = f"[green]{adapter}[/green]" if supported else f"[red]{adapter} (needs impl)[/red]"
        size_gb = _estimate_weights_gb(path)
        table.add_row(str(i), name, adapter_display, f"{size_gb:.1f} GB")

    console.print(table)
    console.print()

    model_idx = IntPrompt.ask("Select target model", default=1) - 1
    model_idx = max(0, min(model_idx, len(local_models) - 1))
    selected_key, selected_name, selected_path = local_models[model_idx]

    console.print(f"  Selected: [cyan]{selected_name}[/cyan]\n")

    # --- Step 2: Select DFlash draft ---
    console.print("[bold]Step 2: Select DFlash draft[/bold]\n")

    drafts = _find_matching_drafts(selected_key)
    if not drafts:
        console.print("[red]No DFlash drafts found in spd/ directory.[/red]")
        console.print("Run: mio download")
        return config

    for i, (name, path) in enumerate(drafts, 1):
        size_gb = _estimate_weights_gb(path)
        console.print(f"  [{i}] {name}  ({size_gb:.1f} GB)")

    console.print()
    draft_idx = IntPrompt.ask("Select DFlash draft", default=1) - 1
    draft_idx = max(0, min(draft_idx, len(drafts) - 1))
    selected_draft_name, selected_draft_path = drafts[draft_idx]

    console.print(f"  Selected: [cyan]{selected_draft_name}[/cyan]\n")

    # --- Step 3: TurboQuant cache settings ---
    console.print("[bold]Step 3: TurboQuant V2 KV-Cache compression[/bold]\n")

    arch = _get_model_arch(selected_path)

    tq_options = [
        (4, "3.6x compression, best speed", "RECOMMENDED"),
        (3, "4.7x compression, moderate quality loss", ""),
        (2, "5.5x compression, aggressive", ""),
        (16, "no compression, full fp16 cache", ""),
    ]

    tq_table = Table()
    tq_table.add_column("#", style="bold yellow", justify="right")
    tq_table.add_column("Mode", style="cyan")
    tq_table.add_column("Compression", justify="right")
    tq_table.add_column("KV @ 32K", justify="right", style="green")
    tq_table.add_column("KV @ 128K", justify="right", style="green")
    tq_table.add_column("Note", style="yellow")

    for i, (bits, desc, note) in enumerate(tq_options, 1):
        enabled = bits in (2, 3, 4)
        label = f"{bits}-bit" if enabled else "OFF"
        compress = f"{16 / bits:.1f}x" if enabled else "1.0x"
        kv_32k = _estimate_kv_cache_gb(arch, 32768, bits)
        kv_128k = _estimate_kv_cache_gb(arch, 131072, bits)
        tq_table.add_row(
            str(i), label, compress,
            f"{kv_32k:.1f} GB", f"{kv_128k:.1f} GB",
            note,
        )

    console.print(tq_table)
    console.print()
    tq_idx = IntPrompt.ask("Select TQ cache bits", default=1) - 1
    tq_idx = max(0, min(tq_idx, len(tq_options) - 1))
    tq_bits, pq_bits, tq_enabled = _resolve_tq_selection(tq_options[tq_idx][0])

    if tq_enabled:
        console.print(f"  Selected: [cyan]{tq_bits}-bit[/cyan]\n")
    else:
        console.print("  Selected: [cyan]OFF[/cyan] (no cache compression)\n")

    # --- Step 4: Context window ---
    console.print("[bold]Step 4: Context window[/bold]\n")

    ctx_options = [
        (8192, "8K"),
        (16384, "16K"),
        (32768, "32K"),
        (65536, "64K"),
        (131072, "128K"),
        (138240, "135K"),
        (168960, "165K"),
        (184320, "180K"),
        (204800, "200K"),
        (230400, "225K"),
        (262144, "256K"),
        (271360, "265K"),
    ]

    model_weights_gb = _estimate_weights_gb(selected_path)
    draft_weights_gb = _estimate_weights_gb(selected_draft_path)
    overhead_gb = 0.5

    ctx_table = Table()
    ctx_table.add_column("#", style="bold yellow", justify="right")
    ctx_table.add_column("Tokens", justify="right", style="cyan")
    ctx_table.add_column("Label", style="cyan")
    ctx_table.add_column("KV Cache", justify="right", style="green")
    ctx_table.add_column("Total VRAM", justify="right", style="bold")
    ctx_table.add_column("Fit 48GB?", style="bold")

    for i, (tokens, label) in enumerate(ctx_options, 1):
        kv_gb = _estimate_kv_cache_gb(arch, tokens, tq_bits)
        total = model_weights_gb + draft_weights_gb + kv_gb + overhead_gb
        fits = total <= 48.0
        fit_str = "[green]YES[/green]" if fits else "[red]NO[/red]"
        ctx_table.add_row(
            str(i), f"{tokens:>7d}", label,
            f"{kv_gb:.1f} GB", f"{total:.1f} GB",
            fit_str,
        )

    custom_idx = len(ctx_options) + 1
    ctx_table.add_row(str(custom_idx), "", "Custom", "", "", "")

    console.print(ctx_table)
    console.print(f"\n[dim]  Weights: {model_weights_gb:.1f} GB model + {draft_weights_gb:.1f} GB draft + 0.5 GB overhead[/dim]")
    console.print()

    choice = IntPrompt.ask("Select context window", default=3)
    choice_idx = choice - 1

    if choice_idx < len(ctx_options):
        context_window = ctx_options[choice_idx][0]
    else:
        context_window = IntPrompt.ask("Enter custom context window (tokens)", default=32768)

    kv_final = _estimate_kv_cache_gb(arch, context_window, tq_bits)
    total_final = model_weights_gb + draft_weights_gb + kv_final + overhead_gb
    max_output = min(context_window // 4, 8192)

    console.print(
        f"  Selected: [cyan]{context_window}[/cyan] tokens "
        f"(KV: {kv_final:.1f} GB, total VRAM: {total_final:.1f} GB)\n"
    )

    # --- Step 5: Assign to tier ---
    console.print("[bold]Step 5: Assign to tier[/bold]\n")

    tier_options = ["large", "medium", "small"]
    for i, t in enumerate(tier_options, 1):
        console.print(f"  [{i}] {t}")

    console.print()
    tier_idx = IntPrompt.ask("Select tier", default=1) - 1
    tier_idx = max(0, min(tier_idx, len(tier_options) - 1))
    tier_name = tier_options[tier_idx]

    # --- Build the tier config ---
    new_tier = TierConfig(
        name=tier_name,
        target_model=str(selected_path),
        draft_model=str(selected_draft_path),
        context_window=context_window,
        max_output_tokens=max_output,
        tq_bits=tq_bits,
        # This wizard selects TurboQuant or an uncompressed cache.  Leaving
        # PolarQuant at its dataclass default would either shadow TQ or make
        # the displayed "OFF" choice compressed in practice.
        pq_bits=pq_bits,
        tq_group_size=64,
        tq_use_rotation=tq_enabled,
        tq_use_normalization=tq_enabled,
        tq_use_qjl=False,
    )

    config.tiers[tier_name] = new_tier

    # --- Summary ---
    console.print()
    console.print(Panel(
        f"[bold]Tier:[/bold]       {tier_name}\n"
        f"[bold]Target:[/bold]     models/{selected_name}\n"
        f"[bold]Draft:[/bold]      spd/{selected_draft_name}\n"
        f"[bold]TQ Cache:[/bold]   {'OFF' if not tq_enabled else f'{tq_bits}-bit'}\n"
        f"[bold]Context:[/bold]    {context_window:,} tokens\n"
        f"[bold]Max Output:[/bold] {max_output:,} tokens\n"
        f"[bold]Est. VRAM:[/bold]  {total_final:.1f} GB "
        f"(weights {model_weights_gb:.1f} + draft {draft_weights_gb:.1f} + KV {kv_final:.1f} + overhead 0.5)",
        title="Configuration Saved",
        border_style="green",
    ))

    # Save to config file
    config_path = config.config_dir / "config.json"
    save_config(config, config_path)
    console.print(f"\n[dim]Saved to {config_path}[/dim]")

    # Ask if user wants to configure another tier
    if Prompt.ask("\nConfigure another tier?", choices=["y", "n"], default="n") == "y":
        return configure_interactive(config)

    return config
