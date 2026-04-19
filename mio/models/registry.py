"""Known model+draft pairs and tier definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mio.config import TierConfig


# --- Path resolution ---

def _project_root() -> Path:
    """Return the mio project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def models_dir() -> Path:
    """Return the local models/ directory (target models)."""
    return _project_root() / "models"


def spd_dir() -> Path:
    """Return the local spd/ directory (speculative decoding drafts)."""
    return _project_root() / "spd"


def resolve_model_path(local_name: str, kind: str = "target") -> str:
    """Resolve a model to a local path if it exists, else return as HF repo ID.

    Args:
        local_name: Local directory name (e.g., "Qwen3.5-4B-4bit")
        kind: "target" checks models/, "draft" checks spd/
    """
    base = models_dir() if kind == "target" else spd_dir()
    local_path = base / local_name
    if local_path.exists() and (local_path / "config.json").exists():
        return str(local_path)
    return local_name  # Fall back to HF repo ID


# --- Model entries ---

@dataclass
class ModelEntry:
    """A known target+draft model pair."""

    target_repo: str       # HuggingFace repo ID (fallback)
    target_local: str      # Local directory name in models/
    draft_repo: str        # HuggingFace repo ID (fallback)
    draft_local: str       # Local directory name in spd/
    adapter: str
    default_tier: str
    context_window: int
    max_output_tokens: int
    description: str = ""

    def resolve_target(self) -> str:
        """Return local path if available, else HF repo ID."""
        return resolve_model_path(self.target_local, kind="target")

    def resolve_draft(self) -> str:
        """Return local path if available, else HF repo ID."""
        return resolve_model_path(self.draft_local, kind="draft")


# All known z-lab DFlash model pairs
KNOWN_MODELS: dict[str, ModelEntry] = {
    # === Qwen 3.5 PARO family (INT4 with pairwise rotation, z-lab) ===
    "qwen3.5-4b": ModelEntry(
        target_repo="z-lab/Qwen3.5-4B-PARO",
        target_local="Qwen3.5-4B-PARO",
        draft_repo="z-lab/Qwen3.5-4B-DFlash",
        draft_local="Qwen3.5-4B-DFlash",
        adapter="qwen3_5",
        default_tier="small",
        context_window=8192,
        max_output_tokens=2048,
        description="Qwen 3.5 4B PARO — fast, small context",
    ),
    "qwen3.5-9b": ModelEntry(
        target_repo="z-lab/Qwen3.5-9B-PARO",
        target_local="Qwen3.5-9B-PARO",
        draft_repo="z-lab/Qwen3.5-9B-DFlash",
        draft_local="Qwen3.5-9B-DFlash",
        adapter="qwen3_5",
        default_tier="medium",
        context_window=16384,
        max_output_tokens=4096,
        description="Qwen 3.5 9B PARO — balanced speed/quality",
    ),
    "qwen3.5-27b": ModelEntry(
        target_repo="z-lab/Qwen3.5-27B-PARO",
        target_local="Qwen3.5-27B-PARO",
        draft_repo="z-lab/Qwen3.5-27B-DFlash",
        draft_local="Qwen3.5-27B-DFlash",
        adapter="qwen3_5",
        default_tier="large",
        context_window=32768,
        max_output_tokens=8192,
        description="Qwen 3.5 27B PARO — highest quality dense, large context",
    ),
    "qwen3.5-35b-a3b": ModelEntry(
        target_repo="z-lab/Qwen3.5-35B-A3B-PARO",
        target_local="Qwen3.5-35B-A3B-PARO",
        draft_repo="z-lab/Qwen3.5-35B-A3B-DFlash",
        draft_local="Qwen3.5-35B-A3B-DFlash",
        adapter="qwen3_5",
        default_tier="large-moe",
        context_window=131072,
        max_output_tokens=8192,
        description="Qwen 3.5 35B-A3B MoE PARO — 35B total, 3B active per token, 128K ctx",
    ),
    # === Qwen 3.5 MLX-community (standard quantization, no PARO) ===
    "qwen3.5-4b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3.5-4B-4bit",
        target_local="Qwen3.5-4B-4bit",
        draft_repo="z-lab/Qwen3.5-4B-DFlash",
        draft_local="Qwen3.5-4B-DFlash",
        adapter="qwen3_5",
        default_tier="small",
        context_window=8192,
        max_output_tokens=2048,
        description="Qwen 3.5 4B 4-bit (mlx-community, no PARO)",
    ),
    "qwen3.5-9b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        target_local="Qwen3.5-9B-4bit",
        draft_repo="z-lab/Qwen3.5-9B-DFlash",
        draft_local="Qwen3.5-9B-DFlash",
        adapter="qwen3_5",
        default_tier="medium",
        context_window=16384,
        max_output_tokens=4096,
        description="Qwen 3.5 9B 4-bit (mlx-community, no PARO)",
    ),
    "qwen3.5-27b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3.5-27B-4bit",
        target_local="Qwen3.5-27B-4bit",
        draft_repo="z-lab/Qwen3.5-27B-DFlash",
        draft_local="Qwen3.5-27B-DFlash",
        adapter="qwen3_5",
        default_tier="large",
        context_window=32768,
        max_output_tokens=8192,
        description="Qwen 3.5 27B 4-bit (mlx-community, no PARO)",
    ),
    "qwen3.5-35b-a3b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3.5-35B-A3B-4bit",
        target_local="Qwen3.5-35B-A3B-4bit",
        draft_repo="z-lab/Qwen3.5-35B-A3B-DFlash",
        draft_local="Qwen3.5-35B-A3B-DFlash",
        adapter="qwen3_5",
        default_tier="large-moe",
        context_window=131072,
        max_output_tokens=8192,
        description="Qwen 3.5 35B-A3B MoE 4-bit (mlx-community, no PARO)",
    ),
    # === Qwen 3.5 Unsloth MLX (Brooooooklyn re-quants) ===
    # Fixes mlx-lm issue #1011 — mlx-community INT4/8 checkpoints degrade
    # tool-calling in multi-turn. Unsloth imatrix is calibrated on tool-call
    # data; UD-Q4_K_XL is the same format that completes 70/70 rounds cleanly.
    "qwen3.5-9b-unsloth": ModelEntry(
        target_repo="Brooooooklyn/Qwen3.5-9B-UD-Q4_K_XL-mlx",
        target_local="Qwen3.5-9B-UD-Q4_K_XL-mlx",
        draft_repo="z-lab/Qwen3.5-9B-DFlash",
        draft_local="Qwen3.5-9B-DFlash",
        adapter="qwen3_5",
        default_tier="medium",
        context_window=16384,
        max_output_tokens=4096,
        description="Qwen 3.5 9B Unsloth MLX UD-Q4_K_XL — tool-calls fixed",
    ),
    "qwen3.5-27b-unsloth": ModelEntry(
        target_repo="Brooooooklyn/Qwen3.5-27B-UD-Q4_K_XL-mlx",
        target_local="Qwen3.5-27B-UD-Q4_K_XL-mlx",
        draft_repo="z-lab/Qwen3.5-27B-DFlash",
        draft_local="Qwen3.5-27B-DFlash",
        adapter="qwen3_5",
        default_tier="large",
        context_window=32768,
        max_output_tokens=8192,
        description="Qwen 3.5 27B Unsloth MLX UD-Q4_K_XL — tool-calls fixed",
    ),
    "qwen3.5-35b-a3b-unsloth": ModelEntry(
        target_repo="Brooooooklyn/Qwen3.5-35B-A3B-UD-Q4_K_XL-mlx",
        target_local="Qwen3.5-35B-A3B-UD-Q4_K_XL-mlx",
        draft_repo="z-lab/Qwen3.5-35B-A3B-DFlash",
        draft_local="Qwen3.5-35B-A3B-DFlash",
        adapter="qwen3_5",
        default_tier="large-moe",
        context_window=131072,
        max_output_tokens=8192,
        description="Qwen 3.5 35B-A3B Unsloth MLX UD-Q4_K_XL — tool-calls fixed",
    ),
    # === Qwen 3.6 (same architecture as 3.5 — qwen3_5_moe) ===
    "qwen3.6-35b-a3b-unsloth": ModelEntry(
        target_repo="Brooooooklyn/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx",
        target_local="Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx",
        draft_repo="z-lab/Qwen3.6-35B-A3B-DFlash",
        draft_local="Qwen3.6-35B-A3B-DFlash",
        adapter="qwen3_5",
        default_tier="large-moe",
        context_window=262144,
        max_output_tokens=8192,
        description="Qwen 3.6 35B-A3B Unsloth MLX UD-Q4_K_XL — 256K ctx native",
    ),
    "qwen3.6-35b-a3b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3.6-35B-A3B-4bit",
        target_local="Qwen3.6-35B-A3B-4bit",
        draft_repo="z-lab/Qwen3.6-35B-A3B-DFlash",
        draft_local="Qwen3.6-35B-A3B-DFlash",
        adapter="qwen3_5",
        default_tier="large-moe",
        context_window=262144,
        max_output_tokens=8192,
        description="Qwen 3.6 35B-A3B 4-bit (mlx-community) — 256K ctx",
    ),
    # === Qwen 3 family (use qwen3 adapter) ===
    "qwen3-4b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3-4B-4bit",
        target_local="Qwen3-4B-4bit",
        draft_repo="z-lab/Qwen3-4B-DFlash-b16",
        draft_local="Qwen3-4B-DFlash-b16",
        adapter="qwen3",
        default_tier="small",
        context_window=8192,
        max_output_tokens=2048,
        description="Qwen 3 4B 4-bit",
    ),
    "qwen3-8b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3-8B-4bit",
        target_local="Qwen3-8B-4bit",
        draft_repo="z-lab/Qwen3-8B-DFlash-b16",
        draft_local="Qwen3-8B-DFlash-b16",
        adapter="qwen3",
        default_tier="small",
        context_window=16384,
        max_output_tokens=4096,
        description="Qwen 3 8B 4-bit",
    ),
    # === Qwen 3 Coder — trained for Cline/Kilo-style XML tool calls ===
    "qwen3-coder-30b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        target_local="Qwen3-Coder-30B-A3B-Instruct-4bit",
        draft_repo="z-lab/Qwen3-Coder-30B-A3B-DFlash",
        draft_local="Qwen3-Coder-30B-A3B-DFlash",
        adapter="qwen3",
        default_tier="large",
        context_window=32768,
        max_output_tokens=8192,
        description="Qwen 3 Coder 30B-A3B MoE 4-bit — agentic coding workflow",
    ),
    # === LLaMA 3.1 (needs llama adapter) ===
    "llama-3.1-8b-4bit": ModelEntry(
        target_repo="mlx-community/Llama-3.1-8B-Instruct-4bit",
        target_local="Llama-3.1-8B-Instruct-4bit",
        draft_repo="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        draft_local="LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        adapter="llama",
        default_tier="small",
        context_window=16384,
        max_output_tokens=4096,
        description="LLaMA 3.1 8B 4-bit — needs adapter",
    ),
    # === GPT-OSS (needs gpt_oss adapter) ===
    "gpt-oss-20b": ModelEntry(
        target_repo="inclusionAI/GPT-OSS-20B",
        target_local="GPT-OSS-20B",
        draft_repo="z-lab/gpt-oss-20b-DFlash",
        draft_local="gpt-oss-20b-DFlash",
        adapter="gpt_oss",
        default_tier="medium",
        context_window=16384,
        max_output_tokens=4096,
        description="GPT-OSS 20B — needs adapter",
    ),
    "gpt-oss-120b": ModelEntry(
        target_repo="inclusionAI/GPT-OSS-120B",
        target_local="GPT-OSS-120B",
        draft_repo="z-lab/gpt-oss-120b-DFlash",
        draft_local="gpt-oss-120b-DFlash",
        adapter="gpt_oss",
        default_tier="large",
        context_window=32768,
        max_output_tokens=8192,
        description="GPT-OSS 120B — needs adapter",
    ),
    # === Kimi (needs kimi adapter) ===
    "kimi-k2.5": ModelEntry(
        target_repo="moonshotai/Kimi-K2.5",
        target_local="Kimi-K2.5",
        draft_repo="z-lab/Kimi-K2.5-DFlash",
        draft_local="Kimi-K2.5-DFlash",
        adapter="kimi",
        default_tier="large",
        context_window=32768,
        max_output_tokens=8192,
        description="Kimi K2.5 — needs adapter",
    ),
}

# Adapters that are currently implemented in dflash-mlx
SUPPORTED_ADAPTERS = {"qwen3", "qwen3_5"}


def _make_tier(name: str, entry_key: str) -> TierConfig:
    """Build a TierConfig that resolves to local path if available."""
    entry = KNOWN_MODELS[entry_key]
    return TierConfig(
        name=name,
        target_model=entry.resolve_target(),
        draft_model=entry.resolve_draft(),
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
    )


def _entry_is_local(key: str) -> bool:
    """Return True if both target and draft dirs for a registry key exist locally."""
    entry = KNOWN_MODELS[key]
    target = models_dir() / entry.target_local / "config.json"
    draft = spd_dir() / entry.draft_local / "config.json"
    return target.exists() and draft.exists()


# Preference order for the large-moe tier:
#   1. Qwen 3.6 Unsloth UD-Q4_K_XL  (256K ctx, tool-calls fixed)
#   2. Qwen 3.6 mlx-community 4-bit (256K ctx, vanilla quantization)
#   3. Qwen 3.5 Unsloth UD-Q4_K_XL  (128K ctx, what `mio pull large-moe` ships)
#
# Any user who manually fetches one of the 3.6 variants into models/ + spd/
# (the 3.6 DFlash draft repo is HF-gated, so it can't be pulled by the CLI)
# will have it picked up automatically — no config edit required.
_LARGE_MOE_PRIORITY = [
    "qwen3.6-35b-a3b-unsloth",
    "qwen3.6-35b-a3b-4bit",
    "qwen3.5-35b-a3b-unsloth",
]


def _pick_large_moe_key() -> str:
    """Pick the best locally-available large-moe variant; fall back to 3.5."""
    for key in _LARGE_MOE_PRIORITY:
        if _entry_is_local(key):
            return key
    return "qwen3.5-35b-a3b-unsloth"


# Default tier configurations — use Unsloth MLX Q4_K_XL re-quants for
# large/medium/large-moe (fixes tool-call degradation from mlx-lm issue #1011).
# Small tier keeps mlx-community 4bit: no tool-call use case, still fast.
DEFAULT_TIERS: dict[str, TierConfig] = {
    "large-moe": _make_tier("large-moe", _pick_large_moe_key()),
    "large": _make_tier("large", "qwen3.5-27b-unsloth"),
    "medium": _make_tier("medium", "qwen3.5-9b-unsloth"),
    "small": _make_tier("small", "qwen3.5-4b-4bit"),
}

# PARO tier configurations (INT4 with pairwise rotation, z-lab)
# Higher quality quantization but slower due to Metal kernel workaround
PARO_TIERS: dict[str, TierConfig] = {
    "large-moe": _make_tier("large-moe", "qwen3.5-35b-a3b"),
    "large": _make_tier("large", "qwen3.5-27b"),
    "medium": _make_tier("medium", "qwen3.5-9b"),
    "small": _make_tier("small", "qwen3.5-4b"),
}


def get_supported_models() -> dict[str, ModelEntry]:
    """Return only models with implemented adapters."""
    return {k: v for k, v in KNOWN_MODELS.items() if v.adapter in SUPPORTED_ADAPTERS}


def get_draft_for_target(target_model: str) -> str | None:
    """Look up the DFlash draft model for a given target."""
    for entry in KNOWN_MODELS.values():
        if entry.target_repo == target_model or entry.resolve_target() == target_model:
            return entry.resolve_draft()
    return None
