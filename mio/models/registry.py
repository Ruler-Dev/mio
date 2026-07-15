"""Known model+draft pairs and tier definitions."""

from __future__ import annotations

import json
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


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _has_target_tokenizer_assets(local_path: Path) -> bool:
    """Recognize tokenizer layouts that MLX/Transformers can load locally."""

    single_file_layouts = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
    )
    has_vocabulary = any(_nonempty_file(local_path / name) for name in single_file_layouts) or (
        _nonempty_file(local_path / "vocab.json") and _nonempty_file(local_path / "merges.txt")
    )
    if not has_vocabulary:
        return False

    # Chat templates can live beside the tokenizer or inside tokenizer/
    # processor metadata. The processor form is common for multimodal repos.
    if _nonempty_file(local_path / "chat_template.jinja"):
        return True
    for directory in ("chat_templates", "additional_chat_templates"):
        templates_dir = local_path / directory
        if templates_dir.is_dir() and any(_nonempty_file(path) for path in templates_dir.glob("*.jinja")):
            return True

    def _template_value_nonempty(value) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return any(_template_value_nonempty(item) for item in value.values())
        if isinstance(value, list):
            return any(_template_value_nonempty(item) for item in value)
        return False

    def _contains_chat_template(value) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "chat_template" and _template_value_nonempty(item):
                    return True
                if isinstance(item, (dict, list)) and _contains_chat_template(item):
                    return True
        elif isinstance(value, list):
            return any(_contains_chat_template(item) for item in value)
        return False

    for name in (
        "chat_template.json",  # legacy processor format
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        path = local_path / name
        if not _nonempty_file(path):
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict) and _contains_chat_template(metadata):
            return True
    return False


def _model_path_is_complete(
    local_path: Path,
    *,
    require_tokenizer: bool = False,
) -> bool:
    """Return whether a local MLX checkpoint has all role-critical assets.

    Hugging Face writes ``config.json`` before the multi-gigabyte weight files.
    Treating that early file as a completed checkpoint makes interrupted pulls
    poison tier auto-detection and fail much later inside ``mlx_lm.load``.
    Drafts use the default weights-only contract; targets additionally require
    a supported tokenizer layout and a chat template.
    """
    if not (local_path / "config.json").is_file():
        return False

    index_path = local_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(index.get("weight_map", {}).values())
        except (OSError, TypeError, ValueError):
            return False
        weights_complete = bool(shard_names) and all(
            (local_path / name).is_file() and (local_path / name).stat().st_size > 0 for name in shard_names
        )
    else:
        weights = list(local_path.glob("*.safetensors"))
        weights_complete = bool(weights) and all(path.stat().st_size > 0 for path in weights)
    return weights_complete and (not require_tokenizer or _has_target_tokenizer_assets(local_path))


def resolve_model_path(local_name: str, kind: str = "target") -> str:
    """Resolve a complete model to a local path, else return its HF reference.

    Args:
        local_name: Local directory name (e.g., "Qwen3.5-4B-4bit")
        kind: "target" checks models/, "draft" checks spd/
    """
    base = models_dir() if kind == "target" else spd_dir()
    local_path = base / local_name
    if _model_path_is_complete(local_path, require_tokenizer=kind == "target"):
        return str(local_path)
    return local_name  # Fall back to HF repo ID


# --- Model entries ---


@dataclass
class ModelEntry:
    """A known target+draft model pair."""

    target_repo: str  # HuggingFace repo ID (fallback)
    target_local: str  # Local directory name in models/
    draft_repo: str  # HuggingFace repo ID (fallback)
    draft_local: str  # Local directory name in spd/
    adapter: str
    default_tier: str
    context_window: int
    max_output_tokens: int
    description: str = ""
    dspark_repo: str | None = None
    dspark_local: str | None = None
    dspark_max_draft_tokens: int = 2
    dspark_lookup_drafts: bool = True

    def resolve_target(self) -> str:
        """Return local path if available, else HF repo ID."""
        local_path = models_dir() / self.target_local
        return str(local_path) if _model_path_is_complete(local_path, require_tokenizer=True) else self.target_repo

    def resolve_draft(self) -> str:
        """Return local path if available, else HF repo ID."""
        local_path = spd_dir() / self.draft_local
        return str(local_path) if _model_path_is_complete(local_path) else self.draft_repo

    def resolve_dspark(self) -> str | None:
        """Return a complete local DSpark checkpoint, never a remote reference.

        DSpark is an optional accelerator.  Startup must not turn a missing
        local copy into an implicit multi-gigabyte Hugging Face download;
        ``mio pull`` is the explicit installation boundary.
        """
        if not self.dspark_local:
            return None
        local_path = spd_dir() / self.dspark_local
        return str(local_path) if _model_path_is_complete(local_path) else None


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
    # === Qwen 3.6 dense (same qwen3_5 text architecture as Qwen 3.5) ===
    "qwen3.6-27b-unsloth": ModelEntry(
        target_repo="Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx",
        target_local="Qwen3.6-27B-UD-Q4_K_XL-mlx",
        draft_repo="z-lab/Qwen3.6-27B-DFlash",
        draft_local="Qwen3.6-27B-DFlash",
        adapter="qwen3_5",
        default_tier="large",
        context_window=262144,
        max_output_tokens=8192,
        description="Qwen 3.6 27B Unsloth MLX UD-Q4_K_XL — 256K ctx native",
        dspark_repo="Avesed/Qwen3.6-27B-DSpark",
        dspark_local="Qwen3.6-27B-DSpark",
        dspark_max_draft_tokens=3,
        dspark_lookup_drafts=False,
    ),
    "qwen3.6-27b-4bit": ModelEntry(
        target_repo="mlx-community/Qwen3.6-27B-4bit",
        target_local="Qwen3.6-27B-4bit",
        draft_repo="z-lab/Qwen3.6-27B-DFlash",
        draft_local="Qwen3.6-27B-DFlash",
        adapter="qwen3_5",
        default_tier="large",
        context_window=262144,
        max_output_tokens=8192,
        description="Qwen 3.6 27B 4-bit (mlx-community) — 256K ctx",
        dspark_repo="Avesed/Qwen3.6-27B-DSpark",
        dspark_local="Qwen3.6-27B-DSpark",
        dspark_max_draft_tokens=3,
        dspark_lookup_drafts=False,
    ),
    # === Qwen 3.6 MoE (qwen3_5_moe) ===
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
    dspark = entry.resolve_dspark()
    local_dflash = spd_dir() / entry.draft_local
    dflash_fallback = str(local_dflash) if _model_path_is_complete(local_dflash) else None
    return TierConfig(
        name=name,
        target_model=entry.resolve_target(),
        draft_model=dspark or entry.resolve_draft(),
        # A DSpark-first tier may fall back only to an already installed,
        # independently compatible DFlash checkpoint.  Missing fallback
        # weights never trigger an implicit download during engine startup.
        draft_fallback_model=dflash_fallback if dspark else None,
        dspark_max_draft_tokens=entry.dspark_max_draft_tokens,
        dspark_lookup_drafts=entry.dspark_lookup_drafts,
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
    )


def _entry_is_local(key: str) -> bool:
    """Return True when the target and at least one drafter are complete."""
    entry = KNOWN_MODELS[key]
    target = models_dir() / entry.target_local
    draft = spd_dir() / entry.draft_local
    dspark = spd_dir() / entry.dspark_local if entry.dspark_local else None
    return _model_path_is_complete(target, require_tokenizer=True) and (
        _model_path_is_complete(draft) or bool(dspark is not None and _model_path_is_complete(dspark))
    )


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

# Preference order for the dense large tier. Qwen 3.6 requires both the target
# and its causal-SWA DFlash draft; an interrupted download safely falls back.
_LARGE_DENSE_PRIORITY = [
    "qwen3.6-27b-unsloth",
    "qwen3.6-27b-4bit",
    "qwen3.5-27b-unsloth",
]


def _pick_large_moe_key() -> str:
    """Pick the best locally-available large-moe variant; fall back to 3.5."""
    for key in _LARGE_MOE_PRIORITY:
        if _entry_is_local(key):
            return key
    return "qwen3.5-35b-a3b-unsloth"


def _pick_large_dense_key() -> str:
    """Pick the best complete local dense variant; fall back to Qwen 3.5."""
    for key in _LARGE_DENSE_PRIORITY:
        if _entry_is_local(key):
            return key
    return "qwen3.5-27b-unsloth"


# Default tier configurations — use Unsloth MLX Q4_K_XL re-quants for
# large/medium/large-moe (fixes tool-call degradation from mlx-lm issue #1011).
# Small tier keeps mlx-community 4bit: no tool-call use case, still fast.
DEFAULT_TIERS: dict[str, TierConfig] = {
    "large-moe": _make_tier("large-moe", _pick_large_moe_key()),
    "large": _make_tier("large", _pick_large_dense_key()),
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


def get_dspark_for_target(target_model: str) -> str | None:
    """Look up the optional DSpark drafter for a given target."""
    for entry in KNOWN_MODELS.values():
        if entry.target_repo == target_model or entry.resolve_target() == target_model:
            return entry.resolve_dspark()
    return None
