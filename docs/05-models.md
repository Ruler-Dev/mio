# Model Guide

## Directory Structure

```
mio/
  models/                          # PARO target models (z-lab, INT4)
    Qwen3.5-4B-PARO/
    Qwen3.5-9B-PARO/
    Qwen3.5-27B-PARO/
    Qwen3.5-35B-A3B-PARO/         # Default (large-moe tier)
  spd/                             # DFlash speculative decoding drafts
    Qwen3.5-4B-DFlash/
    Qwen3.5-9B-DFlash/
    Qwen3.5-27B-DFlash/
    Qwen3.5-35B-A3B-DFlash/
```

## PARO Quantization

All target models use **PARO** (Pairwise Rotation Quantization, z-lab, ICLR 2026):
- INT4 weight quantization with learned Givens rotations
- 2.4% more accurate than AWQ on reasoning tasks
- Custom Metal kernel for rotation at inference time
- Same speed as standard INT4, just better quality

PARO is orthogonal to DFlash (speculative decoding) and TurboQuant (KV cache). All three stack.

## Pulling Models

```bash
mio pull                         # List all available keys
mio pull qwen3.5-35b-a3b        # Default model + DFlash draft
mio pull qwen3.5-27b            # Dense 27B
mio pull qwen3.5-9b             # 9B
mio pull qwen3.5-4b             # 4B
```

## Default Tiers

| Tier | Target (models/) | Draft (spd/) | Context | Notes |
|------|---|---|---|---|
| **large-moe** (default) | `Qwen3.5-35B-A3B-PARO` | `Qwen3.5-35B-A3B-DFlash` | 128K | MoE: 35B total, 3B active, 204 tok/s |
| large | `Qwen3.5-27B-PARO` | `Qwen3.5-27B-DFlash` | 32K | Dense 27B, 56 tok/s |
| medium | `Qwen3.5-9B-PARO` | `Qwen3.5-9B-DFlash` | 16K | 108 tok/s |
| small | `Qwen3.5-4B-PARO` | `Qwen3.5-4B-DFlash` | 8K | 187 tok/s |

## All Known Models

| Key | Target | Draft | Adapter | Status |
|-----|--------|-------|---------|--------|
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B-PARO | Qwen3.5-35B-A3B-DFlash | qwen3_5 | Ready (default) |
| `qwen3.5-27b` | Qwen3.5-27B-PARO | Qwen3.5-27B-DFlash | qwen3_5 | Ready |
| `qwen3.5-9b` | Qwen3.5-9B-PARO | Qwen3.5-9B-DFlash | qwen3_5 | Ready |
| `qwen3.5-4b` | Qwen3.5-4B-PARO | Qwen3.5-4B-DFlash | qwen3_5 | Ready |
| `qwen3-4b-4bit` | Qwen3-4B-4bit | Qwen3-4B-DFlash-b16 | qwen3 | Available |
| `qwen3-8b-4bit` | Qwen3-8B-4bit | Qwen3-8B-DFlash-b16 | qwen3 | Available |
| `qwen3-coder-30b-4bit` | Qwen3-Coder-30B-A3B-4bit | Qwen3-Coder-30B-A3B-DFlash | qwen3 | Available |
| `llama-3.1-8b-4bit` | Llama-3.1-8B-Instruct-4bit | LLaMA3.1-8B-DFlash | llama | Needs adapter |
| `gpt-oss-20b` | GPT-OSS-20B | gpt-oss-20b-DFlash | gpt_oss | Needs adapter |
| `kimi-k2.5` | Kimi-K2.5 | Kimi-K2.5-DFlash | kimi | Needs adapter |

"Ready" = downloaded + PARO + DFlash tested. "Available" = DFlash draft exists, download needed. "Needs adapter" = requires new MLX adapter.

## How Model Loading Works

1. Mio reads `config.json` in the model directory
2. If `quantization_config.quant_method == "paroquant"`, uses the PARO loader (applies `RotateQuantizedLinear` with Metal rotation kernel)
3. If standard quantization, uses DFlash's `load_target_bundle()`
4. DFlash draft is loaded separately via `load_draft_bundle()`
5. TurboQuant monkey-patches the SDPA for KV cache compression

All three layers are applied transparently -- you just pick a model and context settings.
