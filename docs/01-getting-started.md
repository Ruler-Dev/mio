# Getting Started

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.10
- 48 GB+ unified memory recommended (16 GB minimum for small tier only)

## Installation

```bash
cd mio
pip install -e .
```

## Download Models

Models are stored locally:
- `models/` -- PARO target models (z-lab, INT4 with pairwise rotation)
- `spd/` -- DFlash speculative decoding drafts (z-lab)

Download all default tier models:
```bash
mio download
```

Or pull a specific model (target + DFlash draft):
```bash
mio pull qwen3.5-35b-a3b    # Default: 35B-A3B MoE
mio pull qwen3.5-27b        # Dense 27B
mio pull qwen3.5-9b         # 9B
mio pull qwen3.5-4b         # 4B
```

Verify models are ready:
```bash
python -m mio.model_check
```

## First Run

### Launch the coding agent (default)
```bash
mio
```

This loads the **large-moe** tier (Qwen3.5-35B-A3B MoE PARO, 128K context, fp16 KV) and drops you into an interactive agent with Caveman Ultra mode. Type `/help` for slash commands.

Default settings for 48 GB M4 Max:
- **Model:** Qwen3.5-35B-A3B MoE PARO (35B brain, 3B active per token)
- **Context:** 128K tokens
- **KV cache:** standard fp16 (TQ4 is opt-in via `--tq4`)
- **Speed:** ~204 tok/s
- **VRAM:** ~18 GB (leaves 30 GB free)
- **Prefix cache:** automatic; ~5-8× TTFT speedup on warm hits when consecutive prompts share a long prefix (system prompt, agent history)

For deeper-context users, pass `--tq4` to enable TurboQuant 4-bit KV cache (KV ≈ 28% of fp16, decode 0.7-0.9× of baseline; on 27B-dense at 32K decode is actually 1.67× faster).

### Other ways to start

```bash
mio --tier medium            # Use 9B model (108 tok/s, less VRAM)
mio --tier small             # Use 4B model (187 tok/s, minimal VRAM)
mio --tandem                 # All tiers loaded, auto-routing
mio "fix the type error"     # Start with a task
mio serve                    # API server mode
mio chat                     # Simple chat (no agent tools)
```

### Inside the agent

Switch models on the fly:
```
> /tier medium          # Switch to 9B (reloads model)
> /tier large-moe       # Switch back to 35B-A3B (reloads model)
```

Change context window and TQ cache interactively:
```
> /context              # Opens interactive selector
Select context window:
  [1]    8K
  [2]   16K
  [3]   32K
  [4]   64K
  [5]  128K  <-- current
  [6]  256K
Context [5]: 6

Select TurboQuant cache:
  [1] TQ 4-bit  <-- current
  [2] TQ 3-bit
  [3] TQ 2-bit
  [4] OFF
TQ mode [1]: 1

Reloading model...
Context set: 256K, TQ 4-bit
Engine ready: large-moe
```

## Model Tiers

| Tier | Model | VRAM | Context | tok/s |
|------|-------|------|---------|-------|
| **large-moe** (default) | Qwen3.5-35B-A3B MoE PARO | ~17 GB | 128K | ~204 |
| large | Qwen3.5-27B PARO | ~14 GB | 32K | ~56 |
| medium | Qwen3.5-9B PARO | ~5 GB | 16K | ~108 |
| small | Qwen3.5-4B PARO | ~2.5 GB | 8K | ~187 |

All models use PARO quantization (2.4% better than AWQ) with DFlash speculative decoding and TurboQuant V2 KV cache.

## Memory Planning (M4 Max 48 GB)

| Configuration | VRAM | Fits? |
|---|---|---|
| large-moe only (35B-A3B, 128K, TQ4) | ~18 GB | Yes (default) |
| large-moe + medium (tandem) | ~24 GB | Yes |
| All four tiers (tandem) | ~40 GB | Yes |
| large-moe at 256K TQ4 | ~20 GB | Yes |
| large-moe at 256K TQ2 | ~19 GB | Yes |
| large only (27B, 128K, TQ4) | ~16 GB | Yes |
