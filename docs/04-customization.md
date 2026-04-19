# Customization Guide

## Changing Context and TQ On the Fly

Inside the agent, type `/context`:

```
Current: 131,072 tokens, TQ 4-bit

Select context window:
  [1]    8K
  [2]   16K
  [3]   32K
  [4]   64K
  [5]  128K  <-- current
  [6]  256K
Context [5]: 6

Select TurboQuant cache:
  [1] TQ 4-bit     (3.6x compression) <-- current
  [2] TQ 3-bit     (4.7x compression)
  [3] TQ 2-bit     (5.5x compression)
  [4] OFF          (no compression)
TQ mode [1]: 1

Reloading model with new context/TQ settings...
Context set: 256K, TQ 4-bit
Engine ready: large-moe
```

The model reloads automatically. Press Enter to keep current values.

## Switching Tiers

```
> /tier large-moe       # 35B-A3B MoE (204 tok/s, default)
> /tier large            # 27B dense (56 tok/s)
> /tier medium           # 9B (108 tok/s)
> /tier small            # 4B (187 tok/s)
```

Switching tiers unloads the old model and loads the new one. You'll see the loading messages.

## Caveman Mode

Control output verbosity:

```
> /caveman ultra        # Default: 75% fewer tokens
> /caveman full         # 60% fewer tokens
> /caveman lite         # 35% fewer tokens
> /caveman off          # Standard verbose output
```

No reload needed -- takes effect immediately on next response.

## `mio configure` -- Full Wizard

For more detailed configuration (model selection, DFlash draft, TQ parameters, context with VRAM estimates):

```bash
mio configure
```

5 steps:
1. **Select target model** from `models/` -- shows weight sizes
2. **Select DFlash draft** from `spd/` -- auto-matched
3. **Select TQ cache bits** (4/3/2/OFF) -- shows compression + KV estimates
4. **Select context window** (8K-265K) -- shows VRAM + "Fits 48GB?" column
5. **Assign to tier** (large-moe/large/medium/small)

## Adding Custom Models

### From the registry

```bash
mio pull qwen3-8b-4bit     # If it's in KNOWN_MODELS
```

### Manual

```bash
# Place PARO model in models/
huggingface-cli download z-lab/Qwen3.5-9B-PARO
cp -rL ~/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-PARO/snapshots/*/  models/Qwen3.5-9B-PARO/

# Place DFlash draft in spd/
huggingface-cli download z-lab/Qwen3.5-9B-DFlash
cp -rL ~/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/*/  spd/Qwen3.5-9B-DFlash/

# Configure
mio configure
```

## VRAM Reference

Context window VRAM for 35B-A3B MoE (default) with TQ 4-bit:

| Context | KV Cache | Total VRAM |
|---------|----------|------------|
| 32K | ~0.5 GB | ~18.5 GB |
| 64K | ~1.0 GB | ~19.0 GB |
| 128K (default) | ~2.0 GB | ~20.0 GB |
| 256K | ~4.0 GB | ~22.0 GB |
| 512K (TQ2) | ~4.0 GB | ~22.0 GB |
| 1M (TQ2) | ~7.8 GB | ~25.8 GB |

All fit within 48 GB M4 Max.
