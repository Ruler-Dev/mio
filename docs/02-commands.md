# CLI Commands Reference

## `mio` (no subcommand)

Launches the **native coding agent** with the default configuration: Qwen3.5-35B-A3B MoE PARO, 128K context, TQ 4-bit, Caveman Ultra.

```bash
mio                              # Default: large-moe tier
mio --tier medium                # Agent with 9B model
mio --tier large                 # Agent with dense 27B model
mio --tandem                     # All tiers loaded, auto-routing
mio "fix the bug in auth.py"     # Start with initial prompt
mio --tq4                        # Enable TurboQuant 4-bit KV cache
mio --mpath 2                    # Enable Batched Multi-Path DFlash (K=2)
```

### Performance flags (all default off)

| Flag | Effect | When to use | Tradeoff |
|------|--------|------------|----------|
| `--tq4` | TurboQuant 4-bit KV cache | Want bigger contexts than fp16 KV fits | KV ≈ 28% of fp16; decode 0.7-0.9× of baseline; on 27B-dense at 32K decode is 1.67× faster |
| `--mpath K` | Batched Multi-Path DFlash (K paths verified per cycle) | Pure-attention models on math-like prompts | Slower on Qwen3.5; on Qwen3-8B math: 1.26× |
| (auto) | Prefix cache | Always-on; auto-detects shared prompt prefixes | 4-8× TTFT win on warm hits when consecutive prompts share ≥64-token prefix; off when --tq4 or --mpath ≥2 |

### Agent Slash Commands

| Command | Description | Reloads? |
|---------|-------------|----------|
| `/model` | Show current model, tier, context, TQ config | No |
| `/tier large-moe\|large\|medium\|small` | Switch model tier | **Yes** |
| `/context` | Interactive context window + TQ selector | **Yes** |
| `/caveman off\|lite\|full\|ultra` | Toggle communication mode | No |
| `/tq` | Show TurboQuant V2 cache settings | No |
| `/status` | Engine status: loaded tiers, VRAM, tok/s | No |
| `/models` | List all available model keys | No |
| `/configure` | Run full configuration wizard | No |
| `/clear` | Clear conversation history | No |
| `/help` | Show all commands | No |
| `/quit` | Exit | No |

Commands marked **Yes** under "Reloads?" will unload the current model, apply the new settings, and reload. You'll see "Reloading model..." and "Engine ready:" messages.

### `/context` Interactive Selector

Type `/context` to get numbered menus for context window and TQ cache:

```
Select context window:
  [1]    8K  (    8,192 tokens)
  [2]   16K  (   16,384 tokens)
  [3]   32K  (   32,768 tokens)
  [4]   64K  (   65,536 tokens)
  [5]  128K  (  131,072 tokens) <-- current
  [6]  256K  (  262,144 tokens)
Context [5]:

Select TurboQuant cache:
  [1] TQ 4-bit     (3.6x compression, best speed) <-- current
  [2] TQ 3-bit     (4.7x compression, moderate quality loss)
  [3] TQ 2-bit     (5.5x compression, max context)
  [4] OFF          (no compression, fp16 cache)
TQ mode [1]:
```

Press Enter to keep current value, or type a number to change.

## `mio serve`

Start the OpenAI-compatible API server.

```bash
mio serve                        # Default: large-moe tier, port 9090
mio serve --tandem               # All tiers, auto-routing
mio serve --tiers large-moe,medium  # Specific tiers
mio serve --tier medium          # Single tier
mio serve --port 8080            # Custom port
mio serve --host 127.0.0.1      # Localhost only
mio serve --validate             # Enable code validation + auto-retry
mio serve --tq4                  # TurboQuant 4-bit KV cache
mio serve --mpath 2              # BMP-DFlash K=2
mio serve --caveman lite         # Caveman level (default ultra)
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Generate (streaming SSE or non-streaming) |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/models/load` | Load a tier dynamically |
| POST | `/v1/models/unload` | Unload a tier to free VRAM |
| POST | `/v1/batch` | Batch inference (array of requests) |
| GET | `/health` | Server health, loaded models, VRAM |
| GET | `/metrics` | Per-tier generation stats |
| GET | `/dashboard` | Live web monitoring dashboard |
| WS | `/ws/metrics` | WebSocket live metrics stream |

### Model Names in API

| Model Name | Routes To |
|---|---|
| `mio-large-moe` | Large MoE tier (35B-A3B, default) |
| `mio-large` | Large dense tier (27B) |
| `mio-medium` | Medium tier (9B) |
| `mio-small` | Small tier (4B) |
| `mio-auto` | Auto-route by complexity (tandem only) |

## `mio chat`

Simple chat -- no agent tools, no slash commands, just raw LLM conversation with streaming.

```bash
mio chat                         # Default tier (large-moe)
mio chat --tier medium           # Chat with 9B
mio chat --no-caveman            # Disable caveman system prompt
mio chat --tq4                   # TQ4 KV cache
mio chat --mpath 2               # BMP-DFlash K=2
```

## `mio configure`

Interactive 5-step wizard: select model, DFlash draft, TQ cache bits, context window, and tier. Shows VRAM estimates for every option.

```bash
mio configure
```

## `mio pull`

Download target model + DFlash draft in one command.

```bash
mio pull                         # List all available model keys
mio pull qwen3.5-35b-a3b        # Download 35B-A3B + its DFlash draft
mio pull qwen3.5-27b            # Download 27B + draft
mio pull qwen3.5-9b             # Download 9B + draft
mio pull qwen3.5-4b             # Download 4B + draft
```

## `mio download`

Download all default tier models from HuggingFace.

```bash
mio download                     # All tiers
mio download --tier large-moe    # Just the default MoE model
```

## `mio batch`

Batch inference from JSONL file.

```bash
mio batch --input prompts.jsonl --output results.jsonl
mio batch --input prompts.jsonl --output results.jsonl --tier medium
```

## `mio bench`

Benchmark all tiers.

```bash
mio bench
```

## `mio status`

Show configuration, known models, and running server status.

```bash
mio status
```

## `mio menu`

Legacy interactive menu with numbered options.

```bash
mio menu
```
