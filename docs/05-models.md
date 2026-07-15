# Model registry and Qwen 3.6

## Storage

Mio keeps target and speculative checkpoints separate:

```text
models/
└── Qwen3.6-27B-UD-Q4_K_XL-mlx/
spd/
├── Qwen3.6-27B-DSpark/
└── Qwen3.6-27B-DFlash/
```

Weights are local runtime data and are not versioned in Git.

## Tested Qwen 3.6 27B stack

| Role | Repository | Local directory |
|---|---|---|
| target | `Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx` | `models/Qwen3.6-27B-UD-Q4_K_XL-mlx` |
| preferred drafter | `Avesed/Qwen3.6-27B-DSpark` | `spd/Qwen3.6-27B-DSpark` |
| fallback drafter | `z-lab/Qwen3.6-27B-DFlash` | `spd/Qwen3.6-27B-DFlash` |

Registry compatibility metadata observed by the benchmark:

- hidden size: 5120;
- target layers: 64;
- vocabulary: 248,320;
- DSpark block size: 7, with a Markov/confidence head;
- DFlash block size: 16;
- effective sliding window: 2048;
- registered maximum context: 262,144 tokens.

The target uses mixed 4-bit/BF16 weights. Both drafter checkpoints are
separate and never replace target verification. Mio classifies checkpoint
metadata, selects DSpark for the hybrid DFlash+Markov layout, and falls back
only to a distinct pure DFlash checkpoint. Compatibility is validated before
generation and the selected backend/reason is exposed in engine telemetry.

## Download

```bash
mio pull large
```

Or explicitly:

```bash
hf download Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --local-dir models/Qwen3.6-27B-UD-Q4_K_XL-mlx
hf download Avesed/Qwen3.6-27B-DSpark \
  --local-dir spd/Qwen3.6-27B-DSpark
hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir spd/Qwen3.6-27B-DFlash
```

Authenticate outside the repository. If a source requires an access token,
use `hf auth login` or a scoped secret environment; never paste the token into
committed commands or documentation.

## Completeness and resolution

A local directory is considered complete only when:

1. `config.json` exists;
2. if a SafeTensors index exists, every referenced shard exists and is
   non-empty;
3. otherwise, at least one non-empty `.safetensors` file exists.

This prevents an interrupted pull from poisoning automatic tier selection.
When the complete local Qwen 3.6 27B target/DFlash pair exists, `large` selects
that target generation. A complete DSpark checkpoint becomes the preferred
drafter; DFlash remains the load-failure fallback. If the target or DFlash
side is missing, resolution falls back to the registered Qwen 3.5 27B pair.

```bash
python3 -m mio.model_check
mio pull
```

## Tier behavior

| Tier | Selection policy | Registered context |
|---|---|---:|
| `large-moe` | Qwen 3.6 35B-A3B local pair, Qwen 3.6 community pair, then Qwen 3.5 35B-A3B | 256K or 128K |
| `large` | Qwen 3.6 27B UD-Q4_K_XL local pair, Qwen 3.6 community pair, then Qwen 3.5 27B | 256K or 32K |
| `medium` | Qwen 3.5 9B UD-Q4_K_XL pair | 16K |
| `small` | Qwen 3.5 4B 4-bit pair | 8K |

The top-level persisted active tier defaults historically to `large-moe`.
Use `--tier large` when the Qwen 3.6 27B pair is intended; do not infer the
loaded architecture from the word "default".

## Loading sequence

1. Build a `TierConfig` from current registry defaults and persisted values.
2. Resolve complete local paths or retain the Hugging Face reference.
3. Load the target through the standard or PARO-specific MLX path.
4. Classify the requested drafter metadata as DSpark, DFlash, or hybrid.
5. Load DSpark when compatible; on load failure use only a validated, distinct
   pure DFlash fallback unless strict mode is enabled.
6. Validate architecture/shape metadata and bind the selected drafter.
7. Apply an explicit capability policy for sampling, tool EOS, caches, BMP,
   and DDTree; unsupported combinations must be observable rather than silent.
8. Dispatch to the selected drafter; retain target AR as the
   control/fallback.

A missing/unreadable optional draft may fall back to target AR. A draft that
loads but declares an incompatible target must fail before generation rather
than silently producing untrusted output.

## Other registry entries

The registry includes Qwen 3, Qwen 3.5, Qwen 3.6, PARO variants, and entries
whose adapters are not implemented. `mio pull` lists the authoritative key
set. Presence in `KNOWN_MODELS` does not mean a pair has been downloaded,
tested on this machine, or supported by the active adapter set.

## Current evidence

The versioned Qwen 3.6 result verifies exact greedy token parity for 32 output
tokens over two DFlash repetitions. It is not a full model-quality validation.
Required follow-up includes longer parity corpora, tool-call transcripts,
multiple prompt lengths, context-boundary tests, and independent reruns.
