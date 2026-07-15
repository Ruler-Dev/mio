# Configuration and prompt policies

## Configuration precedence

For server settings and tiers, Mio applies:

1. current registry defaults;
2. persisted `~/.mio/config.json` values;
3. one-shot CLI overrides.

Missing fields inherit current defaults. Unknown fields are ignored. Malformed
custom tiers do not prevent every Mio command from starting.

```bash
mio configure
```

The wizard persists active tiers, tandem mode, host/port, and each tier's
model, context, sampling, cache, and speculative settings.

## Tier fields

Important `TierConfig` fields:

| Field | Current dataclass default | Meaning |
|---|---:|---|
| `context_window` | model-specific | maximum prompt/context budget exposed by the tier |
| `max_output_tokens` | model-specific | configured output cap |
| `temperature` | `0.0` | exact greedy speculation; positive values use exact DSpark sampling or target-only fallback for greedy-only backends |
| `top_p` | `0.95` | nucleus probability |
| `top_k` | `20` | top-k filter |
| `pq_bits` | `4` | PolarQuant selected; `16` disables it |
| `tq_bits` | `16` | TurboQuant off; `2`, `3`, or `4` enables it |
| `bmp_paths` | `1` | vanilla DFlash; values above one request BMP |
| `ddtree_budget` | `0` | DDTree off |

PQ and TQ are mutually exclusive. A legacy persisted file containing active
values for both is normalized in favor of the explicitly configured TQ mode.

The DFlash and DDTree verifiers use exact greedy acceptance. Mio therefore
keeps an omitted/zero API temperature on those accelerated paths. DSpark can
perform exact speculative sampling with the requested `top_p`, `top_k`, and
`seed`; when the selected backend is greedy-only, Mio reports an explicit
MLX-LM target-only fallback rather than applying a biased sampler. Textual `stop` values are removed from
both complete and streamed output, including matches split across chunks.
They currently filter exposed text rather than cancelling the underlying
backend iterator, so they may not reduce generation compute.

## One-shot context/cache overrides

```bash
mio --tier large --context 32k
mio chat --tier large --context 64k --tq4
mio serve --tier large --context 128k --mpath 2
```

`--context` accepts suffix forms such as `32k` or a raw integer. Mio clamps
the one-shot output limit to at most one quarter of that context and 8192
tokens. Large advertised model contexts can still exceed the machine's usable
memory.

## Prompt policies

Prompt policy is independent from model execution and cache selection.
`mio chat` and `mio serve` support:

| Mode | Levels | Behavior |
|---|---|---|
| `none` | none | do not inject a Mio policy |
| `caveman` | `lite`, `full`, `ultra` | request increasingly concise prose |
| `ponytail` | `lite`, `full`, `ultra` | request the smallest sufficient engineering change |

Examples:

```bash
mio chat --prompt-mode none
mio chat --prompt-mode caveman --prompt-level lite
mio serve --prompt-mode ponytail --prompt-level full
```

Legacy aliases:

```bash
mio serve --caveman off
mio serve --caveman ultra
mio serve --ponytail lite
```

Mode selectors are mutually exclusive. `--prompt-level` is valid only with
`--prompt-mode caveman` or `--prompt-mode ponytail`. With no selector,
agent/chat/server use `caveman/full`.

The policy text is prepended to an existing system message or inserted as a
new one. Mio skips policy injection when the leading system prompt contains a
known exact XML tool-protocol marker, because changing those instructions can
break external clients.

### Evidence boundary

No versioned Qwen 3.6 task corpus currently demonstrates a token reduction,
coding-success gain, or tool-accuracy improvement for either policy. Levels
describe instruction intensity, not measured percentages. A future harness
must compare task success, retries, tool correctness, elapsed time, and output
tokens against `none`.

## Native agent policy state

The native agent uses the same `none`/`caveman`/`ponytail` policy object and
defaults to `caveman/full`. It supports top-level flags and runtime commands:

```bash
mio --prompt-mode ponytail --prompt-level full
mio --caveman lite
```

```text
/caveman off|lite|full|ultra
/ponytail off|lite|full|ultra
```

## MCP configuration

MCP has a separate file because it contains transport and permission policy:

```bash
mio mcp list
mio serve --mcp-config ~/.mio/mcp.json
```

Set `MIO_MCP_CONFIG` or pass `--mcp-config`/`--config` to use another file.
Provider toggles are persisted with mode `0600`. See [15 — MCP](15-mcp.md).

## Relocating Mio data

The external skill installer honors `MIO_HOME`:

```bash
MIO_HOME=/Volumes/Fast/Mio python3 scripts/install_mio_skills.py
```

Some older Web UI/config modules still use `~/.mio` directly. Until the
centralized data-root task lands, test relocation subsystem by subsystem.
