# Getting started

## Requirements

- macOS on Apple Silicon;
- Python 3.10 or newer;
- Git;
- enough disk and unified memory for the selected checkpoint;
- optional: Hugging Face CLI for explicit downloads, Node.js for Ponytail MCP,
  and Tesseract for OCR tools.

The measured Qwen 3.6 27B target occupies about 20-22 GB on disk before the
DSpark/DFlash checkpoints and runtime state. A 48 GB Mac was used for the checked-in benchmark.
Smaller tiers are available for lower-memory machines.

## Install Mio

```bash
git clone https://github.com/Ruler-Dev/mio.git
cd mio
python3 -m pip install -e .
mio --help
```

Verify the installed dependency graph when changing environments:

```bash
python3 -m pip check
python3 -c 'import mlx; print(mlx.__version__)'
```

## Download the tested Qwen 3.6 stack

```bash
mio pull large
python3 -m mio.model_check
```

`mio pull large` resolves to:

- `models/Qwen3.6-27B-UD-Q4_K_XL-mlx`;
- `spd/Qwen3.6-27B-DSpark` as the preferred drafter;
- `spd/Qwen3.6-27B-DFlash` as an independently compatible fallback.

Use `mio pull large --no-dspark` for a target+DFlash installation or
`mio pull large --no-fallback` for target+DSpark without a local fallback. The
default downloads all three so runtime fallback never reuses the hybrid
DSpark checkpoint as if it were a pure DFlash model.

The downloader resumes into stable local directories and considers a model
complete only when `config.json` and every declared SafeTensors shard exist.
Never commit model weights or Hugging Face credentials.

## Install the external skills inside Mio

```bash
python3 scripts/install_mio_skills.py
```

The command installs the pinned, installer-verified managed snapshot (916
skills at the reviewed revisions) under `~/.mio/skills`. Unmanaged local
skills can make the live discovery total differ. The installer does not modify
`~/.codex`, `~/.claude`, or another product's configuration. See
[14 — External skills](14-external-skills.md).

## Check Mio-local MCP

```bash
mio mcp install-tools
mio mcp doctor
mio mcp check --json
mio mcp list
```

The local `llm-wiki`, `headroom`, and `ponytail` presets are enabled in the
registry by default and exposed through bounded discovery/call tools. Provider
processes start lazily on first use. In Mio UI, local MCP orchestration is a
sensitive capability: a model request needs both the exact operator grant and
per-request consent before it can use the bridge; a direct sensitive run also
needs confirmation. LLM Wiki ships with Mio; the packaged installer creates
the isolated Headroom and Ponytail runtime trees. The source-checkout script
is only a thin compatibility wrapper. Missing optional providers are isolated
from engine startup. See [15 — MCP](15-mcp.md).

## First run

Use the measured dense tier explicitly:

```bash
mio --tier large --workspace .
```

The agent displays its selected roots and capabilities. It prefers the nearest
Git root, grants READ/WRITE/SHELL there, and keeps network disabled unless the
session explicitly adds `--agent-network`. Use repeatable `--agent-root` only
for additional directories the agent is meant to inspect or modify.

Other entry points:

```bash
mio chat --tier large --prompt-mode none
mio serve --tier large
mio serve --tier large --webui
```

Open the UI at `http://127.0.0.1:9090/ui`. The server binds to loopback by
default. Non-loopback binds are rejected unless the operator supplies
`--unsafe-remote-bind`; that opt-in does not add authentication.

## Prompt policies

For the native agent, chat, or server, select one policy:

```bash
mio serve --prompt-mode none
mio serve --prompt-mode caveman --prompt-level full
mio serve --prompt-mode ponytail --prompt-level full
mio --ponytail full --tier large
```

The default is `caveman/full`. Caveman and Ponytail are prompt policies, not
inference accelerators. Their effect on Qwen 3.6 coding quality and total task
time has not yet been measured.

## Context and cache overrides

```bash
mio serve --tier large --context 32k
mio serve --tier large --context 128k
mio serve --tier large --tq4
mio serve --tier large --mpath 2
```

`--tq4` selects TurboQuant and disables PolarQuant for that run. `--mpath 2`
is experimental BMP verification. Do not assume either is faster: the current
256+32-token Qwen 3.6 ablation found TQ4 slower end to end and PQ4 changed the
greedy output. Benchmark the intended prompt lengths first.

## Minimal verification

```bash
python3 -m pytest -q
python3 scripts/bench_qwen36_matrix.py \
  --tier large --prompt-tokens 256 --max-tokens 32 \
  --warmup 1 --reps 2 --modes baseline,dflash
```

For a release or research claim, use more repetitions and multiple prompt
lengths/workloads. The two-repetition command above is only a quick local
parity/performance smoke test.

## Paths

| Path | Purpose |
|---|---|
| `models/` | target checkpoints |
| `spd/` | DSpark and DFlash speculative checkpoints |
| `~/.mio/config.json` | persisted engine/server configuration |
| `~/.mio/mcp.json` | Mio MCP overrides |
| `~/.mio/skills/` | Mio external instruction skills |
| `~/.mio/tools/` | isolated optional MCP tools/source trees |
| `~/.mio/wiki/` | local LLM Wiki data |
| `~/.mio/sessions/` | Web UI sessions |
| `benchmarks/results/` | versioned raw benchmark artifacts |

Set `MIO_HOME` for the skill installer when a different Mio data root is
needed. Not every older subsystem has been migrated to that environment
variable yet; centralizing all persistence is tracked in the development plan.
