# Mio — Install & Use

Everything you need to get Mio running on an Apple Silicon Mac, from zero to a working chat + API + coding agent.

---

## 1. Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python ≥ 3.10
- ~25 GB free disk per large-tier model
- 48 GB unified memory recommended (16 GB minimum — `small` tier only)
- Optional: `tesseract` system binary for OCR (`describe_image`). Install with `brew install tesseract`.

---

## 2. Install

```bash
git clone https://github.com/Ruler-Dev/mio.git
cd mio
pip install -e .
```

One command. This pulls in the full stack: inference engine, agent, OpenAI-compatible API, Mio UI, all 99 skills, and `mlx-vlm` for multimodal loaders. No extras to remember.

Verify:
```bash
mio --help
```

---

## 3. Download a model

```bash
mio pull large-moe       # Qwen 3.5 35B-A3B MoE target + DFlash draft (~22 GB)
mio pull large           # Qwen 3.5 27B + draft (~14 GB)
mio pull medium          # Qwen 3.5 9B + draft (~5 GB)
mio pull small           # Qwen 3.5 4B + draft (~2.5 GB)
mio pull                 # list every tier + raw model key, no download
```

Tier shortcuts map to Unsloth UD-Q4_K_XL re-quantizations (tool-call safe, imatrix-calibrated). Files land in `models/` (target) and `spd/` (DFlash draft) under the repo root.

### Using Qwen 3.6 (256K context)

The 3.6 DFlash draft is HF-gated, so `mio pull` can't fetch it. To use 3.6:

1. Request access to `z-lab/Qwen3.6-35B-A3B-DFlash` on HuggingFace.
2. Once approved:
   ```bash
   hf download Brooooooklyn/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx --local-dir models/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx
   hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir spd/Qwen3.6-35B-A3B-DFlash
   ```
3. No config edit needed — Mio auto-detects the 3.6 folders and uses them for `large-moe` automatically (256K context). Remove either folder to fall back to 3.5.

---

## 4. Run Mio

Three ways, pick whichever fits.

### A. Native coding agent (terminal)

```bash
mio                                  # default tier (large-moe), interactive REPL
mio --tier medium                    # 9B model instead
mio --context 32k                    # shorter context, less memory
mio --tandem                         # load every tier, route by complexity
mio "write a quicksort in python"    # optional initial prompt
```

Inside the agent:

| Slash command | Purpose |
|---------------|---------|
| `/model` | Current tier + context + cache config |
| `/tier large-moe\|large\|medium\|small` | Switch tier (reloads if needed) |
| `/context` | Interactive context window + cache-mode picker |
| `/caveman off\|lite\|full\|ultra` | Toggle system-prompt compression |
| `/status` | Loaded tiers, VRAM, last tok/s |
| `/models` | Registry of known model keys |
| `/configure` | Run config wizard inline |
| `/clear` | Wipe conversation history |
| `/help` | Show all commands |
| `/quit` | Exit |

The agent has real tools: `bash`, `read`, `write`, `edit`. Up to 5 rounds of generation ↔ tool execution per turn, so it can actually make changes to your project.

### B. OpenAI-compatible API server

```bash
mio serve                            # :9090, default tier
mio serve --tier medium --context 32k
mio serve --tandem                   # all tiers, TandemRouter picks per request
mio serve --webui                    # + Mio UI at http://localhost:9090/ui
```

Point any OpenAI-SDK client at `http://localhost:9090/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9090/v1", api_key="local")
response = client.chat.completions.create(
    model="mio-large-moe",
    messages=[{"role": "user", "content": "Write quicksort in Python"}],
    stream=True,
)
```

Works as a backend for Cline, Kilo Code, Aider, Continue, and any OpenAI-spec client. Endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/batch`, `/health`, `/metrics`, `/dashboard` (live metrics page), `ws://localhost:9090/ws/metrics`.

### C. Mio UI — Claude-style chat

```bash
mio serve --webui
```

Open http://localhost:9090/ui. What's there:

- **Artifacts** — ask for a diagram and get Mermaid / Graphviz / d3-force / mindmap rendered in a side panel. Ask for a chart, a React component, a three.js scene, a Pyodide REPL, a Tone.js synth, a flashcard deck — you get a working interactive app embedded in the chat. ~100 artifact types.
- **Skills** — 99 tools the model can call: web search, fetch_url, anime / manga / movie / TV / game lookups, PDF / DOCX / XLSX / PPTX generation (64 PDF presets × 39 colors), local RAG (SQLite FTS5), todos, habits, journal, bookmarks, code execution (sandboxed Pyodide), regex explainer, symbolic math, etc.
- **Personas** — `/as teacher`, `/as skeptic`, `/as pirate`, `/as haiku` … 27 total, `/as-list` shows all.
- **Workspace** at `/ui/dashboard` — scheduled prompts, webhook triggers, indexed RAG folders.
- **Stats** at `/ui/stats`, **Playground** at `/ui/playground`, **Attachments** at `/ui/attachments`, **Compare** at `/ui/compare`.
- **Keyboard** — `⌘K` command palette, `⌘N` new chat, `⌘F` in-chat find, `⌘⇧V` paste-as-hidden-context, `⌘/` shortcut cheatsheet.
- **Persistent state** — everything lives under `~/.mio/` (sessions, projects, todos, habits, journal, RAG index, caches). Wipeable from Settings → Cache.

---

## 5. Common recipes

```bash
# Serve large-moe with a smaller context (free memory for other apps)
mio serve --tier large-moe --context 32k

# Serve small 4B model with a bigger context than its default
mio serve --tier small --context 64k

# Open the UI with medium tier at 32K
mio serve --tier medium --context 32k --webui

# Coding agent on medium tier with 64K context
mio --tier medium --context 64k

# Headless chat (no tools) on small tier at 32K
mio chat --tier small --context 32k

# Tandem mode: all tiers, auto-routed by complexity
mio serve --tandem --webui

# Batch inference
mio batch --input prompts.jsonl --output results.jsonl --tier medium
```

`--context` accepts `8k`, `16k`, `32k`, `64k`, `128k`, `256k`, or a raw integer.

---

## 6. Configuration

Defaults live at `mio/models/registry.py` (`DEFAULT_TIERS`, `KNOWN_MODELS`) and `mio/config.py` (`TierConfig`).

Overridable at runtime via CLI flags (`--tier`, `--context`, `--tq4`, `--mpath`, `--caveman`) or persistently via:

```bash
mio configure
```

Interactive wizard writes to `~/.mio/config.json`.

### Caveman modes

| Level | Behavior | Output token reduction |
|-------|----------|------------------------|
| `off` | Standard | baseline |
| `lite` | No filler, full sentences (agent default) | ~15% |
| `full` | Drop articles, fragments OK (server default) | ~40% |
| `ultra` | Telegraph style, abbreviations, arrows | ~75% |

Code blocks are never compressed.

### KV-cache

| Mode | Setting | Notes |
|------|---------|-------|
| PolarQuant 4-bit *(default)* | `pq_bits = 4` | Zero speed overhead; faster than uncompressed because reads 4× less memory |
| Off (fp16 cache) | `pq_bits = 16` | More memory, same speed |
| TurboQuant 4/3/2-bit | `tq_bits = 4/3/2` | Legacy path; slower than PQ4. `--tq4` enables. |

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `describe_image` fails with tesseract error | `brew install tesseract` |
| `mio pull large-moe` — 404 on draft | Check your HuggingFace login (`hf auth login`). Qwen 3.6 draft is gated — use 3.5 instead. |
| Out of memory on large-moe | Use `--context 32k` or switch to `--tier medium` |
| Web UI artifacts render blank | Check browser console; `--webui` requires the full dependency install |
| Model loads but generation is garbled | You may have a non-Unsloth 4-bit quant — tool-call data is often miscalibrated. Use an Unsloth UD-Q4_K_XL variant (`mio pull` gives you these by default) |
| Port 9090 already in use | `mio serve --port 9100` |

---

## 8. Where things live

| Path | What |
|------|------|
| `models/` | Target model weights (populated by `mio pull`) |
| `spd/` | DFlash draft weights |
| `~/.mio/config.json` | User config (tier, context, caveman default) |
| `~/.mio/sessions/` | Mio UI chat history (one JSON per session) |
| `~/.mio/projects.json`, `prompts.json`, `memory.json` | Mio UI project/prompt/memory library |
| `~/.mio/rag.sqlite` | Local RAG full-text index |
| `~/.mio/todos.sqlite`, `habits.sqlite`, `bookmarks.sqlite` | Productivity skills |
| `~/.mio/schedules.json`, `webhooks.json` | Workspace dashboard state |
| `~/.mio/image-cache/`, `web-cache/`, `files-cache/` | Skill output caches — wipeable from Settings → Cache |
| `~/.mio/journal/<YYYY-MM-DD>.md` | Daily journal entries |

Everything under `~/.mio/` is local, yours, and safe to delete if you want to reset.

---

## 9. Uninstall

```bash
pip uninstall mio
rm -rf ~/.mio              # user data (chats, todos, cache, …)
rm -rf models/ spd/        # model weights, if not needed elsewhere
```

---

Questions, bugs, or patches: open an issue or PR at https://github.com/Ruler-Dev/mio. It's MIT — fork it and make it yours.
