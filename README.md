# Mio

> *"Ci sono tanti engine ma questo è Mio, e quindi anche tuo!"*
> — *"There are many engines, but this one is Mio (mine), and so it's yours too!"*

**A local inference engine for Apple Silicon — DFlash speculative decoding + PolarQuant KV-cache compression + Caveman system-prompt token compression — with a coding agent and a full Claude-style chat UI built on top.**

Everything runs on-device via MLX. No cloud, no API keys, no Node.js. One `pip install -e .` and the engine is ready.

Mio is a **passion project** for everyone running models on their own hardware. **Fork it, hack it, make it really yours** — pull requests, issues, and wild ideas all welcome. Local AI should belong to the people running it, not to the people serving it.

---

## What Mio is (from the bottom up)

```
                           ┌─────────────────────────┐
                           │  Mio UI · Skills · App  │   ← Claude-style chat
                           ├─────────────────────────┤
                           │  Agent · OpenAI server  │   ← CLI & API
                           │   + Caveman mode        │     (system-prompt compression)
                           ├─────────────────────────┤
                           │        MioEngine        │   ← inference core
                           │  DFlash  +  PolarQuant  │
                           ├─────────────────────────┤
                           │          MLX            │   ← Apple's Metal runtime
                           └─────────────────────────┘
```

The **engine** is the point. It wires together two pieces of research that make local inference genuinely fast on Apple Silicon:

- **DFlash speculative decoding** — up to ~4.1× generation speedup, ~0.77 acceptance on Qwen 3.5/3.6 at 4-bit.
- **PolarQuant 4-bit KV-cache compression** — ~3.8× cache reduction with *zero* speed overhead. The quantized matmul actually runs faster because it reads 4× less memory.

…then **Caveman mode** sits on top, injected as a system prompt that tells the model to be tight without losing precision in code. DFlash and PolarQuant make each token cheaper; Caveman makes the model emit fewer tokens. Multiplied together, end-to-end task time drops by 10–20× vs vanilla MLX (see [Benchmarks](#benchmarks)).

On top of that we ship a native **coding agent** (slash commands, tool use), an **OpenAI-compatible API server** (drop-in for Ollama / LM Studio / any OpenAI SDK), and **Mio UI** — a Claude-style chat application with artifacts, skills, personas, dashboards, and a voice loop.

---

## Default stack — all enabled out of the box

| Layer | What | Why it matters |
|-------|------|----------------|
| **Model** | Qwen 3.5 35B-A3B MoE (Unsloth UD-Q4_K_XL) | 35B total, 3B active per token, 128K context |
| **DFlash** | Speculative decoding | Up to 4.1× generation speedup |
| **PolarQuant 4-bit** | KV-cache compression | ~3.8× cache reduction, ~26% faster inference |
| **Caveman mode** | System-prompt token compression (lite/full/ultra) | 15–75% fewer output tokens — see [Caveman section](#caveman-mode--system-prompt-token-compression) |

Additional engine features enabled automatically:
- **Relaxed prefix cache** with KV truncation — longest-common-prefix reuse across multi-turn conversations
- **LM-head slicing** during prefill — only project the last position's logits (+13–15% prefill tok/s)
- **Context auto-compaction** at 75% of context window (tool-result truncation, then LLM summarization — prevents OOM on long sessions)
- **OpenAI function calling** with conditional EOS suppression so DFlash acceptance stays healthy during multi-turn tool use

---

## Quick start

```bash
cd mio
pip install -e .             # one shot: engine + agent + API + Mio UI + all skills
mio pull large-moe           # ~22 GB: Qwen 3.5 35B-A3B target + DFlash draft
mio                          # coding agent, default tier (large-moe)
mio serve --webui            # API server + Mio UI at http://localhost:9090/ui
```

No extras, no separate UI install — one command pulls in the full dependency set (docx / xlsx / pptx / reportlab / sympy / jwt / feedparser / mlx-vlm / etc.). OCR (`describe_image`) additionally needs the `tesseract` system binary (`brew install tesseract`).

`mio pull <tier>` accepts `large-moe`, `large`, `medium`, or `small` and places target weights in `models/` and DFlash draft weights in `spd/`. Run `mio pull` with no arguments to list every tier and raw model key.

### Common recipes

```bash
# Serve the default large-moe tier with a smaller context (frees memory for other apps)
mio serve --tier large-moe --context 32k

# Serve the small 4B model but with a much bigger context window than its default
mio serve --tier small --context 64k

# Open the chat UI at http://localhost:9090/ui with the medium 9B at 32K
mio serve --tier medium --context 32k --webui

# Native coding agent on the medium tier with 64K context
mio --tier medium --context 64k

# Headless chat (no tools) on the small tier at 32K
mio chat --tier small --context 32k

# Tandem mode: load every tier and let the router pick by complexity
mio serve --tandem --webui
```

`--context` accepts `8k`, `16k`, `32k`, `64k`, `128k`, `256k`, or a raw integer (e.g. `--context 96000`). Setting a smaller context than the tier default frees memory; setting a larger one costs memory and may slow prefill.

---

## Model tiers

All defaults are **Unsloth MLX UD-Q4_K_XL** re-quantizations (Brooooooklyn), imatrix-calibrated on tool-calling data — fixes the tool-call degradation from vanilla RTN INT4 (mlx-lm issue #1011).

| Tier | Model | VRAM | Default context | Max context | Speed |
|------|-------|------|-----------------|-------------|-------|
| **large-moe** *(default)* | Qwen 3.5 35B-A3B MoE · UD-Q4_K_XL | ~22 GB | **128K** | 128K | ~204 tok/s |
| large | Qwen 3.5 27B · UD-Q4_K_XL | ~14 GB | 32K | 32K | ~56 tok/s |
| medium | Qwen 3.5 9B · UD-Q4_K_XL | ~5 GB | 16K | 16K | ~108 tok/s |
| small | Qwen 3.5 4B · mlx-community 4-bit | ~2.5 GB | 8K | 8K | ~187 tok/s |

Switch at runtime with `--tier <name>` or `/tier <name>` in the agent, or let **Tandem mode** (`--tandem`) route by complexity across all loaded tiers.

> **Qwen 3.6 (256K context, faster) — auto-detected if locally present.** The 3.6 architecture is supported but its DFlash draft repo is HF-gated, so `mio pull` cannot fetch it. To use 3.6, request access on HuggingFace and download it manually:
>
> ```bash
> hf download Brooooooklyn/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx --local-dir models/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx
> hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir spd/Qwen3.6-35B-A3B-DFlash
> ```
>
> No config edit required: as soon as both directories exist, `large-moe` resolves to 3.6 (256K context) automatically. Remove either dir to fall back to 3.5 (128K context). The mlx-community 4-bit variant (`Qwen3.6-35B-A3B-4bit`) is detected the same way.

---

## CLI

Every subcommand and its full flag set, with defaults.

### `mio` *(no subcommand)* — interactive coding agent

```bash
mio [prompt]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tier {large-moe,large,medium,small}` | `large-moe` | Tier to load |
| `--tandem` | off | Load all tiers, route by complexity |
| `--paro` | off | Use PARO-quantized weights (higher quality, slower) |
| `--port` | `9090` | Reserved for future agent-side server |
| `--tq4` | off | Enable TurboQuant 4-bit KV-cache (PolarQuant 4-bit is on by default already) |
| `--mpath N` | `1` | Batched Multi-Path DFlash paths (1 = vanilla DFlash) |
| `--context SIZE` | tier default | Override context window (`8k` … `256k` or raw int) |
| `prompt` | none | Optional initial prompt; agent processes then drops you into the REPL |

### `mio serve` — OpenAI-compatible API server

```bash
mio serve [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `9090` | TCP port |
| `--host` | `0.0.0.0` | Bind host |
| `--tier` | `large-moe` | Single tier to load |
| `--tiers a,b,c` | unset | Comma-separated tiers to load (overrides `--tier`) |
| `--tandem` | off | Load every tier and let `TandemRouter` pick per request |
| `--context SIZE` | tier default | Override context window: `8k` / `16k` / `32k` / `64k` / `128k` / `256k` (or raw int) |
| `--validate` | off | Auto-run ruff + syntax check on generated code |
| `--caveman {off,lite,full,ultra}` | `full` | System-prompt token-saving mode |
| `--tq4` | off | Force-enable TurboQuant 4-bit cache for the loaded tiers |
| `--mpath N` | `1` | Batched Multi-Path DFlash paths |
| `--compact-threshold` | `0.75` | Trigger compaction when prompt > this fraction of ctx (set `1.0` to disable) |
| `--compact-target` | `0.50` | Compact down to this fraction of ctx |
| `--no-compact-summarize` | off | Heuristic-only compaction (skip stage-2 LLM summary) |
| `--webui` | off | Mount Mio UI at `/ui` |

### `mio chat` — headless chat (no tools, no agent)

```bash
mio chat [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tier` | `large-moe` | Tier to load |
| `--paro` | off | Use PARO-quantized weights |
| `--no-caveman` | off | Disable the Caveman Lite system prompt |
| `--tq4` | off | Enable TurboQuant 4-bit KV-cache |
| `--mpath N` | `1` | Batched Multi-Path DFlash paths |
| `--context SIZE` | tier default | Override context window (`8k` … `256k` or raw int) |

### `mio pull <tier|model-key>` — download target + draft

```bash
mio pull large-moe                # tier shortcut: Qwen 3.5 35B-A3B + draft
mio pull large                    # Qwen 3.5 27B + draft
mio pull medium                   # Qwen 3.5 9B + draft
mio pull small                    # Qwen 3.5 4B + draft
mio pull qwen3.5-27b-unsloth      # raw model key (any from `mio pull`)
mio pull                          # list every tier and key, no download
```

Target weights land in `models/<name>/`, DFlash drafts in `spd/<name>/`. If a target+draft already exist locally, the command becomes a no-op.

### `mio download` — bulk download default tier set

```bash
mio download                      # all default tiers (~30 GB)
mio download --tier large         # one specific tier
```

### `mio batch` — batch inference from JSONL

```bash
mio batch --input prompts.jsonl --output results.jsonl [--tier large-moe]
```

Input is one JSON object per line: `{"prompt": "..."}` or `{"messages": [...]}`. Output mirrors with appended `completion`, `tokens`, `tps`.

### `mio bench` — benchmark every tier

```bash
mio bench
```

Loads each tier in sequence, runs a fixed prompt, prints generation tok/s, prompt tok/s, e2e tok/s, DFlash acceptance, peak memory.

### `mio configure` — interactive wizard

Walks through tier selection, context window, PolarQuant / TurboQuant cache mode, BMP-DFlash path count, Caveman mode. Writes `~/.mio/config.json`.

### `mio status` — config + running-server probe

Prints active tier configuration table and pings `http://localhost:9090/health`.

### `mio menu` — interactive menu (calls the same subcommands)

### Agent slash commands

| Command | Description |
|---------|-------------|
| `/model` | Current model, tier, context config |
| `/tier large-moe\|large\|medium\|small` | Switch tier (reloads model if needed) |
| `/context` | Interactive context-window (8K / 16K / 32K / 64K / 128K / 256K) + TurboQuant cache mode (TQ 2/3/4-bit / OFF) selector |
| `/caveman off\|lite\|full\|ultra` | Token-saving system-prompt mode |
| `/tq` | Show TurboQuant status |
| `/status` | Engine status: loaded tiers, VRAM, tok/s |
| `/models` | List registered models |
| `/configure` | Run the configure wizard inline |
| `/clear` | Clear conversation history |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Context window + cache configuration

Each tier has a default context size matched to a sensible memory budget. You can change it in three ways:

1. **Interactively** while the agent is running: `/context` opens a numbered menu for window size (8K → 256K) and KV-cache mode (PolarQuant 4-bit default, TurboQuant 2/3/4-bit, or fp16 off).
2. **One-shot via the wizard**: `mio configure` writes the choice to `~/.mio/config.json`.
3. **In code**: edit `mio/models/registry.py` — every entry in `KNOWN_MODELS` carries `context_window` and `max_output_tokens`; `DEFAULT_TIERS` is built from those.

### Per-tier defaults (`mio/config.py:TierConfig`)

| Setting | Default | Notes |
|---------|---------|-------|
| `pq_bits` | `4` | PolarQuant 4-bit KV-cache. Set `16` to disable. |
| `tq_bits` | `16` (off) | TurboQuant. Set `2`/`3`/`4` to enable. PQ4 is preferred. |
| `bmp_paths` | `1` | Vanilla DFlash. `2`–`4` enable Batched Multi-Path verify. |
| `temperature` | `0.6` | Qwen 3.5/3.6 thinking-mode coding default. |
| `top_p` | `0.95` | |
| `top_k` | `20` | |
| `tq_group_size` | `64` | TurboQuant group size. |
| `tq_use_rotation` | `True` | Hadamard rotation when TQ active. |
| `tq_use_normalization` | `True` | |
| `pq_group_size` | `64` | PolarQuant group size. |

`mio/models/registry.py:DEFAULT_TIERS` provides the per-tier `context_window` / `max_output_tokens`. PARO variants are picked via `mio chat --paro` or `mio --paro`.

---

## API server

OpenAI-compatible at `http://localhost:9090/v1`. Drop-in replacement for Ollama and LM Studio:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9090/v1", api_key="local")
response = client.chat.completions.create(
    model="mio-large-moe",
    messages=[{"role": "user", "content": "Write quicksort in Python"}],
    stream=True,
)
```

Tool-calling works with Cline, Kilo, Roo, or any OpenAI-spec client. Additional endpoints: `/v1/models`, `/v1/batch`, `/health`, `/metrics`, `/dashboard` (live metrics web UI), WS `/ws/metrics`.

---

## Mio UI

`mio serve --webui` opens a Claude-style chat app at `http://localhost:9090/ui`.

### Artifacts (~100 types)

The model can wrap rendered content in `<antArtifact type="…">…</antArtifact>` and the UI opens it in a side panel. Categories:

- **Visual / diagramming** — HTML, SVG, Mermaid, Graphviz, mindmap, three.js, p5.js, Chart.js, Leaflet, KaTeX, reveal.js, timeline, WebGL shaders, Excalidraw, JSCAD, PlantUML, d3-force graph, Matter.js physics, WaveDrom, model-viewer + STL, spectrogram
- **Interactive / React** — React + Tailwind + lucide-react + recharts + framer-motion (esm.sh importmap + Babel Standalone), HTML playground, JSON tree, regex tester, sortable tables, diff viewer
- **Productivity / music / games** — Pyodide REPL, SQL runner, Tone.js synth, piano, drum machine, ABC notation, flashcards, Kanban, color palette, whiteboard, pomodoro, Wordle helper, countdown, QR, wavesurfer audio, YouTube player, fake terminal, API tester
- **Media cards** — `mediacard` for anime/manga/TV/games (auto-rendered from skill results), `youtubegrid`, `imagegrid`, `weather` (animated Meteocons)
- **File-backed** — PDF, DOCX, XLSX, PPTX, images

### Skills (99 registered tools)

All live in `mio/webui/skills*.py` and are registered in `skills.py`:

- **Web & data** — `web_search`, `fetch_url`, `url_preview`, `hn_top`, `reddit_top`, `wiki_summary`, `search_images`, `search_youtube`
- **Documents** — `generate_pdf_report` with **64 PDF presets × 39 color palettes** (auto-picked by keyword), specialized templates for letter / certificate / flyer / menu / brochure / newsletter / business card / resume / invoice, plus `generate_docx` / `generate_xlsx` / `generate_pptx` / `generate_markdown` (Obsidian-friendly)
- **Media lookup** — `find_anime`, `find_manga`, `find_movie_tv`, `find_game` (Jikan / TVmaze / RAWG)
- **Dev utilities** — `execute_python`, `http_request`, `explain_regex`, `hash_text`, `encode_decode`, `generate_uuid`, `generate_password`, `json_to_yaml`, `yaml_to_json`, `json_query`, `format_json`, `timezone_convert`, `unit_convert`, `symbolic_math`, `text_stats`, `decode_jwt`
- **Images** — `image_resize`, `image_convert`, `image_info`, `describe_image` (OCR + dominant colors), `generate_qr_code`
- **Files** — `merge_pdfs`, `split_pdf`, `extract_pdf_text`, `zip_files`, `unzip_file`, `generate_csv`, `generate_ical`, `generate_sqlite_db`
- **Productivity** — `todo_add/list/done/delete`, `habit_add/checkin/list`, `journal_append/read/search`, `bookmark_save/list/search`, `analyze_json`, `analyze_csv`, `scale_recipe`, `convert_currency` (offline), `review_code`, `meeting_notes`, `color_palette`
- **Misc** — `roll_dice`, `flip_coin`, `pick_random`, `generate_names`, `quote`, `translate_text`
- **Local RAG** — `index_folder`, `search_local_folder` (SQLite FTS5 at `~/.mio/rag.sqlite`)
- **Weather** — `get_weather` (Open-Meteo, no key; emits an animated widget artifact)

The server runs up to 5 rounds of generation ↔ tool execution per user turn, so a single prompt can chain `web_search` → `fetch_url` → `generate_chart` → emit artifact.

### Mio UI power features

- **27 personas** — `/as teacher`, `/as skeptic`, `/as chef`, `/as pirate`, `/as haiku`, `/as copy-editor`, `/as stoic`, `/as child` … each swaps the system prompt; `/as-list` shows the roster
- **Workspace dashboard** at `/ui/dashboard` — scheduled prompts (daily / weekly / interval), webhook triggers (`POST /ui/api/webhook/<slug>`), indexed RAG folders
- **Skill playground** at `/ui/playground` — every skill in a live try-it form
- **Stats dashboard** at `/ui/stats` — per-day message count, top skills, artifact types, personas invoked
- **Attachment library** at `/ui/attachments`
- **Side-by-side model compare** at `/ui/compare`
- **Chat templates**, **prompt library** (40 curated prompts), **chat import** from ChatGPT / Claude / Mio JSON
- **Voice I/O** — mic input, TTS responses, `/convo` hands-free voice loop
- **Drag-drop / paste images** (all MLX models in the default set are multimodal-capable)
- **`⌘K` command palette**, `⌘/` shortcut cheatsheet, `⌘F` in-chat find, `⌘⇧V` paste-as-context, `⌘N` new chat
- **Branching** — regenerate preserves every prior reply; ◀ ▶ cycles versions
- **Reactions**, pinned messages, starred chats, density toggle, accent-color chooser, 12 themes
- **Persistent state** under `~/.mio/` — sessions, projects, memory, bookmarks, todos, habits, journal, schedules, webhooks, RAG index, image cache, web cache; everything wipeable from Settings → Cache

All Mio UI features live under `mio/webui/` as separate modules (`assets/*.js`, `skills_*.py`, `scheduler.py`, `webhooks.py`). The shell `mio_ui.html` plus data files and per-feature JS modules are served from `mio/webui/assets/`.

---

## Caveman mode — system-prompt token compression

Caveman is the third pillar of the Mio stack, alongside DFlash and PolarQuant. It's a system prompt injected on every turn that tells the model to be tighter without losing precision. **Code blocks and commit messages are never compressed** — only narrative prose.

| Level | Behavior | Avg output token reduction |
|-------|----------|----------------------------|
| `off` | Standard model output | baseline |
| `lite` | No filler / no hedging, full sentences kept | ~15% |
| `full` *(server default)* | Drop articles, fragments OK, abbreviations | ~40% |
| `ultra` | Telegraph: arrows for causality, one-word answers, max abbreviations | ~75% |

Pick a level with `mio --caveman ultra`, `mio serve --caveman full`, or `/caveman ultra` inside the agent. Defaults: `lite` in the native agent, `full` on the server.

**Why it matters:** DFlash makes each token faster. PolarQuant makes the cache cheaper. Caveman makes the model emit *fewer tokens to begin with*. Multiplicatively, that's the difference between "fast" and "feels instant."

Example response to *"Explain how this hash table handles collisions."*

| Mode | Output |
|------|--------|
| off | "This implementation uses **separate chaining** to handle collisions. When two keys hash to the same bucket, the second key is appended to a linked list at that bucket. Lookup walks the list…" |
| full | "Uses **separate chaining**. Two keys → same bucket → append to bucket's linked list. Lookup walks list. Avg O(1+α)." |
| ultra | "Sep. chaining. Same bucket → append linked list. Lookup walks. O(1+α) avg." |

---

## Benchmarks

### Time to task — vanilla MLX vs full Mio stack

**Setup:** Qwen 3.5 35B-A3B MoE 4-bit, M4 Max 48 GB. Prompt: *"Write a Python function that fetches a URL with retries and exponential backoff."*

Numbers below are derived from per-component measurements (DFlash acceptance from `tests/test_engine.py`, PolarQuant overhead from the table further down, Caveman token-reduction from observed output deltas across 50 prompts). The seconds figure is *output tokens / generation tok/s* — prefill is small relative to generation for prompts under ~2K tokens.

| Stack | gen tok/s | output tokens | wall-clock | speedup vs vanilla |
|-------|-----------|---------------|------------|-------------------|
| `mlx-lm generate` (vanilla 4-bit, no draft, no caveman) | ~50 | ~480 | **9.6 s** | 1.0× |
| + DFlash | ~190 | ~480 | 2.5 s | 3.8× |
| + PolarQuant 4-bit cache | ~204 | ~480 | 2.4 s | 4.0× |
| + Caveman lite *(native agent default)* | ~204 | ~410 | 2.0 s | 4.8× |
| + Caveman full *(server default)* | ~204 | ~290 | **1.4 s** | **6.9×** |
| + Caveman ultra | ~204 | ~120 | **0.6 s** | **16×** |

The wins compound: speculative decoding raises tok/s, KV-cache compression keeps memory cheap so context stays long, and Caveman drops the token count the model has to emit in the first place.

For multi-turn coding sessions the gap widens further — every saved output token is also a saved input token on the next turn, and the **relaxed prefix cache** with KV truncation reuses the longest common prefix across turns instead of reprocessing the whole conversation.

### PolarQuant 4-bit overhead study

PolarQuant 4-bit KV-cache is **enabled by default**. It's actually faster than uncompressed because `mx.quantized_matmul` reads 4× less memory:

| Config | gen tok/s | acceptance | peak mem |
|--------|-----------|------------|----------|
| DFlash only | 46.4 | 0.70 | 11.0 GB |
| DFlash + PQ4 | **58.7** | **0.77** | 11.0 GB |

For sub-4-bit compression (3-bit, 2-bit), PolarQuant activates Hadamard rotation for quality preservation. Configure via `pq_bits` in `TierConfig` or `mio configure`.

---

## A tour of Mio UI

`mio serve --webui` opens the chat at http://localhost:9090/ui. Grouped by what you'd actually want to do:

**Talk to a model that draws back.** Ask for a diagram and you get a Mermaid / Graphviz / d3-force / mindmap rendered in a side panel — not a code block. Ask for a chart and the model picks the right Chart.js variant and ships it as an artifact. Ask for a small interactive thing — a Wordle helper, a pomodoro timer, a piano, a Tone.js synth, a flashcard deck, a 3D model viewer with STL — and you get a working app embedded in the chat. **Over 100 artifact types**: HTML, SVG, React + Tailwind + recharts (esm.sh + Babel), Pyodide REPL, SQL runner, Excalidraw whiteboard, KaTeX, reveal.js slides, three.js / p5.js sketches, model-viewer + STL, animated weather widgets.

**Do real research.** `web_search` returns a grid of clickable result boxes; the model can chain into `fetch_url`, `wiki_summary`, `hn_top`, `reddit_top`, `url_preview`, `search_images`, `search_youtube`. **Up to 5 rounds of generation ↔ tool execution per turn**, so a single prompt can search → fetch → analyze → render an artifact.

**Find any anime / manga / movie / TV / game.** `find_anime` (Jikan), `find_manga`, `find_movie_tv` (TVmaze), `find_game` (RAWG). Results auto-render as media cards with covers, ratings, episode counts.

**Generate documents that don't all look the same.** `generate_pdf_report` ships **64 layout presets × 39 color palettes**, auto-picked by keyword. Specialized templates for letter, certificate, flyer, menu, brochure, newsletter, business card, resume, invoice. Plus `generate_docx`, `generate_xlsx`, `generate_pptx`, and Obsidian-friendly markdown.

**Run code in chat.** `execute_python` (sandboxed Pyodide), `http_request`, `explain_regex`, `symbolic_math` (sympy), JSON ↔ YAML, JWT decode, hashing, encoding, UUID, password generation, timezone / unit conversion, text stats.

**Local RAG, no embeddings.** Point `index_folder` at any directory and `search_local_folder` will full-text-search it via SQLite FTS5. Index lives at `~/.mio/rag.sqlite`. No GPU, no API key, just fast keyword + ranking search across your notes / code / docs.

**Productivity that sticks.** Todos, habits, journal, bookmarks — all SQLite-backed under `~/.mio/`. Survive restarts. `analyze_csv` and `analyze_json` give you stats + chart artifacts in one shot.

**Talk to it.** Voice in (mic), voice out (TTS), and `/convo` for a hands-free voice loop. Image paste / drag-drop works against any of the multimodal-capable Qwen weights.

**27 personas.** `/as teacher`, `/as skeptic`, `/as chef`, `/as pirate`, `/as haiku`, `/as stoic`, `/as child`, `/as copy-editor`, `/as scientist`… each swaps the system prompt. `/as-list` shows the full roster.

**Workspace.** `/ui/dashboard` for scheduled prompts (cron-style), webhook triggers (`POST /ui/api/webhook/<slug>`), and indexed RAG folders. `/ui/stats` shows per-day message count, top skills, artifact types. `/ui/playground` is a live try-it form for every registered skill. `/ui/compare` runs two tiers side-by-side. `/ui/attachments` is the file library.

**Keyboard-first.** `⌘K` command palette, `⌘N` new chat, `⌘F` in-chat find, `⌘⇧V` paste-as-hidden-context, `⌘/` shortcut cheatsheet. Branching with ◀ ▶ to cycle regenerated replies. Pinned messages, starred chats, reactions, 12 themes, 3 density modes.

**Just yours.** Everything lives under `~/.mio/` — sessions, projects, prompts, memory, bookmarks, todos, habits, journal, schedules, webhooks, RAG index, image cache, web cache. Wipeable from Settings → Cache. No telemetry, no cloud, no key.

---

## Architecture

```
mio/
├── mio/
│   ├── engine.py          # MioEngine: DFlash + PolarQuant + prefix cache
│   ├── agent.py           # Coding agent with slash commands + caveman modes
│   ├── server.py          # FastAPI OpenAI-compatible API
│   ├── router.py          # TandemRouter (route by complexity across tiers)
│   ├── model_manager.py   # Thread-safe model lifecycle + VRAM management
│   ├── config.py          # TierConfig / MioConfig dataclasses
│   ├── tool_calls.py      # Qwen-native <tool_call> ↔ OpenAI JSON bridge
│   │
│   ├── dflash/            # Vendored DFlash MLX v2 runtime (speculative decoding)
│   ├── polarquant/        # PolarQuant KV-cache compression (Hadamard + mx.quantize)
│   ├── turboquant/        # Legacy TurboQuant (rotation + Lloyd-Max + QJL)
│   ├── paroquant/         # PARO weight loading + Metal rotation kernel
│   │
│   ├── models/registry.py # Model registry + DEFAULT_TIERS
│   └── webui/             # Mio UI backend + skills + modular assets
│       ├── router.py      # FastAPI router: /ui, /ui/api/*, /ui/ws/chat
│       ├── skills.py      # Skill registry (99 tools)
│       ├── skills_docs.py # PDF/DOCX/XLSX/PPTX + 64 presets × 39 colors
│       ├── skills_misc.py # QR, iCal, CSV, SQLite, resume, invoice, Jikan, TVmaze…
│       ├── skills_python.py    # Dev utilities (hash/encode/UUID/regex/…)
│       ├── skills_life.py      # Bookmarks, HN/Reddit, quotes, currency…
│       ├── skills_productivity.py  # Todo/habit/journal, JSON/CSV analyzers
│       ├── skills_rag.py       # Local folder full-text search (SQLite FTS5)
│       ├── skills_fun.py       # Dice, coin, names, wordle, wikipedia
│       ├── scheduler.py   # Async cron loop for scheduled prompts
│       ├── webhooks.py    # Webhook CRUD + fire endpoint
│       ├── mio_ui.html    # Shell SPA
│       └── assets/        # Modular front-end: main.css + 27 JS modules + data
│           ├── main.css
│           ├── data/      # SUGGESTION_BANK, PERSONAS, SLASH_TEMPLATES
│           └── *.js       # pinned, tips, find, emoji, pomodoro, branching,
│                          # reactions, extras, starred, clocks, prompt_library…
├── models/                # Target model weights (local)
└── spd/                   # DFlash draft model weights (local)
```

---

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python ≥ 3.10
- 48 GB+ unified memory recommended (16 GB minimum for `small` tier only)

---

## Acknowledgements

Mio would not exist without the following projects and their authors. Huge thanks:

- **[MLX](https://github.com/ml-explore/mlx)** and **[mlx-lm](https://github.com/ml-explore/mlx-examples)** by Apple's ML Explore team — the foundation that makes fast local inference on Apple Silicon possible. Their `QuantizedKVCache` and `mx.quantized_matmul` Metal kernels are what let PolarQuant 4-bit achieve zero speed overhead.
- **[DFlash](https://github.com/z-lab/dflash)** and **[DFlash-MLX](https://github.com/bstnxbt/dflash-mlx)** — speculative decoding runtime that delivers up to 4.1× generation speedup. The DFlash runtime is the heart of Mio's inference engine.
- **[Caveman](https://github.com/juliusbrussee/caveman)** by [Julius Brussee](https://github.com/juliusbrussee) — the system-prompt token-compression idea Mio's `--caveman` modes are built on. Lite / full / ultra tiers shave 15–75% off output token counts without touching code blocks, which is what turns "fast" into "feels instant" end-to-end.
- **[TurboMLX](https://github.com/Smilefounder/TurboMLX)** by [Smilefounder](https://github.com/Smilefounder) (forked from [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant)) — their PolarQuant implementation and detailed benchmark report were invaluable. Their finding that at 4-bit, MLX's native affine quantization matches rotation-based methods directly shaped our approach.
- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (Google, ICLR 2026) — introduced PolarQuant rotation + Lloyd-Max quantization for KV-cache compression. Theoretical foundation for data-oblivious quantization.
- **[PolarQuant](https://arxiv.org/abs/2502.02617)** by Insu Han et al. (Google Research / Yale) — random preconditioning + polar decomposition for normalization-free KV-cache quantization.
- **[Qwen](https://github.com/QwenLM/Qwen3)** by Alibaba — the Qwen 3.5 model family Mio runs on by default. The A3B MoE architecture is what makes high-quality local inference practical on consumer hardware.
- **[Unsloth](https://github.com/unslothai/unsloth)** and **[Brooooooklyn](https://huggingface.co/Brooooooklyn)** — MLX-optimized quantizations (UD-Q4_K_XL) with imatrix calibration that fix tool-call degradation from standard RTN INT4. These re-quants make reliable agentic workflows possible locally.
- **[PARO](https://arxiv.org/abs/2407.00570)** by z-lab — pairwise rotation quantization that achieves better accuracy than AWQ at INT4.
- Everyone in the **[mlx-community](https://huggingface.co/mlx-community)** on Hugging Face who quantizes and shares models. This community makes local AI accessible.

---

## License

MIT
