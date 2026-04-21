# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mio is a fast local MLX-based inference server and interactive coding agent for Apple Silicon Macs. It combines five acceleration technologies: PARO (INT4 weight quantization), DFlash speculative decoding (4.1x speedup), PolarQuant KV-cache compression (Hadamard rotation + 4-bit, zero speed overhead), DDTree speculative decoding (opt-in tree-attention verify — +10-15% over DFlash on code, hybrid_gdn models only), and the Caveman Ultra system prompt (75% fewer output tokens).

Default configuration (`large-moe`) runs **Qwen3.6-35B-A3B MoE** at ~204 tok/s on M4 Max with 128K context.

## Commands

```bash
# Install (editable/dev mode)
pip install -e .

# Run native interactive agent
mio                            # default large-moe tier
mio --tier medium              # 9B model
mio --tandem                   # auto-route by complexity across all tiers

# Start OpenAI-compatible API server (http://localhost:9090/v1)
mio serve
mio serve --tandem --validate

# Simple chat (no tools)
mio chat

# Model management
mio download                   # download all default tiers
mio pull qwen3.5-35b-a3b      # download specific model + DFlash draft
mio pull                       # list available models

# Batch inference
mio batch --input prompts.jsonl --output results.jsonl

# Config & diagnostics
mio configure                  # interactive wizard
mio status                     # show config + server status
mio bench                      # benchmark all tiers

# Tests
pytest tests/
pytest tests/test_engine.py -v
pytest tests/test_server.py -v
```

## Architecture

### Core Stack
```
User Input → CLI (main.py) → Agent / Server / Chat
                                      ↓
                             ModelManager (model_manager.py)
                                      ↓
                           TandemRouter (optional, router.py)
                                      ↓
                             MioEngine (engine.py)
                             ├── Target model (PARO-quantized)
                             ├── DFlash draft model
                             └── TurboQuant KV-cache
                                      ↓
                            MLX Metal (Apple Silicon)
```

### Key Modules

| File | Role |
|------|------|
| `mio/main.py` | CLI entry point; argparse subcommand dispatcher |
| `mio/engine.py` | Core inference: `MioEngine` wraps DFlash + TurboQuant |
| `mio/agent.py` | Native interactive coding agent (no external deps); slash commands |
| `mio/server.py` | FastAPI OpenAI-compatible API server |
| `mio/model_manager.py` | Thread-safe model lifecycle and VRAM management |
| `mio/router.py` | `TandemRouter` — routes requests to optimal tier by complexity |
| `mio/models/registry.py` | `DEFAULT_TIERS` — 4-tier model definitions and tier configs |
| `mio/config.py` | `MioConfig` dataclass; defaults: large-moe, 128K ctx |

### Vendored Sub-packages

- **`mio/dflash/`** — Speculative decoding runtime (DFlash MLX v2). `runtime.py` (64KB) is the main generate loop; 89% acceptance rate on Qwen3.5/3.6.
- **`mio/polarquant/`** — KV-cache quantization (preferred). Hadamard rotation + `mx.quantize`/`mx.quantized_matmul`. Zero speed overhead with DFlash — no forced `mx.eval()`, no normalization, no QJL.
- **`mio/turboquant/`** — Legacy KV-cache quantization. `patch.py` hot-patches MLX attention layers; `attention_v2.py`/`attention_v3.py` are successive optimizations. Slower than PolarQuant due to forced `mx.eval()`.
- **`mio/paroquant/`** — PARO quantization loader. `modules.py` defines `RotateQuantizedLinear`; `kernels/rotation.py` is the Metal rotation kernel.
- **`mio/ddtree/`** — Diffusion Draft Tree speculative decoding (ported from humanrouter/ddtree-mlx). Extends DFlash by verifying a tree of candidate paths in one target forward with tree-attention masks + parent-indexed GatedDelta Metal kernels. Opt-in via `TierConfig.ddtree_budget > 0` or `MIO_DDTREE_BUDGET` env var. Hybrid_gdn models only (Qwen3.5-27B, Qwen3.5/3.6-35B-A3B). Incompatible with PolarQuant/TurboQuant/BMP/prefix-cache — when enabled, the engine swaps PQ/TQ for mlx_lm `QuantizedKVCache` (8-bit) and forces `DDTREE_EXACT_COMMIT=1` (sequential commit, quantized-safe). Gain: +10-15% over DFlash on code/structured output; ~0% on creative prose (draft acceptance too low for the tree to add value).

### Generation Flow

1. Apply Qwen chat template (`<|im_start|>role\ncontent<|im_end|>`)
2. Load PARO model via `paroquant/load.py` (custom `RotateQuantizedLinear`)
3. DFlash speculative decoding: draft generates K tokens → target verifies → accept/reject
4. TurboQuant compresses KV-cache during generation (4/3/2-bit)
5. Caveman mode optionally prepends abbreviation-heavy system prompt
6. Tokens streamed via SSE (OpenAI format) or returned as batch

### Model Tiers

| Tier | Model | VRAM | Context | Speed |
|------|-------|------|---------|-------|
| `large-moe` | Qwen3.6-35B-A3B | ~22 GB | 128K | ~204 tok/s |
| `large` | Qwen3.5-27B | ~14 GB | 32K | ~56 tok/s |
| `medium` | Qwen3.5-9B | ~5 GB | 16K | ~108 tok/s |
| `small` | Qwen3.5-4B | ~2.5 GB | 8K | ~187 tok/s |

Models are stored under `models/` (target weights) and `spd/` (DFlash draft weights).

### API Endpoints (default port 9090)

`GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming SSE), `POST /v1/models/load`, `POST /v1/models/unload`, `POST /v1/batch`, `GET /metrics`, `GET /dashboard`, `WS /ws/metrics`

### Mio UI — Artifacts & Skills

Enable with `mio serve --webui`, open http://localhost:9090/ui. Install full-feature deps with `pip install -e .[webui]`.

**Artifacts**: the model can wrap rendered content in `<antArtifact identifier="..." type="..." title="...">...</antArtifact>` and the UI opens it in a side panel. Supported types:
- `text/html` — sandboxed iframe
- `image/svg+xml` — inline SVG
- `text/markdown` — rendered markdown
- `application/vnd.ant.mermaid` — Mermaid diagrams (mermaid.js CDN)
- `application/vnd.ant.react` — React + Tailwind + Babel standalone (CDN, sandboxed iframe)
- `application/vnd.ant.code` — syntax-highlighted code
- `application/vnd.pimio.weather` — animated weather widget with Meteocons icons

**Skills** — 97 tools registered in `mio/webui/skills.py`, implemented across:
- `skills.py` — core web (web_search, fetch_url), get_weather, execute_python
- `skills_docs.py` — PDF / DOCX / XLSX / PPTX with 64-preset + 39-color system + specialized templates (letter, certificate, flyer, menu, brochure, newsletter, business card) + generate_pdf_report / generate_chart
- `skills_misc.py` — QR, iCal, CSV, SQLite, resume, invoice, extract_pdf_text, translate, find_anime/manga/movie_tv/game, search_images, search_youtube, generate_markdown
- `skills_python.py` — 21 dev utilities: hash, encode, uuid, password, fake data, JWT, YAML↔JSON, timezone, unit, text_stats, RSS, ZIP, PDF merge/split, symbolic math, markdown↔html, etc.
- `skills_life.py` — bookmarks, color_palette, describe_image, review_code, meeting_notes, explain_regex, convert_currency, url_preview, hn_top, reddit_top, quote library
- `skills_productivity.py` — todo list, habit tracker, journal, analyze_json, analyze_csv
- `skills_rag.py` — local folder RAG (index_folder, search_local_folder)
- `skills_fun.py` — roll_dice, flip_coin, pick_random, generate_names, wordle_helper, wiki_summary

**Modular front-end** — `mio/webui/assets/` serves 32 ES module files at `/ui/assets/<name>`. Each is a self-contained feature (pinned, density, tips, clipboard-context, presentation, artifact-export, chat-import, chat-export, prompt-library, onboarding, templates, followups, branching, reactions, extras/surprise, chat-system-prompt, compress, keys, pomodoro, emoji, starred, find, micro, theme, clocks, image-paste, etc.). Shared state is exposed on `window.Mio.<feature>`.

**Auxiliary pages**:
- `/ui/playground` — live try-it form for every registered skill
- `/ui/dashboard` — scheduled prompts, webhooks, indexed folders
- `/ui/stats` — per-day messages, top skills, artifact types, personas
- `/ui/attachments` — every generated / uploaded file in ~/Downloads
- `/ui/compare` — side-by-side two-tier model compare
- `/ui/share/<id>` — read-only artifact share

**Persistent state** (all under `~/.mio/`, gitignored):
- `sessions/` — chat history (one JSON per session, includes artifacts + chat_system_prompt + pinned flags)
- `projects.json` / `prompts.json` / `memory.json`
- `image-cache/` / `web-cache/` / `files-cache/` (wipeable from Settings → Cache)
- `todos.sqlite` / `habits.sqlite` / `bookmarks.sqlite` / `rag.sqlite`
- `schedules.json` / `schedules-log.jsonl`
- `webhooks.json` / `webhooks-log.jsonl`
- `chat-templates.json`
- `journal/<YYYY-MM-DD>.md`

**27 personas** (`/as <name>`), **~100 slash commands** covering every artifact type + every skill + every dashboard. `⌘/` opens the keyboard-shortcut cheatsheet; `⌘K` opens the command palette; `⌘⇧V` pastes clipboard as hidden context; `⌘F` opens in-chat find.

**Tool-use loop**: `_handle_chat` in `mio/webui/router.py` runs up to 5 rounds of generation ↔ tool execution. Each round the model may emit multiple tool calls; results are fed back with URLs preserved so the model can chain `web_search` → `fetch_url` → `generate_chart` → artifact.

**System prompt injection**: every turn gets today's ISO date + browsing protocol + artifacts protocol + document-type + preset-handling + media-skill-gating + personas prepended. Per-chat system prompt (`/chat-prompt`) overrides the global setting when set.

Model name format: `mio-{tier}` (e.g. `mio-large-moe`, `mio-auto` for tandem).
