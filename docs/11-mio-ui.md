# Mio UI — web interface

Mio UI is served by `mio serve --webui` at
`http://127.0.0.1:9090/ui`. It combines chat, artifacts, local tools, projects,
knowledge views, workflows, schedules, and Mio's external instruction catalog.
Feature presence in this document is not a browser-compatibility guarantee;
the release gate still requires live desktop/mobile browser QA.

## Install

```bash
python3 -m pip install -e .
mio serve --tier large --webui
```

The current package includes the Web UI/document stack in the main install;
there is no separate `[webui]` extra. Optional system tools such as Tesseract
still require their platform installation.

The server binds to loopback by default and refuses a non-loopback bind unless
`--unsafe-remote-bind` is supplied. That opt-in does not add authentication.
Do not expose Mio UI to untrusted network users: it includes local files,
executable skills, flows, schedules, webhooks, and generated-code artifacts.

## Core UX features

### Chat

- Full GFM markdown rendering with vendored Marked 12.0.2 and Prism 1.29.0 assets; Python, TS, JSX, Rust, Go, SQL, YAML, JSON, CSS, and Bash highlighting works without a CDN at application boot. Optional sandboxed artifact renderers may still load their own libraries on demand
- Multi-round tool-use loop (up to 5 rounds per turn) — model can chain `web_search` → `fetch_url` → `generate_chart` → emit artifact
- Live streaming with per-message metrics (tok/s, acceptance, prefill)
- Session persistence (`~/.mio/sessions/*.json`) with auto-titles and
  schema-v2 artifact revision state
- Import/export chat as JSON, including artifact chains, project association,
  and the per-chat system prompt; legacy `artifacts` lists remain importable
- Hover any message for **Copy · Regenerate · Speak · Pin** actions
- Global search across every saved chat (sidebar debounced, snippet-highlighted)

### Input
- Drag-drop any file onto the window → extracted + attached (PDF via pdfplumber, plain text for .md/.txt/.code, size note for images)
- Attachment chips preview what's attached; click × to remove
- **Slash commands**: type `/` to open a filterable popup with all commands and template prompts (↑↓ nav, Tab/Enter picks)
- **Voice input** (🎤 button) via Web Speech API, interim transcription into the input
- **Voice conversation mode** (`/convo`): hands-free, 1.4s-silence auto-send, TTS reads replies, reopens mic on end
- **Clipboard watcher**: on window focus, offers to attach any image in the clipboard

### Commands & keyboard

| Shortcut | Action |
|---|---|
| `⌘K` | Command palette (search commands + artifacts) |
| `⌘N` | New chat |
| `⌘,` | Settings |
| `?` | Keyboard + feature cheatsheet |
| `/` | Slash command popup |
| `Esc` | Close overlays / artifact panel / stop voice / focus mode |
| `Enter` | Send (Shift+Enter newline) |

Slash commands: `/weather /chart /pdf /docx /xlsx /pptx /qr /ical /resume /invoice /mindmap /timeline /math /map /diagram /3d /search /new /settings /theme /export /export-json /import /clear /fullscreen /voice /convo /screenshot /ambient /focus /gallery /workspace /save /stop /dashboard /compare /help`.

### Power features

- **Artifact panel**: desktop side panel with Preview/Source tabs,
  drag-left-edge resize, double-click expansion, and fullscreen control; below
  768 px it becomes a fixed sheet beside the 48 px navigation rail instead
  of adding an overflowing fourth grid column
- **Artifact versioning**: same-identifier artifacts stack into a chain with
  prev/next arrows; session schema v2 retains the complete bounded chain and
  selected revision
- **Edit-in-place**: Source is editable; Save appends a revision whose
  provenance refers to the prior content identifier
- **Share**: mints a short-URL at `/ui/share/<id>` for read-only view of an artifact; copies to clipboard
- **Copy source / Download file**: each artifact's source or template-wrapped standalone HTML
- **Artifact gallery** (`/gallery`): thumbnail grid of every artifact in the current chat, filterable
- **Focus/zen mode** (`/focus`): hides chrome, centers content at 720px, 15px type, 1.7 leading
- **Fullscreen artifact**: hides chat and sidebar, panel takes the viewport
- **Ambient mode** (`/ambient`): streams last response onto a canvas → Picture-in-Picture → floats above all Mac apps
- **Screen capture** (`/screenshot`): `getDisplayMedia` → PNG attached to next message
- **Live workspace** (`/workspace`): File System Access API picks a folder with R/W scope; `/save` drops current artifact there
- **Wake Lock**: acquired at send, released at done — screen stays awake during long generations
- **Notifications**: browser ping when a reply finishes if the tab is hidden
- **Side-by-side compare** (`/compare`): operates only when two distinct loaded
  models are available; zero/one-model states are explicit and cannot submit a
  fake or self-comparison
- **MCP health in Settings**: shows redacted readiness, latency, and tool count
  for bounded probes of local unauthenticated providers, with a manual Retry
  action; every stdio probe is sandboxed, while remote, authenticated and
  unisolatable HTTP/SSE providers are never contacted by this view

### Keyboard and accessibility structure

Preview, Source, and Diff are native buttons in a `tablist`, and their
`aria-selected` value follows the visible artifact view. Help, Gallery,
Command Palette, and Settings expose dialog roles and accessible names; the
artifact inspector is a named region. Artifact and gallery cards are buttons,
the send control has an explicit label, the Caveman, temperature, maximum-
token, and response-style settings have associated labels, and keyboard-
focusable controls receive a global `:focus-visible` outline.

These are structural improvements, not a WCAG conformance claim. Complete
focus trapping/restoration, screen-reader passes, zoom testing, and live
desktop/mobile browser QA remain release work.

### Flow Mode

Flow Mode is a persistent visual DAG editor, not a UI-only placeholder. Its
inspector covers every shipped node type, saves graphs under `~/.mio/flows`,
runs them server-side, and streams per-node status and final results over SSE.
The executor supports model/skill calls, bounded HTTP fetches, branching,
iteration, user input, transforms, local memory, RAG, and artifact emission.
It executes nodes serially in topological order; there is no intra-flow
parallel scheduler or retry/backoff policy yet.

An artifact node emits a bounded `artifact_emitted` SSE event. The browser
passes the payload through the stable `window.Mio.artifacts.ingestAndOpen`
interface, registers it in the same version/gallery store as chat artifacts,
and opens the artifact panel. It does not write an undocumented JSONL stub or
pretend the artifact was sent to chat. A run can emit at most 16 artifacts,
each with at most 512 KiB of UTF-8 content and at most 2 MiB in aggregate.

A saved flow can be published persistently as a Mio skill. The model sees two
stable schemas, `list_flow_skills` and `run_flow_skill`, rather than one schema
per graph. Current bounds are 200 graph nodes and 200 execution hops, a 2 MiB
flow document, 64 KiB of run arguments, a 256 KiB tool result, and a 120-second
published-flow timeout. Recursive `run_flow_skill` dispatch is rejected.

## Supported artifact renderer matrix

The model wraps rendered content in
`<antArtifact identifier="…" type="…" title="…">…</antArtifact>`. The
matrix below lists MIME keys that are routed to maintained renderers in the
current UI; it is not derived from a hard-coded marketing count. Some known
`vnd.ant.*`, `vnd.pimio.*`, and legacy spellings are aliases, but support is
defined by the canonical renderer key rather than by accepting every possible
name in either namespace. For compactness, older matrix rows written as
`vnd.*` omit the leading `application/` from their on-wire MIME value; the
three Artifact Lab rows show their complete values.

Alias normalization is idempotent. In particular,
`application/vnd.pimio.react`, `application/vnd.pimio.code`, and
`application/vnd.pimio.mermaid` normalize to the corresponding
`application/vnd.ant.*` renderer keys, while an already canonical Ant key
remains unchanged. Known aliases for Three.js, p5, Chart.js, Leaflet, math,
Graphviz, mind maps, slides, and timelines converge on their
`application/vnd.pimio.*` renderer keys. Legacy ERD, Gantt, and state chart
types converge on Mermaid; `presenter` converges on Reveal.js. Running
normalization again leaves those results unchanged.

Identifiers must begin with an ASCII letter or digit and may then contain up
to 127 letters, digits, dots, underscores, or hyphens. Invalid or prototype-
sensitive identifiers are replaced with a deterministic `art-…` identifier.
Consequently a dotted identifier such as `mlx.run-01` is valid in the message
placeholder, version store, gallery, and session reload path.

### Native MLX research

| Type | Body | Renderer |
|---|---|---|
| `application/vnd.pimio.benchmark+json` | Matched-run JSON | native DOM comparison |
| `application/vnd.pimio.model-card+json` | Checkpoint metadata JSON | native DOM compatibility card |
| `application/vnd.pimio.inference-trace+json` | Timed span JSON | native DOM timeline |

### Visual / interactive
| Type | Body | Library |
|---|---|---|
| `text/html` | Full HTML page | sandboxed iframe |
| `image/svg+xml` | SVG source | inline |
| `text/markdown` | Markdown | marked.js |
| `vnd.ant.mermaid` | Mermaid DSL | mermaid.js |
| `vnd.ant.react` | React JSX | React 18 + Tailwind + lucide-react + recharts + framer-motion |
| `vnd.ant.code` | Source | Prism.js syntax highlight |
| `vnd.pimio.threejs` | JS (scene/camera/lights pre-wired) | three.js + OrbitControls |
| `vnd.pimio.p5` | p5.js setup/draw | p5.js |
| `vnd.pimio.chartjs` | Chart.js config JSON | Chart.js |
| `vnd.pimio.leaflet` | `{center, zoom, markers, polylines, geojson}` | Leaflet |
| `vnd.pimio.math` | LaTeX with `$..$` `$$..$$` | KaTeX |
| `vnd.pimio.graphviz` | DOT source | viz.js |
| `vnd.pimio.mindmap` | Markdown outline | markmap.js |
| `vnd.pimio.revealjs` | Markdown slides | reveal.js |
| `vnd.pimio.timeline` | vis-timeline JSON | vis-timeline |
| `vnd.pimio.shader` | GLSL fragment | WebGL2 |

### Engineering / CAD
| Type | Body | Library |
|---|---|---|
| `vnd.pimio.wavedrom` | WaveDrom JSON | WaveDrom |
| `vnd.pimio.physics` | Matter.js JS | Matter.js |
| `vnd.pimio.graph` | `{nodes, links}` JSON | d3-force |
| `vnd.pimio.plantuml` | PlantUML source | kroki.io |
| `vnd.pimio.jscad` | `function main(){…}` | @jscad/modeling + three.js |
| `vnd.pimio.modelviewer` | GLB/GLTF URL or `{src, poster?}` | @google/model-viewer (+ AR) |

### Productivity
| Type | Body | Library |
|---|---|---|
| `vnd.pimio.pyrepl` | Python code | Pyodide + numpy/pandas/matplotlib |
| `vnd.pimio.tone` | Tone.js JS | Tone.js |
| `vnd.pimio.jsonviewer` | JSON string | native |
| `vnd.pimio.table` | `[…] / {headers, rows}` | native sortable |
| `vnd.pimio.diff` | `{oldStr, newStr}` | diff2html |
| `vnd.pimio.regex` | `{pattern, flags, test}` | native |
| `vnd.pimio.piano` | ignored | Tone.js synth |
| `vnd.pimio.flashcards` | `{cards:[{front, back}]}` | native |
| `vnd.pimio.kanban` | `{columns:[{name, cards}]}` | Sortable.js |
| `vnd.pimio.palette` | `{colors:[{name, hex}]}` | native |
| `vnd.pimio.whiteboard` | ignored | HTML canvas drawing |
| `vnd.pimio.pomodoro` | `{focus, break, longBreak, rounds}` | native + Notifications |
| `vnd.pimio.gradient` | `{gradients:[{name, css}]}` | native |
| `vnd.pimio.countdown` | `{title, target: ISO-datetime}` | native |
| `vnd.pimio.qrview` | URL/text | qrcode-generator |
| `vnd.pimio.excalidraw` | ignored | Excalidraw iframe |
| `vnd.pimio.audio` | URL or `{url, title}` | wavesurfer.js |
| `vnd.pimio.youtube` | URL or 11-char ID | YouTube embed |
| `vnd.pimio.terminal` | `[{prompt, cmd, output}]` | typed-out replay |
| `vnd.pimio.weather` | JSON from `get_weather` | Meteocons + gradient |

### File-backed
| Type | Body | Rendered as |
|---|---|---|
| `application/pdf` | filename | iframe embed |
| `vnd.pimio.image` | filename | inline preview |
| `vnd.pimio.file` | filename | download card |

## Native Artifact Lab payloads

The three MLX research renderers are implemented by the packaged
`/ui/assets/artifact_lab.js` asset. They do not fetch a CDN library, navigate
to a remote URL, or execute payload code. The module parses a JSON object and
builds ordinary DOM nodes; payload values enter those nodes through
`textContent`, not `innerHTML`. This is a narrower rendering surface than the
executable iframe-based artifact types, but it is not a signature or a claim
that the supplied measurements are true.

All three types reject a UTF-8 payload larger than 512 KiB and show an error
card for malformed JSON or a top-level value other than an object. The Source
tab remains available for inspection and editing. A successful download is
pretty-printed JSON with a trailing newline; malformed source is downloaded
unchanged rather than silently rewritten.

### MLX benchmark comparison

`application/vnd.pimio.benchmark+json` accepts:

- optional top-level `title`, `subtitle` or `workload`, and `device` fields;
- a required non-empty `runs` array, bounded to 48 objects;
- `label` (or `name`) for each run;
- `prefill_tps` (or `prompt_tps`), `decode_tps` (or `generation_tps`),
  `ttft_ms`, `memory_gb` (or `peak_memory_gb`), and `acceptance` (or
  `acceptance_ratio`) as finite numeric values;
- at least one prefill, decode, or TTFT value across the compared runs.

The renderer highlights the highest decode and prefill throughput and the
lowest TTFT among values present in the payload. It does not decide whether
runs were matched, validate parity, or produce a benchmark itself. This
minimal example reuses the verified snapshot reported in the root README:

```json
{
  "title": "Qwen 3.6 short-workload snapshot",
  "runs": [
    {
      "label": "target baseline",
      "prefill_tps": 234.77,
      "decode_tps": 19.31
    },
    {
      "label": "DFlash",
      "prefill_tps": 232.92,
      "decode_tps": 33.64
    }
  ]
}
```

### Model compatibility card

`application/vnd.pimio.model-card+json` requires `name` or `model`. It can
also present `description`, `family`, `parameters`, `quantization`, `format`,
`context_window`, `size_gb`, `memory_gb`, `revision`, `source`, and `sha256`.
The optional `features` and `drafters` arrays are combined into at most 24
display tags. This is presentation metadata: Mio does not infer compatibility
or verify the source/hash from the card.

```json
{
  "name": "organization/checkpoint",
  "format": "MLX",
  "quantization": "example-format",
  "drafters": ["compatible-drafter-id"],
  "source": "local model registry"
}
```

Names in this example are placeholders that demonstrate the payload shape;
they are not a compatibility result.

### Inference trace

`application/vnd.pimio.inference-trace+json` requires a non-empty `spans`
array bounded to 256 entries. Each span needs a finite, non-negative
`duration_ms`; `start_ms` is non-negative and defaults to zero when omitted.
Optional `name`, `category`, and `detail` fields label a span. The renderer
sorts spans by start time and uses the greatest of `total_ms` and the observed
span end times as its timeline extent.

```json
{
  "title": "Schema example — not a measurement",
  "spans": [
    {
      "name": "prefill",
      "start_ms": 0,
      "duration_ms": 12.5,
      "category": "prefill"
    }
  ]
}
```

The model-facing system prompt instructs the model to use benchmark and trace
artifacts only for supplied measurements and never to fabricate values or
spans.

## Artifact persistence and JSON export

Mio UI stores artifacts as chains keyed by identifier. `allArtifacts` points
at the selected revision of each chain, while the schema-v2
`artifact_state` retains the revisions themselves:

```json
{
  "schema_version": 2,
  "active_artifact_id": "mlx.run-01",
  "chains": [
    {
      "id": "mlx.run-01",
      "active_index": 0,
      "revisions": [
        {
          "id": "mlx.run-01",
          "type": "application/vnd.pimio.benchmark+json",
          "title": "Matched MLX runs",
          "language": "",
          "content": "{\"runs\":[{\"label\":\"baseline\",\"decode_tps\":19.31}]}",
          "created_at": "2026-07-16T00:00:00.000Z",
          "content_id": "fnv1a32:31dc2a55",
          "provenance": {"producer": "chat"}
        }
      ]
    }
  ]
}
```

The `content_id` is a 32-bit FNV-1a value over the JavaScript code units of the
canonical type, a NUL separator, and the content. It is deterministic metadata
for relating revisions, not a cryptographic digest or integrity check. A
Source-tab save appends a revision with `producer: "editor"` and the preceding
`content_id` in `parent`; it does not overwrite the earlier revision.

Auto-save sends both the schema-v2 state and a legacy projection of the
currently selected artifacts. JSON export adds a top-level
`schema_version: 2`, export timestamp, session ID, title, messages, project
association, and per-chat system prompt. On import, `messages` is required.
If `artifact_state.schema_version` is 2, Mio rebuilds at most 256 chains and
64 revisions per chain and clamps each `active_index` to the available range.
If that state is absent, it ingests the legacy top-level `artifacts` array.
The serialized `active_artifact_id` records which panel was active at save
time; current restoration rebuilds revision selection from each
`active_index` and does not automatically reopen that panel.

## Built-in tools and external instruction skills

The table below is a representative subset of executable built-in tools. It is
not the external skill catalog and is not an authoritative count; use
`GET /ui/api/skills` or the playground for the live registry.

| Skill | Library | Purpose |
|---|---|---|
| `web_search` | DuckDuckGo HTML | |
| `fetch_url` | urllib + Reddit case-resolver | |
| `generate_pdf` | **fpdf2** | Quick Unicode PDF |
| `generate_pdf_report` | **reportlab + matplotlib** | Styled PDF with charts + tables |
| `generate_chart` | **matplotlib** | bar/hbar/line/pie PNG |
| `generate_docx` | **python-docx** | Word w/ tables |
| `generate_xlsx` | **openpyxl** | Styled Excel |
| `generate_pptx` | **python-pptx** | PowerPoint deck |
| `generate_resume` | **reportlab** | Styled CV |
| `generate_invoice` | **reportlab** | PDF invoice with tax |
| `generate_qr_code` | **qrcode** | QR PNG |
| `generate_ical` | **icalendar** | .ics calendar |
| `generate_csv` | csv stdlib | CSV export |
| `generate_sqlite_db` | sqlite3 stdlib | Single-table DB |
| `extract_pdf_text` | **pdfplumber** | Read any PDF |
| `translate_text` | MyMemory (no key) | Short translations |
| `get_weather` | Open-Meteo (no key) | Current + hourly + 7-day |
| `execute_python` | subprocess | Run Python |

Separately, Mio's reviewed managed snapshot currently validates **916 external
instruction skills** under `~/.mio/skills`. This is the expected count for the
pinned sources, not a hard-coded live total; unmanaged Mio-local skills may be
discovered alongside them. Skills are searched through `list_mio_skills` and
loaded on demand through `read_mio_skill`; reading instructions never executes
bundled repository code. See [14 — External skills](14-external-skills.md).

Prompts, persistent memory, projects, schedules, and webhooks are mutated with
locked atomic JSON transactions. Schema or JSON corruption returns HTTP 409
and leaves the original file untouched; it is never converted silently into
an empty collection by a later write.

## Security boundaries

- session identifiers and storage paths are validated server-side;
- Host, Origin, same-origin mutation, and WebSocket session/CSRF checks run
  before browser handlers;
- HTTP request bodies are capped globally at 32 MiB;
- uploads are read in chunks and capped at 25 MiB;
- rendered chat/Markdown passes through Mio's sanitizer;
- the main shell loads its Marked/Prism runtime from versioned package assets,
  not a boot-time CDN;
- executable artifacts use sandboxed iframes without `allow-same-origin`;
- external URL fetches reject private/non-global destinations and revalidate
  DNS and redirects;
- a Content Security Policy and security headers are emitted, while legacy
  inline assets still require CSP compatibility allowances;
- modules share state through `window.Mio` rather than accidental globals.

The Web UI exposes only explicitly allow-listed public read tools to the model
by default. Local reads/writes, execution, private-network tools, flows, and
MCP orchestration are fail-closed: auto-use requires the exact tool name in
both `MIO_WEBUI_SKILL_GRANTS` and the request's `skill_grants`; direct
sensitive runs require the operator grant plus per-call confirmation. The
registry/playground may list a capability that the model is not authorized to
invoke.

These controls reduce risk but are not a complete browser or operating-system
sandbox. Generated artifacts and executable built-in skills remain untrusted
actions. Built-in bearer authentication, executable-artifact consent,
eliminating legacy inline CSP allowances, and cross-browser accessibility QA
remain release work.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /ui/` | Single-page HTML app |
| `WS /ui/ws/chat` | Streaming chat + tool loop |
| `GET /ui/api/config` | Session config |
| `POST /ui/api/config` | Update config (caveman, temp, sys prompt, max_tokens) |
| `GET/POST/DELETE /ui/api/sessions[/id]` | Chat persistence |
| `GET /ui/api/model-info` | Active tier, VRAM, context |
| `POST /v1/mcp/health` | CSRF-protected, redacted and bounded health probe for eligible local MCP providers |
| `GET /ui/compare` | Distinct-loaded-model side-by-side comparison |
| `GET /dashboard` / `WS /ws/metrics` | Live schema-v1 inference metrics and reconnecting dashboard |
| `POST /ui/api/tier` | Switch tier |
| `POST /ui/api/upload` | File attachment (multipart) |
| `GET /ui/api/search?q=…` | Global cross-chat search |
| `POST /ui/api/share` | Mint a share link |
| `GET /ui/share/{id}` | Standalone read-only artifact view |
| `GET /ui/files/{name}[?download=1]` | Serve generated file inline or as attachment |
| `GET/POST /ui/api/flows` | List/save bounded Flow Mode graphs |
| `GET/DELETE /ui/api/flows/{id}` | Read/delete one graph |
| `POST/DELETE /ui/api/flows/{id}/expose` | Publish/unpublish a flow as a Mio skill |
| `POST /ui/api/flows/{id}/run` | Start a server-side graph run |
| `GET /ui/api/flows/runs/{id}/events` | Stream per-node run events over SSE |
