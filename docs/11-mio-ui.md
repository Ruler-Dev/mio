# Mio UI — Web Interface

A Claude-style chat UI served by `mio serve --webui` at http://localhost:9090/ui. Aims to be the best Mac-native LLM chat UI by leaning hard on browser APIs most cloud chatbots ignore.

## Install

```bash
pip install -e .[webui]
mio serve --webui
```

The `[webui]` extra pulls in the full stack for document generation, weather, QR codes, calendars, and file uploads: `fpdf2 matplotlib python-docx openpyxl python-pptx reportlab pdfplumber pypdf markdown qrcode icalendar python-barcode python-multipart`.

## Core UX features

### Chat
- Full GFM markdown rendering (marked.js) with Prism.js syntax highlighting for Python, TS, JSX, Rust, Go, SQL, YAML, JSON, CSS, Bash
- Multi-round tool-use loop (up to 5 rounds per turn) — model can chain `web_search` → `fetch_url` → `generate_chart` → emit artifact
- Live streaming with per-message metrics (tok/s, acceptance, prefill)
- Session persistence (`~/.mio/sessions/*.json`) with auto-titles
- Import/export chat as JSON (round-trip saves)
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

Slash commands: `/weather /chart /pdf /docx /xlsx /pptx /qr /ical /resume /invoice /mindmap /timeline /math /map /diagram /3d /search /new /settings /theme /export /export-json /import /clear /fullscreen /voice /convo /screenshot /ambient /focus /gallery /workspace /save /stop /help`.

### Power features

- **Artifact panel**: side panel with Preview/Source tabs, drag-left-edge to resize, double-click to toggle expanded, fullscreen button (**no** more blank/stuck states during drag)
- **Artifact versioning**: same-identifier artifacts stack into a chain with prev/next arrows
- **Edit-in-place**: Source tab is editable → Save creates a new version (undo via prev arrow)
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

## Artifact types (41 total)

Model wraps rendered content in `<antArtifact identifier="…" type="…" title="…">…</antArtifact>`. Both `vnd.ant.*` and `vnd.pimio.*` namespaces are accepted (aliased).

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

## Skills (18 total)

Mirrors library choices of [anthropics/skills](https://github.com/anthropics/skills).

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

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /ui/` | Single-page HTML app |
| `WS /ui/ws/chat` | Streaming chat + tool loop |
| `GET /ui/api/config` | Session config |
| `POST /ui/api/config` | Update config (caveman, temp, sys prompt, max_tokens) |
| `GET/POST/DELETE /ui/api/sessions[/id]` | Chat persistence |
| `GET /ui/api/model-info` | Active tier, VRAM, context |
| `POST /ui/api/tier` | Switch tier |
| `POST /ui/api/upload` | File attachment (multipart) |
| `GET /ui/api/search?q=…` | Global cross-chat search |
| `POST /ui/api/share` | Mint a share link |
| `GET /ui/share/{id}` | Standalone read-only artifact view |
| `GET /ui/files/{name}[?download=1]` | Serve generated file inline or as attachment |
