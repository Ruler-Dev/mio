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

## July 2026 UI reliability checkpoint

This checkpoint is a set of implementation contracts and failure-state fixes,
not a redesign claim or a benchmark result. It covers the artifact type
boundary, view lifecycle, shared skill calls, model/workspace activation, and
the standalone surfaces that previously depended on optimistic assumptions.

### Artifact registry and MIME execution boundary

`/ui/assets/artifact_registry.js` exposes `window.Mio.artifactTypes` with the
following operations:

- `register(definition)` validates and records a canonical MIME type, bounded
  display metadata, aliases, a required renderer, and optional download and
  standalone handlers;
- `normalize(type)` follows registered aliases without looping;
- `definition`, `supports`, and `catalog` report only definitions that were
  successfully registered with a renderer;
- `render` and `download` pass a canonical artifact record to the owning
  definition.

Registration rejects malformed MIME names, duplicate canonical types,
duplicate alias ownership, missing render functions, and empty or oversized
labels. The native MLX benchmark, model-card, inference-trace, and speculative
acceptance-atlas definitions are currently the first consumers. Labels, help
entries, preview dispatch,
normalization, and structured downloads for those types therefore derive from
the same definition. The large historical renderer set has not all migrated
to this interface; its maintained branches still live in the main artifact
dispatcher. The registry should be read as the stable migration boundary, not
as an assertion that legacy dispatch no longer exists.

Preview dispatch follows this order:

1. normalize the incoming type;
2. ask the registered native renderer;
3. select an exact maintained legacy renderer where one exists;
4. execute generic markup only when the canonical type is exactly
   `text/html`;
5. otherwise render a parent-DOM **Renderer not installed** panel whose source
   uses `textContent`.

The share route uses the same registry for the four native Artifact Lab
types and accepts dotted identifiers. Exact `text/html` remains an
`allow-scripts` sandboxed iframe. An unknown vendor type is displayed as
source, rather than being turned into an HTML iframe merely because its name
starts with `application/vnd.pimio.`. This is an execution routing boundary;
it does not authenticate the producer or make known executable types safe to
trust.

### New local artifact implementations

The node-editor artifact no longer waits on a remote Rete.js bootstrap. Its
packaged template accepts either `{nodes, connections}` or `{nodes, edges}`.
It constructs nodes with DOM APIs, draws cubic SVG edges, preserves labels and
ports as text, supports pointer dragging, and provides Add node, Auto layout,
and Center controls. Input is bounded to 80 nodes, 160 connections, eight
input/output ports per node, and bounded label fields. Invalid JSON produces a
visible error panel. Added and moved nodes alter the current preview only;
there is no serializer that writes those visual changes back to the Source
revision.

`/ui/assets/artifact_periodic.js` carries a complete local list of 118 element
symbols and names plus conventional grid positions. The renderer lays out
groups 1–18, places lanthanides and actinides in separate f-block rows, assigns
display categories, supports search by name/symbol/number, and updates a live
detail region on selection. Module initialization asserts exactly 118 records
and a position for each symbol. The main shell retains its older 36-element
template only as an asset-load fallback; normal packaged rendering takes the
complete asset. The data is a navigational periodic table, not a chemistry
database: it does not include masses, isotopes, oxidation states, or computed
properties.

### Asynchronous view lifecycle

`/ui/assets/views.js` now treats a route change as an asynchronous lifecycle:

```text
previous deactivate -> previous cleanup -> new mount/render -> new activate
                    -> publish host -> persist active view
```

Each non-chat view receives `{view, token, signal, isCurrent}`. The router
awaits `mount` or `render`, then `activate`, and awaits `deactivate`, returned
cleanup handles, and the explicit `cleanup` hook during disposal. It displays
an `aria-busy` loading panel while mounting, publishes the new host only after
successful activation, and displays a bounded error panel with Retry if the
current navigation fails. A later navigation aborts the earlier signal and a
monotonic token prevents a stale view from becoming active. Synchronous hooks
continue to work because their return values are normalized to promises.

Cancellation remains cooperative. A hook that starts detached work and does
not return its promise cannot be awaited. A returned promise that ignores the
signal may still settle later; its late cleanup handle is executed, and the
aborted host is not committed, but the underlying operation itself cannot be
forcibly stopped by the router.

### Shared skill API client

`/ui/assets/api_client.js` defines `Mio.api.runSkill(name, args, options)`. It
validates the local call shape, delegates to `Mio.security.runSkill` when that
transport is present, and otherwise performs a credentialed same-origin
`POST /ui/api/skills/run`. A sensitive call carries both
`confirm_sensitive: true` and the matching `X-Mio-Dangerous-Action` header.
The client distinguishes transport failure, non-JSON response, non-successful
HTTP status, `{ok: false}`, and a missing `result`; successful calls return the
unwrapped `result` object.

Design Research, Notebook, ShaderToy import, Playground, and both workspace
surfaces use that contract. The Blender integration sends only the
allow-listed `blender_exec` and `blender_snapshot` operations through a
sandbox-safe message bridge and requires explicit confirmation for the
execution path. The client is not an authorization layer: a listed skill can
still be denied by Mio's server-side grants, confirmation, or tool policy, and
each surface remains responsible for clear progress and error feedback.

### Prompt-policy controls and checked workspace activation

The main Settings panel exposes one prompt-policy selector with `none` and all
Caveman/Ponytail `lite`, `full`, and `ultra` combinations. It reads
`prompt_mode`, `prompt_level`, and `prompt_policy`, with a fallback for older
`caveman` responses. Config saves and all three WebSocket chat paths (new
turn, regenerate, and edit/resend) use the modern fields; `prompt_level` is
not sent with `none`. The sidebar label therefore reports the effective mode
and level rather than always claiming Caveman.

The Workspaces view persists a reusable project plus an optional runtime
profile: tier, minimum context capacity, and prompt policy. A workspace can
inherit the current policy or pin `none`, Caveman, or Ponytail with a level.
Legacy `caveman_level` projects remain readable and are rewritten to modern
fields on their next save. Clicking **Open chat** first calls
`POST /ui/api/projects/{id}/activate`. The backend validates the persisted
project shape, known tier, positive minimum context, and prompt policy before
entering the engine lock. It then:

1. resolves the requested or active tier and validates its configured context
   capacity;
2. rejects an unsatisfied requirement before loading/unloading a model or
   changing prompt policy;
3. loads a requested tier before retiring the old loaded tier;
4. publishes the resolved Caveman policy only after the tier operation has
   succeeded;
5. returns the effective runtime, context check, project-context summary, and
   explicit warnings.

Only after this response succeeds does the browser refresh projects, select
the project ID, refresh config, and switch to Chat. Project files and system
prompt remain request-scoped: they take effect because later chat requests
carry the selected project, not because the activation endpoint copies them
into global configuration. Legacy `pinned_prompts` are not supported and are
reported as unapplied instead of silently presented as active.

This sequence avoids the previous optimistic UI state, but it is not a
distributed rollback protocol. A model/policy activation can succeed and a
subsequent browser-local project-selection step can fail; the card reports
that distinction (`Runtime profile applied, but chat selection failed`) and
does not pretend to have reverted engine memory.

### Explicit two-model Compare

`/ui/compare` reads `all_tiers` and `loaded_tiers` from
`GET /ui/api/config`; it does not derive residency from a static tier list.
The page displays configured and loaded inventories separately. A comparison
is runnable only when the prompt is non-empty and two distinct selected tiers
are confirmed loaded by the latest response.

Selecting a configured but unloaded tier does not trigger work. **Load
selected tiers** shows a unified-memory warning and, only after confirmation,
calls `POST /v1/models/load` once for each missing selected tier. It refreshes
configuration after the load and leaves Run disabled if residency is not
confirmed. Compare never calls an unload endpoint, so existing resident tiers
remain in memory. Once ready, it opens the existing per-tier WebSocket paths
in parallel and reports each completion's returned token/tok/s metadata and
elapsed wall time.

This behavior favors explicitness over memory automation. Loading two large
tiers can fail or pressure unified memory; the operator must unload elsewhere
when appropriate. Compare also does not establish prompt parity beyond using
the same entered text, normalize generation settings into a benchmark
protocol, score output quality, or save a research result. Use the benchmark
harness and native benchmark artifact for measured claims.

### Backend-driven MLX HUD

`/ui/assets/engine_hud.js` combines `GET /ui/api/config` with
`GET /ui/api/model-info`. It renders explicit loading, empty, ready, switching,
and error states; reports configured/loaded and active tiers; and shows the
backend's `last_gen_tps`, last prompt tokens, context capacity/use, model name,
and reported VRAM where available. Expanding the HUD exposes only tiers from
the backend inventory. Switching delegates to the main shell's tier operation
and refreshes state afterward; there is no phantom hard-coded tier fallback.

The HUD refreshes every five seconds while the document is visible and also
on expansion or return to the page. `last_gen_tps` describes the last completed
generation available to the endpoint. It is not a real-time sampling stream,
an average, or a substitute for the benchmark harness. Context use can use the
shell's last observed value before falling back to model-info data.

### Standalone states and responsive behavior

The standalone pages now declare a mobile viewport and avoid treating a
pending request as a final UI:

| Surface | Implemented states and narrow-layout behavior | Remaining limits |
|---|---|---|
| Stats | loading, empty, error, Retry; 4/2/1-column responsive cards | saved-session summaries, not live engine telemetry |
| Compare | configuration/residency inventory, explicit loading and WebSocket errors; one response column below 720 px | keeping two resident tiers can exceed memory |
| Attachments | loading, empty, error, Retry, local search; same-origin generated-file URLs | only Mio's generated-file inventory, not a general file browser |
| Playground | shared skill response handling with disabled/loading and surfaced failures | a skill may still require a grant or confirmation |
| Workspace | shared skill response handling and explicit failures | separate from File System Access live-folder permissions |

These changes target the known indefinite-loading, empty-crash, response-
envelope, and horizontal-overflow failures. They do not imply uniform feature
parity between the main shell and every standalone page.

### Sequential onboarding

Two first-run surfaces remain, but they cannot open simultaneously. The
local-data/sovereignty card stores `mio.sovereignty.onboarded.v1` and emits
`mio:sovereignty-onboarded` when dismissed. The feature tour opens only when
that flag already exists or after that event. Its copy refers to the live
command/template catalog and named native MLX artifacts rather than a fixed
marketing count. Both flows remain replayable through their existing UI
entrypoints.

The sequencing removes overlapping overlays and the sovereignty card has
dark/light surface variables that do not depend on an accidental global card
color. Complete focus trapping, focus restoration, reduced-motion review,
screen-reader validation, and all onboarding actions' failure feedback remain
release work.

### Local parent-DOM PNG export

`/ui/assets/artifact_export.js` implements `/screenshot-artifact` without a
remote screenshot library. For an active parent-DOM artifact it measures and
clones the rendered node, copies an allow-listed set of computed styles and
the current values of form controls, replaces local canvas pixels and
same-origin/data images with embedded data, serializes the clone inside an SVG
`foreignObject`, draws that image to a canvas, and downloads an `image/png`.
Object URLs are revoked after use.

The pipeline fails closed at these bounds:

| Resource | Bound |
|---|---:|
| CSS dimensions | 2,048 × 4,096 |
| cloned DOM nodes | 1,800 |
| output pixels | 8 MiPixels |
| embedded images in aggregate | 4 MiB |
| serialized SVG | 10 MiB |
| PNG | 16 MiB |

An artifact body containing an iframe is not inspected. Mio's executable
artifact frames deliberately omit `allow-same-origin`, so reading their
document would violate the intended opaque-origin boundary. The command
instead reports `sandboxed-frame` and directs the operator to **Download as
file** or the operating-system screenshot tool. External images, CSS
background URLs, tainted canvases, live audio/video, excessive dimensions,
and browsers that cannot rasterize SVG `foreignObject` also receive explicit
failures. Native Artifact Lab output is a primary supported parent-DOM case;
PNG export is not guaranteed for every legacy renderer.

### Verification scope

Focused automated tests cover registry ownership/aliases, native artifact
parsing and downloads, unknown-MIME preview/share behavior, dotted share IDs,
schema-v2 revision persistence, the 118-element invariant, node-editor script
syntax, asynchronous navigation ordering/cancellation/cleanup, skill response
validation, transactional workspace activation, explicit Compare loads, the
standalone response states, and PNG bounds/fallbacks. JavaScript syntax checks
run with Node, while backend contracts run through pytest.

A manual visual smoke was run in Mio's integrated in-app browser at desktop
and narrow/mobile viewport sizes. It checked the sequential first-run overlays,
dark-surface contrast, the main shell/artifact layout, and responsive
standalone structure. It did not use or claim control of the user's Chrome
session. This smoke is narrower than a release matrix: Safari, Chrome,
Firefox, physical touch devices, zoom, screen readers, and full keyboard-only
flows still require dedicated passes.

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
four Artifact Lab rows show their complete values.

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
| `application/vnd.pimio.speculative-acceptance-atlas+json` | Versioned matched-experiment JSON | native DOM acceptance/robustness atlas |

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

The four MLX research renderers are implemented by the packaged
`/ui/assets/artifact_lab.js` asset. They do not fetch a CDN library, navigate
to a remote URL, or execute payload code. The module parses a JSON object and
builds ordinary DOM nodes; payload values enter those nodes through
`textContent`, not `innerHTML`. This is a narrower rendering surface than the
executable iframe-based artifact types, but it is not a signature or a claim
that the supplied measurements are true.

All four types reject a UTF-8 payload larger than 512 KiB and show an error
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

The model-facing system prompt instructs the model to use benchmark, trace,
and acceptance-atlas artifacts only for supplied measurements and never to
fabricate values or spans.

### Speculative acceptance atlas

`application/vnd.pimio.speculative-acceptance-atlas+json` is a decision aid
for matched speculative-decoding experiments. It requires schema
`pimio.speculative-acceptance-atlas`, version `1`, positive baseline and
candidate prefill/decode rates, peak-memory values, and:

- 1–64 unique draft-position rows with an acceptance fraction and sample
  count;
- 1–24 named workload/context phases with token bounds, acceptance, speedup,
  peak memory, and sample count;
- a reliability block with matched-run count, confidence level, ordered
  speedup interval, and regression rate;
- an explicit `promote`, `hold`, `reject`, or `collect-more` decision and a
  bounded rationale.

The renderer derives the headline prefill/decode ratios, weighted acceptance,
and memory delta, then keeps the phase table and uncertainty next to the
decision. It does not run an experiment or validate parity. The packaged
`artifactLab.sample()` payload is labeled synthetic and exists only to
exercise the schema and renderer; replace every value with benchmark output
before using the artifact as evidence.

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
| `POST /ui/api/projects/{id}/activate` | Validate and activate the supported workspace runtime profile |
| `GET/POST/DELETE /ui/api/sessions[/id]` | Chat persistence |
| `GET /ui/api/model-info` | Active tier, VRAM, context |
| `POST /v1/models/load` | Explicitly load an additional configured tier, including from Compare |
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
