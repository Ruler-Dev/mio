"""Focused regressions for WebUI lifecycle, imports, hooks, and scheduling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from mio.webui import router, scheduler, webhooks


def test_sovereignty_onboarding_opens_the_real_network_monitor():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "onboarding_sovereignty.js"
    source = asset.read_text(encoding="utf-8")

    assert 'data-act="noop"' not in source
    assert 'data-act="network"' in source
    assert 'document.querySelector(".mio-sovereignty .mio-sov-net")' in source
    assert "networkMonitor.click();" in source


def test_first_run_onboarding_is_sequenced_and_uses_current_capabilities():
    root = Path(__file__).parents[1]
    sovereignty = (root / "mio" / "webui" / "assets" / "onboarding_sovereignty.js").read_text(encoding="utf-8")
    tour = (root / "mio" / "webui" / "assets" / "onboarding.js").read_text(encoding="utf-8")
    css = (root / "mio" / "webui" / "assets" / "main.css").read_text(encoding="utf-8")

    assert 'new CustomEvent("mio:sovereignty-onboarded")' in sovereignty
    assert "SOVEREIGNTY_SEEN_KEY" in tour
    assert "mio:sovereignty-onboarded" in tour
    assert "100+ templates" not in tour
    assert "90+ more" not in tour
    assert "native MLX benchmarks" in tour
    assert "--sov-surface" in css
    assert "background: var(--sov-surface)" in css


def test_compare_requires_two_distinct_real_models():
    page = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "compare.html"
    source = page.read_text(encoding="utf-8")

    assert "ALL_TIERS.length < 2" in source
    assert "leftTier !== rightTier" in source
    assert "Two distinct tiers are required" in source
    assert "LOADED_TIERS.has(leftTier) && LOADED_TIERS.has(rightTier)" in source
    assert "runButton.disabled = busy || Boolean(configLoadError) || !selectedLoaded || !hasPrompt" in source
    assert "['small', 'medium', 'large', 'large-moe']" not in source
    assert "if (!prompt || !validateComparison()) return;" in source


def test_settings_mcp_health_panel_is_retryable_and_uses_safe_dom_rendering():
    root = Path(__file__).parents[1]
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    css = (root / "mio" / "webui" / "assets" / "main.css").read_text(encoding="utf-8")
    health_ui = shell.split("function renderMcpHealth(payload)", 1)[1].split(
        "function openSettings()", 1
    )[0]

    assert 'id="mcpHealthRetry"' in shell
    assert "fetch('/v1/mcp/health'" in health_ui
    assert "method: 'POST'" in health_ui
    assert "list.replaceChildren();" in health_ui
    assert "name.textContent" in health_ui
    assert ".innerHTML" not in health_ui
    assert "retry.disabled = true" in health_ui
    assert "refreshMcpHealth();" in shell.split("function openSettings()", 1)[1]
    assert ".mcp-health-card" in css
    assert '.mcp-health-state[data-state="degraded"]' in css


def test_emoji_button_uses_parent_prepend_instead_of_descendant_insert_before():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "emoji.js"
    source = asset.read_text(encoding="utf-8")
    add_button = source.split("function addButton()", 1)[1].split(
        "function injectCSS()", 1
    )[0]

    assert "actions.prepend(btn);" in add_button
    assert "actions.insertBefore(" not in add_button
    assert "actions.querySelector('button')" not in add_button


def test_mobile_layout_does_not_create_an_implicit_metrics_column():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "main.css"
    source = asset.read_text(encoding="utf-8")
    responsive = source.split("/* ===== Responsive ===== */", 1)[1].split(
        "@media (prefers-reduced-motion", 1
    )[0]

    assert "grid-template-columns: 48px minmax(0, 1fr)" in responsive
    assert ".metrics-bar" in responsive
    assert "grid-column: 2" in responsive


def test_flow_has_a_canvas_first_mobile_layout():
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "main.css"
    source = asset.read_text(encoding="utf-8")
    flow_mobile = source.split("@keyframes pulse", 1)[1].split(
        "/* ===== Journal view", 1
    )[0]

    assert "@media (max-width: 768px)" in flow_mobile
    assert "grid-template-columns: minmax(0, 1fr)" in flow_mobile
    assert "grid-template-rows: 140px minmax(320px, 1fr)" in flow_mobile
    assert ".flow-nodes" in flow_mobile
    assert "flex-direction: row" in flow_mobile


def test_native_mlx_artifact_lab_renders_without_remote_dependencies():
    root = Path(__file__).parents[1]
    registry = root / "mio" / "webui" / "assets" / "artifact_registry.js"
    asset = root / "mio" / "webui" / "assets" / "artifact_lab.js"
    registry_source = registry.read_text(encoding="utf-8")
    source = asset.read_text(encoding="utf-8")
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")

    assert '<script src="/ui/assets/artifact_registry.js"></script>' in shell
    assert '<script src="/ui/assets/artifact_lab.js"></script>' in shell
    assert shell.index("artifact_registry.js") < shell.index("artifact_lab.js")
    assert "https://" not in registry_source
    assert ".innerHTML" not in registry_source
    assert "https://" not in source
    assert ".innerHTML" not in source
    assert "application/vnd.pimio.benchmark+json" in source
    assert "application/vnd.pimio.model-card+json" in source
    assert "application/vnd.pimio.inference-trace+json" in source
    assert "application/vnd.pimio.speculative-acceptance-atlas+json" in source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    smoke = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.attributes = {};
    this.dataset = {}; this.style = {}; this.className = ''; this.textContent = '';
    this.classList = { add: (value) => { this.className += ' ' + value; } };
  }
  append(...values) { this.children.push(...values); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
}
global.window = {};
global.document = { createElement: (tag) => new Element(tag) };
require(process.argv[1]);
require(process.argv[2]);
const lab = window.Mio.artifactLab;
const registry = window.Mio.artifactTypes;
assert.equal(lab.catalog().length, 4);
assert.equal(registry.catalog().length, 4);
for (const artifact of [
  {type:'application/vnd.pimio.benchmark+json', content:JSON.stringify({runs:[{label:'base',prefill_tps:20,decode_tps:10,ttft_ms:5}]})},
  {type:'application/vnd.pimio.model-card+json', content:JSON.stringify({name:'Qwen',quantization:'Q4'})},
  {type:'application/vnd.pimio.inference-trace+json', content:JSON.stringify({spans:[{name:'prefill',start_ms:0,duration_ms:4}]})},
  {type:'application/vnd.pimio.speculative-acceptance-atlas+json', content:lab.sample('application/vnd.pimio.speculative-acceptance-atlas+json')},
]) {
  const body = new Element('div');
  assert.equal(lab.render(body, artifact), true);
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].className.includes('has-error'), false);
  assert.equal(lab.download(artifact).extension, '.json');
  const registeredBody = new Element('div');
  assert.equal(registry.render(registeredBody, artifact), true);
  assert.equal(registeredBody.children.length, 1);
  assert.equal(registry.download(artifact).extension, '.json');
}
assert.equal(registry.render(new Element('div'), {type:'application/vnd.pimio.unknown',content:'{}'}), false);
"""
    subprocess.run(
        [node, "-e", smoke, str(registry), str(asset)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_artifact_mime_normalization_is_idempotent_and_reaches_real_renderers():
    root = Path(__file__).parents[1]
    shell_path = root / "mio" / "webui" / "mio_ui.html"
    shell = shell_path.read_text(encoding="utf-8")
    start = shell.index("function _normalizeArtifactType(t)")
    end = shell.index("function renderArtifactPreview(body, art)", start)
    function_source = shell[start:end]
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    check = "global.window = {};\n" + function_source + r"""
const assert = require('node:assert/strict');
const cases = new Map([
  ['application/vnd.ant.react', 'application/vnd.ant.react'],
  ['application/vnd.pimio.react', 'application/vnd.ant.react'],
  ['application/vnd.ant.code', 'application/vnd.ant.code'],
  ['application/vnd.pimio.code', 'application/vnd.ant.code'],
  ['application/vnd.pimio.mermaid', 'application/vnd.ant.mermaid'],
  ['application/vnd.pimio.dxf', 'application/vnd.pimio.dxfviewer'],
]);
for (const [input, expected] of cases) {
  const once = _normalizeArtifactType(input);
  assert.equal(once, expected);
  assert.equal(_normalizeArtifactType(once), expected);
}
"""
    subprocess.run([node, "-e", check], check=True, capture_output=True, text=True)


def test_unknown_artifact_types_are_preserved_as_source_instead_of_executed():
    root = Path(__file__).parents[1]
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    css = (root / "mio" / "webui" / "assets" / "main.css").read_text(encoding="utf-8")
    preview = shell.split("function renderArtifactPreview(body, art)", 1)[1].split(
        "function makeSandboxedIframe", 1
    )[0]

    assert "if (art.type === 'text/html')" in preview
    assert "Renderer not installed" in preview
    assert "source.textContent = String(art.content || '')" in preview
    assert "Default: text/html" not in preview
    assert ".artifact-unsupported" in css

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    start = shell.index("function legacyArtifactDocument(art)")
    end = shell.index("function artifactIcon(t)", start)
    helpers = shell[start:end]
    check = "global.window = {Mio: {}};\n" + helpers + r"""
const assert = require('node:assert/strict');
assert.deepEqual(
  artifactDownloadPayload({type:'application/vnd.pimio.unknown', content:'<script>boom()</script>'}),
  {content:'<script>boom()</script>', extension:'.txt', mime:'text/plain'}
);
assert.equal(
  artifactDownloadPayload({type:'application/vnd.ant.code', content:'const x = 1'}).extension,
  '.txt'
);
assert.equal(
  artifactDownloadPayload({type:'text/html', content:'<h1>safe boundary</h1>'}).extension,
  '.html'
);
window.reactTemplate = (source) => '<!doctype html>' + source;
assert.equal(
  artifactDownloadPayload({type:'application/vnd.ant.react', content:'<App />'}).extension,
  '.html'
);
"""
    subprocess.run([node, "-e", check], check=True, capture_output=True, text=True)


def test_node_editor_is_a_real_local_interactive_renderer():
    root = Path(__file__).parents[1]
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    start = shell.index("function nodeEditorTemplate(json)")
    end = shell.index("function abcMusicTemplate", start)
    template = shell[start:end]

    assert "Loading Rete.js" not in template
    assert "rete.min.js" not in template
    assert "function drawEdges()" in template
    assert "function autoLayout()" in template
    assert "pointerdown" in template
    assert "Add node" in template

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    embedded = template.split("<script>\n", 1)[1].split("<\\/script>", 1)[0]
    embedded = embedded.replace("${safeJSONForScript(json)}", "'{}'")
    subprocess.run(
        [node, "--check", "-"],
        input=embedded,
        check=True,
        capture_output=True,
        text=True,
    )


def test_periodic_table_has_all_118_elements_and_local_interactions():
    root = Path(__file__).parents[1]
    asset = root / "mio" / "webui" / "assets" / "artifact_periodic.js"
    source = asset.read_text(encoding="utf-8")
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")

    assert '<script src="/ui/assets/artifact_periodic.js"></script>' in shell
    assert "window.Mio.artifactPeriodic.template()" in shell
    assert "SYMBOLS.length !== 118 || NAMES.length !== 118" in source
    assert "complete local dataset" in source
    assert "Search name, symbol, or number" in source
    assert "https://" not in source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    smoke = r"""
global.window = {};
require(process.argv[1]);
if (window.Mio.artifactPeriodic.count() !== 118) throw new Error('incomplete table');
const html = window.Mio.artifactPeriodic.template();
if (!html.includes('Oganesson') || !html.includes("data-number='118'")) throw new Error('missing final element');
if (!html.includes("id='query'")) throw new Error('missing search');
"""
    subprocess.run(
        [node, "-e", smoke, str(asset)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_artifact_state_v2_round_trips_revisions_and_dotted_identifiers():
    root = Path(__file__).parents[1]
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")

    assert "function serializeArtifactState()" in shell
    assert "function restoreArtifactState(state, legacyArtifacts = [])" in shell
    assert "artifact_state: serializeArtifactState()" in shell
    assert "schema_version: 2" in shell
    assert "active_index:" in shell and "revisions:" in shell
    assert "parent: cur.content_id" in shell
    assert r"([A-Za-z0-9][A-Za-z0-9._-]{0,127})" in shell


@pytest.mark.asyncio
async def test_session_endpoint_preserves_artifact_revision_state(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "_sessions_dir", tmp_path)
    state = {
        "schema_version": 2,
        "active_artifact_id": "mlx.trace.v1",
        "chains": [
            {
                "id": "mlx.trace.v1",
                "active_index": 1,
                "revisions": [
                    {
                        "id": "mlx.trace.v1",
                        "type": "application/vnd.pimio.inference-trace+json",
                        "title": "Trace",
                        "content": '{"spans":[{"name":"prefill","start_ms":0,"duration_ms":4}]}',
                        "content_id": "fnv1a32:00000001",
                        "provenance": {"producer": "benchmark", "run_id": "run-1"},
                    },
                    {
                        "id": "mlx.trace.v1",
                        "type": "application/vnd.pimio.inference-trace+json",
                        "title": "Trace (annotated)",
                        "content": '{"spans":[{"name":"prefill","start_ms":0,"duration_ms":3.8}]}',
                        "content_id": "fnv1a32:00000002",
                        "provenance": {"producer": "editor", "parent": "fnv1a32:00000001"},
                    },
                ],
            }
        ],
    }

    result = await router.save_session(
        {
            "id": "artifact-roundtrip",
            "messages": [{"role": "user", "content": "Compare these runs"}],
            "artifact_state": state,
        }
    )
    loaded = await router.load_session(result["id"])

    assert loaded["artifact_state"] == state
    assert loaded["artifact_state"]["chains"][0]["active_index"] == 1
    assert loaded["artifact_state"]["chains"][0]["revisions"][1]["provenance"]["parent"] == "fnv1a32:00000001"


@pytest.mark.asyncio
async def test_shared_native_artifact_uses_the_registry_and_accepts_dotted_ids():
    router._shared_artifacts.clear()
    result = await router.share_artifact(
        {
            "identifier": "mlx.trace.v1",
            "type": "application/vnd.pimio.inference-trace+json",
            "title": "Measured trace",
            "content": '{"spans":[{"name":"prefill","start_ms":0,"duration_ms":4}]}',
        }
    )
    response = await router.view_shared_artifact(result["id"])
    html = response.body.decode("utf-8")

    assert result["url"] == "/ui/share/mlx.trace.v1"
    assert "/ui/assets/artifact_registry.js" in html
    assert "/ui/assets/artifact_lab.js" in html
    assert "artifactTypes?.render(mount, art)" in html
    assert "art.type.startsWith('application/vnd.pimio.')" not in html


def test_artifact_panel_is_mobile_sheet_and_core_controls_are_accessible():
    root = Path(__file__).parents[1]
    shell = (root / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    css = (root / "mio" / "webui" / "assets" / "main.css").read_text(encoding="utf-8")

    assert '.app.artifact-open .artifact-panel' in css
    assert "inset: 0 0 0 48px" in css
    assert ".artifact-resizer { display: none; }" in css
    assert ":focus-visible" in css
    assert 'role="tablist"' in shell
    assert 'role="dialog" aria-modal="true"' in shell
    assert 'aria-label="Send message"' in shell
    assert 'type="button" class="artifact-card"' in shell


@pytest.mark.asyncio
async def test_tier_switch_loads_new_tier_before_retiring_old(monkeypatch):
    lock = threading.Lock()
    events: list[str] = []

    class Manager:
        config = SimpleNamespace(tiers={"old": object(), "new": object()})

        def __init__(self):
            self.loaded = ["old"]

        def loaded_tiers(self):
            assert lock.locked()
            events.append("snapshot")
            return list(self.loaded)

        def load_tier(self, tier):
            assert lock.locked()
            events.append(f"load:{tier}")
            self.loaded.append(tier)

        def unload_tier(self, tier):
            assert lock.locked()
            events.append(f"unload:{tier}")
            self.loaded.remove(tier)

    manager = Manager()
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)

    result = await router.switch_tier({"tier": "new"})

    assert result == {"ok": True, "tier": "new", "already_loaded": False}
    assert events == ["snapshot", "load:new", "unload:old"]
    assert manager.loaded == ["new"]


@pytest.mark.asyncio
async def test_tier_switch_load_failure_keeps_serving_tier(monkeypatch):
    lock = threading.Lock()

    class Manager:
        config = SimpleNamespace(tiers={"old": object(), "new": object()})

        def __init__(self):
            self.loaded = ["old"]
            self.unloaded: list[str] = []

        def loaded_tiers(self):
            assert lock.locked()
            return list(self.loaded)

        def load_tier(self, _tier):
            assert lock.locked()
            raise MemoryError("insufficient unified memory")

        def unload_tier(self, tier):
            self.unloaded.append(tier)

    manager = Manager()
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)

    with pytest.raises(HTTPException) as raised:
        await router.switch_tier({"tier": "new"})

    assert raised.value.status_code == 409
    assert manager.loaded == ["old"]
    assert manager.unloaded == []


@pytest.mark.asyncio
async def test_tier_switch_retires_other_tiers_when_target_is_already_loaded(
    monkeypatch,
):
    lock = threading.Lock()

    class Manager:
        config = SimpleNamespace(tiers={"old": object(), "new": object()})

        def __init__(self):
            self.loaded = ["old", "new"]
            self.load_calls: list[str] = []

        def loaded_tiers(self):
            assert lock.locked()
            return list(self.loaded)

        def load_tier(self, tier):
            self.load_calls.append(tier)

        def unload_tier(self, tier):
            assert lock.locked()
            self.loaded.remove(tier)

    manager = Manager()
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)

    result = await router.switch_tier({"tier": "new"})

    assert result == {"ok": True, "tier": "new", "already_loaded": True}
    assert manager.loaded == ["new"]
    assert manager.load_calls == []


def _chatgpt_node(
    parent: str | None,
    role: str,
    text: str,
    *,
    created: float,
) -> dict:
    return {
        "parent": parent,
        "children": [],
        "message": {
            "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "create_time": created,
        },
    }


def test_chatgpt_import_uses_selected_branch_instead_of_flattening_siblings():
    mapping = {
        "root": _chatgpt_node(None, "user", "question", created=1),
        "old": _chatgpt_node("root", "assistant", "old answer", created=2),
        "active": _chatgpt_node("root", "assistant", "active answer", created=3),
    }
    mapping["root"]["children"] = ["old", "active"]

    messages = router._normalize_imported_chat(
        {"mapping": mapping, "current_node": "active"}
    )

    assert messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "active answer"},
    ]


def test_chatgpt_import_is_iterative_and_falls_back_to_newest_leaf():
    mapping: dict[str, dict] = {}
    parent = None
    for index in range(2_000):
        node_id = f"n{index}"
        mapping[node_id] = _chatgpt_node(
            parent,
            "user" if index % 2 == 0 else "assistant",
            str(index),
            created=index,
        )
        parent = node_id

    messages = router._normalize_imported_chat({"mapping": mapping})

    assert len(messages) == 2_000
    assert messages[-1]["content"] == "1999"


def test_chat_import_validates_full_batch_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "_sessions_dir", tmp_path)
    too_large = "x" * (router._MAX_IMPORT_MESSAGE_BYTES + 1)
    body = {
        "source": "mio",
        "data": [
            {"messages": [{"role": "user", "content": "valid"}]},
            {"messages": [{"role": "user", "content": too_large}]},
        ],
    }

    with pytest.raises(HTTPException, match="2 MiB"):
        asyncio.run(router.import_chats(body))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "invalid",
    [True, False, "16", 16.0, None, 0, -1, 32_769],
)
def test_webui_config_rejects_non_integer_or_unbounded_max_tokens(
    invalid,
    monkeypatch,
):
    monkeypatch.setattr(router, "_max_tokens", 123)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(router.update_config({"max_tokens": invalid}))

    assert raised.value.status_code == 400
    assert router._max_tokens == 123


@pytest.mark.parametrize("valid", [1, 32_768])
def test_webui_config_accepts_bounded_max_tokens(valid, monkeypatch):
    monkeypatch.setattr(router, "_max_tokens", 123)

    result = asyncio.run(router.update_config({"max_tokens": valid}))

    assert result["max_tokens"] == valid
    assert router._max_tokens == valid


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [True, "8", 8.0, None, 0, 32_769])
async def test_webui_socket_rejects_invalid_max_tokens_before_side_effects(
    invalid,
    monkeypatch,
):
    class Socket:
        def __init__(self):
            self.events: list[dict] = []

        async def send_json(self, value):
            self.events.append(value)

    socket = Socket()
    monkeypatch.setattr(router, "_manager", None)

    await router._handle_chat(
        socket,
        {
            "max_tokens": invalid,
            "messages": [{"role": "user", "content": "/remember must-not-save"}],
        },
    )

    assert socket.events == [
        {
            "type": "error",
            "message": "max_tokens must be an integer between 1 and 32768",
        }
    ]


def test_webhook_requires_secret_and_never_returns_auth_material(tmp_path, monkeypatch):
    hooks_file = tmp_path / "webhooks.json"
    monkeypatch.setattr(webhooks, "_WEBHOOKS_FILE", hooks_file)

    with pytest.raises(ValueError, match="secret is required"):
        webhooks.create_webhook("build", "Build {{target}}")

    created = webhooks.create_webhook(
        "build",
        "Build {{target}}",
        secret="correct horse battery staple",
    )
    [stored] = json.loads(hooks_file.read_text())
    [public] = webhooks.public_webhooks()

    assert created["ok"] is True
    assert "secret" not in stored
    assert stored["secret_hash"].startswith("sha256:")
    assert webhooks.verify_secret(stored, "correct horse battery staple")
    assert not webhooks.verify_secret(stored, "wrong secret value")
    assert public["has_secret"] is True
    assert "secret" not in public and "secret_hash" not in public


def test_webhook_log_redacts_bounds_and_rotates(tmp_path, monkeypatch):
    log = tmp_path / "webhooks-log.jsonl"
    monkeypatch.setattr(webhooks, "_WEBHOOKS_LOG", log)
    monkeypatch.setattr(webhooks, "_LOG_MAX_BYTES", 700)
    monkeypatch.setattr(webhooks, "_LOG_BACKUPS", 2)

    for index in range(12):
        webhooks.append_log(
            "build",
            {
                "secret": "never-write-this-secret",
                "Authorization": "Bearer never-write-this-token",
                "safe": "x" * 180,
                "index": index,
            },
            {"ok": True, "output": "y" * 180},
        )

    paths = [log, log.with_name(log.name + ".1"), log.with_name(log.name + ".2")]
    content = b"".join(path.read_bytes() for path in paths if path.exists())
    assert b"never-write-this" not in content
    assert b"<redacted>" in content
    assert all(path.stat().st_size <= 700 for path in paths if path.exists())
    assert log.with_name(log.name + ".1").exists()
    assert len(webhooks.recent_runs(10_000)) <= 100


def test_once_schedule_runs_exactly_once_and_normalizes_timestamp():
    cadence = scheduler.validate_cadence(
        {"kind": "once", "at": "2026-07-15T12:30:00"}
    )
    schedule = {"enabled": True, "cadence": cadence}

    assert cadence == {"kind": "once", "at": "2026-07-15T12:30:00"}
    assert not scheduler._should_run_now(
        schedule,
        scheduler._dt.datetime(2026, 7, 15, 12, 29),
        None,
    )
    assert scheduler._should_run_now(
        schedule,
        scheduler._dt.datetime(2026, 7, 15, 12, 30),
        None,
    )
    assert not scheduler._should_run_now(
        schedule,
        scheduler._dt.datetime(2026, 7, 15, 12, 31),
        "2026-07-15T12:30:01",
    )


def test_natural_language_schedule_emits_scheduler_kind_schema():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "nl_schedule.js"
    harness = f"""
global.window = {{ Mio: {{}} }};
global.document = {{ readyState: 'loading', addEventListener() {{}} }};
eval(require('fs').readFileSync({json.dumps(str(asset))}, 'utf8'));
const api = window.Mio.nlSchedule;
console.log(JSON.stringify({{
  daily: api.toCadence(api.parse('every day at 8:30am stretch')),
  weekly: api.toCadence(api.parse('every Sunday at 9am plan')),
  once: api.toCadence(api.parse('in 30 minutes stretch')),
  unsupported: api.parse('every weekday at 8am plan'),
}}));
"""

    completed = subprocess.run(
        [node, "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["daily"] == {"kind": "daily", "hour": 8, "minute": 30}
    assert result["weekly"] == {
        "kind": "weekly",
        "weekday": 6,
        "hour": 9,
        "minute": 0,
    }
    assert result["once"]["kind"] == "once"
    assert set(result["once"]) == {"kind", "at"}
    assert result["unsupported"] is None
