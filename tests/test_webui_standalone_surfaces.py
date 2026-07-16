"""Regressions for Mio's standalone dashboards and engine HUD."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "mio" / "webui" / "assets"


class _InlineScripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self._parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and not dict(attrs).get("src"):
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._parts is not None:
            self.scripts.append("".join(self._parts))
            self._parts = None


def _source(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def test_standalone_pages_declare_mobile_layouts() -> None:
    stats = _source("stats.html")
    compare = _source("compare.html")
    attachments = _source("attachments.html")

    for source in (stats, compare, attachments):
        assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in source

    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in stats
    assert "@media (max-width: 900px)" in stats
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in stats
    assert "@media (max-width: 560px)" in stats
    assert "grid-template-columns: minmax(0, 1fr)" in stats

    assert "@media (max-width: 720px)" in compare
    assert ".columns { grid-template-columns: minmax(0, 1fr)" in compare
    assert "overflow-wrap: anywhere" in compare

    assert "@media (max-width: 600px)" in attachments
    assert ".grid { grid-template-columns: minmax(0, 1fr)" in attachments
    assert "grid-template-columns: 72px minmax(0, 1fr)" in attachments


def test_stats_handles_empty_and_error_payloads_without_dynamic_html() -> None:
    source = _source("stats.html")

    assert "if (!response.ok)" in source
    assert "data.error" in source
    assert "data.error === 'no sessions dir'" in source
    assert "totals[0][1] === 0" in source
    assert "No saved sessions yet" in source
    assert "Stats are unavailable" in source
    assert "retryButton.addEventListener('click', loadStats)" in source
    assert "replaceChildren" in source
    assert ".innerHTML" not in source


def test_attachments_has_safe_urls_and_distinct_empty_filter_state() -> None:
    source = _source("attachments.html")

    assert "function safeFileUrl(value)" in source
    assert "url.origin !== location.origin" in source
    assert "url.pathname.startsWith('/ui/files/')" in source
    assert "No generated attachments yet" in source
    assert "No attachments match" in source
    assert "Attachments are unavailable" in source
    assert "retryButton.addEventListener('click', loadAttachments)" in source
    assert "image.addEventListener('error'" in source
    assert ".innerHTML" not in source


def test_compare_reports_loading_transport_and_stream_errors() -> None:
    source = _source("compare.html")

    assert "Loading available models" in source
    assert "Generating both responses" in source
    assert "Promise.all([run('l'), run('r')])" in source
    assert "socket.onerror" in source
    assert "socket.onclose" in source
    assert "JSON.parse(event.data)" in source
    assert "The connection closed before the response completed" in source
    assert "output.replaceChildren(document.createTextNode(text), cursor)" in source
    assert ".innerHTML" not in source


def test_engine_hud_uses_backend_metrics_and_real_tier_lists() -> None:
    source = _source("engine_hud.js")
    shell = (ROOT / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")

    assert "'density', 'engine_hud', 'tips'" in shell
    assert 'fetch("/ui/api/model-info")' in source
    assert 'fetch("/ui/api/config")' in source
    assert "info.last_gen_tps" in source
    assert "configData.all_tiers" in source
    assert "configData.loaded_tiers" in source
    assert "status = \"empty\"" in source
    assert "status = \"error\"" in source
    assert "No model is loaded" in source
    assert "last_tps" not in source
    assert '["small", "medium", "large", "large-moe"]' not in source


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_standalone_inline_javascript_has_valid_syntax(tmp_path: Path) -> None:
    failures: list[str] = []
    for page_name in ("stats.html", "compare.html", "attachments.html"):
        collector = _InlineScripts()
        collector.feed(_source(page_name))
        assert collector.scripts, f"no inline scripts found in {page_name}"
        for index, script in enumerate(collector.scripts):
            script_path = tmp_path / f"{page_name}-{index}.js"
            script_path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                ["node", "--check", str(script_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                failures.append(f"{page_name}: {result.stderr}")

    hud_result = subprocess.run(
        ["node", "--check", str(ASSETS / "engine_hud.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    if hud_result.returncode:
        failures.append(f"engine_hud.js: {hud_result.stderr}")

    assert failures == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("page_name", ["stats.html", "attachments.html"])
@pytest.mark.parametrize("mode", ["empty", "error"])
def test_dashboard_empty_and_error_states_execute_without_crashing(
    page_name: str,
    mode: str,
    tmp_path: Path,
) -> None:
    collector = _InlineScripts()
    collector.feed(_source(page_name))
    script_path = tmp_path / f"{page_name}.js"
    script_path.write_text(collector.scripts[0], encoding="utf-8")

    harness = r"""
class FakeNode {
  constructor() {
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.textContent = '';
  }
  addEventListener() {}
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = String(value); }
}

const page = process.argv[2];
const mode = process.argv[3];
const ids = {};
for (const id of ['state', 'state-message', 'retry', 'kpis', 'sections', 'filter', 'grid']) {
  ids[id] = new FakeNode();
}
global.document = {
  getElementById: id => ids[id],
  createElement: () => new FakeNode(),
};
global.fetch = async () => mode === 'error' ? {
  ok: false,
  status: 503,
  json: async () => ({ error: 'unavailable' }),
} : {
  ok: true,
  status: 200,
  json: async () => page === 'stats.html' ? { error: 'no sessions dir' } : { files: [] },
};

require(process.argv[1]);
setImmediate(() => {
  const expected = mode === 'error' ? 'error' : 'empty';
  if (ids.state.dataset.kind !== expected) {
    throw new Error(`expected ${expected}, got ${ids.state.dataset.kind}`);
  }
  if (mode === 'error' && ids.retry.hidden) throw new Error('retry must be visible');
  if (mode === 'empty' && !ids.retry.hidden) throw new Error('retry must be hidden');
  if (page === 'stats.html' && mode === 'empty' && ids.kpis.children.length !== 4) {
    throw new Error('empty stats must render four zero KPIs');
  }
  if (page === 'attachments.html' && mode === 'empty' && ids.filter.disabled) {
    throw new Error('filter must remain usable in an empty library');
  }
});
"""
    result = subprocess.run(
        ["node", "-e", harness, str(script_path), page_name, mode],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_engine_hud_snapshot_reflects_api_payload() -> None:
    harness = r"""
class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.hidden = false;
    this.textContent = '';
  }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = String(value); }
  addEventListener() {}
  remove() {}
}

global.window = { Mio: {}, lastContextUsed: 512 };
global.document = {
  body: new FakeNode('body'),
  hidden: false,
  createElement: tag => new FakeNode(tag),
  addEventListener() {},
  removeEventListener() {},
};
global.setTimeout = callback => { callback(); return 1; };
global.setInterval = () => 2;
global.clearInterval = () => {};
global.fetch = async url => ({
  ok: true,
  json: async () => url.endsWith('model-info') ? {
    tier: 'mlx-27b', model_name: 'Qwen-27B-MLX', context_window: 131072,
    last_prompt_tokens: 4096, last_gen_tps: 42.25, vram_gb: 18.5,
  } : {
    active_tier: 'mlx-27b', loaded_tiers: ['mlx-27b'],
    all_tiers: ['mlx-27b', 'mlx-7b'],
  },
});

require(process.argv[1]);

(async () => {
  await window.Mio.engineHud.refresh();
  const snapshot = window.Mio.engineHud.snapshot();
  if (snapshot.status !== 'ready') throw new Error(JSON.stringify(snapshot));
  if (snapshot.active_tier !== 'mlx-27b') throw new Error(JSON.stringify(snapshot));
  if (snapshot.last_gen_tps !== 42.25) throw new Error(JSON.stringify(snapshot));
  if (snapshot.all_tiers.join(',') !== 'mlx-27b,mlx-7b') throw new Error(JSON.stringify(snapshot));
  if (snapshot.loaded_tiers.join(',') !== 'mlx-27b') throw new Error(JSON.stringify(snapshot));
  window.Mio.engineHud.unmount();
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness, str(ASSETS / "engine_hud.js")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
