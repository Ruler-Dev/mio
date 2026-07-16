from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "mio" / "webui" / "assets" / "compare.html"


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


def _node() -> str:
    executable = shutil.which("node")
    if not executable:
        pytest.skip("node is required for WebUI JavaScript tests")
    return executable


def _inline_script() -> str:
    collector = _InlineScripts()
    collector.feed(COMPARE.read_text(encoding="utf-8"))
    assert len(collector.scripts) == 1
    return collector.scripts[0]


def test_compare_uses_verified_residency_and_only_additive_manual_loads() -> None:
    source = COMPARE.read_text(encoding="utf-8")

    assert "fetch('/ui/api/config')" in source
    assert "data.all_tiers" in source
    assert "data.loaded_tiers" in source
    assert "Configured:" in source
    assert "Loaded:" in source
    assert 'id="load-btn"' in source
    assert "fetch('/v1/models/load'" in source
    assert "window.confirm(" in source
    assert "!selectedLoaded" in source
    assert "return selectedLoaded && !configLoadError && !busy" in source
    assert "Run remains disabled" in source
    assert "substantial unified memory" in source
    assert "never unloads them" in source
    assert "fetch('/v1/models')" not in source
    assert "/v1/models/unload" not in source
    assert "/ui/api/tier" not in source
    assert "Promise.all([run('l'), run('r')])" in source
    assert "socket.onmessage" in source
    assert "@media (max-width: 720px)" in source
    assert ".innerHTML" not in source


def test_compare_inline_script_has_valid_javascript() -> None:
    completed = subprocess.run(
        [_node(), "--check", "-"],
        input=_inline_script(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_compare_requires_explicit_load_and_rechecks_server_state() -> None:
    harness = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeNode {
  constructor(tag = "div") {
    this.tag = tag;
    this.value = "";
    this.textContent = "";
    this.disabled = false;
    this.hidden = false;
    this.dataset = {};
    this.children = [];
    this.listeners = {};
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute(name, value) { this[name] = String(value); }
  replaceChildren(...children) {
    this.children = children;
    if (this.tag === "select") this.value = children[0]?.value || "";
  }
  appendChild(child) {
    this.children.push(child);
    if (this.tag === "select" && !this.value) this.value = child.value || "";
    return child;
  }
}

const ids = [
  "compare-status", "configured-tiers", "loaded-tiers", "refresh-btn", "load-btn",
  "run-btn", "prompt", "tier-l", "tier-r", "out-l", "out-r", "metric-l", "metric-r",
];
const nodes = Object.fromEntries(ids.map(id => [id, new FakeNode(id.startsWith("tier-") ? "select" : "div")]));
nodes.prompt.value = "Explain speculative decoding";
const calls = [];
const resident = new Set(["small"]);
let confirmations = 0;

function response(payload, ok = true, status = 200) {
  return {ok, status, async json() { return payload; }};
}

async function fetchStub(url, options = {}) {
  calls.push({url, options});
  if (url === "/ui/api/config") {
    return response({
      all_tiers: ["small", "large", "huge"],
      loaded_tiers: [...resident],
    });
  }
  if (url === "/v1/models/load") {
    const {tier} = JSON.parse(options.body);
    if (tier === "huge") return response({detail: "load refused"}, false, 400);
    resident.add(tier);
    return response({status: "loaded", tier});
  }
  throw new Error("unexpected request " + url);
}

global.document = {
  getElementById(id) { return nodes[id]; },
  createElement(tag) { return new FakeNode(tag); },
  createTextNode(text) { const node = new FakeNode("text"); node.textContent = text; return node; },
};
global.window = {
  Mio: {security: {openWebSocket() { throw new Error("not used"); }}},
  confirm() { confirmations += 1; return true; },
};
global.fetch = fetchStub;
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), {filename: "compare-inline.js"});

async function settle() {
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));
}

(async () => {
  await settle();
  assert.deepEqual(calls.map(call => call.url), ["/ui/api/config"]);
  assert.equal(nodes["configured-tiers"].textContent, "small, large, huge");
  assert.equal(nodes["loaded-tiers"].textContent, "small");
  assert.equal(nodes["tier-l"].value, "small");
  assert.equal(nodes["tier-r"].value, "large");
  assert.equal(nodes["run-btn"].disabled, true);
  assert.equal(nodes["load-btn"].disabled, false);

  await nodes["load-btn"].listeners.click();
  assert.equal(confirmations, 1);
  assert.deepEqual(calls.map(call => call.url), [
    "/ui/api/config", "/v1/models/load", "/ui/api/config",
  ]);
  const loadCall = calls.find(call => call.url === "/v1/models/load");
  assert.equal(loadCall.options.method, "POST");
  assert.deepEqual(JSON.parse(loadCall.options.body), {tier: "large"});
  assert.equal(nodes["loaded-tiers"].textContent, "small, large");
  assert.equal(nodes["run-btn"].disabled, false);

  nodes["tier-r"].value = "huge";
  nodes["tier-r"].listeners.change();
  assert.equal(nodes["run-btn"].disabled, true);
  assert.equal(nodes["load-btn"].disabled, false);
  await nodes["load-btn"].listeners.click();
  assert.equal(confirmations, 2);
  assert.match(nodes["compare-status"].textContent, /Could not load huge: load refused/);
  assert.match(nodes["compare-status"].textContent, /No tier was unloaded/);
  assert.equal(nodes["run-btn"].disabled, true);
  assert.equal(calls.some(call => call.url.includes("unload")), false);
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [_node(), "-e", harness, "/dev/stdin"],
        input=_inline_script(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
