"""Focused browser-contract tests for local artifact PNG export."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
ASSET = ROOT / "mio" / "webui" / "assets" / "artifact_export.js"


def _source() -> str:
    return ASSET.read_text(encoding="utf-8")


def test_export_has_no_remote_renderer_or_sandbox_document_access() -> None:
    source = _source()
    executable = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//"))

    assert "html2canvas" not in source.lower()
    assert "cdn.jsdelivr" not in source
    assert ".contentDocument" not in executable
    assert ".contentWindow" not in executable
    assert "body.querySelector('iframe')" in source
    assert "'sandboxed-frame'" in source
    assert "PNG snapshot is unavailable for this sandboxed artifact" in source
    assert "Use Download as file" in source
    assert "system screenshot tool" in source


def test_export_uses_shared_artifact_state_and_bounded_local_pipeline() -> None:
    source = _source()

    assert "NS.store" in source
    assert "store.activeArtifactId ?? window.activeArtifactId" in source
    assert "store.allArtifacts ?? window.allArtifacts" in source
    assert "Object.prototype.hasOwnProperty.call(artifacts, id)" in source
    assert "new XMLSerializer().serializeToString(clone)" in source
    assert "<foreignObject" in source
    assert "canvas.toBlob" in source
    assert "maxWidth: 2048" in source
    assert "maxHeight: 4096" in source
    assert "maxPixels: 8 * 1024 * 1024" in source
    assert "maxNodes: 1800" in source
    assert "maxEmbeddedImageBytes: 4 * 1024 * 1024" in source
    assert source.count("URL.revokeObjectURL") >= 3


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_artifact_export_javascript_has_valid_syntax() -> None:
    result = subprocess.run(
        ["node", "--check", str(ASSET)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_sandboxed_artifact_returns_honest_actionable_status() -> None:
    harness = r"""
const notices = [];
global.window = {
  Mio: { store: {
    activeArtifactId: 'demo',
    allArtifacts: { demo: { id: 'demo', title: 'Sandbox demo', type: 'text/html' } },
  } },
  toast: message => notices.push(message),
};
global.document = {
  getElementById: id => id === 'artifactBody' ? {
    querySelector: selector => selector === 'iframe' ? { sandbox: 'allow-scripts' } : null,
  } : null,
};

require(process.argv[1]);

(async () => {
  const result = await window.Mio.artifactExport.screenshot();
  if (result.ok || result.code !== 'sandboxed-frame') throw new Error(JSON.stringify(result));
  if (!result.alternative.includes('Download as file')) throw new Error(JSON.stringify(result));
  if (!notices[0] || !notices[0].includes('sandboxed artifact')) throw new Error(JSON.stringify(notices));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness, str(ASSET)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_oversized_parent_dom_is_rejected_before_rendering() -> None:
    harness = r"""
global.window = {
  Mio: { store: {
    activeArtifactId: 'large',
    allArtifacts: { large: { id: 'large', title: 'Large artifact', type: 'text/markdown' } },
  } },
  toast() {},
};
const target = {
  clientWidth: 2049,
  scrollWidth: 2049,
  clientHeight: 100,
  scrollHeight: 100,
  cloneNode() { throw new Error('clone must not run'); },
  getBoundingClientRect() { return { width: 2049, height: 100 }; },
};
global.document = {
  getElementById: id => id === 'artifactBody' ? {
    firstElementChild: target,
    querySelector: () => null,
  } : null,
};

require(process.argv[1]);

(async () => {
  const result = await window.Mio.artifactExport.screenshot();
  if (result.ok || result.code !== 'dimensions-exceeded') throw new Error(JSON.stringify(result));
  if (!result.message.includes('2049×100')) throw new Error(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness, str(ASSET)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_parent_dom_export_downloads_png_and_revokes_blob_urls() -> None:
    harness = r"""
const createdUrls = [];
const revokedUrls = [];
const notices = [];
let clicked = false;

class FakeStyle {
  constructor() { this.values = {}; this.cssText = ''; }
  setProperty(name, value) { this.values[name] = value; }
}

class FakeElement {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.nodeType = 1;
    this.parentElement = null;
    this.children = [];
    this.style = new FakeStyle();
    this.clientWidth = 320;
    this.scrollWidth = 320;
    this.clientHeight = 180;
    this.scrollHeight = 180;
    this.value = '';
  }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  cloneNode(deep) {
    const clone = new FakeElement(this.tagName);
    clone.clientWidth = this.clientWidth;
    clone.scrollWidth = this.scrollWidth;
    clone.clientHeight = this.clientHeight;
    clone.scrollHeight = this.scrollHeight;
    if (deep) this.children.forEach(child => clone.appendChild(child.cloneNode(true)));
    return clone;
  }
  getBoundingClientRect() { return { width: this.clientWidth, height: this.clientHeight }; }
  querySelector(selector) {
    if (selector === 'iframe') return null;
    return null;
  }
  querySelectorAll(selector) {
    if (selector !== '*') return [];
    const descendants = [];
    const visit = node => node.children.forEach(child => { descendants.push(child); visit(child); });
    visit(this);
    return descendants;
  }
  setAttribute(name, value) { this[name] = String(value); }
  removeAttribute(name) { delete this[name]; }
  replaceWith() {}
  remove() {}
  click() { clicked = true; }
}

const target = new FakeElement('article');
const surface = new FakeElement('div');
surface.firstElementChild = target;
surface.querySelector = selector => selector === 'iframe' ? null : null;

const pageBody = new FakeElement('body');
const context = {
  setTransform() {},
  fillRect() {},
  drawImage() {},
  fillStyle: '',
};

global.window = {
  devicePixelRatio: 2,
  Mio: { store: {
    activeArtifactId: 'benchmark',
    allArtifacts: { benchmark: { id: 'benchmark', title: 'MLX benchmark', type: 'application/vnd.pimio.benchmark+json' } },
  } },
  toast: message => notices.push(message),
};
global.location = { href: 'http://127.0.0.1/ui', origin: 'http://127.0.0.1' };
global.document = {
  body: pageBody,
  getElementById: id => id === 'artifactBody' ? surface : null,
  createElement: tag => {
    if (tag === 'canvas') {
      const canvas = new FakeElement('canvas');
      canvas.getContext = () => context;
      canvas.toBlob = callback => callback(new Blob(['png'], { type: 'image/png' }));
      return canvas;
    }
    return new FakeElement(tag);
  },
};
global.getComputedStyle = () => ({
  backgroundColor: 'rgb(255, 255, 255)',
  getPropertyValue: property => property === 'background-color' ? 'rgb(255, 255, 255)' : '',
  getPropertyPriority: () => '',
});
global.XMLSerializer = class { serializeToString() { return '<article xmlns="http://www.w3.org/1999/xhtml">benchmark</article>'; } };
global.Image = class {
  set src(value) { this.currentSrc = value; queueMicrotask(() => this.onload()); }
};

URL.createObjectURL = () => {
  const value = 'blob:local-' + (createdUrls.length + 1);
  createdUrls.push(value);
  return value;
};
URL.revokeObjectURL = value => revokedUrls.push(value);
const nativeSetTimeout = global.setTimeout;
global.setTimeout = (callback, delay) => delay === 2000 ? (callback(), 1) : nativeSetTimeout(callback, delay);

require(process.argv[1]);

(async () => {
  const result = await window.Mio.artifactExport.screenshot();
  if (!result.ok || result.code !== 'saved') throw new Error(JSON.stringify(result));
  if (result.width !== 640 || result.height !== 360) throw new Error(JSON.stringify(result));
  if (!clicked) throw new Error('download link was not clicked');
  if (createdUrls.length !== 2) throw new Error(JSON.stringify(createdUrls));
  if (revokedUrls.join(',') !== createdUrls.join(',')) {
    throw new Error(JSON.stringify({ createdUrls, revokedUrls }));
  }
  if (!notices[0] || !notices[0].includes('MLX-benchmark')) throw new Error(JSON.stringify(notices));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness, str(ASSET)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
