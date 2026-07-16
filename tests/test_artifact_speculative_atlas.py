from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "mio" / "webui" / "assets" / "artifact_registry.js"
LAB = ROOT / "mio" / "webui" / "assets" / "artifact_lab.js"
CSS = ROOT / "mio" / "webui" / "assets" / "main.css"


def test_speculative_acceptance_atlas_contract_and_safe_renderer() -> None:
    source = LAB.read_text(encoding="utf-8")
    assert "application/vnd.pimio.speculative-acceptance-atlas+json" in source
    assert "pimio.speculative-acceptance-atlas" in source
    assert ".innerHTML" not in source
    assert "https://" not in source

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")

    smoke = r"""
const assert = require('node:assert/strict');

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.className = '';
    this.textContent = '';
    this.classList = {
      add: (...values) => { this.className += ` ${values.join(' ')}`; },
    };
  }
  append(...values) { this.children.push(...values); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
}

function descendants(node) {
  return [node, ...node.children.flatMap((child) => child instanceof Element ? descendants(child) : [])];
}

global.window = {};
global.document = { createElement: (tag) => new Element(tag) };
require(process.argv[1]);
require(process.argv[2]);

const canonical = 'application/vnd.pimio.speculative-acceptance-atlas+json';
const alias = 'application/vnd.pimio.speculative-atlas+json';
const legacyAlias = 'application/vnd.pimio.acceptance-atlas+json';
const registry = window.Mio.artifactTypes;
const lab = window.Mio.artifactLab;

assert.equal(registry.normalize(alias), canonical);
assert.equal(registry.normalize(legacyAlias), canonical);
assert.equal(registry.supports(alias), true);
assert.equal(lab.supports(legacyAlias), true);
assert.equal(lab.label(alias), 'Speculative acceptance atlas');
assert.equal(registry.definition(alias).type, canonical);

const schema = lab.schema(alias);
assert.deepEqual(
  {id: schema.id, version: schema.version, mime: schema.mime},
  {id: 'pimio.speculative-acceptance-atlas', version: 1, mime: canonical},
);
assert.deepEqual(schema.aliases, [alias, legacyAlias]);

const sample = JSON.parse(lab.sample(canonical));
assert.equal(sample.schema, schema.id);
assert.equal(sample.version, schema.version);
assert.ok(sample.positions.length >= 6);
assert.ok(sample.phases.length >= 3);
assert.equal(sample.reliability.speedup_ci.length, 2);

sample.title = '<img src=x onerror=global.pwned=true> Atlas';
sample.decision.rationale = '<script>global.pwned=true</script> measured rationale';
const body = new Element('main');
assert.equal(registry.render(body, {type: alias, content: JSON.stringify(sample)}), true);
assert.equal(body.children.length, 1);
const rendered = descendants(body.children[0]);
assert.equal(body.children[0].dataset.artifactType, canonical);
assert.equal(body.children[0].className.includes('has-error'), false);
assert.ok(rendered.some((node) => node.className.includes('mio-lab-atlas-positions')));
assert.ok(rendered.some((node) => node.className.includes('mio-lab-atlas-phases')));
assert.ok(rendered.some((node) => node.className.includes('mio-lab-atlas-reliability')));
assert.ok(rendered.some((node) => node.textContent.includes('<img src=x onerror=')));
assert.ok(rendered.some((node) => node.textContent.includes('<script>global.pwned')));
assert.equal(rendered.some((node) => node.tag === 'img' || node.tag === 'script'), false);
assert.equal(global.pwned, undefined);

const positionBars = rendered.filter((node) => node.className.includes('mio-lab-atlas-position-bar'));
assert.equal(positionBars.length, sample.positions.length);
assert.ok(positionBars.every((node) => /^\d+(?:\.\d+)?%$/.test(node.style.height)));

const directAliasBody = new Element('main');
assert.equal(lab.render(directAliasBody, {type: legacyAlias, content: JSON.stringify(sample)}), true);
assert.equal(directAliasBody.children[0].dataset.artifactType, canonical);

const download = registry.download({type: alias, content: JSON.stringify(sample)});
assert.deepEqual(
  {extension: download.extension, mime: download.mime},
  {extension: '.json', mime: 'application/json'},
);
assert.deepEqual(JSON.parse(download.content), sample);

const invalid = {...sample, version: 2};
const invalidBody = new Element('main');
assert.equal(registry.render(invalidBody, {type: canonical, content: JSON.stringify(invalid)}), true);
assert.equal(invalidBody.children[0].className.includes('has-error'), true);
assert.ok(descendants(invalidBody).some((node) => node.textContent.includes('version 1')));
"""
    subprocess.run(
        [node, "-e", smoke, str(REGISTRY), str(LAB)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_speculative_acceptance_atlas_is_theme_aware_and_responsive() -> None:
    css = CSS.read_text(encoding="utf-8")
    assert ".mio-lab-atlas-decision" in css
    assert ".mio-lab-atlas-positions" in css
    assert ".mio-lab-atlas-phases" in css
    assert ".mio-lab-atlas-reliability-grid" in css
    assert "overflow-x: auto" in css
    mobile = css.split("@media (max-width: 760px)", 1)[1].split(
        "/* ===== Command palette", 1
    )[0]
    assert ".mio-lab-atlas-decision { grid-template-columns: 1fr;" in mobile
    assert ".mio-lab-atlas-reliability-grid { grid-template-columns: repeat(2" in mobile
