"""Behavioral regression tests for the asynchronous WebUI view router."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_view_router_awaits_lifecycle_cancels_stale_navigation_and_retries_safely():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    asset = Path(__file__).parents[1] / "mio" / "webui" / "assets" / "views.js"
    harness = f"""
const fs = require('fs');

class ClassList {{
  constructor() {{ this.values = new Set(); }}
  toggle(name, enabled) {{
    if (enabled) this.values.add(name); else this.values.delete(name);
  }}
  contains(name) {{ return this.values.has(name); }}
}}

function connectTree(element, connected) {{
  element.isConnected = connected;
  for (const child of element.children) connectTree(child, connected);
}}

class Element {{
  constructor(tagName) {{
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.style = {{}};
    this.dataset = {{}};
    this.attributes = {{}};
    this.classList = new ClassList();
    this.className = '';
    this.id = '';
    this.hidden = false;
    this.disabled = false;
    this.type = '';
    this.isConnected = false;
    this.listeners = new Map();
    this._text = '';
  }}
  appendChild(child) {{
    child.remove();
    this.children.push(child);
    child.parentNode = this;
    connectTree(child, this.isConnected);
    return child;
  }}
  replaceChildren(...children) {{
    for (const child of this.children) {{
      child.parentNode = null;
      connectTree(child, false);
    }}
    this.children = [];
    this._text = '';
    for (const child of children) this.appendChild(child);
  }}
  remove() {{
    if (!this.parentNode) return;
    const siblings = this.parentNode.children;
    const index = siblings.indexOf(this);
    if (index >= 0) siblings.splice(index, 1);
    this.parentNode = null;
    connectTree(this, false);
  }}
  setAttribute(name, value) {{ this.attributes[name] = String(value); }}
  getAttribute(name) {{ return this.attributes[name] ?? null; }}
  addEventListener(name, callback, options = {{}}) {{
    const entries = this.listeners.get(name) || [];
    entries.push({{ callback, once: Boolean(options.once) }});
    this.listeners.set(name, entries);
  }}
  click() {{
    const entries = [...(this.listeners.get('click') || [])];
    this.listeners.set('click', entries.filter((entry) => !entry.once));
    for (const entry of entries) entry.callback({{ currentTarget: this }});
  }}
  set textContent(value) {{
    this._text = String(value);
    this.replaceChildren();
    this._text = String(value);
  }}
  get textContent() {{
    return this._text + this.children.map((child) => child.textContent).join('');
  }}
}}

function walk(root, predicate) {{
  if (predicate(root)) return root;
  for (const child of root.children) {{
    const match = walk(child, predicate);
    if (match) return match;
  }}
  return null;
}}

const body = new Element('body');
connectTree(body, true);
const app = new Element('div');
app.id = 'app';
app.className = 'app';
body.appendChild(app);
const sidebar = new Element('aside');
sidebar.className = 'sidebar';
app.appendChild(sidebar);
const main = new Element('main');
main.className = 'main';
app.appendChild(main);
const rails = ['chat', 'slow', 'sync', 'stale', 'fast', 'failure', 'render-only'].map((view) => {{
  const button = new Element('button');
  button.className = 'nav-rail-btn';
  button.dataset.view = view;
  return button;
}});

global.window = {{ Mio: {{}} }};
global.document = {{
  readyState: 'loading',
  body,
  createElement(tagName) {{ return new Element(tagName); }},
  getElementById(id) {{ return walk(body, (element) => element.id === id); }},
  querySelector(selector) {{
    if (selector === '.app > .sidebar') return sidebar;
    if (selector === '.app > .main') return main;
    return null;
  }},
  querySelectorAll(selector) {{ return selector === '.nav-rail-btn' ? rails : []; }},
  addEventListener() {{}},
}};
const storage = new Map();
global.localStorage = {{
  getItem(key) {{ return storage.has(key) ? storage.get(key) : null; }},
  setItem(key, value) {{ storage.set(key, String(value)); }},
}};

eval(fs.readFileSync({json.dumps(str(asset))}, 'utf8'));

const events = [];
const tick = () => new Promise((resolve) => setImmediate(resolve));
let releaseMount;
let releaseDeactivate;
let releaseCleanup;
let staleResolve;
let staleSignal;
let failureAttempts = 0;

window.Mio.views.register('slow', {{
  title: '<Unsafe title>',
  mount(host, context) {{
    events.push('slow:mount');
    host.textContent = 'slow content';
    return new Promise((resolve) => {{
      releaseMount = () => resolve(async () => {{
        events.push('slow:cleanup:start');
        await new Promise((done) => {{ releaseCleanup = done; }});
        events.push('slow:cleanup:end');
      }});
    }});
  }},
  async activate() {{ events.push('slow:activate'); }},
  async deactivate() {{
    events.push('slow:deactivate:start');
    await new Promise((resolve) => {{ releaseDeactivate = resolve; }});
    events.push('slow:deactivate:end');
  }},
  async cleanup() {{ events.push('slow:explicit-cleanup'); }},
}});
window.Mio.views.register('sync', {{
  mount() {{ events.push('sync:mount'); return () => events.push('sync:cleanup'); }},
  activate() {{ events.push('sync:activate'); }},
}});
window.Mio.views.register('stale', {{
  mount(_host, context) {{
    staleSignal = context.signal;
    events.push('stale:mount');
    return new Promise((resolve) => {{ staleResolve = resolve; }});
  }},
  cleanup() {{ events.push('stale:explicit-cleanup'); }},
}});
window.Mio.views.register('fast', {{
  mount() {{ events.push('fast:mount'); }},
  activate() {{ events.push('fast:activate'); }},
}});
window.Mio.views.register('failure', {{
  mount() {{
    failureAttempts += 1;
    events.push('failure:mount:' + failureAttempts);
    if (failureAttempts === 1) throw new Error('<img src=x onerror=attack()>');
  }},
  async deactivate() {{ events.push('failure:deactivate-without-activate'); }},
}});
window.Mio.views.register('render-only', {{
  async render() {{ await tick(); events.push('render-only:render'); }},
}});

(async () => {{
  await window.Mio.views._boot();

  const slowNavigation = window.Mio.views.switch('slow');
  await tick();
  const stage = document.getElementById('view-stage');
  const loadingText = stage.children[0].textContent;
  const loadingState = stage.children[0].dataset.state;
  const activeWhileLoading = window.Mio.views.getActive();
  releaseMount();
  const slowResult = await slowNavigation;

  const syncNavigation = window.Mio.views.switch('sync');
  await tick();
  const syncMountedBeforeDeactivate = events.includes('sync:mount');
  releaseDeactivate();
  await tick();
  const syncMountedBeforeCleanup = events.includes('sync:mount');
  releaseCleanup();
  const syncResult = await syncNavigation;

  const staleNavigation = window.Mio.views.switch('stale');
  await tick();
  const fastNavigation = window.Mio.views.switch('fast');
  const fastResult = await fastNavigation;
  const staleResult = await staleNavigation;
  const staleWasAborted = staleSignal.aborted;
  staleResolve(async () => {{ events.push('stale:late-cleanup'); }});
  await tick();
  await tick();

  const firstFailure = await window.Mio.views.switch('failure');
  const errorPanel = stage.children[0];
  const errorDetail = errorPanel.children[1];
  const retryButton = errorPanel.children[2];
  const errorSnapshot = {{
    state: errorPanel.dataset.state,
    role: errorPanel.getAttribute('role'),
    text: errorDetail.textContent,
    childCount: errorDetail.children.length,
  }};
  retryButton.click();
  await tick();
  const retryResult = await window.Mio.views.switch('failure');

  const renderResult = await window.Mio.views.switch('render-only');
  console.log(JSON.stringify({{
    loadingText,
    loadingState,
    activeWhileLoading,
    slowResult,
    syncResult,
    syncMountedBeforeDeactivate,
    syncMountedBeforeCleanup,
    staleResult,
    staleWasAborted,
    fastResult,
    firstFailure,
    retryResult,
    renderResult,
    errorSnapshot,
    failureAttempts,
    active: window.Mio.views.getActive(),
    stored: storage.get('mio.activeView'),
    events,
  }}));
}})().catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    completed = subprocess.run(
        [node, "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    result = json.loads(completed.stdout)

    assert result["loadingState"] == "loading"
    assert result["loadingText"] == "Loading <Unsafe title>…"
    assert result["activeWhileLoading"] is None
    assert result["slowResult"] is True
    assert result["syncResult"] is True
    assert result["syncMountedBeforeDeactivate"] is False
    assert result["syncMountedBeforeCleanup"] is False
    assert result["events"].index("slow:deactivate:end") < result["events"].index(
        "slow:cleanup:start"
    )
    assert result["events"].index("slow:cleanup:end") < result["events"].index(
        "slow:explicit-cleanup"
    )
    assert result["events"].index("slow:explicit-cleanup") < result["events"].index(
        "sync:mount"
    )

    assert result["staleResult"] is False
    assert result["staleWasAborted"] is True
    assert result["fastResult"] is True
    assert "stale:late-cleanup" in result["events"]
    assert result["events"].index("fast:activate") < result["events"].index(
        "stale:late-cleanup"
    )

    assert result["firstFailure"] is False
    assert result["retryResult"] is True
    assert result["failureAttempts"] == 2
    assert result["errorSnapshot"] == {
        "state": "error",
        "role": "alert",
        "text": "<img src=x onerror=attack()>",
        "childCount": 0,
    }
    assert result["renderResult"] is True
    assert "render-only:render" in result["events"]
    assert result["events"].index(
        "failure:deactivate-without-activate"
    ) < result["events"].index("render-only:render")
    assert result["active"] == "render-only"
    assert result["stored"] == "render-only"
