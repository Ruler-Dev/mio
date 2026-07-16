from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "mio" / "webui" / "assets"


def _node() -> str:
    executable = shutil.which("node")
    if not executable:
        pytest.skip("node is required for WebUI JavaScript tests")
    return executable


def test_shared_skill_client_enforces_transport_contract() -> None:
    script = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function response(payload, {ok = true, status = 200, invalidJson = false} = {}) {
  return {
    ok,
    status,
    async json() {
      if (invalidJson) throw new SyntaxError("bad json");
      return payload;
    },
  };
}

async function expectFailure(action, pattern, expectedStatus) {
  let failure;
  try { await action(); } catch (error) { failure = error; }
  assert.ok(failure, "expected request to fail");
  assert.match(failure.message, pattern);
  if (expectedStatus !== undefined) assert.equal(failure.status, expectedStatus);
  return failure;
}

(async () => {
  const calls = [];
  global.window = {
    Mio: {
      security: {
        async runSkill(...args) {
          calls.push(args);
          return response({ok: true, result: {results: ["one"]}});
        },
      },
    },
  };
  vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), {filename: process.argv[1]});

  const result = await window.Mio.api.runSkill("web_search", {query: "MLX"});
  assert.deepEqual(result, {results: ["one"]});
  assert.deepEqual(calls, [["web_search", {query: "MLX"}, {}]]);

  window.Mio.security.runSkill = async () => response(
    {detail: {error: "sensitive_skill_denied", reason: "confirmation required"}},
    {ok: false, status: 403},
  );
  const denied = await expectFailure(
    () => window.Mio.api.runSkill("blender_exec", {code: "pass"}),
    /HTTP 403: sensitive_skill_denied: confirmation required/,
    403,
  );
  assert.equal(denied.payload.detail.error, "sensitive_skill_denied");

  window.Mio.security.runSkill = async () => response(
    {ok: false, error: "RuntimeError: boom"},
  );
  await expectFailure(
    () => window.Mio.api.runSkill("broken", {}),
    /RuntimeError: boom/,
    200,
  );

  window.Mio.security.runSkill = async () => response(
    {error: "unknown skill: missing"},
  );
  await expectFailure(
    () => window.Mio.api.runSkill("missing", {}),
    /unknown skill: missing/,
  );

  window.Mio.security.runSkill = async () => response({}, {invalidJson: true});
  await expectFailure(
    () => window.Mio.api.runSkill("bad_json", {}),
    /invalid JSON/,
  );

  window.Mio.security.runSkill = async () => response({ok: true});
  await expectFailure(
    () => window.Mio.api.runSkill("missing_result", {}),
    /missing result/,
  );

  window.Mio.security.runSkill = async () => { throw new TypeError("offline"); };
  await expectFailure(
    () => window.Mio.api.runSkill("offline", {}),
    /request failed: offline/,
  );

  await expectFailure(
    () => window.Mio.api.runSkill("", {}),
    /non-empty string/,
  );
  await expectFailure(
    () => window.Mio.api.runSkill("bad_args", []),
    /arguments must be an object/,
  );

  delete window.Mio.security;
  let directRequest;
  window.fetch = async (url, options) => {
    directRequest = {url, options};
    return response({ok: true, result: null});
  };
  const directResult = await window.Mio.api.runSkill(
    "blender_exec",
    {code: "print('ok')"},
    {confirmSensitive: true},
  );
  assert.equal(directResult, null);
  assert.equal(directRequest.url, "/ui/api/skills/run");
  assert.equal(directRequest.options.method, "POST");
  assert.equal(directRequest.options.credentials, "same-origin");
  assert.equal(directRequest.options.headers.get("X-Mio-Dangerous-Action"), "blender_exec");
  assert.deepEqual(JSON.parse(directRequest.options.body), {
    name: "blender_exec",
    args: {code: "print('ok')"},
    confirm_sensitive: true,
  });
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [_node(), "-e", script, str(ASSETS / "api_client.js")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_external_skill_consumers_use_shared_client() -> None:
    shell = (ROOT / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    design = (ASSETS / "view_design.js").read_text(encoding="utf-8")
    notebook = (ASSETS / "view_notebook.js").read_text(encoding="utf-8")

    assert '<script src="/ui/assets/api_client.js"></script>' in shell
    assert shell.index("security.js") < shell.index("api_client.js") < shell.index("artifact_registry.js")
    assert "window.Mio.api.runSkill('import_shadertoy'" in shell
    assert 'window.Mio.api.runSkill("web_search"' in design
    assert 'window.Mio.api.runSkill("search_images"' in design
    assert "{ query: q, max_results: 5 }" in design
    assert "{ query: q, count: 6 }" in design
    assert "window.Mio.api.runSkill(body.skill, body.args || {})" in notebook
    assert "__mioDesignSkillRun" in design
    assert "__mioDesignSkillResult" in design
    assert 'new Set(["blender_exec", "blender_snapshot"])' in design
    assert "confirmSensitive: true" in design

    allowed = {"api_client.js", "security.js"}
    direct_consumers = []
    for path in ASSETS.glob("*.js"):
        if path.name in allowed:
            continue
        if "/ui/api/skills/run" in path.read_text(encoding="utf-8"):
            direct_consumers.append(path.name)
    assert direct_consumers == []


def test_blender_viewer_bridge_script_is_valid_javascript() -> None:
    source = (ASSETS / "view_design.js").read_text(encoding="utf-8")
    start = source.index("<script>\nconst code = ${json};", source.index("function buildBlenderViewer"))
    end = source.index("\n</script>", start)
    embedded = source[start + len("<script>\n") : end].replace("${json}", '"import bpy"')
    completed = subprocess.run(
        [_node(), "--check", "-"],
        input=embedded,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_standalone_skill_pages_use_shared_client_and_have_valid_scripts() -> None:
    for name in ("playground.html", "workspace.html"):
        source = (ASSETS / name).read_text(encoding="utf-8")
        assert source.index('src="/ui/assets/security.js"') < source.index(
            'src="/ui/assets/api_client.js"'
        )
        assert "/ui/api/skills/run" not in source
        inline_scripts = [
            match
            for match in re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", source)
            if match.strip()
        ]
        assert inline_scripts
        for script in inline_scripts:
            completed = subprocess.run(
                [_node(), "--check", "-"],
                input=script,
                capture_output=True,
                check=False,
                text=True,
            )
            assert completed.returncode == 0, f"{name}: {completed.stderr}"

    playground = (ASSETS / "playground.html").read_text(encoding="utf-8")
    assert "window.Mio.api.runSkill(active, args, { confirmSensitive })" in playground
    assert "showResult(result, Boolean(result?.error || result?.ok === false))" in playground
    assert "finally" in playground

    workspace = (ASSETS / "workspace.html").read_text(encoding="utf-8")
    for skill in ("list_indexes", "index_folder", "drop_index"):
        assert re.search(rf"window\.Mio\.api\.runSkill\(\s*'{skill}'", workspace)
    assert workspace.count("{ confirmSensitive: true }") == 3
    assert "Loading indexed folders…" in workspace
    assert "renderIndexPanel(c, INDEXES, error.message, true)" in workspace
