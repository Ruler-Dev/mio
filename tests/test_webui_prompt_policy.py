"""Focused contracts for the WebUI PromptPolicy selector and chat payloads."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SHELL = ROOT / "mio" / "webui" / "mio_ui.html"


def _node() -> str:
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node.js is required for WebUI policy tests")
    return executable


def test_settings_and_every_chat_path_use_modern_prompt_policy_fields():
    source = SHELL.read_text(encoding="utf-8")

    assert 'id="settPromptPolicy"' in source
    for value in (
        "none",
        "caveman/lite",
        "caveman/full",
        "caveman/ultra",
        "ponytail/lite",
        "ponytail/full",
        "ponytail/ultra",
    ):
        assert f'value="{value}"' in source
    assert 'aria-describedby="settPromptPolicyHint"' in source
    assert "currentConfig.caveman" not in source
    assert "settCaveman" not in source
    assert "cavemanLabel" not in source

    chat_sends = source.count("action: 'chat'")
    assert chat_sends == 3
    assert source.count("...promptPolicyRequestFields()") == chat_sends
    assert "if (config.prompt_mode !== 'none') fields.prompt_level" in source
    assert "if (candidate.prompt_mode === 'none') delete body.prompt_level" in source
    assert "delete body.caveman" in source


def test_policy_config_normalization_labels_and_none_payload_are_behavioral():
    source = SHELL.read_text(encoding="utf-8")
    start = source.index("let currentConfig = {")
    end = source.index("async function loadConfig()", start)
    policy_source = source[start:end]
    harness = f"""
const assert = require('node:assert/strict');
const labels = {{}};
global.document = {{
  getElementById(id) {{
    if (!labels[id]) labels[id] = {{ textContent: '' }};
    return labels[id];
  }},
}};
{policy_source}

assert.deepEqual(
  normalizePromptPolicyConfig({{prompt_mode: 'ponytail', prompt_level: 'ultra', caveman: 'lite'}}),
  {{prompt_mode: 'ponytail', prompt_level: 'ultra'}},
);
assert.deepEqual(
  normalizePromptPolicyConfig({{prompt_policy: 'ponytail/lite'}}),
  {{prompt_mode: 'ponytail', prompt_level: 'lite'}},
);
assert.deepEqual(
  normalizePromptPolicyConfig({{caveman: 'off'}}),
  {{prompt_mode: 'none', prompt_level: null}},
);
assert.deepEqual(
  normalizePromptPolicyConfig({{caveman: 'full'}}),
  {{prompt_mode: 'caveman', prompt_level: 'full'}},
);
assert.deepEqual(
  promptPolicyRequestFields({{prompt_mode: 'none', prompt_level: 'ultra'}}),
  {{prompt_mode: 'none'}},
);
assert.deepEqual(
  promptPolicyRequestFields({{prompt_mode: 'ponytail', prompt_level: 'lite'}}),
  {{prompt_mode: 'ponytail', prompt_level: 'lite'}},
);
applyPromptPolicyConfig({{prompt_policy: 'ponytail/ultra'}});
assert.equal(promptPolicySettingValue(), 'ponytail/ultra');
assert.equal(labels.promptPolicyLabel.textContent, 'Prompt: Ponytail Ultra');
"""
    completed = subprocess.run(
        [_node(), "-e", harness],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
