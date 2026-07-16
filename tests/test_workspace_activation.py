"""Focused contract tests for truthful, atomic workspace activation."""

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from mio.prompt_policy import PromptMode, PromptPolicy
from mio.webui import router


@pytest.mark.asyncio
async def test_workspace_activation_commits_supported_runtime_profile(monkeypatch):
    lock = threading.Lock()
    events: list[str] = []

    class Manager:
        config = SimpleNamespace(
            tiers={
                "old": SimpleNamespace(context_window=8192),
                "research": SimpleNamespace(context_window=131072),
            }
        )

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

    project = {
        "id": "research",
        "name": "Research",
        "system_prompt": "Cite retrieved sources.",
        "files": ["paper.pdf"],
        "tier": "research",
        "context_window": 65536,
        "caveman_level": "lite",
        "pinned_prompts": ["legacy prompt"],
    }
    manager = Manager()
    monkeypatch.setattr(router, "_load_projects", lambda: [project])
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)
    monkeypatch.setattr(router, "_prompt_policy", PromptPolicy.resolve(caveman="full"))
    monkeypatch.setattr(router, "_caveman_level", "full")

    result = await router.activate_project("research")

    assert manager.loaded == ["research"]
    assert events == ["snapshot", "load:research", "unload:old"]
    assert router._prompt_policy.label == "caveman/lite"
    assert router._caveman_level == "lite"
    assert result["runtime"] == {
        "tier": "research",
        "tier_changed": True,
        "tier_pinned": True,
        "prompt_policy": "caveman/lite",
        "prompt_policy_pinned": True,
        "prompt_policy_source": "workspace",
        "prompt_policy_changed": True,
        "context_window": 131072,
    }
    assert result["project_context"] == {"system_prompt": True, "files": 1}
    assert result["context_requirement"] == {
        "requested": 65536,
        "available": 131072,
        "satisfied": True,
    }
    assert result["warnings"][0]["field"] == "pinned_prompts"
    assert "were not applied" in result["warnings"][0]["message"]


@pytest.mark.asyncio
async def test_workspace_activation_rejects_context_before_any_mutation(monkeypatch):
    lock = threading.Lock()

    class Manager:
        config = SimpleNamespace(
            tiers={
                "old": SimpleNamespace(context_window=8192),
                "too-small": SimpleNamespace(context_window=16384),
            }
        )

        def __init__(self):
            self.loaded = ["old"]
            self.load_calls: list[str] = []
            self.unload_calls: list[str] = []

        def loaded_tiers(self):
            assert lock.locked()
            return list(self.loaded)

        def load_tier(self, tier):
            self.load_calls.append(tier)

        def unload_tier(self, tier):
            self.unload_calls.append(tier)

    project = {
        "id": "coding",
        "name": "Coding",
        "system_prompt": "Test every change.",
        "files": [],
        "tier": "too-small",
        "context_window": 32768,
        "caveman_level": "lite",
    }
    manager = Manager()
    original_policy = PromptPolicy.resolve(caveman="full")
    monkeypatch.setattr(router, "_load_projects", lambda: [project])
    monkeypatch.setattr(router, "_manager", manager)
    monkeypatch.setattr(router, "_gpu_lock", lock)
    monkeypatch.setattr(router, "_prompt_policy", original_policy)
    monkeypatch.setattr(router, "_caveman_level", "full")

    with pytest.raises(HTTPException) as raised:
        await router.activate_project("coding")

    assert raised.value.status_code == 409
    assert "requires 32768 context tokens" in raised.value.detail
    assert manager.loaded == ["old"]
    assert manager.load_calls == []
    assert manager.unload_calls == []
    assert router._prompt_policy == original_policy
    assert router._caveman_level == "full"


@pytest.mark.asyncio
async def test_workspace_activation_supports_modern_ponytail_and_explicit_none(
    monkeypatch,
):
    projects = [
        {
            "id": "ponytail",
            "name": "Small coherent diffs",
            "system_prompt": "Test the affected flow.",
            "files": [],
            "prompt_mode": "ponytail",
            "prompt_level": "ultra",
        },
        {
            "id": "plain",
            "name": "No policy",
            "system_prompt": "",
            "files": [],
            "prompt_mode": "none",
            "prompt_level": None,
        },
    ]
    monkeypatch.setattr(router, "_load_projects", lambda: projects)
    monkeypatch.setattr(router, "_manager", None)
    monkeypatch.setattr(router, "_gpu_lock", threading.Lock())
    monkeypatch.setattr(router, "_prompt_policy", PromptPolicy.resolve(caveman="lite"))
    monkeypatch.setattr(router, "_caveman_level", "lite")

    activated = await router.activate_project("ponytail")
    assert router._prompt_policy.label == "ponytail/ultra"
    assert router._caveman_level == "off"
    assert activated["runtime"]["prompt_policy"] == "ponytail/ultra"
    assert activated["runtime"]["prompt_policy_pinned"] is True
    assert activated["runtime"]["prompt_policy_source"] == "workspace"

    activated = await router.activate_project("plain")
    assert router._prompt_policy.mode is PromptMode.NONE
    assert activated["runtime"]["prompt_policy"] == "none"
    assert activated["runtime"]["prompt_policy_pinned"] is True


@pytest.mark.asyncio
async def test_workspace_without_policy_honestly_inherits_current_runtime(monkeypatch):
    project = {
        "id": "inherit",
        "name": "Inherited policy",
        "system_prompt": "",
        "files": [],
    }
    original = PromptPolicy.resolve(ponytail="full")
    monkeypatch.setattr(router, "_load_projects", lambda: [project])
    monkeypatch.setattr(router, "_manager", None)
    monkeypatch.setattr(router, "_gpu_lock", threading.Lock())
    monkeypatch.setattr(router, "_prompt_policy", original)
    monkeypatch.setattr(router, "_caveman_level", "off")

    activated = await router.activate_project("inherit")

    assert router._prompt_policy is original
    assert activated["runtime"]["prompt_policy"] == "ponytail/full"
    assert activated["runtime"]["prompt_policy_pinned"] is False
    assert activated["runtime"]["prompt_policy_source"] == "current runtime"


@pytest.mark.asyncio
async def test_workspace_save_persists_modern_fields_and_upgrades_legacy(monkeypatch):
    saved: list[dict] = []

    def update(transform):
        saved[:] = transform(saved)

    monkeypatch.setattr(router, "_update_projects", update)

    ponytail = await router.save_project(
        {"id": "modern", "prompt_mode": "ponytail", "prompt_level": "lite"}
    )
    assert ponytail["prompt_mode"] == "ponytail"
    assert ponytail["prompt_level"] == "lite"
    assert "caveman_level" not in ponytail

    upgraded = await router.save_project({"id": "legacy", "caveman_level": "ultra"})
    assert upgraded["prompt_mode"] == "caveman"
    assert upgraded["prompt_level"] == "ultra"
    assert "caveman_level" not in upgraded

    disabled = await router.save_project(
        {"id": "disabled", "prompt_mode": "none", "prompt_level": None}
    )
    assert disabled["prompt_mode"] == "none"
    assert disabled["prompt_level"] is None

    with pytest.raises(HTTPException) as raised:
        await router.save_project({"id": "invalid", "prompt_level": "lite"})
    assert raised.value.status_code == 400
    assert "requires a prompt mode" in raised.value.detail


def test_workspace_ui_waits_for_activation_and_has_no_fake_pinned_contract():
    source = (
        Path(__file__).parents[1]
        / "mio"
        / "webui"
        / "assets"
        / "view_workspaces.js"
    ).read_text(encoding="utf-8")
    activation = source.split("async function openInChat", 1)[1].split(
        "function openEditor", 1
    )[0]

    endpoint = "${encodeURIComponent(p.id)}/activate"
    assert endpoint in activation
    assert activation.index(endpoint) < activation.index("await window.setActiveProject(p.id)")
    assert "if (!res.ok || !activation.ok)" in activation
    assert "window.switchTier" not in activation
    assert "Workspace not activated" in activation
    assert "pinned_prompts:" not in source
    assert "Minimum context" in source
    assert "this does not resize a loaded model" in source
    assert 'id="ws-f-policy"' in source
    assert 'value="ponytail/ultra"' in source
    assert "prompt_mode:" in source
    assert "prompt_level:" in source
    assert "caveman_level:" not in source
    assert "workspace?.caveman_level" in source
    assert "prompt_policy_pinned" in source
