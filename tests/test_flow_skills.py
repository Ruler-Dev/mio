from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from mio.webui import flow_skills
from mio.webui import router as webui
from mio.webui import skills


def _flow_document() -> dict:
    return {
        "id": "research",
        "name": "Research helper",
        "nodes": {
            "ask": {
                "class": "user_input",
                "data": {"key": "topic", "label": "Topic"},
                "inputs": {},
            },
            "format": {
                "class": "template",
                "data": {"template": "Research {{input}}"},
                "inputs": {"input_1": {"connections": [{"node": "ask", "input": "output_1"}]}},
            },
            "out": {
                "class": "output",
                "data": {},
                "inputs": {"input_1": {"connections": [{"node": "format", "input": "output_1"}]}},
            },
        },
        "edges": [],
    }


def _write_flow(root: Path, document: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "research.json"
    path.write_text(json.dumps(document or _flow_document()), encoding="utf-8")
    return path


def test_published_flow_is_discovered_and_executed_through_bounded_tools(tmp_path):
    _write_flow(tmp_path)
    published = flow_skills.publish_flow(
        "research",
        name="research_topic",
        description="Format a research topic",
        root=tmp_path,
    )
    assert published["skill"]["exposed"] is True
    assert len(published["skill"]["graph_sha256"]) == 64

    catalog = flow_skills.list_flow_skills(root=tmp_path)
    assert catalog["count"] == 1
    assert catalog["flows"][0]["name"] == "research_topic"
    assert catalog["flows"][0]["inputs"] == [
        {"key": "topic", "label": "Topic", "required": True}
    ]

    result = flow_skills.run_flow_skill("research_topic", input="MLX", root=tmp_path)
    assert result["ok"] is True
    assert result["outputs"] == ["Research MLX"]

    flow_skills.unpublish_flow("research", root=tmp_path)
    assert flow_skills.list_flow_skills(root=tmp_path)["count"] == 0
    with pytest.raises(flow_skills.FlowSkillError, match="unpublished"):
        flow_skills.run_flow_skill("research_topic", root=tmp_path)


def test_flow_skill_names_are_unique_and_recursive_dispatch_is_rejected(tmp_path):
    _write_flow(tmp_path)
    second = _flow_document()
    second["id"] = "second"
    (tmp_path / "second.json").write_text(json.dumps(second), encoding="utf-8")
    flow_skills.publish_flow("research", name="shared_name", root=tmp_path)
    with pytest.raises(flow_skills.FlowSkillError, match="already in use"):
        flow_skills.publish_flow("second", name="shared_name", root=tmp_path)

    recursive = _flow_document()
    recursive["id"] = "recursive"
    recursive["nodes"] = {
        "call": {
            "class": "skill_call",
            "data": {"skill": "run_flow_skill", "args": "{}"},
            "inputs": {},
        }
    }
    (tmp_path / "recursive.json").write_text(json.dumps(recursive), encoding="utf-8")
    flow_skills.publish_flow("recursive", name="recursive_flow", root=tmp_path)
    with pytest.raises(flow_skills.FlowSkillError, match="recursively"):
        flow_skills.run_flow_skill("recursive_flow", root=tmp_path)


def test_published_flow_timeout_does_not_wait_for_cancelled_executor_work(
    tmp_path,
    monkeypatch,
):
    document = _flow_document()
    document["nodes"] = {
        "slow": {
            "class": "skill_call",
            "data": {"skill": "format_json", "args": "{}"},
            "inputs": {},
        }
    }
    _write_flow(tmp_path, document)
    flow_skills.publish_flow("research", name="slow_flow", root=tmp_path)
    monkeypatch.setattr(flow_skills, "_RUN_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(skills, "execute_skill", lambda *_args: time.sleep(0.25))

    started = time.perf_counter()
    result = flow_skills.run_flow_skill("slow_flow", root=tmp_path)
    elapsed = time.perf_counter() - started

    assert result["ok"] is False
    assert "timeout" in result["error"]
    assert elapsed < 0.15


def test_flow_exposure_endpoint_persists_only_when_graph_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_flows_dir", lambda: tmp_path)
    document = _flow_document()
    asyncio.run(webui.save_flow(document))
    asyncio.run(
        webui.expose_flow(
            "research",
            {"name": "research_topic", "description": "Format a topic"},
        )
    )

    document["name"] = "Renamed"
    asyncio.run(webui.save_flow(document))
    loaded = asyncio.run(webui.get_flow("research"))
    assert loaded["skill"]["exposed"] is True
    assert loaded["skill"]["name"] == "research_topic"
    assert loaded["skill"]["description"] == "Format a topic"
    assert len(loaded["skill"]["graph_sha256"]) == 64

    document["nodes"]["format"]["data"]["template"] = "Changed {{input}}"
    asyncio.run(webui.save_flow(document))
    assert asyncio.run(webui.get_flow("research"))["skill"] == {"exposed": False}

    asyncio.run(
        webui.expose_flow(
            "research",
            {"name": "research_topic", "description": "Format a topic"},
        )
    )

    asyncio.run(webui.unexpose_flow("research"))
    assert asyncio.run(webui.get_flow("research"))["skill"] == {"exposed": False}


def test_published_flow_manifest_rejects_graph_tampering(tmp_path):
    path = _write_flow(tmp_path)
    flow_skills.publish_flow("research", name="research_topic", root=tmp_path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["nodes"]["format"]["data"]["template"] = "Tampered {{input}}"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    assert flow_skills.list_flow_skills(root=tmp_path)["count"] == 0
    with pytest.raises(flow_skills.FlowSkillError, match="unpublished"):
        flow_skills.run_flow_skill("research_topic", root=tmp_path)


def test_nested_sensitive_flow_skill_needs_request_scoped_grant(
    tmp_path,
    monkeypatch,
):
    from mio.web_security import model_request_skill_grants

    document = _flow_document()
    document["nodes"] = {
        "call": {
            "class": "skill_call",
            "data": {"skill": "custom_sensitive", "args": '{}'},
            "inputs": {},
        }
    }
    _write_flow(tmp_path, document)
    flow_skills.publish_flow("research", name="sensitive_flow", root=tmp_path)
    monkeypatch.setenv(
        "MIO_WEBUI_SKILL_GRANTS",
        "run_flow_skill,custom_sensitive",
    )
    monkeypatch.setattr(skills, "execute_skill", lambda *_args: {"ok": True})

    with model_request_skill_grants(["run_flow_skill"]):
        denied = flow_skills.run_flow_skill("sensitive_flow", root=tmp_path)
    assert denied["ok"] is False
    assert denied["error"] == "flow contains sensitive skills without required grants"

    with model_request_skill_grants(["run_flow_skill", "custom_sensitive"]):
        allowed = flow_skills.run_flow_skill("sensitive_flow", root=tmp_path)
    assert allowed["ok"] is True


def test_flow_registry_uses_two_stable_schemas_instead_of_one_per_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(flow_skills, "_configured_root", tmp_path)
    names = [tool["function"]["name"] for tool in skills.get_tools_spec()]
    assert names.count("list_flow_skills") == 1
    assert names.count("run_flow_skill") == 1
    assert not any(name.startswith("flow_research") for name in names)
    error = skills.execute_skill("run_flow_skill", {"name": "not_published"})
    assert error["ok"] is False
    assert "unpublished" in error["error"]


def test_flow_editor_has_complete_inspector_and_valid_javascript():
    source = Path("mio/webui/assets/view_flow.js").read_text(encoding="utf-8")
    for node_type in (
        "llm_call", "skill_call", "http_fetch", "if_else", "iterate", "user_input",
        "output", "constant", "template", "parse_json", "to_json", "regex_extract",
        "split", "join", "mem_get", "mem_set", "delay", "clock", "random",
        "rag_search", "artifact_emit",
    ):
        assert f"{node_type}:" in source
    assert "updateNodeDataFromId" in source
    assert "normalizeImportedGraph" in source
    assert 'data-action="expose"' in source
    assert "/expose`" in source
    assert 'data.type === "artifact_emitted"' in source
    assert "api.ingestAndOpen(evt.artifact)" in source
    assert "append to the chat" not in source
    assert "→ chat artifact" not in source
    assert "→ gallery + panel" in source

    shell = Path("mio/webui/mio_ui.html").read_text(encoding="utf-8")
    assert "window.Mio.artifacts = api" in shell
    assert "api.ingestAndOpen" in shell
    assert "mio:artifact-ingested" in shell

    node = shutil.which("node")
    if node:
        result = subprocess.run(
            [node, "--check", "mio/webui/assets/view_flow.js"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_flow_empty_library_uses_inline_feedback_instead_of_alert():
    source = Path("mio/webui/assets/view_flow.js").read_text(encoding="utf-8")

    assert 'flash(host, "No saved flows yet.");' in source
    assert 'alert("No saved flows yet.")' not in source
    assert "alert('No saved flows yet.')" not in source
