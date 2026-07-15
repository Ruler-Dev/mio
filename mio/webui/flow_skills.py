"""Persistent, bounded bridge between Flow Mode and Mio's skill registry.

Flows are not expanded into one model tool schema per graph: that would make
the prompt grow without bound.  Mio instead exposes two stable tools — list
and run — while publication metadata lives beside each flow JSON document.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mio.paths import mio_home
from mio.persistence import atomic_write_bytes
from mio.web_security import current_model_request_skill_grants
from mio.webui.flow_runner import _Run, _execute_run, required_flow_skill_grants

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_FLOW_FILES = 1_000
_MAX_FLOW_FILE_BYTES = 2 * 1024 * 1024
_MAX_ARGUMENT_BYTES = 64 * 1024
_MAX_RESULT_BYTES = 256 * 1024
_RUN_TIMEOUT_SECONDS = 120.0

_manager: Any = None
_gpu_lock: Any = None
_configured_root: Path | None = None


class FlowSkillError(ValueError):
    """Raised when a flow cannot safely be published or executed as a skill."""


def configure_runtime(manager: Any, gpu_lock: Any = None, root: Path | None = None) -> None:
    """Inject process-local execution state used by the stable run tool."""
    global _manager, _gpu_lock, _configured_root
    _manager = manager
    _gpu_lock = gpu_lock
    if root is not None:
        _configured_root = Path(root).expanduser().resolve()


def _root(root: Path | None = None) -> Path:
    directory = Path(root or _configured_root or (mio_home() / "flows")).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def _flow_path(flow_id: str, root: Path | None = None, *, must_exist: bool = True) -> Path:
    if not isinstance(flow_id, str) or not _IDENTIFIER_RE.fullmatch(flow_id):
        raise FlowSkillError("invalid flow id")
    directory = _root(root)
    candidate = directory / f"{flow_id}.json"
    if candidate.is_symlink():
        raise FlowSkillError("flow documents may not be symlinks")
    path = candidate.resolve()
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise FlowSkillError("flow path escapes Mio storage") from exc
    if must_exist and (not path.is_file() or path.stat().st_size > _MAX_FLOW_FILE_BYTES):
        raise FlowSkillError("flow not found or too large")
    return path


def _load_flow(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlowSkillError(f"invalid flow document: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("nodes", {}), dict):
        raise FlowSkillError("invalid flow document")
    return data


def _write_flow(path: Path, data: dict[str, Any]) -> None:
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(payload) > _MAX_FLOW_FILE_BYTES:
        raise FlowSkillError("flow document exceeds the 2 MiB limit")
    atomic_write_bytes(path, payload)


def _iter_flows(root: Path | None = None):
    directory = _root(root)
    seen = 0
    for path in sorted(directory.glob("*.json")):
        if seen >= _MAX_FLOW_FILES:
            break
        seen += 1
        try:
            if path.is_symlink() or path.stat().st_size > _MAX_FLOW_FILE_BYTES:
                continue
            yield path, _load_flow(path)
        except (FlowSkillError, OSError):
            continue


def flow_capability_digest(flow: dict[str, Any]) -> str:
    """Hash the executable graph bound to one published capability."""

    canonical = json.dumps(
        {
            "nodes": flow.get("nodes") or {},
            "edges": flow.get("edges") or [],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _published_metadata(flow: dict[str, Any]) -> dict[str, str] | None:
    skill = flow.get("skill")
    if not isinstance(skill, dict) or not skill.get("exposed"):
        return None
    name = skill.get("name")
    if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name):
        return None
    digest = skill.get("graph_sha256")
    if not isinstance(digest, str) or digest != flow_capability_digest(flow):
        return None
    description = str(skill.get("description") or f"Run the {flow.get('name', name)} Mio flow")
    return {"name": name, "description": description[:500]}


def _input_descriptors(flow: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for node_id, node in list((flow.get("nodes") or {}).items())[:200]:
        if not isinstance(node, dict) or (node.get("class") or node.get("name")) != "user_input":
            continue
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        descriptors.append(
            {
                "key": str(data.get("key") or node_id),
                "label": str(data.get("label") or data.get("key") or node_id)[:200],
                "required": "default" not in data,
            }
        )
    return descriptors


def publish_flow(
    flow_id: str,
    *,
    name: str,
    description: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name):
        raise FlowSkillError("skill name must match [a-z][a-z0-9_]{0,63}")
    description = str(description or "").strip()
    if len(description) > 500:
        raise FlowSkillError("skill description must be at most 500 characters")
    path = _flow_path(flow_id, root)
    for other_path, other in _iter_flows(root):
        metadata = _published_metadata(other)
        if metadata and metadata["name"] == name and other_path != path:
            raise FlowSkillError(f"flow skill name {name!r} is already in use")
    flow = _load_flow(path)
    flow["skill"] = {
        "exposed": True,
        "name": name,
        "description": description or f"Run the {flow.get('name', name)} Mio flow",
        "graph_sha256": flow_capability_digest(flow),
    }
    _write_flow(path, flow)
    return {"ok": True, "flow_id": flow_id, "skill": flow["skill"]}


def unpublish_flow(flow_id: str, *, root: Path | None = None) -> dict[str, Any]:
    path = _flow_path(flow_id, root)
    flow = _load_flow(path)
    flow["skill"] = {"exposed": False}
    _write_flow(path, flow)
    return {"ok": True, "flow_id": flow_id, "skill": flow["skill"]}


def list_flow_skills(query: str = "", limit: int = 50, *, root: Path | None = None) -> dict[str, Any]:
    query = str(query or "").strip().casefold()
    limit = max(1, min(int(limit), 100))
    results: list[dict[str, Any]] = []
    for path, flow in _iter_flows(root):
        metadata = _published_metadata(flow)
        if not metadata:
            continue
        haystack = f"{metadata['name']} {metadata['description']} {flow.get('name', '')}".casefold()
        if query and query not in haystack:
            continue
        results.append(
            {
                "name": metadata["name"],
                "description": metadata["description"],
                "flow_id": str(flow.get("id") or path.stem),
                "inputs": _input_descriptors(flow),
                "missing_operator_grants": required_flow_skill_grants(flow),
            }
        )
        if len(results) >= limit:
            break
    return {"skill": "list_flow_skills", "count": len(results), "flows": results}


def _find_published(name: str, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name):
        raise FlowSkillError("invalid flow skill name")
    for path, flow in _iter_flows(root):
        metadata = _published_metadata(flow)
        if metadata and metadata["name"] == name:
            return path, flow
    raise FlowSkillError(f"unknown or unpublished flow skill: {name}")


def run_flow_skill(
    name: str,
    input: Any = None,
    variables: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Execute one published flow through the stable, bounded Mio tool."""
    _path, flow = _find_published(name, root)
    if variables is not None and not isinstance(variables, dict):
        raise FlowSkillError("flow variables must be an object")
    variables = dict(variables or {})
    try:
        argument_bytes = len(
            json.dumps({"input": input, "variables": variables}, ensure_ascii=False).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise FlowSkillError("flow arguments must be JSON serializable") from exc
    if argument_bytes > _MAX_ARGUMENT_BYTES:
        raise FlowSkillError("flow arguments exceed the 64 KiB limit")

    # Avoid unbounded recursive graph spawning. Composition can be introduced
    # later with an explicit depth token carried across worker threads.
    for node in (flow.get("nodes") or {}).values():
        if not isinstance(node, dict) or (node.get("class") or node.get("name")) != "skill_call":
            continue
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        if data.get("skill") == "run_flow_skill":
            raise FlowSkillError("published flows cannot recursively call run_flow_skill")

    user_input = dict(variables)
    descriptors = _input_descriptors(flow)
    if input is not None and len(descriptors) == 1:
        user_input.setdefault(descriptors[0]["key"], input)
    env = {**variables, "input": input, "user_input": user_input}
    run = _Run(
        f"skill-{name}",
        flow,
        env,
        manager=_manager,
        gpu_lock=_gpu_lock,
        request_skill_grants=current_model_request_skill_grants(),
    )

    async def execute() -> None:
        await asyncio.wait_for(_execute_run(run), timeout=_RUN_TIMEOUT_SECONDS)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(execute())
    except TimeoutError:
        return {
            "skill": "run_flow_skill",
            "name": name,
            "ok": False,
            "error": f"flow exceeded {_RUN_TIMEOUT_SECONDS:.0f}s timeout",
        }
    finally:
        asyncio.set_event_loop(None)
        # Unlike asyncio.run(), closing a manually-owned loop does not wait
        # indefinitely for a cancelled default-executor job. The Run's
        # cancellation event cooperatively stops Mio's streaming model worker;
        # bounded third-party skills may finish their current call off-loop.
        loop.close()

    final = run.final_event or {"ok": False, "error": "flow ended without a final event"}
    result: dict[str, Any] = {
        "skill": "run_flow_skill",
        "name": name,
        "ok": bool(final.get("ok")),
        "outputs": final.get("outputs", []),
        "elapsed_ms": final.get("elapsed_ms"),
    }
    if final.get("error"):
        result["error"] = final["error"]
    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
    if len(encoded) > _MAX_RESULT_BYTES:
        preview = json.dumps(result.get("outputs"), ensure_ascii=False)[: _MAX_RESULT_BYTES // 2]
        result["outputs"] = preview
        result["truncated"] = True
    return result
