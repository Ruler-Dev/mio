"""Flow Mode — async DAG executor.

Takes a Drawflow-exported graph and runs it server-side, emitting
SSE events per node. Supported node classes in this MVP:

  llm_call     chat completion via the OpenAI-compatible endpoint
  skill_call   dispatch to the registered Mio skills
  http_fetch   simple GET/POST
  if_else      branch on a truthy-expression against the input
  iterate      run downstream subgraph per item in the input list
  user_input   NOT executed server-side (stubs to empty output)
  output       terminal — collects into the run's result bucket

Nodes carry a `data` payload edited in the UI. Variable interpolation
uses simple `{{...}}` templating:
  {{input}}    — the last upstream output
  {{n.out}}    — the output of node N (id)
  {{env.X}}    — environment values set at run start

This is a deliberately small executor: no parallelism inside a
single flow, no retries / backoff, no loop detection beyond max
200 node-hops. Enough to be useful; big enough to iterate.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import urllib.request
import uuid
from typing import Any

_runs: dict[str, "_Run"] = {}
_MAX_HOPS = 200


class _Run:
    def __init__(self, run_id: str, flow: dict, env: dict | None = None):
        self.id = run_id
        self.flow = flow
        self.env = env or {}
        self.outputs: dict[str, Any] = {}
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.done = False
        self.started = time.time()

    async def emit(self, event: dict) -> None:
        await self.queue.put(event)

    def close(self) -> None:
        self.done = True


def _interpolate(tpl: str, ctx: dict) -> str:
    def repl(m: re.Match) -> str:
        expr = m.group(1).strip()
        if expr == "input":
            return str(ctx.get("_last", ""))
        if expr.startswith("env."):
            return str(ctx.get("env", {}).get(expr[4:], ""))
        # nodeid.out — use output of that node
        if "." in expr:
            nid, _ = expr.split(".", 1)
            val = ctx.get("outputs", {}).get(nid, "")
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            return str(val)
        return ""
    return re.sub(r"\{\{\s*(.*?)\s*\}\}", repl, tpl or "")


async def _run_llm(data: dict, ctx: dict) -> str:
    prompt = _interpolate(data.get("prompt", "") or "{{input}}", ctx)
    system = _interpolate(data.get("system", ""), ctx)
    body = {
        "model": "mio-auto",
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": prompt},
        ],
        "temperature": float(data.get("temperature", 0.7) or 0.7),
        "max_tokens": int(data.get("max_tokens", 1024) or 1024),
        "stream": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:9090/v1/chat/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode(),
    )
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=120))
    data_resp = json.loads(resp.read())
    return data_resp.get("choices", [{}])[0].get("message", {}).get("content", "")


async def _run_skill(data: dict, ctx: dict) -> Any:
    from mio.webui.skills import execute_skill
    skill = data.get("skill", "")
    args_raw = data.get("args", "{}")
    try:
        args = json.loads(_interpolate(args_raw if isinstance(args_raw, str) else json.dumps(args_raw), ctx))
    except Exception:
        args = {}
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: execute_skill(skill, args))


async def _run_http(data: dict, ctx: dict) -> Any:
    method = (data.get("method") or "GET").upper()
    url = _interpolate(data.get("url", ""), ctx)
    if not url:
        return {"error": "url required"}
    req = urllib.request.Request(url, method=method)
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30))
        body = resp.read()
        try:
            return json.loads(body)
        except Exception:
            return body.decode(errors="replace")
    except Exception as e:
        return {"error": str(e)}


async def _run_if_else(data: dict, ctx: dict) -> dict:
    expr = _interpolate(data.get("expr", ""), ctx)
    # Refuse anything that smells risky — we only allow simple comparison
    # or truthy tests over the input.
    safe = re.match(r"^([\w\.\[\]'\"]+)\s*(==|!=|<|>|<=|>=)?\s*([\w\.\[\]'\"]*)$", expr.strip())
    branch = "false"
    try:
        if safe:
            lhs, op, rhs = safe.groups()
            # Very small evaluator — compare strings as-is, numbers coerce
            def coerce(v):
                try: return float(v)
                except Exception: return str(v).strip('"\'')
            lv = coerce(lhs); rv = coerce(rhs) if rhs else None
            if op is None:     branch = "true" if lv else "false"
            elif op == "==":   branch = "true" if lv == rv else "false"
            elif op == "!=":   branch = "true" if lv != rv else "false"
            elif op == "<":    branch = "true" if lv < rv else "false"
            elif op == ">":    branch = "true" if lv > rv else "false"
            elif op == "<=":   branch = "true" if lv <= rv else "false"
            elif op == ">=":   branch = "true" if lv >= rv else "false"
    except Exception:
        branch = "false"
    return {"_branch": branch, "value": ctx.get("_last", "")}


async def _execute_run(run: _Run) -> None:
    """Topologically walk the graph in a single pass from user_input /
    first llm_call nodes to output nodes. For this MVP we do a naïve BFS
    respecting `connections_in` dependencies — no parallelism."""
    nodes = run.flow.get("nodes") or {}
    if not nodes:
        await run.emit({"type": "run_finished", "ok": False, "error": "empty flow"})
        run.close(); return

    # Build indegree + dependency map.
    in_edges: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for port in (node.get("inputs") or {}).values():
            for conn in port.get("connections", []):
                in_edges[nid].append(str(conn.get("node")))

    # Toposort
    order: list[str] = []
    indeg = {nid: len(deps) for nid, deps in in_edges.items()}
    queue = [nid for nid, d in indeg.items() if d == 0]
    hops = 0
    while queue:
        if hops > _MAX_HOPS: break
        hops += 1
        nid = queue.pop(0)
        order.append(nid)
        for other, deps in in_edges.items():
            if nid in deps:
                deps.remove(nid)
                indeg[other] -= 1
                if indeg[other] == 0 and other not in order:
                    queue.append(other)

    ctx = {"env": run.env, "outputs": run.outputs, "_last": ""}
    outputs_bucket: list[Any] = []

    await run.emit({"type": "run_started", "node_order": order})
    for nid in order:
        node = nodes.get(nid)
        if not node:
            continue
        cls = node.get("class") or node.get("name") or ""
        data = node.get("data", {}) or {}

        # Gather last upstream output (first predecessor)
        preds = in_edges.get(nid) or []
        if preds:
            ctx["_last"] = run.outputs.get(preds[0], "")

        await run.emit({"type": "node_started", "node_id": nid, "class": cls})
        try:
            if cls == "llm_call":
                out = await _run_llm(data, ctx)
            elif cls == "skill_call":
                out = await _run_skill(data, ctx)
            elif cls == "http_fetch":
                out = await _run_http(data, ctx)
            elif cls == "if_else":
                out = await _run_if_else(data, ctx)
            elif cls == "iterate":
                # MVP: pass the list through unchanged. Real iteration
                # lands in the next commit once we have subgraph tagging.
                raw = _interpolate(data.get("list_expr", "{{input}}"), ctx)
                try:
                    out = json.loads(raw)
                except Exception:
                    out = raw.split("\n") if raw else []
            elif cls == "user_input":
                out = ""  # paused; UI fills in later
            elif cls == "output":
                out = ctx["_last"]
                outputs_bucket.append(out)
            else:
                out = {"error": f"unknown class: {cls}"}

            run.outputs[nid] = out
            await run.emit({"type": "node_finished", "node_id": nid, "class": cls, "output": _preview(out)})
        except Exception as e:
            await run.emit({"type": "node_error", "node_id": nid, "class": cls, "error": str(e)})
            run.outputs[nid] = {"_error": str(e)}

    await run.emit({
        "type": "run_finished",
        "ok":   True,
        "outputs": outputs_bucket,
        "elapsed_ms": int((time.time() - run.started) * 1000),
    })
    run.close()


def _preview(val: Any) -> Any:
    if isinstance(val, str):
        return val[:400]
    if isinstance(val, (dict, list)):
        try:
            s = json.dumps(val)[:400]
            return json.loads(s) if s.strip().startswith(("[", "{")) else s
        except Exception:
            return str(val)[:400]
    return val


def start_run(flow: dict, env: dict | None = None) -> str:
    run_id = str(uuid.uuid4())[:12]
    run = _Run(run_id, flow, env)
    _runs[run_id] = run
    asyncio.create_task(_execute_run(run))
    return run_id


def get_run(run_id: str) -> _Run | None:
    return _runs.get(run_id)
