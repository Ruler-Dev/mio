"""Regression tests for Flow Mode DAG semantics."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from mio.webui import flow_runner
from mio.webui.flow_runner import _Run, _execute_run, _validate_http_url


def connection(source: str, output: str = "output_1") -> dict:
    # This is Drawflow's real target-side export shape: the source output port
    # is called `input` inside inputs.*.connections[].
    return {"input_1": {"connections": [{"node": source, "input": output}]}}


def run_flow(nodes: dict, env: dict | None = None) -> _Run:
    run = _Run("test-run", {"nodes": nodes}, env)
    asyncio.run(_execute_run(run))
    return run


def drain_events(run: _Run) -> list[dict]:
    events = []
    while not run.queue.empty():
        events.append(run.queue.get_nowait())
    return events


def test_topological_execution_propagates_upstream_value():
    run = run_flow(
        {
            "1": {"class": "constant", "data": {"value": "hello"}, "inputs": {}},
            "2": {
                "class": "template",
                "data": {"template": "{{input}} world"},
                "inputs": connection("1"),
            },
            "3": {"class": "output", "data": {}, "inputs": connection("2")},
        }
    )

    assert run.done
    assert run.outputs == {"1": "hello", "2": "hello world", "3": "hello world"}
    assert drain_events(run)[-1]["outputs"] == ["hello world"]


def test_user_input_is_resolved_from_run_environment():
    run = run_flow(
        {
            "ask": {
                "class": "user_input",
                "data": {"key": "topic", "label": "Topic"},
                "inputs": {},
            },
            "format": {
                "class": "template",
                "data": {"template": "Research {{input}}"},
                "inputs": connection("ask"),
            },
            "out": {"class": "output", "data": {}, "inputs": connection("format")},
        },
        env={"user_input": {"topic": "MLX prefill"}},
    )

    assert run.outputs["ask"] == "MLX prefill"
    assert run.outputs["out"] == "Research MLX prefill"


def test_missing_user_input_emits_node_error_instead_of_empty_stub():
    run = run_flow(
        {
            "ask": {"class": "user_input", "data": {"label": "Required"}, "inputs": {}},
        }
    )

    assert "missing user input" in run.outputs["ask"]["_error"]
    events = drain_events(run)
    assert any(event["type"] == "node_error" for event in events)
    assert events[-1]["type"] == "run_finished"
    assert events[-1]["ok"] is False


def test_node_error_fails_fast_before_downstream_side_effects():
    run = run_flow(
        {
            "bad": {"class": "unknown_node", "data": {}, "inputs": {}},
            "after": {
                "class": "constant",
                "data": {"value": "must not run"},
                "inputs": connection("bad"),
            },
        }
    )

    events = drain_events(run)
    assert "after" not in run.outputs
    assert events[-1]["type"] == "run_finished"
    assert events[-1]["ok"] is False
    assert events[-1]["failed_node"] == "bad"


def test_iterate_preserves_native_list_and_maps_each_item():
    run = run_flow(
        {
            "seed": {"class": "constant", "data": {"value": ["a", "b"]}, "inputs": {}},
            "map": {
                "class": "iterate",
                "data": {"list_expr": "{{input}}", "template": "{{index}}:{{item}}"},
                "inputs": connection("seed"),
            },
            "out": {"class": "output", "data": {}, "inputs": connection("map")},
        }
    )

    assert run.outputs["map"] == ["0:a", "1:b"]
    assert run.outputs["out"] == ["0:a", "1:b"]


def test_if_else_executes_only_the_selected_output_port():
    run = run_flow(
        {
            "seed": {"class": "constant", "data": {"value": True}, "inputs": {}},
            "branch": {
                "class": "if_else",
                "data": {"expr": "{{input}} == true"},
                "inputs": connection("seed"),
            },
            "yes": {
                "class": "template",
                "data": {"template": "selected {{input}}"},
                "inputs": connection("branch", "output_1"),
            },
            "no": {
                "class": "template",
                "data": {"template": "wrong {{input}}"},
                "inputs": connection("branch", "output_2"),
            },
            "out_yes": {"class": "output", "data": {}, "inputs": connection("yes")},
            "out_no": {"class": "output", "data": {}, "inputs": connection("no")},
        }
    )

    assert run.outputs["out_yes"] == "selected True"
    assert "no" not in run.outputs
    assert "out_no" not in run.outputs
    assert any(event["type"] == "node_skipped" for event in drain_events(run))


@pytest.mark.parametrize(
    ("value", "expression", "expected"),
    [
        (True, "{{input}} == true", "true"),
        (False, "{{input}} == true", "false"),
        (None, "{{input}} == true", "false"),
        ([], "{{input}} == true", "false"),
        ("true", "{{input}} == true", "false"),
        ("true", '{{input}} == "true"', "true"),
        (2, "{{input}} > 1", "true"),
        (2.0, "{{input}} == 2", "true"),
        ("2", "{{input}} == 2", "false"),
        ("2", '{{input}} == "2"', "true"),
        (1, "{{input}} == true", "false"),
    ],
)
def test_if_else_preserves_native_types(value, expression, expected):
    result = asyncio.run(
        flow_runner._run_if_else(
            {"expr": expression},
            {"_last": value, "outputs": {}, "env": {}},
        )
    )

    assert result == {"_branch": expected, "value": value}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, "false"),
        (None, "false"),
        ([], "false"),
        ({}, "false"),
        (0, "false"),
        ("", "false"),
        ([0], "true"),
        ("false", "true"),
    ],
)
def test_if_else_truth_test_uses_native_json_truthiness(value, expected):
    result = asyncio.run(
        flow_runner._run_if_else(
            {"expr": "{{input}}"},
            {"_last": value, "outputs": {}, "env": {}},
        )
    )

    assert result["_branch"] == expected


def test_if_else_compound_values_keep_strict_nested_types():
    result = asyncio.run(
        flow_runner._run_if_else(
            {"expr": "{{input}} == {{env.expected}}"},
            {
                "_last": {"items": [True]},
                "outputs": {},
                "env": {"expected": {"items": [1]}},
            },
        )
    )

    assert result["_branch"] == "false"


def test_cycle_fails_without_executing_partial_graph():
    run = run_flow(
        {
            "a": {"class": "template", "data": {}, "inputs": connection("b")},
            "b": {"class": "template", "data": {}, "inputs": connection("a")},
        }
    )

    events = drain_events(run)
    assert run.outputs == {}
    assert events[-1]["type"] == "run_finished"
    assert events[-1]["ok"] is False
    assert "cycle" in events[-1]["error"]


def test_legacy_connection_output_key_remains_readable():
    run = run_flow(
        {
            "seed": {"class": "constant", "data": {"value": "ok"}, "inputs": {}},
            "out": {
                "class": "output",
                "data": {},
                "inputs": {"input_1": {"connections": [{"node": "seed", "output": "output_1"}]}},
            },
        }
    )
    assert run.outputs["out"] == "ok"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://127.0.0.1:9090/private",
        "http://[::1]/private",
        "http://user:pass@example.com/",
    ],
)
def test_http_node_rejects_local_or_non_http_targets(url, monkeypatch):
    monkeypatch.delenv("MIO_FLOW_ALLOW_PRIVATE_HTTP", raising=False)
    with pytest.raises(ValueError):
        _validate_http_url(url)


def test_http_connection_uses_validated_ip_instead_of_resolving_hostname_again(monkeypatch):
    calls = []

    class Socket:
        pass

    monkeypatch.setattr(
        flow_runner.socket,
        "create_connection",
        lambda address, timeout, source: calls.append((address, timeout, source)) or Socket(),
    )
    connection = flow_runner._PinnedHTTPConnection(
        "attacker.example",
        80,
        "93.184.216.34",
        timeout=3.0,
    )
    connection.connect()
    assert calls == [(('93.184.216.34', 80), 3.0, None)]


def test_http_redirect_revalidates_dns_before_opening_second_connection(monkeypatch):
    monkeypatch.delenv("MIO_FLOW_ALLOW_PRIVATE_HTTP", raising=False)

    def resolve(host, port, **_kwargs):
        ip = "93.184.216.34" if host == "public.example" else "127.0.0.1"
        return [(flow_runner.socket.AF_INET, flow_runner.socket.SOCK_STREAM, 6, "", (ip, port))]

    class Response:
        status = 302

        def read(self, _limit):
            return b""

        def getheader(self, name):
            return "http://internal.example/private" if name == "Location" else None

    connections = []

    class Connection:
        def __init__(self, host, port, pinned_ip, *, timeout):
            connections.append((host, port, pinned_ip, timeout))

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(flow_runner.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(flow_runner, "_PinnedHTTPConnection", Connection)

    with pytest.raises(ValueError, match="private"):
        flow_runner._fetch_pinned_http("http://public.example/start", "GET", None)
    assert connections == [("public.example", 80, "93.184.216.34", 30.0)]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_https_redirect_never_downgrades_to_http(status, monkeypatch):
    def resolve(_host, port, **_kwargs):
        return [
            (
                flow_runner.socket.AF_INET,
                flow_runner.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    class Response:
        def read(self, _limit):
            return b""

        def getheader(self, name):
            return "http://public.example/insecure" if name == "Location" else None

    Response.status = status
    requests = []

    class Connection:
        def __init__(self, host, port, pinned_ip, *, timeout):
            self.address = (host, port, pinned_ip, timeout)

        def request(self, method, path, *, body, headers):
            requests.append((self.address, method, path, body, headers))

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(flow_runner.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(flow_runner, "_PinnedHTTPSConnection", Connection)
    monkeypatch.setattr(flow_runner, "_PinnedHTTPConnection", Connection)

    with pytest.raises(ValueError, match="downgrade"):
        flow_runner._fetch_pinned_http(
            "https://secure.example/start",
            "POST",
            b'{"secret": true}',
        )

    assert len(requests) == 1
    assert requests[0][1:4] == (
        "POST",
        "/start",
        b'{"secret": true}',
    )


def test_https_downgrade_is_blocked_after_an_intermediate_secure_hop(monkeypatch):
    def resolve(_host, port, **_kwargs):
        return [
            (
                flow_runner.socket.AF_INET,
                flow_runner.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    redirects = iter(
        [
            (307, "https://second.example/continue"),
            (308, "http://public.example/insecure"),
        ]
    )
    requests = []

    class Response:
        def __init__(self):
            self.status, self.location = next(redirects)

        def read(self, _limit):
            return b""

        def getheader(self, name):
            return self.location if name == "Location" else None

    class Connection:
        def __init__(self, host, port, pinned_ip, *, timeout):
            self.address = (host, port, pinned_ip, timeout)

        def request(self, method, path, *, body, headers):
            requests.append((self.address, method, path, body, headers))

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(flow_runner.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(flow_runner, "_PinnedHTTPSConnection", Connection)
    monkeypatch.setattr(flow_runner, "_PinnedHTTPConnection", Connection)

    with pytest.raises(ValueError, match="downgrade"):
        flow_runner._fetch_pinned_http(
            "https://first.example/start",
            "POST",
            b'{"secret": true}',
        )

    assert [request[0][0] for request in requests] == [
        "first.example",
        "second.example",
    ]
    assert all(request[1] == "POST" for request in requests)
    assert all(request[3] == b'{"secret": true}' for request in requests)


def test_regex_extract_runs_bounded_pattern_in_isolated_worker():
    result = asyncio.run(
        flow_runner._run_regex_extract(
            {"pattern": r"user=(\w+)", "flags": "i"},
            {"_last": "USER=Mio"},
        )
    )

    assert result == "Mio"


@pytest.mark.parametrize(
    ("data", "context", "message"),
    [
        (
            {"pattern": "x" * (flow_runner._MAX_REGEX_PATTERN_BYTES + 1)},
            {"_last": "input"},
            "pattern exceeds",
        ),
        (
            {"pattern": "x"},
            {"_last": "x" * (flow_runner._MAX_REGEX_INPUT_BYTES + 1)},
            "input exceeds",
        ),
    ],
)
def test_regex_extract_rejects_oversized_work(data, context, message):
    with pytest.raises(ValueError, match=message):
        asyncio.run(flow_runner._run_regex_extract(data, context))


def test_regex_extract_terminates_catastrophic_backtracking(monkeypatch):
    monkeypatch.setattr(flow_runner, "_REGEX_TIMEOUT_SECONDS", 0.15)
    started = time.monotonic()

    with pytest.raises(ValueError, match="regex timed out"):
        asyncio.run(
            flow_runner._run_regex_extract(
                {"pattern": r"(a+)+$"},
                {"_last": "a" * 20_000 + "!"},
            )
        )

    assert time.monotonic() - started < 2.0
    # A killed pathological match cannot occupy the executor or poison the
    # next invocation.
    assert asyncio.run(
        flow_runner._run_regex_extract(
            {"pattern": r"(Mio)"},
            {"_last": "Mio"},
        )
    ) == "Mio"


def test_mem_set_read_modify_write_is_atomic_across_threads(tmp_path, monkeypatch):
    home = tmp_path / "mio-home"
    monkeypatch.setenv("MIO_HOME", str(home))
    workers = 8
    writes_per_worker = 5
    barrier = threading.Barrier(workers)

    def write(worker: int) -> None:
        barrier.wait(timeout=5)
        for index in range(writes_per_worker):
            flow_runner._memory_set(f"thread-{worker}-{index}", index)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write, range(workers)))

    memory = json.loads((home / "flow-memory.json").read_text(encoding="utf-8"))
    assert len(memory) == workers * writes_per_worker
    assert memory["thread-0-0"] == 0
    assert memory["thread-7-4"] == 4


def test_mem_set_read_modify_write_is_atomic_across_processes(tmp_path):
    home = tmp_path / "mio-home"
    start = tmp_path / "start"
    workers = 4
    writes_per_worker = 8
    program = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path

        from mio.webui.flow_runner import _memory_set

        prefix, ready_name, start_name, count = sys.argv[1:]
        ready = Path(ready_name)
        start = Path(start_name)
        ready.touch()
        deadline = time.monotonic() + 15
        while not start.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test start signal was not received")
            time.sleep(0.01)
        for index in range(int(count)):
            _memory_set(f"process-{prefix}-{index}", index)
        """
    )
    environment = os.environ.copy()
    environment["MIO_HOME"] = str(home)
    processes: list[subprocess.Popen[str]] = []
    try:
        for worker in range(workers):
            ready = tmp_path / f"ready-{worker}"
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        program,
                        str(worker),
                        str(ready),
                        str(start),
                        str(writes_per_worker),
                    ],
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        deadline = time.monotonic() + 15
        ready_paths = [tmp_path / f"ready-{worker}" for worker in range(workers)]
        while not all(path.exists() for path in ready_paths):
            if time.monotonic() >= deadline:
                raise TimeoutError("memory writers did not become ready")
            time.sleep(0.01)
        start.touch()

        for process in processes:
            _stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()

    memory_path = Path(home) / "flow-memory.json"
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert len(memory) == workers * writes_per_worker
    assert memory["process-0-0"] == 0
    assert memory["process-3-7"] == 7


def test_flow_run_registry_is_bounded_and_evicts_completed_runs(monkeypatch):
    monkeypatch.setattr(flow_runner, "_MAX_RUNS", 2)
    monkeypatch.setattr(flow_runner, "_RUN_TTL_SECONDS", 10)
    flow_runner._runs.clear()
    try:
        expired = _Run("expired", {"nodes": {}})
        expired.close()
        expired.finished = 1.0
        flow_runner._runs["expired"] = expired
        active = _Run("active", {"nodes": {}})
        flow_runner._runs["active"] = active

        flow_runner._prune_runs(now=20.0)
        assert set(flow_runner._runs) == {"active"}

        second = _Run("second", {"nodes": {}})
        flow_runner._runs["second"] = second
        with pytest.raises(RuntimeError, match="too many active"):
            flow_runner.start_run({"nodes": {"x": {"class": "clock", "inputs": {}}}})

        flow_runner.discard_run("second")
        assert set(flow_runner._runs) == {"active"}
    finally:
        flow_runner._runs.clear()


def test_discarding_an_active_run_cancels_its_owned_task():
    async def scenario():
        flow_runner._runs.clear()
        run_id = flow_runner.start_run(
            {
                "nodes": {
                    "wait": {
                        "class": "delay",
                        "data": {"seconds": 10},
                        "inputs": {},
                    }
                }
            }
        )
        run = flow_runner.get_run(run_id)
        assert run is not None and run.task is not None
        await asyncio.sleep(0)
        flow_runner.discard_run(run_id)
        try:
            await run.task
        except asyncio.CancelledError:
            pass
        assert run.cancelled.is_set()
        assert run.done is True
        assert flow_runner.get_run(run_id) is None

    asyncio.run(scenario())


def test_llm_node_uses_injected_manager_and_gpu_lock_off_event_loop():
    main_thread = threading.get_ident()

    class Lock:
        active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, *_args):
            self.active = False

    lock = Lock()

    class Engine:
        def generate_stream(self, messages, **kwargs):
            assert threading.get_ident() != main_thread
            assert lock.active
            assert messages[-1]["content"] == "Say hello seed"
            assert kwargs["max_tokens"] == 9
            assert kwargs["temperature"] == 0.0
            yield "hello", None
            yield "", object()

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    run = _Run(
        "llm-run",
        {
            "nodes": {
                "seed": {"class": "constant", "data": {"value": "seed"}, "inputs": {}},
                "llm": {
                    "class": "llm_call",
                    "data": {"prompt": "Say hello {{input}}", "max_tokens": 9},
                    "inputs": connection("seed"),
                },
                "out": {"class": "output", "data": {}, "inputs": connection("llm")},
            }
        },
        manager=Manager(),
        gpu_lock=lock,
    )
    asyncio.run(_execute_run(run))
    assert run.outputs["out"] == "hello"


def test_cancelled_flow_never_generates_after_waiting_for_gpu_lock():
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    engine_calls = 0

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, _tier):
            nonlocal engine_calls
            engine_calls += 1
            raise AssertionError("cancelled flow must not resolve an engine")

    run = _Run(
        "cancelled-lock-wait",
        {
            "nodes": {
                "llm": {
                    "class": "llm_call",
                    "data": {"prompt": "never run"},
                    "inputs": {},
                }
            }
        },
        manager=Manager(),
        gpu_lock=gpu_lock,
    )

    async def scenario():
        task = asyncio.create_task(_execute_run(run))
        await asyncio.sleep(0.02)
        run.cancel()
        await task

    try:
        asyncio.run(scenario())
    finally:
        gpu_lock.release()

    assert engine_calls == 0
    assert run.final_event["ok"] is False
    assert "cancelled" in run.final_event["error"]


def test_malformed_graph_always_reaches_a_terminal_state():
    run = _Run(
        "malformed",
        {
            "nodes": {
                "bad": {
                    "class": "output",
                    "data": {},
                    "inputs": {"input_1": "not-a-port-object"},
                }
            }
        },
    )

    asyncio.run(_execute_run(run))

    assert run.done is True
    assert run.final_event is not None
    assert run.final_event["ok"] is False
    assert "invalid flow graph" in run.final_event["error"]


def test_sensitive_flow_skills_require_an_exact_operator_grant(monkeypatch):
    from mio.webui import skills

    calls = []
    monkeypatch.setattr(skills, "execute_skill", lambda name, args: calls.append((name, args)) or "ok")
    monkeypatch.delenv("MIO_WEBUI_SKILL_GRANTS", raising=False)
    graph = {
        "nodes": {
            "call": {
                "class": "skill_call",
                "data": {"skill": "custom_sensitive", "args": '{"x": 1}'},
                "inputs": {},
            }
        }
    }

    denied = _Run("denied", graph)
    asyncio.run(_execute_run(denied))
    assert denied.final_event["ok"] is False
    assert denied.final_event["missing_grants"] == [
        {"skill": "custom_sensitive", "risk": "sensitive"}
    ]
    assert calls == []

    monkeypatch.setenv("MIO_WEBUI_SKILL_GRANTS", "custom_sensitive")
    granted = _Run("granted", graph)
    asyncio.run(_execute_run(granted))
    assert granted.final_event["ok"] is True
    assert calls == [("custom_sensitive", {"x": 1})]


def test_error_shaped_http_node_fails_run_before_downstream_nodes():
    graph = {
        "nodes": {
            "fetch": {
                "class": "http_fetch",
                "data": {"url": "https://example.invalid", "method": "DELETE"},
                "inputs": {},
            },
            "out": {
                "class": "output",
                "data": {},
                "inputs": connection("fetch"),
            },
        }
    }
    run = _Run("http-error", graph)

    asyncio.run(_execute_run(run))

    assert run.final_event["ok"] is False
    assert run.final_event["failed_node"] == "fetch"
    assert "only GET and POST" in run.final_event["error"]
    assert "out" not in run.outputs


def test_artifact_node_emits_structured_gallery_event():
    run = run_flow(
        {
            "seed": {
                "class": "constant",
                "data": {"value": "<main>Hello Flow</main>"},
                "inputs": {},
            },
            "artifact": {
                "class": "artifact_emit",
                "data": {"type": "text/html", "title": "Flow result"},
                "inputs": connection("seed"),
            },
            "out": {
                "class": "output",
                "data": {},
                "inputs": connection("artifact"),
            },
        }
    )

    artifact = run.outputs["artifact"]["artifact"]
    assert artifact == {
        "id": artifact["id"],
        "type": "text/html",
        "title": "Flow result",
        "content": "<main>Hello Flow</main>",
        "source": "flow",
    }
    assert artifact["id"].startswith("flow-")
    assert "<antArtifact" not in str(run.outputs)
    emitted = [event for event in drain_events(run) if event["type"] == "artifact_emitted"]
    assert emitted == [
        {
            "type": "artifact_emitted",
            "node_id": "artifact",
            "class": "artifact_emit",
            "artifact": artifact,
        }
    ]


def test_artifact_node_rejects_oversized_content():
    oversized = "x" * (flow_runner._MAX_ARTIFACT_CONTENT_BYTES + 1)
    run = run_flow(
        {
            "seed": {"class": "constant", "data": {"value": oversized}, "inputs": {}},
            "artifact": {
                "class": "artifact_emit",
                "data": {"type": "text/html", "title": "Too large"},
                "inputs": connection("seed"),
            },
        }
    )

    events = drain_events(run)
    assert "exceeds" in run.outputs["artifact"]["_error"]
    assert not any(event["type"] == "artifact_emitted" for event in events)
    assert events[-1]["type"] == "run_finished"
    assert events[-1]["ok"] is False


def test_flow_artifact_output_has_an_aggregate_run_bound(monkeypatch):
    monkeypatch.setattr(flow_runner, "_MAX_ARTIFACT_RUN_BYTES", 5)
    run = run_flow(
        {
            "seed": {"class": "constant", "data": {"value": "1234"}, "inputs": {}},
            "first": {
                "class": "artifact_emit",
                "data": {"type": "text/plain", "title": "First"},
                "inputs": connection("seed"),
            },
            "second": {
                "class": "artifact_emit",
                "data": {"type": "text/plain", "title": "Second"},
                "inputs": connection("seed"),
            },
        }
    )

    events = drain_events(run)
    assert run.artifact_count == 1
    assert run.artifact_bytes == 4
    assert "output exceeds" in run.outputs["second"]["_error"]
    assert len([event for event in events if event["type"] == "artifact_emitted"]) == 1
    assert events[-1]["ok"] is False
