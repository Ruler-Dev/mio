"""Tests for the Mio API server (unit tests, no model loading)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mio import server
from mio.engine import GenerationMetrics
from mio.mcp import MCPRegistry, MCPServerConfig, MCPTransport
from mio.router import TandemRouter
from mio.server import _cors_origins
from mio.web_security import host_allowed, reset_web_security_state


def test_router_explicit_tier():
    """Router should respect explicit tier names."""
    router = TandemRouter(["small", "medium", "large"])
    assert router.route([], model_hint="mio-large") == "large"
    assert router.route([], model_hint="mio-small") == "small"
    assert router.route([], model_hint="mio-medium") == "medium"


def test_router_simple_message():
    """Short simple messages should route to small tier."""
    router = TandemRouter(["small", "medium", "large"])
    messages = [{"role": "user", "content": "fix the typo"}]
    assert router.route(messages, model_hint="mio-auto") == "small"


def test_router_complex_message():
    """Complex multi-file tasks should route to large tier."""
    router = TandemRouter(["small", "medium", "large"])
    messages = [{"role": "user", "content": "refactor the entire authentication system across multiple files"}]
    assert router.route(messages, model_hint="mio-auto") == "large"


def test_router_medium_message():
    """Medium-length standard tasks should route to medium tier."""
    router = TandemRouter(["small", "medium", "large"])
    # A moderate task description
    msg = "Write a function that parses JSON config files and validates them against a schema, including error messages for each invalid field"
    messages = [{"role": "user", "content": msg}]
    assert router.route(messages, model_hint="mio-auto") == "medium"


def test_router_fallback_when_tier_unavailable():
    """Router should fall back when preferred tier not available."""
    router = TandemRouter(["small", "large"])  # no medium
    # Explicit medium → should fall to closest available
    result = router.route([], model_hint="mio-medium")
    assert result in ["small", "large"]


def test_router_single_tier():
    """Router with single tier should always return that tier."""
    router = TandemRouter(["large"])
    messages = [{"role": "user", "content": "fix typo"}]
    assert router.route(messages, model_hint="mio-auto") == "large"


def test_chat_endpoint_auto_routes_real_prompt_against_current_tiers(monkeypatch):
    selected: list[str] = []

    class Engine:
        tier_config = SimpleNamespace(max_output_tokens=32)

        def generate(self, _messages, **_kwargs):
            return (
                "ok",
                GenerationMetrics(prompt_tokens=8, completion_tokens=1, generation_tps=10.0),
            )

    class Manager:
        def loaded_tiers(self):
            # Deliberately differs from the router's stale startup snapshot.
            return ["medium", "large"]

        def get_engine(self, tier):
            selected.append(tier)
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", TandemRouter(["small"]))
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_compact_enabled", False)
    request = server.ChatCompletionRequest(
        model="mio-auto",
        messages=[
            {
                "role": "user",
                "content": "refactor the entire authentication system across multiple files",
            }
        ],
    )

    response = asyncio.run(server.chat_completions(request))

    assert response["model"] == "mio-large"
    assert selected and set(selected) == {"large"}


def test_dynamic_second_tier_refreshes_tandem_router(monkeypatch):
    class Manager:
        def __init__(self):
            self.tiers = ["small"]

        def loaded_tiers(self):
            return list(self.tiers)

        def load_tier(self, tier):
            self.tiers.append(tier)

    manager = Manager()
    monkeypatch.setattr(server, "_manager", manager)
    monkeypatch.setattr(server, "_tandem_enabled", True)
    monkeypatch.setattr(server, "_router", TandemRouter(["small"]))

    response = asyncio.run(server.load_model(server.TierLoadRequest(tier="large")))

    assert response == {"status": "loaded", "tier": "large"}
    assert server._router is not None
    assert server._router.available_tiers == ["small", "large"]
    assert server._resolve_tier(
        "mio-auto",
        [{"role": "user", "content": "refactor the entire system across files"}],
    ) == "large"


def test_nonstream_rest_uses_engine_generate_for_bmp_mode(monkeypatch):
    calls: list[str] = []

    class Engine:
        tier_config = SimpleNamespace(max_output_tokens=32, bmp_paths=4)

        def generate(self, _messages, **_kwargs):
            calls.append("generate")
            return (
                "bmp answer",
                GenerationMetrics(
                    prompt_tokens=7,
                    completion_tokens=2,
                    total_tokens=9,
                    generation_tps=20.0,
                ),
            )

        def generate_stream(self, _messages, **_kwargs):
            calls.append("generate_stream")
            raise AssertionError("non-streaming REST must not bypass BMP mode selection")

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_compact_enabled", False)
    request = server.ChatCompletionRequest(
        model="mio-small",
        messages=[{"role": "user", "content": "answer"}],
    )

    response = asyncio.run(server.chat_completions(request))

    assert calls == ["generate"]
    assert response["choices"][0]["message"]["content"] == "bmp answer"
    assert response["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


def test_batch_endpoint_auto_routes_each_real_prompt(monkeypatch):
    selected: list[str] = []

    class Engine:
        tier_config = SimpleNamespace(
            temperature=0.0,
            top_p=0.95,
            top_k=20,
            max_output_tokens=32,
        )

        def generate_batch(self, messages, **_kwargs):
            return [
                (
                    request[-1]["content"],
                    GenerationMetrics(prompt_tokens=2, completion_tokens=1, generation_tps=10.0),
                )
                for request in messages
            ]

    class Manager:
        def loaded_tiers(self):
            return ["small", "large"]

        def get_engine(self, tier):
            selected.append(tier)
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", TandemRouter(["medium"]))
    request = server.BatchCompletionRequest(
        requests=[
            {
                "model": "mio-auto",
                "messages": [{"role": "user", "content": "fix the typo"}],
            },
            {
                "model": "mio-auto",
                "messages": [
                    {
                        "role": "user",
                        "content": "refactor the entire authentication system across multiple files",
                    }
                ],
            },
        ]
    )

    response = asyncio.run(server.batch_completions(request))

    assert [item["text"] for item in response] == [
        "fix the typo",
        "refactor the entire authentication system across multiple files",
    ]
    assert set(selected) == {"small", "large"}


def test_cors_defaults_to_loopback_origins(monkeypatch):
    monkeypatch.delenv("MIO_CORS_ORIGINS", raising=False)
    origins = _cors_origins(9090)
    assert origins == [
        "http://127.0.0.1:9090",
        "http://localhost:9090",
        "http://[::1]:9090",
    ]
    assert "*" not in origins


def test_cors_explicit_override(monkeypatch):
    monkeypatch.setenv(
        "MIO_CORS_ORIGINS",
        "https://app.example:443/, http://localhost:3000",
    )
    assert _cors_origins(9090) == ["https://app.example", "http://localhost:3000"]


@pytest.mark.parametrize(
    "origin",
    ["*", "file:///tmp/client.html", "https://user:pass@app.example", "https://app.example/path"],
)
def test_cors_explicit_override_rejects_broad_or_malformed_origins(monkeypatch, origin):
    monkeypatch.setenv("MIO_CORS_ORIGINS", origin)
    with pytest.raises(ValueError, match="invalid explicit CORS origin"):
        _cors_origins(9090)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.42.0.1", "::1", "[::1]", "localhost"])
def test_server_loopback_bind_detection(host):
    assert server._is_loopback_bind(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.2", "mio.example", ""])
def test_server_rejects_ambiguous_or_remote_bind_by_default(host, monkeypatch):
    monkeypatch.delenv("MIO_UNSAFE_REMOTE_BIND", raising=False)
    with pytest.raises(RuntimeError, match="refusing non-loopback") as error:
        server.start_server(object(), host=host, port=19090, live_panel=False)
    assert "--unsafe-remote-bind" in str(error.value)
    assert "MIO_TRUSTED_HOSTS" in str(error.value)


def test_unsafe_remote_bind_configures_exact_host_and_browser_origin(monkeypatch):
    class Manager:
        def loaded_tiers(self):
            return []

        def get_model_names(self):
            return []

    monkeypatch.setattr(server, "_probe_server_bind", lambda _host, _port: None)
    monkeypatch.setattr(server, "_install_plain_line_printer", lambda: None)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("MIO_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("MIO_TRUSTED_HOSTS", raising=False)
    reset_web_security_state()
    try:
        server.start_server(
            Manager(),
            host="192.168.50.20",
            port=19093,
            live_panel=False,
            unsafe_remote_bind=True,
            mcp_registry=MCPRegistry([]),
        )

        assert host_allowed("192.168.50.20:19093")
        assert not host_allowed("192.168.50.21:19093")
        assert "http://192.168.50.20:19093" in _cors_origins(19093)
    finally:
        reset_web_security_state()


def test_server_never_kills_port_owner_without_explicit_replace(monkeypatch):
    killed = []
    monkeypatch.setattr(server, "_kill_port_holders", lambda port: killed.append(port))
    monkeypatch.setattr(
        server,
        "_probe_server_bind",
        lambda host, port: (_ for _ in ()).throw(RuntimeError("occupied")),
    )

    with pytest.raises(RuntimeError, match="occupied"):
        server.start_server(object(), host="127.0.0.1", port=19091, live_panel=False)
    assert killed == []

    with pytest.raises(RuntimeError, match="occupied"):
        server.start_server(
            object(),
            host="127.0.0.1",
            port=19091,
            live_panel=False,
            replace_existing=True,
        )
    assert killed == [19091]


def test_validate_alias_retries_off_event_loop_under_gpu_lock(monkeypatch):
    main_thread = threading.get_ident()
    calls = []

    class Engine:
        def generate(self, _messages, **_kwargs):
            calls.append((threading.get_ident(), server._GPU_LOCK.locked()))
            text = "first attempt" if len(calls) == 1 else "validated attempt"
            return text, GenerationMetrics(prompt_tokens=2, completion_tokens=2, generation_tps=10.0)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    validation_calls = []

    def validate_response(text):
        validation_calls.append(text)
        return SimpleNamespace(passed=len(validation_calls) > 1, errors=["retry"])

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_validate_enabled", True)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr("mio.validator.validate_response", validate_response)
    monkeypatch.setattr("mio.validator.build_retry_message", lambda errors: "fix: " + ",".join(errors))

    request = server.ChatCompletionRequest(
        model="mio-small",
        messages=[{"role": "user", "content": "generate"}],
        validate=True,
    )
    assert request.validate_output is True
    response = asyncio.run(server.chat_completions(request))

    assert response["choices"][0]["message"]["content"] == "validated attempt"
    assert len(calls) == 2
    assert all(thread_id != main_thread and locked for thread_id, locked in calls)


def test_mcp_endpoint_lists_only_config_and_redacts_environment(monkeypatch):
    config = MCPServerConfig(
        name="local",
        transport=MCPTransport.STDIO,
        command=("missing-tool",),
        environment={"TOKEN_LIKE_VALUE": "do-not-return"},
    )
    monkeypatch.setattr(server, "_mcp_registry", MCPRegistry([config]))
    response = asyncio.run(server.list_mcp_servers())
    assert response["data"][0]["environment"] == {"TOKEN_LIKE_VALUE": "<redacted>"}


def test_batch_endpoint_uses_continuous_engine_and_preserves_order(monkeypatch):
    class Engine:
        tier_config = SimpleNamespace(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_output_tokens=32,
        )

        def generate_batch(self, messages, **_kwargs):
            return [
                (
                    request[-1]["content"].upper(),
                    GenerationMetrics(
                        prompt_tokens=2,
                        completion_tokens=1,
                        generation_tps=50.0,
                        metrics_scope="batch",
                        batch_size=len(messages),
                    ),
                )
                for request in messages
            ]

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    request = server.BatchCompletionRequest(
        requests=[
            {
                "model": "mio-small",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "one"}]}],
                "max_tokens": 5,
                "temperature": 0.2,
            },
            {
                "model": "mio-small",
                "messages": [{"role": "user", "content": "two"}],
                "max_tokens": 5,
                "temperature": 0.2,
            },
        ]
    )

    response = asyncio.run(server.batch_completions(request))

    assert [item["text"] for item in response] == ["ONE", "TWO"]
    assert all(item["backend"] == "mlx-continuous" for item in response)
    assert all(item["generation_tps"] is None for item in response)
    assert all(item["batch_generation_tps"] == 50.0 for item in response)
    assert all(item["metrics_scope"] == "batch" and item["batch_size"] == 2 for item in response)


def test_batch_endpoint_rejects_empty_or_oversized_batches(monkeypatch):
    monkeypatch.setattr(server, "_manager", object())
    with pytest.raises(HTTPException) as empty:
        asyncio.run(server.batch_completions(server.BatchCompletionRequest(requests=[])))
    assert empty.value.status_code == 400

    item = {"messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(HTTPException) as oversized:
        asyncio.run(
            server.batch_completions(server.BatchCompletionRequest(requests=[item] * 65))
        )
    assert oversized.value.status_code == 413
