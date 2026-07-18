#!/usr/bin/env python3
"""Run MioCodeBench v1 smoke/development arms with the native Mio agent.

This module contains only the non-confirmatory 4-task smoke and 8-task
development corpus.  Public files are materialized by ``bench_coding_quality``;
private evaluators stay in the host process and are never copied under an agent
workspace root.  The command writes only the source-free aggregate schema.

The MLX stack is imported lazily.  Unit tests can inject a callback runner and
exercise every protocol boundary without loading a model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from scripts.bench_coding_quality import (
    GATE_OFF,
    GATE_ON,
    BenchmarkExecution,
    CodingFixture,
    EvaluationRequest,
    GenerationObservation,
    GenerationRequest,
    HiddenEvaluation,
    Preregistration,
    PublicFile,
    fixture_suite_sha256,
    run_benchmark,
    serialize_source_free_aggregate,
)


_PUBLIC_TOOL_NAMES = ("bash", "read", "write", "edit")
_GATE_TOOL_NAMES = (*_PUBLIC_TOOL_NAMES, "validate")


@dataclass(frozen=True)
class HiddenOracle:
    """Evaluator programs that never enter an agent-visible workspace."""

    public_regression: str
    hidden_checks: str


@dataclass(frozen=True)
class CorpusCase:
    split: str
    fixture: CodingFixture
    oracle: HiddenOracle
    editable_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split not in {"smoke", "development"}:
            raise ValueError("corpus split must be smoke or development")
        public_names = {item.relative_name for item in self.fixture.public_files}
        if not self.editable_names or not set(self.editable_names) <= public_names:
            raise ValueError("editable names must be a non-empty subset of public files")
        compile(self.oracle.public_regression, f"<{self.fixture.fixture_id}-public>", "exec")
        compile(self.oracle.hidden_checks, f"<{self.fixture.fixture_id}-hidden>", "exec")


def _public_test(module: str, body: str) -> str:
    return f"""import unittest

from {module} import *


class PublicContractTests(unittest.TestCase):
{body}


if __name__ == "__main__":
    unittest.main()
"""


def _case(
    *,
    split: str,
    fixture_id: str,
    instruction: str,
    module: str,
    source: str,
    public_test_body: str,
    public_regression: str,
    hidden_checks: str,
) -> CorpusCase:
    return CorpusCase(
        split=split,
        fixture=CodingFixture(
            fixture_id=fixture_id,
            instruction=(
                f"{instruction} Work only in this workspace. Preserve the public API and use only "
                "the Python standard library. Do not edit the public test or create extra files. "
                "Before finishing run: python3 -B -m unittest discover -s . -p test_public_*.py"
            ),
            public_files=(
                PublicFile(relative_name=f"{module}.py", content=source),
                PublicFile(
                    relative_name=f"test_public_{module}.py",
                    content=_public_test(module, public_test_body),
                ),
            ),
        ),
        oracle=HiddenOracle(public_regression=public_regression, hidden_checks=hidden_checks),
        editable_names=(f"{module}.py",),
    )


CORPUS: tuple[CorpusCase, ...] = (
    _case(
        split="smoke",
        fixture_id="s01",
        instruction="Implement normalize_whitespace(text) in text_utils.py.",
        module="text_utils",
        source='''"""Small text normalization helpers."""\n\n\ndef normalize_whitespace(text):\n    """Collapse Unicode whitespace runs and strip the ends."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_words_and_newline(self):
        self.assertEqual(normalize_whitespace("  hello   local\\nworld  "), "hello local world")

    def test_empty(self):
        self.assertEqual(normalize_whitespace("   "), "")""",
        public_regression="""from text_utils import normalize_whitespace
assert normalize_whitespace("  hello   local\\nworld  ") == "hello local world"
assert normalize_whitespace("   ") == ""
""",
        hidden_checks="""from text_utils import normalize_whitespace
assert normalize_whitespace("a\\tb\\r\\nc") == "a b c"
assert normalize_whitespace("\u2003alpha\u00a0beta\u2009") == "alpha beta"
""",
    ),
    _case(
        split="smoke",
        fixture_id="s02",
        instruction="Implement clamp(value, lower, upper) in math_utils.py; reject an inverted interval.",
        module="math_utils",
        source='''"""Numeric boundary helpers."""\n\n\ndef clamp(value, lower, upper):\n    """Return value constrained to the inclusive [lower, upper] interval."""\n    return value\n''',
        public_test_body="""    def test_inside_and_edges(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(12, 0, 10), 10)""",
        public_regression="""from math_utils import clamp
assert clamp(5, 0, 10) == 5
assert clamp(-1, 0, 10) == 0
assert clamp(12, 0, 10) == 10
""",
        hidden_checks="""from math_utils import clamp
assert clamp(0.25, 0.5, 1.0) == 0.5
assert clamp(1.0, 1.0, 1.0) == 1.0
try:
    clamp(1, 4, 2)
except ValueError:
    pass
else:
    raise AssertionError("inverted intervals must fail")
""",
    ),
    _case(
        split="smoke",
        fixture_id="s03",
        instruction="Implement batched(iterable, size) in iter_utils.py as a lazy iterator of tuples.",
        module="iter_utils",
        source='''"""Iterator helpers."""\n\n\ndef batched(iterable, size):\n    """Yield tuples of at most size items without materializing iterable."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_full_and_partial_batches(self):
        self.assertEqual(list(batched([1, 2, 3, 4, 5], 2)), [(1, 2), (3, 4), (5,)])""",
        public_regression="""from iter_utils import batched
assert list(batched([1, 2, 3, 4, 5], 2)) == [(1, 2), (3, 4), (5,)]
""",
        hidden_checks="""from iter_utils import batched
assert list(batched((value for value in range(4)), 3)) == [(0, 1, 2), (3,)]
assert list(batched([], 2)) == []
try:
    list(batched([1], 0))
except ValueError:
    pass
else:
    raise AssertionError("non-positive size must fail")
""",
    ),
    _case(
        split="smoke",
        fixture_id="s04",
        instruction="Implement deep_get(mapping, dotted_path, default=None) in mapping_utils.py.",
        module="mapping_utils",
        source='''"""Nested mapping helpers."""\n\n\ndef deep_get(mapping, dotted_path, default=None):\n    """Follow dot-separated mapping keys; return default when traversal fails."""\n    return default\n''',
        public_test_body="""    def test_present_and_missing(self):
        payload = {"user": {"profile": {"name": "Mio"}}}
        self.assertEqual(deep_get(payload, "user.profile.name"), "Mio")
        self.assertEqual(deep_get(payload, "user.id", 7), 7)""",
        public_regression="""from mapping_utils import deep_get
payload = {"user": {"profile": {"name": "Mio"}}}
assert deep_get(payload, "user.profile.name") == "Mio"
assert deep_get(payload, "user.id", 7) == 7
""",
        hidden_checks="""from mapping_utils import deep_get
marker = object()
assert deep_get({"a": {"b": None}}, "a.b", marker) is None
assert deep_get({"a": 3}, "a.b", marker) is marker
assert deep_get({"": {"x": 1}}, ".x", marker) == 1
""",
    ),
    _case(
        split="development",
        fixture_id="d01",
        instruction=(
            "Implement backoff_delays(attempts, base=1.0, factor=2.0, cap=None) in retry.py "
            "with validation and per-delay capping."
        ),
        module="retry",
        source='''"""Retry scheduling utilities."""\n\n\ndef backoff_delays(attempts, base=1.0, factor=2.0, cap=None):\n    """Return deterministic exponential delays for attempts after the first call."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_exponential_and_cap(self):
        self.assertEqual(backoff_delays(4), [1.0, 2.0, 4.0, 8.0])
        self.assertEqual(backoff_delays(4, cap=3.0), [1.0, 2.0, 3.0, 3.0])""",
        public_regression="""from retry import backoff_delays
assert backoff_delays(4) == [1.0, 2.0, 4.0, 8.0]
assert backoff_delays(4, cap=3.0) == [1.0, 2.0, 3.0, 3.0]
""",
        hidden_checks="""from retry import backoff_delays
assert backoff_delays(0) == []
assert backoff_delays(3, base=0.5, factor=3, cap=2) == [0.5, 1.5, 2]
for args in [(-1,), (2, -1), (2, 1, 0), (2, 1, 2, -1)]:
    try:
        backoff_delays(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(args)
""",
    ),
    _case(
        split="development",
        fixture_id="d02",
        instruction=(
            "Implement topological_sort(graph) in graph.py. Include neighbor-only nodes, use lexical "
            "tie-breaking, do not mutate input, and reject cycles."
        ),
        module="graph",
        source='''"""Deterministic directed-graph algorithms."""\n\n\ndef topological_sort(graph):\n    """Return a deterministic ordering for a mapping of node to dependencies."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_dependency_order(self):
        graph = {"deploy": {"test"}, "test": {"build"}, "build": set()}
        self.assertEqual(topological_sort(graph), ["build", "test", "deploy"])""",
        public_regression="""from graph import topological_sort
graph = {"deploy": {"test"}, "test": {"build"}, "build": set()}
assert topological_sort(graph) == ["build", "test", "deploy"]
""",
        hidden_checks="""from graph import topological_sort
graph = {"z": {"a"}, "b": set()}
snapshot = {key: set(value) for key, value in graph.items()}
assert topological_sort(graph) == ["a", "b", "z"]
assert graph == snapshot
try:
    topological_sort({"a": {"b"}, "b": {"a"}})
except ValueError:
    pass
else:
    raise AssertionError("cycle must fail")
""",
    ),
    _case(
        split="development",
        fixture_id="d03",
        instruction="Implement the bounded LRUCache API in cache.py using Python standard-library data structures.",
        module="cache",
        source='''"""Small in-memory caches."""\n\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n\n    def get(self, key, default=None):\n        return default\n\n    def put(self, key, value):\n        pass\n\n    def __len__(self):\n        return 0\n''',
        public_test_body="""    def test_eviction(self):
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        cache.put("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(len(cache), 2)""",
        public_regression="""from cache import LRUCache
cache = LRUCache(2)
cache.put("a", 1); cache.put("b", 2)
assert cache.get("a") == 1
cache.put("c", 3)
assert cache.get("b") is None and len(cache) == 2
""",
        hidden_checks="""from cache import LRUCache
cache = LRUCache(2)
cache.put("a", 1); cache.put("b", 2); cache.put("a", 9); cache.put("c", 3)
assert cache.get("a") == 9 and cache.get("b", "missing") == "missing"
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError("zero capacity must fail")
""",
    ),
    _case(
        split="development",
        fixture_id="d04",
        instruction=(
            "Implement redact_secrets(value, keys, replacement='[REDACTED]') in redaction.py. "
            "Recursively copy dict/list/tuple containers and match dictionary keys case-insensitively."
        ),
        module="redaction",
        source='''"""Structured-data redaction."""\n\n\ndef redact_secrets(value, keys, replacement="[REDACTED]"):\n    """Return a redacted deep copy of built-in containers."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_nested_mapping(self):
        value = {"token": "abc", "nested": [{"PASSWORD": "xyz", "ok": 1}]}
        expected = {"token": "***", "nested": [{"PASSWORD": "***", "ok": 1}]}
        self.assertEqual(redact_secrets(value, {"token", "password"}, "***"), expected)""",
        public_regression="""from redaction import redact_secrets
value = {"token": "abc", "nested": [{"PASSWORD": "xyz", "ok": 1}]}
expected = {"token": "***", "nested": [{"PASSWORD": "***", "ok": 1}]}
assert redact_secrets(value, {"token", "password"}, "***") == expected
""",
        hidden_checks="""from redaction import redact_secrets
source = {"Auth": ("keep", {"secret": 4}), "items": [1, 2]}
result = redact_secrets(source, {"auth", "secret"})
assert result == {"Auth": "[REDACTED]", "items": [1, 2]}
assert source == {"Auth": ("keep", {"secret": 4}), "items": [1, 2]}
assert result is not source and result["items"] is not source["items"]
""",
    ),
    _case(
        split="development",
        fixture_id="d05",
        instruction=(
            "Implement group_totals(rows, group_key, value_key) in records.py using Decimal for exact "
            "accumulation; accept int/float/str/Decimal values and return Decimal totals."
        ),
        module="records",
        source='''"""Tabular record aggregation."""\n\n\ndef group_totals(rows, group_key, value_key):\n    """Aggregate numeric record values without binary floating-point drift."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_grouping(self):
        from decimal import Decimal
        rows = [{"team": "a", "amount": "0.1"}, {"team": "a", "amount": 0.2}, {"team": "b", "amount": 2}]
        self.assertEqual(group_totals(rows, "team", "amount"), {"a": Decimal("0.3"), "b": Decimal("2")})""",
        public_regression="""from decimal import Decimal
from records import group_totals
rows = [{"team": "a", "amount": "0.1"}, {"team": "a", "amount": 0.2}, {"team": "b", "amount": 2}]
assert group_totals(rows, "team", "amount") == {"a": Decimal("0.3"), "b": Decimal("2")}
""",
        hidden_checks="""from decimal import Decimal
from records import group_totals
assert group_totals([], "k", "v") == {}
assert group_totals([{"k": None, "v": Decimal("1.25")}, {"k": None, "v": "2.75"}], "k", "v") == {None: Decimal("4.00")}
rows = ({"k": index % 2, "v": index} for index in range(4))
assert group_totals(rows, "k", "v") == {0: Decimal("2"), 1: Decimal("4")}
""",
    ),
    _case(
        split="development",
        fixture_id="d06",
        instruction=(
            "Implement canonicalize_url(url) in urls.py: lowercase scheme/host, remove fragments and "
            "default ports, normalize an empty path to '/', and sort query pairs while preserving duplicates/blanks."
        ),
        module="urls",
        source='''"""Stable URL canonicalization."""\n\n\ndef canonicalize_url(url):\n    """Return a conservative canonical representation of an HTTP(S) URL."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_basic_http_url(self):
        value = "HTTPS://Example.COM:443/path?z=2&a=1#section"
        self.assertEqual(canonicalize_url(value), "https://example.com/path?a=1&z=2")""",
        public_regression="""from urls import canonicalize_url
value = "HTTPS://Example.COM:443/path?z=2&a=1#section"
assert canonicalize_url(value) == "https://example.com/path?a=1&z=2"
""",
        hidden_checks="""from urls import canonicalize_url
assert canonicalize_url("http://EXAMPLE.com:80") == "http://example.com/"
assert canonicalize_url("https://example.com/?b=&a=2&a=1") == "https://example.com/?a=1&a=2&b="
assert canonicalize_url("https://user:pass@EXAMPLE.com:444/x") == "https://user:pass@example.com:444/x"
""",
    ),
    _case(
        split="development",
        fixture_id="d07",
        instruction=(
            "Implement rolling_mean(values, window) in window.py as a single-pass function returning floats; "
            "support generators, reject non-positive windows, and return [] when the window is too large."
        ),
        module="window",
        source='''"""Streaming numeric windows."""\n\n\ndef rolling_mean(values, window):\n    """Return the arithmetic mean of each complete consecutive window."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_three_value_window(self):
        self.assertEqual(rolling_mean([1, 2, 3, 6], 3), [2.0, 11 / 3])""",
        public_regression="""from window import rolling_mean
assert rolling_mean([1, 2, 3, 6], 3) == [2.0, 11 / 3]
""",
        hidden_checks="""from window import rolling_mean
assert rolling_mean((value for value in [2, 4, 8]), 1) == [2.0, 4.0, 8.0]
assert rolling_mean([1, 2], 3) == []
try:
    rolling_mean([1], 0)
except ValueError:
    pass
else:
    raise AssertionError("invalid window")
""",
    ),
    _case(
        split="development",
        fixture_id="d08",
        instruction=(
            "Implement deduplicate_events(events) in events.py. Keep first-seen id order but retain the last "
            "full event for each id, accept any iterable, copy outputs, and reject events missing id."
        ),
        module="events",
        source='''"""Event-stream normalization."""\n\n\ndef deduplicate_events(events):\n    """Deduplicate mapping events by id without mutating input mappings."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_last_value_first_order(self):
        events = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "a", "v": 3}]
        self.assertEqual(deduplicate_events(events), [{"id": "a", "v": 3}, {"id": "b", "v": 2}])""",
        public_regression="""from events import deduplicate_events
events = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "a", "v": 3}]
assert deduplicate_events(events) == [{"id": "a", "v": 3}, {"id": "b", "v": 2}]
""",
        hidden_checks="""from events import deduplicate_events
source = [{"id": 0, "v": []}, {"id": 0, "v": [1]}]
result = deduplicate_events(iter(source))
assert result == [{"id": 0, "v": [1]}]
assert result[0] is not source[1]
try:
    deduplicate_events([{"value": 1}])
except (KeyError, ValueError):
    pass
else:
    raise AssertionError("missing id must fail")
""",
    ),
)


# Explicit seals make corpus edits visible during review.  Update them only as
# a preregistered protocol revision, never in response to benchmark outcomes.
SMOKE_SUITE_SHA256 = "d0fef6c7ccfcccbf6dcbc70f973d931f5dba023f45f4335482d72f626c824afc"
DEVELOPMENT_SUITE_SHA256 = "3b9cd3611486e5b3d20a5786249fdc9446af3aecf3a648d507f46a5f5c3208e5"
ALL_SUITE_SHA256 = "32f4a59ab1831b5130fccbcdc3a9affcfe5e03204f7de0423aec106d4251857c"


def select_cases(split: str) -> tuple[CorpusCase, ...]:
    if split == "all":
        return CORPUS
    if split not in {"smoke", "development"}:
        raise ValueError("split must be smoke, development, or all")
    return tuple(case for case in CORPUS if case.split == split)


def sealed_suite_sha256(cases: Sequence[CorpusCase]) -> str:
    """Return the explicit split seal and reject a silently changed corpus."""

    identifiers = tuple(case.fixture.fixture_id for case in cases)
    seals = {
        tuple(case.fixture.fixture_id for case in select_cases("smoke")): SMOKE_SUITE_SHA256,
        tuple(case.fixture.fixture_id for case in select_cases("development")): DEVELOPMENT_SUITE_SHA256,
        tuple(case.fixture.fixture_id for case in select_cases("all")): ALL_SUITE_SHA256,
    }
    expected = seals.get(identifiers)
    if expected is None:
        raise ValueError("cases must be one complete frozen MioCodeBench split")
    actual = fixture_suite_sha256(tuple(case.fixture for case in cases))
    if actual != expected:
        raise RuntimeError("MioCodeBench corpus no longer matches its explicit suite seal")
    return expected


def fixture_tree_sha256(fixture: CodingFixture) -> str:
    digest = hashlib.sha256()
    for public_file in sorted(fixture.public_files, key=lambda item: item.relative_name):
        encoded_name = public_file.relative_name.encode("utf-8")
        encoded_content = public_file.content.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(b"F")
        digest.update(len(encoded_content).to_bytes(8, "big"))
        digest.update(encoded_content)
    return digest.hexdigest()


def workspace_tree_sha256(workspace: Path) -> str:
    digest = hashlib.sha256()
    root = workspace.resolve()
    entries = sorted(root.rglob("*"))
    for path in entries:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"L")
        elif path.is_dir():
            digest.update(b"D")
        elif path.is_file():
            content = path.read_bytes()
            digest.update(b"F")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        else:
            digest.update(b"O")
    return digest.hexdigest()


def build_agent_tool_surface(
    condition: str, agent_module: Any | None = None
) -> tuple[Mapping[str, Any], tuple[dict, ...]]:
    """Return the frozen benchmark-only tool registry for one arm."""

    if condition not in {GATE_OFF, GATE_ON}:
        raise ValueError("unknown benchmark condition")
    if agent_module is None:
        from mio import agent as agent_module

    names = _PUBLIC_TOOL_NAMES if condition == GATE_OFF else _GATE_TOOL_NAMES
    registry = MappingProxyType({name: agent_module.AGENT_TOOLS[name] for name in names})
    specs_by_name = {
        spec["function"]["name"]: spec
        for spec in agent_module.AGENT_TOOLS_SPEC
        if isinstance(spec, dict) and isinstance(spec.get("function"), dict)
    }
    specs = tuple(specs_by_name[name] for name in names)
    return registry, specs


def agent_turn_to_observation(result: Any, condition: str) -> GenerationObservation:
    """Adapt a content-free AgentTurnResult to the benchmark runner contract."""

    if condition not in {GATE_OFF, GATE_ON}:
        raise ValueError("unknown benchmark condition")
    rounds = tuple(getattr(result, "rounds", ()) or ())
    events = tuple(getattr(result, "tool_events", ()) or ())
    mutations = [
        event
        for event in events
        if getattr(event, "operation", "") in {"write", "edit"}
        and bool(getattr(event, "allowed", False))
        and getattr(event, "outcome", "") == "ok"
    ]
    validations = [event for event in events if getattr(event, "operation", "") == "validate"]
    gate_record = getattr(result, "quality_gate", None)
    terminal_complete = getattr(result, "terminal_reason", "") == "model_final"
    if condition == GATE_ON:
        if not isinstance(gate_record, Mapping):
            terminal_complete = False
        else:
            decision = gate_record.get("decision", gate_record.get("status"))
            terminal_complete = terminal_complete and decision in {
                "satisfied",
                "not_applicable",
                "complete",
                "pass",
            }

    return GenerationObservation(
        completed=bool(terminal_complete),
        mutation_count=len(mutations),
        tool_calls=int(getattr(result, "tool_calls", len(events)) or 0),
        output_tokens=sum(int(getattr(item, "completion_tokens", 0) or 0) for item in rounds),
        validation_attempted=bool(validations),
        validation_succeeded=any(
            bool(getattr(event, "allowed", False)) and getattr(event, "outcome", "") == "ok" for event in validations
        ),
        model_seconds=sum(float(getattr(item, "total_time_s", 0.0) or 0.0) for item in rounds),
        wall_seconds=float(getattr(result, "wall_time_s", 0.0) or 0.0),
    )


class AgentTurnExecutor(Protocol):
    def __call__(
        self,
        *,
        request: GenerationRequest,
        tool_registry: Mapping[str, Any],
        tool_specs: Sequence[dict],
        tool_policy: Any,
        quality_gate_enabled: bool,
        effort: str,
    ) -> Any: ...


class NativeAgentTurnExecutor:
    """One loaded Mio engine with fresh conversation and cache state per arm."""

    def __init__(self, *, config: Any, manager: Any, engine: Any, tier: str) -> None:
        self.config = config
        self.manager = manager
        self.engine = engine
        self.tier = tier

    def _reset_engine_state(self) -> None:
        invalidator = getattr(self.engine, "_prefix_cache_invalidate", None)
        if callable(invalidator):
            invalidator()
        if hasattr(self.engine, "_last_prompt_tokens"):
            self.engine._last_prompt_tokens = []
        if hasattr(self.engine, "_pending_assistant_prefill"):
            self.engine._pending_assistant_prefill = ""
        dspark = getattr(self.engine, "_dspark_runtime", None)
        prefix_cache = getattr(dspark, "_prefix_cache", None)
        executor = getattr(dspark, "_executor", None)
        if prefix_cache is not None and executor is not None:
            executor.submit(prefix_cache.reset).result()

    def __call__(
        self,
        *,
        request: GenerationRequest,
        tool_registry: Mapping[str, Any],
        tool_specs: Sequence[dict],
        tool_policy: Any,
        quality_gate_enabled: bool,
        effort: str,
    ) -> Any:
        from mio import agent
        from mio.prompt_policy import PromptPolicy

        self._reset_engine_state()
        state = {
            "tier": self.tier,
            "prompt_policy": PromptPolicy(),
            "tool_policy": tool_policy,
            "tool_registry": tool_registry,
            "tool_specs": tuple(tool_specs),
            "messages": [],
            "quality_gate_enabled": quality_gate_enabled,
            "coding_effort": effort,
        }
        previous_console = agent.console
        try:
            from rich.console import Console

            agent.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
            return agent._process_user_input(
                request.instruction,
                self.engine,
                self.manager,
                self.config,
                state,
            )
        finally:
            agent.console = previous_console


class RealMioGenerationRunner:
    """Benchmark callback enforcing identical bytes and a network-free policy."""

    def __init__(
        self,
        *,
        executor: AgentTurnExecutor,
        fixtures: Sequence[CodingFixture],
        effort: str,
        agent_module: Any | None = None,
    ) -> None:
        self.executor = executor
        self.effort = effort
        self.agent_module = agent_module
        self._initial_digests = {fixture.fixture_id: fixture_tree_sha256(fixture) for fixture in fixtures}

    def __call__(self, request: GenerationRequest) -> GenerationObservation:
        expected = self._initial_digests.get(request.fixture_id)
        if expected is None or workspace_tree_sha256(request.workspace) != expected:
            raise RuntimeError("agent workspace does not match the frozen initial fixture bytes")

        from mio.agent_policy import AgentToolPermission, AgentToolPolicy

        policy = AgentToolPolicy.coding_workspace(request.workspace, allow_network=False)
        if AgentToolPermission.NETWORK in policy.permissions:
            raise RuntimeError("coding benchmark policy unexpectedly grants network access")
        registry, specs = build_agent_tool_surface(request.condition, self.agent_module)
        result = self.executor(
            request=request,
            tool_registry=registry,
            tool_specs=specs,
            tool_policy=policy,
            quality_gate_enabled=request.condition == GATE_ON,
            effort=self.effort,
        )
        return agent_turn_to_observation(result, request.condition)


class CorpusHiddenEvaluator:
    """Run pristine public and private oracles after all agent arms are sealed."""

    def __init__(self, cases: Sequence[CorpusCase], *, timeout_s: float = 5.0) -> None:
        self._cases = {case.fixture.fixture_id: case for case in cases}
        self.timeout_s = timeout_s

    @staticmethod
    def _scope_is_valid(case: CorpusCase, workspace: Path) -> bool:
        root = workspace.resolve()
        initial = {item.relative_name: item.content.encode("utf-8") for item in case.fixture.public_files}
        entries = list(root.rglob("*"))
        if any(path.is_symlink() or not path.is_file() for path in entries):
            return False
        observed_names = {path.relative_to(root).as_posix() for path in entries}
        if observed_names != set(initial):
            return False

        edited = False
        editable = set(case.editable_names)
        for relative_name, original in initial.items():
            current = (root / relative_name).read_bytes()
            if relative_name in editable:
                edited = edited or current != original
            elif current != original:
                return False
        return edited

    def _run_oracle(self, workspace: Path, source: str) -> bool:
        from mio.agent_policy import AgentToolPolicy, sandboxed_command

        bootstrap = "import sys; sys.path.insert(0, '.')\n" + source
        argv = [sys.executable, "-I", "-B", "-c", bootstrap]
        policy = AgentToolPolicy.read_only(workspace)
        sandboxed_argv, environment = sandboxed_command(
            argv,
            policy,
            allow_process_fork=False,
        )
        environment = dict(environment)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                sandboxed_argv,
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def __call__(self, request: EvaluationRequest) -> HiddenEvaluation:
        case = self._cases.get(request.fixture_id)
        if case is None:
            raise RuntimeError("hidden oracle is missing for a corpus fixture")
        scope_valid = self._scope_is_valid(case, request.workspace)
        regression_free = self._run_oracle(request.workspace, case.oracle.public_regression)
        hidden_passed = self._run_oracle(request.workspace, case.oracle.hidden_checks)
        # ``passed`` is the preregistered composite primary outcome, not merely
        # the private assertion bit.
        return HiddenEvaluation(
            passed=scope_valid and regression_free and hidden_passed,
            regression_free=regression_free,
        )


def execute_corpus(
    *,
    cases: Sequence[CorpusCase],
    runner: Callable[[GenerationRequest], GenerationObservation],
    work_root: Path,
    hidden_evaluator: Callable[[EvaluationRequest], HiddenEvaluation] | None = None,
    seed: int = 20260718,
) -> BenchmarkExecution:
    """Execute a non-confirmatory split through the shared two-phase harness."""

    if not cases:
        raise ValueError("at least one corpus case is required")
    fixtures = tuple(case.fixture for case in cases)
    return run_benchmark(
        fixtures=fixtures,
        preregistration=Preregistration(
            expected_suite_sha256=sealed_suite_sha256(cases),
            seed=seed,
            bootstrap_samples=10_000,
            alpha=0.05,
            minimum_pairs_for_claim=16,
        ),
        runner=runner,
        hidden_evaluator=hidden_evaluator or CorpusHiddenEvaluator(cases),
        work_root=work_root,
    )


def _load_native_executor(*, tier: str, config_path: Path | None) -> tuple[NativeAgentTurnExecutor, Any]:
    from mio.config import load_config
    from mio.model_manager import ModelManager

    config = load_config(config_path)
    if tier not in config.tiers:
        raise ValueError(f"unknown configured tier: {tier}")
    config.active_tiers = [tier]
    tier_config = config.tiers[tier]
    tier_config.temperature = 0.0
    tier_config.top_p = 1.0
    tier_config.top_k = 0
    # Both arms are cold by protocol. Disable DSpark's private prefix slots;
    # MioEngine's own cache is explicitly invalidated before every arm.
    tier_config.dspark_prefix_cache = False
    manager = ModelManager(config)
    manager.load_tier(tier)
    engine = manager.get_engine(tier)
    return NativeAgentTurnExecutor(config=config, manager=manager, engine=engine, tier=tier), manager


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["smoke", "development", "all"], default="smoke")
    parser.add_argument("--tier", default="small")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "ultra"], default="medium")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = select_cases(args.split)
    fixtures = tuple(case.fixture for case in cases)
    manager = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        # Model loaders are verbose; reserve stdout exclusively for the JSON
        # artifact and keep content-bearing agent output in an in-memory sink.
        with redirect_stdout(sys.stderr):
            executor, manager = _load_native_executor(tier=args.tier, config_path=args.config)
        runner = RealMioGenerationRunner(
            executor=executor,
            fixtures=fixtures,
            effort=args.effort,
        )
        if args.work_root is None:
            temporary = tempfile.TemporaryDirectory(prefix="mio-codebench-")
            work_root = Path(temporary.name)
        else:
            work_root = args.work_root
        with redirect_stdout(sys.stderr):
            execution = execute_corpus(cases=cases, runner=runner, work_root=work_root)
        serialized = serialize_source_free_aggregate(execution.aggregate)
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
    finally:
        if manager is not None:
            manager.unload_all()
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
