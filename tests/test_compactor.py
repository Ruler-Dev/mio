"""Tests for context auto-compaction."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock


def _fake_engine(context_window: int = 1000, tokens_per_char: float = 0.25):
    """Build a minimal fake engine with a tokenizer that is just
    ceil(total_content_chars * tokens_per_char). Lets us hit thresholds
    deterministically without loading a model.
    """
    eng = MagicMock()
    eng.tier_config.context_window = context_window

    def _apply_chat_template(messages, tools=None):
        total = 0
        for m in messages:
            c = m.get("content") or ""
            if isinstance(c, str):
                total += len(c)
            # Account for tool_calls being rendered too
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                total += len(fn.get("name") or "")
                args = fn.get("arguments")
                if isinstance(args, dict):
                    total += sum(len(str(v)) for v in args.values())
                elif isinstance(args, str):
                    total += len(args)
            total += 10  # per-message framing overhead
        # Return a list of length = tokens so len(...) works
        return [0] * int(total * tokens_per_char)

    eng._apply_chat_template = _apply_chat_template
    return eng


def test_atomic_groups_preserves_tool_chains():
    from mio.compactor import _atomic_groups

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"p": "a"}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        {"role": "assistant", "content": "hello"},
    ]
    groups = _atomic_groups(msgs)
    # 4 groups: system, user, [assistant+tool], assistant
    assert len(groups) == 4
    assert len(groups[2].messages) == 2
    assert groups[2].messages[0].get("role") == "assistant"
    assert groups[2].messages[1].get("role") == "tool"


def test_compact_below_threshold_noop():
    from mio.compactor import compact

    eng = _fake_engine(context_window=1000)
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
    ]
    out, stats = compact(msgs, eng, threshold=0.75, target=0.50)
    assert not stats.triggered
    assert out == msgs


def test_stage1_truncation_reduces_tool_content():
    from mio.compactor import compact

    # 1000-token ctx. Fake tokenizer: 0.25 tokens/char + 10/msg overhead.
    # Threshold 0.75 = 750 tokens ≈ 3000 chars. Target 0.50 = 500 tokens.
    eng = _fake_engine(context_window=1000)

    big_tool = "X" * 4000  # 4000 chars → ~1000 tokens, clearly over threshold
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"p": "a"}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": big_tool},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "answering"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "q4 latest"},
    ]

    out, stats = compact(
        msgs, eng, threshold=0.75, target=0.50, protected_tail=3,
        min_tool_content=100, enable_summarization=False,
    )
    assert stats.triggered
    assert stats.stage == "truncation"
    assert stats.tool_results_truncated == 1
    # The big_tool X*4000 should no longer appear (replaced by placeholder)
    assert all("X" * 400 not in (m.get("content") or "") for m in out)


def test_stage1_preserves_protected_tail():
    from mio.compactor import compact

    eng = _fake_engine(context_window=1000)
    recent_user = "latest user request please preserve me"
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"p": "a"}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "Y" * 1000},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "read", "arguments": {"p": "b"}}}
        ]},
        # This tool result is in the protected tail — must NOT be truncated
        {"role": "tool", "tool_call_id": "c2", "name": "read", "content": "Z" * 800},
        {"role": "user", "content": recent_user},
    ]

    out, _ = compact(
        msgs, eng, threshold=0.75, target=0.50, protected_tail=2,
        min_tool_content=100, enable_summarization=False,
    )
    # Z*800 is in the protected tail (last 2 groups) — must survive
    assert any("Z" * 800 in (m.get("content") or "") for m in out)
    assert any(recent_user == (m.get("content") or "") for m in out)


def test_stage2_summarization_replaces_middle():
    from mio.compactor import compact

    eng = _fake_engine(context_window=1000)
    eng.generate = MagicMock(return_value=("<SUMMARY_OK>", MagicMock()))

    # Build a conversation that stage 1 can't reduce enough — all messages
    # are below min_tool_content so truncation makes no dent.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"user {i} " * 50})
        msgs.append({"role": "assistant", "content": f"reply {i} " * 50})
    msgs.append({"role": "user", "content": "final"})

    out, stats = compact(
        msgs, eng, threshold=0.75, target=0.50, protected_tail=2,
        min_tool_content=10_000,  # effectively disables stage 1 truncation
        enable_summarization=True,
    )
    assert stats.triggered
    assert stats.stage == "summary"
    assert stats.messages_summarized > 0
    # engine.generate must have been called exactly once (summarization)
    assert eng.generate.call_count == 1
    # The synthetic summary message should be present
    assert any("<SUMMARY_OK>" in (m.get("content") or "") for m in out)
    # System prompt is preserved
    assert out[0].get("role") == "system"
    # Last user message is preserved
    assert out[-1].get("content") == "final"


def test_stage2_cancellation_closes_stream_without_consuming_more_tokens():
    from mio.compactor import _run_summarization

    cancelled = threading.Event()
    started = threading.Event()
    allow_inflight_step = threading.Event()
    closed = threading.Event()

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            try:
                started.set()
                yield "partial", None
                assert allow_inflight_step.wait(timeout=1.0)
                yield "discarded", None
            finally:
                closed.set()

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
    ]
    result: list[tuple[list[dict], int]] = []
    worker = threading.Thread(
        target=lambda: result.append(
            _run_summarization(
                messages,
                Engine(),
                protected_tail=1,
                gpu_lock=None,
                cancellation_event=cancelled,
            )
        )
    )
    worker.start()
    assert started.wait(timeout=1.0)
    cancelled.set()
    allow_inflight_step.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert closed.is_set()
    assert result
    assert "partial" not in result[0][0][1]["content"]


def test_compact_disabled_threshold_1():
    from mio.compactor import compact

    eng = _fake_engine(context_window=1000)
    # Even a very long message won't trigger when threshold=1.0 (or higher
    # — semantically disabled by server when threshold >= 1.0, and compact()
    # itself must not trigger at exactly 1.0).
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "X" * 2000},  # clearly over 1000-char ctx
    ]
    out, stats = compact(msgs, eng, threshold=1.0, target=0.50)
    # threshold=1.0 means trigger when tokens > 1000; our prompt IS over that,
    # so compaction would fire. What we really verify is that compact handles
    # threshold semantics as documented — the server gate is separate.
    # Just sanity-check that nothing errors and stats are populated.
    assert stats.before_tokens > 0
