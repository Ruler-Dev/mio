"""Parse Qwen3.5 native tool_call XML into OpenAI tool_calls JSON.

Qwen3.5's chat template (when `tools=[...]` is passed) instructs the model to
reply in this format:

    <tool_call>
    <function=read>
    <parameter=filePath>
    C:\\Users\\...\\setup.bat
    </parameter>
    </function>
    </tool_call>

OpenAI / Kilo / most SDKs expect:

    {"id": "call_abc", "type": "function",
     "function": {"name": "read", "arguments": '{"filePath": "..."}'}}

This module converts the former to the latter, with a streaming state machine
that emits tool_call deltas as tokens arrive.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field


# Qwen3.5 is trained to emit <function=NAME><parameter=KEY>val</parameter></function>
# but on MLX Unsloth quants it frequently drops the "function=" / "parameter="
# prefixes and uses bare <NAME>...<KEY>val</KEY>...</NAME>. Both shapes show up
# in practice; we accept either.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<inner>.*?)\s*</tool_call>",
    re.DOTALL,
)

# Qwen native: <function=NAME>...<parameter=KEY>val</parameter>...</function>
_QWEN_FUNC_RE = re.compile(
    r"<function=([^>\s]+)>\s*(?P<body>.*?)\s*</function>",
    re.DOTALL,
)
_QWEN_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(?P<val>.*?)\s*</parameter>",
    re.DOTALL,
)

# Bare-tag variant: <NAME>...<KEY>val</KEY>...</NAME>
_BARE_FUNC_RE = re.compile(
    r"<([A-Za-z_][A-Za-z0-9_\-]*)>\s*(?P<body>.*?)\s*</\1>",
    re.DOTALL,
)
_BARE_PARAM_RE = re.compile(
    r"<([A-Za-z_][A-Za-z0-9_\-]*)>\s*(?P<val>.*?)\s*</\1>",
    re.DOTALL,
)


def _coerce_value(v: str) -> object:
    """Best-effort JSON-friendly coercion of a parameter string.

    Qwen emits every parameter as free-form text between <parameter> tags.
    OpenAI arguments is a JSON string — bool/int/float/null should round-trip
    as typed, but anything ambiguous stays a string (safer for paths, code).
    """
    s = v.strip()
    if s == "true":
        return True
    if s == "false":
        return False
    if s == "null":
        return None
    # Ints and floats
    if s and (s[0] in "-0123456789"):
        try:
            if "." in s or "e" in s or "E" in s:
                return float(s)
            return int(s)
        except ValueError:
            pass
    # JSON object or array — best effort
    if s.startswith(("{", "[")):
        try:
            return json.loads(s)
        except ValueError:
            pass
    return s


def _parse_call_inner(inner: str) -> tuple[str | None, dict]:
    """Parse the body of a single <tool_call>...</tool_call> block.

    Returns (function_name, args_dict). Tries Qwen's native
    <function=NAME><parameter=KEY>val</parameter></function> first, then
    falls back to the bare-tag variant <NAME><KEY>val</KEY></NAME>.
    """
    m = _QWEN_FUNC_RE.search(inner)
    if m:
        name = m.group(1)
        body = m.group("body") or ""
        args: dict = {}
        for p in _QWEN_PARAM_RE.finditer(body):
            args[p.group(1)] = _coerce_value(p.group("val"))
        return name, args

    # Bare tag: <NAME>...</NAME> wrapping <KEY>val</KEY> params.
    m = _BARE_FUNC_RE.search(inner)
    if m:
        name = m.group(1)
        body = m.group("body") or ""
        args = {}
        for p in _BARE_PARAM_RE.finditer(body):
            args[p.group(1)] = _coerce_value(p.group("val"))
        return name, args

    return None, {}


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract all <tool_call> blocks from `text`.

    Returns (leading_content, tool_calls) where leading_content is any text
    emitted before the first tool call and tool_calls is a list of OpenAI-
    format objects. Accepts both Qwen's native
    <function=NAME><parameter=KEY>val</parameter></function> form and the
    bare <NAME><KEY>val</KEY></NAME> variant (frequent on MLX Unsloth quants).
    """
    calls: list[dict] = []
    first_match_start: int | None = None
    for m in _TOOL_CALL_RE.finditer(text):
        if first_match_start is None:
            first_match_start = m.start()
        name, args = _parse_call_inner(m.group("inner") or "")
        if not name:
            continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    if first_match_start is None:
        return text, []
    leading = text[:first_match_start].rstrip()
    return leading, calls


@dataclass
class StreamingToolCallParser:
    """Incremental parser for SSE streaming.

    Feeds chunks; emits ordinary content when outside `<tool_call>`, and
    collects complete tool calls to be flushed after the stream ends.

    Design note: we do NOT stream partial tool_calls. Qwen emits every
    call atomically (usually one per turn), and OpenAI streaming clients
    tolerate tool_calls arriving only in the final delta + finish_reason.
    This keeps the state machine simple and correct for the common case.
    """

    _buf: str = ""
    _inside_call: bool = False
    _call_text: str = ""
    _complete_calls: list[dict] = field(default_factory=list)

    def feed(self, chunk: str) -> str:
        """Return any content that is NOT inside a tool call."""
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._inside_call:
                end_idx = self._buf.find("</tool_call>")
                if end_idx < 0:
                    # Keep buffering until we see the closing tag.
                    break
                self._call_text += self._buf[: end_idx + len("</tool_call>")]
                self._buf = self._buf[end_idx + len("</tool_call>"):]
                # Parse and stash. One call per block by Qwen convention.
                _, calls = parse_tool_calls(self._call_text)
                self._complete_calls.extend(calls)
                self._call_text = ""
                self._inside_call = False
            else:
                start_idx = self._buf.find("<tool_call>")
                if start_idx < 0:
                    # Might be forming a partial "<tool_call" — hold the tail.
                    tag = "<tool_call>"
                    hold = 0
                    max_hold = min(len(self._buf), len(tag) - 1)
                    for k in range(max_hold, 0, -1):
                        if tag.startswith(self._buf[-k:]):
                            hold = k
                            break
                    if hold:
                        out.append(self._buf[:-hold])
                        self._buf = self._buf[-hold:]
                    else:
                        out.append(self._buf)
                        self._buf = ""
                    break
                if start_idx > 0:
                    out.append(self._buf[:start_idx])
                self._buf = self._buf[start_idx + len("<tool_call>"):]
                self._call_text = "<tool_call>"
                self._inside_call = True
        return "".join(out)

    def flush(self) -> tuple[str, list[dict]]:
        """Emit any remaining safe content + all completed tool calls."""
        tail = ""
        if not self._inside_call and self._buf:
            tail = self._buf
        # Drop any incomplete tool_call that never closed.
        self._buf = ""
        self._call_text = ""
        self._inside_call = False
        calls = self._complete_calls
        self._complete_calls = []
        return tail, calls
