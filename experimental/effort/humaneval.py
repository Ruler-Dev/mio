"""Pinned HumanEval corpus and sandboxed verifier for effort R&D.

The model receives only the public prompt.  Evaluation tests remain outside
the generation context and execute in Mio's inherited macOS sandbox with no
network and no child-process creation.  This is an experimental evaluator,
not a claim that HumanEval alone measures repository-level coding quality.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shlex
import ssl
import stat
import subprocess
import sys
import tempfile
import time
from typing import Literal
from urllib.request import Request, urlopen

import certifi

from experimental.markov_effort_controller import ValidationOutcome
from mio.agent import (
    _BoundedCommandResult,
    _run_bounded_process,
    _shell_argv,
    _terminate_process_group,
)
from mio.agent_policy import (
    AgentToolPermission,
    AgentToolPolicy,
    sandboxed_command,
)


HUMANEVAL_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/"
    f"{HUMANEVAL_REVISION}/data/HumanEval.jsonl.gz"
)
SPLIT_SALT = "mio-humaneval-effort-v1"
CALIBRATION_TASKS = 32
_MAX_ARCHIVE_BYTES = 1_000_000
_MAX_DECOMPRESSED_BYTES = 8_000_000
_MAX_CASE_CHARS = 200_000
_CODE_FENCE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class HumanEvalError(ValueError):
    """Raised when the pinned corpus or a candidate is malformed."""


@dataclass(frozen=True)
class HumanEvalCase:
    task_id: str
    prompt: str
    test: str
    entry_point: str

    def __post_init__(self) -> None:
        fields = (self.task_id, self.prompt, self.test, self.entry_point)
        if not all(isinstance(value, str) and value for value in fields):
            raise HumanEvalError("HumanEval fields must be non-empty strings")
        if sum(len(value) for value in fields) > _MAX_CASE_CHARS:
            raise HumanEvalError("HumanEval case exceeds the size limit")
        if not self.entry_point.isidentifier():
            raise HumanEvalError("HumanEval entry point is not an identifier")

    @property
    def public(self) -> PublicHumanEvalCase:
        """Return the only task view permitted in generation and routing."""

        return PublicHumanEvalCase(
            task_id=self.task_id,
            prompt=self.prompt,
            entry_point=self.entry_point,
        )


@dataclass(frozen=True)
class PublicHumanEvalCase:
    """HumanEval data visible to prompts, validators, and the controller."""

    task_id: str
    prompt: str
    entry_point: str

    def __post_init__(self) -> None:
        fields = (self.task_id, self.prompt, self.entry_point)
        if not all(isinstance(value, str) and value for value in fields):
            raise HumanEvalError("public HumanEval fields must be non-empty strings")
        if len(self.prompt) > _MAX_CASE_CHARS:
            raise HumanEvalError("public HumanEval case exceeds the size limit")
        if not self.entry_point.isidentifier():
            raise HumanEvalError("HumanEval entry point is not an identifier")


@dataclass(frozen=True)
class PreparedCandidate:
    completion: str
    source: str
    source_sha256: str


@dataclass(frozen=True)
class PublicValidationResult:
    """Controller-visible evidence that contains no hidden-test outcome."""

    outcome: ValidationOutcome
    status: str
    feedback: str
    elapsed_seconds: float
    source_sha256: str


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    status: str
    feedback: str
    elapsed_seconds: float
    source_sha256: str
    output_sha256: str
    output_chars: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_archive(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HumanEvalError("cannot inspect HumanEval archive") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise HumanEvalError("HumanEval archive must be a regular non-symlink file")
        if file_stat.st_size > _MAX_ARCHIVE_BYTES:
            raise HumanEvalError("HumanEval archive exceeds the size limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(_MAX_ARCHIVE_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise HumanEvalError("HumanEval archive exceeds the size limit")
    return payload


def default_corpus_path() -> Path:
    return Path.home() / ".cache" / "mio" / "benchmarks" / "HumanEval.jsonl.gz"


def fetch_humaneval(destination: Path | None = None) -> Path:
    """Fetch the immutable official archive and verify it before publishing."""

    path = (destination or default_corpus_path()).expanduser()
    if path.is_file():
        payload = _read_archive(path)
        if _sha256(payload) != HUMANEVAL_SHA256:
            raise HumanEvalError("cached HumanEval archive has the wrong SHA-256")
        return path

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    request = Request(HUMANEVAL_URL, headers={"User-Agent": "mio-effort-research/1"})
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(  # noqa: S310 - immutable HTTPS URL with pinned content digest
        request,
        timeout=30.0,
        context=tls_context,
    ) as response:
        payload = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise HumanEvalError("HumanEval download exceeds the size limit")
    if _sha256(payload) != HUMANEVAL_SHA256:
        raise HumanEvalError("downloaded HumanEval archive has the wrong SHA-256")

    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_humaneval(path: Path, *, require_official: bool = True) -> tuple[HumanEvalCase, ...]:
    payload = _read_archive(path.expanduser())
    if require_official and _sha256(payload) != HUMANEVAL_SHA256:
        raise HumanEvalError("HumanEval archive does not match the pinned corpus")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as archive:
            decompressed = archive.read(_MAX_DECOMPRESSED_BYTES + 1)
    except (EOFError, OSError) as exc:
        raise HumanEvalError("HumanEval archive is not valid gzip") from exc
    if len(decompressed) > _MAX_DECOMPRESSED_BYTES:
        raise HumanEvalError("decompressed HumanEval corpus exceeds the size limit")

    cases: list[HumanEvalCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(decompressed.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
            case = HumanEvalCase(
                task_id=row["task_id"],
                prompt=row["prompt"],
                test=row["test"],
                entry_point=row["entry_point"],
            )
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HumanEvalError(f"invalid HumanEval row {line_number}") from exc
        if case.task_id in seen:
            raise HumanEvalError(f"duplicate HumanEval task id: {case.task_id}")
        seen.add(case.task_id)
        cases.append(case)
    if not cases:
        raise HumanEvalError("HumanEval corpus is empty")
    return tuple(cases)


def split_humaneval(
    cases: tuple[HumanEvalCase, ...],
    split: Literal["calibration", "heldout", "all"],
) -> tuple[HumanEvalCase, ...]:
    """Return the preregistered hash split without inspecting task contents."""

    if split == "all":
        return cases
    if split not in {"calibration", "heldout"}:
        raise ValueError("split must be calibration, heldout, or all")
    if len(cases) <= CALIBRATION_TASKS:
        raise HumanEvalError("corpus is too small for the preregistered split")
    ranked = sorted(
        cases,
        key=lambda case: hashlib.sha256(
            f"{SPLIT_SALT}\0{case.task_id}".encode("utf-8")
        ).digest(),
    )
    calibration_ids = {case.task_id for case in ranked[:CALIBRATION_TASKS]}
    return tuple(
        case
        for case in cases
        if (case.task_id in calibration_ids) == (split == "calibration")
    )


def corpus_manifest(cases: tuple[HumanEvalCase, ...]) -> dict[str, object]:
    rows = [
        {
            "task_id": case.task_id,
            "prompt_sha256": _sha256(case.prompt.encode("utf-8")),
            "test_sha256": _sha256(case.test.encode("utf-8")),
            "entry_point": case.entry_point,
        }
        for case in cases
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "tasks": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "manifest_sha256": _sha256(canonical),
    }


def _defines_entry_point(source: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in ast.walk(tree)
    )


def prepare_candidate(case: PublicHumanEvalCase, model_text: str) -> PreparedCandidate:
    """Extract either a full module or a HumanEval-style function completion."""

    if not isinstance(model_text, str) or not model_text.strip():
        raise HumanEvalError("candidate output is empty")
    visible = model_text.rsplit("</think>", 1)[-1].strip("\r\n")
    fenced = [
        match.strip("\r\n")
        for match in _CODE_FENCE.findall(visible)
        if match.strip()
    ]
    snippets = fenced or [visible]

    for snippet in snippets:
        if _defines_entry_point(snippet, case.entry_point):
            source = snippet.rstrip() + "\n"
            return PreparedCandidate(snippet, source, _sha256(source.encode("utf-8")))
    for snippet in snippets:
        source = case.prompt + snippet.rstrip() + "\n"
        if _defines_entry_point(source, case.entry_point):
            return PreparedCandidate(snippet, source, _sha256(source.encode("utf-8")))
    raise HumanEvalError("candidate is not a parseable implementation of the entry point")


def validate_candidate_public(
    case: PublicHumanEvalCase,
    model_text: str,
) -> PublicValidationResult:
    """Validate only public syntax/shape evidence for controller routing.

    A parseable candidate remains ``UNKNOWN``: compilation cannot establish
    semantic correctness.  Hidden tests are deliberately unavailable through
    this interface and are reserved for terminal evaluation.
    """

    started = time.perf_counter()
    try:
        prepared = prepare_candidate(case, model_text)
        compile(prepared.source, "<mio-public-candidate>", "exec", dont_inherit=True)
    except (HumanEvalError, SyntaxError, ValueError) as exc:
        return PublicValidationResult(
            outcome=ValidationOutcome.FAIL,
            status="format_error",
            feedback=_safe_feedback(str(exc)),
            elapsed_seconds=time.perf_counter() - started,
            source_sha256=_sha256(model_text.encode("utf-8", errors="replace")),
        )
    return PublicValidationResult(
        outcome=ValidationOutcome.UNKNOWN,
        status="parseable",
        feedback="parseable",
        elapsed_seconds=time.perf_counter() - started,
        source_sha256=prepared.source_sha256,
    )


def _safe_feedback(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(
        r"(?:/Users|/private|/Volumes|/Network)/[^\s:'\"]+",
        "<path>",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[:256]


def _top_level_defined_names(node: ast.stmt) -> frozenset[str]:
    """Return names bound by one declarative top-level prompt statement."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return frozenset({node.name})
    if isinstance(node, ast.Import):
        return frozenset(
            alias.asname or alias.name.partition(".")[0] for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        return frozenset(
            alias.asname or alias.name
            for alias in node.names
            if alias.name != "*"
        )

    def assignment_names(target: ast.expr) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {
                name
                for item in target.elts
                for name in assignment_names(item)
            }
        return set()

    if isinstance(node, ast.Assign):
        return frozenset(
            name
            for target in node.targets
            for name in assignment_names(target)
        )
    if isinstance(node, ast.AnnAssign):
        return frozenset(assignment_names(node.target))
    return frozenset()


def _public_verifier_support_source(case: HumanEvalCase) -> str:
    """Extract public declarations transitively needed by hidden tests.

    HumanEval tests sometimes call helpers declared in the public prompt (for
    example ``poly``) or call the entry point by name as well as through their
    ``candidate`` argument.  Executing the complete prompt would also install
    the incomplete target stub and could run unrelated top-level statements.
    Instead, this extracts only declarative imports, helper definitions/classes,
    and assignments reachable by name from the pinned test source.  The target
    itself is always supplied by the RPC proxy.

    Extracted code still executes inside the trusted verifier sandbox.  A
    non-official corpus can place executable imports, decorators, defaults, or
    class bodies in a public declaration, so this static filter complements --
    and does not replace -- the no-network/no-fork sandbox.
    """

    try:
        prompt_tree = ast.parse(case.prompt)
        test_tree = ast.parse(case.test)
    except SyntaxError as exc:
        raise HumanEvalError("HumanEval prompt or test is not parseable") from exc

    declarations: list[tuple[ast.stmt, frozenset[str]]] = []
    last_declaration: dict[str, int] = {}
    for node in prompt_tree.body:
        names = _top_level_defined_names(node)
        if not names or case.entry_point in names:
            continue
        index = len(declarations)
        declarations.append((node, names))
        for name in names:
            last_declaration[name] = index

    pending = {
        node.id
        for node in ast.walk(test_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    pending.discard(case.entry_point)
    selected: set[int] = set()
    inspected: set[str] = set()
    while pending:
        name = pending.pop()
        if name in inspected:
            continue
        inspected.add(name)
        index = last_declaration.get(name)
        if index is None:
            continue
        selected.add(index)
        declaration, _names = declarations[index]
        pending.update(
            node.id
            for node in ast.walk(declaration)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )

    snippets: list[str] = []
    for index in sorted(selected):
        node, _names = declarations[index]
        snippet = ast.get_source_segment(case.prompt, node) or ast.unparse(node)
        snippets.append(snippet.rstrip())
    return "\n\n".join(snippets)


def _candidate_worker_source(
    prepared: PreparedCandidate,
    entry_point: str,
    marker: str,
    request_path: Path,
) -> str:
    """Build the candidate-only RPC worker.

    This file deliberately contains neither the hidden tests nor the verifier
    success marker.  A candidate may forge any RPC return value, which is
    equivalent to choosing its function result, but it cannot forge the
    isolated verifier's final verdict without learning its independent marker.
    """

    return (
        prepared.source.rstrip()
        + "\n\n"
        + "import base64 as __mio_base64\n"
        + "import json as __mio_json\n"
        + "import math as __mio_math\n"
        + "import sys as __mio_sys\n\n"
        + "def __mio_decode(value, depth=0):\n"
        + "    if depth > 32 or not isinstance(value, dict):\n"
        + "        raise ValueError('invalid rpc value')\n"
        + "    tag = value.get('t')\n"
        + "    data = value.get('v')\n"
        + "    if tag == 'none': return None\n"
        + "    if tag == 'bool' and isinstance(data, bool): return data\n"
        + "    if tag == 'int' and isinstance(data, str): return int(data)\n"
        + "    if tag == 'float' and isinstance(data, str):\n"
        + "        result = float(data)\n"
        + "        if not __mio_math.isfinite(result): raise ValueError('nonfinite')\n"
        + "        return result\n"
        + "    if tag == 'str' and isinstance(data, str): return data\n"
        + "    if tag == 'bytes' and isinstance(data, str):\n"
        + "        return __mio_base64.b64decode(data, validate=True)\n"
        + "    if tag in ('list', 'tuple', 'set', 'frozenset') and isinstance(data, list):\n"
        + "        items = [__mio_decode(item, depth + 1) for item in data]\n"
        + "        if tag == 'list': return items\n"
        + "        if tag == 'tuple': return tuple(items)\n"
        + "        if tag == 'set': return set(items)\n"
        + "        return frozenset(items)\n"
        + "    if tag == 'dict' and isinstance(data, list):\n"
        + "        return {__mio_decode(pair[0], depth + 1): "
        + "__mio_decode(pair[1], depth + 1) for pair in data "
        + "if isinstance(pair, list) and len(pair) == 2}\n"
        + "    raise ValueError('unsupported rpc value')\n\n"
        + "def __mio_encode(value, depth=0):\n"
        + "    if depth > 32: raise ValueError('rpc result is too deep')\n"
        + "    if value is None: return {'t': 'none'}\n"
        + "    if isinstance(value, bool): return {'t': 'bool', 'v': value}\n"
        + "    if isinstance(value, int): return {'t': 'int', 'v': str(value)}\n"
        + "    if isinstance(value, float):\n"
        + "        if not __mio_math.isfinite(value): raise ValueError('nonfinite')\n"
        + "        return {'t': 'float', 'v': repr(value)}\n"
        + "    if isinstance(value, str): return {'t': 'str', 'v': value}\n"
        + "    if isinstance(value, bytes):\n"
        + "        return {'t': 'bytes', 'v': __mio_base64.b64encode(value).decode('ascii')}\n"
        + "    if isinstance(value, (list, tuple, set, frozenset)):\n"
        + "        tag = type(value).__name__\n"
        + "        return {'t': tag, 'v': [__mio_encode(item, depth + 1) for item in value]}\n"
        + "    if isinstance(value, dict):\n"
        + "        return {'t': 'dict', 'v': [[__mio_encode(key, depth + 1), "
        + "__mio_encode(item, depth + 1)] for key, item in value.items()]}\n"
        + "    raise ValueError('unsupported rpc result')\n\n"
        + "try:\n"
        + f"    with open({os.fspath(request_path)!r}, encoding='utf-8') as __mio_stream:\n"
        + "        __mio_request = __mio_json.loads(__mio_stream.read(65537))\n"
        + "    __mio_args = __mio_decode(__mio_request['args'])\n"
        + "    __mio_kwargs = __mio_decode(__mio_request['kwargs'])\n"
        + f"    __mio_result = {entry_point}(*__mio_args, **__mio_kwargs)\n"
        + "    __mio_payload = {'ok': True, 'result': __mio_encode(__mio_result)}\n"
        + "except BaseException as __mio_exc:\n"
        + "    __mio_payload = {'ok': False, 'error': type(__mio_exc).__name__}\n"
        + f"__mio_sys.stdout.write({marker!r} + ':' + "
        + "__mio_json.dumps(__mio_payload, separators=(',', ':')) + '\\n')\n"
    )


def _hidden_verifier_source(
    case: HumanEvalCase,
    *,
    request_marker: str,
    response_marker: str,
    verdict_marker: str,
    timeout_s: float,
) -> str:
    """Build a sandboxed verifier whose only candidate handle is host RPC."""

    public_support = _public_verifier_support_source(case)
    settings = json.dumps(
        {
            "request_marker": request_marker,
            "response_marker": response_marker,
            "verdict_marker": verdict_marker,
            "timeout_s": timeout_s,
        },
        sort_keys=True,
    )
    return (
        "import base64 as __mio_base64\n"
        "import json as __mio_json\n"
        "import math as __mio_math\n"
        "import sys as __mio_sys\n"
        "import time as __mio_time\n\n"
        f"__MIO_SETTINGS = __mio_json.loads({settings!r})\n"
        "__MIO_DEADLINE = __mio_time.monotonic() + __MIO_SETTINGS['timeout_s']\n"
        "__MIO_MAX_MESSAGE = 65536\n\n"
        "class __MioCandidateTimeout(Exception): pass\n"
        "class __MioCandidateOutputLimit(Exception): pass\n"
        "class __MioCandidateInvalidExit(Exception): pass\n"
        "class __MioCandidateRuntime(Exception): pass\n\n"
        "def __mio_encode(value, depth=0):\n"
        "    if depth > 32: raise __MioCandidateRuntime('rpc input is too deep')\n"
        "    if value is None: return {'t': 'none'}\n"
        "    if isinstance(value, bool): return {'t': 'bool', 'v': value}\n"
        "    if isinstance(value, int): return {'t': 'int', 'v': str(value)}\n"
        "    if isinstance(value, float):\n"
        "        if not __mio_math.isfinite(value): raise __MioCandidateRuntime('nonfinite')\n"
        "        return {'t': 'float', 'v': repr(value)}\n"
        "    if isinstance(value, str): return {'t': 'str', 'v': value}\n"
        "    if isinstance(value, bytes):\n"
        "        return {'t': 'bytes', 'v': __mio_base64.b64encode(value).decode('ascii')}\n"
        "    if isinstance(value, (list, tuple, set, frozenset)):\n"
        "        return {'t': type(value).__name__, 'v': [__mio_encode(item, depth + 1) for item in value]}\n"
        "    if isinstance(value, dict):\n"
        "        return {'t': 'dict', 'v': [[__mio_encode(key, depth + 1), "
        "__mio_encode(item, depth + 1)] for key, item in value.items()]}\n"
        "    raise __MioCandidateRuntime('unsupported rpc input')\n\n"
        "def __mio_decode(value, depth=0):\n"
        "    if depth > 32 or not isinstance(value, dict):\n"
        "        raise __MioCandidateRuntime('invalid rpc result')\n"
        "    tag, data = value.get('t'), value.get('v')\n"
        "    if tag == 'none': return None\n"
        "    if tag == 'bool' and isinstance(data, bool): return data\n"
        "    if tag == 'int' and isinstance(data, str): return int(data)\n"
        "    if tag == 'float' and isinstance(data, str):\n"
        "        result = float(data)\n"
        "        if not __mio_math.isfinite(result): raise __MioCandidateRuntime('nonfinite')\n"
        "        return result\n"
        "    if tag == 'str' and isinstance(data, str): return data\n"
        "    if tag == 'bytes' and isinstance(data, str):\n"
        "        return __mio_base64.b64decode(data, validate=True)\n"
        "    if tag in ('list', 'tuple', 'set', 'frozenset') and isinstance(data, list):\n"
        "        items = [__mio_decode(item, depth + 1) for item in data]\n"
        "        if tag == 'list': return items\n"
        "        if tag == 'tuple': return tuple(items)\n"
        "        if tag == 'set': return set(items)\n"
        "        return frozenset(items)\n"
        "    if tag == 'dict' and isinstance(data, list):\n"
        "        result = {}\n"
        "        for pair in data:\n"
        "            if not isinstance(pair, list) or len(pair) != 2:\n"
        "                raise __MioCandidateRuntime('invalid rpc mapping')\n"
        "            result[__mio_decode(pair[0], depth + 1)] = __mio_decode(pair[1], depth + 1)\n"
        "        return result\n"
        "    raise __MioCandidateRuntime('unsupported rpc result')\n\n"
        "def __mio_invoke(*args, **kwargs):\n"
        "    if __mio_time.monotonic() >= __MIO_DEADLINE:\n"
        "        raise __MioCandidateTimeout('deadline')\n"
        "    request = __mio_json.dumps({'args': __mio_encode(args), "
        "'kwargs': __mio_encode(kwargs)}, separators=(',', ':'))\n"
        "    if len(request.encode('utf-8')) > __MIO_MAX_MESSAGE:\n"
        "        raise __MioCandidateRuntime('rpc input limit')\n"
        "    __mio_sys.stdout.write(__MIO_SETTINGS['request_marker'] + ':' + request + '\\n')\n"
        "    __mio_sys.stdout.flush()\n"
        "    response = __mio_sys.stdin.readline(__MIO_MAX_MESSAGE + 1)\n"
        "    prefix = __MIO_SETTINGS['response_marker'] + ':'\n"
        "    if not response.startswith(prefix) or len(response.encode('utf-8')) > __MIO_MAX_MESSAGE:\n"
        "        raise __MioCandidateRuntime('host rpc response invalid')\n"
        "    try: payload = __mio_json.loads(response[len(prefix):])\n"
        "    except Exception as exc: raise __MioCandidateRuntime('host rpc response invalid') from exc\n"
        "    if not isinstance(payload, dict):\n"
        "        raise __MioCandidateRuntime('host rpc response invalid')\n"
        "    if payload.get('ok') is not True:\n"
        "        status = payload.get('status')\n"
        "        if status == 'timeout': raise __MioCandidateTimeout('deadline')\n"
        "        if status == 'output_limit': raise __MioCandidateOutputLimit('output')\n"
        "        if status == 'invalid_exit': raise __MioCandidateInvalidExit('exit')\n"
        "        raise __MioCandidateRuntime('candidate runtime')\n"
        "    return __mio_decode(payload.get('result'))\n\n"
        # Bind the public entry-point name before loading public helpers/tests.
        # Official HumanEval tests sometimes reference it directly instead of
        # using only the ``candidate`` argument passed to ``check``.
        + f"\n{case.entry_point} = __mio_invoke\n"
        + (f"\n{public_support}\n" if public_support else "")
        + "\n"
        + case.test.rstrip()
        + "\n\ntry:\n"
        + "    check(__mio_invoke)\n"
        + "except __MioCandidateTimeout:\n"
        + f"    print({verdict_marker!r} + ':timeout')\n"
        + "except __MioCandidateOutputLimit:\n"
        + f"    print({verdict_marker!r} + ':output_limit')\n"
        + "except __MioCandidateInvalidExit:\n"
        + f"    print({verdict_marker!r} + ':invalid_exit')\n"
        + "except AssertionError:\n"
        + f"    print({verdict_marker!r} + ':assertion_failed')\n"
        + "except BaseException as __mio_exc:\n"
        + f"    print({verdict_marker!r} + ':exception:' + type(__mio_exc).__name__)\n"
        + "else:\n"
        + f"    print({verdict_marker!r} + ':passed')\n"
    )


_RPC_MAX_MESSAGE_BYTES = 65_536
_RPC_MAX_CALLS = 4_096


def _candidate_rpc_response(
    result: _BoundedCommandResult,
    *,
    marker: str,
) -> dict[str, object]:
    """Convert one isolated candidate process into a bounded RPC response."""

    if result.timed_out:
        return {"ok": False, "status": "timeout"}
    if result.output_exceeded:
        return {"ok": False, "status": "output_limit"}
    prefix = marker + ":"
    rows = [line[len(prefix) :] for line in result.output.splitlines() if line.startswith(prefix)]
    if not rows:
        status = "invalid_exit" if result.returncode == 0 else "exception"
        return {"ok": False, "status": status}
    try:
        payload = json.loads(rows[-1])
    except json.JSONDecodeError:
        return {"ok": False, "status": "invalid_exit"}
    if not isinstance(payload, dict) or type(payload.get("ok")) is not bool:
        return {"ok": False, "status": "invalid_exit"}
    if payload["ok"] is True and "result" in payload:
        return {"ok": True, "result": payload["result"]}
    return {"ok": False, "status": "exception"}


def _publish_candidate_request(path: Path, payload: bytes) -> None:
    """Atomically publish one host-validated request to the candidate sandbox."""

    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o400)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_hidden_verifier_rpc(
    verifier_argv: list[str],
    *,
    verifier_directory: Path,
    verifier_environment: dict[str, str],
    candidate_argv: list[str],
    candidate_directory: Path,
    candidate_environment: dict[str, str],
    candidate_request_path: Path,
    request_marker: str,
    response_marker: str,
    candidate_marker: str,
    timeout_s: float,
    output_limit_chars: int,
) -> _BoundedCommandResult:
    """Mediate hidden-test calls without sharing either sandbox's files.

    The trusted verifier owns hidden tests and can only request a function call
    over tagged JSON.  The host launches a fresh no-fork candidate process for
    every call, then returns only its serialized value.  Request rows are never
    included in captured verifier output.
    """

    max_output_bytes = max(128, output_limit_chars * 4)
    process = subprocess.Popen(
        verifier_argv,
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=verifier_directory,
        env=verifier_environment,
        start_new_session=True,
        close_fds=True,
        bufsize=0,
    )
    if process.stdin is None or process.stdout is None:  # pragma: no cover
        _terminate_process_group(process)
        raise RuntimeError("verifier RPC pipes were not created")

    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    pending = bytearray()
    captured = bytearray()
    deadline = time.monotonic() + timeout_s
    timed_out = False
    output_exceeded = False
    rpc_calls = 0
    eof = False

    request_prefix = (request_marker + ":").encode("utf-8")

    def append_visible(line: bytes) -> None:
        nonlocal output_exceeded
        remaining = max_output_bytes - len(captured)
        if len(line) > remaining:
            captured.extend(line[: max(0, remaining)])
            output_exceeded = True
        else:
            captured.extend(line)

    def send_response(payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _RPC_MAX_MESSAGE_BYTES:
            encoded = b'{"ok":false,"status":"output_limit"}'
        row = response_marker.encode("utf-8") + b":" + encoded + b"\n"
        process.stdin.write(row)
        process.stdin.flush()

    def handle_line(line: bytes) -> None:
        nonlocal rpc_calls, output_exceeded
        if not line.startswith(request_prefix):
            append_visible(line)
            return
        rpc_calls += 1
        request = line[len(request_prefix) :].rstrip(b"\r\n")
        if rpc_calls > _RPC_MAX_CALLS:
            send_response({"ok": False, "status": "output_limit"})
            output_exceeded = True
            return
        if not request or len(request) > _RPC_MAX_MESSAGE_BYTES:
            send_response({"ok": False, "status": "output_limit"})
            return
        try:
            parsed = json.loads(request)
        except json.JSONDecodeError:
            send_response({"ok": False, "status": "exception"})
            return
        if not isinstance(parsed, dict) or set(parsed) != {"args", "kwargs"}:
            send_response({"ok": False, "status": "exception"})
            return
        canonical = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        if len(canonical) > _RPC_MAX_MESSAGE_BYTES:
            send_response({"ok": False, "status": "output_limit"})
            return
        _publish_candidate_request(candidate_request_path, canonical)
        remaining = max(0.001, deadline - time.monotonic())
        candidate_result = _run_bounded_process(
            candidate_argv,
            cwd=candidate_directory,
            env=candidate_environment,
            timeout_s=remaining,
            output_limit_chars=output_limit_chars,
        )
        send_response(_candidate_rpc_response(candidate_result, marker=candidate_marker))

    def drain() -> None:
        nonlocal eof, output_exceeded
        while True:
            try:
                chunk = os.read(descriptor, 65_536)
            except BlockingIOError:
                return
            if not chunk:
                eof = True
                return
            pending.extend(chunk)
            if len(pending) > max_output_bytes + _RPC_MAX_MESSAGE_BYTES:
                output_exceeded = True
                return
            while b"\n" in pending:
                raw, _, rest = pending.partition(b"\n")
                pending[:] = rest
                handle_line(raw + b"\n")
                if output_exceeded:
                    return
            if len(chunk) < 65_536:
                return

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                timed_out = True
                break
            for _key, _mask in selector.select(timeout=min(0.05, remaining)):
                drain()
            if output_exceeded:
                break
            if process.poll() is not None:
                drain()
                if pending:
                    handle_line(bytes(pending))
                    pending.clear()
                break
            if eof:
                break
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        _terminate_process_group(process)
        selector.close()
        process.stdout.close()

    return _BoundedCommandResult(
        output=bytes(captured).decode("utf-8", errors="replace"),
        returncode=int(process.returncode if process.returncode is not None else -1),
        timed_out=timed_out,
        output_exceeded=output_exceeded,
    )


def verify_candidate(
    case: HumanEvalCase,
    model_text: str,
    *,
    timeout_s: float = 10.0,
) -> VerificationResult:
    """Evaluate a candidate without exposing hidden tests to its process.

    A trusted, sandboxed verifier owns the tests and final verdict marker.  A
    host broker launches a fresh no-network/no-fork candidate sandbox for each
    bounded tagged-JSON RPC. Candidate and verifier files live in disjoint
    directories and neither process can inspect the other's workspace.
    """

    started = time.perf_counter()
    try:
        prepared = prepare_candidate(case.public, model_text)
    except HumanEvalError as exc:
        elapsed = time.perf_counter() - started
        return VerificationResult(
            passed=False,
            status="format_error",
            feedback=_safe_feedback(str(exc)),
            elapsed_seconds=elapsed,
            source_sha256=_sha256(model_text.encode("utf-8", errors="replace")),
            output_sha256=_sha256(b""),
            output_chars=0,
        )
    if not 0.1 <= timeout_s <= 30.0:
        raise ValueError("timeout_s must be between 0.1 and 30 seconds")

    request_marker = f"__MIO_REQUEST_{secrets.token_hex(16)}__"
    response_marker = f"__MIO_RESPONSE_{secrets.token_hex(16)}__"
    candidate_marker = f"__MIO_CANDIDATE_{secrets.token_hex(16)}__"
    verdict_marker = f"__MIO_VERIFY_{secrets.token_hex(16)}__"

    try:
        with tempfile.TemporaryDirectory(prefix="mio-humaneval-") as directory:
            workspace = Path(directory).resolve()
            candidate_workspace = workspace / "candidate"
            verifier_workspace = workspace / "verifier"
            candidate_workspace.mkdir(mode=0o700)
            verifier_workspace.mkdir(mode=0o700)

            candidate_path = candidate_workspace / "candidate.py"
            candidate_request_path = candidate_workspace / "request.json"
            candidate_path.write_text(
                _candidate_worker_source(
                    prepared,
                    case.entry_point,
                    candidate_marker,
                    candidate_request_path,
                ),
                encoding="utf-8",
            )
            candidate_path.chmod(0o400)
            candidate_policy = AgentToolPolicy(
                workspace_roots=(candidate_workspace,),
                permissions=frozenset(
                    {AgentToolPermission.READ, AgentToolPermission.SHELL}
                ),
                output_limit_chars=16_384,
                file_limit_chars=1_000_000,
                command_timeout_s=timeout_s,
                audit_sink=lambda _event: None,
            )
            python = shlex.quote(sys.executable)
            candidate_command = (
                "ulimit -S -f 2048 >/dev/null 2>&1; "
                "ulimit -S -d 1048576 >/dev/null 2>&1 || true; "
                "ulimit -S -m 1048576 >/dev/null 2>&1 || true; "
                f"exec {python} -I -B {shlex.quote(candidate_path.name)}"
            )
            candidate_argv, candidate_environment = sandboxed_command(
                _shell_argv(candidate_command, timeout_s=timeout_s),
                candidate_policy,
                allow_process_fork=False,
            )

            verifier_path = verifier_workspace / "verifier.py"
            verifier_path.write_text(
                _hidden_verifier_source(
                    case,
                    request_marker=request_marker,
                    response_marker=response_marker,
                    verdict_marker=verdict_marker,
                    timeout_s=timeout_s,
                ),
                encoding="utf-8",
            )
            verifier_path.chmod(0o400)
            verifier_policy = AgentToolPolicy(
                workspace_roots=(verifier_workspace,),
                permissions=frozenset(
                    {AgentToolPermission.READ, AgentToolPermission.SHELL}
                ),
                output_limit_chars=16_384,
                file_limit_chars=1_000_000,
                command_timeout_s=timeout_s,
                audit_sink=lambda _event: None,
            )
            verifier_command = (
                "ulimit -S -f 2048 >/dev/null 2>&1; "
                "ulimit -S -d 1048576 >/dev/null 2>&1 || true; "
                "ulimit -S -m 1048576 >/dev/null 2>&1 || true; "
                f"exec {python} -I -B {shlex.quote(verifier_path.name)}"
            )
            argv, environment = sandboxed_command(
                _shell_argv(verifier_command, timeout_s=timeout_s),
                verifier_policy,
                allow_process_fork=False,
            )
            result = _run_hidden_verifier_rpc(
                argv,
                verifier_directory=verifier_workspace,
                verifier_environment=environment,
                candidate_argv=candidate_argv,
                candidate_directory=candidate_workspace,
                candidate_environment=candidate_environment,
                candidate_request_path=candidate_request_path,
                request_marker=request_marker,
                response_marker=response_marker,
                candidate_marker=candidate_marker,
                timeout_s=timeout_s,
                output_limit_chars=verifier_policy.output_limit_chars,
            )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return VerificationResult(
            passed=False,
            status="sandbox_error",
            feedback=_safe_feedback(type(exc).__name__),
            elapsed_seconds=elapsed,
            source_sha256=prepared.source_sha256,
            output_sha256=_sha256(b""),
            output_chars=0,
        )

    elapsed = time.perf_counter() - started
    output = result.output
    output_digest = _sha256(output.encode("utf-8", errors="replace"))
    marker_rows = [
        line for line in output.splitlines() if line.startswith(verdict_marker + ":")
    ]
    status_payload = (
        marker_rows[-1][len(verdict_marker) + 1 :] if marker_rows else ""
    )
    if result.timed_out:
        status, feedback = "timeout", "validator_timeout"
    elif result.output_exceeded:
        status, feedback = "output_limit", "validator_output_limit"
    elif status_payload == "passed" and result.returncode == 0:
        status, feedback = "passed", "passed"
    elif status_payload == "assertion_failed":
        status, feedback = "assertion_failed", "assertion_failed"
    elif status_payload == "timeout":
        status, feedback = "timeout", "validator_timeout"
    elif status_payload == "output_limit":
        status, feedback = "output_limit", "validator_output_limit"
    elif status_payload == "invalid_exit":
        status, feedback = "invalid_exit", "candidate_exit_without_result"
    elif status_payload.startswith("exception:"):
        status, feedback = "exception", _safe_feedback(status_payload)
    elif result.returncode == 0:
        status, feedback = "invalid_exit", "validator_marker_missing"
    else:
        status, feedback = "runtime_error", f"validator_returncode_{result.returncode}"
    return VerificationResult(
        passed=status == "passed",
        status=status,
        feedback=feedback,
        elapsed_seconds=elapsed,
        source_sha256=prepared.source_sha256,
        output_sha256=output_digest,
        output_chars=len(output),
    )
