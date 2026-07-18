from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import pytest

from experimental.effort.humaneval import (
    HumanEvalCase,
    HumanEvalError,
    corpus_manifest,
    fetch_humaneval,
    load_humaneval,
    prepare_candidate,
    split_humaneval,
    validate_candidate_public,
    verify_candidate,
)
from experimental.markov_effort_controller import ValidationOutcome


def case(task_id: str = "HumanEval/0") -> HumanEvalCase:
    return HumanEvalCase(
        task_id=task_id,
        prompt="def increment(value: int) -> int:\n    \"\"\"Add one.\"\"\"\n",
        test=(
            "def check(candidate):\n"
            "    assert candidate(0) == 1\n"
            "    assert candidate(9) == 10\n"
        ),
        entry_point="increment",
    )


def write_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    payload = b"\n".join(
        json.dumps(row, sort_keys=True).encode("utf-8") for row in rows
    ) + b"\n"
    path.write_bytes(gzip.compress(payload, mtime=0))


def test_load_custom_corpus_and_manifest_are_deterministic(tmp_path: Path) -> None:
    sample = case()
    path = tmp_path / "corpus.jsonl.gz"
    write_corpus(
        path,
        [
            {
                "task_id": sample.task_id,
                "prompt": sample.prompt,
                "test": sample.test,
                "entry_point": sample.entry_point,
            }
        ],
    )
    loaded = load_humaneval(path, require_official=False)
    assert loaded == (sample,)
    assert corpus_manifest(loaded) == corpus_manifest(loaded)
    assert corpus_manifest(loaded)["tasks"] == 1


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    sample = case()
    row = {
        "task_id": sample.task_id,
        "prompt": sample.prompt,
        "test": sample.test,
        "entry_point": sample.entry_point,
    }
    path = tmp_path / "duplicate.jsonl.gz"
    write_corpus(path, [row, row])
    with pytest.raises(HumanEvalError, match="duplicate"):
        load_humaneval(path, require_official=False)


def test_hash_split_is_stable_disjoint_and_content_blind() -> None:
    cases = tuple(case(f"HumanEval/{index}") for index in range(100))
    calibration = split_humaneval(cases, "calibration")
    heldout = split_humaneval(cases, "heldout")
    assert len(calibration) == 32
    assert len(heldout) == 68
    assert {item.task_id for item in calibration}.isdisjoint(
        item.task_id for item in heldout
    )
    assert split_humaneval(cases, "calibration") == calibration
    assert split_humaneval(cases, "all") == cases


def test_prepare_candidate_accepts_completion_or_full_module() -> None:
    sample = case()
    completion = prepare_candidate(sample.public, "    return value + 1")
    assert completion.source.startswith(sample.prompt)
    assert completion.source_sha256

    module = prepare_candidate(
        sample.public,
        "Here is the code:\n```python\ndef increment(value: int) -> int:\n"
        "    return value + 1\n```",
    )
    assert module.source.startswith("def increment")
    assert module.completion.startswith("def increment")


def test_prepare_candidate_rejects_missing_entry_point() -> None:
    with pytest.raises(HumanEvalError, match="entry point"):
        prepare_candidate(case().public, "this is prose, not valid Python ???")


def test_public_validator_cannot_turn_hidden_semantics_into_a_pass() -> None:
    sample = case()
    wrong = validate_candidate_public(sample.public, "    return value")
    correct = validate_candidate_public(sample.public, "    return value + 1")
    malformed = validate_candidate_public(sample.public, "not valid Python ???")

    assert wrong.outcome is ValidationOutcome.UNKNOWN
    assert correct.outcome is ValidationOutcome.UNKNOWN
    assert wrong.status == correct.status == "parseable"
    assert malformed.outcome is ValidationOutcome.FAIL
    assert not hasattr(sample.public, "test")


def test_fetch_rejects_poisoned_cache_without_network(tmp_path: Path) -> None:
    path = tmp_path / "HumanEval.jsonl.gz"
    path.write_bytes(b"not-the-pinned-corpus")
    with pytest.raises(HumanEvalError, match="wrong SHA-256"):
        fetch_humaneval(path)


def test_load_rejects_archive_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.gz"
    target.write_bytes(gzip.compress(b""))
    link = tmp_path / "link.gz"
    link.symlink_to(target)
    with pytest.raises(HumanEvalError, match="archive"):
        load_humaneval(link, require_official=False)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_verifier_passes_and_redacts_assertion_details() -> None:
    passed = verify_candidate(case(), "    return value + 1")
    assert passed.passed is True
    assert passed.status == "passed"
    assert passed.source_sha256

    failed = verify_candidate(case(), "    return value")
    assert failed.passed is False
    assert failed.status == "assertion_failed"
    assert failed.feedback == "assertion_failed"
    assert "candidate(9)" not in failed.feedback


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_verifier_loads_needed_public_helpers_and_binds_entry_point() -> None:
    helper_case = HumanEvalCase(
        task_id="HumanEval/helper-rpc",
        prompt=(
            "import math\n\n"
            "def public_helper(value):\n"
            "    return math.sqrt(value)\n\n"
            "def unused_helper():\n"
            "    raise RuntimeError('must not execute')\n\n"
            "def solve(value):\n"
            "    \"\"\"Return the public helper result.\"\"\"\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate(4) == public_helper(4)\n"
            "    assert solve(9) == public_helper(9)\n"
        ),
        entry_point="solve",
    )

    result = verify_candidate(
        helper_case,
        "    return public_helper(value)\n",
    )

    assert result.passed is True
    assert result.status == "passed"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_default_verifier_budget_covers_many_isolated_rpc_calls() -> None:
    many_call_case = HumanEvalCase(
        task_id="HumanEval/many-rpc-calls",
        prompt="def add(x, y):\n    \"\"\"Add two integers.\"\"\"\n",
        test=(
            "def check(candidate):\n"
            "    for value in range(105):\n"
            "        assert candidate(value, 1) == value + 1\n"
        ),
        entry_point="add",
    )

    result = verify_candidate(many_call_case, "    return x + y\n")

    assert result.passed is True
    assert result.status == "passed"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_verifier_rejects_early_exit_and_times_out_loop() -> None:
    early_exit = verify_candidate(
        case(),
        "    import os\n    os._exit(0)",
    )
    assert early_exit.passed is False
    assert early_exit.status == "invalid_exit"

    timed_out = verify_candidate(case(), "    while True:\n        pass", timeout_s=0.1)
    assert timed_out.passed is False
    assert timed_out.status == "timeout"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_verifier_isolates_hidden_tests_and_verdict_marker() -> None:
    exploit = (
        "    import os\n"
        "    from pathlib import Path\n"
        "    own_source = Path(__file__).read_text(encoding='utf-8')\n"
        "    markers = [word for word in own_source.split() "
        "if '__MIO_VERIFY_' in word and len(word) > 40]\n"
        "    if markers:\n"
        "        print(markers[0].strip(\"'\\\"+ :()\") + ':passed')\n"
        "        os._exit(0)\n"
        "    return value\n"
    )
    result = verify_candidate(case(), exploit)
    assert result.passed is False
    assert result.status == "assertion_failed"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_verifier_candidate_cannot_read_sibling_hidden_harness() -> None:
    candidate = (
        "    from pathlib import Path\n"
        "    try:\n"
        "        hidden = next(Path('..').glob('verifier/*.py')).read_text()\n"
        "    except Exception:\n"
        "        return value\n"
        "    return value + (1 if 'candidate(9)' in hidden else 0)\n"
    )
    result = verify_candidate(case(), candidate)
    assert result.passed is False
    assert result.status == "assertion_failed"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_verifier_cannot_read_a_user_file_outside_its_workspace(tmp_path: Path) -> None:
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("10", encoding="utf-8")
    candidate = (
        "    from pathlib import Path\n"
        f"    return int(Path({str(secret)!r}).read_text())"
    )
    result = verify_candidate(case(), candidate)
    assert result.passed is False
    assert result.status == "exception"
    assert "outside-secret" not in result.feedback
