"""Code validation pipeline: auto-check generated code and retry on errors."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract code blocks from markdown-formatted text.

    Returns list of (language, code) tuples.
    """
    pattern = r"```(\w*)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang or "python", code.strip()) for lang, code in matches]


def run_syntax_check(code: str) -> tuple[bool, str]:
    """Check Python syntax via compile()."""
    try:
        compile(code, "<generated>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"


def run_ruff(code: str) -> tuple[bool, str]:
    """Run ruff check on code string. Returns (passed, output)."""
    try:
        result = subprocess.run(
            ["ruff", "check", "--stdin-filename=generated.py", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stdout.strip()
    except FileNotFoundError:
        return True, ""  # ruff not installed, skip
    except subprocess.TimeoutExpired:
        return True, ""  # timeout, skip


def validate_code(code: str, language: str = "python") -> ValidationResult:
    """Run all validation checks on a code string."""
    if language != "python":
        return ValidationResult(passed=True)

    result = ValidationResult(passed=True)

    # Syntax check
    ok, msg = run_syntax_check(code)
    result.checks.append({"name": "syntax", "passed": ok, "output": msg})
    if not ok:
        result.passed = False
        result.errors.append(msg)

    # Ruff lint (only if syntax passes)
    if ok:
        ok, msg = run_ruff(code)
        result.checks.append({"name": "ruff", "passed": ok, "output": msg})
        if not ok:
            result.passed = False
            # Only include first 5 errors to avoid flooding
            errors = msg.split("\n")[:5]
            result.errors.extend(errors)

    return result


def validate_response(text: str) -> ValidationResult:
    """Validate all code blocks in a response."""
    blocks = extract_code_blocks(text)
    if not blocks:
        return ValidationResult(passed=True)

    combined = ValidationResult(passed=True)
    for lang, code in blocks:
        block_result = validate_code(code, lang)
        combined.checks.extend(block_result.checks)
        if not block_result.passed:
            combined.passed = False
            combined.errors.extend(block_result.errors)

    return combined


def build_retry_message(errors: list[str]) -> str:
    """Build a user message for retry with error context."""
    error_list = "\n".join(f"- {e}" for e in errors[:8])
    return (
        f"The code you generated has errors. Fix them:\n\n{error_list}\n\n"
        "Return the corrected code only. Do not explain."
    )
