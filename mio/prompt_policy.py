"""Mutually exclusive system-prompt policies used by Mio frontends.

The policy layer deliberately only transforms messages.  It cannot execute
tools, grant permissions, or change model/runtime configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


class PromptMode(str, Enum):
    """Available prompt policies.  ``none`` means no policy injection."""

    NONE = "none"
    CAVEMAN = "caveman"
    PONYTAIL = "ponytail"


class PromptLevel(str, Enum):
    """Intensity shared by Caveman and Ponytail."""

    LITE = "lite"
    FULL = "full"
    ULTRA = "ultra"


# Original, short wording inspired by DietrichGebert/ponytail's public
# "laziest senior developer" ladder.  It is intentionally not a vendored copy
# of the upstream skill: Mio needs a small inference-time system policy, while
# the full plugin also contains commands, hooks, audits, and review workflows.
_PONYTAIL_PROMPTS = {
    PromptLevel.LITE: """
ENGINEERING MODE: PONYTAIL LITE.
Implement exactly the requested outcome after reading the affected flow.
Prefer an existing or platform-native solution; mention a materially simpler
alternative in one sentence. Keep validation, security, error handling, data
safety, accessibility, and explicitly requested behavior intact.
""",
    PromptLevel.FULL: """
ENGINEERING MODE: PONYTAIL FULL.
Before adding code, choose the first sufficient rung: do not build an
unneeded feature; reuse project code; use the standard library; use a native
platform feature; use an installed dependency; then write the minimum new
code that works. Make the smallest coherent diff and leave one focused check
for non-trivial behavior. Do not remove validation, security, data-loss
protection, error handling, accessibility, or explicitly requested behavior.
Avoid speculative abstractions, dependencies, configuration, and prose.
""",
    PromptLevel.ULTRA: """
ENGINEERING MODE: PONYTAIL ULTRA.
Prove each addition is necessary. Prefer deletion, reuse, standard-library and
native features before new code; add only the smallest complete implementation.
Reject speculative layers and options, but still deliver explicitly requested
behavior. Never trade away understanding, tests for non-trivial logic,
validation, security, data integrity, error handling, or accessibility.
Explain only decisions the user needs.
""",
}


DEFAULT_TOOL_PROTOCOL_MARKERS = (
    "<read_file>",
    "<write_to_file>",
    "<replace_in_file>",
    "<execute_command>",
    "<list_files>",
    "<search_files>",
    "<attempt_completion>",
    "<use_mcp_tool>",
    "<ask_followup_question>",
    "<apply_diff>",
    "<edit_file>",
    "<new_task>",
    "<plan_mode_response>",
)


@dataclass(frozen=True)
class PromptPolicy:
    """A validated prompt mode/level pair."""

    mode: PromptMode = PromptMode.CAVEMAN
    level: PromptLevel = PromptLevel.FULL

    @classmethod
    def resolve(
        cls,
        *,
        prompt_mode: str | PromptMode | None = None,
        prompt_level: str | PromptLevel | None = None,
        caveman: str | None = None,
        ponytail: str | None = None,
    ) -> PromptPolicy:
        """Resolve modern flags and legacy ``--caveman`` into one policy.

        With no arguments the historical Mio default remains Caveman Full.
        Legacy ``caveman="off"`` maps to ``none``.  Passing more than one
        mode selector is rejected even when this function is used outside
        argparse (for example by an embedding application).
        """
        selectors = sum(value is not None for value in (prompt_mode, caveman, ponytail))
        if selectors > 1:
            raise ValueError("prompt-mode, caveman, and ponytail are mutually exclusive")
        if prompt_level is not None and (caveman is not None or ponytail is not None):
            raise ValueError("prompt-level is only valid with prompt-mode")

        if caveman is not None:
            if caveman == "off":
                return cls(PromptMode.NONE, PromptLevel.FULL)
            return cls(PromptMode.CAVEMAN, PromptLevel(caveman))
        if ponytail is not None:
            return cls(PromptMode.PONYTAIL, PromptLevel(ponytail))

        mode = PromptMode(prompt_mode) if prompt_mode is not None else PromptMode.CAVEMAN
        if mode is PromptMode.NONE:
            if prompt_level is not None:
                raise ValueError("prompt-level cannot be used with prompt-mode=none")
            return cls(mode, PromptLevel.FULL)
        level = PromptLevel(prompt_level) if prompt_level is not None else PromptLevel.FULL
        return cls(mode, level)

    @property
    def label(self) -> str:
        if self.mode is PromptMode.NONE:
            return "none"
        return f"{self.mode.value}/{self.level.value}"

    def system_text(self) -> str:
        """Return the text to inject, or an empty string for ``none``."""
        if self.mode is PromptMode.NONE:
            return ""
        if self.mode is PromptMode.PONYTAIL:
            return _PONYTAIL_PROMPTS[self.level].strip()

        # Keep Caveman's established prompt text as the single source of truth.
        from mio.agent import CAVEMAN_LEVELS

        return CAVEMAN_LEVELS[self.level.value].strip()


def apply_prompt_policy(
    messages: Sequence[Mapping],
    policy: PromptPolicy,
    *,
    skip_system_markers: Iterable[str] = DEFAULT_TOOL_PROTOCOL_MARKERS,
) -> list[dict]:
    """Return copied messages with the selected policy injected.

    Policy text is skipped when the leading system message declares a known
    XML tool protocol.  Those clients require exact syntax and historically
    Caveman was skipped for the same reason.  Input mappings are never mutated.
    """
    out = [dict(message) for message in messages]
    text = policy.system_text()
    if not text:
        return out

    if out and out[0].get("role") == "system":
        system_content = out[0].get("content", "") or ""
        if not isinstance(system_content, str):
            system_content = str(system_content)
        if any(marker in system_content for marker in skip_system_markers):
            return out
        out[0]["content"] = f"{text}\n\n{system_content}"
    else:
        out.insert(0, {"role": "system", "content": text})
    return out
