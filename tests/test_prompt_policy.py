from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from mio.main import _add_prompt_policy_arguments, _resolve_prompt_policy_args
from mio.prompt_policy import PromptLevel, PromptMode, PromptPolicy, apply_prompt_policy


def test_prompt_policy_defaults_preserve_caveman_full():
    policy = PromptPolicy.resolve()
    assert policy.mode is PromptMode.CAVEMAN
    assert policy.level is PromptLevel.FULL
    assert policy.label == "caveman/full"


@pytest.mark.parametrize("level", ["lite", "full", "ultra"])
def test_ponytail_levels(level):
    policy = PromptPolicy.resolve(ponytail=level)
    assert policy.mode is PromptMode.PONYTAIL
    assert policy.level.value == level
    assert "PONYTAIL" in policy.system_text()


def test_none_and_legacy_off_inject_nothing():
    messages = [{"role": "user", "content": "hello"}]
    assert apply_prompt_policy(messages, PromptPolicy.resolve(caveman="off")) == messages
    assert PromptPolicy.resolve(prompt_mode="none").label == "none"


def test_prompt_selectors_are_strictly_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PromptPolicy.resolve(prompt_mode="caveman", ponytail="lite")
    with pytest.raises(ValueError, match="only valid"):
        PromptPolicy.resolve(caveman="lite", prompt_level="ultra")
    with pytest.raises(ValueError, match="cannot be used"):
        PromptPolicy.resolve(prompt_mode="none", prompt_level="lite")


def test_apply_policy_copies_input_and_preserves_protocol_system_prompt():
    messages = [{"role": "system", "content": "Use <read_file> exactly"}, {"role": "user", "content": "x"}]
    result = apply_prompt_policy(messages, PromptPolicy.resolve(ponytail="full"))
    assert result == messages
    assert result is not messages
    assert result[0] is not messages[0]


def test_apply_policy_prepends_existing_system_message():
    messages = [{"role": "system", "content": "Project rules"}, {"role": "user", "content": "x"}]
    result = apply_prompt_policy(messages, PromptPolicy.resolve(ponytail="lite"))
    assert result[0]["content"].startswith("ENGINEERING MODE: PONYTAIL LITE")
    assert result[0]["content"].endswith("Project rules")
    assert messages[0]["content"] == "Project rules"


def test_cli_prompt_group_rejects_two_selectors():
    parser = argparse.ArgumentParser()
    _add_prompt_policy_arguments(parser, include_no_caveman=True)
    with pytest.raises(SystemExit):
        parser.parse_args(["--caveman", "lite", "--ponytail", "full"])


def test_cli_legacy_no_caveman_resolves_none():
    args = SimpleNamespace(
        no_caveman=True,
        prompt_mode=None,
        prompt_level=None,
        caveman=None,
        ponytail=None,
    )
    assert _resolve_prompt_policy_args(args).mode is PromptMode.NONE
