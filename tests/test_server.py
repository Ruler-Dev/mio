"""Tests for the Mio API server (unit tests, no model loading)."""

from __future__ import annotations

import pytest

from mio.router import TandemRouter


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
