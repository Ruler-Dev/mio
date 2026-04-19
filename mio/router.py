"""Tandem mode routing — routes requests to appropriate model tier."""

from __future__ import annotations

import re


# Complexity signals for routing
_COMPLEX_PATTERNS = [
    r"multi.?file",
    r"refactor.*(entire|whole|all)",
    r"architect",
    r"redesign",
    r"implement.*system",
    r"complex",
    r"across.*files",
    r"migration",
]

_SIMPLE_PATTERNS = [
    r"^(fix|complete|finish)\s",
    r"autocomplete",
    r"docstring",
    r"type\s?hint",
    r"rename",
    r"^what\s(is|does)",
    r"explain",
]


class TandemRouter:
    """Routes requests to the best tier based on task complexity."""

    def __init__(self, available_tiers: list[str]) -> None:
        self.available_tiers = available_tiers
        self._tier_order = ["small", "medium", "large", "large-moe"]

    def route(self, messages: list[dict], model_hint: str | None = None) -> str:
        """Determine which tier to use.

        Args:
            messages: OpenAI-format message list.
            model_hint: Explicit model name from request (e.g., "mio-large").

        Returns:
            Tier name ("small", "medium", "large").
        """
        # Explicit tier from model name
        if model_hint and model_hint.startswith("mio-"):
            tier = model_hint.removeprefix("mio-")
            if tier in self.available_tiers and tier != "auto":
                return tier

        # Auto-route based on last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_msg = content
                break

        return self._classify(last_user_msg)

    def _classify(self, text: str) -> str:
        """Classify task complexity from the user message."""
        text_lower = text.lower()

        # Check complex patterns first
        for pattern in _COMPLEX_PATTERNS:
            if re.search(pattern, text_lower):
                return self._best_available("large")

        # Check simple patterns
        for pattern in _SIMPLE_PATTERNS:
            if re.search(pattern, text_lower):
                return self._best_available("small")

        # Heuristic: message length
        word_count = len(text.split())
        if word_count < 20:
            return self._best_available("small")
        if word_count < 80:
            return self._best_available("medium")
        return self._best_available("large")

    def _best_available(self, preferred: str) -> str:
        """Return preferred tier if available, otherwise closest in tier order."""
        if preferred in self.available_tiers:
            return preferred

        # Build priority: tiers closest to preferred in the tier order
        if preferred in self._tier_order:
            idx = self._tier_order.index(preferred)
        else:
            idx = 1  # default to middle

        # Sort available tiers by distance from preferred
        available_sorted = sorted(
            self.available_tiers,
            key=lambda t: abs(self._tier_order.index(t) - idx) if t in self._tier_order else 99,
        )
        return available_sorted[0] if available_sorted else self.available_tiers[0]
