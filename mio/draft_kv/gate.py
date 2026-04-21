"""Confidence gate for Speculative Prefill.

Before committing to SP (which skips most target-model layers), check a
cheap signal from the intermediate hidden state to decide whether the
projector's output is likely to be trustworthy. If the signal suggests
high uncertainty, fall back to full prefill instead.

Signal used: per-token L2 norm of the intermediate hidden state. Prompts
whose hidden states fall outside the distribution the projector was
trained on tend to have anomalous norms (either very low or very high).
This is a stand-in — a real deployment would train a per-prompt classifier
on (hidden → prefill_success) pairs harvested via HarvestRecorder.

The gate is deliberately simple: two-sided threshold on the mean and
fraction-outliers on the tail. Deterministic, no state, cheap to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class GateDecision:
    """Result of a gate evaluation."""

    proceed: bool
    reason: str
    mean_norm: float
    outlier_fraction: float


class ConfidenceGate:
    """Two-sided norm threshold + outlier-fraction gate.

    Args:
        min_norm: hidden-state L2 norm must exceed this per-token average.
        max_norm: ...and stay below this per-token average.
        max_outlier_fraction: at most this fraction of tokens can fall
            outside [min_norm, max_norm] individually.

    Default thresholds bracket the empirical range for Qwen3.5/3.6 at
    layer 15 on natural text (~3-50). They're conservative: most prompts
    will proceed. Bad prompts with anomalous hiddens (all-zero, giant
    scale) will trigger fallback.
    """

    def __init__(
        self,
        *,
        min_norm: float = 1.0,
        max_norm: float = 100.0,
        max_outlier_fraction: float = 0.05,
    ) -> None:
        if not (min_norm > 0):
            raise ValueError("min_norm must be positive")
        if max_norm <= min_norm:
            raise ValueError("max_norm must exceed min_norm")
        if not (0.0 <= max_outlier_fraction <= 1.0):
            raise ValueError("max_outlier_fraction must be in [0, 1]")
        self.min_norm = float(min_norm)
        self.max_norm = float(max_norm)
        self.max_outlier_fraction = float(max_outlier_fraction)

    def evaluate(self, hidden: mx.array) -> GateDecision:
        """Assess whether the hidden state is in-distribution enough to proceed.

        Args:
            hidden: (B, L, D) intermediate hidden from the early target
                layers. Norm is computed per-token.

        Returns:
            GateDecision with .proceed True/False and diagnostics.
        """
        if hidden.ndim != 3:
            raise ValueError(f"expected (B, L, D), got shape {hidden.shape}")

        per_token_norm = mx.sqrt(mx.sum(hidden * hidden, axis=-1))  # (B, L)
        per_token_norm = per_token_norm.flatten()
        mean_norm = float(mx.mean(per_token_norm).item())
        inside = (per_token_norm >= self.min_norm) & (per_token_norm <= self.max_norm)
        outlier_fraction = 1.0 - float(mx.mean(inside.astype(mx.float32)).item())

        if not (self.min_norm <= mean_norm <= self.max_norm):
            return GateDecision(
                proceed=False,
                reason=f"mean norm {mean_norm:.3f} outside [{self.min_norm}, {self.max_norm}]",
                mean_norm=mean_norm,
                outlier_fraction=outlier_fraction,
            )
        if outlier_fraction > self.max_outlier_fraction:
            return GateDecision(
                proceed=False,
                reason=(
                    f"{outlier_fraction:.3f} of tokens outside norm band "
                    f"(max {self.max_outlier_fraction})"
                ),
                mean_norm=mean_norm,
                outlier_fraction=outlier_fraction,
            )
        return GateDecision(
            proceed=True,
            reason="in-distribution",
            mean_norm=mean_norm,
            outlier_fraction=outlier_fraction,
        )
