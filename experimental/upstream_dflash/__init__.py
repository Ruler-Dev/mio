"""Isolated prototypes for evaluating the upstream ``dflash_mlx`` runtime.

Nothing in this package is imported by Mio's production engine.  Promotion is
deliberately gated on capability and parity evidence; see ``compatibility``.
"""

from .adapter import (
    MioEventAdapter,
    UpstreamGenerationRequest,
    adapt_upstream_stream,
    stream_bundle_as_mio,
)
from .compatibility import (
    CompatibilityGate,
    CompatibilityReport,
    ParityCertificate,
    PromotionRequest,
    assess_promotion,
)
from .deferred_priming import (
    DeferredPrimingStats,
    deferred_drafter_priming,
    stream_with_deferred_drafter_priming,
)


def compare_benchmark_artifacts(*args, **kwargs):
    """Lazy import keeps ``python -m ...comparison`` free of runpy warnings."""

    from .comparison import compare_benchmark_artifacts as compare

    return compare(*args, **kwargs)

__all__ = [
    "CompatibilityGate",
    "CompatibilityReport",
    "DeferredPrimingStats",
    "MioEventAdapter",
    "ParityCertificate",
    "PromotionRequest",
    "UpstreamGenerationRequest",
    "adapt_upstream_stream",
    "assess_promotion",
    "compare_benchmark_artifacts",
    "deferred_drafter_priming",
    "stream_bundle_as_mio",
    "stream_with_deferred_drafter_priming",
]
