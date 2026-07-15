"""Request-level mixture-of-drafters research prototype.

The prototype is deliberately isolated from Mio's production inference path.
It chooses one speculative drafter for a request and never executes both arms
for the same online decision.
"""

from .model import DrafterObservation, RouteContext
from .router import Decision, OnlineDrafterRouter, RouterConfig

__all__ = [
    "Decision",
    "DrafterObservation",
    "OnlineDrafterRouter",
    "RouteContext",
    "RouterConfig",
]
