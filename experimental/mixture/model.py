"""Data model and online acceptance/cost/verification curves."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class RouteContext:
    """Information available before selecting a drafter."""

    request_id: str
    prompt_tokens: int
    requested_tokens: int
    workload: str = "default"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.requested_tokens <= 0:
            raise ValueError("requested_tokens must be positive")

    @property
    def bucket(self) -> str:
        """A deliberately coarse, deterministic routing context."""

        if self.prompt_tokens < 64:
            prompt_bucket = "p<64"
        elif self.prompt_tokens < 256:
            prompt_bucket = "p<256"
        elif self.prompt_tokens < 1024:
            prompt_bucket = "p<1024"
        else:
            prompt_bucket = "p>=1024"
        if self.requested_tokens <= 64:
            output_bucket = "o<=64"
        elif self.requested_tokens <= 256:
            output_bucket = "o<=256"
        else:
            output_bucket = "o>256"
        return f"{self.workload}:{prompt_bucket}:{output_bucket}"


@dataclass(frozen=True)
class DrafterObservation:
    """Feedback produced by exactly one selected drafter.

    ``rounds`` is a verification-cycle count (``rounds`` for DSpark and
    ``cycles_completed`` for DFlash).  ``accepted_per_round`` and
    ``total_seconds_per_round`` form the online acceptance/cost curve.  A
    directly measured verification duration is optional because current
    DSpark telemetry exposes target-forward counts but not a separate target
    verification timer.
    """

    arm: str
    wall_seconds: float
    ttft_seconds: float
    output_tokens: int
    rounds: int | None = None
    accepted_per_round: float | None = None
    verify_seconds: float | None = None
    target_forwards: int | None = None
    fallback: bool = False
    parity: bool | None = None
    peak_memory_bytes: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm:
            raise ValueError("arm must not be empty")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be finite and positive")
        if not math.isfinite(self.ttft_seconds) or self.ttft_seconds < 0:
            raise ValueError("ttft_seconds must be finite and non-negative")
        if self.ttft_seconds > self.wall_seconds:
            raise ValueError("ttft_seconds cannot exceed wall_seconds")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.rounds is not None and self.rounds <= 0:
            raise ValueError("rounds must be positive when present")
        if self.accepted_per_round is not None:
            if not math.isfinite(self.accepted_per_round) or self.accepted_per_round <= 0:
                raise ValueError("accepted_per_round must be finite and positive")
        if self.verify_seconds is not None:
            if not math.isfinite(self.verify_seconds) or self.verify_seconds < 0:
                raise ValueError("verify_seconds must be finite and non-negative")
        if self.target_forwards is not None and self.target_forwards <= 0:
            raise ValueError("target_forwards must be positive when present")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative")

    @property
    def decode_seconds(self) -> float:
        return max(0.0, self.wall_seconds - self.ttft_seconds)

    @property
    def seconds_per_output_token(self) -> float:
        return self.wall_seconds / self.output_tokens

    @property
    def total_seconds_per_round(self) -> float | None:
        if self.rounds is None:
            return None
        return self.decode_seconds / self.rounds

    @property
    def verify_seconds_per_round(self) -> float | None:
        if self.rounds is None or self.verify_seconds is None:
            return None
        return self.verify_seconds / self.rounds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunningMoments:
    """Stable-enough cumulative moments for a small online experiment."""

    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("curve samples must be finite")
        self.count += 1
        self.total += value
        self.total_squares += value * value

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    @property
    def variance(self) -> float | None:
        if self.count < 2:
            return None
        mean = self.total / self.count
        return max(0.0, self.total_squares / self.count - mean * mean)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
        }


@dataclass
class ArmCurve:
    """Online per-arm acceptance/cost/verification model.

    Prediction uses a measured TTFT plus:

    ``requested_tokens / accepted_per_round * total_seconds_per_round``.

    This makes acceptance and verification-round cost first-class inputs while
    avoiding double-counting the separately measured verify timer.  When cycle
    telemetry is absent, prediction falls back to direct seconds per token.
    """

    arm: str
    observations: int = 0
    fallbacks: int = 0
    parity_known: int = 0
    parity_passes: int = 0
    ttft_seconds: RunningMoments = field(default_factory=RunningMoments)
    seconds_per_output_token: RunningMoments = field(default_factory=RunningMoments)
    accepted_per_round: RunningMoments = field(default_factory=RunningMoments)
    total_seconds_per_round: RunningMoments = field(default_factory=RunningMoments)
    verify_seconds_per_round: RunningMoments = field(default_factory=RunningMoments)
    target_forwards_per_output_token: RunningMoments = field(default_factory=RunningMoments)

    def update(self, observation: DrafterObservation) -> None:
        if observation.arm != self.arm:
            raise ValueError(f"observation for {observation.arm!r} cannot update {self.arm!r}")
        self.observations += 1
        self.fallbacks += int(observation.fallback)
        if observation.parity is not None:
            self.parity_known += 1
            self.parity_passes += int(observation.parity)
        self.ttft_seconds.add(observation.ttft_seconds)
        self.seconds_per_output_token.add(observation.seconds_per_output_token)
        if observation.accepted_per_round is not None:
            self.accepted_per_round.add(observation.accepted_per_round)
        total_per_round = observation.total_seconds_per_round
        if total_per_round is not None:
            self.total_seconds_per_round.add(total_per_round)
        verify_per_round = observation.verify_seconds_per_round
        if verify_per_round is not None:
            self.verify_seconds_per_round.add(verify_per_round)
        if observation.target_forwards is not None:
            self.target_forwards_per_output_token.add(
                observation.target_forwards / observation.output_tokens
            )

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.observations if self.observations else 0.0

    @property
    def parity_rate(self) -> float | None:
        return self.parity_passes / self.parity_known if self.parity_known else None

    def predict_seconds(self, context: RouteContext) -> float:
        if not self.observations:
            return math.inf
        ttft = self.ttft_seconds.mean or 0.0
        acceptance = self.accepted_per_round.mean
        round_cost = self.total_seconds_per_round.mean
        if acceptance is not None and round_cost is not None:
            projected_rounds = context.requested_tokens / acceptance
            return ttft + projected_rounds * round_cost
        direct = self.seconds_per_output_token.mean
        if direct is None:
            return math.inf
        return context.requested_tokens * direct

    def risk_adjusted_prediction(
        self,
        context: RouteContext,
        *,
        fallback_penalty: float,
    ) -> float:
        predicted = self.predict_seconds(context)
        return predicted * (1.0 + fallback_penalty * self.fallback_rate)

    def static_cost(self, *, fallback_penalty: float) -> float:
        direct = self.seconds_per_output_token.mean
        if direct is None:
            return math.inf
        return direct * (1.0 + fallback_penalty * self.fallback_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "observations": self.observations,
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallback_rate,
            "parity_known": self.parity_known,
            "parity_rate": self.parity_rate,
            "ttft_seconds": self.ttft_seconds.to_dict(),
            "seconds_per_output_token": self.seconds_per_output_token.to_dict(),
            "accepted_per_round": self.accepted_per_round.to_dict(),
            "total_seconds_per_round": self.total_seconds_per_round.to_dict(),
            "verify_seconds_per_round": self.verify_seconds_per_round.to_dict(),
            "target_forwards_per_output_token": (
                self.target_forwards_per_output_token.to_dict()
            ),
        }
