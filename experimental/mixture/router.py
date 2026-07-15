"""Deterministic and guarded online router for a mixture of drafters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

from .model import ArmCurve, DrafterObservation, RouteContext


@dataclass(frozen=True)
class RouterConfig:
    """Safety controls for the experimental router.

    The default exploration budget covers one calibration observation per arm
    and then stops.  Additional deterministic exploration can be enabled by
    increasing ``max_exploration_decisions`` and setting a positive interval.
    """

    warmup_samples_per_arm: int = 1
    max_exploration_decisions: int = 2
    exploration_interval: int = 0
    min_context_samples_per_arm: int = 2
    switch_margin: float = 0.05
    switch_patience: int = 2
    guard_regression_margin: float = 0.10
    guard_patience: int = 2
    fallback_penalty: float = 2.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.warmup_samples_per_arm,
            self.max_exploration_decisions,
            self.exploration_interval,
            self.min_context_samples_per_arm,
            self.switch_patience,
            self.guard_patience,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("router integer controls must be non-negative")
        if self.warmup_samples_per_arm < 1:
            raise ValueError("warmup_samples_per_arm must be at least one")
        if self.switch_patience < 1 or self.guard_patience < 1:
            raise ValueError("patience controls must be at least one")
        if not 0 <= self.switch_margin < 1:
            raise ValueError("switch_margin must be in [0, 1)")
        if self.guard_regression_margin < 0:
            raise ValueError("guard_regression_margin must be non-negative")
        if self.fallback_penalty < 0:
            raise ValueError("fallback_penalty must be non-negative")


@dataclass(frozen=True)
class Decision:
    request_id: str
    arm: str
    reason: str
    decision_index: int
    exploration: bool
    static_arm: str | None
    curve_source: str
    predicted_seconds: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OnlineDrafterRouter:
    """Choose one drafter without paying for both candidates.

    The router learns only from the chosen arm.  It first performs bounded,
    deterministic calibration, estimates an acceptance/cost/verification curve
    per arm, and defaults to the globally best observed static arm.  A
    contextual challenger must clear a hysteresis margin for consecutive
    decisions.  Realized regressions or fallbacks latch the router back to the
    best static arm.
    """

    def __init__(
        self,
        arms: Iterable[str],
        *,
        config: RouterConfig | None = None,
    ) -> None:
        ordered = tuple(dict.fromkeys(str(arm) for arm in arms))
        if len(ordered) < 2:
            raise ValueError("mixture routing needs at least two distinct arms")
        if any(not arm for arm in ordered):
            raise ValueError("arm names must not be empty")
        self.arms = ordered
        self.config = config or RouterConfig(
            max_exploration_decisions=len(ordered)
        )
        minimum_calibration = self.config.warmup_samples_per_arm * len(self.arms)
        if self.config.max_exploration_decisions < minimum_calibration:
            raise ValueError(
                "max_exploration_decisions cannot be smaller than required calibration"
            )
        self.global_curves = {arm: ArmCurve(arm) for arm in self.arms}
        self.context_curves: dict[str, dict[str, ArmCurve]] = {}
        self.telemetry: list[dict[str, Any]] = []
        self.decision_count = 0
        self.observation_count = 0
        self.exploration_count = 0
        self._last_exploration_decision = 0
        self._active_arm: str | None = None
        self._challenger: str | None = None
        self._challenger_streak = 0
        self._guard_streak = 0
        self._fallback_latched = False

    @property
    def fallback_latched(self) -> bool:
        return self._fallback_latched

    def _context_curves(self, bucket: str) -> dict[str, ArmCurve]:
        if bucket not in self.context_curves:
            self.context_curves[bucket] = {
                arm: ArmCurve(arm) for arm in self.arms
            }
        return self.context_curves[bucket]

    def best_static_arm(self) -> str | None:
        observed = [
            arm for arm in self.arms if self.global_curves[arm].observations > 0
        ]
        if not observed:
            return None
        return min(
            observed,
            key=lambda arm: (
                self.global_curves[arm].static_cost(
                    fallback_penalty=self.config.fallback_penalty
                ),
                self.arms.index(arm),
            ),
        )

    def _predictions(
        self,
        context: RouteContext,
    ) -> tuple[dict[str, float], str]:
        local = self._context_curves(context.bucket)
        local_ready = all(
            local[arm].observations >= self.config.min_context_samples_per_arm
            for arm in self.arms
        )
        curves = local if local_ready else self.global_curves
        source = "context" if local_ready else "global"
        return (
            {
                arm: curves[arm].risk_adjusted_prediction(
                    context,
                    fallback_penalty=self.config.fallback_penalty,
                )
                for arm in self.arms
            },
            source,
        )

    def _decision(
        self,
        context: RouteContext,
        *,
        arm: str,
        reason: str,
        exploration: bool,
        static_arm: str | None,
        curve_source: str,
        predictions: dict[str, float] | None = None,
    ) -> Decision:
        encoded = {
            name: (value if math.isfinite(value) else None)
            for name, value in (predictions or {}).items()
        }
        return Decision(
            request_id=context.request_id,
            arm=arm,
            reason=reason,
            decision_index=self.decision_count,
            exploration=exploration,
            static_arm=static_arm,
            curve_source=curve_source,
            predicted_seconds=encoded,
        )

    def route(self, context: RouteContext) -> Decision:
        self.decision_count += 1

        calibration_candidates = [
            arm
            for arm in self.arms
            if self.global_curves[arm].observations
            < self.config.warmup_samples_per_arm
        ]
        if calibration_candidates:
            arm = min(
                calibration_candidates,
                key=lambda name: (
                    self.global_curves[name].observations,
                    self.arms.index(name),
                ),
            )
            self.exploration_count += 1
            self._last_exploration_decision = self.decision_count
            return self._decision(
                context,
                arm=arm,
                reason="deterministic_calibration",
                exploration=True,
                static_arm=self.best_static_arm(),
                curve_source="calibration",
            )

        static_arm = self.best_static_arm()
        if static_arm is None:
            raise RuntimeError("calibration completed without an observed arm")
        predictions, curve_source = self._predictions(context)

        if self._fallback_latched:
            self._active_arm = static_arm
            return self._decision(
                context,
                arm=static_arm,
                reason="guarded_static_fallback",
                exploration=False,
                static_arm=static_arm,
                curve_source=curve_source,
                predictions=predictions,
            )

        due_for_exploration = (
            self.config.exploration_interval > 0
            and self.exploration_count < self.config.max_exploration_decisions
            and self.decision_count - self._last_exploration_decision
            >= self.config.exploration_interval
        )
        if due_for_exploration:
            challengers = [arm for arm in self.arms if arm != static_arm]
            arm = min(
                challengers,
                key=lambda name: (
                    self.global_curves[name].observations,
                    self.arms.index(name),
                ),
            )
            self.exploration_count += 1
            self._last_exploration_decision = self.decision_count
            return self._decision(
                context,
                arm=arm,
                reason="bounded_exploration",
                exploration=True,
                static_arm=static_arm,
                curve_source=curve_source,
                predictions=predictions,
            )

        candidate = min(
            self.arms,
            key=lambda arm: (predictions[arm], self.arms.index(arm)),
        )
        if self._active_arm is None:
            # The first post-calibration choice establishes the best measured
            # static arm; hysteresis applies only to subsequent changes.
            self._active_arm = static_arm

        active = self._active_arm
        active_score = predictions[active]
        candidate_score = predictions[candidate]
        improves_enough = (
            candidate != active
            and math.isfinite(candidate_score)
            and (
                not math.isfinite(active_score)
                or candidate_score <= active_score * (1.0 - self.config.switch_margin)
            )
        )
        if improves_enough:
            if self._challenger == candidate:
                self._challenger_streak += 1
            else:
                self._challenger = candidate
                self._challenger_streak = 1
            if self._challenger_streak >= self.config.switch_patience:
                self._active_arm = candidate
                self._challenger = None
                self._challenger_streak = 0
                reason = "hysteresis_switch"
            else:
                reason = "hysteresis_hold"
        else:
            self._challenger = None
            self._challenger_streak = 0
            # A contextual model without a material win falls back to the
            # globally best static arm.
            self._active_arm = static_arm
            reason = "best_static"

        return self._decision(
            context,
            arm=self._active_arm,
            reason=reason,
            exploration=False,
            static_arm=static_arm,
            curve_source=curve_source,
            predictions=predictions,
        )

    def observe(
        self,
        context: RouteContext,
        decision: Decision,
        observation: DrafterObservation,
    ) -> None:
        if decision.request_id != context.request_id:
            raise ValueError("decision and context request ids differ")
        if decision.arm != observation.arm:
            raise ValueError("feedback must belong to the selected arm")

        static_prediction = decision.predicted_seconds.get(decision.static_arm or "")
        if observation.fallback:
            self._fallback_latched = True
            self._guard_streak = self.config.guard_patience
        elif (
            decision.static_arm is not None
            and decision.arm != decision.static_arm
            and static_prediction is not None
            and observation.wall_seconds
            > static_prediction * (1.0 + self.config.guard_regression_margin)
        ):
            self._guard_streak += 1
            if self._guard_streak >= self.config.guard_patience:
                self._fallback_latched = True
        else:
            self._guard_streak = 0

        self.global_curves[observation.arm].update(observation)
        self._context_curves(context.bucket)[observation.arm].update(observation)
        self.observation_count += 1
        self.telemetry.append(
            {
                "decision": decision.to_dict(),
                "context": {
                    "request_id": context.request_id,
                    "prompt_tokens": context.prompt_tokens,
                    "requested_tokens": context.requested_tokens,
                    "workload": context.workload,
                    "bucket": context.bucket,
                },
                "observation": observation.to_dict(),
                "guard": {
                    "streak": self._guard_streak,
                    "fallback_latched": self._fallback_latched,
                },
                "online_update_arms": [observation.arm],
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "config": asdict(self.config),
            "decision_count": self.decision_count,
            "observation_count": self.observation_count,
            "exploration_count": self.exploration_count,
            "active_arm": self._active_arm,
            "best_static_arm": self.best_static_arm(),
            "fallback_latched": self._fallback_latched,
            "curves": {
                arm: self.global_curves[arm].to_dict() for arm in self.arms
            },
        }
