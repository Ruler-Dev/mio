"""Deterministic Markov-tree controller for conditional quality effort.

This module is intentionally independent from Mio's production runtime.  It
implements a finite-state greedy Markov policy: the next action is a pure
function of immutable request state, a frozen offline transition model, and a
fixed effort profile.  Candidate generations form a tree, while explicit
token and end-to-end latency allocations bound extra work.  It does not do
Bellman backups, value iteration, look-ahead search, or online planning.

The controller does not alter decoding or prefill.  A one-candidate request
therefore has the same model path as the baseline; extra generations are only
issued after a validator failure or a calibrated uncertainty trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import math
from types import MappingProxyType
from typing import Iterable, Mapping


class EffortTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    ULTRA = "ultra"


class ValidationOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Trigger(StrEnum):
    VALIDATOR_FAILURE = "validator_failure"
    CALIBRATED_UNCERTAINTY = "calibrated_uncertainty"


class ControllerAction(StrEnum):
    GENERATE_DIRECT = "generate_direct"
    GENERATE_REPAIR = "generate_repair"
    GENERATE_ALTERNATIVE = "generate_alternative"
    GENERATE_REFINE = "generate_refine"
    ACCEPT = "accept"
    STOP = "stop"

    @property
    def generates_candidate(self) -> bool:
        return self in {
            ControllerAction.GENERATE_DIRECT,
            ControllerAction.GENERATE_REPAIR,
            ControllerAction.GENERATE_ALTERNATIVE,
            ControllerAction.GENERATE_REFINE,
        }


EXTRA_ACTIONS = (
    ControllerAction.GENERATE_REPAIR,
    ControllerAction.GENERATE_ALTERNATIVE,
    ControllerAction.GENERATE_REFINE,
)


@dataclass(frozen=True)
class CalibrationIdentity:
    """Exact experiment identity required to reuse frozen transition bounds."""

    model: str
    config: str
    prompt: str
    sampler: str
    corpus: str
    split: str
    backend: str

    def __post_init__(self) -> None:
        values = (
            self.model,
            self.config,
            self.prompt,
            self.sampler,
            self.corpus,
            self.split,
            self.backend,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("all calibration identity fields must be non-empty strings")


@dataclass(frozen=True)
class BootstrapMetadata:
    """Provenance for conservative bounds computed by an offline harness."""

    task_cluster_count: int
    resamples: int
    confidence_level: float
    method: str
    seed: int

    def __post_init__(self) -> None:
        if self.task_cluster_count < 1:
            raise ValueError("task_cluster_count must be positive")
        if self.resamples < 1:
            raise ValueError("bootstrap resamples must be positive")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if not self.method:
            raise ValueError("bootstrap method must not be empty")
        if self.seed < 0:
            raise ValueError("bootstrap seed must be non-negative")


@dataclass(frozen=True)
class EffortProfile:
    """A hard envelope around conditional generation.

    ``max_latency_ratio`` is converted to a per-candidate end-to-end allocation
    after the direct baseline is observed.  The experiment harness must enforce
    it across generation, validation, and controller overhead; any overshoot is
    retained as a measurable budget violation rather than silently discarded.
    """

    max_candidates: int
    max_extra_output_tokens: int
    max_output_tokens_per_candidate: int
    max_latency_ratio: float
    uncertainty_threshold: float
    min_task_clusters: int = 8
    min_success_lcb: float = 0.10

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.max_extra_output_tokens < 0:
            raise ValueError("max_extra_output_tokens must be non-negative")
        if self.max_output_tokens_per_candidate < 1:
            raise ValueError("max_output_tokens_per_candidate must be positive")
        if self.max_latency_ratio < 1.0 or not math.isfinite(self.max_latency_ratio):
            raise ValueError("max_latency_ratio must be finite and at least one")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")
        if self.min_task_clusters < 2:
            raise ValueError("min_task_clusters must be at least two")
        if not 0.0 <= self.min_success_lcb <= 1.0:
            raise ValueError("min_success_lcb must be in [0, 1]")


EFFORT_PROFILES: Mapping[EffortTier, EffortProfile] = MappingProxyType(
    {
        # Low is a zero-overhead direct path: it never schedules a second
        # generation.  The remaining profiles expose increasingly wider hard
        # envelopes, but the controller still spends only after observable,
        # calibrated evidence warrants an extra transition.
        EffortTier.LOW: EffortProfile(1, 0, 96, 1.0, 1.0),
        # Medium spends compute only after an exact validator failure.
        EffortTier.MEDIUM: EffortProfile(2, 128, 128, 1.75, 1.0),
        EffortTier.HIGH: EffortProfile(3, 256, 160, 2.50, 0.80),
        EffortTier.XHIGH: EffortProfile(4, 384, 192, 3.25, 0.72),
        EffortTier.ULTRA: EffortProfile(5, 640, 224, 4.50, 0.65),
    }
)


@dataclass(frozen=True)
class GenerationMetrics:
    """Backend timing split needed to audit prefill and decode separately."""

    prompt_tokens: int
    output_tokens: int
    prefill_seconds: float
    decode_seconds: float
    other_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        values = (self.prefill_seconds, self.decode_seconds, self.other_seconds)
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            raise ValueError("timings must be finite and non-negative")

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds + self.other_seconds


@dataclass(frozen=True)
class CandidateEvidence:
    """Observable evidence returned by one generation and its cheap verifier.

    ``evaluation_score`` is held-out reporting data.  The controller never
    reads it for routing or candidate selection.
    """

    metrics: GenerationMetrics
    validator: ValidationOutcome = ValidationOutcome.UNKNOWN
    evaluation_score: float | None = None
    uncertainty: float | None = None
    validation_seconds: float = 0.0
    controller_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.evaluation_score is not None and not 0.0 <= self.evaluation_score <= 1.0:
            raise ValueError("evaluation_score must be in [0, 1]")
        if self.uncertainty is not None and not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        overhead = (self.validation_seconds, self.controller_seconds)
        if any(value < 0.0 or not math.isfinite(value) for value in overhead):
            raise ValueError("validation/controller timings must be finite and non-negative")

    @property
    def e2e_seconds(self) -> float:
        return (
            self.metrics.total_seconds
            + self.validation_seconds
            + self.controller_seconds
        )


@dataclass(frozen=True)
class CandidateNode:
    node_id: int
    parent_id: int | None
    action: ControllerAction
    evidence: CandidateEvidence
    allocated_e2e_seconds: float | None
    transition_source: str | None

    @property
    def deadline_exceeded(self) -> bool:
        return bool(
            self.allocated_e2e_seconds is not None
            and self.evidence.e2e_seconds > self.allocated_e2e_seconds
        )


@dataclass(frozen=True)
class TransitionEstimate:
    """Frozen offline evidence for one Markov transition.

    A success means that taking ``action`` from the given trigger/depth changed
    a failing or unknown candidate into a validator pass on held-out data.
    The offline latency UCB is stored as an end-to-end ratio to that request's
    direct baseline so the table is portable while remaining auditable on each
    machine.
    """

    context_bucket: str
    trigger: Trigger
    depth: int
    action: ControllerAction
    conservative_success_lcb: float
    extra_output_tokens_ucb: float
    extra_e2e_latency_ratio_ucb: float
    bootstrap: BootstrapMetadata

    def __post_init__(self) -> None:
        if not self.context_bucket:
            raise ValueError("context_bucket must not be empty")
        if self.action not in EXTRA_ACTIONS:
            raise ValueError("transition estimates apply only to extra actions")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= self.conservative_success_lcb <= 1.0:
            raise ValueError("conservative_success_lcb must be in [0, 1]")
        if self.extra_output_tokens_ucb <= 0 or not math.isfinite(
            self.extra_output_tokens_ucb
        ):
            raise ValueError("extra_output_tokens_ucb must be finite and positive")
        if self.extra_e2e_latency_ratio_ucb <= 0 or not math.isfinite(
            self.extra_e2e_latency_ratio_ucb
        ):
            raise ValueError("extra_e2e_latency_ratio_ucb must be finite and positive")


class FrozenTransitionModel:
    """Immutable lookup table learned outside the request hot path."""

    def __init__(
        self,
        identity: CalibrationIdentity,
        estimates: Iterable[TransitionEstimate] = (),
    ) -> None:
        self.identity = identity
        rows: dict[
            tuple[str, Trigger, int, ControllerAction], TransitionEstimate
        ] = {}
        for estimate in estimates:
            key = (
                estimate.context_bucket,
                estimate.trigger,
                estimate.depth,
                estimate.action,
            )
            if key in rows:
                raise ValueError(f"duplicate transition estimate: {key}")
            rows[key] = estimate
        self._rows = MappingProxyType(rows)

    def lookup(
        self,
        *,
        context_bucket: str,
        trigger: Trigger,
        depth: int,
        action: ControllerAction,
    ) -> tuple[TransitionEstimate | None, str | None]:
        exact_key = (context_bucket, trigger, depth, action)
        if exact_key in self._rows:
            return self._rows[exact_key], context_bucket
        global_key = ("global", trigger, depth, action)
        if global_key in self._rows:
            return self._rows[global_key], "global"
        return None, None


@dataclass(frozen=True)
class ControllerState:
    """Complete observable state; no hidden controller history is consulted."""

    request_id: str
    context_bucket: str
    tier: EffortTier
    nodes: tuple[CandidateNode, ...] = ()
    terminal_action: ControllerAction | None = None
    terminal_reason: str | None = None
    selected_node_id: int | None = None

    @property
    def terminal(self) -> bool:
        return self.terminal_action is not None


@dataclass(frozen=True)
class Decision:
    action: ControllerAction
    reason: str
    node_id: int | None = None
    parent_id: int | None = None
    max_output_tokens: int | None = None
    max_additional_e2e_seconds: float | None = None
    seed: int | None = None
    transition_source: str | None = None
    predicted_success_lcb: float | None = None


@dataclass(frozen=True)
class TraceMetrics:
    candidates: int
    baseline_prefill_tokens_per_second: float | None
    aggregate_prefill_tokens_per_second: float | None
    prefill_speed_ratio: float | None
    baseline_decode_tokens_per_second: float | None
    aggregate_decode_tokens_per_second: float | None
    decode_speed_ratio: float | None
    generation_latency_ratio: float | None
    e2e_latency_ratio: float | None
    effective_selected_tokens_per_second: float | None
    controller_overhead_fraction: float
    validation_overhead_fraction: float
    evaluation_delta: float | None
    deadline_violations: int


def _safe_rate(tokens: int, seconds: float) -> float | None:
    return tokens / seconds if tokens > 0 and seconds > 0.0 else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _best_node(nodes: tuple[CandidateNode, ...]) -> CandidateNode:
    if not nodes:
        raise ValueError("at least one candidate is required")

    validation_rank = {
        ValidationOutcome.FAIL: 0,
        ValidationOutcome.UNKNOWN: 1,
        ValidationOutcome.PASS: 2,
    }

    def rank(node: CandidateNode) -> tuple[float, ...]:
        evidence = node.evidence
        uncertainty = evidence.uncertainty if evidence.uncertainty is not None else 1.0
        return (
            float(validation_rank[evidence.validator]),
            -uncertainty,
            -float(evidence.metrics.output_tokens),
            -float(node.node_id),
        )

    return max(nodes, key=rank)


class MarkovTreeEffortController:
    """Finite-state greedy Markov policy over a bounded candidate tree.

    This is a replayable one-step policy, not a Bellman-optimal controller or
    a tree-planning algorithm.
    """

    _action_order = {
        ControllerAction.GENERATE_REPAIR: 0,
        ControllerAction.GENERATE_ALTERNATIVE: 1,
        ControllerAction.GENERATE_REFINE: 2,
    }

    def __init__(
        self,
        *,
        tier: EffortTier | str,
        transition_model: FrozenTransitionModel,
        calibration_identity: CalibrationIdentity,
        initial_max_output_tokens: int,
        profile: EffortProfile | None = None,
        seed_salt: str = "mio-markov-effort-v1",
    ) -> None:
        self.tier = EffortTier(tier)
        self.profile = profile or EFFORT_PROFILES[self.tier]
        self.transition_model = transition_model
        if transition_model.identity != calibration_identity:
            raise ValueError("transition calibration identity mismatch")
        self.calibration_identity = calibration_identity
        if initial_max_output_tokens < 1:
            raise ValueError("initial_max_output_tokens must be positive")
        if not seed_salt:
            raise ValueError("seed_salt must not be empty")
        self.initial_max_output_tokens = initial_max_output_tokens
        self.seed_salt = seed_salt

    def initial_state(
        self,
        request_id: str,
        *,
        context_bucket: str = "global",
    ) -> ControllerState:
        if not request_id or not context_bucket:
            raise ValueError("request_id and context_bucket must not be empty")
        return ControllerState(request_id, context_bucket, self.tier)

    def _seed(self, state: ControllerState, action: ControllerAction, node_id: int) -> int:
        payload = (
            f"{self.seed_salt}\0{state.request_id}\0{state.context_bucket}\0"
            f"{node_id}\0{action.value}"
        ).encode()
        return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")

    def _terminal_decision(
        self,
        state: ControllerState,
        *,
        reason: str,
        force_stop: bool = False,
    ) -> Decision:
        best = _best_node(state.nodes)
        action = (
            ControllerAction.STOP
            if force_stop or best.evidence.validator is ValidationOutcome.FAIL
            else ControllerAction.ACCEPT
        )
        return Decision(action=action, reason=reason, node_id=best.node_id)

    def _trigger(self, state: ControllerState) -> Trigger | None:
        latest = state.nodes[-1].evidence
        if latest.validator is ValidationOutcome.FAIL:
            return Trigger.VALIDATOR_FAILURE
        if (
            latest.validator is ValidationOutcome.UNKNOWN
            and latest.uncertainty is not None
            and latest.uncertainty >= self.profile.uncertainty_threshold
        ):
            return Trigger.CALIBRATED_UNCERTAINTY
        return None

    def decide(self, state: ControllerState) -> Decision:
        if state.tier is not self.tier:
            raise ValueError("state tier does not match controller tier")
        if state.terminal:
            raise RuntimeError("terminal state cannot be advanced")
        if not state.nodes:
            action = ControllerAction.GENERATE_DIRECT
            return Decision(
                action=action,
                reason="direct_fast_path",
                node_id=0,
                max_output_tokens=self.initial_max_output_tokens,
                seed=self._seed(state, action, 0),
            )

        if any(
            node.evidence.validator is ValidationOutcome.PASS for node in state.nodes
        ):
            return self._terminal_decision(state, reason="validator_pass")

        trigger = self._trigger(state)
        if trigger is None:
            return self._terminal_decision(state, reason="no_quality_trigger")

        if len(state.nodes) >= self.profile.max_candidates:
            return self._terminal_decision(
                state,
                reason="candidate_budget_exhausted",
                force_stop=trigger is Trigger.VALIDATOR_FAILURE,
            )

        extra_tokens_used = sum(
            node.evidence.metrics.output_tokens for node in state.nodes[1:]
        )
        remaining_tokens = self.profile.max_extra_output_tokens - extra_tokens_used
        if remaining_tokens <= 0:
            return self._terminal_decision(
                state,
                reason="token_budget_exhausted",
                force_stop=trigger is Trigger.VALIDATOR_FAILURE,
            )

        baseline_e2e_seconds = state.nodes[0].evidence.e2e_seconds
        if baseline_e2e_seconds <= 0.0:
            return self._terminal_decision(
                state,
                reason="zero_baseline_e2e_latency",
                force_stop=True,
            )
        total_e2e_seconds = sum(node.evidence.e2e_seconds for node in state.nodes)
        remaining_e2e_seconds = (
            baseline_e2e_seconds * self.profile.max_latency_ratio
            - total_e2e_seconds
        )
        if remaining_e2e_seconds <= 0.0:
            return self._terminal_decision(
                state,
                reason="latency_budget_exhausted",
                force_stop=trigger is Trigger.VALIDATOR_FAILURE,
            )

        used_actions = {node.action for node in state.nodes}
        allowed_actions = (
            EXTRA_ACTIONS
            if trigger is Trigger.VALIDATOR_FAILURE
            else (
                ControllerAction.GENERATE_ALTERNATIVE,
                ControllerAction.GENERATE_REFINE,
            )
        )
        token_allocation = min(
            remaining_tokens,
            self.profile.max_output_tokens_per_candidate,
        )
        eligible: list[
            tuple[
                float,
                float,
                int,
                ControllerAction,
                TransitionEstimate,
                str,
            ]
        ] = []
        for action in allowed_actions:
            if action in used_actions:
                continue
            estimate, source = self.transition_model.lookup(
                context_bucket=state.context_bucket,
                trigger=trigger,
                depth=len(state.nodes),
                action=action,
            )
            if estimate is None or source is None:
                continue
            success_lcb = estimate.conservative_success_lcb
            predicted_e2e_seconds = (
                baseline_e2e_seconds * estimate.extra_e2e_latency_ratio_ucb
            )
            if (
                estimate.bootstrap.task_cluster_count
                < self.profile.min_task_clusters
            ):
                continue
            if success_lcb < self.profile.min_success_lcb:
                continue
            if estimate.extra_output_tokens_ucb > token_allocation:
                continue
            if predicted_e2e_seconds > remaining_e2e_seconds:
                continue
            # Maximize conservative quality gain per unit of extra latency.
            utility = success_lcb / estimate.extra_e2e_latency_ratio_ucb
            eligible.append(
                (
                    -utility,
                    -success_lcb,
                    self._action_order[action],
                    action,
                    estimate,
                    source,
                )
            )

        if not eligible:
            return self._terminal_decision(
                state,
                reason="no_calibrated_transition_within_budget",
                force_stop=trigger is Trigger.VALIDATOR_FAILURE,
            )

        _, negative_lcb, _, action, _, source = min(eligible)
        best = _best_node(state.nodes)
        if action is ControllerAction.GENERATE_REPAIR:
            parent_id = state.nodes[-1].node_id
        elif action is ControllerAction.GENERATE_ALTERNATIVE:
            parent_id = state.nodes[0].node_id
        else:
            parent_id = best.node_id
        node_id = len(state.nodes)
        return Decision(
            action=action,
            reason=f"calibrated_{trigger.value}",
            node_id=node_id,
            parent_id=parent_id,
            max_output_tokens=token_allocation,
            max_additional_e2e_seconds=remaining_e2e_seconds,
            seed=self._seed(state, action, node_id),
            transition_source=source,
            predicted_success_lcb=-negative_lcb,
        )

    def observe(
        self,
        state: ControllerState,
        decision: Decision,
        evidence: CandidateEvidence,
    ) -> ControllerState:
        expected = self.decide(state)
        if decision != expected:
            raise ValueError("decision does not match deterministic policy replay")
        if not decision.action.generates_candidate:
            raise ValueError("terminal decisions cannot receive candidate evidence")
        if decision.node_id is None or decision.max_output_tokens is None:
            raise RuntimeError("generation decision is missing an allocation")
        if evidence.metrics.output_tokens > decision.max_output_tokens:
            raise ValueError("candidate exceeded its output-token allocation")
        if decision.parent_id is not None and not any(
            node.node_id == decision.parent_id for node in state.nodes
        ):
            raise ValueError("candidate parent is not present in the tree")
        node = CandidateNode(
            node_id=decision.node_id,
            parent_id=decision.parent_id,
            action=decision.action,
            evidence=evidence,
            allocated_e2e_seconds=decision.max_additional_e2e_seconds,
            transition_source=decision.transition_source,
        )
        return replace(state, nodes=(*state.nodes, node))

    def finish(self, state: ControllerState, decision: Decision) -> ControllerState:
        expected = self.decide(state)
        if decision != expected:
            raise ValueError("decision does not match deterministic policy replay")
        if decision.action not in {ControllerAction.ACCEPT, ControllerAction.STOP}:
            raise ValueError("generation decisions cannot finish a trace")
        if decision.node_id is None:
            raise RuntimeError("terminal decision has no selected node")
        return replace(
            state,
            terminal_action=decision.action,
            terminal_reason=decision.reason,
            selected_node_id=decision.node_id,
        )


def trace_metrics(
    state: ControllerState,
) -> TraceMetrics:
    """Compute honest backend and end-to-end rates for one replayable trace."""

    if not state.nodes:
        raise ValueError("metrics require at least one candidate")
    baseline = state.nodes[0].evidence.metrics
    baseline_evidence = state.nodes[0].evidence
    all_metrics = [node.evidence.metrics for node in state.nodes]
    prompt_tokens = sum(metrics.prompt_tokens for metrics in all_metrics)
    output_tokens = sum(metrics.output_tokens for metrics in all_metrics)
    prefill_seconds = sum(metrics.prefill_seconds for metrics in all_metrics)
    decode_seconds = sum(metrics.decode_seconds for metrics in all_metrics)
    generation_seconds = sum(metrics.total_seconds for metrics in all_metrics)
    validation_seconds = sum(
        node.evidence.validation_seconds for node in state.nodes
    )
    controller_seconds = sum(
        node.evidence.controller_seconds for node in state.nodes
    )
    e2e_seconds = generation_seconds + validation_seconds + controller_seconds

    baseline_prefill_tps = _safe_rate(
        baseline.prompt_tokens,
        baseline.prefill_seconds,
    )
    aggregate_prefill_tps = _safe_rate(prompt_tokens, prefill_seconds)
    baseline_decode_tps = _safe_rate(baseline.output_tokens, baseline.decode_seconds)
    aggregate_decode_tps = _safe_rate(output_tokens, decode_seconds)

    selected_id = state.selected_node_id
    selected = (
        next(node for node in state.nodes if node.node_id == selected_id)
        if selected_id is not None
        else _best_node(state.nodes)
    )
    effective_tps = _safe_rate(selected.evidence.metrics.output_tokens, e2e_seconds)
    baseline_score = baseline_evidence.evaluation_score
    selected_score = selected.evidence.evaluation_score
    evaluation_delta = (
        selected_score - baseline_score
        if selected_score is not None and baseline_score is not None
        else None
    )
    controller_overhead_fraction = (
        controller_seconds / e2e_seconds if e2e_seconds > 0.0 else 0.0
    )
    validation_overhead_fraction = (
        validation_seconds / e2e_seconds if e2e_seconds > 0.0 else 0.0
    )

    return TraceMetrics(
        candidates=len(state.nodes),
        baseline_prefill_tokens_per_second=baseline_prefill_tps,
        aggregate_prefill_tokens_per_second=aggregate_prefill_tps,
        prefill_speed_ratio=_safe_ratio(aggregate_prefill_tps, baseline_prefill_tps),
        baseline_decode_tokens_per_second=baseline_decode_tps,
        aggregate_decode_tokens_per_second=aggregate_decode_tps,
        decode_speed_ratio=_safe_ratio(aggregate_decode_tps, baseline_decode_tps),
        generation_latency_ratio=(
            generation_seconds / baseline.total_seconds
            if baseline.total_seconds > 0.0
            else None
        ),
        e2e_latency_ratio=(
            e2e_seconds / baseline_evidence.e2e_seconds
            if baseline_evidence.e2e_seconds > 0.0
            else None
        ),
        effective_selected_tokens_per_second=effective_tps,
        controller_overhead_fraction=controller_overhead_fraction,
        validation_overhead_fraction=validation_overhead_fraction,
        evaluation_delta=evaluation_delta,
        deadline_violations=sum(node.deadline_exceeded for node in state.nodes),
    )
