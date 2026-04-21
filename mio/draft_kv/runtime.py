"""SP runtime — half-depth prefill + projected KV for late layers.

This is the orchestrator. In the final design it would:
 1. Run the target model on the prompt through layers 0..N_early-1, capturing
    per-layer K/V in a normal cache + the hidden state at layer N_early.
 2. Apply the ConfidenceGate to the captured hidden.
 3. If gate fails → fall back to `fallback_fn()` (full prefill) and return.
 4. If gate passes → for each late layer l in [N_early, N_layers):
        K_l, V_l = projector.project(hidden, layer_idx=l, kv_shape=shapes[l])
        install K_l, V_l into cache[l]
 5. Return (cache, target_hidden_for_draft, offset) so decode can start.

SCAFFOLD: step 1 requires a model-family-specific partial forward that
captures intermediate hiddens — not yet wired. The current `sp_prefill`
always takes the fallback path, calls `fallback_fn()`, and returns its
result. Integration tests exercise the control flow (gate eval, projector
invocation, fallback dispatch) using synthetic tensors.

When a trained projector ships and the partial-forward hook exists, only
the `_run_partial_target_prefill` body needs filling in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx

from mio.draft_kv.gate import ConfidenceGate, GateDecision
from mio.draft_kv.projector import KVProjector, KVShape


@dataclass
class SPResult:
    """Outcome of an SP prefill attempt.

    Either fallback=True and `fallback_value` holds whatever the full-
    prefill callable returned, or fallback=False and `caches`/`hidden`
    are the populated SP cache list and the target_hidden feature vector
    needed to kick off the first decode cycle.
    """

    fallback: bool
    gate: Optional[GateDecision] = None
    fallback_value: Any = None
    caches: Optional[list[Any]] = None
    hidden: Optional[mx.array] = None
    stats: dict[str, Any] = field(default_factory=dict)


def _run_partial_target_prefill(
    *,
    target_model: Any,
    input_ids: mx.array,
    early_layer: int,
) -> tuple[mx.array, list[Any]]:
    """Run target layers 0..early_layer and return (hidden, caches_so_far).

    NOT IMPLEMENTED in the scaffold. When implemented, this should behave
    like `target_forward_with_hidden_states` but stop after `early_layer`
    layers and return the live hidden state at that boundary plus any
    caches populated so far.
    """
    raise NotImplementedError(
        "Partial target prefill requires a model-family-specific hook that "
        "doesn't exist yet. SP currently always takes the fallback path."
    )


def sp_prefill(
    *,
    target_model: Any,
    input_ids: mx.array,
    projector: KVProjector,
    gate: ConfidenceGate,
    kv_shapes: list[KVShape],
    early_layer: int,
    fallback_fn: Callable[[], Any],
    force_fallback: bool = False,
) -> SPResult:
    """Run SP prefill if gate passes; else call fallback_fn and return.

    Args:
        target_model: loaded MLX target model (unused in scaffold).
        input_ids: (B, L) prompt tokens.
        projector: maps intermediate hidden → (K, V) per late layer.
        gate: confidence gate on the intermediate hidden.
        kv_shapes: per-layer KVShape for layers [early_layer, N_layers).
        early_layer: how many layers to run fully through target (≥ 1).
        fallback_fn: zero-arg callable invoked when SP is skipped.
        force_fallback: debug / dry-run switch; bypasses the try block and
            always calls fallback_fn.

    Returns:
        SPResult with either populated caches or the fallback value.
    """
    if early_layer <= 0:
        raise ValueError("early_layer must be positive")
    if not kv_shapes:
        raise ValueError("kv_shapes must not be empty")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be (B, L), got {input_ids.shape}")

    stats: dict[str, Any] = {
        "early_layer": early_layer,
        "late_layer_count": len(kv_shapes),
        "input_len": int(input_ids.shape[-1]),
    }

    if force_fallback:
        stats["path"] = "forced_fallback"
        return SPResult(
            fallback=True,
            fallback_value=fallback_fn(),
            stats=stats,
        )

    try:
        hidden, partial_caches = _run_partial_target_prefill(
            target_model=target_model,
            input_ids=input_ids,
            early_layer=early_layer,
        )
    except NotImplementedError as exc:
        stats["path"] = "fallback"
        stats["reason"] = str(exc)
        return SPResult(
            fallback=True,
            fallback_value=fallback_fn(),
            stats=stats,
        )

    decision = gate.evaluate(hidden)
    stats["gate_mean_norm"] = decision.mean_norm
    stats["gate_outlier_fraction"] = decision.outlier_fraction
    if not decision.proceed:
        stats["path"] = "gate_reject"
        stats["reason"] = decision.reason
        return SPResult(
            fallback=True,
            gate=decision,
            fallback_value=fallback_fn(),
            stats=stats,
        )

    late_caches: list[tuple[mx.array, mx.array]] = []
    for offset, shape in enumerate(kv_shapes):
        layer_idx = early_layer + offset
        k, v = projector.project(hidden, layer_idx=layer_idx, kv_shape=shape)
        late_caches.append((k, v))

    stats["path"] = "sp"
    return SPResult(
        fallback=False,
        gate=decision,
        caches=partial_caches + late_caches,
        hidden=hidden,
        stats=stats,
    )
