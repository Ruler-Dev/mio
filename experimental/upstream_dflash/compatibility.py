"""Evidence-based promotion gates for the upstream DFlash fast path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


GateStatus = Literal["pass", "conditional", "block"]
ToolMode = Literal["none", "prompt_only", "required_dynamic_eos"]
PrefixCacheMode = Literal["none", "upstream_snapshot", "mio_warm_state"]


@dataclass(frozen=True)
class ParityCertificate:
    """Model/config/version-scoped evidence; never a global parity claim."""

    target_reference: str
    draft_reference: str
    target_config_sha256: str
    draft_config_sha256: str
    mlx_version: str
    mlx_lm_version: str
    dflash_mlx_version: str
    verify_mode: str
    draft_quant: str | None
    block_tokens: int | None
    verify_len_cap: int | None
    parity_rate: float
    eligible_pairs: int
    distinct_prompts: int
    fallback_count: int
    strict_passed: bool
    source_created_at: str
    source_git_revision: str

    @classmethod
    def from_matched_benchmark(
        cls,
        payload: dict[str, Any],
        *,
        candidate: str = "dflash-mlx",
    ) -> "ParityCertificate":
        comparison = payload["paired_comparisons"][candidate]
        provenance = payload["provenance"]
        models = provenance["models"]
        software = provenance["software"]
        config = payload["configuration"]
        pairs = [pair for pair in comparison.get("pairs", ()) if pair.get("eligible")]
        prompt_ids = {str(pair["prompt_id"]) for pair in pairs}
        draft = models[f"{candidate}_draft"]
        return cls(
            target_reference=str(models["target"]["reference"]),
            draft_reference=str(draft["reference"]),
            target_config_sha256=str(models["target"]["config_sha256"]),
            draft_config_sha256=str(draft["config_sha256"]),
            mlx_version=str(software["mlx"]),
            mlx_lm_version=str(software["mlx-lm"]),
            dflash_mlx_version=str(software["dflash-mlx"]),
            verify_mode=str(config["dflash_verify_mode"]),
            draft_quant=(
                str(config["dflash_draft_quant"])
                if config.get("dflash_draft_quant") is not None
                else None
            ),
            block_tokens=(
                int(config["dflash_block_tokens"])
                if config.get("dflash_block_tokens") is not None
                else None
            ),
            verify_len_cap=(
                int(config["dflash_verify_cap"])
                if config.get("dflash_verify_cap") is not None
                else None
            ),
            parity_rate=float(comparison["parity_rate"]),
            eligible_pairs=int(comparison["eligible_pairs"]),
            distinct_prompts=len(prompt_ids),
            fallback_count=int(comparison["fallback_count"]),
            strict_passed=bool(payload["checks"]["strict_passed"]),
            source_created_at=str(payload["created_at"]),
            source_git_revision=str(provenance["git"]["revision"]),
        )


@dataclass(frozen=True)
class PromotionRequest:
    target_reference: str
    draft_reference: str
    target_config_sha256: str
    draft_config_sha256: str
    mlx_version: str
    mlx_lm_version: str
    dflash_mlx_version: str
    verify_mode: str = "dflash"
    draft_quant: str | None = "w4:gs64"
    block_tokens: int | None = None
    verify_len_cap: int | None = None
    sampling: bool = False
    streaming: bool = True
    uses_token_stop_ids: bool = True
    uses_static_suppress_ids: bool = False
    dynamic_suppress_after: bool = False
    text_stop_sequences: bool = False
    tool_mode: ToolMode = "none"
    tq_bits: int | None = None
    pq_bits: int | None = None
    quantize_kv_cache: bool = False
    prefix_cache_mode: PrefixCacheMode = "none"
    require_logprobs: bool = False
    parity: ParityCertificate | None = None


@dataclass(frozen=True)
class CompatibilityGate:
    name: str
    status: GateStatus
    detail: str


@dataclass(frozen=True)
class CompatibilityReport:
    gates: tuple[CompatibilityGate, ...]

    @property
    def prototype_eligible(self) -> bool:
        """No known semantic blocker; conditional work may remain."""

        return not any(gate.status == "block" for gate in self.gates)

    @property
    def ready_for_default(self) -> bool:
        """Every required behavior is implemented and certified."""

        return all(gate.status == "pass" for gate in self.gates)

    @property
    def blockers(self) -> tuple[CompatibilityGate, ...]:
        return tuple(gate for gate in self.gates if gate.status == "block")

    @property
    def conditions(self) -> tuple[CompatibilityGate, ...]:
        return tuple(gate for gate in self.gates if gate.status == "conditional")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prototype_eligible": self.prototype_eligible,
            "ready_for_default": self.ready_for_default,
            "gates": [
                {"name": gate.name, "status": gate.status, "detail": gate.detail}
                for gate in self.gates
            ],
        }


def _gate(name: str, status: GateStatus, detail: str) -> CompatibilityGate:
    return CompatibilityGate(name=name, status=status, detail=detail)


def _parity_gate(request: PromotionRequest) -> CompatibilityGate:
    evidence = request.parity
    if evidence is None:
        return _gate("parity", "block", "A model/config/version-scoped parity certificate is required.")

    identity_fields = (
        "target_reference",
        "draft_reference",
        "target_config_sha256",
        "draft_config_sha256",
        "mlx_version",
        "mlx_lm_version",
        "dflash_mlx_version",
        "verify_mode",
        "draft_quant",
        "block_tokens",
        "verify_len_cap",
    )
    mismatches = [
        field
        for field in identity_fields
        if getattr(request, field) != getattr(evidence, field)
    ]
    if mismatches:
        return _gate(
            "parity",
            "block",
            "Certificate identity mismatch: " + ", ".join(mismatches),
        )
    if not evidence.strict_passed:
        return _gate("parity", "block", "The source benchmark did not pass strict checks.")
    if evidence.parity_rate != 1.0 or evidence.fallback_count != 0:
        return _gate("parity", "block", "Promotion requires 100% parity and zero fallbacks.")
    if evidence.eligible_pairs < 12 or evidence.distinct_prompts < 4:
        return _gate("parity", "block", "Promotion requires at least 12 pairs across 4 prompts.")
    return _gate(
        "parity",
        "pass",
        "Exact identity matched a strict 12-pair/4-prompt-or-larger parity corpus.",
    )


def assess_promotion(request: PromotionRequest) -> CompatibilityReport:
    """Assess semantic compatibility without mutating production configuration."""

    gates: list[CompatibilityGate] = []
    gates.append(
        _gate(
            "greedy_sampling",
            "block" if request.sampling else "pass",
            (
                "Upstream DFlash has no exact speculative sampling contract; use target-only/DSpark."
                if request.sampling
                else "Greedy generation matches the upstream acceptance contract."
            ),
        )
    )
    gates.append(
        _gate(
            "streaming",
            "pass",
            "Native prefill/token/summary events are translated and generator close propagates cancellation."
            if request.streaming
            else "The event stream can also be collected for non-streaming requests.",
        )
    )
    if request.uses_token_stop_ids or request.uses_static_suppress_ids:
        gates.append(
            _gate(
                "token_stop_and_suppress",
                "pass",
                "Upstream accepts stop_token_ids and a static suppress_token_ids mask.",
            )
        )
    if request.dynamic_suppress_after:
        gates.append(
            _gate(
                "dynamic_suppression",
                "block",
                "Mio's suppress-then-relax tool policy has no upstream request parameter.",
            )
        )
    if request.text_stop_sequences:
        gates.append(
            _gate(
                "text_stops",
                "conditional",
                "Text stop strings remain a buffered downstream Mio concern; test split-token boundaries.",
            )
        )
    if request.tool_mode == "prompt_only":
        gates.append(
            _gate(
                "tools",
                "conditional",
                "Prepared tool-template tokens pass through exactly; add tool-call corpus parity and parser tests.",
            )
        )
    elif request.tool_mode == "required_dynamic_eos":
        gates.append(
            _gate(
                "tools",
                "block",
                "Required tool calls depend on Mio's dynamic EOS suppression/relaxation semantics.",
            )
        )
    else:
        gates.append(_gate("tools", "pass", "No tool-specific generation semantics requested."))

    if request.tq_bits is not None or request.pq_bits is not None:
        gates.append(
            _gate(
                "pq_tq",
                "block",
                "Upstream TargetOps does not construct Mio PolarQuant/TurboQuant cache classes.",
            )
        )
    else:
        gates.append(_gate("pq_tq", "pass", "PQ/TQ are disabled for this fast-path request."))

    gates.append(
        _gate(
            "kv8",
            "conditional" if request.quantize_kv_cache else "pass",
            (
                "Upstream supports native 8-bit KV, but it needs a separate parity/performance certificate."
                if request.quantize_kv_cache
                else "Unquantized target KV matches the current certificate configuration."
            ),
        )
    )
    if request.prefix_cache_mode == "mio_warm_state":
        gates.append(
            _gate(
                "prefix_cache",
                "block",
                "Mio warm_state cache objects are not ABI-compatible with DFlashPrefixSnapshot.",
            )
        )
    elif request.prefix_cache_mode == "upstream_snapshot":
        gates.append(
            _gate(
                "prefix_cache",
                "conditional",
                "Use upstream SnapshotService end to end and certify restore/seam parity before defaulting it.",
            )
        )
    else:
        gates.append(_gate("prefix_cache", "pass", "Cold/no-prefix-cache request."))

    gates.append(
        _gate(
            "logprobs",
            "block" if request.require_logprobs else "pass",
            (
                "Upstream TokenEvent does not expose per-token log probabilities."
                if request.require_logprobs
                else "No logprob payload requested."
            ),
        )
    )
    gates.append(_parity_gate(request))
    return CompatibilityReport(gates=tuple(gates))
