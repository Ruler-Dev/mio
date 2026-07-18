#!/usr/bin/env python3
"""Pure protocol primitives for the hard-blocked SWE-bench quality v2 study.

This module validates content-free efficiency telemetry and offline Docker
image-lock records.  It never loads a model, invokes Docker, starts the SWE-
bench evaluator, or enables confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bench_swebench_quality as v1  # noqa: E402

PREREGISTRATION_PATH = ROOT / "benchmarks" / "swebench-quality-preregistration-v2.json"
V1_PREREGISTRATION_PATH = ROOT / "benchmarks" / "swebench-quality-preregistration-v1.json"
V1_PREREGISTRATION_CANONICAL_SHA256 = "834b205733c02a81adaa8ad1cbfd3ab66bdb65575fc162742c53af246422d708"
V2_PREREGISTRATION_CANONICAL_SHA256 = "bf9429105651d5c06d14720ba9fe096c78197319072e7b9fef6de86984e0b1e2"
SCHEMA = "mio.swebench-verified-quality-preregistration.v2"
ARM_METRIC_SCHEMA = "mio.swebench-verified-quality-arm-metrics.v2"
ROUND_METRIC_SCHEMA = "mio.swebench-verified-quality-round-metrics.v2"
TOOL_METRIC_SCHEMA = "mio.swebench-verified-quality-tool-metrics.v2"
GUARDRAIL_SCHEMA = "mio.swebench-verified-quality-efficiency-aggregate.v2"
PROMOTION_SCHEMA = "mio.swebench-verified-quality-promotion-decision.v2"
DOCKER_RECORD_SCHEMA = "mio.swebench-verified-quality-docker-image-record.v2"
DOCKER_LOCK_SCHEMA = "mio.swebench-verified-quality-docker-image-lock.v2"

EXPECTED_PAIRS = 500
TARGET_CONTEXT_TOKENS = 32_768
ARM_WALL_NS = 1_800_000_000_000
MAX_OUTPUT_TOKENS_PER_ARM = 24_576
MAX_AGENT_ROUNDS = 12
MAX_TOOL_CALLS_PER_ARM = 32
MAX_PAIR_ATTEMPTS = 3
MAX_TOOL_OUTPUT_CHARS = 10_000
PROCESS_GROUP_KILL_GRACE_NS = 1_000_000_000
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_718
WALL_RATIO_LIMIT = 1.25
PREFILL_COST_RATIO_LIMIT = 1.10
DECODE_COST_RATIO_LIMIT = 1.05

CONFIRMATORY_ENABLED = False
CONFIRMATORY_BLOCKERS = (
    "v2_clean_commit_and_runtime_not_sealed",
    "native_executor_loaded_model_tree_chain_of_custody_not_attested",
    "executed_commit_and_import_origins_chain_of_custody_not_attested",
    "supervisor_private_per_run_hmac_key_not_attested",
    "promotion_input_authority_and_receipt_binding_not_implemented",
    "supervisor_monotonic_timeline_not_attested",
    "hard_arm_watchdog_not_attested",
    "known_tool_full_invocation_watchdog_not_attested",
    "raw_phase_and_tool_telemetry_not_attested",
    "whole_pair_retry_v2_not_attested",
    "docker_x86_64_image_lock_not_materialized",
    "official_harness_locked_image_flags_not_attested",
    "evaluator_timeout_outcome_classifier_not_attested",
)
RETRYABLE_INFRASTRUCTURE_REASONS = frozenset(
    {
        "infrastructure_host_loss",
        "infrastructure_process_external_kill",
        "infrastructure_storage_io_failure",
        "infrastructure_telemetry_corruption_pre_completion",
        "infrastructure_docker_daemon_failure",
        "infrastructure_locked_image_corruption",
        "infrastructure_harness_failure",
    }
)
NONRETRYABLE_OUTCOMES = frozenset(
    {
        "arm_wall_timeout",
        "model_round_timeout",
        "model_error",
        "tool_timeout",
        "tool_nonzero",
        "protocol_violation",
        "budget_exhaustion",
        "quality_gate_incomplete",
        "empty_patch",
        "patch_validation_failure",
        "official_test_timeout",
        "official_patch_apply_failure",
        "official_evaluation_unresolved",
    }
)
CONDITIONS = ("gate_off", "gate_on")
TOOL_NAMES = ("bash", "validate", "read", "write", "edit", "unknown")
KNOWN_TOOL_NAMES = frozenset(TOOL_NAMES[:-1])
TERMINAL_STATUSES = frozenset({"completed", "incomplete", "model_error", "timeout"})
TERMINATION_STATUS = {
    "completed": "completed",
    "model_final_incomplete": "incomplete",
    "quality_gate_incomplete": "incomplete",
    "budget_exhaustion": "incomplete",
    "tool_nonzero": "incomplete",
    "protocol_violation": "incomplete",
    "model_error": "model_error",
    "model_round_timeout": "timeout",
    "tool_timeout": "timeout",
    "arm_wall_timeout": "timeout",
}
TOOL_OUTCOMES = frozenset(
    {
        "ok",
        "nonzero",
        "timeout",
        "output_limit",
        "denied",
        "unrecognized",
        "unscoped",
        "untrusted_executable",
        "no_work",
        "error",
        "not_found",
        "old_string_not_found",
    }
)
TOOL_OUTCOMES_BY_NAME = {
    "bash": frozenset({"ok", "nonzero", "timeout", "output_limit", "denied", "error"}),
    "validate": frozenset(
        {
            "ok",
            "nonzero",
            "timeout",
            "output_limit",
            "denied",
            "error",
            "no_work",
            "unscoped",
            "untrusted_executable",
        }
    ),
    "read": frozenset({"ok", "not_found", "denied", "error", "timeout"}),
    "write": frozenset({"ok", "denied", "error", "timeout"}),
    "edit": frozenset({"ok", "not_found", "old_string_not_found", "denied", "error", "timeout"}),
    "unknown": frozenset({"unrecognized"}),
}
FROZEN_TIMEOUTS_SECONDS: dict[str, Any] = {
    "bash": 300,
    "validate": 300,
    "read": 30,
    "write": 30,
    "edit": 30,
    "model_round": 600,
    "trusted_git_clone": 300,
    "trusted_git_checkout": 300,
    "trusted_patch_capture": 60,
    "official_evaluator_per_instance": 1800,
    "effective_model_visible_timeout_formula": ("min(operation_timeout, arm_deadline_monotonic-now_monotonic)"),
    "process_group_shutdown": "SIGTERM_then_one_second_grace_then_SIGKILL",
}
TIMEOUT_CENSORING = {
    "administrative_wall_limit_ns": ARM_WALL_NS,
    "observed_wall_formula": "min(actual_wall_ns,administrative_wall_limit_ns)",
    "wall_timeout_status": "timeout",
    "wall_timeout_is_retryable": False,
    "tool_or_model_round_timeout_is_retryable": False,
    "official_test_timeout_outcome": "unresolved_not_infrastructure",
    "official_patch_apply_failure_outcome": "unresolved_not_infrastructure",
    "phase_censoring_policy": ("primary_quality_remains_admissible_but_efficiency_promotion_fails_closed"),
}
ROUND_REQUIRED_FIELDS = (
    "round_index",
    "generation_backend",
    "drafter_requested",
    "drafter_selected",
    "drafter_ref",
    "timing_source",
    "effective_timeout_ns",
    "logical_prompt_tokens",
    "warm_offset_tokens",
    "physical_prefill_tokens",
    "completion_tokens",
    "prefill_ns",
    "decode_ns",
    "model_total_ns",
    "phase_censored",
    "deadline_hit",
)
TOOL_REQUIRED_FIELDS = (
    "sequence",
    "round_index",
    "tool_name",
    "allowed",
    "outcome",
    "duration_ns",
    "effective_timeout_ns",
    "exit_code_or_signal",
    "output_chars",
    "target_hmac_sha256",
)
ARM_REQUIRED_FIELDS = (
    "pair_index",
    "condition",
    "status",
    "termination_reason",
    "wall_elapsed_ns",
    "wall_observed_ns",
    "wall_limit_ns",
    "wall_censored",
    "watchdog_overrun_ns",
    "round_count",
    "tool_call_count",
    "output_tokens",
    "physical_prefill_tokens",
    "prefill_ns",
    "decode_tokens",
    "decode_ns",
    "phase_censored",
    "telemetry_complete",
)
FORBIDDEN_TELEMETRY_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "model_patch",
    "assistant_text",
    "tool_arguments",
    "tool_output",
    "official_result",
)
TELEMETRY_RECEIPT_BINDINGS = (
    "round_telemetry_sha256",
    "tool_telemetry_sha256",
    "pair_attempt_ledger_sha256",
    "docker_image_lock_sha256",
    "execution_chain_of_custody_sha256",
)
TELEMETRY_SEMANTICS = {
    "round_completion_tokens": ("physically_decoded_tokens_including_tokens_not_delivered_when_phase_censored"),
    "round_completion_tokens_source": "AgentRoundTrace.physical_decode_tokens",
    "arm_output_tokens": "tokens_delivered_to_agent_loop_and_charged_to_generation_budget",
    "arm_output_tokens_source": "sum_AgentRoundTrace.completion_tokens",
    "arm_decode_tokens": ("sum_round_physical_completion_tokens_used_for_decode_cost_guardrail"),
    "censored_output_decode_relation": ("output_tokens_le_decode_tokens_and_difference_only_when_phase_censored"),
    "tool_record_unit": ("exactly_one_record_per_admitted_tool_invocation_not_per_audit_event"),
    "pre_admission_budget_block": "no_tool_record_because_no_invocation_was_admitted",
    "tool_output_chars": ("visible_capped_result_chars_including_denial_unknown_and_error_messages"),
    "known_tool_adapter_precondition": (
        "AgentToolTrace.timeout_enforced_must_be_true_but_is_not_serialized_as_evidence"
    ),
    "blocked_command_adapter_names": "bash_and_validate_until_full_invocation_watchdog_attested",
    "nonterminal_visible_tool_outcomes": (
        "not_found_and_old_string_not_found_are_allowed_results_not_dispatch_failures"
    ),
    "tool_round_topology": "tool_round_index_nondecreasing",
    "terminal_tool_topology": "last_tool_in_final_round_with_no_later_round",
    "arm_wall_timeout_topology": "at_most_one_terminal_timeout_and_no_event_after_it",
    "round_timing_source": "runtime_raw_ns",
    "round_effective_timeout": "positive_ns_capped_at_frozen_600_second_model_round_timeout",
    "tool_target_identifier": ("hmac_sha256_pseudonymous_within_run_tag_not_authenticity_or_public_commitment"),
    "tool_target_hmac_source": ("HMAC_SHA256(private_per_run_key,AgentToolTrace.target_sha256)"),
    "tool_target_hmac_key": ("caller_bytes_shape_checked_only_supervisor_private_random_per_run_key_not_attested"),
    "tool_trace_adapter_authority": "shape_only_non_authoritative_non_evidence",
    "tool_target_cross_run_comparison": "forbidden_tags_intentionally_noncomparable_across_runs",
    "unknown_tool_name_representation": (
        "tool_name_unknown_outcome_unrecognized_allowed_false_requested_name_never_serialized"
    ),
}
EXECUTION_CHAIN_OF_CUSTODY = {
    "status": "required_not_attested",
    "trusted_observer": "supervisor_outside_model_visible_loop",
    "receipt_binding": "execution_chain_of_custody_sha256",
    "model_tree_binding": {
        "executor": "scripts.run_swebench_quality_generation.NativeMioArmExecutor",
        "caller_supplied_model_path_is_sufficient": False,
        "required_receipt_fields": [
            "executor_engine_object_identity",
            "resolved_loaded_target_root",
            "complete_loaded_model_tree_manifest_sha256_before",
            "engine_loaded_target_tree_manifest_sha256",
            "complete_loaded_model_tree_manifest_sha256_after",
            "frozen_target_model_identity",
        ],
        "required_equivalence": (
            "before_tree_equals_engine_loaded_target_tree_equals_after_tree_equals_frozen_target_tree"
        ),
    },
    "executed_code_binding": {
        "caller_supplied_repo_commit_is_sufficient": False,
        "required_receipt_fields": [
            "clean_repo_head_commit",
            "clean_repo_head_tree",
            "native_executor_source_sha256",
            "executed_mio_import_origins_sha256",
            "executed_runtime_import_origins_sha256",
            "sealed_dependency_lock_sha256",
        ],
        "required_equivalence": ("executed_mio_origins_resolve_to_sealed_repo_tree_and_runtime_origins_to_sealed_lock"),
    },
    "activation_rule": "both_bindings_verified_before_and_after_confirmatory_generation",
}
QUALITY_AND_PROMOTION_GATES = {
    "quality_improvement": ("paired_bootstrap_lower_95_gt_0_and_exact_one_sided_mcnemar_p_lt_0_05"),
    "practical_resolution_difference_minimum": 0.02,
    "full_500_pairs_required": True,
    "promotion": ("quality_improvement_and_resolution_difference_ge_0_02_and_all_efficiency_guardrails_pass"),
    "current_v2_decision_mode": ("shape_only_all_admissible_criteria_false_without_authority_receipt_binding"),
    "caller_counts_bootstrap_checksum_are_evidence": False,
}
PREREGISTRATION_STATUS = "draft_hard_blocked_no_confirmatory_activation"
STUDY_ID = "mio-qwen36-27b-quality-gate-swebench-verified-v2"
RESULT_STATUS = "no_generation_no_evaluation_no_quality_result"
INHERITANCE_POLICY = {
    "schema": "mio.swebench-verified-quality-preregistration.v1",
    "canonical_sha256": V1_PREREGISTRATION_CANONICAL_SHA256,
    "inheritance_rule": "all_v1_controls_remain_frozen_unless_explicitly_replaced_below",
    "dataset_model_schedule_and_quality_analysis_changed": False,
}
CONFIRMATORY_SCOPE = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "dataset_revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "full_snapshot_sha256": "52ccbc6ec0e03085f95191b261e0ed881cd6a0752a3c5247c1aba258ec2993da",
    "required_pairs": EXPECTED_PAIRS,
    "required_arms": EXPECTED_PAIRS * 2,
    "target_model_identity": ("local-sha256-v1:ba3975accc6b6398f47f82ff7640b39f5541abb49f1d3c6f34113aa7fb040c87"),
    "target_runtime": "target_ar",
    "draft_model": None,
    "generation_backend_metric": "baseline",
}
FROZEN_LIMITS = {
    "arm_wall_seconds": ARM_WALL_NS // 1_000_000_000,
    "max_output_tokens_per_round": 4_096,
    "max_output_tokens_per_arm": MAX_OUTPUT_TOKENS_PER_ARM,
    "max_agent_rounds": MAX_AGENT_ROUNDS,
    "max_tool_calls_per_arm": MAX_TOOL_CALLS_PER_ARM,
    "maximum_pair_attempts": MAX_PAIR_ATTEMPTS,
    "maximum_retries_after_initial_attempt": MAX_PAIR_ATTEMPTS - 1,
}
WHOLE_PAIR_RETRY_POLICY = {
    "unit": "both_generation_arms_for_one_instance",
    "authority": "trusted_supervisor_only",
    "classifier_is_authority": False,
    "automatic_retry": False,
    "sealed_incident_receipt_required": True,
    "sealed_pair_attempt_ledger_required": True,
    "infrastructure_reason_allowlist_is_exclusive": True,
    "same_within_pair_order_and_seed": True,
    "fresh_workspace_cache_and_conversation": True,
    "partial_success_is_never_promoted": True,
    "first_completed_attempt_is_final": True,
    "retry_after_completed_attempt": False,
    "open_attempt_requires_explicit_blinded_abort": True,
    "aborted_attempt_manifest_must_be_hash_bound": True,
    "retry_reason_enum": sorted(RETRYABLE_INFRASTRUCTURE_REASONS),
    "nonretryable_outcomes": sorted(NONRETRYABLE_OUTCOMES),
}
CONTENT_FREE_TELEMETRY_POLICY = {
    "clock": "monotonic_integer_nanoseconds",
    "tool_name_vocabulary": list(TOOL_NAMES),
    "tool_outcome_vocabulary": sorted(TOOL_OUTCOMES),
    "tool_outcomes_by_name": {name: sorted(outcomes) for name, outcomes in TOOL_OUTCOMES_BY_NAME.items()},
    "exit_status_semantics": {
        "file_and_unknown_tools": "null_only",
        "denied_preexecution_and_error_outcomes": "null_only",
        "command_ok_and_validate_no_work": "integer_zero",
        "command_nonzero": "nonzero_integer_or_signal",
        "command_timeout_or_output_limit": "null_integer_or_signal",
    },
    "round_required_fields": list(ROUND_REQUIRED_FIELDS),
    "tool_required_fields": list(TOOL_REQUIRED_FIELDS),
    "arm_required_fields": list(ARM_REQUIRED_FIELDS),
    "forbidden_fields": list(FORBIDDEN_TELEMETRY_FIELDS),
    "receipt_bindings": list(TELEMETRY_RECEIPT_BINDINGS),
    "field_semantics": TELEMETRY_SEMANTICS,
}
TERMINATION_TAXONOMY = {
    "completed": {"status": "completed", "wall_censored": False},
    "model_final_incomplete": {"status": "incomplete", "wall_censored": False},
    "quality_gate_incomplete": {"status": "incomplete", "wall_censored": False},
    "budget_exhaustion": {"status": "incomplete", "wall_censored": False},
    "tool_nonzero": {
        "status": "incomplete",
        "wall_censored": False,
        "requires": "terminal_tool_outcome_nonzero",
    },
    "protocol_violation": {"status": "incomplete", "wall_censored": False},
    "model_error": {"status": "model_error", "wall_censored": False},
    "model_round_timeout": {
        "status": "timeout",
        "wall_censored": False,
        "requires": "terminal_round_deadline_hit",
    },
    "tool_timeout": {
        "status": "timeout",
        "wall_censored": False,
        "requires": "terminal_tool_outcome_timeout",
    },
    "arm_wall_timeout": {"status": "timeout", "wall_censored": True},
}
EFFICIENCY_GUARDRAIL_POLICY = {
    "population": "all_500_complete_pairs",
    "cost_direction": "gate_on_divided_by_gate_off_lower_is_better",
    "wall_ratio_formula": "sum(observed_wall_on)/sum(observed_wall_off)",
    "prefill_cost_ratio_formula": (
        "(sum(prefill_ns_on)/sum(physical_prefill_tokens_on))/(sum(prefill_ns_off)/sum(physical_prefill_tokens_off))"
    ),
    "decode_cost_ratio_formula": (
        "(sum(decode_ns_on)/sum(decode_tokens_on))/(sum(decode_ns_off)/sum(decode_tokens_off))"
    ),
    "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
    "paired_bootstrap_seed": BOOTSTRAP_SEED,
    "paired_bootstrap_rng": "splitmix64_rejection_sampling_v1",
    "one_sided_upper_confidence": 0.95,
    "upper_percentile_zero_based_index": 9_499,
    "thresholds": {
        "wall_ratio_upper_95": WALL_RATIO_LIMIT,
        "prefill_cost_per_token_ratio_upper_95": PREFILL_COST_RATIO_LIMIT,
        "decode_cost_per_token_ratio_upper_95": DECODE_COST_RATIO_LIMIT,
    },
    "missing_or_censored_phase_telemetry": "fail_closed",
    "all_three_required": True,
    "aggregate_origin": "pure_offline_protocol_calculation_non_evidence",
    "aggregate_integrity": "canonical_sha256_checksum_not_authenticity_or_commitment",
}
DOCKER_IMAGE_LOCK_POLICY = {
    "status": "required_not_materialized",
    "manifest_sha256": None,
    "required_images": EXPECTED_PAIRS,
    "harness_arch": "x86_64",
    "oci_platform": "linux/amd64",
    "namespace": "swebench",
    "source_tag": "latest",
    "locked_local_alias_tag": "mio-swe-v2-locked",
    "required_record_identities": [
        "registry_index_digest",
        "linux_amd64_manifest_digest",
        "config_digest",
        "docker_image_id",
        "compressed_layer_digests",
        "rootfs_diff_ids",
    ],
    "offline_validator_result": "syntactic_non_evidence",
    "expected_instance_digests_required_for_binding": True,
    "official_harness_flags": {
        "cache_level": "instance",
        "clean": False,
        "force_rebuild": False,
        "instance_image_tag": "mio-swe-v2-locked",
    },
    "verification_points": [
        "before_gate_off_evaluation",
        "between_gate_off_and_gate_on_evaluation",
        "after_gate_on_evaluation",
    ],
    "pull_or_build_during_confirmatory_evaluation": False,
    "bind_manifest_sha256_into": [
        "preregistration",
        "run_id",
        "evaluation_seal",
        "evaluation_receipt",
    ],
}
CONFIRMATORY_ACTIVATION_POLICY = {
    "enabled": False,
    "fail_closed": True,
    "blockers": list(CONFIRMATORY_BLOCKERS),
}
EXPECTED_PREREGISTRATION_DOCUMENT = {
    "schema": SCHEMA,
    "status": PREREGISTRATION_STATUS,
    "study_id": STUDY_ID,
    "inherits": INHERITANCE_POLICY,
    "confirmatory_scope": CONFIRMATORY_SCOPE,
    "frozen_limits": FROZEN_LIMITS,
    "frozen_timeouts_seconds": FROZEN_TIMEOUTS_SECONDS,
    "timeout_censoring": TIMEOUT_CENSORING,
    "whole_pair_retry": WHOLE_PAIR_RETRY_POLICY,
    "content_free_telemetry": CONTENT_FREE_TELEMETRY_POLICY,
    "termination_taxonomy": TERMINATION_TAXONOMY,
    "execution_chain_of_custody": EXECUTION_CHAIN_OF_CUSTODY,
    "efficiency_guardrails": EFFICIENCY_GUARDRAIL_POLICY,
    "quality_and_promotion_gates": QUALITY_AND_PROMOTION_GATES,
    "docker_image_lock": DOCKER_IMAGE_LOCK_POLICY,
    "confirmatory_activation": CONFIRMATORY_ACTIVATION_POLICY,
    "result_status": RESULT_STATUS,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_IMAGE_RE = re.compile(r"^swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:latest$")
_LOCKED_IMAGE_RE = re.compile(r"^swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:mio-swe-v2-locked$")
_MASK_64 = (1 << 64) - 1

ProtocolError = v1.ProtocolError


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation used for every v2 digest."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ProtocolError(f"duplicate JSON key in v2 preregistration: {key}")
        document[key] = value
    return document


def _load_json_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot load canonical SWE-bench quality v2 JSON") from exc
    if not isinstance(document, dict):
        raise ProtocolError("SWE-bench quality v2 document must be a JSON object")
    return document


def preregistration_digest(path: Path = PREREGISTRATION_PATH) -> str:
    document = _load_json_document(path)
    if document.get("schema") != SCHEMA:
        raise ProtocolError("unexpected SWE-bench quality v2 preregistration schema")
    return sha256_bytes(canonical_json_bytes(document))


def verify_v1_inheritance(path: Path = V1_PREREGISTRATION_PATH) -> str:
    """Fail if the inherited v1 protocol differs from the frozen canonical digest."""

    observed = v1.preregistration_digest(path)
    if observed != V1_PREREGISTRATION_CANONICAL_SHA256:
        raise ProtocolError("SWE-bench quality v1 canonical digest differs from the v2 inheritance seal")
    return observed


def load_and_validate_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Validate every frozen v2 block, its full seal, and v1 inheritance."""

    verify_v1_inheritance()
    document = _load_json_document(path)
    observed_keys = set(document)
    expected_keys = set(EXPECTED_PREREGISTRATION_DOCUMENT)
    if observed_keys != expected_keys:
        raise ProtocolError(
            "v2 preregistration top-level fields differ: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    for block, expected in EXPECTED_PREREGISTRATION_DOCUMENT.items():
        if canonical_json_bytes(document[block]) != canonical_json_bytes(expected):
            raise ProtocolError(f"v2 frozen top-level block differs: {block}")
    observed_digest = sha256_bytes(canonical_json_bytes(document))
    if observed_digest != V2_PREREGISTRATION_CANONICAL_SHA256:
        raise ProtocolError("v2 full canonical preregistration digest differs from its frozen seal")
    return document


def require_confirmatory_ready() -> None:
    """Hard-block confirmatory generation/evaluation in this implementation."""

    load_and_validate_preregistration()
    raise ProtocolError("confirmatory SWE-bench quality v2 is hard-blocked: " + ", ".join(CONFIRMATORY_BLOCKERS))


def classify_pair_retry_request(
    *,
    next_attempt_index: int,
    reason_code: str,
) -> dict[str, Any]:
    """Classify syntax only; never authorize or admit a whole-pair retry.

    A trusted supervisor must separately seal both an incident receipt and the
    pair-attempt ledger. Those authority-bearing artifacts are deliberately not
    accepted by this pure helper, so ``admissible_for_retry`` is always false.
    """

    attempt = _integer(next_attempt_index, "next_attempt_index", minimum=1)
    if attempt >= MAX_PAIR_ATTEMPTS:
        raise ProtocolError("whole-pair retry exceeds the frozen three total attempts")
    if reason_code not in RETRYABLE_INFRASTRUCTURE_REASONS:
        if reason_code in NONRETRYABLE_OUTCOMES:
            raise ProtocolError("model, timeout, protocol, budget, patch, and evaluation outcomes are non-retryable")
        raise ProtocolError("whole-pair retry reason is outside the exclusive infrastructure allowlist")
    return {
        "next_attempt_index": attempt,
        "reason_code": reason_code,
        "rerun_unit": "whole_pair",
        "reason_allowlisted": True,
        "admissible_for_retry": False,
        "missing_authority_artifacts": [
            "sealed_supervisor_incident_receipt",
            "sealed_pair_attempt_ledger",
        ],
    }


def _require_exact_keys(raw: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    keys = set(raw)
    if keys != set(expected):
        raise ProtocolError(
            f"{label} fields differ: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{label} exceeds its frozen maximum")
    return value


@dataclass(frozen=True)
class RoundMetricRecord:
    """Content-free raw timing and backend provenance for one model round.

    ``completion_tokens`` counts physically decoded tokens, including a token
    computed before a censored deadline but never delivered to the agent loop.
    """

    round_index: int
    generation_backend: str
    drafter_requested: str
    drafter_selected: str
    drafter_ref: None
    timing_source: str
    effective_timeout_ns: int
    logical_prompt_tokens: int
    warm_offset_tokens: int
    physical_prefill_tokens: int
    completion_tokens: int
    prefill_ns: int
    decode_ns: int
    model_total_ns: int
    phase_censored: bool
    deadline_hit: bool

    def as_dict(self) -> dict[str, Any]:
        return {"schema": ROUND_METRIC_SCHEMA, **self.__dict__}


_ROUND_METRIC_KEYS = frozenset({"schema", *RoundMetricRecord.__dataclass_fields__})


def validate_round_metric_record(raw: Mapping[str, Any]) -> RoundMetricRecord:
    """Validate target-only provenance, token arithmetic, and raw phase time."""

    if not isinstance(raw, Mapping):
        raise ProtocolError("round metric record must be a mapping")
    _require_exact_keys(raw, _ROUND_METRIC_KEYS, "round metric record")
    if raw.get("schema") != ROUND_METRIC_SCHEMA:
        raise ProtocolError("unexpected round metric schema")
    if (
        raw.get("generation_backend") != "baseline"
        or raw.get("drafter_requested") != "target_ar"
        or raw.get("drafter_selected") != "baseline"
        or raw.get("drafter_ref") is not None
    ):
        raise ProtocolError("round metric is not the frozen target_ar baseline with no drafter ref")
    if raw.get("timing_source") != "runtime_raw_ns":
        raise ProtocolError("confirmatory round telemetry requires runtime_raw_ns timing")
    effective_timeout_ns = _integer(
        raw["effective_timeout_ns"],
        "round effective_timeout_ns",
        minimum=1,
        maximum=int(FROZEN_TIMEOUTS_SECONDS["model_round"] * 1_000_000_000),
    )
    round_index = _integer(
        raw["round_index"],
        "round_index",
        maximum=MAX_AGENT_ROUNDS - 1,
    )
    logical = _integer(
        raw["logical_prompt_tokens"],
        "logical_prompt_tokens",
        maximum=TARGET_CONTEXT_TOKENS,
    )
    warm = _integer(raw["warm_offset_tokens"], "warm_offset_tokens", maximum=logical)
    physical = _integer(raw["physical_prefill_tokens"], "physical_prefill_tokens")
    completion = _integer(
        raw["completion_tokens"],
        "completion_tokens",
        maximum=4_096,
    )
    if physical != logical - warm:
        raise ProtocolError("physical prefill tokens must equal logical prompt minus warm offset")
    if logical + completion > TARGET_CONTEXT_TOKENS:
        raise ProtocolError("round prompt plus completion exceeds the frozen context")
    prefill_ns = _integer(raw["prefill_ns"], "prefill_ns")
    decode_ns = _integer(raw["decode_ns"], "decode_ns")
    model_total_ns = _integer(raw["model_total_ns"], "model_total_ns")
    if model_total_ns != prefill_ns + decode_ns:
        raise ProtocolError("model_total_ns must equal prefill_ns plus decode_ns")
    if (physical == 0) != (prefill_ns == 0):
        raise ProtocolError("round prefill token/time zero values are inconsistent")
    if (completion == 0) != (decode_ns == 0):
        raise ProtocolError("round decode token/time zero values are inconsistent")
    if not isinstance(raw["phase_censored"], bool) or not isinstance(raw["deadline_hit"], bool):
        raise ProtocolError("round censoring fields must be boolean")
    if raw["deadline_hit"] and not raw["phase_censored"]:
        raise ProtocolError("a round deadline hit must mark its model phase censored")
    maximum_model_ns = effective_timeout_ns + (PROCESS_GROUP_KILL_GRACE_NS if raw["deadline_hit"] else 0)
    if model_total_ns > maximum_model_ns:
        raise ProtocolError("round raw model time exceeds its effective timeout envelope")
    return RoundMetricRecord(
        round_index=round_index,
        generation_backend="baseline",
        drafter_requested="target_ar",
        drafter_selected="baseline",
        drafter_ref=None,
        timing_source="runtime_raw_ns",
        effective_timeout_ns=effective_timeout_ns,
        logical_prompt_tokens=logical,
        warm_offset_tokens=warm,
        physical_prefill_tokens=physical,
        completion_tokens=completion,
        prefill_ns=prefill_ns,
        decode_ns=decode_ns,
        model_total_ns=model_total_ns,
        phase_censored=raw["phase_censored"],
        deadline_hit=raw["deadline_hit"],
    )


def round_metric_from_agent_trace(
    trace: Any,
    *,
    effective_timeout_ns: int,
) -> RoundMetricRecord:
    """Map physical AgentRoundTrace decode work, never delivered-token count."""

    delivered_tokens = _integer(
        getattr(trace, "completion_tokens", None),
        "AgentRoundTrace.completion_tokens",
        maximum=4_096,
    )
    physical_decode_tokens = _integer(
        getattr(trace, "physical_decode_tokens", None),
        "AgentRoundTrace.physical_decode_tokens",
        maximum=4_096,
    )
    phase_censored = getattr(trace, "phase_censored", None)
    if not isinstance(phase_censored, bool):
        raise ProtocolError("AgentRoundTrace.phase_censored must be boolean")
    if physical_decode_tokens < delivered_tokens:
        raise ProtocolError("physical decode tokens cannot be below delivered completion tokens")
    if not phase_censored and physical_decode_tokens != delivered_tokens:
        raise ProtocolError("physical/delivered decode divergence requires phase censoring")
    if getattr(trace, "fallback_ar", None) is not False:
        raise ProtocolError("target_ar trace cannot report speculative fallback")
    warm_offset = getattr(trace, "warm_offset", None)
    warm_offset_tokens = getattr(trace, "warm_offset_tokens", None)
    if warm_offset != warm_offset_tokens:
        raise ProtocolError("AgentRoundTrace warm-offset aliases disagree")
    return validate_round_metric_record(
        {
            "schema": ROUND_METRIC_SCHEMA,
            "round_index": getattr(trace, "round_index", None),
            "generation_backend": getattr(trace, "generation_backend", None),
            "drafter_requested": getattr(trace, "drafter_requested", None),
            "drafter_selected": getattr(trace, "drafter_selected", None),
            "drafter_ref": getattr(trace, "drafter_ref", None),
            "timing_source": getattr(trace, "timing_source", None),
            "effective_timeout_ns": effective_timeout_ns,
            "logical_prompt_tokens": getattr(trace, "logical_prompt_tokens", None),
            "warm_offset_tokens": warm_offset_tokens,
            "physical_prefill_tokens": getattr(trace, "physical_prefill_tokens", None),
            # Deliberately physical, despite the legacy v2 field name.
            "completion_tokens": physical_decode_tokens,
            "prefill_ns": getattr(trace, "prefill_ns", None),
            "decode_ns": getattr(trace, "decode_ns", None),
            "model_total_ns": getattr(trace, "model_total_ns", None),
            "phase_censored": phase_censored,
            "deadline_hit": getattr(trace, "deadline_hit", None),
        }
    )


def arm_output_tokens_from_agent_round_traces(traces: Sequence[Any]) -> int:
    """Sum only AgentRoundTrace tokens delivered to the loop/budget."""

    total = 0
    for index, trace in enumerate(traces):
        total += _integer(
            getattr(trace, "completion_tokens", None),
            f"AgentRoundTrace[{index}].completion_tokens",
            maximum=4_096,
        )
        if total > MAX_OUTPUT_TOKENS_PER_ARM:
            raise ProtocolError("AgentRoundTrace delivered tokens exceed the frozen arm budget")
    return total


@dataclass(frozen=True)
class ToolMetricRecord:
    """One bounded, argument-free record per model-visible tool invocation.

    Audit events around one invocation must be folded into this single record;
    they must never be emitted as extra ToolMetricRecord rows.
    """

    sequence: int
    round_index: int
    tool_name: str
    allowed: bool
    outcome: str
    duration_ns: int
    effective_timeout_ns: int
    exit_code_or_signal: int | str | None
    output_chars: int
    target_hmac_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"schema": TOOL_METRIC_SCHEMA, **self.__dict__}


_TOOL_METRIC_KEYS = frozenset({"schema", *ToolMetricRecord.__dataclass_fields__})


def validate_tool_metric_record(raw: Mapping[str, Any]) -> ToolMetricRecord:
    """Reject tool content and enforce the frozen effective timeout envelope."""

    if not isinstance(raw, Mapping):
        raise ProtocolError("tool metric record must be a mapping")
    _require_exact_keys(raw, _TOOL_METRIC_KEYS, "tool metric record")
    if raw.get("schema") != TOOL_METRIC_SCHEMA:
        raise ProtocolError("unexpected tool metric schema")
    sequence = _integer(
        raw["sequence"],
        "tool sequence",
        maximum=MAX_TOOL_CALLS_PER_ARM - 1,
    )
    round_index = _integer(
        raw["round_index"],
        "tool round_index",
        maximum=MAX_AGENT_ROUNDS - 1,
    )
    tool_name = raw["tool_name"]
    outcome = raw["outcome"]
    if tool_name not in TOOL_NAMES or outcome not in TOOL_OUTCOMES:
        raise ProtocolError("tool metric name or outcome is outside the frozen vocabulary")
    if outcome not in TOOL_OUTCOMES_BY_NAME[str(tool_name)]:
        raise ProtocolError("tool metric name/outcome combination is outside the frozen taxonomy")
    if not isinstance(raw["allowed"], bool):
        raise ProtocolError("tool metric allowed must be boolean")
    duration_ns = _integer(raw["duration_ns"], "tool duration_ns")
    effective_timeout_ns = _integer(raw["effective_timeout_ns"], "tool effective_timeout_ns")
    if tool_name == "unknown":
        if raw["allowed"] is not False or outcome != "unrecognized":
            raise ProtocolError("unknown tool names require the redacted unrecognized sentinel")
        if effective_timeout_ns != 0 or duration_ns > PROCESS_GROUP_KILL_GRACE_NS:
            raise ProtocolError("an unrecognized tool has no operation timeout and only bounded dispatch duration")
    else:
        if effective_timeout_ns < 1:
            raise ProtocolError("known tool effective timeout must be positive")
        maximum_timeout_ns = int(FROZEN_TIMEOUTS_SECONDS[str(tool_name)] * 1_000_000_000)
        if effective_timeout_ns > maximum_timeout_ns:
            raise ProtocolError("tool effective timeout exceeds its frozen operation timeout")
        if outcome == "timeout":
            if not (effective_timeout_ns <= duration_ns <= effective_timeout_ns + PROCESS_GROUP_KILL_GRACE_NS):
                raise ProtocolError("tool timeout duration is outside its frozen one-second kill grace")
        elif duration_ns > effective_timeout_ns:
            raise ProtocolError("non-timeout tool duration exceeds its effective timeout")
    denied_outcomes = {"denied", "unrecognized", "unscoped", "untrusted_executable"}
    if (raw["allowed"] and outcome in denied_outcomes) or (not raw["allowed"] and outcome not in denied_outcomes):
        raise ProtocolError("tool allowed flag and outcome are inconsistent")
    exit_value = raw["exit_code_or_signal"]
    valid_exit = (
        exit_value is None
        or (isinstance(exit_value, int) and not isinstance(exit_value, bool) and -64 <= exit_value <= 255)
        or (isinstance(exit_value, str) and re.fullmatch(r"signal:(?:[1-9]|[1-5][0-9]|6[0-4])", exit_value))
    )
    if not valid_exit:
        raise ProtocolError("tool exit_code_or_signal is malformed")
    file_tools = {"read", "write", "edit"}
    if tool_name in file_tools | {"unknown"} and exit_value is not None:
        raise ProtocolError("file and unknown tool telemetry cannot carry a process exit status")
    if outcome in denied_outcomes | {"error"} and exit_value is not None:
        raise ProtocolError("pre-execution denial and error outcomes cannot carry an exit status")
    if outcome in {"ok", "no_work"} and tool_name in {"bash", "validate"} and exit_value != 0:
        raise ProtocolError("successful command telemetry must record exit code zero")
    if outcome == "nonzero" and not (
        (isinstance(exit_value, int) and not isinstance(exit_value, bool) and exit_value != 0)
        or (isinstance(exit_value, str) and exit_value.startswith("signal:"))
    ):
        raise ProtocolError("nonzero command telemetry must record a nonzero exit code or signal")
    output_chars = _integer(
        raw["output_chars"],
        "tool output_chars",
        maximum=MAX_TOOL_OUTPUT_CHARS,
    )
    target_hmac_sha256 = raw["target_hmac_sha256"]
    if not isinstance(target_hmac_sha256, str) or not _SHA256_RE.fullmatch(target_hmac_sha256):
        raise ProtocolError("tool target_hmac_sha256 must be lowercase HMAC-SHA-256")
    return ToolMetricRecord(
        sequence=sequence,
        round_index=round_index,
        tool_name=str(tool_name),
        allowed=raw["allowed"],
        outcome=str(outcome),
        duration_ns=duration_ns,
        effective_timeout_ns=effective_timeout_ns,
        exit_code_or_signal=exit_value,
        output_chars=output_chars,
        target_hmac_sha256=target_hmac_sha256,
    )


def tool_metric_shape_from_agent_trace(
    trace: Any,
    *,
    target_hmac_key: bytes,
) -> ToolMetricRecord:
    """Shape-map one AgentToolTrace without granting authority or evidence.

    The byte string is only length-checked here. This helper cannot attest that
    it is random, private, unique per run, or supervisor-held. The key remains
    absent from the returned record, and the corresponding supervisor blocker
    remains active. Incomplete traces are rejected rather than upgraded.
    """

    if not isinstance(target_hmac_key, bytes) or len(target_hmac_key) < 32:
        raise ProtocolError("target HMAC key shape must contain at least 32 bytes")
    if getattr(trace, "telemetry_complete", None) is not True:
        raise ProtocolError("incomplete AgentToolTrace cannot map to frozen v2 telemetry")
    tool_name = getattr(trace, "tool_name", None)
    if tool_name in {"bash", "validate"}:
        raise ProtocolError("bash/validate adapter is blocked until the full-invocation watchdog is attested")
    if tool_name != "unknown" and getattr(trace, "timeout_enforced", None) is not True:
        raise ProtocolError("known AgentToolTrace timeout_enforced must be true before frozen v2 mapping")
    target_sha256 = getattr(trace, "target_sha256", None)
    if not isinstance(target_sha256, str) or not _SHA256_RE.fullmatch(target_sha256):
        raise ProtocolError("AgentToolTrace target_sha256 is malformed")
    effective_timeout = getattr(trace, "effective_timeout_ns", None)
    raw = {
        "schema": TOOL_METRIC_SCHEMA,
        "sequence": getattr(trace, "sequence", None),
        "round_index": getattr(trace, "round_index", None),
        "tool_name": tool_name,
        "allowed": getattr(trace, "allowed", None),
        "outcome": getattr(trace, "outcome", None),
        "duration_ns": getattr(trace, "duration_ns", None),
        "effective_timeout_ns": 0 if effective_timeout is None else effective_timeout,
        "exit_code_or_signal": getattr(trace, "exit_code_or_signal", None),
        "output_chars": getattr(trace, "output_chars", None),
        "target_hmac_sha256": hmac.new(
            target_hmac_key,
            target_sha256.encode("ascii"),
            hashlib.sha256,
        ).hexdigest(),
    }
    return validate_tool_metric_record(raw)


@dataclass(frozen=True)
class ArmMetricRecord:
    """The exact source-free per-arm telemetry admitted to guardrail analysis."""

    pair_index: int
    condition: str
    status: str
    termination_reason: str
    wall_elapsed_ns: int
    wall_observed_ns: int
    wall_limit_ns: int
    wall_censored: bool
    watchdog_overrun_ns: int
    round_count: int
    tool_call_count: int
    output_tokens: int
    physical_prefill_tokens: int
    prefill_ns: int
    decode_tokens: int
    decode_ns: int
    phase_censored: bool
    telemetry_complete: bool

    def as_dict(self) -> dict[str, Any]:
        return {"schema": ARM_METRIC_SCHEMA, **self.__dict__}


_ARM_METRIC_KEYS = frozenset({"schema", *ArmMetricRecord.__dataclass_fields__})


def validate_arm_metric_record(raw: Mapping[str, Any]) -> ArmMetricRecord:
    """Validate one content-free arm record and its censoring invariants."""

    if not isinstance(raw, Mapping):
        raise ProtocolError("arm metric record must be a mapping")
    _require_exact_keys(raw, _ARM_METRIC_KEYS, "arm metric record")
    if raw.get("schema") != ARM_METRIC_SCHEMA:
        raise ProtocolError("unexpected arm metric schema")
    pair_index = _integer(raw["pair_index"], "pair_index", maximum=EXPECTED_PAIRS - 1)
    condition = raw["condition"]
    status = raw["status"]
    termination_reason = raw["termination_reason"]
    if condition not in CONDITIONS or status not in TERMINAL_STATUSES or termination_reason not in TERMINATION_STATUS:
        raise ProtocolError("arm metric condition or terminal status is invalid")
    if TERMINATION_STATUS[str(termination_reason)] != status:
        raise ProtocolError("arm status and termination_reason taxonomy are inconsistent")
    for name in ("wall_censored", "phase_censored", "telemetry_complete"):
        if not isinstance(raw[name], bool):
            raise ProtocolError(f"{name} must be boolean")

    wall_elapsed_ns = _integer(raw["wall_elapsed_ns"], "wall_elapsed_ns")
    wall_observed_ns = _integer(raw["wall_observed_ns"], "wall_observed_ns")
    wall_limit_ns = _integer(raw["wall_limit_ns"], "wall_limit_ns")
    watchdog_overrun_ns = _integer(raw["watchdog_overrun_ns"], "watchdog_overrun_ns")
    if wall_limit_ns != ARM_WALL_NS:
        raise ProtocolError("arm metric wall limit differs from the frozen 1800 seconds")
    if raw["wall_censored"]:
        if termination_reason != "arm_wall_timeout" or wall_observed_ns != ARM_WALL_NS or wall_elapsed_ns < ARM_WALL_NS:
            raise ProtocolError("censored wall metric must be a capped terminal timeout")
        if watchdog_overrun_ns != wall_elapsed_ns - ARM_WALL_NS:
            raise ProtocolError("censored wall metric watchdog overrun is inconsistent")
        if watchdog_overrun_ns > PROCESS_GROUP_KILL_GRACE_NS:
            raise ProtocolError("wall watchdog overrun exceeds the frozen one-second kill grace")
    elif (
        termination_reason == "arm_wall_timeout"
        or wall_elapsed_ns != wall_observed_ns
        or wall_observed_ns > ARM_WALL_NS
        or watchdog_overrun_ns != 0
    ):
        raise ProtocolError("uncensored wall metric is inconsistent")

    round_count = _integer(raw["round_count"], "round_count", maximum=MAX_AGENT_ROUNDS)
    tool_call_count = _integer(
        raw["tool_call_count"],
        "tool_call_count",
        maximum=MAX_TOOL_CALLS_PER_ARM,
    )
    output_tokens = _integer(
        raw["output_tokens"],
        "output_tokens",
        maximum=MAX_OUTPUT_TOKENS_PER_ARM,
    )
    physical_prefill_tokens = _integer(
        raw["physical_prefill_tokens"],
        "physical_prefill_tokens",
        maximum=MAX_AGENT_ROUNDS * TARGET_CONTEXT_TOKENS,
    )
    prefill_ns = _integer(raw["prefill_ns"], "prefill_ns")
    decode_tokens = _integer(
        raw["decode_tokens"],
        "decode_tokens",
        maximum=MAX_OUTPUT_TOKENS_PER_ARM + 1,
    )
    decode_ns = _integer(raw["decode_ns"], "decode_ns")
    if output_tokens > decode_tokens:
        raise ProtocolError("budget output_tokens cannot exceed physically decoded tokens")
    if not raw["phase_censored"] and output_tokens != decode_tokens:
        raise ProtocolError("output/decode token divergence requires phase-censored telemetry")
    if (physical_prefill_tokens == 0) != (prefill_ns == 0):
        raise ProtocolError("prefill token/time zero values are inconsistent")
    if (decode_tokens == 0) != (decode_ns == 0):
        raise ProtocolError("decode token/time zero values are inconsistent")
    if prefill_ns + decode_ns > wall_elapsed_ns:
        raise ProtocolError("model phase time exceeds observed agent wall time")
    if raw["phase_censored"] and raw["telemetry_complete"]:
        raise ProtocolError("phase-censored telemetry cannot be declared complete")
    if termination_reason == "model_round_timeout" and not raw["phase_censored"]:
        raise ProtocolError("model_round_timeout requires phase-censored telemetry")
    return ArmMetricRecord(
        pair_index=pair_index,
        condition=str(condition),
        status=str(status),
        termination_reason=str(termination_reason),
        wall_elapsed_ns=wall_elapsed_ns,
        wall_observed_ns=wall_observed_ns,
        wall_limit_ns=wall_limit_ns,
        wall_censored=raw["wall_censored"],
        watchdog_overrun_ns=watchdog_overrun_ns,
        round_count=round_count,
        tool_call_count=tool_call_count,
        output_tokens=output_tokens,
        physical_prefill_tokens=physical_prefill_tokens,
        prefill_ns=prefill_ns,
        decode_tokens=decode_tokens,
        decode_ns=decode_ns,
        phase_censored=raw["phase_censored"],
        telemetry_complete=raw["telemetry_complete"],
    )


def validate_arm_telemetry_bundle(
    arm: Mapping[str, Any] | ArmMetricRecord,
    rounds: Sequence[Mapping[str, Any] | RoundMetricRecord],
    tools: Sequence[Mapping[str, Any] | ToolMetricRecord],
) -> dict[str, Any]:
    """Recompute one arm summary and bind its exact content-free event streams."""

    arm_record = validate_arm_metric_record(arm.as_dict() if isinstance(arm, ArmMetricRecord) else arm)
    round_records = tuple(
        validate_round_metric_record(item.as_dict())
        if isinstance(item, RoundMetricRecord)
        else validate_round_metric_record(item)
        for item in rounds
    )
    tool_records = tuple(
        validate_tool_metric_record(item.as_dict())
        if isinstance(item, ToolMetricRecord)
        else validate_tool_metric_record(item)
        for item in tools
    )
    if tuple(item.round_index for item in round_records) != tuple(range(len(round_records))):
        raise ProtocolError("round telemetry indices must be contiguous and ordered")
    if tuple(item.sequence for item in tool_records) != tuple(range(len(tool_records))):
        raise ProtocolError("tool telemetry sequence must be contiguous and ordered")
    if any(item.round_index >= len(round_records) for item in tool_records):
        raise ProtocolError("tool telemetry refers to a missing model round")
    tool_round_indices = tuple(item.round_index for item in tool_records)
    if tool_round_indices != tuple(sorted(tool_round_indices)):
        raise ProtocolError("tool telemetry round_index values must be nondecreasing")
    model_duration_ns = sum(item.model_total_ns for item in round_records)
    tool_duration_ns = sum(item.duration_ns for item in tool_records)
    if any(item.duration_ns > arm_record.wall_elapsed_ns for item in tool_records):
        raise ProtocolError("a tool invocation duration exceeds the complete arm wall duration")
    if model_duration_ns + tool_duration_ns > arm_record.wall_elapsed_ns:
        raise ProtocolError("sequential model and tool durations exceed the complete arm wall duration")

    deadline_rounds = [item.round_index for item in round_records if item.deadline_hit]
    timeout_tools = [item.sequence for item in tool_records if item.outcome == "timeout"]

    def require_terminal_tool(sequence: int, label: str) -> None:
        if (
            sequence != len(tool_records) - 1
            or not tool_records
            or tool_records[sequence].round_index != len(round_records) - 1
        ):
            raise ProtocolError(f"{label} must be the last tool in the final model round")

    if arm_record.termination_reason == "model_round_timeout":
        if deadline_rounds != [len(round_records) - 1] or timeout_tools:
            raise ProtocolError("model_round_timeout requires exactly the terminal round deadline")
        if any(item.round_index >= deadline_rounds[0] for item in tool_records):
            raise ProtocolError("tool telemetry cannot follow a terminal model-round timeout")
    elif arm_record.termination_reason == "tool_timeout":
        if deadline_rounds or timeout_tools != [len(tool_records) - 1]:
            raise ProtocolError("tool_timeout requires exactly the terminal tool invocation timeout")
        require_terminal_tool(timeout_tools[0], "tool_timeout")
    elif arm_record.termination_reason == "arm_wall_timeout":
        if len(deadline_rounds) + len(timeout_tools) > 1:
            raise ProtocolError("arm_wall_timeout cannot contain multiple terminal timeout events")
        if deadline_rounds:
            deadline_round = deadline_rounds[0]
            if deadline_round != len(round_records) - 1 or any(
                item.round_index >= deadline_round for item in tool_records
            ):
                raise ProtocolError("no telemetry event may follow the wall-terminal round deadline")
        if timeout_tools:
            require_terminal_tool(timeout_tools[0], "wall-terminal tool timeout")
    elif deadline_rounds or timeout_tools:
        raise ProtocolError("round/tool timeout telemetry disagrees with arm termination")
    if arm_record.termination_reason == "tool_nonzero":
        if not tool_records or tool_records[-1].outcome != "nonzero":
            raise ProtocolError("tool_nonzero requires a terminal nonzero tool invocation")
        require_terminal_tool(tool_records[-1].sequence, "tool_nonzero")
    expected = {
        "round_count": len(round_records),
        "tool_call_count": len(tool_records),
        "physical_prefill_tokens": sum(item.physical_prefill_tokens for item in round_records),
        "prefill_ns": sum(item.prefill_ns for item in round_records),
        "decode_tokens": sum(item.completion_tokens for item in round_records),
        "decode_ns": sum(item.decode_ns for item in round_records),
        "phase_censored": any(item.phase_censored for item in round_records),
        "telemetry_complete": not any(item.phase_censored for item in round_records),
    }
    differences = {
        name: {"observed": getattr(arm_record, name), "expected": value}
        for name, value in expected.items()
        if getattr(arm_record, name) != value
    }
    if differences:
        raise ProtocolError(f"arm summary differs from its raw telemetry: {differences}")
    round_payload = [item.as_dict() for item in round_records]
    tool_payload = [item.as_dict() for item in tool_records]
    return {
        "arm": arm_record.as_dict(),
        "round_telemetry_sha256": sha256_bytes(canonical_json_bytes(round_payload)),
        "tool_telemetry_sha256": sha256_bytes(canonical_json_bytes(tool_payload)),
        "round_records": len(round_records),
        "tool_records": len(tool_records),
        "content_policy": "source_free_no_model_or_tool_content",
        "timeline_validation": "partial_sums_only_supervisor_timeline_not_attested",
    }


def _validated_pairs(
    records: Sequence[Mapping[str, Any] | ArmMetricRecord],
    *,
    expected_pairs: int,
) -> tuple[tuple[ArmMetricRecord, ArmMetricRecord], ...]:
    if (
        isinstance(expected_pairs, bool)
        or not isinstance(expected_pairs, int)
        or not 1 <= expected_pairs <= EXPECTED_PAIRS
    ):
        raise ProtocolError(f"expected_pairs must be an integer in [1, {EXPECTED_PAIRS}]")
    validated = tuple(
        validate_arm_metric_record(record.as_dict())
        if isinstance(record, ArmMetricRecord)
        else validate_arm_metric_record(record)
        for record in records
    )
    if len(validated) != expected_pairs * 2:
        raise ProtocolError("guardrail analysis requires exactly two arms per expected pair")
    by_pair: dict[int, dict[str, ArmMetricRecord]] = {}
    for record in validated:
        arms = by_pair.setdefault(record.pair_index, {})
        if record.condition in arms:
            raise ProtocolError("guardrail telemetry contains a duplicate pair condition")
        arms[record.condition] = record
    if set(by_pair) != set(range(expected_pairs)):
        raise ProtocolError("guardrail pair indices must be contiguous and complete")
    pairs = []
    for pair_index in range(expected_pairs):
        arms = by_pair[pair_index]
        if set(arms) != set(CONDITIONS):
            raise ProtocolError("guardrail telemetry contains an incomplete pair")
        pairs.append((arms["gate_off"], arms["gate_on"]))
    return tuple(pairs)


class _SplitMix64:
    """Protocol-local deterministic generator with unbiased bounded sampling."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK_64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK_64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
        return (value ^ (value >> 31)) & _MASK_64

    def randbelow(self, upper: int) -> int:
        if upper < 1:
            raise ValueError("upper must be positive")
        domain = 1 << 64
        limit = domain - (domain % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def _cost_ratio(
    on_time: int,
    on_units: int,
    off_time: int,
    off_units: int,
) -> float | None:
    if min(on_time, on_units, off_time, off_units) <= 0:
        return None
    return (on_time * off_units) / (on_units * off_time)


def _ratios_for_indices(
    pairs: Sequence[tuple[ArmMetricRecord, ArmMetricRecord]],
    indices: Sequence[int],
    *,
    phases_available: bool,
) -> tuple[float | None, float | None, float | None]:
    selected = [pairs[index] for index in indices]
    off_wall = sum(off.wall_observed_ns for off, _on in selected)
    on_wall = sum(on.wall_observed_ns for _off, on in selected)
    wall = on_wall / off_wall if off_wall > 0 else None
    if not phases_available:
        return wall, None, None
    off_prefill_ns = sum(off.prefill_ns for off, _on in selected)
    on_prefill_ns = sum(on.prefill_ns for _off, on in selected)
    off_prefill_tokens = sum(off.physical_prefill_tokens for off, _on in selected)
    on_prefill_tokens = sum(on.physical_prefill_tokens for _off, on in selected)
    off_decode_ns = sum(off.decode_ns for off, _on in selected)
    on_decode_ns = sum(on.decode_ns for _off, on in selected)
    off_decode_tokens = sum(off.decode_tokens for off, _on in selected)
    on_decode_tokens = sum(on.decode_tokens for _off, on in selected)
    return (
        wall,
        _cost_ratio(on_prefill_ns, on_prefill_tokens, off_prefill_ns, off_prefill_tokens),
        _cost_ratio(on_decode_ns, on_decode_tokens, off_decode_ns, off_decode_tokens),
    )


def _upper_95(values: Sequence[float | None], samples: int) -> float | None:
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    ordered = sorted(value for value in values if value is not None)
    index = math.ceil(0.95 * samples) - 1
    return ordered[index]


def aggregate_efficiency_guardrails(
    records: Sequence[Mapping[str, Any] | ArmMetricRecord],
    *,
    expected_pairs: int = EXPECTED_PAIRS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute source-free paired non-inferiority guardrails.

    Resampling operates on whole instance pairs. Missing/censored phase
    telemetry never gets dropped; it fails prefill/decode promotion closed.
    """

    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or not 1 <= bootstrap_samples <= BOOTSTRAP_SAMPLES
    ):
        raise ProtocolError(f"bootstrap_samples must be an integer in [1, {BOOTSTRAP_SAMPLES}]")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or not 0 <= bootstrap_seed <= _MASK_64:
        raise ProtocolError("bootstrap_seed must be an unsigned 64-bit integer")
    pairs = _validated_pairs(records, expected_pairs=expected_pairs)
    phase_ready = all(arm.telemetry_complete and not arm.phase_censored for pair in pairs for arm in pair)
    full_indices = tuple(range(expected_pairs))
    estimates = _ratios_for_indices(pairs, full_indices, phases_available=phase_ready)
    samples: tuple[list[float | None], list[float | None], list[float | None]] = (
        [],
        [],
        [],
    )
    rng = _SplitMix64(bootstrap_seed)
    for _sample_index in range(bootstrap_samples):
        indices = tuple(rng.randbelow(expected_pairs) for _ in range(expected_pairs))
        ratios = _ratios_for_indices(pairs, indices, phases_available=phase_ready)
        for bucket, value in zip(samples, ratios, strict=True):
            bucket.append(value)
    uppers = tuple(_upper_95(values, bootstrap_samples) for values in samples)

    def row(name: str, estimate: float | None, upper: float | None, limit: float) -> dict[str, Any]:
        return {
            "name": name,
            "estimate": estimate,
            "upper_95": upper,
            "limit": limit,
            "passed": upper is not None and upper <= limit,
        }

    guardrails = {
        "wall_ratio": row("wall_ratio", estimates[0], uppers[0], WALL_RATIO_LIMIT),
        "prefill_cost_per_token_ratio": row(
            "prefill_cost_per_token_ratio",
            estimates[1],
            uppers[1],
            PREFILL_COST_RATIO_LIMIT,
        ),
        "decode_cost_per_token_ratio": row(
            "decode_cost_per_token_ratio",
            estimates[2],
            uppers[2],
            DECODE_COST_RATIO_LIMIT,
        ),
    }
    normalized_records = [record.as_dict() for pair in pairs for record in pair]
    frozen_shape = (
        expected_pairs == EXPECTED_PAIRS and bootstrap_samples == BOOTSTRAP_SAMPLES and bootstrap_seed == BOOTSTRAP_SEED
    )
    result = {
        "schema": GUARDRAIL_SCHEMA,
        "status": "protocol_calculation_only_hard_blocked",
        "confirmatory_evidence_admissible": False,
        "aggregate_origin": "pure_offline_protocol_calculation_non_evidence",
        "v2_preregistration_sha256": V2_PREREGISTRATION_CANONICAL_SHA256,
        "input_records_sha256": sha256_bytes(canonical_json_bytes(normalized_records)),
        "frozen_confirmatory_shape": frozen_shape,
        "pairs": expected_pairs,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_rng": "splitmix64_rejection_sampling_v1",
        "upper_confidence": 0.95,
        "phase_telemetry_complete": phase_ready,
        "guardrails": guardrails,
        "all_passed": phase_ready and all(row["passed"] for row in guardrails.values()),
        "content_policy": "source_free_no_instance_rows",
    }
    return {
        **result,
        "calculation_integrity_sha256": sha256_bytes(canonical_json_bytes(result)),
    }


def promotion_decision(
    quality: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate caller-supplied shape without producing promotion evidence.

    Counts, bootstrap metadata, and checksums remain caller-recalculable. No
    authority/receipt binding is implemented, so every admissible promotion
    criterion and ``promote`` remain false regardless of formula outputs.
    """

    expected = frozenset(
        {
            "resolution_difference",
            "paired_bootstrap_lower_95",
            "exact_one_sided_mcnemar_p",
            "full_500_pairs",
        }
    )
    _require_exact_keys(quality, expected, "quality summary")
    numeric = (
        quality["resolution_difference"],
        quality["paired_bootstrap_lower_95"],
        quality["exact_one_sided_mcnemar_p"],
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric):
        raise ProtocolError("quality summary metrics must be numeric")
    delta, lower, p_value = (float(value) for value in numeric)
    if (
        not all(math.isfinite(value) for value in (delta, lower, p_value))
        or not -1 <= delta <= 1
        or not -1 <= lower <= 1
        or not 0 <= p_value <= 1
    ):
        raise ProtocolError("quality summary metrics are outside their valid domains")
    if not isinstance(quality["full_500_pairs"], bool):
        raise ProtocolError("quality full_500_pairs must be boolean")
    efficiency_keys = frozenset(
        {
            "schema",
            "status",
            "confirmatory_evidence_admissible",
            "aggregate_origin",
            "v2_preregistration_sha256",
            "input_records_sha256",
            "frozen_confirmatory_shape",
            "pairs",
            "bootstrap_samples",
            "bootstrap_seed",
            "bootstrap_rng",
            "upper_confidence",
            "phase_telemetry_complete",
            "guardrails",
            "all_passed",
            "content_policy",
            "calculation_integrity_sha256",
        }
    )
    _require_exact_keys(efficiency, efficiency_keys, "efficiency aggregate")
    integrity = efficiency.get("calculation_integrity_sha256")
    unsigned_efficiency = {key: value for key, value in efficiency.items() if key != "calculation_integrity_sha256"}
    if (
        not isinstance(integrity, str)
        or not _SHA256_RE.fullmatch(integrity)
        or integrity != sha256_bytes(canonical_json_bytes(unsigned_efficiency))
    ):
        raise ProtocolError("efficiency aggregate calculation-integrity checksum differs")
    if (
        efficiency.get("schema") != GUARDRAIL_SCHEMA
        or efficiency.get("status") != "protocol_calculation_only_hard_blocked"
        or efficiency.get("confirmatory_evidence_admissible") is not False
        or efficiency.get("aggregate_origin") != "pure_offline_protocol_calculation_non_evidence"
        or efficiency.get("v2_preregistration_sha256") != V2_PREREGISTRATION_CANONICAL_SHA256
        or not isinstance(efficiency.get("input_records_sha256"), str)
        or not _SHA256_RE.fullmatch(str(efficiency.get("input_records_sha256")))
        or efficiency.get("frozen_confirmatory_shape") is not True
        or efficiency.get("pairs") != EXPECTED_PAIRS
        or efficiency.get("bootstrap_samples") != BOOTSTRAP_SAMPLES
        or efficiency.get("bootstrap_seed") != BOOTSTRAP_SEED
        or efficiency.get("bootstrap_rng") != "splitmix64_rejection_sampling_v1"
        or efficiency.get("upper_confidence") != 0.95
        or efficiency.get("phase_telemetry_complete") is not True
        or efficiency.get("content_policy") != "source_free_no_instance_rows"
        or not isinstance(efficiency.get("all_passed"), bool)
    ):
        raise ProtocolError("efficiency aggregate is not a v2 guardrail result")
    guardrails = efficiency.get("guardrails")
    limits = {
        "wall_ratio": WALL_RATIO_LIMIT,
        "prefill_cost_per_token_ratio": PREFILL_COST_RATIO_LIMIT,
        "decode_cost_per_token_ratio": DECODE_COST_RATIO_LIMIT,
    }
    if not isinstance(guardrails, dict) or set(guardrails) != set(limits):
        raise ProtocolError("efficiency aggregate guardrail set differs from v2")
    observed_passes = []
    for name, limit in limits.items():
        row = guardrails[name]
        if not isinstance(row, Mapping):
            raise ProtocolError("efficiency aggregate guardrail row is malformed")
        _require_exact_keys(
            row,
            frozenset({"name", "estimate", "upper_95", "limit", "passed"}),
            "efficiency guardrail row",
        )
        if row.get("name") != name or row.get("limit") != limit or not isinstance(row.get("passed"), bool):
            raise ProtocolError("efficiency aggregate guardrail identity differs from v2")
        estimate = row.get("estimate")
        upper = row.get("upper_95")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in (estimate, upper)
            )
            or float(estimate) <= 0
            or float(upper) <= 0
        ):
            raise ProtocolError("passing efficiency guardrail values must be finite numbers")
        expected_pass = float(upper) <= limit
        if row["passed"] is not expected_pass:
            raise ProtocolError("efficiency aggregate guardrail verdict is inconsistent")
        observed_passes.append(expected_pass)
    if efficiency["all_passed"] is not all(observed_passes):
        raise ProtocolError("efficiency aggregate combined verdict is inconsistent")
    quality_passed = bool(quality["full_500_pairs"] and lower > 0 and p_value < 0.05)
    practical_delta_passed = delta >= 0.02
    efficiency_passed = bool(efficiency["all_passed"])
    criteria_met = quality_passed and practical_delta_passed and efficiency_passed
    return {
        "schema": PROMOTION_SCHEMA,
        "status": "hard_blocked",
        "confirmatory_enabled": False,
        "inputs_receipt_bound": False,
        "input_authority_binding_implemented": False,
        "caller_counts_or_checksum_are_evidence": False,
        "confirmatory_blockers": list(CONFIRMATORY_BLOCKERS),
        "shape_only_formula_outputs": {
            "quality_formula": quality_passed,
            "practical_delta_formula": practical_delta_passed,
            "efficiency_formula": efficiency_passed,
            "combined_formula": criteria_met,
        },
        "quality_improvement_passed": False,
        "practical_resolution_difference_passed": False,
        "efficiency_guardrails_passed": False,
        "mathematical_criteria_met_unverified": False,
        "promote": False,
    }


_DOCKER_RECORD_KEYS = frozenset(
    {
        "schema",
        "instance_digest",
        "expected_local_alias",
        "source_tag",
        "registry_index_digest",
        "linux_amd64_manifest_digest",
        "config_digest",
        "docker_image_id",
        "compressed_layer_digests",
        "rootfs_diff_ids",
        "os",
        "architecture",
        "harness_arch",
    }
)


def _digest_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > 512
        or any(not isinstance(item, str) or not _OCI_DIGEST_RE.fullmatch(item) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ProtocolError(f"{label} must be a non-empty unique OCI SHA-256 list")
    return list(value)


def validate_docker_lock_record_syntax(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate record syntax only; this is not daemon/materialization evidence."""

    if not isinstance(raw, Mapping):
        raise ProtocolError("Docker lock record must be a mapping")
    _require_exact_keys(raw, _DOCKER_RECORD_KEYS, "Docker lock record")
    if raw.get("schema") != DOCKER_RECORD_SCHEMA:
        raise ProtocolError("unexpected Docker lock record schema")
    instance_digest = raw["instance_digest"]
    source_tag = raw["source_tag"]
    alias = raw["expected_local_alias"]
    if not isinstance(instance_digest, str) or not _SHA256_RE.fullmatch(instance_digest):
        raise ProtocolError("Docker lock instance digest must be lowercase SHA-256")
    if not isinstance(source_tag, str) or not _SOURCE_IMAGE_RE.fullmatch(source_tag):
        raise ProtocolError("Docker lock source must be an official x86_64 latest tag")
    if not isinstance(alias, str) or not _LOCKED_IMAGE_RE.fullmatch(alias):
        raise ProtocolError("Docker lock alias must use the frozen local v2 tag")
    if source_tag.rsplit(":", 1)[0] != alias.rsplit(":", 1)[0]:
        raise ProtocolError("Docker lock source and local alias name differ")
    for name in (
        "registry_index_digest",
        "linux_amd64_manifest_digest",
        "config_digest",
        "docker_image_id",
    ):
        if not isinstance(raw[name], str) or not _OCI_DIGEST_RE.fullmatch(raw[name]):
            raise ProtocolError(f"Docker lock {name} must be an OCI SHA-256")
    if raw["config_digest"] != raw["docker_image_id"]:
        raise ProtocolError("Docker image ID must equal the locked OCI config digest")
    compressed_layers = _digest_list(raw["compressed_layer_digests"], "compressed layers")
    rootfs = _digest_list(raw["rootfs_diff_ids"], "RootFS diff IDs")
    if len(compressed_layers) != len(rootfs):
        raise ProtocolError("compressed layer and RootFS identity counts differ")
    if raw["os"] != "linux" or raw["architecture"] != "amd64" or raw["harness_arch"] != "x86_64":
        raise ProtocolError("Docker lock record is not the frozen linux/amd64 x86_64 image")
    return {
        key: (list(raw[key]) if key in {"compressed_layer_digests", "rootfs_diff_ids"} else raw[key])
        for key in sorted(_DOCKER_RECORD_KEYS)
    }


_DOCKER_LOCK_KEYS = frozenset(
    {
        "schema",
        "v1_preregistration_sha256",
        "dataset_full_snapshot_sha256",
        "harness_commit",
        "platform",
        "harness_arch",
        "namespace",
        "locked_alias_tag",
        "images",
    }
)


def validate_docker_lock_manifest_syntax(
    raw: Mapping[str, Any],
    *,
    expected_images: int = EXPECTED_PAIRS,
    expected_instance_digests: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return an explicitly non-evidentiary offline syntax check.

    When supplied, ``expected_instance_digests`` binds image records to the
    exact frozen 500-instance population. No result from this function attests
    Docker daemon state, local materialization, or evaluation-time reuse.
    """

    if not isinstance(raw, Mapping):
        raise ProtocolError("Docker lock manifest must be a mapping")
    _require_exact_keys(raw, _DOCKER_LOCK_KEYS, "Docker lock manifest")
    if raw.get("schema") != DOCKER_LOCK_SCHEMA:
        raise ProtocolError("unexpected Docker lock manifest schema")
    if raw.get("v1_preregistration_sha256") != V1_PREREGISTRATION_CANONICAL_SHA256:
        raise ProtocolError("Docker lock manifest v1 inheritance mismatch")
    if raw.get("dataset_full_snapshot_sha256") != v1.FULL_SNAPSHOT_SHA256:
        raise ProtocolError("Docker lock manifest dataset identity mismatch")
    if raw.get("harness_commit") != v1.HARNESS_COMMIT:
        raise ProtocolError("Docker lock manifest harness commit mismatch")
    if (
        raw.get("platform") != "linux/amd64"
        or raw.get("harness_arch") != "x86_64"
        or raw.get("namespace") != "swebench"
        or raw.get("locked_alias_tag") != "mio-swe-v2-locked"
    ):
        raise ProtocolError("Docker lock manifest platform or alias controls mismatch")
    if (
        isinstance(expected_images, bool)
        or not isinstance(expected_images, int)
        or not 1 <= expected_images <= EXPECTED_PAIRS
    ):
        raise ProtocolError(f"expected_images must be an integer in [1, {EXPECTED_PAIRS}]")
    raw_images = raw.get("images")
    if not isinstance(raw_images, list) or len(raw_images) != expected_images:
        raise ProtocolError("Docker lock manifest image cardinality mismatch")
    images = [validate_docker_lock_record_syntax(item) for item in raw_images]
    if images != sorted(images, key=lambda item: item["instance_digest"]):
        raise ProtocolError("Docker lock records must be sorted by instance digest")
    if len({item["instance_digest"] for item in images}) != len(images):
        raise ProtocolError("Docker lock manifest contains duplicate instance identities")
    if len({item["expected_local_alias"] for item in images}) != len(images):
        raise ProtocolError("Docker lock manifest contains duplicate image aliases")
    bound_instance_digests: list[str] | None = None
    if expected_instance_digests is not None:
        if isinstance(expected_instance_digests, (str, bytes)):
            raise ProtocolError("expected instance digests must be a sequence of SHA-256 values")
        bound_instance_digests = list(expected_instance_digests)
        if (
            len(bound_instance_digests) != expected_images
            or len(set(bound_instance_digests)) != expected_images
            or any(not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in bound_instance_digests)
        ):
            raise ProtocolError("expected instance digest population is malformed")
        bound_instance_digests.sort()
        if [item["instance_digest"] for item in images] != bound_instance_digests:
            raise ProtocolError("Docker lock records differ from expected instance digests")
    document = {key: (images if key == "images" else raw[key]) for key in sorted(_DOCKER_LOCK_KEYS)}
    return {
        "status": "syntactic_non_evidence",
        "confirmatory_evidence_admissible": False,
        "daemon_materialization_attested": False,
        "expected_instance_digests_bound": bound_instance_digests is not None,
        "expected_instance_digests_sha256": (
            sha256_bytes(canonical_json_bytes(bound_instance_digests)) if bound_instance_digests is not None else None
        ),
        "document": document,
        "canonical_sha256_checksum_not_authenticity": sha256_bytes(canonical_json_bytes(document)),
    }


def main() -> int:
    document = load_and_validate_preregistration()
    print(
        json.dumps(
            {
                "schema": document["schema"],
                "confirmatory_enabled": CONFIRMATORY_ENABLED,
                "blockers": list(CONFIRMATORY_BLOCKERS),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
