"""Machine-readable SWE-bench quality v2 protocol tests."""

from __future__ import annotations

import json
import subprocess

import pytest

from mio import agent
from scripts import bench_swebench_quality_v2 as v2


def _metric(pair_index: int, condition: str, **changes):
    factor = 1.0 if condition == "gate_off" else 1.0
    values = {
        "schema": v2.ARM_METRIC_SCHEMA,
        "pair_index": pair_index,
        "condition": condition,
        "status": "completed",
        "termination_reason": "completed",
        "wall_elapsed_ns": int(1_000_000_000 * factor),
        "wall_observed_ns": int(1_000_000_000 * factor),
        "wall_limit_ns": v2.ARM_WALL_NS,
        "wall_censored": False,
        "watchdog_overrun_ns": 0,
        "round_count": 2,
        "tool_call_count": 3,
        "output_tokens": 100,
        "physical_prefill_tokens": 200,
        "prefill_ns": 200_000_000,
        "decode_tokens": 100,
        "decode_ns": 400_000_000,
        "phase_censored": False,
        "telemetry_complete": True,
    }
    values.update(changes)
    return values


def _records(pairs: int, *, wall_ratio=1.0, prefill_ratio=1.0, decode_ratio=1.0):
    records = []
    for pair_index in range(pairs):
        records.append(_metric(pair_index, "gate_off"))
        records.append(
            _metric(
                pair_index,
                "gate_on",
                wall_elapsed_ns=int(1_000_000_000 * wall_ratio),
                wall_observed_ns=int(1_000_000_000 * wall_ratio),
                prefill_ns=int(200_000_000 * prefill_ratio),
                decode_ns=int(400_000_000 * decode_ratio),
            )
        )
    return records


def _docker_record(index: int = 1):
    image = f"swebench/sweb.eval.x86_64.owner_1776_repo-{index}"
    config = f"sha256:{index:064x}"
    return {
        "schema": v2.DOCKER_RECORD_SCHEMA,
        "instance_digest": f"{index:064x}",
        "expected_local_alias": f"{image}:mio-swe-v2-locked",
        "source_tag": f"{image}:latest",
        "registry_index_digest": f"sha256:{index + 10:064x}",
        "linux_amd64_manifest_digest": f"sha256:{index + 20:064x}",
        "config_digest": config,
        "docker_image_id": config,
        "compressed_layer_digests": [f"sha256:{index + 30:064x}"],
        "rootfs_diff_ids": [f"sha256:{index + 40:064x}"],
        "os": "linux",
        "architecture": "amd64",
        "harness_arch": "x86_64",
    }


def _docker_manifest(images):
    return {
        "schema": v2.DOCKER_LOCK_SCHEMA,
        "v1_preregistration_sha256": v2.V1_PREREGISTRATION_CANONICAL_SHA256,
        "dataset_full_snapshot_sha256": v2.v1.FULL_SNAPSHOT_SHA256,
        "harness_commit": v2.v1.HARNESS_COMMIT,
        "platform": "linux/amd64",
        "harness_arch": "x86_64",
        "namespace": "swebench",
        "locked_alias_tag": "mio-swe-v2-locked",
        "images": images,
    }


def _round(**changes):
    values = {
        "schema": v2.ROUND_METRIC_SCHEMA,
        "round_index": 0,
        "generation_backend": "baseline",
        "drafter_requested": "target_ar",
        "drafter_selected": "baseline",
        "drafter_ref": None,
        "timing_source": "runtime_raw_ns",
        "effective_timeout_ns": 600_000_000_000,
        "logical_prompt_tokens": 200,
        "warm_offset_tokens": 0,
        "physical_prefill_tokens": 200,
        "completion_tokens": 100,
        "prefill_ns": 200_000_000,
        "decode_ns": 400_000_000,
        "model_total_ns": 600_000_000,
        "phase_censored": False,
        "deadline_hit": False,
    }
    values.update(changes)
    return values


def _tool(**changes):
    values = {
        "schema": v2.TOOL_METRIC_SCHEMA,
        "sequence": 0,
        "round_index": 0,
        "tool_name": "bash",
        "allowed": True,
        "outcome": "ok",
        "duration_ns": 10_000_000,
        "effective_timeout_ns": 300_000_000_000,
        "exit_code_or_signal": 0,
        "output_chars": 10,
        "target_hmac_sha256": "a" * 64,
    }
    values.update(changes)
    return values


def _two_round_arm(**changes):
    values = _metric(
        0,
        "gate_off",
        wall_elapsed_ns=2_000_000_000,
        wall_observed_ns=2_000_000_000,
        round_count=2,
        tool_call_count=0,
        output_tokens=200,
        physical_prefill_tokens=400,
        prefill_ns=400_000_000,
        decode_tokens=200,
        decode_ns=800_000_000,
    )
    values.update(changes)
    return values


def test_v2_inherits_exact_current_v1_canonical_digest_and_remains_blocked():
    assert v2.verify_v1_inheritance() == ("834b205733c02a81adaa8ad1cbfd3ab66bdb65575fc162742c53af246422d708")
    document = v2.load_and_validate_preregistration()
    assert document["inherits"]["canonical_sha256"] == v2.V1_PREREGISTRATION_CANONICAL_SHA256
    assert document["confirmatory_activation"]["enabled"] is False
    assert document["confirmatory_activation"]["blockers"] == list(v2.CONFIRMATORY_BLOCKERS)
    assert v2.preregistration_digest() == v2.V2_PREREGISTRATION_CANONICAL_SHA256
    assert document["docker_image_lock"]["manifest_sha256"] is None
    assert document["execution_chain_of_custody"]["status"] == "required_not_attested"
    assert (
        "native_executor_loaded_model_tree_chain_of_custody_not_attested"
        in document["confirmatory_activation"]["blockers"]
    )
    assert "supervisor_private_per_run_hmac_key_not_attested" in document["confirmatory_activation"]["blockers"]
    assert (
        "promotion_input_authority_and_receipt_binding_not_implemented"
        in document["confirmatory_activation"]["blockers"]
    )
    assert "v2_clean_commit_and_runtime_not_sealed" in document["confirmatory_activation"]["blockers"]
    assert v2.CONFIRMATORY_ENABLED is False
    with pytest.raises(v2.ProtocolError, match="hard-blocked"):
        v2.require_confirmatory_ready()


def test_machine_readable_protocol_freezes_limits_guardrails_and_quality_gates():
    document = json.loads(v2.PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    assert document["frozen_limits"] == {
        "arm_wall_seconds": 1800,
        "max_output_tokens_per_round": 4096,
        "max_output_tokens_per_arm": 24576,
        "max_agent_rounds": 12,
        "max_tool_calls_per_arm": 32,
        "maximum_pair_attempts": 3,
        "maximum_retries_after_initial_attempt": 2,
    }
    assert document["efficiency_guardrails"]["thresholds"] == {
        "wall_ratio_upper_95": 1.25,
        "prefill_cost_per_token_ratio_upper_95": 1.1,
        "decode_cost_per_token_ratio_upper_95": 1.05,
    }
    assert document["efficiency_guardrails"]["paired_bootstrap_samples"] == 10_000
    assert document["quality_and_promotion_gates"]["practical_resolution_difference_minimum"] == 0.02
    assert document["timeout_censoring"]["wall_timeout_is_retryable"] is False
    assert document["whole_pair_retry"]["retry_after_completed_attempt"] is False
    assert document["whole_pair_retry"]["authority"] == "trusted_supervisor_only"
    assert document["whole_pair_retry"]["classifier_is_authority"] is False
    assert document["whole_pair_retry"]["automatic_retry"] is False
    assert document == v2.EXPECTED_PREREGISTRATION_DOCUMENT


@pytest.mark.parametrize("block", tuple(v2.EXPECTED_PREREGISTRATION_DOCUMENT))
def test_preregistration_validator_rejects_mutation_of_every_top_level_block(tmp_path, block):
    document = json.loads(v2.PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    value = document[block]
    if isinstance(value, dict):
        value["_adversarial_mutation"] = True
    elif isinstance(value, bool):
        document[block] = not value
    else:
        document[block] = None
    path = tmp_path / "tampered-v2.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(v2.ProtocolError, match=f"frozen top-level block differs: {block}"):
        v2.load_and_validate_preregistration(path)


def test_pair_retry_is_supervisor_only_allowlisted_and_capped_at_three_attempts():
    classification = v2.classify_pair_retry_request(
        next_attempt_index=2,
        reason_code="infrastructure_host_loss",
    )
    assert classification["rerun_unit"] == "whole_pair"
    assert classification["reason_allowlisted"] is True
    assert classification["admissible_for_retry"] is False
    assert classification["missing_authority_artifacts"] == [
        "sealed_supervisor_incident_receipt",
        "sealed_pair_attempt_ledger",
    ]
    with pytest.raises(v2.ProtocolError, match="three total attempts"):
        v2.classify_pair_retry_request(
            next_attempt_index=3,
            reason_code="infrastructure_host_loss",
        )
    for nonretryable in (
        "arm_wall_timeout",
        "model_error",
        "protocol_violation",
        "budget_exhaustion",
        "patch_validation_failure",
        "official_patch_apply_failure",
        "official_evaluation_unresolved",
    ):
        with pytest.raises(v2.ProtocolError, match="non-retryable"):
            v2.classify_pair_retry_request(
                next_attempt_index=1,
                reason_code=nonretryable,
            )


def test_arm_metric_validation_enforces_content_free_schema_and_censoring():
    observed = v2.validate_arm_metric_record(_metric(0, "gate_off"))
    assert observed.wall_observed_ns == 1_000_000_000

    censored = _metric(
        0,
        "gate_on",
        status="timeout",
        termination_reason="arm_wall_timeout",
        wall_elapsed_ns=v2.ARM_WALL_NS + 2_000_000,
        wall_observed_ns=v2.ARM_WALL_NS,
        wall_censored=True,
        watchdog_overrun_ns=2_000_000,
    )
    assert v2.validate_arm_metric_record(censored).wall_censored is True

    censored["status"] = "completed"
    censored["termination_reason"] = "completed"
    with pytest.raises(v2.ProtocolError, match="capped terminal timeout"):
        v2.validate_arm_metric_record(censored)

    leaked = _metric(0, "gate_off")
    leaked["instance_id"] = "secret-instance"
    with pytest.raises(v2.ProtocolError, match="extra=.*instance_id"):
        v2.validate_arm_metric_record(leaked)


def test_round_metric_validator_enforces_target_ar_arithmetic_and_raw_ns():
    observed = v2.validate_round_metric_record(_round())
    assert observed.generation_backend == "baseline"
    assert observed.drafter_requested == "target_ar"
    assert observed.drafter_ref is None
    assert observed.timing_source == "runtime_raw_ns"
    assert observed.completion_tokens == _round()["completion_tokens"]

    for changes, message in (
        ({"generation_backend": "dflash"}, "target_ar baseline"),
        ({"drafter_ref": "/private/draft"}, "no drafter ref"),
        ({"timing_source": "wall_estimate_ns"}, "runtime_raw_ns"),
        ({"effective_timeout_ns": 600_000_000_001}, "frozen maximum"),
        ({"effective_timeout_ns": 500_000_000}, "effective timeout envelope"),
        ({"warm_offset_tokens": 10}, "logical prompt minus warm"),
        ({"model_total_ns": 599_999_999}, "must equal"),
        ({"deadline_hit": True}, "must mark"),
        ({"assistant_text": "secret"}, "extra=.*assistant_text"),
    ):
        with pytest.raises(v2.ProtocolError, match=message):
            v2.validate_round_metric_record(_round(**changes))


def test_real_agent_round_trace_maps_physical_and_delivered_tokens_separately():
    trace = agent.AgentRoundTrace(
        round_index=0,
        prompt_tokens=200,
        completion_tokens=100,
        total_time_s=0.21,
        prompt_tps=1_000.0,
        generation_tps=10.0,
        generation_backend="baseline",
        fallback_ar=False,
        prefill_ns=200_000_000,
        decode_ns=10_000_000,
        model_total_ns=210_000_000,
        logical_prompt_tokens=200,
        physical_prefill_tokens=200,
        physical_decode_tokens=101,
        warm_offset=0,
        warm_offset_tokens=0,
        timing_source="runtime_raw_ns",
        drafter_requested="target_ar",
        drafter_selected="baseline",
        drafter_ref=None,
        phase_censored=True,
        deadline_hit=True,
    )
    record = v2.round_metric_from_agent_trace(trace, effective_timeout_ns=200_000_000)
    assert record.completion_tokens == trace.physical_decode_tokens == 101
    assert v2.arm_output_tokens_from_agent_round_traces([trace]) == trace.completion_tokens == 100

    uncensored = agent.AgentRoundTrace(**{**trace.__dict__, "phase_censored": False})
    with pytest.raises(v2.ProtocolError, match="divergence requires phase censoring"):
        v2.round_metric_from_agent_trace(uncensored, effective_timeout_ns=200_000_000)


def test_tool_metric_validator_enforces_timeout_envelope_and_no_content():
    observed = v2.validate_tool_metric_record(_tool())
    assert observed.tool_name == "bash"
    timeout = _tool(
        outcome="timeout",
        duration_ns=301_000_000_000,
        exit_code_or_signal=None,
    )
    assert v2.validate_tool_metric_record(timeout).outcome == "timeout"
    read_timeout = _tool(
        tool_name="read",
        outcome="timeout",
        effective_timeout_ns=30_000_000_000,
        duration_ns=31_000_000_000,
        exit_code_or_signal=None,
    )
    assert v2.validate_tool_metric_record(read_timeout).effective_timeout_ns == 30_000_000_000

    for changes, message in (
        ({"effective_timeout_ns": 300_000_000_001}, "exceeds"),
        ({"outcome": "timeout", "duration_ns": 299_999_999_999}, "kill grace"),
        ({"outcome": "timeout", "duration_ns": 301_000_000_001}, "one-second"),
        ({"outcome": "ok", "exit_code_or_signal": 1}, "exit code zero"),
        ({"target_hmac_sha256": "A" * 64}, "lowercase"),
        ({"tool_arguments": "rm -rf"}, "extra=.*tool_arguments"),
    ):
        with pytest.raises(v2.ProtocolError, match=message):
            v2.validate_tool_metric_record(_tool(**changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "tool_name": "read",
                "outcome": "unrecognized",
                "allowed": False,
                "effective_timeout_ns": 30_000_000_000,
                "exit_code_or_signal": None,
            },
            "name/outcome combination",
        ),
        (
            {
                "tool_name": "unknown",
                "outcome": "denied",
                "allowed": False,
                "effective_timeout_ns": 0,
                "exit_code_or_signal": None,
            },
            "name/outcome combination",
        ),
        (
            {
                "tool_name": "bash",
                "outcome": "untrusted_executable",
                "allowed": False,
                "exit_code_or_signal": None,
            },
            "name/outcome combination",
        ),
        (
            {
                "tool_name": "write",
                "outcome": "no_work",
                "effective_timeout_ns": 30_000_000_000,
                "exit_code_or_signal": None,
            },
            "name/outcome combination",
        ),
        (
            {
                "tool_name": "read",
                "effective_timeout_ns": 30_000_000_000,
                "exit_code_or_signal": 0,
            },
            "file and unknown",
        ),
    ],
)
def test_tool_name_outcome_and_exit_taxonomy_is_bidirectional(changes, message):
    with pytest.raises(v2.ProtocolError, match=message):
        v2.validate_tool_metric_record(_tool(**changes))

    validate_no_work = v2.validate_tool_metric_record(
        _tool(tool_name="validate", outcome="no_work", exit_code_or_signal=0)
    )
    assert validate_no_work.outcome == "no_work"
    signaled_nonzero = v2.validate_tool_metric_record(_tool(outcome="nonzero", exit_code_or_signal="signal:9"))
    assert signaled_nonzero.exit_code_or_signal == "signal:9"


def test_arm_bundle_recomputes_round_tool_summaries_and_hashes():
    arm = _metric(0, "gate_off", round_count=1, tool_call_count=1)
    bundle = v2.validate_arm_telemetry_bundle(arm, [_round()], [_tool()])
    assert bundle["round_records"] == 1
    assert bundle["tool_records"] == 1
    assert len(bundle["round_telemetry_sha256"]) == 64
    assert len(bundle["tool_telemetry_sha256"]) == 64

    with pytest.raises(v2.ProtocolError, match="raw telemetry"):
        v2.validate_arm_telemetry_bundle(
            {**arm, "prefill_ns": arm["prefill_ns"] + 1},
            [_round()],
            [_tool()],
        )
    with pytest.raises(v2.ProtocolError, match="sequence"):
        v2.validate_arm_telemetry_bundle(arm, [_round()], [_tool(sequence=1)])


def test_censored_round_seals_physical_decode_not_delivered_to_agent_loop():
    round_record = _round(
        effective_timeout_ns=200_000_000,
        completion_tokens=1,
        decode_ns=10_000_000,
        model_total_ns=210_000_000,
        phase_censored=True,
        deadline_hit=True,
    )
    arm = _metric(
        0,
        "gate_off",
        status="timeout",
        termination_reason="model_round_timeout",
        round_count=1,
        tool_call_count=0,
        output_tokens=0,
        decode_tokens=1,
        decode_ns=10_000_000,
        phase_censored=True,
        telemetry_complete=False,
    )
    bundle = v2.validate_arm_telemetry_bundle(arm, [round_record], [])
    assert bundle["arm"]["output_tokens"] == 0
    assert bundle["arm"]["decode_tokens"] == 1
    assert v2.TELEMETRY_SEMANTICS["round_completion_tokens_source"] == "AgentRoundTrace.physical_decode_tokens"
    assert v2.TELEMETRY_SEMANTICS["arm_output_tokens_source"] == "sum_AgentRoundTrace.completion_tokens"

    with pytest.raises(v2.ProtocolError, match="divergence requires phase-censored"):
        v2.validate_arm_metric_record(_metric(0, "gate_off", output_tokens=99, decode_tokens=100))


def test_operation_timeouts_are_distinct_from_the_arm_wall_censor():
    tool_timeout = _tool(
        outcome="timeout",
        duration_ns=301_000_000_000,
        exit_code_or_signal="signal:9",
    )
    arm = _metric(
        0,
        "gate_off",
        status="timeout",
        termination_reason="tool_timeout",
        wall_elapsed_ns=400_000_000_000,
        wall_observed_ns=400_000_000_000,
        round_count=1,
        tool_call_count=1,
    )
    bundle = v2.validate_arm_telemetry_bundle(arm, [_round()], [tool_timeout])
    assert bundle["arm"]["wall_censored"] is False
    assert bundle["arm"]["termination_reason"] == "tool_timeout"

    with pytest.raises(v2.ProtocolError, match="disagrees with arm termination"):
        v2.validate_arm_telemetry_bundle(
            {**arm, "status": "completed", "termination_reason": "completed"},
            [_round()],
            [tool_timeout],
        )


def test_bundle_enforces_monotonic_tool_round_topology():
    rounds = [_round(round_index=0), _round(round_index=1)]
    arm = _two_round_arm(tool_call_count=2)
    tools = [
        _tool(
            sequence=0,
            round_index=1,
            tool_name="read",
            effective_timeout_ns=30_000_000_000,
            exit_code_or_signal=None,
        ),
        _tool(
            sequence=1,
            round_index=0,
            tool_name="read",
            effective_timeout_ns=30_000_000_000,
            exit_code_or_signal=None,
        ),
    ]
    with pytest.raises(v2.ProtocolError, match="round_index values must be nondecreasing"):
        v2.validate_arm_telemetry_bundle(arm, rounds, tools)


def test_terminal_tool_timeout_and_nonzero_must_be_in_final_round():
    rounds = [_round(round_index=0), _round(round_index=1)]
    timeout_arm = _two_round_arm(
        status="timeout",
        termination_reason="tool_timeout",
        wall_elapsed_ns=40_000_000_000,
        wall_observed_ns=40_000_000_000,
        tool_call_count=1,
    )
    timeout_tool = _tool(
        round_index=0,
        tool_name="read",
        outcome="timeout",
        duration_ns=31_000_000_000,
        effective_timeout_ns=30_000_000_000,
        exit_code_or_signal=None,
    )
    with pytest.raises(v2.ProtocolError, match="last tool in the final model round"):
        v2.validate_arm_telemetry_bundle(timeout_arm, rounds, [timeout_tool])

    nonzero_arm = _two_round_arm(
        status="incomplete",
        termination_reason="tool_nonzero",
        tool_call_count=1,
    )
    nonzero_tool = _tool(round_index=0, outcome="nonzero", exit_code_or_signal=1)
    with pytest.raises(v2.ProtocolError, match="last tool in the final model round"):
        v2.validate_arm_telemetry_bundle(nonzero_arm, rounds, [nonzero_tool])

    terminal_nonzero = _tool(round_index=1, outcome="nonzero", exit_code_or_signal=1)
    bundle = v2.validate_arm_telemetry_bundle(nonzero_arm, rounds, [terminal_nonzero])
    assert bundle["arm"]["termination_reason"] == "tool_nonzero"


def test_arm_wall_timeout_rejects_multiple_or_followed_terminal_events():
    wall_arm = _metric(
        0,
        "gate_off",
        status="timeout",
        termination_reason="arm_wall_timeout",
        wall_elapsed_ns=v2.ARM_WALL_NS + 1_000_000,
        wall_observed_ns=v2.ARM_WALL_NS,
        wall_censored=True,
        watchdog_overrun_ns=1_000_000,
        round_count=1,
        tool_call_count=2,
    )
    timeout_tools = [
        _tool(
            sequence=index,
            tool_name="read",
            outcome="timeout",
            duration_ns=31_000_000_000,
            effective_timeout_ns=30_000_000_000,
            exit_code_or_signal=None,
        )
        for index in range(2)
    ]
    with pytest.raises(v2.ProtocolError, match="multiple terminal timeout events"):
        v2.validate_arm_telemetry_bundle(wall_arm, [_round()], timeout_tools)

    rounds = [
        _round(round_index=0, phase_censored=True, deadline_hit=True),
        _round(round_index=1),
    ]
    followed_arm = _two_round_arm(
        status="timeout",
        termination_reason="arm_wall_timeout",
        wall_elapsed_ns=v2.ARM_WALL_NS + 1_000_000,
        wall_observed_ns=v2.ARM_WALL_NS,
        wall_censored=True,
        watchdog_overrun_ns=1_000_000,
        phase_censored=True,
        telemetry_complete=False,
    )
    with pytest.raises(v2.ProtocolError, match="no telemetry event may follow"):
        v2.validate_arm_telemetry_bundle(followed_arm, rounds, [])


def test_unknown_tool_name_uses_a_content_free_sentinel():
    unknown = _tool(
        tool_name="unknown",
        allowed=False,
        outcome="unrecognized",
        duration_ns=1_000,
        effective_timeout_ns=0,
        exit_code_or_signal=None,
        output_chars=19,
    )
    assert v2.validate_tool_metric_record(unknown).tool_name == "unknown"
    assert v2.TELEMETRY_SEMANTICS["tool_target_hmac_source"] == (
        "HMAC_SHA256(private_per_run_key,AgentToolTrace.target_sha256)"
    )
    with pytest.raises(v2.ProtocolError, match="frozen vocabulary"):
        v2.validate_tool_metric_record({**unknown, "tool_name": "secret_custom_tool"})


def test_validate_trace_is_shape_valid_but_command_adapter_is_hard_blocked():
    visible_result = "(validation rejected: executable must resolve outside the workspace)"
    event = agent.AgentAuditEvent(
        timestamp=1.0,
        operation="validate",
        permission="shell",
        target="sha256:" + "b" * 64,
        allowed=False,
        outcome="untrusted_executable",
        detail="content-free detail",
    )
    trace = agent._tool_trace(
        sequence=0,
        round_index=0,
        tool_name="validate",
        args={"argv": ["pytest", "-q"]},
        events=(event,),
        result=visible_result,
        fallback_outcome="error",
        duration_ns=10,
        effective_timeout_ns=300_000_000_000,
        timeout_enforced=False,
        telemetry_complete=True,
        known_tool=True,
        permission_fallback="shell",
    )
    shape_record = v2.validate_tool_metric_record(
        _tool(
            tool_name="validate",
            allowed=False,
            outcome="untrusted_executable",
            exit_code_or_signal=None,
            output_chars=len(visible_result),
        )
    )
    assert shape_record.outcome == "untrusted_executable"
    bash_trace = agent.AgentToolTrace(**{**trace.__dict__, "tool_name": "bash"})
    for blocked_trace in (trace, bash_trace):
        with pytest.raises(v2.ProtocolError, match="bash/validate adapter is blocked"):
            v2.tool_metric_shape_from_agent_trace(blocked_trace, target_hmac_key=b"k" * 32)

    unknown_result = "(unknown tool: secret-unregistered-tool)"
    unknown_trace = agent._tool_trace(
        sequence=1,
        round_index=0,
        tool_name="secret-unregistered-tool",
        args={"path": "private.py"},
        events=(),
        result=unknown_result,
        fallback_outcome="unrecognized",
        duration_ns=10,
        effective_timeout_ns=None,
        timeout_enforced=False,
        telemetry_complete=True,
        known_tool=False,
    )
    unknown_record = v2.tool_metric_shape_from_agent_trace(
        unknown_trace,
        target_hmac_key=b"k" * 32,
    )
    assert unknown_record.tool_name == "unknown"
    assert unknown_record.effective_timeout_ns == 0
    assert unknown_record.output_chars == len(unknown_result)
    assert (
        unknown_record.target_hmac_sha256
        != v2.tool_metric_shape_from_agent_trace(
            unknown_trace,
            target_hmac_key=b"z" * 32,
        ).target_hmac_sha256
    )


@pytest.mark.parametrize(
    ("tool_name", "outcome", "visible_result"),
    [
        ("read", "not_found", "(file not found: missing.py)"),
        ("edit", "old_string_not_found", "(old_string not found in value.py)"),
    ],
)
def test_real_allowed_nonterminal_tool_outcomes_map_exactly(
    tool_name,
    outcome,
    visible_result,
):
    event = agent.AgentAuditEvent(
        timestamp=1.0,
        operation=tool_name,
        permission="read" if tool_name == "read" else "write",
        target="sha256:" + "c" * 64,
        allowed=True,
        outcome=outcome,
    )
    trace = agent._tool_trace(
        sequence=0,
        round_index=0,
        tool_name=tool_name,
        args={"path": "missing.py"},
        events=(event,),
        result=visible_result,
        fallback_outcome="error",
        duration_ns=10,
        effective_timeout_ns=30_000_000_000,
        timeout_enforced=True,
        telemetry_complete=True,
        known_tool=True,
    )
    record = v2.tool_metric_shape_from_agent_trace(trace, target_hmac_key=b"k" * 32)
    assert record.allowed is True
    assert record.outcome == outcome
    assert record.output_chars == len(visible_result)
    assert record.target_hmac_sha256 != trace.target_sha256
    assert "target_sha256" not in record.as_dict()

    unenforced_trace = agent.AgentToolTrace(**{**trace.__dict__, "timeout_enforced": False})
    with pytest.raises(v2.ProtocolError, match="timeout_enforced"):
        v2.tool_metric_shape_from_agent_trace(unenforced_trace, target_hmac_key=b"k" * 32)

    with pytest.raises(v2.ProtocolError, match="frozen vocabulary"):
        v2.validate_tool_metric_record({**record.as_dict(), "outcome": "arbitrary_result"})


def test_tool_metric_is_exactly_one_record_per_invocation_not_audit_event():
    arm = _metric(0, "gate_off", round_count=1, tool_call_count=2)
    with pytest.raises(v2.ProtocolError, match="sequence"):
        v2.validate_arm_telemetry_bundle(
            arm,
            [_round()],
            [_tool(sequence=0), _tool(sequence=0)],
        )


def test_bundle_rejects_physically_impossible_tool_and_model_duration_sum():
    arm = _metric(0, "gate_off", round_count=1, tool_call_count=1)
    with pytest.raises(v2.ProtocolError, match="sequential model and tool durations"):
        v2.validate_arm_telemetry_bundle(
            arm,
            [_round()],
            [_tool(duration_ns=500_000_001)],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"wall_limit_ns": v2.ARM_WALL_NS - 1}, "frozen 1800"),
        ({"round_count": 13}, "frozen maximum"),
        ({"tool_call_count": 33}, "frozen maximum"),
        ({"pair_index": 500}, "frozen maximum"),
        ({"output_tokens": 101}, "cannot exceed physically decoded"),
        ({"phase_censored": True, "telemetry_complete": True}, "cannot be declared complete"),
        ({"prefill_ns": 0}, "prefill token/time"),
    ],
)
def test_arm_metric_validation_fails_closed_on_inconsistent_limits(changes, message):
    metric = _metric(0, "gate_off")
    metric.update(changes)
    with pytest.raises(v2.ProtocolError, match=message):
        v2.validate_arm_metric_record(metric)


def test_guardrail_aggregate_uses_paired_bootstrap_and_frozen_thresholds():
    aggregate = v2.aggregate_efficiency_guardrails(
        _records(4, wall_ratio=1.20, prefill_ratio=1.08, decode_ratio=1.04),
        expected_pairs=4,
        bootstrap_samples=128,
    )
    assert aggregate["phase_telemetry_complete"] is True
    assert aggregate["confirmatory_evidence_admissible"] is False
    assert aggregate["guardrails"]["wall_ratio"]["estimate"] == pytest.approx(1.20)
    assert aggregate["guardrails"]["wall_ratio"]["upper_95"] == pytest.approx(1.20)
    assert aggregate["guardrails"]["prefill_cost_per_token_ratio"]["upper_95"] == pytest.approx(1.08)
    assert aggregate["guardrails"]["decode_cost_per_token_ratio"]["upper_95"] == pytest.approx(1.04)
    assert aggregate["all_passed"] is True
    assert aggregate == v2.aggregate_efficiency_guardrails(
        _records(4, wall_ratio=1.20, prefill_ratio=1.08, decode_ratio=1.04),
        expected_pairs=4,
        bootstrap_samples=128,
    )


@pytest.mark.parametrize(
    ("ratios", "guardrail"),
    [
        ({"wall_ratio": 1.26}, "wall_ratio"),
        ({"prefill_ratio": 1.11}, "prefill_cost_per_token_ratio"),
        ({"decode_ratio": 1.06}, "decode_cost_per_token_ratio"),
    ],
)
def test_each_efficiency_threshold_independently_blocks_promotion(ratios, guardrail):
    aggregate = v2.aggregate_efficiency_guardrails(
        _records(4, **ratios),
        expected_pairs=4,
        bootstrap_samples=64,
    )
    assert aggregate["guardrails"][guardrail]["passed"] is False
    assert aggregate["all_passed"] is False


def test_missing_or_phase_censored_telemetry_fails_efficiency_closed():
    records = _records(2)
    records[1]["telemetry_complete"] = False
    aggregate = v2.aggregate_efficiency_guardrails(
        records,
        expected_pairs=2,
        bootstrap_samples=32,
    )
    assert aggregate["phase_telemetry_complete"] is False
    assert aggregate["guardrails"]["wall_ratio"]["passed"] is True
    assert aggregate["guardrails"]["prefill_cost_per_token_ratio"]["upper_95"] is None
    assert aggregate["guardrails"]["decode_cost_per_token_ratio"]["upper_95"] is None
    assert aggregate["all_passed"] is False


def test_pair_validation_rejects_missing_duplicate_or_noncontiguous_arms():
    with pytest.raises(v2.ProtocolError, match="exactly two arms"):
        v2.aggregate_efficiency_guardrails(
            _records(2)[:-1],
            expected_pairs=2,
            bootstrap_samples=8,
        )
    duplicate = _records(2)
    duplicate[1]["condition"] = "gate_off"
    with pytest.raises(v2.ProtocolError, match="duplicate pair condition"):
        v2.aggregate_efficiency_guardrails(duplicate, expected_pairs=2, bootstrap_samples=8)


def test_promotion_combines_inherited_quality_and_v2_efficiency_gates():
    small = v2.aggregate_efficiency_guardrails(
        _records(2),
        expected_pairs=2,
        bootstrap_samples=16,
    )
    relabeled = {
        **small,
        "pairs": v2.EXPECTED_PAIRS,
        "bootstrap_samples": v2.BOOTSTRAP_SAMPLES,
        "frozen_confirmatory_shape": True,
    }
    with pytest.raises(v2.ProtocolError, match="calculation-integrity"):
        v2.promotion_decision(
            {
                "resolution_difference": 0.02,
                "paired_bootstrap_lower_95": 0.001,
                "exact_one_sided_mcnemar_p": 0.049,
                "full_500_pairs": True,
            },
            relabeled,
        )
    unsigned_relabeled = {key: value for key, value in relabeled.items() if key != "calculation_integrity_sha256"}
    relabeled["calculation_integrity_sha256"] = v2.sha256_bytes(v2.canonical_json_bytes(unsigned_relabeled))
    with pytest.raises(v2.ProtocolError, match="not a v2 guardrail"):
        v2.promotion_decision(
            {
                "resolution_difference": 0.02,
                "paired_bootstrap_lower_95": 0.001,
                "exact_one_sided_mcnemar_p": 0.049,
                "full_500_pairs": True,
            },
            small,
        )

    efficiency = v2.aggregate_efficiency_guardrails(_records(v2.EXPECTED_PAIRS))
    quality = {
        "resolution_difference": 0.02,
        "paired_bootstrap_lower_95": 0.001,
        "exact_one_sided_mcnemar_p": 0.049,
        "full_500_pairs": True,
    }
    decision = v2.promotion_decision(quality, efficiency)
    assert decision["shape_only_formula_outputs"]["combined_formula"] is True
    assert decision["mathematical_criteria_met_unverified"] is False
    assert decision["quality_improvement_passed"] is False
    assert decision["efficiency_guardrails_passed"] is False
    assert decision["promote"] is False
    assert decision["confirmatory_enabled"] is False
    assert decision["inputs_receipt_bound"] is False
    assert decision["input_authority_binding_implemented"] is False
    assert decision["caller_counts_or_checksum_are_evidence"] is False

    relabeled_decision = v2.promotion_decision(quality, relabeled)
    assert relabeled_decision["shape_only_formula_outputs"]["combined_formula"] is True
    assert relabeled_decision["mathematical_criteria_met_unverified"] is False
    assert relabeled_decision["promote"] is False
    assert v2.promotion_decision({**quality, "resolution_difference": 0.019}, efficiency)["promote"] is False
    assert (
        v2.promotion_decision({**quality, "resolution_difference": 0.019}, efficiency)[
            "mathematical_criteria_met_unverified"
        ]
        is False
    )
    assert (
        v2.promotion_decision({**quality, "paired_bootstrap_lower_95": 0.0}, efficiency)[
            "mathematical_criteria_met_unverified"
        ]
        is False
    )

    forged = {**efficiency, "pairs": 499}
    with pytest.raises(v2.ProtocolError, match="calculation-integrity"):
        v2.promotion_decision(quality, forged)


def test_docker_image_lock_record_validates_amd64_digests_and_rootfs():
    observed = v2.validate_docker_lock_record_syntax(_docker_record())
    assert observed["architecture"] == "amd64"
    assert observed["harness_arch"] == "x86_64"
    assert observed["config_digest"] == observed["docker_image_id"]

    for changes, message in (
        ({"architecture": "arm64"}, "linux/amd64"),
        ({"docker_image_id": "sha256:" + "f" * 64}, "equal the locked"),
        ({"rootfs_diff_ids": []}, "RootFS diff IDs"),
        ({"expected_local_alias": "swebench/wrong:mio-swe-v2-locked"}, "frozen local"),
    ):
        record = _docker_record()
        record.update(changes)
        with pytest.raises(v2.ProtocolError, match=message):
            v2.validate_docker_lock_record_syntax(record)


def test_complete_docker_lock_manifest_is_canonical_unique_and_digest_bound():
    images = [_docker_record(1), _docker_record(2)]
    result = v2.validate_docker_lock_manifest_syntax(
        _docker_manifest(images),
        expected_images=2,
        expected_instance_digests=[item["instance_digest"] for item in images],
    )
    assert result["status"] == "syntactic_non_evidence"
    assert result["confirmatory_evidence_admissible"] is False
    assert result["daemon_materialization_attested"] is False
    assert result["expected_instance_digests_bound"] is True
    assert len(result["document"]["images"]) == 2
    assert result["canonical_sha256_checksum_not_authenticity"] == v2.sha256_bytes(
        v2.canonical_json_bytes(result["document"])
    )

    with pytest.raises(v2.ProtocolError, match="sorted"):
        v2.validate_docker_lock_manifest_syntax(
            _docker_manifest(list(reversed(images))),
            expected_images=2,
        )
    with pytest.raises(v2.ProtocolError, match="duplicate instance"):
        v2.validate_docker_lock_manifest_syntax(
            _docker_manifest([images[0], {**images[1], "instance_digest": images[0]["instance_digest"]}]),
            expected_images=2,
        )
    with pytest.raises(v2.ProtocolError, match="differ from expected"):
        v2.validate_docker_lock_manifest_syntax(
            _docker_manifest(images),
            expected_images=2,
            expected_instance_digests=["f" * 64, images[1]["instance_digest"]],
        )


def test_v2_module_has_no_docker_or_benchmark_execution_surface():
    assert not hasattr(v2, "run_evaluation")
    assert not hasattr(v2, "run_generation")
    assert not hasattr(v2, "pull_docker_images")


def test_v2_script_entrypoint_reports_the_hard_block() -> None:
    completed = subprocess.run(
        ["python3", str(v2.PREREGISTRATION_PATH.parents[1] / "scripts" / "bench_swebench_quality_v2.py")],
        cwd=v2.PREREGISTRATION_PATH.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(completed.stdout)

    assert status["confirmatory_enabled"] is False
    assert status["blockers"] == list(v2.CONFIRMATORY_BLOCKERS)
