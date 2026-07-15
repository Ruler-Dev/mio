"""BMP-DFlash runtime loop — batched multi-path speculative decoding.

Drop-in alternative to `generate_dflash_once` / `stream_dflash_generate` that
branches verification across the batch dimension. See mio/dflash/bmp.py for
the primitives and docs/08-bmp-dflash.md for the full writeup.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx

from mio.dflash.bmp import (
    build_bmp_batch,
    expand_cache_batch,
    extract_top_k,
    filter_cache_batch,
    per_row_acceptance,
)
from mio.dflash.model import DFlashDraftModel


def generate_bmp_dflash_once(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    num_paths: int = 2,
    tree_budget: Optional[int] = None,
    block_tokens: int = 16,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
) -> dict[str, Any]:
    """DFlash with K-path batched verification.

    Each round: one draft pass → K candidate paths → batch-K target verify → pick
    row with longest accepted prefix → filter caches to winner → commit.

    Constraints:
      - Linear/SSM layers use ArraysCache (non-rollback) so batch-expand/filter is safe.
      - BMP is incompatible with TurboQuant caches at the moment (tq_bits must be None).
      - block_tokens capped by draft_model.block_size.
    """
    # Local imports to keep runtime.py the source of truth for these helpers.
    from mio.dflash.runtime import (
        _arm_target_rollback_with_prefix,
        _lm_head_logits,
        _prepare_prompt_tokens,
        _target_embed_tokens,
        _resolve_dflash_max_ctx,
        _resolve_draft_window,
        _effective_draft_window,
        _resolve_verify_len_cap,
        build_suppress_token_mask,
        configure_full_attention_split,
        extract_context_feature_from_dict,
        generate_baseline_once,
        greedy_tokens_with_mask,
        make_target_cache,
        target_forward_with_hidden_states,
    )
    from mio.dflash.model import ContextOnlyDraftKVCache

    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

    # BMP expands the batch dim mid-run, so disable split-attention
    # (its hook assumes a single trajectory) and speculative linear rollback
    # (the tape would be ambiguous across batch rows).
    try:
        configure_full_attention_split(target_model, enabled=False)
    except Exception:
        pass

    draft_sink_size, draft_window_size = _resolve_draft_window()
    draft_window_size = _effective_draft_window(draft_model, draft_window_size)

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(
            tokenizer, prompt, use_chat_template=use_chat_template
        )
    )
    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        baseline = generate_baseline_once(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
        )
        baseline.update({
            "accepted_from_draft": 0,
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "phase_timings_us": {
                "prefill": baseline["elapsed_us"], "draft": 0.0,
                "draft_prefill": 0.0, "draft_incremental": 0.0,
                "verify": 0.0, "replay": 0.0, "commit": 0.0,
            },
            "fallback_ar": True,
            "fallback_reason": fallback_reason,
            "num_paths": num_paths,
            "bmp": True,
        })
        return baseline

    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    # Rollback-capable caches for SSM layers; lets us avoid a replay forward
    # by rolling back the tape from block_len → commit_count after batch filter.
    target_cache = make_target_cache(
        target_model, enable_speculative_linear_cache=True,
    )
    draft_cache = [
        ContextOnlyDraftKVCache(sink_size=draft_sink_size, window_size=draft_window_size)
        for _ in range(len(draft_model.layers))
    ]
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
        target_model,
        input_ids=prompt_array,
        cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
    mx.eval(prefill_logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns

    suppress_token_mask = build_suppress_token_mask(
        int(prefill_logits.shape[-1]), suppress_token_ids
    )
    staged_first = greedy_tokens_with_mask(
        prefill_logits[:, -1, :], suppress_token_mask
    ).reshape(-1)
    target_hidden = extract_context_feature_from_dict(
        prefill_hidden_states, list(draft_model.target_layer_ids),
    )

    effective_block_tokens = max(1, min(int(block_tokens or 1), int(draft_model.block_size)))
    verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)

    generated_token_buffer = mx.full((max_new_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
    block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
    generated_token_count = 0
    accepted_from_draft = 0
    cycles_completed = 0
    start = prompt_len

    draft_ns_total = 0
    draft_prefill_ns = 0
    draft_incremental_ns = 0
    verify_ns_total = 0
    commit_ns_total = 0
    seen_draft_cycle = False
    acceptance_history: list[int] = []
    winner_history: list[int] = []

    while generated_token_count < max_new_tokens:
        remaining = max_new_tokens - generated_token_count
        block_len = max(1, min(effective_block_tokens, remaining))
        block_token_buffer[:block_len] = draft_model.mask_token_id
        block_token_buffer[:1] = staged_first
        block_token_ids = block_token_buffer[:block_len]
        bonus_token = int(staged_first.item())

        # ----- Draft -----
        path_rows: mx.array
        if block_len > 1:
            draft_start_ns = time.perf_counter_ns()
            noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
            mx.eval(draft_logits)
            draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
            draft_ns_total += draft_cycle_ns
            if not seen_draft_cycle:
                draft_prefill_ns += draft_cycle_ns
                seen_draft_cycle = True
            else:
                draft_incremental_ns += draft_cycle_ns

            # Top-K per position, then BMP path batch.
            verify_block_len = min(block_len, verify_len_cap)
            top_k_tokens, top_k_logps = extract_top_k(
                draft_logits, k=max(num_paths, 2), suppress_token_mask=suppress_token_mask,
            )
            path_rows, _paths_py = build_bmp_batch(
                bonus_token=bonus_token,
                top_k_tokens=top_k_tokens,
                top_k_logprobs=top_k_logps,
                num_paths=num_paths,
                block_len=verify_block_len,
                tree_budget=tree_budget,
            )
        else:
            # block_len == 1: no speculation possible. Just run bonus token.
            path_rows = mx.array([[bonus_token]], dtype=mx.uint32)

        K_effective = int(path_rows.shape[0])
        winner_history.append(K_effective)  # placeholder, overwritten below

        # ----- Verify batch=K -----
        # Expand caches BEFORE arming rollback so snapshots are taken at batch=K.
        if K_effective > 1:
            expand_cache_batch(target_cache, K_effective)
        _arm_target_rollback_with_prefix(target_cache, prefix_len=start)

        verify_start_ns = time.perf_counter_ns()
        verify_logits, verify_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=path_rows,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        mx.eval(verify_logits)
        verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
        verify_ns_total += verify_cycle_ns

        posterior = greedy_tokens_with_mask(verify_logits, suppress_token_mask)  # (K, block_len)

        # ----- Per-row acceptance -----
        if path_rows.shape[1] >= 2:
            per_row = per_row_acceptance(path_rows, posterior)
        else:
            per_row = [0] * K_effective
        winner = 0
        best = per_row[0]
        for i in range(1, len(per_row)):
            if per_row[i] > best:
                winner = i
                best = per_row[i]
        acceptance_len = best
        winner_history[-1] = winner
        acceptance_history.append(acceptance_len)

        # ----- Commit winner: filter caches to winning row, then rollback tape -----
        commit_start_ns = time.perf_counter_ns()
        commit_count = 1 + acceptance_len
        winning_row = path_rows[winner]
        committed_segment = winning_row[:commit_count]

        # Hidden states of the winner before rollback (for next draft round).
        committed_hidden = extract_context_feature_from_dict(
            verify_hidden_states, list(draft_model.target_layer_ids),
        )[winner : winner + 1, :commit_count, :]
        mx.eval(committed_hidden, committed_segment)
        target_hidden = committed_hidden

        # Filter caches (incl. rollback tapes) to winner, then rollback to acceptance_len.
        if K_effective > 1:
            filter_cache_batch(target_cache, winner)

        # Use DFlash's existing rollback to rewind SSM state from block_len → commit_count.
        from mio.dflash.runtime import _restore_target_cache_after_acceptance
        _restore_target_cache_after_acceptance(
            target_cache,
            target_len=start + commit_count,
            acceptance_length=acceptance_len,
            drafted_tokens=int(path_rows.shape[1]) - 1,
        )

        generated_token_buffer[generated_token_count : generated_token_count + commit_count] = committed_segment
        generated_token_count += commit_count
        accepted_from_draft += acceptance_len

        start += commit_count
        cycles_completed += 1
        commit_ns_total += time.perf_counter_ns() - commit_start_ns

        # Stop token?
        if stop_token_array is not None:
            stop_hit = bool(
                mx.any(
                    mx.equal(
                        committed_segment[:, None], stop_token_array[None, :]
                    )
                ).item()
            )
            if stop_hit:
                break

        staged_first = posterior[winner : winner + 1, acceptance_len].reshape(-1)
        mx.eval(staged_first)

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    generated_token_ids = (
        generated_token_buffer[:generated_token_count].tolist()
        if generated_token_count > 0 else []
    )
    return {
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_token_ids,
        "generation_tokens": len(generated_token_ids),
        "accepted_from_draft": accepted_from_draft,
        "acceptance_ratio": (
            accepted_from_draft / len(generated_token_ids)
            if generated_token_ids else 0.0
        ),
        "cycles_completed": cycles_completed,
        "tokens_per_cycle": (
            len(generated_token_ids) / cycles_completed
            if cycles_completed > 0 else 0.0
        ),
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": draft_ns_total / 1_000.0,
            "draft_prefill": draft_prefill_ns / 1_000.0,
            "draft_incremental": draft_incremental_ns / 1_000.0,
            "verify": verify_ns_total / 1_000.0,
            "replay": 0.0,
            "commit": commit_ns_total / 1_000.0,
        },
        "peak_memory_gb": (
            float(mx.get_peak_memory()) / 1e9
            if hasattr(mx, "get_peak_memory") else None
        ),
        "bmp": True,
        "num_paths": num_paths,
        "winner_history": winner_history[-40:],
        "acceptance_history": acceptance_history[-40:],
    }
