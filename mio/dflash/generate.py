# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from mio.dflash.runtime import (
    generate_baseline_once,
    generate_dflash_once,
    load_draft_bundle,
    load_target_bundle,
    stream_baseline_generate,
    stream_dflash_generate,
)


DRAFT_REGISTRY = {
    # Original HF repos
    "Qwen/Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen/Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen/Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen/Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    # mlx-community quantized
    "mlx-community/Qwen3.5-27B-8bit": "z-lab/Qwen3.5-27B-DFlash",
    "mlx-community/Qwen3.5-27B-4bit": "z-lab/Qwen3.5-27B-DFlash",
    "mlx-community/Qwen3.5-35B-A3B-4bit": "z-lab/Qwen3.5-35B-A3B-DFlash",
    # PARO quantized (z-lab)
    "z-lab/Qwen3.5-4B-PARO": "z-lab/Qwen3.5-4B-DFlash",
    "z-lab/Qwen3.5-9B-PARO": "z-lab/Qwen3.5-9B-DFlash",
    "z-lab/Qwen3.5-27B-PARO": "z-lab/Qwen3.5-27B-DFlash",
    "z-lab/Qwen3.5-35B-A3B-PARO": "z-lab/Qwen3.5-35B-A3B-DFlash",
}


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids


def resolve_optional_draft_ref(model_ref: str, draft_ref: Optional[str]) -> Optional[str]:
    if draft_ref:
        return draft_ref
    return DRAFT_REGISTRY.get(model_ref)


def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return str(tokenizer.decode(int(token_id)))


def generation_tps_from_summary(summary: dict[str, Any]) -> float:
    elapsed_us = float(summary.get("elapsed_us", 0.0))
    phase_timings = dict(summary.get("phase_timings_us", {}))
    prefill_us = float(summary.get("prefill_us", phase_timings.get("prefill", 0.0)))
    generation_tokens = int(summary.get("generation_tokens", 0))
    generation_us = max(0.0, elapsed_us - prefill_us)
    return (generation_tokens / (generation_us / 1e6)) if generation_us > 0.0 else 0.0


def load_runtime_components(
    *,
    model_ref: str,
    draft_ref: Optional[str],
):
    target_model, tokenizer, _ = load_target_bundle(model_ref, lazy=True)
    resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)
    if not resolved_draft_ref:
        return target_model, tokenizer, None, None
    try:
        draft_model, _ = load_draft_bundle(resolved_draft_ref, lazy=True)
    except Exception:
        return target_model, tokenizer, None, None
    return target_model, tokenizer, draft_model, resolved_draft_ref


def run_generate(
    *,
    model_ref: str,
    prompt: str,
    max_tokens: int,
    use_chat_template: bool,
    draft_ref: Optional[str],
) -> int:
    target_model, tokenizer, draft_model, _ = load_runtime_components(
        model_ref=model_ref,
        draft_ref=draft_ref,
    )
    stop_token_ids = get_stop_token_ids(tokenizer)
    if draft_model is None:
        stream = stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
        )
    else:
        stream = stream_dflash_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=prompt,
            max_new_tokens=max_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
        )

    summary: Optional[dict[str, Any]] = None
    for event in stream:
        if event.get("event") == "token":
            sys.stdout.write(decode_token(tokenizer, int(event["token_id"])))
            sys.stdout.flush()
        elif event.get("event") == "summary":
            summary = event

    if summary is None:
        return 1

    tps = generation_tps_from_summary(summary)
    acceptance_pct = float(summary.get("acceptance_ratio", 0.0)) * 100.0
    token_count = int(summary.get("generation_tokens", 0))
    sys.stderr.write(
        f"\n{token_count} tokens | {tps:.1f} tok/s | {acceptance_pct:.1f}% acceptance\n"
    )
    sys.stderr.flush()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with DFlash on MLX.")
    parser.add_argument("--model", required=True, help="Target model reference.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--draft", default=None, help="Optional draft model override.")
    args = parser.parse_args()
    raise SystemExit(
        run_generate(
            model_ref=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            use_chat_template=not args.no_chat_template,
            draft_ref=args.draft,
        )
    )


if __name__ == "__main__":
    main()
