"""Unified DFlash + TurboQuant inference engine.

Uses the vendored mio.dflash runtime (the fast DFlash path that benchmarks
240+ tok/s on Qwen3.5-35B-A3B-4bit) for both PARO and standard quantized models.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from mio.config import TierConfig


@dataclass
class GenerationMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    end_to_end_tps: float = 0.0
    avg_acceptance_length: float = 0.0
    acceptance_ratio: float = 0.0
    peak_memory_gb: float = 0.0
    total_time_s: float = 0.0
    cycles: int = 0
    fallback_ar: bool = False
    warm_offset: int = 0        # tokens skipped via prefix cache (0 = cold/miss)
    cache_entries: int = 0      # prefix-cache size at the time of this call


@dataclass
class MioEngine:
    """DFlash-first engine using the vendored mio.dflash fast runtime."""

    tier_config: TierConfig
    _target_model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)
    _draft_model: Any = field(default=None, repr=False)
    _target_meta: dict = field(default_factory=dict, repr=False)
    _loaded: bool = False
    _last_metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    # Prefix cache: maps token-tuple → (cached_state, offset). Populated automatically
    # after each successful generate() and consulted before the next one to skip
    # prefill of shared prompt prefixes. See docs/09-prefix-cache.md.
    _prefix_cache: dict = field(default_factory=dict, repr=False)
    _prefix_cache_max_entries: int = field(default=2, repr=False)
    _prefix_cache_min_tokens: int = field(default=64, repr=False)
    _prefix_cache_margin: int = field(default=32, repr=False)
    # Approximate token-budget cap across all live entries. A 100K-token entry
    # on Qwen3.5-35B-A3B is ~25 GB of KV; four such entries thrash Metal on a
    # 48 GB M4 Max. We evict by age until total_tokens <= budget. Default sized
    # for large-moe at 128K context; tiers override at load time.
    _prefix_cache_token_budget: int = field(default=200_000, repr=False)
    _last_prompt_tokens: list = field(default_factory=list, repr=False)
    # Tool-call prefill: set to "<" when the last chat-template call detected a
    # Cline/Kilo/Roo system prompt. The generate() methods prepend this to
    # their output so the client sees the full XML tag.
    _pending_assistant_prefill: str = field(default="", repr=False)

    def load(self) -> None:
        if self._loaded:
            return

        tc = self.tier_config
        # Size the prefix-cache token budget and entry count to match VRAM
        # headroom. KV per token roughly: 64 KB (4B), 96 KB (9B), 196 KB
        # (27B dense), 256 KB (35B-A3B MoE). On a 48 GB M4 Max with a 17 GB
        # model resident, we have ~25 GB usable for cache. A single 128K
        # entry on large-moe already eats 32 GB of that — two would OOM and
        # Metal starts killing command buffers as "innocent victims".
        ctx = tc.context_window
        if ctx >= 65536:        # 64K+ (large-moe)
            self._prefix_cache_max_entries = 1
            self._prefix_cache_token_budget = ctx  # one full-context entry
        elif ctx >= 16384:      # 16-32K (medium, large dense)
            self._prefix_cache_max_entries = 2
            self._prefix_cache_token_budget = ctx * 2
        else:                   # <=8K (small)
            self._prefix_cache_max_entries = 4
            self._prefix_cache_token_budget = ctx * 4
        print(f"Loading {tc.name} tier: {tc.target_model}")
        print(f"  Draft: {tc.draft_model}")
        print(
            f"  Prefix cache: max {self._prefix_cache_max_entries} entries, "
            f"{self._prefix_cache_token_budget:,}-token budget",
        )

        if self._detect_paro(tc.target_model):
            self._load_target_paro(tc)
        else:
            self._load_target_standard(tc)

        self._load_draft(tc)

        self._loaded = True
        print("  Loaded successfully.")

    def _load_target_standard(self, tc: TierConfig) -> None:
        """Load a standard (non-PARO) MLX model via the vendored DFlash runtime."""
        from mio.dflash.runtime import load_target_bundle

        self._target_model, self._tokenizer, self._target_meta = load_target_bundle(
            tc.target_model,
            lazy=True,
            split_full_attention_sdpa=True,
            quantize_kv_cache=False,
        )
        print(f"  Target: standard MLX ({self._target_meta.get('target_family', 'unknown')})")

    def _load_target_paro(self, tc: TierConfig) -> None:
        """Load a PARO-quantized model and wire in DFlash hooks."""
        from mio.dflash.runtime import (
            configure_full_attention_split,
            detect_target_family,
            _install_target_speculative_hooks,
        )
        from mio.paroquant.load import load as paro_load

        self._target_model, processor, _is_vlm = paro_load(tc.target_model, force_text=True)
        self._tokenizer = getattr(processor, "tokenizer", processor)
        target_family = detect_target_family(self._target_model)
        if target_family == "hybrid_gdn":
            _install_target_speculative_hooks(self._target_model)
            configure_full_attention_split(self._target_model, enabled=True, chunk_size=8)
        self._target_meta = {"paro": True, "target_family": target_family}
        print(f"  Target: PARO INT4 ({target_family})")

    def _load_draft(self, tc: TierConfig) -> None:
        """Load the DFlash draft model. No draft -> baseline AR fallback."""
        try:
            from mio.dflash.runtime import load_draft_bundle
            self._draft_model, draft_meta = load_draft_bundle(tc.draft_model)
            print(f"  Draft loaded: {draft_meta.get('resolved_model_ref', tc.draft_model)}")
        except Exception as e:
            print(f"  WARNING: Draft load failed ({e}), baseline AR fallback")
            self._draft_model = None

    @staticmethod
    def _detect_paro(model_path: str) -> bool:
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text())
            return config.get("quantization_config", {}).get("quant_method") == "paroquant"
        except Exception:
            return False

    def unload(self) -> None:
        if not self._loaded:
            return
        self._target_model = None
        self._tokenizer = None
        self._draft_model = None
        self._target_meta = {}
        self._loaded = False
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def last_metrics(self) -> GenerationMetrics:
        return self._last_metrics

    def _resolved_tq_bits(self) -> int | None:
        """Return {2, 3, 4} if TurboQuant is enabled for this tier; else None."""
        bits = getattr(self.tier_config, "tq_bits", 16)
        return bits if bits in (2, 3, 4) else None

    def _resolved_pq_bits(self) -> int | None:
        """Return {2, 3, 4} if PolarQuant is enabled for this tier; else None."""
        bits = getattr(self.tier_config, "pq_bits", 16)
        return bits if bits in (2, 3, 4) else None

    def _resolved_ddtree_budget(self) -> int:
        """Return DDTree tree-node budget or 0 (off).

        DDTree only runs when: a draft model is loaded, target family is
        hybrid_gdn, BMP is off, and the tier (or MIO_DDTREE_BUDGET env) sets a
        positive budget. The engine forces DDTREE_EXACT_COMMIT=1 before calling
        the runtime so the commit path is compatible with QuantizedKVCache.
        """
        import os as _os
        if self._draft_model is None:
            return 0
        if self._target_meta.get("target_family") != "hybrid_gdn":
            return 0
        if int(getattr(self.tier_config, "bmp_paths", 1) or 1) >= 2:
            return 0
        env = _os.environ.get("MIO_DDTREE_BUDGET")
        if env:
            try:
                return max(0, int(env))
            except ValueError:
                pass
        return max(0, int(getattr(self.tier_config, "ddtree_budget", 0) or 0))

    @staticmethod
    def _prepare_ddtree_env() -> None:
        """Force exact-commit mode so DDTree works with any KV cache type.

        tree_aware_path_commit writes per-position into cache.keys — fine for
        plain KVCache, broken for QuantizedKVCache (keys is a 3-tuple of
        quantized arrays). EXACT_COMMIT skips that path and uses a sequential
        forward instead, which update_and_fetch handles correctly for both.
        """
        import os as _os
        _os.environ.setdefault("DDTREE_EXACT_COMMIT", "1")

    # --- Prefix cache ---
    #
    # The cache is populated opportunistically: after each generate(), we look
    # at the previous prompt and the current one and store a warm-cache entry
    # for their longest common prefix. The first generate() is always a miss.
    # The cache is bypassed when TurboQuant, PolarQuant, or BMP is active
    # (their cache states include pre-allocated buffers and tapes that don't
    # snapshot cleanly).

    def _prefix_cache_enabled(self) -> bool:
        # Quantized caches are pre-allocated and not safe to freeze via simple dict.
        # BMP expects fresh caches per call (batch-expand/filter logic).
        if self._resolved_tq_bits() is not None:
            return False
        if self._resolved_pq_bits() is not None:
            return False
        if int(getattr(self.tier_config, "bmp_paths", 1) or 1) >= 2:
            return False
        return True

    def _longest_common_prefix(self, a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _prefix_cache_lookup(self, prompt_tokens: list[int]) -> dict | None:
        """Return a warm state truncated to the longest matching prefix.

        Unlike a strict "cached_tokens is a prefix of prompt_tokens" check, this
        finds the longest common prefix between any cached entry and the new
        prompt. If that prefix is long enough (>= min_tokens), we rent the
        entry out (remove it from the map), truncate its KV cache + hidden
        states to the match length, and return it for warm-start prefill.

        Qwen3.5's chat template only wraps the CURRENT assistant turn in
        `<think>\\n\\n</think>\\n\\n`, so prior-turn cache keys always diverge
        from the next turn's rendered prompt at that 10-token wrapper. Strict
        prefix-match would miss every cross-turn lookup; this relaxed version
        recovers 99% of the tokens (everything up to the divergence point).
        """
        best_key: tuple | None = None
        best_entry: dict | None = None
        best_match = 0
        for cached_tokens_tuple, entry in self._prefix_cache.items():
            cached_tokens = list(cached_tokens_tuple)
            match = self._longest_common_prefix(cached_tokens, prompt_tokens)
            if match > best_match and match >= self._prefix_cache_min_tokens:
                best_key = cached_tokens_tuple
                best_entry = entry
                best_match = match
        if best_entry is None or best_key is None:
            return None
        # Rent: remove from map so concurrent/future requests don't see the
        # about-to-be-mutated state. The new post-generation state is re-stored
        # under its own key at the end of this request.
        del self._prefix_cache[best_key]
        entry = dict(best_entry)
        entry["offset"] = best_match
        # Apply truncation to the KV structures so runtime's warm-start begins
        # writing at position best_match.
        self._truncate_warm_state(entry, best_match)
        return entry

    @staticmethod
    def _truncate_warm_state(entry: dict, length: int) -> None:
        """Truncate target_cache / draft_cache / target_hidden to `length` positions.

        Safe to call when length == current offset (no-op). Length must be
        <= current offset — we never grow caches here.
        """
        from mio.dflash.runtime import trim_cache_to

        target_cache = entry.get("target_cache")
        if target_cache is not None:
            trim_cache_to(target_cache, length)

        draft_cache = entry.get("draft_cache")
        if draft_cache is not None:
            for dc in draft_cache:
                if dc.keys is None or dc.values is None:
                    continue
                cur = int(dc.keys.shape[2])
                if cur > length:
                    dc.keys = dc.keys[:, :, :length, :]
                    dc.values = dc.values[:, :, :length, :]
                    dc.offset = length

        target_hidden = entry.get("target_hidden")
        if target_hidden is not None:
            cur = int(target_hidden.shape[1])
            if cur > length:
                entry["target_hidden"] = target_hidden[:, :length, :]

    def _prefix_cache_store(self, prompt_tokens: list[int], summary_event: dict | None) -> None:
        """Store the current request's post-generation cache state directly.

        No extra prefill pass, no warm-up. The key is `prompt_tokens +
        generated_tokens` — the full sequence the cache has just processed.
        A subsequent request that starts with this full sequence (chat-style
        extension) will hit and skip re-prefill of those tokens.

        Memory: each entry holds ~KV_bytes(prompt+gen). At 80K tokens on the
        large-moe tier that's multiple GB; `_prefix_cache_max_entries` bounds
        the total.
        """
        self._last_prompt_tokens = list(prompt_tokens)
        if summary_event is None:
            return
        final = summary_event.get("final_state")
        if not isinstance(final, dict):
            print("[prefix-cache] store skipped: no final_state in summary", flush=True)
            return
        if final.get("target_cache") is None or final.get("draft_cache") is None:
            print("[prefix-cache] store skipped: final_state missing caches", flush=True)
            return

        gen_ids = list(summary_event.get("generated_token_ids") or [])
        key_tokens = list(prompt_tokens) + gen_ids
        if len(key_tokens) < self._prefix_cache_min_tokens:
            return
        key = tuple(key_tokens)
        if key in self._prefix_cache:
            return

        # LRU evict by BOTH entry count AND cumulative token budget. The
        # token-budget evict protects Metal from OOM thrash on long-context
        # agent sessions — each rented-out entry holds ~256 KB/token of KV
        # on 35B-A3B, so 4 entries × 100K tokens ≈ 100 GB on a 48 GB machine.
        def _total_tokens() -> int:
            return sum(len(k) for k in self._prefix_cache.keys())

        while (
            len(self._prefix_cache) >= self._prefix_cache_max_entries
            or _total_tokens() + len(key_tokens) > self._prefix_cache_token_budget
        ):
            if not self._prefix_cache:
                break
            oldest = next(iter(self._prefix_cache))
            del self._prefix_cache[oldest]

        # Overwrite offset to match full key length — the cache is at this
        # position after prefill + decode.
        final = dict(final)
        final["offset"] = len(key_tokens)
        self._prefix_cache[key] = final
        print(
            f"[prefix-cache] STORED key_len={len(key_tokens)} "
            f"(entries={len(self._prefix_cache)})",
            flush=True,
        )

    def _prefix_cache_invalidate(self) -> None:
        """Drop all cached entries. Needed when the model's caches are externally mutated."""
        self._prefix_cache.clear()
        self._last_prompt_tokens = []

    def _apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> list[int]:
        # Force enable_thinking=False for Qwen3.5 — otherwise the model
        # prepends <think>...</think> reasoning before any tool call.
        # `tools` is the OpenAI-style tools array (Kilo sends 11 functions
        # in this field). Qwen3.5's chat template has native handling:
        # it injects a '# Tools\n<tools>[...]</tools>' system prefix and
        # instructs the model to respond in its native
        # <tool_call><function=X><parameter=Y>val</parameter></function></tool_call>
        # format. We parse that back to OpenAI tool_calls JSON server-side.
        tokens: list[int] | None = None
        text: str | None = None
        tmpl_kwargs_tries = []
        base = {"add_generation_prompt": True, "enable_thinking": False}
        if tools:
            tmpl_kwargs_tries.append({**base, "tools": tools})
            tmpl_kwargs_tries.append({"add_generation_prompt": True, "tools": tools})
        tmpl_kwargs_tries.append(base)
        tmpl_kwargs_tries.append({"add_generation_prompt": True})

        if hasattr(self._tokenizer, "apply_chat_template"):
            # Some chat templates can't render multimodal content-list
            # messages (text + image_url parts) — collapse them to plain
            # text as a fallback shape before trying the template.
            def _flatten(msgs):
                out = []
                for m in msgs:
                    c = m.get("content")
                    if isinstance(c, list):
                        bits = []
                        for part in c:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    bits.append(str(part.get("text") or ""))
                                elif part.get("type") == "image_url":
                                    bits.append("[image attached]")
                        out.append({**m, "content": "\n".join(bits) or ""})
                    else:
                        out.append(m)
                return out

            candidates = [messages, _flatten(messages)]
            for msg_set in candidates:
                for kw in tmpl_kwargs_tries:
                    try:
                        text = self._tokenizer.apply_chat_template(
                            msg_set, tokenize=False, **kw,
                        )
                        break
                    except Exception:
                        text = None
                if text is not None:
                    break
        if text is None:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            text = "\n".join(parts)

        self._pending_assistant_prefill = ""
        tokens = list(self._tokenizer.encode(text, add_special_tokens=False))

        # Optional debug logging — enable with MIO_DEBUG_LOG=1.
        import os
        if os.environ.get("MIO_DEBUG_LOG", "") in ("1", "true", "yes"):
            try:
                path = os.environ.get("MIO_DEBUG_LOG_PATH", "/tmp/mio-serve-debug.log")
                import json, time
                with open(path, "a") as f:
                    f.write(json.dumps({
                        "ts": time.time(), "event": "chat_template",
                        "token_count": len(tokens),
                        "prompt_text_head": (text or "")[:800],
                        "prompt_text_tail": (text or "")[-800:],
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return tokens

    def _eos_token_ids(self) -> list[int]:
        eos_ids = list(getattr(self._tokenizer, "eos_token_ids", None) or [])
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_id is not None and eos_id not in eos_ids:
            eos_ids.append(int(eos_id))
        return eos_ids

    def _metrics_from_result(self, result: dict[str, Any]) -> GenerationMetrics:
        gen_ids = result.get("generated_token_ids", [])
        elapsed_us = result.get("elapsed_us", 0)
        prefill_us = result.get("prefill_us")
        if prefill_us is None:
            phase = result.get("phase_timings_us") or {}
            prefill_us = phase.get("prefill", 0)
        gen_tokens = result.get("generation_tokens", len(gen_ids))
        prompt_tok_count = result.get("prompt_token_count", 0)
        decode_us = max(0, elapsed_us - prefill_us)

        return GenerationMetrics(
            prompt_tokens=prompt_tok_count,
            completion_tokens=gen_tokens,
            total_tokens=prompt_tok_count + gen_tokens,
            prompt_tps=prompt_tok_count / max(prefill_us / 1e6, 1e-9) if prefill_us > 0 else 0,
            generation_tps=gen_tokens / max(decode_us / 1e6, 1e-9) if decode_us > 0 else 0,
            end_to_end_tps=gen_tokens / max(elapsed_us / 1e6, 1e-9) if elapsed_us > 0 else 0,
            acceptance_ratio=result.get("acceptance_ratio", 0),
            avg_acceptance_length=(
                result.get("tokens_per_cycle", 0)
                or result.get("avg_acceptance", 0)  # DDTree's key name
                or result.get("acceptance_ratio", 0) * 16
            ),
            peak_memory_gb=result.get("peak_memory_gb", 0) or 0,
            total_time_s=elapsed_us / 1e6,
            cycles=result.get("cycles_completed", 0),
            fallback_ar=result.get("fallback_ar", False),
            warm_offset=int(result.get("warm_offset", 0) or 0),
            cache_entries=len(self._prefix_cache),
        )

    def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str, GenerationMetrics]:
        if not self._loaded:
            raise RuntimeError("Engine not loaded.")

        max_tokens = max_tokens or self.tier_config.max_output_tokens
        prompt_tokens = self._apply_chat_template(messages, tools=tools)
        stop_ids = self._eos_token_ids()
        # Non-streaming path keeps simpler behaviour: suppress EOS entirely
        # when tools are present and trim at the last </tool_call>. Streaming
        # path (used by Kilo) gets the smarter per-step relaxation.
        suppress_ids = list(stop_ids) if tools else None

        tq_bits = self._resolved_tq_bits()
        pq_bits = self._resolved_pq_bits()
        bmp_paths = int(getattr(self.tier_config, "bmp_paths", 1) or 1)
        ddtree_budget = self._resolved_ddtree_budget()

        if self._draft_model is not None and bmp_paths >= 2:
            # TQ4/PQ + BMP is untested; fall back to vanilla DFlash.
            if tq_bits is not None or pq_bits is not None:
                bmp_paths = 1

        if ddtree_budget > 0:
            # DDTree owns its own cache policy: PQ/TQ off, 8-bit KV on for some
            # compression, sequential-forward commit for quantized compatibility.
            self._prepare_ddtree_env()
            from mio.ddtree.runtime import generate_ddtree_once
            result = generate_ddtree_once(
                target_model=self._target_model,
                draft_model=self._draft_model,
                tokenizer=self._tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                tree_budget=ddtree_budget,
                stop_token_ids=stop_ids,
                suppress_token_ids=suppress_ids,
                quantize_kv_cache=True,
            )
            result.setdefault("prompt_token_count", len(prompt_tokens))
        elif self._draft_model is not None and bmp_paths >= 2:
            from mio.dflash.bmp_runtime import generate_bmp_dflash_once
            result = generate_bmp_dflash_once(
                target_model=self._target_model,
                tokenizer=self._tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_tokens,
                num_paths=bmp_paths,
            )
        elif self._draft_model is not None:
            from mio.dflash.runtime import generate_dflash_once
            warm_state = None
            if self._prefix_cache_enabled():
                warm_state = self._prefix_cache_lookup(prompt_tokens)
            result = generate_dflash_once(
                target_model=self._target_model,
                tokenizer=self._tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                suppress_token_ids=suppress_ids,
                prompt_tokens_override=prompt_tokens,
                quantize_kv_cache=False,
                tq_bits=tq_bits,
                pq_bits=pq_bits,
                warm_state=warm_state,
            )
            if self._prefix_cache_enabled():
                self._prefix_cache_store(prompt_tokens, result)
        else:
            from mio.dflash.runtime import generate_baseline_once
            result = generate_baseline_once(
                target_model=self._target_model,
                tokenizer=self._tokenizer,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                suppress_token_ids=suppress_ids,
                prompt_tokens_override=prompt_tokens,
                quantize_kv_cache=False,
                tq_bits=tq_bits,
                pq_bits=pq_bits,
            )

        gen_ids = result.get("generated_token_ids", [])
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        if self._pending_assistant_prefill:
            text = self._pending_assistant_prefill + text
            self._pending_assistant_prefill = ""
        # With EOS suppressed we may overshoot past </tool_call>. Trim to the
        # end of the last complete tool call block.
        if tools:
            end = text.rfind("</tool_call>")
            if end >= 0:
                text = text[: end + len("</tool_call>")]
        metrics = self._metrics_from_result(result)
        self._last_metrics = metrics
        return text, metrics

    def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
    ) -> Generator[tuple[str, GenerationMetrics | None], None, None]:
        if not self._loaded:
            raise RuntimeError("Engine not loaded.")

        max_tokens = max_tokens or self.tier_config.max_output_tokens
        prompt_tokens = self._apply_chat_template(messages, tools=tools)
        stop_ids = self._eos_token_ids()
        # Tool-calling path: suppress EOS for the first N tokens so the model
        # can't escape via "Now I'll update..." + immediate EOS. After N we
        # re-enable EOS so natural end-of-turn after a tool call terminates
        # cleanly and the draft model's EOS predictions are accepted (keeps
        # DFlash acceptance healthy).
        suppress_ids = list(stop_ids) if tools else None
        relax_after = 40 if tools else 0
        relax_ids = list(stop_ids) if tools else None

        tq_bits = self._resolved_tq_bits()
        pq_bits = self._resolved_pq_bits()
        ddtree_budget = self._resolved_ddtree_budget()

        # Prefix cache lookup — same machinery as generate(). Skip when DDTree
        # is driving: tree_aware verify mutates caches beyond what a dict
        # snapshot models.
        warm_state = None
        if ddtree_budget == 0 and self._prefix_cache_enabled():
            warm_state = self._prefix_cache_lookup(prompt_tokens)

        if ddtree_budget > 0:
            self._prepare_ddtree_env()
            from mio.ddtree.runtime import stream_ddtree_generate
            stream = stream_ddtree_generate(
                target_model=self._target_model,
                draft_model=self._draft_model,
                tokenizer=self._tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                tree_budget=ddtree_budget,
                stop_token_ids=stop_ids,
                suppress_token_ids=suppress_ids,
                quantize_kv_cache=True,
            )
        elif self._draft_model is not None:
            from mio.dflash.runtime import stream_dflash_generate
            stream = stream_dflash_generate(
                target_model=self._target_model,
                tokenizer=self._tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                suppress_token_ids=suppress_ids,
                relax_suppress_after=relax_after,
                relax_suppress_token_ids=relax_ids,
                prompt_tokens_override=prompt_tokens,
                quantize_kv_cache=False,
                tq_bits=tq_bits,
                pq_bits=pq_bits,
                warm_state=warm_state,
            )
        else:
            from mio.dflash.runtime import stream_baseline_generate
            stream = stream_baseline_generate(
                target_model=self._target_model,
                tokenizer=self._tokenizer,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                suppress_token_ids=suppress_ids,
                prompt_tokens_override=prompt_tokens,
                quantize_kv_cache=False,
                tq_bits=tq_bits,
                pq_bits=pq_bits,
            )

        decode_pending: list[int] = []
        prefill_us = 0
        prefill_emitted = False

        for event in stream:
            ev_type = event.get("event")

            if ev_type == "prefill":
                prefill_us = event.get("prefill_us", 0)
                if self._pending_assistant_prefill and not prefill_emitted:
                    yield self._pending_assistant_prefill, None
                    prefill_emitted = True

            elif ev_type == "token":
                decode_pending.append(event["token_id"])
                if len(decode_pending) >= 8:
                    chunk = self._tokenizer.decode(decode_pending, skip_special_tokens=True)
                    decode_pending.clear()
                    if chunk:
                        yield chunk, None

            elif ev_type == "summary":
                if decode_pending:
                    chunk = self._tokenizer.decode(decode_pending, skip_special_tokens=True)
                    decode_pending.clear()
                    if chunk:
                        yield chunk, None
                # Belt-and-braces: emit prefill if no prefill event fired.
                if self._pending_assistant_prefill and not prefill_emitted:
                    yield self._pending_assistant_prefill, None
                    prefill_emitted = True
                self._pending_assistant_prefill = ""
                event.setdefault("prefill_us", prefill_us)
                metrics = self._metrics_from_result(event)
                self._last_metrics = metrics
                if self._prefix_cache_enabled():
                    self._prefix_cache_store(prompt_tokens, event)
                yield "", metrics
