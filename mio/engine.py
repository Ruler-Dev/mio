"""Unified DFlash + TurboQuant inference engine.

Uses the vendored mio.dflash runtime (the fast DFlash path that benchmarks
240+ tok/s on Qwen3.5-35B-A3B-4bit) for both PARO and standard quantized models.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    fallback_reason: str | None = None
    metrics_scope: str = "request"  # request, or batch for MLX aggregate timings
    batch_size: int = 1
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
        config_path = Path(tc.target_model) / "config.json"
        try:
            target_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            target_config = {}
        self._target_meta = {
            "paro": True,
            "target_family": target_family,
            "config": target_config,
        }
        print(f"  Target: PARO INT4 ({target_family})")

    def _load_draft(self, tc: TierConfig) -> None:
        """Load the DFlash draft model. No draft -> baseline AR fallback."""
        try:
            from mio.dflash.runtime import (
                bind_draft_target_model,
                load_draft_bundle,
                validate_draft_target_compatibility,
            )
            self._draft_model, draft_meta = load_draft_bundle(tc.draft_model)
        except Exception as e:
            print(f"  WARNING: Draft load failed ({e}), baseline AR fallback")
            self._draft_model = None
            return

        # A model that loads but targets a different architecture is not a safe
        # autoregressive fallback condition: surface the configuration error.
        validate_draft_target_compatibility(
            self._target_meta.get("config") or {},
            draft_meta.get("config") or {},
        )
        bind_draft_target_model(self._draft_model, self._target_model)
        print(f"  Draft loaded: {draft_meta.get('resolved_model_ref', tc.draft_model)}")

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
            # Runtime warm-start must always leave at least one uncached prompt
            # token so it can produce the first next-token logit.
            if match >= len(prompt_tokens):
                continue
            # Qwen hybrid targets combine attention KV caches with recurrent
            # GDN state.  The latter is not rewindable, so a divergent cached
            # tail may only be reused when *every* target cache can trim it.
            needs_rewind = match < len(cached_tokens)
            if needs_rewind and not self._warm_state_can_rewind(entry):
                continue
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
    def _warm_state_can_rewind(entry: dict) -> bool:
        target_cache = entry.get("target_cache")
        if not target_cache:
            return True
        try:
            from mlx_lm.models import cache as cache_mod

            return bool(cache_mod.can_trim_prompt_cache(target_cache))
        except (AttributeError, TypeError):
            return False

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
                keys = getattr(dc, "keys", None)
                values = getattr(dc, "values", None)
                positions = getattr(dc, "positions", None)

                # dflash-mlx caches keep a fixed sink plus a moving tail. Their
                # physical length is therefore unrelated to the absolute RoPE
                # offset. Filter by absolute positions so a divergent cached
                # tail can never leak into the next prompt.
                if keys is not None and values is not None and positions is not None:
                    position_values = [int(position) for position in positions.tolist()]
                    keep = sum(position < length for position in position_values)
                    if keep:
                        dc.keys = keys[:, :, :keep, :]
                        dc.values = values[:, :, :keep, :]
                        dc.positions = positions[:keep]
                    else:
                        dc.keys = None
                        dc.values = None
                        dc.positions = None
                elif keys is not None and values is not None:
                    # Compatibility with the old contiguous draft cache.
                    keep = min(int(keys.shape[2]), length)
                    dc.keys = keys[:, :, :keep, :]
                    dc.values = values[:, :, :keep, :]
                if hasattr(dc, "offset"):
                    dc.offset = length

        target_hidden = entry.get("target_hidden")
        if target_hidden is not None:
            cur = int(target_hidden.shape[1])
            if cur > length:
                entry["target_hidden"] = target_hidden[:, :length, :]

        draft_context = entry.get("draft_context")
        if draft_context is not None:
            cur = int(draft_context.shape[1])
            if cur > length:
                entry["draft_context"] = draft_context[:, :length, :]

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
                import time
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

    def _resolve_sampling(
        self,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        seed: int | None,
    ) -> tuple[float, float, int, int | None]:
        """Resolve and validate the sampling contract shared by every backend."""

        # ``None`` is Mio's fast/speculative default.  A positive value is an
        # explicit request for stochastic target-only sampling; it must never
        # silently disable DFlash merely because an older persisted tier used
        # the historical 0.6 recommendation.
        resolved_temperature = 0.0 if temperature is None else float(temperature)
        resolved_top_p = float(self.tier_config.top_p) if top_p is None else float(top_p)
        resolved_top_k = int(self.tier_config.top_k) if top_k is None else int(top_k)
        resolved_seed = None if seed is None else int(seed)
        if not 0.0 <= resolved_temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 < resolved_top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if resolved_top_k < 0:
            raise ValueError("top_k must be non-negative")
        if resolved_seed is not None and not 0 <= resolved_seed <= (2**63 - 1):
            raise ValueError("seed must be between 0 and 2^63-1")
        return resolved_temperature, resolved_top_p, resolved_top_k, resolved_seed

    @staticmethod
    def _make_sampler(
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
    ):
        """Build the canonical MLX-LM sampler and initialize its RNG stream."""

        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            mx.random.seed(seed)
        return make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

    @staticmethod
    def _truncate_at_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
        matches = [text.find(value) for value in (stop or []) if value and value in text]
        if not matches:
            return text, False
        return text[: min(matches)], True

    def _adjust_completion_metrics(self, metrics: GenerationMetrics, text: str) -> None:
        """Keep usage coherent when textual stop/tool trimming shortens output."""

        try:
            count = len(self._tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            count = len(self._tokenizer.encode(text))
        metrics.completion_tokens = count
        metrics.total_tokens = metrics.prompt_tokens + count

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
            fallback_reason=result.get("fallback_reason"),
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
        tool_required: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> tuple[str, GenerationMetrics]:
        if not self._loaded:
            raise RuntimeError("Engine not loaded.")

        configured_bmp_paths = int(getattr(self.tier_config, "bmp_paths", 1) or 1)
        bmp_nonstream = self._draft_model is not None and configured_bmp_paths >= 2
        if any(value for value in (stop or []) if value) and not bmp_nonstream:
            chunks: list[str] = []
            final_metrics: GenerationMetrics | None = None
            for chunk, metrics in self.generate_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                tools=tools,
                tool_required=tool_required,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            ):
                if chunk:
                    chunks.append(chunk)
                if metrics is not None:
                    final_metrics = metrics
            if final_metrics is None:
                raise RuntimeError("streaming stop path ended without generation metrics")
            return "".join(chunks), final_metrics

        max_tokens = self.tier_config.max_output_tokens if max_tokens is None else max_tokens
        temperature, top_p, top_k, seed = self._resolve_sampling(
            temperature, top_p, top_k, seed
        )
        sampling = temperature > 0.0
        sampler = self._make_sampler(temperature, top_p, top_k, seed)
        prompt_tokens = self._apply_chat_template(messages, tools=tools)
        stop_ids = self._eos_token_ids()
        # Non-streaming path keeps simpler behaviour: suppress EOS entirely
        # when tools are present and trim at the last </tool_call>. Streaming
        # path (used by Kilo) gets the smarter per-step relaxation.
        suppress_ids = list(stop_ids) if tools and tool_required else None

        tq_bits = self._resolved_tq_bits()
        pq_bits = self._resolved_pq_bits()
        bmp_paths = configured_bmp_paths
        ddtree_budget = self._resolved_ddtree_budget()

        if self._draft_model is not None and bmp_paths >= 2:
            # TQ4/PQ + BMP is untested; fall back to vanilla DFlash.
            if tq_bits is not None or pq_bits is not None:
                bmp_paths = 1

        if sampling:
            # The vendored DFlash and DDTree verifiers use greedy exact-match
            # acceptance.  Applying a stochastic sampler only to their bonus
            # token would bias the distribution, so stochastic requests take
            # Mio's optimized target-only baseline instead.
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
                sampler=sampler,
            )
            result["fallback_ar"] = True
            result["fallback_reason"] = "stochastic_sampling_requires_target_only"
        elif ddtree_budget > 0:
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
                return_final_state=self._prefix_cache_enabled(),
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
        trimmed = False
        if tools:
            end = text.rfind("</tool_call>")
            if end >= 0:
                text = text[: end + len("</tool_call>")]
                trimmed = True
        text, stopped = self._truncate_at_stop(text, stop)
        metrics = self._metrics_from_result(result)
        if trimmed or stopped:
            self._adjust_completion_metrics(metrics, text)
        self._last_metrics = metrics
        return text, metrics

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        *,
        max_tokens: int | list[int] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop: list[list[str] | None] | None = None,
        prefill_batch_size: int = 8,
        completion_batch_size: int = 32,
        prefill_step_size: int = 2048,
    ) -> list[tuple[str, GenerationMetrics]]:
        """Generate multiple independent sessions with MLX continuous batching.

        DFlash accelerates a single speculative stream.  For two or more
        unrelated prompts, MLX-LM's ``BatchGenerator`` instead keeps separate
        KV caches and continuously moves completed sequences out of the active
        batch.  This method exposes that throughput-oriented path while keeping
        the ordinary ``generate`` method latency-oriented.

        Sampling is intentionally uniform within one call; ``mio.batch``
        groups requests by the complete sampler configuration before invoking
        this method. Textual stop strings are applied independently per output.
        """

        if not self._loaded:
            raise RuntimeError("Engine not loaded.")
        if not messages_batch:
            return []
        if len(messages_batch) == 1:
            one_max = max_tokens[0] if isinstance(max_tokens, list) else max_tokens
            one_stop = stop[0] if stop else None
            return [
                self.generate(
                    messages_batch[0],
                    max_tokens=one_max,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    stop=one_stop,
                )
            ]

        if stop is None:
            stops: list[list[str] | None] = [None] * len(messages_batch)
        elif len(stop) != len(messages_batch):
            raise ValueError("stop list must match the number of prompts")
        else:
            stops = stop
        if max_tokens is None:
            resolved_max: int | list[int] = self.tier_config.max_output_tokens
        elif isinstance(max_tokens, list):
            if len(max_tokens) != len(messages_batch):
                raise ValueError("max_tokens list must match the number of prompts")
            if any(int(value) < 1 for value in max_tokens):
                raise ValueError("every max_tokens value must be positive")
            resolved_max = [min(int(value), 32768) for value in max_tokens]
        else:
            resolved_max = max(1, min(int(max_tokens), 32768))

        temperature, top_p, top_k, seed = self._resolve_sampling(
            temperature, top_p, top_k, seed
        )
        if seed is not None and temperature > 0.0:
            # MLX continuous batching owns one RNG stream for the whole active
            # batch, so a seeded request would otherwise change output when a
            # neighbour is inserted or reordered.  Preserve the public seed
            # contract by resetting the same seed for each independent request.
            return [
                self.generate(
                    messages,
                    max_tokens=(
                        resolved_max[index]
                        if isinstance(resolved_max, list)
                        else resolved_max
                    ),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    stop=stops[index],
                )
                for index, messages in enumerate(messages_batch)
            ]
        import mlx_lm

        prompts = [self._apply_chat_template(messages) for messages in messages_batch]
        sampler = self._make_sampler(temperature, top_p, top_k, seed)
        response = mlx_lm.batch_generate(
            self._target_model,
            self._tokenizer,
            prompts,
            max_tokens=resolved_max,
            sampler=sampler,
            prefill_batch_size=max(1, int(prefill_batch_size)),
            completion_batch_size=max(1, int(completion_batch_size)),
            prefill_step_size=max(1, int(prefill_step_size)),
        )
        stats = response.stats
        results: list[tuple[str, GenerationMetrics]] = []
        for prompt, text, request_stop in zip(prompts, response.texts, stops, strict=True):
            text, _ = self._truncate_at_stop(text, request_stop)
            completion_tokens = len(self._tokenizer.encode(text, add_special_tokens=False))
            metrics = GenerationMetrics(
                prompt_tokens=len(prompt),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt) + completion_tokens,
                prompt_tps=float(stats.prompt_tps),
                generation_tps=float(stats.generation_tps),
                end_to_end_tps=(
                    float(stats.generation_tokens)
                    / max(float(stats.prompt_time) + float(stats.generation_time), 1e-9)
                ),
                peak_memory_gb=float(stats.peak_memory),
                total_time_s=float(stats.prompt_time) + float(stats.generation_time),
                fallback_ar=True,
                fallback_reason="continuous_batch_uses_target_only",
                metrics_scope="batch",
                batch_size=len(messages_batch),
            )
            results.append((text, metrics))
        self._last_metrics = results[-1][1]
        # Batched target-only generation owns independent caches; old speculative
        # prefix snapshots cannot be reused safely after a different execution path.
        self._prefix_cache_invalidate()
        return results

    def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        tool_required: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> Generator[tuple[str, GenerationMetrics | None], None, None]:
        """Stream generation while withholding partial textual stop matches."""

        stops = [value for value in (stop or []) if value]
        stop_signal = threading.Event()
        raw = self._generate_stream_raw(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_required=tool_required,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_signal=stop_signal,
            decode_chunk_tokens=1 if stops else 8,
        )
        if not stops:
            yield from raw
            return

        buffer = ""
        emitted: list[str] = []
        stopped = False
        try:
            for chunk, metrics in raw:
                if metrics is not None:
                    if buffer and not stopped:
                        emitted.append(buffer)
                        yield buffer, None
                        buffer = ""
                    self._adjust_completion_metrics(metrics, "".join(emitted))
                    self._last_metrics = metrics
                    yield "", metrics
                    continue
                if not chunk or stopped:
                    continue

                buffer += chunk
                match_positions = [buffer.find(value) for value in stops if value in buffer]
                if match_positions:
                    before = buffer[: min(match_positions)]
                    if before:
                        emitted.append(before)
                        yield before, None
                    buffer = ""
                    stopped = True
                    stop_signal.set()
                    continue

                # Keep only a suffix that could still grow into a stop string.
                # Everything before it is safe to expose to the client.
                hold = 0
                for value in stops:
                    limit = min(len(value) - 1, len(buffer))
                    for size in range(limit, 0, -1):
                        if buffer.endswith(value[:size]):
                            hold = max(hold, size)
                            break
                safe = buffer[:-hold] if hold else buffer
                buffer = buffer[-hold:] if hold else ""
                if safe:
                    emitted.append(safe)
                    yield safe, None
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()

    def _new_streaming_detokenizer(self):
        """Create request-local, UTF-8-safe incremental tokenizer state."""

        try:
            detokenizer = self._tokenizer.detokenizer
        except (AttributeError, TypeError):
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

            detokenizer = NaiveStreamingDetokenizer(self._tokenizer)
        detokenizer.reset()
        return detokenizer

    def _generate_stream_raw(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tool_required: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop_signal: threading.Event | None = None,
        decode_chunk_tokens: int = 8,
    ) -> Generator[tuple[str, GenerationMetrics | None], None, None]:
        if not self._loaded:
            raise RuntimeError("Engine not loaded.")

        max_tokens = self.tier_config.max_output_tokens if max_tokens is None else max_tokens
        temperature, top_p, top_k, seed = self._resolve_sampling(
            temperature, top_p, top_k, seed
        )
        sampling = temperature > 0.0
        sampler = self._make_sampler(temperature, top_p, top_k, seed)
        prompt_tokens = self._apply_chat_template(messages, tools=tools)
        stop_ids = self._eos_token_ids()
        # Tool-calling path: suppress EOS for the first N tokens so the model
        # can't escape via "Now I'll update..." + immediate EOS. After N we
        # re-enable EOS so natural end-of-turn after a tool call terminates
        # cleanly and the draft model's EOS predictions are accepted (keeps
        # DFlash acceptance healthy).
        suppress_ids = list(stop_ids) if tools and tool_required else None
        relax_after = 40 if tools and tool_required else 0
        relax_ids = list(stop_ids) if tools and tool_required else None

        tq_bits = self._resolved_tq_bits()
        pq_bits = self._resolved_pq_bits()
        ddtree_budget = self._resolved_ddtree_budget()

        # Prefix cache lookup — same machinery as generate(). Skip when DDTree
        # is driving: tree_aware verify mutates caches beyond what a dict
        # snapshot models.
        warm_state = None
        if not sampling and ddtree_budget == 0 and self._prefix_cache_enabled():
            warm_state = self._prefix_cache_lookup(prompt_tokens)

        if sampling:
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
                sampler=sampler,
                fallback_reason="stochastic_sampling_requires_target_only",
            )
        elif ddtree_budget > 0:
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
        decode_chunk_tokens = max(1, int(decode_chunk_tokens))
        detokenizer = self._new_streaming_detokenizer()
        special_ids = {
            int(token_id)
            for token_id in (getattr(self._tokenizer, "all_special_ids", None) or [])
        }

        def flush_decode(*, final: bool = False) -> str:
            for token_id in decode_pending:
                if int(token_id) not in special_ids:
                    detokenizer.add_token(int(token_id))
            decode_pending.clear()
            if final:
                detokenizer.finalize()
            return str(detokenizer.last_segment)

        prefill_us = 0
        prefill_emitted = False
        generation_tokens = 0
        acceptance_ratio = 0.0
        cycles_completed = 0
        warm_offset = 0
        summary_emitted = False
        started = time.perf_counter()

        try:
            for event in stream:
                ev_type = event.get("event")

                if ev_type == "prefill":
                    prefill_us = event.get("prefill_us", 0)
                    warm_offset = int(event.get("warm_offset", 0) or 0)
                    if self._pending_assistant_prefill and not prefill_emitted:
                        yield self._pending_assistant_prefill, None
                        prefill_emitted = True
                        if stop_signal is not None and stop_signal.is_set():
                            break

                elif ev_type == "token":
                    generation_tokens = int(
                        event.get("generated_tokens", generation_tokens + 1)
                    )
                    acceptance_ratio = float(
                        event.get("acceptance_ratio", acceptance_ratio) or 0.0
                    )
                    cycles_completed = int(
                        event.get("cycles_completed", cycles_completed) or 0
                    )
                    decode_pending.append(event["token_id"])
                    if len(decode_pending) >= decode_chunk_tokens:
                        chunk = flush_decode()
                        if chunk:
                            yield chunk, None
                            if stop_signal is not None and stop_signal.is_set():
                                break

                elif ev_type == "summary":
                    generation_tokens = int(
                        event.get("generation_tokens", generation_tokens) or generation_tokens
                    )
                    acceptance_ratio = float(
                        event.get("acceptance_ratio", acceptance_ratio) or 0.0
                    )
                    cycles_completed = int(
                        event.get("cycles_completed", cycles_completed) or 0
                    )
                    chunk = flush_decode(final=True)
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
                    if not sampling and self._prefix_cache_enabled():
                        self._prefix_cache_store(prompt_tokens, event)
                    summary_emitted = True
                    yield "", metrics
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if stop_signal is not None and stop_signal.is_set() and not summary_emitted:
            self._pending_assistant_prefill = ""
            elapsed_us = (time.perf_counter() - started) * 1e6
            stopped_result = {
                "prompt_token_count": len(prompt_tokens),
                "generation_tokens": generation_tokens,
                "prefill_us": prefill_us,
                "elapsed_us": elapsed_us,
                "acceptance_ratio": acceptance_ratio,
                "cycles_completed": cycles_completed,
                "warm_offset": warm_offset,
                "fallback_ar": sampling,
                "fallback_reason": (
                    "stochastic_sampling_requires_target_only" if sampling else None
                ),
                "stopped_early": True,
            }
            metrics = self._metrics_from_result(stopped_result)
            self._last_metrics = metrics
            yield "", metrics
