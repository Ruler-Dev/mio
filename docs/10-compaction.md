# Context Auto-Compaction

Mio v0.4 introduces automatic context compaction: when an incoming prompt
approaches the model's context window, mio reduces it in place before
prefill. This prevents Metal OOM crashes on long agent sessions and keeps
attention quality high by avoiding near-full contexts where Qwen3.5 starts to
drift.

Implementation: `mio/compactor.py`, wired into `server.chat_completions`
before `_apply_caveman`.

## Trigger

Compaction fires when `len(prompt_tokens) > compact_threshold * context_window`.
Reduces to roughly `compact_target * context_window`. Defaults:

```
--compact-threshold 0.75   # fire at 75% of context window
--compact-target 0.50      # reduce to ~50%
--no-compact-summarize     # disable stage 2 (heuristic only)
```

Disable entirely with `--compact-threshold 1.0`.

## Two stages

### Stage 1 — tool-result truncation (cheap, no LLM)

Walks messages oldest → newest, excluding a **protected tail** of the last 4
atomic groups. For each `role=tool` message with content larger than 500
characters, replaces the content with a short placeholder:

```
[compacted: tool=read call_id=call_abc123, original output was 11612 chars — elided to save context]
```

In a typical Kilo session this reclaims 80-95% of prompt tokens in under 5ms.
No LLM round-trip.

### Stage 2 — LLM summarization (only if stage 1 is insufficient)

When stage 1 can't reach the target (e.g. the old turns are mostly prose or
tool arguments rather than huge tool results), stage 2 fires. It takes the
"middle" of the conversation — everything except the system prompt and the
last 4 atomic groups — and sends it to the current engine with:

```
You are a conversation summarizer. Produce a concise plain-text summary
(max 300 words) of the prior agent conversation. Preserve:
(1) what files/commands were read or executed,
(2) key facts discovered about the codebase,
(3) decisions made,
(4) any open questions. Do not invent. Output ONLY the summary.
```

The middle is replaced with a single synthetic `user` message containing the
summary. Stage 2 takes 1-2 s on large-moe; the summarization call runs under
the same GPU lock that gates the request's main generate, so there's no
concurrency risk.

## Atomic-group invariant

The compactor walks messages as **atomic groups**:

- Any system / user / standalone-assistant message → one group.
- `assistant (with tool_calls) + all contiguous role=tool responses` → one group.

A group is kept or dropped as a unit. This preserves the chat-template
requirement that every `tool_call_id` in an assistant message has a matching
`role=tool` message — splitting them would make Qwen's template raise and
break Kilo's client-side state.

## Interaction with the prefix cache

Compaction changes the prompt tokens past the truncation / summary point, so
the prefix cache misses for the compacted span on that turn. Subsequent turns
that start with the same post-compaction prefix then rebuild the cache
normally — effectively one miss every K turns where K is the compaction
frequency (usually tens of turns for a Kilo session).

This is strictly better than the alternative: without compaction, long agent
sessions either OOM on Metal ("innocent victim" command-buffer discards) or
run with degraded attention quality on near-full contexts.

## Observability

The live serve panel shows cumulative compactions and tokens reclaimed:

```
compactions    3 (42,100 tok reclaimed)
```

Per-compaction detail goes to stdout and the debug log
(`MIO_DEBUG_LOG=1` → `/tmp/mio-serve-debug.log`):

```
[compact] stage=truncation 78,432 → 42,100 tokens (36,332 saved, 7 tool results truncated, 0 turns summarized)
```

Debug-log `compact` events carry full `CompactStats`:

```json
{"event": "compact", "triggered": true, "stage": "truncation",
 "before_tokens": 78432, "after_tokens": 42100, "tokens_saved": 36332,
 "tool_results_truncated": 7, "messages_summarized": 0}
```

## Limits and future work

- The last user message is always in the protected tail — if a user pastes a
  single message larger than the context window, compaction cannot help. The
  server proceeds with the oversized prompt and mlx-lm's own chunked prefill
  will either finish or fail.
- Stage 2 summarization runs on the current tier. For tandem deployments we
  could route summarization to the medium (9B) tier for lower latency;
  currently not wired.
- The summarizer uses the live engine's greedy decoding. A small amount of
  deterministic content hashing to cache summaries across sessions would let
  us skip stage 2 entirely on exact prefix repeats — future work.
