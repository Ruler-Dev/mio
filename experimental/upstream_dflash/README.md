# Upstream DFlash fast-path R&D

This directory is an isolated prototype. It is not imported by `mio.engine`, does not alter production model loading, and must not be enabled by default until every compatibility and parity gate passes.

## Result

Two independent effects explain the current Qwen3.6-27B behavior:

1. Mio's vendored verifier pays for timewise-exact execution across quantized linears, attention, MLP, GDN and the head. Upstream `dflash_mlx 0.1.8` verifies a candidate block in batch and rolls recurrent/KV state back after acceptance.
2. Upstream's cold speculative prefill still uses a snapshot-oriented `prompt_len - 1` pass followed by a singleton seam, even when no snapshot exists. The target-only baseline uses one full-prompt pass.

The second observation produced a promising experimental candidate: use the existing full-tail path for cold/no-snapshot requests and defer only the draft-context projection until the first draft-context read. On the local Qwen3.6-27B model, the 12-pair pilot measured:

| Metric | Result | 95% prompt-cluster bootstrap CI |
| --- | ---: | ---: |
| Token parity | 12/12 (100%) | descriptive |
| TTFT speedup | 1.1555x | 1.1122x–1.1614x |
| End-to-end speedup | 1.0794x | 1.0657x–1.0938x |

This is a short, single-model, single-machine pilot—not a production or global breakthrough claim. The machine-readable artifact is [`results/qwen36-27b-fused-cold-prefill-20260715.json`](results/qwen36-27b-fused-cold-prefill-20260715.json).

## Exact verifier difference

| Stage | Mio vendored `mio.dflash` | Upstream `dflash_mlx 0.1.8` |
| --- | --- | --- |
| Target verify | `_verify_target_block` enters a timewise-exact component context | `TargetOps.verify_block` executes the full candidate block in batch |
| Quantized linear | Custom exact QMV when eligible, otherwise concatenated singleton calls | Normal batched quantized matmul; eligible M=16/M=4 shapes use upstream verify QMM kernels |
| Full attention | Exact verification splits SDPA query positions at chunk size 1 | Normal attention, with a long-prefix short-query GQA hook where applicable |
| MLP/head | Exact per-token projection path | Batched module/head path |
| Qwen GDN state | Innovation tape plus exact component execution | Innovation tape with batched projections and rollback |
| Cache commit | `timewise_exact_tape` | `TargetOps.restore_after_acceptance` rollback/trim |

Existing artifacts provide the following provenance-scoped evidence:

| Artifact | Decode | E2E | Parity | Notes |
| --- | ---: | ---: | ---: | --- |
| `speculative-matched-qwen36-27b-20260715.json` | 2.3728x | 2.0029x | 12/12 | Upstream, four prompts × three repetitions; TTFT is 0.9098x |
| `qwen36-20260715-193332.json` | 0.6207x | 0.9424x | 1/1 | Vendored, one prompt/repetition, `timewise_exact_tape` |

The upstream median target verification cost is 3,745.7 µs/cycle across 12 runs. The vendored artifact reports 337,626.0 µs/cycle, a 90.14x diagnostic gap. The source workloads and schedules differ, so this ratio identifies the implementation bottleneck but is not a paired speedup claim.

Reproduce the read-only extraction:

```bash
python3 experimental/upstream_dflash/comparison.py \
  benchmarks/results/speculative-matched-qwen36-27b-20260715.json \
  benchmarks/results/qwen36-20260715-193332.json
```

## TTFT experiment

Upstream already emits `state.staged_first` before starting the first speculative cycle. Therefore, "early token emission" did not require inventing a new token event. The avoidable work before that event was:

```text
cold prompt
  -> target forward(prompt_len - 1)
  -> synchronize / cache boundary
  -> target forward(singleton seam)
  -> concatenate selected target layers
  -> evaluate draft context projection
  -> PrefillCompleteEvent
  -> TokenEvent(first greedy target token)
```

The experimental cold/no-snapshot path is:

```text
cold prompt
  -> one target forward(full prompt)
  -> preserve selected hidden values
  -> PrefillCompleteEvent
  -> TokenEvent(first greedy target token)
  -> evaluate the same draft context projection on first drafter access
```

The projection-only ablation was parity-safe but small: 1.0043x median TTFT and 0.9969x E2E over 12 pairs. The material gain comes from removing the needless singleton seam and its synchronization boundary. The candidate is deliberately limited to prompts of at most 512 tokens in the prototype and must not be used with prefix snapshots.

Run the paired pilot (JSON is printed to stdout):

```bash
python3 -m experimental.upstream_dflash.bench_deferred_priming \
  --model models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --draft spd/Qwen3.6-27B-DFlash \
  --max-tokens 16 \
  --repetitions 3
```

Use `--no-fuse-cold-prefill` for the projection-only ablation.

## Adapter

`adapter.py` translates upstream event dataclasses into the dictionaries consumed by Mio's stream loop. It retains prefill accounting, tokens, summary metrics, fallback state and cycle diagnostics, and closes the upstream generator on cancellation.

```python
from experimental.upstream_dflash import (
    UpstreamGenerationRequest,
    stream_bundle_as_mio,
)

request = UpstreamGenerationRequest(
    max_new_tokens=64,
    prompt_tokens_override=tuple(prompt_tokens),
    stop_token_ids=tuple(stop_ids),
    verify_mode="dflash",
)
for event in stream_bundle_as_mio(upstream_bundle, request):
    consume(event)
```

To exercise the cold-prefill prototype, wrap the factory with `stream_with_deferred_drafter_priming(..., fuse_cold_prefill=True)`. The flag defaults to false and is valid only for a confirmed cold request with no snapshot or snapshot service. This uses a temporary upstream module patch and is intentionally single-threaded; a production implementation needs a session-level upstream policy/injection point.

## Promotion gates

`compatibility.py` produces explicit `pass`, `conditional`, or `block` results. `ready_for_default` is true only when every gate passes.

| Capability | Current status | Promotion requirement |
| --- | --- | --- |
| Greedy generation | Supported | Keep model/config/version-scoped parity at 100% |
| Streaming/cancellation | Adapter implemented | Load/disconnect tests under concurrent requests |
| Token stop IDs | Supported upstream | Stop-at-first-token and stop-inside-block parity tests |
| Static suppression | Supported upstream | Suppressed-token parity corpus |
| Dynamic suppress/relax | Blocked | Add Mio's timed EOS relaxation semantics upstream or retain vendored path |
| Tool prompts | Conditional | Prepared prompt tokens work; add tool-call parser/output corpus |
| Tool-required mode | Blocked | Depends on dynamic EOS suppress/relax behavior |
| PQ/TQ | Blocked | Upstream TargetOps cannot construct Mio PolarQuant/TurboQuant caches |
| Native KV8 | Conditional | Separate parity, latency and memory certificate |
| Prefix cache | Conditional upstream / blocked for Mio warm state | Use upstream `SnapshotService` end to end or design an explicit conversion; certify restore seams |
| Logprobs | Blocked | Extend upstream token events and acceptance accounting |
| Sampling | Blocked | Use target-only or an exact speculative-sampling backend |
| Parity | Model-scoped pass for upstream 0.1.8 matched artifact | Re-run after any model hash, MLX, MLX-LM, DFlash, quant, verify or block-size change |

The fused cold-prefill candidate requires its own stronger promotion certificate. The current 16-token pilot is insufficient by design.

## Required next experiments

1. Repeat the fused candidate at 64 and 256 generated tokens with the same matched-target schedule and report TTFT, decode, E2E, p95 and peak memory.
2. Add prompt buckets at 512, 2K, 8K and 32K tokens. The current deferred graph is capped at 512 tokens to avoid unbounded lazy-state retention.
3. Test prefix snapshot miss, partial hit, exact hit, publish and restore. Until then, fused mode must be disabled whenever a snapshot/service is present.
4. Run stop/suppress/tool-required corpora, generator cancellation, repeated load/unload and concurrent serving.
5. Validate one more Qwen hybrid scale and a second Apple Silicon machine before any default-route decision.
6. Only then replace the temporary global patch with an upstream `PrefillPolicy` or `TargetFeatureStore` injection point and integrate through Mio's normal engine selection.

## Tests

```bash
python3 -m pytest -q experimental/tests/test_upstream_dflash_adapter.py
python3 -m ruff check experimental/upstream_dflash experimental/tests/test_upstream_dflash_adapter.py
```

The pure contract suite currently contains 13 passing tests covering event translation, cancellation cleanup, request validation, patch restoration, exact parity-certificate identity, unsupported-feature blockers and provenance-aware comparison.
