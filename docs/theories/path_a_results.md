# Path A — adaptive KV with semantic clustering: results

## Setup

- Target: Qwen3.6-35B-A3B-UD-Q4_K_XL (40 layers, d_model=2048).
- Embedding: target's own early-layer hidden, mean-pooled across sequence, L2-normalized.
- Calibration: 8 topic-distinct prompts ("fib", "binsearch", "list_vs_tuple", "stack", "inverted_index", "mergesort", "linkedlist", "palindrome").
- Probes: (a) 8 near-duplicate rewordings of each calibration prompt (should match), (b) 4 topically-novel prompts (should NOT match).

## Retrieval metrics by embedding depth (n_early)

| n_early | correct top-1 | MRR | avg sim on correct | avg sim on novel | gap |
|---:|---:|---:|---:|---:|---:|
| 4 | 7/8 | 0.917 | 0.984 | 0.977 | **0.007** |
| 10 | 8/8 | 1.000 | 0.970 | 0.943 | 0.027 |
| 20 | 8/8 | 1.000 | 0.984 | 0.962 | 0.022 |
| 30 | 8/8 | 1.000 | 0.981 | 0.956 | 0.025 |

All 8/8 near-duplicates found in top-1 at n_early ≥ 10. But the gap between "correct-hit similarity" and "novel-prompt best-match similarity" is only 0.02-0.03. That's too narrow for a reliable threshold.

## Why novel prompts produce 0.94+ similarity

All 12 probes are "Write a Python function that ...". At the mean-pooled hidden level, they cluster by **domain** (Python coding tasks) not by **specific topic**. The model's early/mid layers do domain tagging; semantic separation happens in the last layers and the LM head.

**Fundamental constraint:** for retrieval to be *cheap on new prompts*, we need a cheap embedding. For the embedding to *discriminate topically*, we need to run most of the model. These two don't compose — the cheap embedding is in the wrong feature space.

## Reframe of what Path A actually delivers

Not a big-win warm-start. It's a **candidate ranker**: given 100 frozen KV snapshots on disk, pick the top-3 most likely to have a long shared prefix with the new prompt. Then fall into `scan_best_prefix_match` (the existing exact-prefix tester from C3) to decide which one's KV to load.

Value: O(N) cosine is cheap; saves scanning all N metadata files on disk when N is large. Linear-time `scan_best_prefix_match` is fine for ~20 prototypes but degrades at 100+. Path A's cosine pre-filter keeps lookup O(1).

**But** on a single-user mio instance, N is typically < 20 frozen snapshots, and the existing linear scan is fast. Path A as a performance optimization provides little value until a multi-user / many-prototype deployment exists.

## What Path A does NOT deliver

The "semantic warm-start" — use prototype KV directly on a non-prefix-matching prompt — is blocked by RoPE position-dependence. A prompt that's semantically similar but has different tokens has no reusable KV at the same positions. You can't just swap in the prototype's KV and expect correct outputs.

To make semantic warm-start work, you'd need **sub-prompt KV splicing with RoPE rewriting** — which is Path C below. **Paths A and C are coupled**: Path A selects the candidate, Path C makes the candidate's KV usable on a different-token prompt.

## Committed artifacts

- `mio/theories/prefill_autolearn/embed.py` — early-layer mean-pool embedder.
- `mio/theories/prefill_autolearn/prototype_store.py` — on-disk prototype index with LRU eviction, cosine search, config filtering.
- `mio/theories/prefill_autolearn/bench.py` — end-to-end retrieval bench.
- `tests/test_prefill_autolearn.py` — 8 unit tests (deterministic, no model load).
- `experiments/prefill_autolearn/bench*.json` — retrieval metrics per n_early.

## Next steps

1. **Path B (per-user LoRA)** — orthogonal to Path A, separate concern. Design doc next.
2. **Path C (substring KV splicing)** — the real prerequisite for Path A's semantic warm-start to matter. Design doc next.
3. **Path A productization** — park until we have a use case with >20 frozen prototypes.
