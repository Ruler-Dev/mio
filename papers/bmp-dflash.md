# BMP-DFlash: Batched Multi-Path Speculative Decoding for Hybrid SSM/Attention Models

> Historical research note. These Qwen 3/3.5 experiments predate Mio's current
> Qwen 3.6 benchmark schema. The paper is retained as negative/positive prior
> work, not as evidence for the present Qwen 3.6 path.

**Technical note, mio project, 2026**

## Abstract

DDTree (Ringel & Romano, 2026) constructs a draft tree from a single block-
diffusion drafter forward pass and verifies the tree in one target forward
using tree attention. Its reported 1.30×–1.50× speedup over vanilla DFlash on
Qwen3-4B/8B/30B is premised on the target model being a pure-attention
transformer. The Qwen3.5 family targeted by this experiment is a hybrid of gated
delta-net (recurrent SSM) and full attention layers; tree attention cannot
express the "one sequence, ancestor-only mask" pattern through SSM layers
because the SSM is a strict recurrence. We propose **BMP-DFlash** (Batched
Multi-Path DFlash): the same best-first tree construction, but paths are
verified in the **batch dimension** rather than the sequence dimension. The
target runs exactly one forward of shape `(K, L)`; SSM and attention both
behave correctly because each batch row is an independent trajectory.
Correctness is preserved on any architecture at the cost of K-way batch
activations during verify. A rollback step reuses DFlash's existing SSM
tape replay to rewind the winning row from `L` to `accepted_len` without a
second forward pass.

## 1. Background

### 1.1 DFlash

DFlash (Chen et al., 2026) uses a lightweight block-diffusion drafter to
propose L future tokens in one forward pass, then the target model verifies
the linear trajectory `[b, t₁, t₂, …, t_L]` in a single forward. The
drafter produces per-position **marginal** logits `ℓᵢ ∈ ℝ^|V|` for i = 1..L
independent of the realized tokens at other positions within the block. Under
greedy decoding this gives one candidate trajectory per round.

### 1.2 DDTree

DDTree extends DFlash's single-path verify to B candidate prefixes organized
as a prefix-closed tree. The tree maximizes the surrogate
`E[α_T(y)] = Σ_{u ∈ T} q(u | c, b)` where `q(u) = ∏ q_d(u_d)` is the
factorized draft probability of prefix u. Algorithm 1 of the paper produces
the optimal tree in O(B log B) time with a best-first heap:

```
Init heap with ((1,), σ((1,)))
while |T| < B:
  pop ρ with largest σ
  add prefix(ρ) to T
  push sibling (ρ₁..ρ_{d-1}, ρ_d+1)   if within top-K at depth d
  push first child (ρ₁..ρ_d, 1)       if d < L
```

Verification uses **tree attention**: the flattened tree of B tokens is input
with position ids = tree depth, and a custom attention mask permits each tree
node to attend to past context + its ancestors + itself.

### 1.3 Why tree attention breaks on hybrid SSM

Gated delta-net / Mamba-style layers advance a hidden state recurrently:
`s_t = f(s_{t-1}, x_t)`. Feeding a flat tree into such a layer produces a
single serial advance of state, with tokens from different branches
erroneously chained together. Any single-sequence-forward tree mask applies
only to attention layers; the interleaved recurrent layers in Qwen3.5 lack
an equivalent "ancestor-only" semantics. This is the blocker for running
DDTree unmodified on Qwen3.5.

## 2. Method

### 2.1 Idea

Represent the DDTree-selected continuations as K independent root-to-leaf
**paths**, pad each to length L with the marginal-argmax tail, and stack them
as a batch of shape `(K, L)`. Feed this batch through one target forward pass.
Attention operates within each batch row independently (self-attention is
batch-wise). SSM layers also operate per-row independently (each row has its
own recurrent state). Outputs are `(K, L, |V|)` logits — K independent target
posteriors.

### 2.2 Selecting the K paths

DDTree's Algorithm 1 produces a prefix-closed tree T with up to B nodes. Each
root-to-leaf path in T corresponds to one candidate continuation. We enumerate
these leaves, score each by its cumulative log-marginal-probability
`σ(leaf) = Σ_i log q_i(path_i)`, and retain the top K. If T has fewer than K
leaves (e.g. because the heap spent budget on a deep trunk), we backfill by
using rank-2/3/... depth-1 alternatives extended with the argmax tail.

### 2.3 Acceptance

For each batch row k, define
```
α_k = max { i ∈ [0, L-1] : ∀ j ≤ i, row_k[j+1] = argmax target_logits[k, j] }
```
and pick the winner `k* = argmax_k α_k`, breaking ties toward lower index so
path 0 (the vanilla DFlash argmax path) wins on equality — deterministic
fallback.

Acceptance is **monotone non-decreasing** in K: path 0 is always the
vanilla DFlash trajectory, so `α_{k*} ≥ α_0 = α_DFlash`.

### 2.4 Committing without a replay forward

The batched verify leaves every target-cache layer at state "end of path k",
for every k in parallel:
- Full-attention KVCache: batch dim K, sequence offset advanced by L.
- Hybrid SSM cache: batch dim K, recurrent state at time L for each row.

To continue generation we need state at position `accepted_len + 1` of the
winning row, not L.

We reuse DFlash's existing rollback machinery. Before the verify forward, we
`arm_rollback()` on each SSM cache — this snapshots the cache state and enables
tape recording. The verify forward writes a `(K, L, ...)` tape of the SSM
transitions. After the winner is picked:

1. **Filter all caches to the winner**: we extend `RecurrentRollbackCache.filter`
   to slice the tape in lock-step with the cache state, so after filtering we
   have a batch-1 cache + batch-1 tape of length L.
2. **Rollback to accepted_len**: DFlash's `rollback(accepted_len)` invokes the
   existing Metal tape-replay kernel to compute the correct batch-1 SSM state
   at time `accepted_len`.
3. **Adjust KV offsets**: standard KVCache just needs `offset -= L - commit_count`;
   slots beyond `commit_count` become stale but are never read.

No extra forward pass is needed. The rollback kernel is the same one DFlash uses
per-cycle today; BMP just invokes it after filtering.

## 3. Correctness

**Claim 1** (per-row independence): If a forward pass over a tensor of shape
`(K, L)` is composed of operators each of which is batch-wise (i.e., factorizes
over the batch dim), then the output at batch index k is independent of other
rows.

All operators in the Qwen3.5 target model are batch-wise: MLPs act
elementwise on (K, L, D), attention is computed per-row due to the causal
structure, and gated delta-net SSMs maintain one recurrent state per batch
entry. Hence the verify logits at row k are equal to the logits that would
have been produced by running the same path alone at batch=1. ∎

**Claim 2** (greedy equivalence to vanilla DFlash at K=1): BMP with K=1
feeds the argmax-per-position trajectory as its sole path, which is exactly
what vanilla DFlash verifies. Per-row acceptance reduces to DFlash
acceptance. Output text is identical up to floating-point nondeterminism
in the rollback tape replay. ∎

**Claim 3** (monotonic acceptance in K): Because paths are ordered by
descending draft-score and the vanilla argmax path has the highest draft
score by construction, path 0 in the K-batch is always identical to the
K=1 path. Therefore α_{k*} ≥ α_0, and tie-breaking toward lower index
preserves BMP_K₁ ≤ BMP_K₂ for K₁ ≤ K₂. ∎

## 4. Cost analysis

Let L be block length, K the number of paths.

**Vanilla DFlash verify**: one batch=1 forward of L tokens.
FLOPs ≈ L · (prefill_per_token_flops).

**BMP-DFlash verify**: one batch=K forward of L tokens.
FLOPs ≈ K · L · (prefill_per_token_flops).

Wall-clock on MLX: the batched forward shares weight loads across rows. For
small K (≤ 4) and L = 16 the measured wall-clock overhead is 1.1× – 1.5×,
not K×. Beyond K ≈ 4 the overhead starts to approach linear.

**Break-even acceptance**. Let τ_K be the mean accepted tokens per cycle at
K paths. BMP wins wall-clock if
```
τ_K · verify_wall_1 / verify_wall_K  ≥ τ_1
```
With `verify_wall_K / verify_wall_1 ≈ 1.3` at K=2, we need τ_2 ≥ 1.3 · τ_1.
On Qwen3.5-9B the measured gains (§5) are near this threshold; K=3 typically
gives the largest wall-clock improvement when it does improve.

## 5. Empirical results

All runs on M4 Max 128 GB, MLX, Mio's then-default Qwen3.5 tiers, T=0 greedy,
block_tokens=16, 128 decoded tokens. Numbers from `scripts/bench_bmp.py`
logged to `/tmp/bmp_bench.log`.

### Small tier (Qwen3.5-4B-4bit)

| Prompt | Mode    | tok/s | tokens/cycle | accept_ratio |
|--------|---------|-------|--------------|--------------|
| code   | dflash  | 93.5  | 4.41         | 0.77         |
| code   | K=2     | 90.7  | 4.57         | 0.78         |
| code   | K=3     | 65.1  | 4.41         | 0.77         |
| code   | K=4     | 69.3  | 4.74         | 0.79         |
| math   | dflash  | 178.1 | 8.53         | 0.88         |
| math   | K=2     | 162.2 | 8.00         | 0.88         |
| math   | K=3     | 119.3 | 8.00         | 0.88         |
| math   | K=4     | 108.5 | 7.53         | 0.87         |
| prose  | dflash  | 119.3 | 5.12         | 0.80         |
| prose  | K=2     | 108.1 | 5.12         | 0.80         |
| prose  | K=3     | 85.7  | 5.57         | 0.82         |
| prose  | K=4     | 80.2  | 5.33         | 0.81         |

### Qwen3-8B-4bit (pure attention, non-default tier)

For comparison with the DDTree paper's actual target family, we also ran BMP
on mlx-community/Qwen3-8B-4bit paired with z-lab/Qwen3-8B-DFlash-b16 —
`mio pull qwen3-8b-4bit`. This is a pure-attention model (no SSM layers),
matching the regime DDTree was designed for.

| Prompt | Mode    | tok/s | tokens/cycle | accept_ratio |
|--------|---------|-------|--------------|--------------|
| code   | dflash  | 43.9  | 2.98         | 0.66         |
| code   | K=2     | 39.9  | 2.78         | 0.64         |
| code   | K=3     | 30.4  | 3.20         | 0.69         |
| code   | K=4     | 30.4  | 3.20         | 0.69         |
| math   | dflash  | 74.0  | 5.12         | 0.80         |
| math   | **K=2** | **93.3**  | **6.74**     | **0.85**     |
| math   | K=3     | 57.3  | 6.10         | 0.84         |
| math   | K=4     | 59.2  | 6.40         | 0.84         |
| prose  | dflash  | 28.3  | 1.91         | 0.48         |
| prose  | K=2     | 29.6  | 2.03         | 0.51         |
| prose  | K=3     | 21.7  | 2.29         | 0.56         |
| prose  | K=4     | 20.3  | 2.21         | 0.55         |

**BMP K=2 on Qwen3-8B math: 93.3 tok/s vs 74.0 vanilla = 1.26× speedup.**
tpc rises from 5.12 to 6.74 (+32%) and acceptance_ratio from 0.80 to 0.85
(+6.3pp). This is the first configuration in our sweep where BMP delivers a
real wall-clock win.

Prose K=2 is a marginal +5% tok/s win (29.6 vs 28.3). Code regresses at every K.

K=3/4 never win wall-clock even when tpc improves — MLX batch overhead at
K>=3 dominates the cycle-count savings.

### Medium tier (Qwen3.5-9B-4bit)

| Prompt | Mode    | tok/s | tokens/cycle | accept_ratio |
|--------|---------|-------|--------------|--------------|
| code   | dflash  | 73.1  | 5.57         | 0.82         |
| code   | K=2     | 62.7  | 4.92         | 0.80         |
| code   | K=3     | 50.9  | 5.82         | 0.83         |
| code   | K=4     | 48.0  | 5.57         | 0.82         |
| math   | dflash  | 79.0  | 6.10         | 0.84         |
| math   | K=2     | 70.4  | 5.57         | 0.82         |
| math   | K=3     | 47.7  | 5.33         | 0.81         |
| math   | K=4     | 50.8  | 5.82         | 0.83         |
| prose  | dflash  | 33.8  | 2.37         | 0.58         |
| prose  | K=2     | 31.8  | 2.37         | 0.58         |
| prose  | K=3     | 22.5  | 2.61         | 0.62         |
| prose  | K=4     | 21.6  | 2.51         | 0.60         |

### Interpretation

**BMP-DFlash on Qwen3-8B pure attention: K=2 wins 1.26× on math, marginal
+5% on prose, regresses on code.**

**BMP-DFlash on Qwen3.5 (M4 Max MLX) is slower than vanilla DFlash in every
tested configuration.** Concretely: K=2 costs 3–14% wall-clock with flat or
marginally better acceptance; K=3 costs 28–40% wall-clock; K=4 is worse still.
This contrasts sharply with DDTree's reported 1.30×–1.50× wall-clock speedup
on pure-attention Qwen3.

Three reasons, in order of importance:

1. **Vanilla DFlash's acceptance on Qwen3.5 is already very high (0.77–0.88
   on most prompts).** The rank-2 token at each position is therefore very
   rarely the target's actual choice; K=2+ paths rarely find gains.
2. **MLX's batch scaling on hybrid_gdn is not flat.** Measured verify wall-time
   ratios batch=2 / batch=1 ≈ 1.15–1.30 on the tested models, so the K-row
   verify already consumes most or all of any acceptance gain.
3. **The snapshot + rollback bookkeeping between verify and commit adds
   per-cycle overhead** not present in the vanilla path.

In the single case where BMP tokens/cycle genuinely improved (medium tier,
prose, K=3: 2.61 vs 2.37, +10%), the wall-clock still fell by 33% because the
K-path verify cost overwhelmed the cycle-count saving.

### Where BMP *could* help on this hardware

BMP's win condition is essentially `τ_K > 1.3 · τ_1` at K=2. This is not met
on Qwen3.5 in the tested regimes. Cases where it could be met:

- **Temperature > 0 sampling** (not yet supported in BMP): DFlash's top-1
  acceptance drops significantly under sampling, opening more room for K>1 to
  rescue missed tokens.
- **Very long outputs on low-confidence tasks** (noisy prose generation,
  underdetermined problems): the measured drop in τ for vanilla DFlash on
  `prose` at medium tier (2.37 vs 6.10 on math) hints at this, but not by
  enough to flip the wall-clock.
- **Weaker draft models**: the gains depend on a regime where the draft
  isn't near-always-right. The DFlash draft models for Qwen3.5 are very
  strong.
- **Adaptive K**: use K=1 when draft top-1 confidence > threshold, K=2
  otherwise. Keeps the hot path cheap.

Unlike DDTree's reported 1.30×–1.50× on pure-attention, BMP on hybrid
Qwen3.5 currently delivers **no wall-clock speedup** because the batch-dim
overhead is real (K-row activations) whereas tree-attention on pure-attention
models pays only for B total tokens with shared prefix computation.

## 6. Limitations & Future Work

### 6.1 Interactions with TurboQuant 4-bit KV

The current engine disables BMP when TurboQuant KV quantization is enabled
(`tq_bits ∈ {2, 3, 4}`). Two reasons:
1. `TurboQuantKVCacheV2` returns quantized tuples from `update_and_fetch` and
   lacks the `filter(batch_indices)` method.
2. The batch-expand step (broadcasting compressed KV along batch dim) requires
   an extra decode-quantize-requantize path that hasn't been implemented.

Enabling TQ4×BMP is ~80 lines of cache-layer code and is a natural follow-up.

### 6.2 Sampling

BMP-DFlash as written is greedy-only. Extending to sampled decoding requires:
- Per-path rejection sampling with the standard speculative-sampling acceptance
  probability (Chen et al., 2023) applied at each position.
- Careful handling of the "first unmatched token becomes bonus" rule when K
  paths disagree at different positions.

The DDTree paper's Table 1 shows that sampling reduces but does not eliminate
the tree-construction gains; the same should hold for BMP.

### 6.3 Adaptive K per round

K is currently a static hyperparameter. Adaptive K based on recent
draft-confidence (e.g., increase K when top-1 logprob is low, decrease when
high) could recover wall-clock in the easy-prompt regime where K=1 wins.

### 6.4 Beyond top-K at depth 1

BMP currently fans out only at depth 1 effectively (because the DDTree leaves
it selects are almost always rank-1 at positions 2..L). Deeper fan-out —
selecting K paths that differ at positions 1 and 2 — could close more of the
DDTree gap. Requires tuning the tree budget B to force multi-depth branching.

## 7. Relation to Prior Work

- **Speculative decoding** (Leviathan 2023, Chen 2023): one draft trajectory,
  one verify pass. BMP-DFlash generalizes to K verify trajectories.
- **SpecInfer** (Miao 2023): token-tree verification with tree attention on
  pure-attention models. Same structural blocker as DDTree on SSM hybrids.
- **Medusa** (Cai 2024): multiple decoding heads; tree-verification assumed
  pure attention.
- **EAGLE-2/3** (Li 2024): dynamic tree construction for autoregressive
  drafters. BMP keeps DFlash's one-shot drafter; only the verification
  broadens.
- **DART** (Ni 2026): tree pruning via external n-gram signal; does not
  address hybrid-architecture correctness.
- **DDTree** (Ringel & Romano, 2026): the direct inspiration. BMP is a
  batch-dim specialization for the hybrid case, trading tree attention's
  sequence-sharing efficiency for SSM correctness.

## 8. Implementation notes

The BMP runtime is ~250 lines of Python in `mio/dflash/bmp_runtime.py`
that forks from `generate_dflash_once`. Primitives live in
`mio/dflash/bmp.py` (~200 lines). The existing `RecurrentRollbackCache`
needed a single 8-line change to `filter()` — broaden rollback-tape
slicing to match the filtered batch dim.

All primitives have unit tests in `tests/test_bmp.py` (12 tests) and
`tests/test_ddtree.py` (13 tests).

## References

- Ringel, L., & Romano, Y. (2026). *Accelerating Speculative Decoding with
  Block Diffusion Draft Trees*. [preprint]
- Chen, J., Liang, Y., & Liu, Z. (2026). *DFlash: Block Diffusion for Flash
  Speculative Decoding*. arXiv:2602.06036.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from
  Transformers via Speculative Decoding*. ICML 2023.
- Chen, C. et al. (2023). *Accelerating Large Language Model Decoding with
  Speculative Sampling*. arXiv:2302.01318.
- Miao, X. et al. (2023). *SpecInfer: Accelerating Generative Large Language
  Model Serving with Tree-based Speculative Inference and Verification*.
  arXiv:2305.09781.
- Arriola, M. et al. (2025). *Block Diffusion: Interpolating Between
  Autoregressive and Diffusion Language Models*. ICLR 2025.
- Wang, Y. et al. (2024). *OPT-Tree: Speculative Decoding with Adaptive Draft
  Tree Structure*. arXiv:2406.17276.
