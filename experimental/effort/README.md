# Mio adaptive-effort research protocol

## Status and claim boundary

This directory is an experimental research surface for conditional test-time
compute in Mio. It is not connected to Mio's production runtime, CLI, UI, or
public API. The five names `low`, `medium`, `high`, `xhigh`, and `ultra` are Mio
experiment labels; they are not claims about, or reimplementations of, a
provider's private reasoning system.

There is no demonstrated breakthrough, quality gain, or production-ready
effort feature yet. The profiles below are initial resource envelopes to be
calibrated, not benchmark results. More permitted work does not imply more
work on every request, and neither implies monotonically better quality.

Public provider documentation establishes only a user-facing pattern: an
effort control trades response thoroughness or reasoning work against latency
and token use. OpenAI documents model-dependent effort values and Anthropic
documents an effort signal used with adaptive thinking. Neither source
discloses enough internal detail to infer a chain, tree, Markov process, or
specific scheduler. In particular, `ultra` is Mio's top orchestration envelope;
it is not a literal OpenAI or Anthropic API value.

## Research question and hypothesis

The question is whether Mio can spend extra local MLX inference only when its
expected value exceeds its measured cost, improving terminal coding quality
without destroying the fast path or correct-completion throughput.

The proposed **Value-of-Compute Gated Markov Tree** (VoC-GMT) hypothesis is:

> A deterministic, calibrated gate over a bounded candidate tree can allocate
> request-level test-time compute more efficiently than a direct completion,
> a fixed sequential refinement chain, or a budget-matched fixed-width search.

This decomposes into three preregistered claims:

1. A public-evidence gate can rescue a material fraction of failed direct
   completions on held-out tasks.
2. Conditional branching can dominate budget-matched static baselines on the
   quality/latency Pareto frontier while preserving the direct fast path.
3. Any effect can be reproduced with a pinned Qwen 3.6 27B MLX configuration,
   rather than existing only in a small-model engineering pilot.

The null hypothesis is that adaptive routing offers no positive held-out
quality delta after correction, or that any gain costs too much end-to-end
latency or correct completions per second.

## Chain, tree, Markov policy, and constrained MDP

These terms describe different properties and must not be used
interchangeably.

- A **chain** has one active lineage. Each extra completion repairs or refines
  its immediate predecessor, so there is no sibling alternative.
- A **candidate tree** records parent/child structure and permits alternatives.
  Mio currently represents a repair as a child of the latest candidate, an
  alternative as a sibling rooted at the direct candidate, and a refinement as
  a child of the best candidate according to public evidence.
- A **Markov policy** chooses its next action from the complete current state,
  without consulting hidden history. A transition table alone does not make a
  policy an optimal Markov decision process solver.
- A **constrained Markov decision process** defines states, actions, transition
  probabilities, a reward, and resource constraints, then optimizes expected
  cumulative value subject to those constraints.

For this experiment, an observable state can be written as

```text
s_t = (tier, context bucket, candidate summaries, public trigger,
       used actions, depth, remaining output-token budget,
       remaining end-to-end latency budget)
```

and the action set is

```text
A = {direct, repair, alternative, refine, accept, stop}.
```

The current `MarkovTreeEffortController` is deliberately narrower than a
constrained-MDP solution. It is a deterministic, replayable, one-step greedy
policy with a frozen offline transition table. It performs no Bellman backup,
value iteration, rollout, look-ahead search, or online learning. Among feasible
actions it currently maximizes the conservative rescue-probability lower bound
per upper-bounded latency ratio:

```text
score(a | s) = rescue_probability_LCB(a | s)
               / extra_e2e_latency_ratio_UCB(a | s)
```

An action is feasible only when its token and latency upper bounds fit the
remaining envelope, its calibration support meets the minimum task-cluster
count, and its conservative rescue probability clears the profile floor.
Missing, stale, or identity-mismatched calibration data causes the controller
to stop rather than improvise.

VoC-GMT is the proposed next step, not the present implementation. Its gate is
to estimate

```text
VoC_LCB(a | s) = expected_quality_gain_LCB(a | s)
                 - lambda_latency * latency_cost_UCB(a | s)
                 - lambda_tokens * token_cost_UCB(a | s)
                 - lambda_risk * selection_risk_UCB(a | s)
```

and branch only when `VoC_LCB > 0` and every hard constraint remains feasible.
The lambdas must be fixed by tier before held-out evaluation. A future dynamic
programming or model-predictive variant is a separate hypothesis and must be
compared against the greedy policy rather than silently replacing it.

## Initial five-mode envelopes

These values are the defaults encoded by the experimental controller. The
initial direct-completion cap is configured separately by the runner.

| Mode | Max candidates | Extra output tokens | Per-candidate cap | Max E2E ratio | Uncertainty trigger |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low` | 1 | 0 | 96 | 1.00x | 1.00 |
| `medium` | 2 | 128 | 128 | 1.75x | 1.00 |
| `high` | 3 | 256 | 160 | 2.50x | 0.80 |
| `xhigh` | 4 | 384 | 192 | 3.25x | 0.72 |
| `ultra` | 5 | 640 | 224 | 4.50x | 0.65 |

All modes first take the same direct generation path. `low` cannot schedule a
second generation. Higher modes expose monotonically wider ceilings but spend
inside them only after an exact public-validator failure or a supported,
calibrated uncertainty trigger. The default support floor is eight independent
task clusters and the default conservative success floor is 0.10.

The latency ratio is an end-to-end request envelope relative to the observed
direct candidate, not a promise about decode speed. Generation, validation,
and controller overhead count toward it. An overshoot is retained as a deadline
violation; it is never removed from the report.

## Evidence firewall: routing versus evaluation

The controller and the terminal evaluator must have distinct interfaces.
Without this firewall, adaptive search can select against the test set and make
the reported gain invalid.

### Controller-visible public evidence

The HumanEval routing view contains only the public task identifier, prompt,
and entry-point name. The public validator may extract code, parse it, and
compile it. A malformed completion produces `FAIL`; a parseable completion
remains `UNKNOWN`. Syntax is not semantic correctness. A calibrated uncertainty
value may be supplied only if its estimator, features, and threshold were
frozen on the calibration split.

Candidate selection may use public validator state, calibrated uncertainty,
bounded output length, and deterministic tie-breaking. It may not use a hidden
test result, terminal evaluation score, hidden-test feedback, or knowledge of
which candidate would pass.

### Terminal hidden evaluation

The pinned HumanEval tests run only after the policy has selected its terminal
candidate. Their pass/fail verdict is reporting data. It cannot trigger a
repair, choose a sibling, update a transition online, or enter a later prompt.
An oracle that selects with hidden results may be reported only as a clearly
labelled, non-deployable upper bound.

The harness pins the official HumanEval archive by source revision and SHA-256
and assigns 32 tasks to calibration through a salted hash of task IDs. The
remaining tasks are held out. Calibration may use its own terminal outcomes to
fit conservative transition estimates; those task IDs must never enter the
confirmatory analysis. Each frozen table is bound to the exact model, model
configuration, prompt, sampler, corpus, split, and backend. No table is
transferred to a different identity.

The confirmatory run fails closed if provenance is incomplete, the Git tree is
dirty, a content hash is partial or missing, or leakage is detected.

## Experimental sequence

### 0. Preregister and freeze

Before evaluating held-out tasks, commit a machine-readable preregistration
that fixes:

- hypotheses, primary metrics, gates, planned comparisons, and exclusions;
- exact task split and manifest hash;
- exact model and tokenizer revisions, quantization, chat template, prompts,
  sampling parameters, stop strings, and output caps;
- controller, scorer, public validator, hidden verifier, and calibration-table
  hashes;
- MLX, MLX-LM, speculative-decoding component, macOS, and Mio revisions;
- hardware, memory, power mode, thermal policy, and cache protocol;
- bootstrap count and seed, generation seeds, and task execution order.

All confirmatory executions must use a clean commit. Exploratory runs are
labelled `pilot` and cannot be promoted after looking at held-out outcomes.

### 1. Calibration and engineering pilot

Use only the 32-task calibration split to verify the harness, estimate public
uncertainty calibration, and fit task-cluster lower/upper bounds for each
`(context, trigger, depth, action)` transition. Sparse context buckets must fall
back to a preregistered global row or be ineligible; they must not borrow hidden
held-out outcomes.

The initial envelopes may be adjusted during this phase, but every adjustment
invalidates the previous preregistration. Small models may be used to find
implementation bugs quickly. Their results are engineering evidence only and
cannot establish the 27B claim.

### 2. Held-out paired comparison

Run every locked strategy on exactly the same independent tasks. Randomize or
counterbalance strategy order to reduce thermal and temporal drift. If multiple
decode seeds are used, preregister a single task-level aggregation; generations
of the same task are not independent samples.

Compare each adaptive mode with `low`, then compare the best adaptive policy
with the strongest budget-matched static baseline. The present statistics
helper defaults to three planned comparisons; a full five-mode campaign has at
least four (`medium`, `high`, `xhigh`, and `ultra` versus `low`) and therefore
must override that default before the run. Any additional confirmatory baseline
also enters the multiplicity correction.

### 3. Qwen 3.6 27B MLX replication

Repeat calibration and the locked held-out protocol with the exact pinned
Qwen 3.6 27B MLX artifact. Do not reuse transition bounds learned on a smaller
model. Report model revision, quantization, peak memory, and every backend
revision.

Treat speculative acceleration as a separate experimental factor:

```text
effort policy:       low / selected adaptive policy
decode path:         base MLX / DFlash or DSpark-compatible path
cache condition:     cold / preregistered warm-prefix condition
```

This factorial separation is required to tell request-level quality routing
from kernel, prefill, caching, or speculative-decode speedups. DFlash and
DSpark results cannot be added arithmetically; measure the combined path
directly and attribute overlap through ablations.

### 4. Generalization

HumanEval alone is too small and too narrow to justify a coding-engine claim.
Replicate on at least one contamination-resistant or newly authored coding set
and one repository-level task set. Freeze their public/hidden boundary before
execution. A benchmark-specific gain is reported as such.

## Baselines and ablations

Each comparison must match the relevant candidate, output-token, and latency
envelope as closely as possible.

Required baselines are:

1. `low`: one direct completion with no controller retry.
2. Fixed chain: deterministic repair/refine steps with no branching or adaptive
   stop.
3. Fixed-width best-of-N or self-consistency: the same candidate budget with a
   public-only selector.
4. Fixed tree schedule: the same repair/alternative/refine actions without the
   transition gate.
5. Random feasible action: a seeded control for the learned action ordering.
6. Hidden-test oracle: analysis-only upper bound, never a deployable baseline.

Required ablations are:

- chain only versus candidate tree;
- validator-failure trigger only versus validator plus uncertainty;
- global transition rows versus supported context buckets;
- point estimates versus conservative LCB/UCB gating;
- current rescue-per-latency score versus the proposed VoC gate;
- token-only, latency-only, and joint hard constraints;
- public selector variants, with the hidden evaluator always inaccessible;
- base MLX versus each acceleration path, with identical effort policy;
- cold versus warm-prefix runs reported separately.

An adaptive policy that beats only an intentionally weak direct baseline, but
not the strongest budget-matched static baseline, is not a breakthrough.

## Metrics and measurement definitions

### Quality

- **Terminal accuracy:** fraction of selected candidates passing all hidden
  tests; this is paired by task with the baseline.
- **Accuracy delta:** adaptive minus baseline accuracy with a task-level 95%
  confidence interval.
- **Rescue and regression counts:** direct-fail/adaptive-pass and
  direct-pass/adaptive-fail discordant pairs.
- **Selection regret:** calibration-only gap between the selected candidate and
  an oracle candidate; never a routing feature.
- **Quality/cost frontier:** non-dominated points across all five modes and
  matched baselines.

### Latency and throughput

- **TTFT:** wall time from request admission to the first emitted token. Report
  the direct request TTFT and any additional-candidate TTFT separately.
- **Prefill throughput:** prompt tokens actually processed divided by measured
  prefill seconds. Cached and uncached tokens are counted separately; a cache
  hit is not labelled raw prefill speed.
- **Decode throughput:** emitted output tokens divided by measured decode
  seconds under one frozen first-token convention.
- **End-to-end latency:** generation plus public validation plus controller and
  orchestration time, reported as p50, p95, p99, and a paired ratio to `low`.
- **Correct completions per second:** number of terminal correct tasks divided
  by total end-to-end wall time. This is the primary joint quality/throughput
  utility, with a paired bootstrap ratio.
- **Effective selected tokens per second:** selected output tokens divided by
  the full request time. This is an orchestration-efficiency metric, not MLX
  decode tok/s.
- **Work and overhead:** candidates, prompt/output tokens, controller fraction,
  validator fraction, cache reuse, and deadline violations.

Also record peak resident memory and energy when a reproducible instrument is
available, but do not block the primary analysis on an uncalibrated power
estimate. Warm-up, cold-start, and steady-state results remain separate. Runs
with thermal throttling or competing inference workloads are invalidated under
the preregistered exclusion rule, not selectively discarded after seeing
quality.

## Statistics and pass gates

The task is the resampling unit. Every task must have exactly one aggregate row
per compared strategy. Use an exact two-sided McNemar test for paired binary
correctness and a deterministic task-cluster bootstrap for accuracy delta,
end-to-end latency ratio, and correct-completions-per-second ratio. Preregister
at least 10,000 bootstrap samples, the seed, and a 95% interval.

The current fail-closed confirmatory gate requires all of the following:

- at least 100 independent paired tasks;
- accuracy delta point estimate of at least `+0.05`;
- accuracy-delta confidence-interval lower bound strictly above zero;
- exact McNemar `p` below Bonferroni-corrected alpha;
- correct-completions-per-second ratio lower bound of at least `0.95`;
- direct fast-path end-to-end overhead no greater than `2%`;
- candidate end-to-end latency ratio point estimate no greater than `1.15`;
- zero candidate deadline violations;
- complete hashes, exact revisions, a clean Git tree, and no detected leakage.

With four preregistered tier comparisons and family-wise alpha `0.05`, the
per-comparison threshold is `0.0125`. If more comparisons are planned, the
denominator increases. Report every planned result, including failures; do not
select the multiplicity count after observing outcomes.

Passing this gate supports a narrow held-out improvement claim. The word
**breakthrough** additionally requires Pareto superiority over the strongest
matched baseline, successful Qwen 3.6 27B MLX replication, the same directional
effect on an independent benchmark family, and a reproducible artifact from a
clean commit. Until all four conditions hold, documentation must say that no
breakthrough has been demonstrated.

## Known threats to validity

- The Markov state is a lossy summary; unobserved prompt and candidate features
  may violate the assumed conditional transition model.
- A parse/compile validator is intentionally weak. Uncertainty may be
  miscalibrated under model, prompt, task, quantization, or backend shift.
- Thirty-two calibration tasks support only coarse estimates; over-bucketing
  creates unstable bounds and must fail closed.
- Candidate outcomes are correlated, so candidate count is never a statistical
  sample size.
- HumanEval may be contaminated in model pretraining and does not represent
  repository-scale coding work.
- Extra reasoning or self-repair can reinforce a wrong solution. Larger effort
  envelopes need not improve quality monotonically.
- Prefix caching and speculative decoding change action costs and can invalidate
  a calibration identity even if the model weights are unchanged.
- TTFT, prefill, and decode boundaries differ between backends. The harness must
  define and preserve one convention rather than comparing unlike timers.
- Thermal state, memory pressure, and background work can dominate small speed
  differences on Apple silicon.

## Required run artifacts

Every publishable run must retain immutable raw rows and a manifest containing:

```text
git revision and dirty flag
model/tokenizer revision and quantization
policy, prompt, task-manifest, scorer, verifier, and preregistration SHA-256
test split ID and explicit leakage flag
MLX/MLX-LM/accelerator revisions
hardware, macOS, power, thermal, and cache condition
generation seeds, task order, timing convention, and warm-up policy
all terminal outcomes, timings, token counts, decisions, and deadline violations
```

Reports must distinguish exploratory, calibration, confirmatory, replication,
and oracle data. Raw results are append-only; corrected analyses receive a new
artifact and explanation rather than overwriting the original.

## Primary sources

Provider sources are cited only for their public interface semantics, never as
evidence of private implementation details:

- [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)
  and [current model catalog](https://developers.openai.com/api/docs/models)
- [OpenAI: Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort)
  and [adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)

Research precedents motivate hypotheses but do not validate Mio:

- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)
- [PonderNet](https://arxiv.org/abs/2107.05407)
- [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [LE-MCTS](https://arxiv.org/abs/2412.15797)
- [Answer Convergence as a Signal for Early Stopping](https://aclanthology.org/2025.emnlp-main.904/)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Evaluating Large Language Models Trained on Code (HumanEval)](https://arxiv.org/abs/2107.03374)

ACT, PonderNet, and CALM allocate compute inside a trained network or decoder;
Mio's current experiment allocates whole request-level candidate generations.
Tree of Thoughts and LE-MCTS use richer search or learned evaluators; Mio's
current controller does neither. The distinctions are central to the ablation
plan and prevent adjacent literature from being presented as Mio evidence.
