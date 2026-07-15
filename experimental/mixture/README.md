# Mixture of drafters: isolated R&D prototype

This experiment asks a narrow question: can Mio choose **either** DSpark or
DFlash for a request and beat the best static drafter without running both?
It is not integrated into Mio's runtime.

## Design

Each routing decision executes one drafter. The selected arm alone updates its
online curve. The curve estimates TTFT, accepted tokens per verification round,
total cost per verification round, directly timed verification cost when the
backend exposes it, and fallback/parity telemetry. Its latency projection is:

```text
TTFT + requested_tokens / accepted_per_round * seconds_per_round
```

This uses acceptance and round cost without double-counting the verification
timer already included in wall time. DSpark currently exposes target-forward
counts rather than a separate verification duration, so those counts are kept
as telemetry and the measured round cost remains authoritative.

The router has deterministic one-sample-per-arm calibration, a hard exploration
budget, static-best fallback, a switch margin plus consecutive-decision
hysteresis, and a regression/fallback guard. Counterfactual outcomes in a full
benchmark matrix are never revealed to the online update; the replay evaluator
uses them only after routing to calculate static and oracle comparators.

Run the existing matched replay with:

```bash
python3 -m experimental.mixture.replay \
  benchmarks/results/speculative-matched-qwen3-4b-20260715-v041.json \
  --pretty

python3 -m experimental.mixture.replay \
  benchmarks/results/speculative-matched-qwen36-27b-20260715.json \
  --pretty
```

## What counts as a real gain

A scoped router candidate must be frozen after calibration and evaluated on
disjoint held-out prompts. It must beat the best static drafter by at least 5%
in paired wall time, with a prompt-cluster bootstrap confidence lower bound
strictly above 1.0. P95 latency may not regress, peak memory may regress by at
most 5%, greedy token parity must be 100%, and fallback count must be zero.

A broader claim additionally needs at least two model scales including 27B,
independent hardware replication, and sustained concurrency/load evidence.
This replay intentionally reports `global_breakthrough: false`; same-corpus
online replay is useful for falsification and router debugging, not a discovery
claim.

## Current matched 4B replay

On `speculative-matched-qwen3-4b-20260715-v041.json`, DSpark is the fastest arm
on all 12 matched requests. Its aggregate wall time is 6.5138 s versus 9.2867 s
for DFlash, so the best static DSpark policy is 1.4257x faster than static
DFlash on this corpus.

With the default two-decision calibration budget, the router selects DFlash
once, DSpark once, then DSpark for the remaining ten requests. It therefore
converges to DSpark (11/12 total selections), but takes 6.7263 s: only 0.9684x
the speed of static DSpark, or a 3.16% regression. Its selected exact-token
parity is 75%. The result is not a mixture gain and not a breakthrough; it says
that this corpus contains no routing opportunity because one static arm
dominates every observed request.

## Current matched 27B replay

On `speculative-matched-qwen36-27b-20260715.json`, DFlash is the fastest arm on
all 12 matched requests. Its aggregate wall time is 26.0635 s versus 45.0937 s
for DSpark, making static DFlash 1.7301x faster than static DSpark on this
corpus.

The same two-decision calibration selects DFlash once, DSpark once, then DFlash
for the remaining ten requests. The router therefore converges to DFlash
(11/12 total selections), but takes 27.1818 s: 0.9589x the speed of static
DFlash, or a 4.11% regression. Selected exact-token parity is 100% and there
are no fallbacks. This is still neither a mixture gain nor a breakthrough: the
calibration cost loses to the arm that dominates every observed request, and
the replay is not a held-out evaluation.

The reversal between scales is useful engineering evidence—DSpark dominates
this 4B corpus while DFlash dominates this 27B corpus—but it does not show a
request-level mixture advantage within either scale. A scale-aware static
selection can exploit that difference without paying online exploration cost;
the stronger router claim still requires heterogeneous held-out prompts where
different arms win different requests.
