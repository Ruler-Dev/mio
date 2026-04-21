# Path B — Per-user LoRA adapter: design

**Status:** design only. Implementation is weeks of training infrastructure; spec captured here so a future session (or a compute-heavy branch) can pick it up.

## Goal

Train a small LoRA adapter per user on their own conversation history, loaded on request. Over time, the adapter shapes the model to their domain, their coding style, their frequently-revisited topics. Result: fewer tokens needed to express intent, faster prefills because common patterns compile well, and better quality on the user's workloads.

## Hypothesis

After 100-500 conversations with a user, a LoRA adapter trained on those conversations captures enough of the user's prompt distribution to:

1. **Reduce average prompt length by 10-30%** — the user doesn't need to restate standards/preferences the adapter has absorbed. This is a PREFILL win in the "fewer tokens to prefill" sense, not a "same tokens prefilled faster" sense.
2. **Improve first-shot code quality** — adapter handles house style, preferred libraries, repeating patterns.
3. **Reduce draft model KL vs target on user's domain** — higher DFlash acceptance → faster decode.

## Architecture

- LoRA rank r = 8 or 16 on target attention projections (q_proj, k_proj, v_proj, o_proj).
- LoRA rank r = 4 on GDN in_proj / out_proj.
- Typical adapter size: ~10-40 MB per user (versus 21 GB for the base model).
- Stored under `~/.mio/adapters/<user_id>/adapter.safetensors`.

## Training pipeline

1. **Harvest:** every completed conversation saved (mio/webui/sessions already does this) is a training example. Format as next-token-prediction on assistant responses, conditioning on preceding dialogue.
2. **Filtering:** drop conversations the user down-voted or aborted mid-stream. Keep only "completed successfully" sessions.
3. **Batching:** 30-day rolling window, batch_size=1 (large context), packed if fits.
4. **Trainer:** MLX-native LoRA trainer. MLX already has nn.Linear.weight + a low-rank perturbation pattern documented. Cost per step: ~forward + backward on an 8B model with rank-8 adapters ≈ similar-to-inference compute.
5. **Cadence:** nightly background training, 1-4 hours on M4 Max. Uses idle compute, not blocking user-facing inference.
6. **Validation:** hold out the last 5 conversations as a dev set. Only ship an adapter update if val loss decreases.

## Inference integration

- At model load: always load the base weights.
- On first request for a user: check if `~/.mio/adapters/<user_id>/adapter.safetensors` exists; if so, merge it at runtime (LoRA merge is O(n_layers × rank × d_model²) ≈ seconds).
- To switch users: unmerge the previous adapter + merge the new one. Negligible latency vs the 10s of seconds first-token-latency.

## Why this is NOT a prefill speedup in the bandwidth sense

The model size stays the same. Prefill FLOPs per token stay the same. What changes:
1. Fewer tokens needed to elicit desired behavior (indirect prefill reduction).
2. Better DFlash draft acceptance on the user's domain (decode win, not prefill win).

If you want a true prefill speedup, Path B does not deliver it directly. It's a quality / tokens-per-intent win that happens to reduce wall-clock for typical tasks.

## Risks

1. **Catastrophic forgetting:** continued training on a narrow user distribution degrades general capability. Mitigation: mix 10-20% of general calibration data from the base model's distribution (FineWeb sample) in every training batch.
2. **Privacy:** user conversations may contain sensitive data. Adapter weights could encode it, reidentifiable via extraction attacks. Mitigation: local-only; never shared; document the risk to the user.
3. **Adapter drift:** adapter works well for 30 days then stops improving. Detect via validation loss plateau; offer one-click reset.
4. **Compute cost to user:** nightly training burns watts. Mitigation: gated on "plugged in + not in active use" + user opt-in.

## Scope for a first implementation

1. **Week 1:** MLX LoRA kernel — low-rank perturbation on nn.Linear. Verified on small model.
2. **Week 2:** Training loop on harvested mio session files. Verified: loss decreases on held-out.
3. **Week 3:** Per-user adapter loading + merge path in MioEngine. Gated by `tier_config.user_adapter: str | None`.
4. **Week 4:** Nightly training orchestration via mio/dashboard or external cron.
5. **Week 5:** User-facing benchmark: avg prompt length, avg TTFT, avg DFlash acceptance — before and after 30 days of adapter training.

**Total: ~5 weeks part-time if everything works.** Research risk is moderate — LoRA is well-established for base models; the novelty is the per-user nightly pipeline on MLX, which is engineering not research.

## Why defer

Path B delivers a different class of win than Paths A and C. It's a long-horizon quality improvement that happens to reduce prefill cost indirectly. It doesn't compose multiplicatively with the other paths — you could ship Path B independently of anything else, or never.

Given the prefill-research scope this week, Path B is parked until there's a concrete use case that motivates per-user personalization in mio. The scaffolding here serves as a starting point.
