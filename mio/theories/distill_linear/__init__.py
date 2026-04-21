"""Week 1 — Linear replacement probe.

For each decoder layer l, fit W_l such that layer_l(x) ~= x + W_l @ LN(x)
on harvested activations. Report per-layer R^2. Layers with high R^2
are replacement candidates.
"""
