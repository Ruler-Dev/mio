"""Adaptive prefill cache with semantic clustering.

Extends mio.frozen_kv from exact-prefix-match to semantic-cluster match.
Key mechanism: embed prompts via the target model's own early-layer
hidden state (free compute), cluster prototypes on disk, nearest-neighbor
retrieve on new prompts, fall through to the existing longest-common-
prefix logic once a prototype is selected.
"""
