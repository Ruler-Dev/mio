"""Path C — substring KV splicing with RoPE rewriting.

Phase 1 (this module): measure K_base context-robustness. Go/no-go for
the rest of Path C.
Phase 2+: if Phase 1 passes, build RoPE-rewriting + splice-aware prefill.
"""
