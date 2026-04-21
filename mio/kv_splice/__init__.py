"""KV splicing — reuse pre-RoPE K/V across prompts where the same chunk
of tokens appears at different absolute positions.

Production interface:
    from mio.kv_splice import ChunkStore, detect_chunks, install_splice_hooks

    store = ChunkStore()
    store.ingest(tokens, target_model, layers_to_store=[3, 7, 11, 15, 19])

    # Later, on a new prompt:
    sites = detect_chunks(new_prompt_tokens, store)
    cleanup = install_splice_hooks(target_model, sites, store)
    try:
        run_prefill(new_prompt_tokens, ...)
    finally:
        cleanup()

Research basis: docs/theories/path_c_results.md.

Layer set [3, 7, 11, 15, 19] is the byte-exact-match sweet spot for
Qwen3.6-35B-A3B (5 of 10 attention layers). Different models need their
own Phase 3 validation to find the right layer set.
"""

# Layer indices that are spliceable on Qwen3.6-35B-A3B.
# Phase 3 K-sweep found this exact set produces byte-exact output.
SPLICEABLE_LAYERS_QWEN36_A3B = (3, 7, 11, 15, 19)
