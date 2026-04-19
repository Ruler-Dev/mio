"""Speculative Prefill (SpecPrefill) experimental implementation for mio.

Based on:
    Yang et al., "Speculative Prefill: Turbocharging TTFT with Lightweight
    and Training-Free Token Importance Estimation", ICML 2025.
    arXiv:2502.02789, github.com/Jingyu6/speculative_prefill

Experimental — kept entirely in `mio/experimental/` so we don't disturb
working code paths. Production wiring decision is deferred until quality
matches dense baseline within tolerance.
"""
