"""C1 — LowRank-QK post-hoc attention rank reduction.

Hypothesis from prefill research program:
  For each attention head h in each layer l, there exists a rank
  r_{h,l} in [4, 32] such that attention computed via low-rank QK
  factorization recovers >= 98% of full attention's Frobenius norm.

Scope of this module:
  1. Calibration capture — hook every attention layer, record Q and K
     tensors during a prefill pass on a calibration corpus.
  2. Rank analysis — per-head SVD, find minimal rank capturing a target
     fraction of Frobenius energy.
  3. Post-hoc projection matrices — P_Q, P_K per (layer, head) saved
     to safetensors for inference-time load.

Scope out of this module (future):
  - Custom Metal kernel for low-rank attention dispatch.
  - Benchmark / quality eval.
"""
