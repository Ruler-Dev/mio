"""Speculative Prefill (aka DraftKV) — experimental prefill acceleration.

Core idea: the target model processes the prompt through only the first
N_early layers. A small learned projector predicts KV cache entries for
the remaining layers directly from the intermediate hidden state. Decode
starts immediately; DFlash's per-token verification self-corrects any
incorrect predictions within the first few decode cycles (the target
re-runs all layers during decode).

This package is a SCAFFOLD. The projectors are untrained identity maps;
the runtime always falls back to full prefill through the production
path. Interfaces and tests exist so a trained projector can slot in.

Public surface:
    KVProjector, IdentityKVProjector   (projector.py)
    ConfidenceGate                     (gate.py)
    HarvestRecorder                    (harvest.py)
    sp_prefill                         (runtime.py)
"""

from mio.draft_kv.projector import IdentityKVProjector, KVProjector
from mio.draft_kv.gate import ConfidenceGate
from mio.draft_kv.harvest import HarvestRecorder
from mio.draft_kv.runtime import SPResult, sp_prefill

__all__ = [
    "KVProjector",
    "IdentityKVProjector",
    "ConfidenceGate",
    "HarvestRecorder",
    "SPResult",
    "sp_prefill",
]
