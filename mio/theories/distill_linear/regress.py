"""Per-layer linear regression on harvested (x_in, x_out) pairs.

For each layer, fit:
  1. Full-output form: Y ≈ W_full @ X
  2. Delta form:       (Y - X) ≈ W_delta @ X

Report R² on a held-out 20% test split. A high R² means the layer's
function (or its residual delta) is linearly approximable.

The matmul W @ x is massively cheaper than a full attention or GDN block:
  - attention: 4 projections + SDPA + output proj ≈ 8-10 matmuls + O(L²)
  - GDN:       5+ projections + conv1d + recurrence + output proj
  - W @ x:     one matmul, d_model × d_model = 2048 × 2048 = 4M FLOPs/token

Layers with high R² are replacement candidates. Report per-layer ranking.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class LayerResult:
    layer_idx: int
    is_linear: bool  # True = GDN, False = attention
    n_train: int
    n_test: int
    d_model: int
    # Full-output fit
    full_r2_train: float
    full_r2_test: float
    # Delta fit (residual contribution)
    delta_r2_train: float
    delta_r2_test: float
    # Spectral
    delta_w_rank98: int
    # Identity baselines
    identity_r2_full: float  # Y ≈ X (no W)
    identity_r2_delta: float  # (Y-X) ≈ 0 (no W)


def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination. Computed in a numerically-safe way."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _fit_and_score(X: np.ndarray, Y: np.ndarray, test_frac: float = 0.2, ridge: float = 1e-3) -> tuple[np.ndarray, float, float]:
    """Fit Y ≈ W @ X^T (row-wise: Y_i ≈ X_i @ W^T), return (W, r2_train, r2_test).

    X: (N, d_in), Y: (N, d_out), solves W such that Y ≈ X @ W.
    With ridge: W = (X^T X + λI)^-1 X^T Y.
    """
    N, d_in = X.shape
    split = int(N * (1 - test_frac))
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    X = X[perm]
    Y = Y[perm]
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    reg = ridge * np.eye(d_in, dtype=np.float32)
    XtX = X_tr.astype(np.float32).T @ X_tr.astype(np.float32)
    XtY = X_tr.astype(np.float32).T @ Y_tr.astype(np.float32)
    W = np.linalg.solve(XtX + reg, XtY)  # (d_in, d_out)

    Y_pred_tr = X_tr.astype(np.float32) @ W
    Y_pred_te = X_te.astype(np.float32) @ W
    r2_tr = _r2(Y_pred_tr, Y_tr.astype(np.float32))
    r2_te = _r2(Y_pred_te, Y_te.astype(np.float32))
    return W, r2_tr, r2_te


def _spectral_rank(W: np.ndarray, threshold: float = 0.98) -> int:
    """Number of singular values needed to capture `threshold` of the
    Frobenius energy of W."""
    s = np.linalg.svd(W.astype(np.float32), compute_uv=False)
    e = s ** 2
    cum = np.cumsum(e) / e.sum()
    return int(np.searchsorted(cum, threshold)) + 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--harvest", default="experiments/distill_linear/harvest",
        help="Harvest directory with per-layer .npy files and meta.json",
    )
    p.add_argument("--out-json", default="experiments/distill_linear/regression.json")
    p.add_argument("--ridge", type=float, default=1e-3)
    args = p.parse_args()
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    harvest_dir = Path(args.harvest)
    meta = json.loads((harvest_dir / "meta.json").read_text())
    print(f"[regress] harvest dir {harvest_dir}: {len(meta['layers'])} layers, d={meta['d_model']}",
          flush=True)

    results: list[LayerResult] = []
    for entry in meta["layers"]:
        i = entry["idx"]
        is_lin = bool(entry["is_linear"])
        X = np.load(harvest_dir / f"layer{i}_X.npy")
        Y = np.load(harvest_dir / f"layer{i}_Y.npy")
        N, d = X.shape
        # Float32 for compute.
        X_f32 = X.astype(np.float32)
        Y_f32 = Y.astype(np.float32)
        delta = Y_f32 - X_f32

        # Fit full-output and delta
        t0 = time.perf_counter()
        W_full, fr_tr, fr_te = _fit_and_score(X_f32, Y_f32, ridge=args.ridge)
        W_del, dr_tr, dr_te = _fit_and_score(X_f32, delta, ridge=args.ridge)
        dt = time.perf_counter() - t0

        # Identity baselines
        id_r2_full = _r2(X_f32, Y_f32)
        id_r2_delta = _r2(np.zeros_like(delta), delta)

        # W_delta spectral
        rank = _spectral_rank(W_del)

        lr = LayerResult(
            layer_idx=i, is_linear=is_lin, n_train=int(N * 0.8),
            n_test=int(N * 0.2), d_model=d,
            full_r2_train=fr_tr, full_r2_test=fr_te,
            delta_r2_train=dr_tr, delta_r2_test=dr_te,
            delta_w_rank98=rank,
            identity_r2_full=id_r2_full,
            identity_r2_delta=id_r2_delta,
        )
        results.append(lr)
        kind = "GDN " if is_lin else "attn"
        print(
            f"  L{i:2d} {kind}  N={N:>6d} d={d}  "
            f"full_r2_te={fr_te:>6.3f}  delta_r2_te={dr_te:>6.3f}  "
            f"(id_full={id_r2_full:>6.3f}, id_delta={id_r2_delta:>6.3f})  "
            f"W_rank98={rank:>4d}/{d}  "
            f"[{dt:.1f}s]",
            flush=True,
        )

    print("\n[regress] === RANKED BY DELTA R² (higher = more replaceable) ===",
          flush=True)
    ranked = sorted(results, key=lambda r: -r.delta_r2_test)
    print(f"  {'layer':>5}  {'kind':>4}  {'delta_r2':>9}  {'full_r2':>7}  {'id_delta':>9}  {'id_full':>7}  {'W_rank':>6}")
    for r in ranked:
        kind = "GDN" if r.is_linear else "attn"
        print(
            f"  L{r.layer_idx:>3d}  {kind:>4}  "
            f"{r.delta_r2_test:>9.3f}  {r.full_r2_test:>7.3f}  "
            f"{r.identity_r2_delta:>9.3f}  {r.identity_r2_full:>7.3f}  "
            f"{r.delta_w_rank98:>4d}/{r.d_model}",
            flush=True,
        )

    # Save
    out = {
        "ridge": args.ridge,
        "layer_results": [asdict(r) for r in results],
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[regress] wrote {args.out_json}")


if __name__ == "__main__":
    main()
