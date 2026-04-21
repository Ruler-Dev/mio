"""Week 2 Path A — MLP replacement for single-layer swap.

For each candidate layer, train a 2-layer MLP:
    delta = W_2 @ silu(W_1 @ x + b_1) + b_2
    y = x + delta

where:
    W_1: (d, bottleneck), W_2: (bottleneck, d)
    bottleneck = 512 (1/4 of d=2048)

Trained via MSE on harvested (x_in, x_out - x_in) pairs. Uses MLX's
autodiff, Adam optimizer, 80/20 train/test split.

Reports MSE + R² on test set, compares to linear baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


class MLPReplacement(nn.Module):
    """Linear skip + MLP residual.

    y = W_skip @ x + W_2 @ silu(W_1 @ x + b_1) + b_2

    The linear skip is the primary path; the MLP branch is a residual
    correction. Initialization: W_skip starts from closed-form OLS, MLP
    branch zero-initialized. Training can only *improve* over linear.
    """

    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        self.d_model = d_model
        self.skip = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, bottleneck, bias=True)
        self.fc2 = nn.Linear(bottleneck, d_model, bias=True)
        # Zero-init the MLP branch so the model starts as pure skip.
        self.fc2.weight = mx.zeros_like(self.fc2.weight)
        self.fc2.bias = mx.zeros_like(self.fc2.bias)

    def init_skip_from_ols(self, W_ols: mx.array) -> None:
        """Install the OLS solution as initial skip weight.

        W_ols has shape (d_in, d_out) (column-major ordering). nn.Linear
        stores weight as (d_out, d_in), so we transpose.
        """
        self.skip.weight = W_ols.astype(self.skip.weight.dtype).T

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.fc1(x))
        return self.skip(x) + self.fc2(h)


def _r2_mx(y_pred: mx.array, y_true: mx.array) -> float:
    resid = (y_true - y_pred) ** 2
    ss_res = float(mx.sum(resid).item())
    y_mean = mx.mean(y_true, axis=0, keepdims=True)
    centered = (y_true - y_mean) ** 2
    ss_tot = float(mx.sum(centered).item())
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_mlp_for_layer(
    X_np: np.ndarray, Y_np: np.ndarray,
    *, bottleneck: int = 512, epochs: int = 8, batch_size: int = 4096,
    lr: float = 1e-3, test_frac: float = 0.2, seed: int = 42,
) -> tuple[MLPReplacement, float, float]:
    """Fit MLP to predict delta = Y - X from X. Returns (model, r2_train, r2_test)."""
    d = X_np.shape[1]
    # Train delta = Y - X
    delta = (Y_np.astype(np.float32) - X_np.astype(np.float32))
    X = X_np.astype(np.float32)

    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    X = X[perm]; delta = delta[perm]
    split = int(N * (1 - test_frac))
    X_tr, X_te = X[:split], X[split:]
    D_tr, D_te = delta[:split], delta[split:]

    model = MLPReplacement(d_model=d, bottleneck=bottleneck)
    # Warm-start skip from closed-form OLS on train split.
    reg = 1e-3
    XtX = X_tr.T @ X_tr + reg * np.eye(d, dtype=np.float32)
    XtD = X_tr.T @ D_tr
    W_ols = np.linalg.solve(XtX, XtD)  # (d, d), maps x -> delta
    model.init_skip_from_ols(mx.array(W_ols))
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=lr)

    # Baseline (pure OLS skip) R² for reference.
    pred_ols = model(mx.array(X_te))
    r2_ols = _r2_mx(pred_ols, mx.array(D_te))
    print(f"  OLS-skip init: test_r2={r2_ols:.4f}  (must match regression.json)", flush=True)

    def loss_fn(mdl, xb, yb):
        pred = mdl(xb)
        return mx.mean((pred - yb) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n_batches = (X_tr.shape[0] + batch_size - 1) // batch_size
    print(
        f"  MLP d={d} bot={bottleneck} params={sum(v.size for _, v in tree_flatten(model.parameters())):.3g} "
        f"N_tr={X_tr.shape[0]} batches/ep={n_batches}",
        flush=True,
    )
    for ep in range(epochs):
        t0 = time.perf_counter()
        ep_losses = []
        perm2 = rng.permutation(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], batch_size):
            idx = perm2[i:i + batch_size]
            xb = mx.array(X_tr[idx])
            yb = mx.array(D_tr[idx])
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            ep_losses.append(float(loss.item()))
        avg_loss = sum(ep_losses) / len(ep_losses)
        # Test R²
        pred_te = model(mx.array(X_te))
        r2_te = _r2_mx(pred_te, mx.array(D_te))
        print(f"  epoch {ep+1}/{epochs}: train_mse={avg_loss:.5f}  test_r2={r2_te:.4f}  [{time.perf_counter()-t0:.1f}s]",
              flush=True)

    # Final R² train + test
    pred_tr = model(mx.array(X_tr))
    r2_tr = _r2_mx(pred_tr, mx.array(D_tr))
    pred_te = model(mx.array(X_te))
    r2_te = _r2_mx(pred_te, mx.array(D_te))
    return model, r2_tr, r2_te


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", default="experiments/distill_linear/harvest")
    p.add_argument("--regression",
                   default="experiments/distill_linear/regression.json")
    p.add_argument("--layers", nargs="+", type=int, default=[1],
                   help="Layer indices to fit MLPs for")
    p.add_argument("--bottleneck", type=int, default=512)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="experiments/distill_linear/mlps")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    reg = json.loads(Path(args.regression).read_text())
    per_layer = {r["layer_idx"]: r for r in reg["layer_results"]}

    report = []
    for idx in args.layers:
        if idx not in per_layer:
            continue
        X = np.load(Path(args.harvest) / f"layer{idx}_X.npy")
        Y = np.load(Path(args.harvest) / f"layer{idx}_Y.npy")
        linear_r2 = per_layer[idx]["delta_r2_test"]
        print(f"\n[mlp-fit] layer {idx}  linear_r2_test={linear_r2:.4f}", flush=True)
        model, r2_tr, r2_te = fit_mlp_for_layer(
            X, Y, bottleneck=args.bottleneck, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
        )
        out_path = Path(args.out_dir) / f"layer{idx}_mlp.safetensors"
        # Save params
        flat = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(out_path), flat)
        improvement = r2_te - linear_r2
        report.append({
            "layer_idx": idx,
            "linear_r2": linear_r2,
            "mlp_r2_train": r2_tr,
            "mlp_r2_test": r2_te,
            "delta_r2": improvement,
            "bottleneck": args.bottleneck,
            "epochs": args.epochs,
            "path": str(out_path),
        })
        print(f"  => linear={linear_r2:.4f}  mlp_test={r2_te:.4f}  improvement={improvement:+.4f}",
              flush=True)

    Path(args.out_dir).joinpath("report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[mlp-fit] wrote {args.out_dir}/report.json", flush=True)


if __name__ == "__main__":
    main()
