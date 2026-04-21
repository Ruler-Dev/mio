"""Upper-bound probe: can a linear projection predict late-layer KV from early hidden?

Before spending compute on projector training, we need an empirical
answer to: "what fraction of the variance in late-layer K/V is linearly
explained by the intermediate hidden state at an earlier layer?"

Method — closed-form ridge regression, per-layer:
  Given N samples of pairs (h_i, kv_i) where
    h_i   ∈ ℝ^{L, D_in}  (intermediate hidden)
    kv_i  ∈ ℝ^{L, D_out} (late-layer K or V, flattened across heads)
  We stack by sample to form matrices
    H   ∈ ℝ^{N*L, D_in}
    Y   ∈ ℝ^{N*L, D_out}
  Fit W = argmin ||HW - Y||_F^2 + λ||W||_F^2 via the closed-form
    W = (H^T H + λI)^-1 H^T Y
  Report R^2 = 1 - ||HW - Y||_F^2 / ||Y - mean(Y)||_F^2.

R^2 interpretation:
  - R^2 > 0.8 per layer → linear projector is strong; SP viable with a
    trained MLP that captures the remaining nonlinearity.
  - 0.4 < R^2 < 0.8 → marginal; projector helps but decode-verify will
    reject often. SP likely wins modestly on long prompts only.
  - R^2 < 0.4 → linear structure doesn't exist. Nonlinear projectors
    might still work but the hypothesis is shakier. Consider other
    prefill-acceleration strategies first.

This script is self-contained: no target model required at runtime. It
consumes harvest shards written by HarvestRecorder and produces a JSON
report with per-layer R^2 scores (separately for K and V).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx

from mio.draft_kv.harvest import load_shard


@dataclass
class LayerScore:
    """Per-layer regression results."""

    layer: int
    n_samples: int
    n_tokens: int
    d_in: int
    d_kv: int
    k_r2: float
    v_r2: float
    k_mse: float
    v_mse: float

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeReport:
    """Aggregated projectability report across layers."""

    early_layer: int
    target_layers: list[int]
    ridge_lambda: float
    shards: list[str]
    layers: list[LayerScore] = field(default_factory=list)

    def dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layers"] = [l.dict() for l in self.layers]
        return d

    def summary(self) -> str:
        if not self.layers:
            return "(no layers scored)"
        lines = [
            f"Probe: early={self.early_layer}, targets={self.target_layers}",
            f"Shards: {len(self.shards)}  ridge λ={self.ridge_lambda:g}",
            "",
            f"{'layer':>5}  {'samples':>7}  {'tokens':>7}  {'K R²':>8}  {'V R²':>8}",
            "-" * 48,
        ]
        for l in self.layers:
            lines.append(
                f"{l.layer:>5}  {l.n_samples:>7}  {l.n_tokens:>7}  "
                f"{l.k_r2:>8.4f}  {l.v_r2:>8.4f}"
            )
        mean_k = sum(l.k_r2 for l in self.layers) / len(self.layers)
        mean_v = sum(l.v_r2 for l in self.layers) / len(self.layers)
        lines.append("-" * 48)
        lines.append(f"{'mean':>5}  {'':>7}  {'':>7}  {mean_k:>8.4f}  {mean_v:>8.4f}")
        return "\n".join(lines)


def _stack_per_layer(
    shards: list[tuple[dict[str, mx.array], dict[str, str]]],
    layer: int,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Concatenate hidden + K + V across all samples for one late layer.

    Returns (H, K, V, n_samples) where shapes are:
        H: (N*L, D_in)
        K: (N*L, n_heads*head_dim)
        V: (N*L, n_heads*head_dim)
    """
    h_rows: list[mx.array] = []
    k_rows: list[mx.array] = []
    v_rows: list[mx.array] = []
    n_samples = 0
    for arrays, meta in shards:
        target_layers = [int(x) for x in meta["target_layers"].split(",")]
        if layer not in target_layers:
            continue
        sample_count = int(meta["sample_count"])
        for i in range(sample_count):
            hid = arrays.get(f"s{i}/hidden")
            k = arrays.get(f"s{i}/layer{layer}/K")
            v = arrays.get(f"s{i}/layer{layer}/V")
            if hid is None or k is None or v is None:
                continue
            # hid: (1, L, D_in); flatten to (L, D_in)
            hid_flat = hid.reshape(-1, hid.shape[-1])
            # K, V: (1, n_heads, L, head_dim); move L to front, flatten heads.
            B, H, L, Dh = k.shape
            k_flat = k.transpose(0, 2, 1, 3).reshape(B * L, H * Dh)
            v_flat = v.transpose(0, 2, 1, 3).reshape(B * L, H * Dh)
            if k_flat.shape[0] != hid_flat.shape[0]:
                continue  # inconsistent sample, skip
            h_rows.append(hid_flat.astype(mx.float32))
            k_rows.append(k_flat.astype(mx.float32))
            v_rows.append(v_flat.astype(mx.float32))
            n_samples += 1

    if not h_rows:
        raise ValueError(f"no (hidden, KV) pairs found for layer {layer}")
    H = mx.concatenate(h_rows, axis=0)
    K = mx.concatenate(k_rows, axis=0)
    V = mx.concatenate(v_rows, axis=0)
    return H, K, V, n_samples


def _ridge_fit_and_r2(
    H: mx.array,
    Y: mx.array,
    lam: float,
) -> tuple[float, float]:
    """Closed-form ridge + R^2 on training set.

    W = (H^T H + λI)^-1 H^T Y
    R^2 = 1 - SSE / SST  where SSE = ||HW - Y||_F^2 and
          SST = ||Y - mean(Y)||_F^2.

    Returns (r2, mse).
    """
    N, D = H.shape
    _, Dy = Y.shape
    HtH = H.T @ H
    HtY = H.T @ Y
    reg = lam * mx.eye(D, dtype=H.dtype)
    W = mx.linalg.solve(HtH + reg, HtY, stream=mx.cpu)
    Y_hat = H @ W
    resid = Y_hat - Y
    sse = float(mx.sum(resid * resid).item())
    y_mean = mx.mean(Y, axis=0, keepdims=True)
    centered = Y - y_mean
    sst = float(mx.sum(centered * centered).item())
    if sst == 0.0:
        return 0.0, sse / max(1, N * Dy)
    r2 = 1.0 - sse / sst
    mse = sse / (N * Dy)
    return r2, mse


def run_probe(
    shard_paths: Iterable[Path],
    *,
    ridge_lambda: float = 1e-2,
    layers: list[int] | None = None,
) -> ProbeReport:
    """Fit ridge per target layer over all shards; return a ProbeReport."""
    loaded = [load_shard(p) for p in shard_paths]
    if not loaded:
        raise ValueError("no shards provided")

    first_meta = loaded[0][1]
    early_layer = int(first_meta["early_layer"])
    all_target_layers = [int(x) for x in first_meta["target_layers"].split(",")]

    # Sanity-check shards share schema.
    for _, m in loaded[1:]:
        if int(m["early_layer"]) != early_layer:
            raise ValueError("shards have differing early_layer; cannot mix")
        if [int(x) for x in m["target_layers"].split(",")] != all_target_layers:
            raise ValueError("shards have differing target_layers; cannot mix")

    target_layers = layers if layers is not None else all_target_layers

    report = ProbeReport(
        early_layer=early_layer,
        target_layers=list(target_layers),
        ridge_lambda=ridge_lambda,
        shards=[str(p) for p in shard_paths],
    )

    for layer in target_layers:
        H, K, V, n_samples = _stack_per_layer(loaded, layer)
        k_r2, k_mse = _ridge_fit_and_r2(H, K, ridge_lambda)
        v_r2, v_mse = _ridge_fit_and_r2(H, V, ridge_lambda)
        report.layers.append(
            LayerScore(
                layer=layer,
                n_samples=n_samples,
                n_tokens=int(H.shape[0]),
                d_in=int(H.shape[1]),
                d_kv=int(K.shape[1]),
                k_r2=k_r2,
                v_r2=v_r2,
                k_mse=k_mse,
                v_mse=v_mse,
            )
        )

    return report


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Upper-bound probe for draft_kv projector viability."
    )
    parser.add_argument(
        "shards",
        nargs="+",
        type=Path,
        help="Paths to HarvestRecorder output safetensors shards.",
    )
    parser.add_argument(
        "--lambda", dest="ridge_lambda", type=float, default=1e-2,
        help="Ridge regularization (L2 on projector weights).",
    )
    parser.add_argument(
        "--layer", action="append", type=int, default=None,
        help="Target layer(s) to score. Default: all in shard metadata.",
    )
    parser.add_argument(
        "--json", dest="json_out", type=Path, default=None,
        help="Also write machine-readable report to this path.",
    )
    args = parser.parse_args(argv)

    report = run_probe(
        args.shards,
        ridge_lambda=args.ridge_lambda,
        layers=args.layer,
    )
    print(report.summary())
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report.dict(), indent=2))
        print(f"\nJSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
