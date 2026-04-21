"""Tests for mio.draft_kv (speculative prefill scaffold).

Deterministic, no model load. Exercises:
- Projector shape contracts (IdentityKVProjector, LinearKVProjector).
- ConfidenceGate thresholding (proceed vs reject).
- HarvestRecorder shape validation + safetensors round-trip.
- sp_prefill control flow (gate reject, forced fallback, not-implemented path).
- probe.run_probe: ridge regression math on synthetic shards with known
  answer (exact linear relation → R^2 ≈ 1; independent noise → R^2 ≈ 0).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mio.draft_kv.gate import ConfidenceGate, GateDecision
from mio.draft_kv.harvest import HarvestRecorder, load_shard
from mio.draft_kv.projector import (
    IdentityKVProjector,
    KVShape,
    LinearKVProjector,
)
from mio.draft_kv.runtime import SPResult, sp_prefill


# --- KVShape / IdentityKVProjector ------------------------------------------


def test_kvshape_channel_dim():
    shape = KVShape(n_kv_heads=4, head_dim=8)
    assert shape.channel_dim == 2 * 4 * 8  # K+V concat


def test_identity_projector_returns_correct_shape():
    proj = IdentityKVProjector()
    shape = KVShape(n_kv_heads=4, head_dim=8)
    hidden = mx.random.normal((1, 16, shape.channel_dim))
    k, v = proj.project(hidden, layer_idx=3, kv_shape=shape)
    assert k.shape == (1, 4, 16, 8)
    assert v.shape == (1, 4, 16, 8)


def test_identity_projector_rejects_wrong_channel_dim():
    proj = IdentityKVProjector()
    shape = KVShape(n_kv_heads=4, head_dim=8)  # channel_dim = 64
    hidden = mx.random.normal((1, 4, 32))  # wrong
    with pytest.raises(ValueError):
        proj.project(hidden, layer_idx=0, kv_shape=shape)


def test_identity_projector_preserves_content():
    """Split + reshape + transpose should round-trip values exactly."""
    proj = IdentityKVProjector()
    shape = KVShape(n_kv_heads=2, head_dim=3)
    hidden = mx.arange(1 * 2 * shape.channel_dim, dtype=mx.float32).reshape(
        1, 2, shape.channel_dim
    )
    k, v = proj.project(hidden, layer_idx=0, kv_shape=shape)
    # Reverse the transform: transpose back, concat, compare to input.
    k_back = k.transpose(0, 2, 1, 3).reshape(1, 2, shape.n_kv_heads * shape.head_dim)
    v_back = v.transpose(0, 2, 1, 3).reshape(1, 2, shape.n_kv_heads * shape.head_dim)
    reconstructed = mx.concatenate([k_back, v_back], axis=-1)
    assert bool(mx.all(reconstructed == hidden).item())


# --- LinearKVProjector ------------------------------------------------------


def test_linear_projector_zero_init_returns_zeros():
    shape = KVShape(n_kv_heads=2, head_dim=4)
    proj = LinearKVProjector(num_layers=3, d_in=16, kv_shape=shape)
    hidden = mx.random.normal((1, 8, 16))
    k, v = proj.project(hidden, layer_idx=1, kv_shape=shape)
    assert k.shape == (1, 2, 8, 4)
    assert v.shape == (1, 2, 8, 4)
    assert float(mx.max(mx.abs(k)).item()) == 0.0
    assert float(mx.max(mx.abs(v)).item()) == 0.0


def test_linear_projector_load_weights_enforces_shape():
    shape = KVShape(n_kv_heads=2, head_dim=4)
    proj = LinearKVProjector(num_layers=3, d_in=16, kv_shape=shape)
    bad = mx.zeros((2, shape.channel_dim, 16))  # wrong num_layers
    with pytest.raises(ValueError):
        proj.load_weights(bad)
    good = mx.random.normal((3, shape.channel_dim, 16))
    proj.load_weights(good)
    assert bool(mx.all(proj.weights == good).item())


def test_linear_projector_rejects_out_of_range_layer():
    shape = KVShape(n_kv_heads=2, head_dim=4)
    proj = LinearKVProjector(num_layers=3, d_in=16, kv_shape=shape)
    hidden = mx.random.normal((1, 8, 16))
    with pytest.raises(ValueError):
        proj.project(hidden, layer_idx=3, kv_shape=shape)
    with pytest.raises(ValueError):
        proj.project(hidden, layer_idx=-1, kv_shape=shape)


def test_linear_projector_rejects_mismatched_shape():
    shape = KVShape(n_kv_heads=2, head_dim=4)
    proj = LinearKVProjector(num_layers=3, d_in=16, kv_shape=shape)
    hidden = mx.random.normal((1, 8, 16))
    other_shape = KVShape(n_kv_heads=4, head_dim=2)  # same channel_dim, still reject
    with pytest.raises(ValueError):
        proj.project(hidden, layer_idx=0, kv_shape=other_shape)


def test_linear_projector_with_trained_weights_reproduces_expected():
    """With a hand-crafted weight, verify projection is exact matmul."""
    shape = KVShape(n_kv_heads=1, head_dim=2)  # channel_dim = 4
    proj = LinearKVProjector(num_layers=1, d_in=3, kv_shape=shape)
    # w[0] @ hidden[b,l,:].T should produce [k0, k1, v0, v1]
    w = mx.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    proj.load_weights(w)
    hidden = mx.array([[[2.0, 3.0, 5.0]]])
    k, v = proj.project(hidden, layer_idx=0, kv_shape=shape)
    # out = hidden @ w[0].T = [2, 3, 5, 10]
    # K = first 2: [2, 3]; V = last 2: [5, 10]
    assert k.shape == (1, 1, 1, 2) and v.shape == (1, 1, 1, 2)
    assert float(k[0, 0, 0, 0].item()) == 2.0
    assert float(k[0, 0, 0, 1].item()) == 3.0
    assert float(v[0, 0, 0, 0].item()) == 5.0
    assert float(v[0, 0, 0, 1].item()) == 10.0


# --- ConfidenceGate ---------------------------------------------------------


def test_gate_proceeds_on_normal_hidden():
    gate = ConfidenceGate(min_norm=1.0, max_norm=100.0, max_outlier_fraction=0.05)
    # Hidden with per-token norm in [5, 10].
    hidden = mx.random.normal((1, 64, 32)) * 1.0  # expected norm ~sqrt(32) ~ 5.6
    decision = gate.evaluate(hidden)
    assert isinstance(decision, GateDecision)
    assert decision.proceed is True
    assert decision.reason == "in-distribution"


def test_gate_rejects_zero_hidden():
    gate = ConfidenceGate(min_norm=1.0, max_norm=100.0, max_outlier_fraction=0.0)
    hidden = mx.zeros((1, 32, 16))
    decision = gate.evaluate(hidden)
    assert decision.proceed is False
    assert "mean norm" in decision.reason
    assert decision.mean_norm == 0.0
    assert decision.outlier_fraction == 1.0


def test_gate_rejects_scaled_up_hidden():
    gate = ConfidenceGate(min_norm=1.0, max_norm=10.0, max_outlier_fraction=0.05)
    hidden = mx.ones((1, 16, 4)) * 100.0  # all tokens norm ~200
    decision = gate.evaluate(hidden)
    assert decision.proceed is False
    assert decision.mean_norm > 10.0


def test_gate_rejects_on_outlier_fraction():
    """Mean is fine but too many individual tokens are out-of-band."""
    gate = ConfidenceGate(min_norm=2.0, max_norm=6.0, max_outlier_fraction=0.1)
    # Build hidden where 30% of tokens have norm 20 (outside), 70% have norm 4.
    values = mx.array(
        [[20.0] * 3 + [4.0] * 7], dtype=mx.float32
    )  # (1, 10)
    hidden = mx.expand_dims(values, axis=-1)  # (1, 10, 1), norm == |value|
    decision = gate.evaluate(hidden)
    assert decision.proceed is False
    assert decision.outlier_fraction >= 0.1


def test_gate_rejects_bad_hidden_shape():
    gate = ConfidenceGate()
    with pytest.raises(ValueError):
        gate.evaluate(mx.zeros((16, 32)))


def test_gate_constructor_rejects_bad_args():
    with pytest.raises(ValueError):
        ConfidenceGate(min_norm=-1.0)
    with pytest.raises(ValueError):
        ConfidenceGate(min_norm=10.0, max_norm=5.0)
    with pytest.raises(ValueError):
        ConfidenceGate(max_outlier_fraction=1.5)


# --- HarvestRecorder --------------------------------------------------------


def test_harvest_recorder_rejects_bad_init():
    with pytest.raises(ValueError):
        HarvestRecorder(early_layer=-1, target_layers=[5, 6])
    with pytest.raises(ValueError):
        HarvestRecorder(early_layer=10, target_layers=[])
    with pytest.raises(ValueError):
        HarvestRecorder(early_layer=5, target_layers=[3, 6])  # target <= early


def test_harvest_recorder_shape_validation():
    rec = HarvestRecorder(early_layer=2, target_layers=[3, 4])
    hidden = mx.zeros((1, 8, 16))
    # Wrong number of KV pairs.
    with pytest.raises(ValueError):
        rec.record(
            sample_id="s",
            hidden_at_early=hidden,
            kvs_per_layer=[(mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4)))],
        )
    # Wrong hidden rank.
    with pytest.raises(ValueError):
        rec.record(
            sample_id="s",
            hidden_at_early=mx.zeros((16,)),
            kvs_per_layer=[
                (mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4))),
                (mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4))),
            ],
        )
    # Wrong seq dim on KV.
    with pytest.raises(ValueError):
        rec.record(
            sample_id="s",
            hidden_at_early=hidden,
            kvs_per_layer=[
                (mx.zeros((1, 2, 7, 4)), mx.zeros((1, 2, 8, 4))),
                (mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4))),
            ],
        )


def test_harvest_recorder_roundtrip(tmp_path: Path):
    rec = HarvestRecorder(
        early_layer=5, target_layers=[10, 11], out_dir=tmp_path, shard_name="test"
    )
    hidden_a = mx.random.normal((1, 16, 32))
    hidden_b = mx.random.normal((1, 24, 32))
    kvs_a = [
        (mx.random.normal((1, 2, 16, 8)), mx.random.normal((1, 2, 16, 8))),
        (mx.random.normal((1, 2, 16, 8)), mx.random.normal((1, 2, 16, 8))),
    ]
    kvs_b = [
        (mx.random.normal((1, 2, 24, 8)), mx.random.normal((1, 2, 24, 8))),
        (mx.random.normal((1, 2, 24, 8)), mx.random.normal((1, 2, 24, 8))),
    ]
    rec.record(sample_id="a", hidden_at_early=hidden_a, kvs_per_layer=kvs_a)
    rec.record(sample_id="b", hidden_at_early=hidden_b, kvs_per_layer=kvs_b)
    assert rec.sample_count() == 2

    path = rec.flush()
    assert path is not None
    assert path.exists()
    assert rec.sample_count() == 0  # cleared after flush

    arrays, meta = load_shard(path)
    assert meta["sample_count"] == "2"
    assert meta["early_layer"] == "5"
    assert meta["target_layers"] == "10,11"
    assert meta["sample_ids"] == "a,b"
    assert "s0/hidden" in arrays
    assert "s0/layer10/K" in arrays
    assert "s1/layer11/V" in arrays
    assert bool(mx.all(arrays["s0/hidden"] == hidden_a).item())
    assert bool(mx.all(arrays["s1/layer11/V"] == kvs_b[1][1]).item())


def test_harvest_recorder_flush_noop_when_empty(tmp_path: Path):
    rec = HarvestRecorder(early_layer=5, target_layers=[6], out_dir=tmp_path)
    assert rec.flush() is None


# --- sp_prefill control flow -------------------------------------------------


def test_sp_prefill_forced_fallback_calls_fallback_fn():
    called = {"n": 0}

    def fallback_fn():
        called["n"] += 1
        return {"result": 42}

    result = sp_prefill(
        target_model=None,
        input_ids=mx.zeros((1, 64), dtype=mx.uint32),
        projector=IdentityKVProjector(),
        gate=ConfidenceGate(),
        kv_shapes=[KVShape(n_kv_heads=4, head_dim=8)],
        early_layer=4,
        fallback_fn=fallback_fn,
        force_fallback=True,
    )
    assert result.fallback is True
    assert result.fallback_value == {"result": 42}
    assert result.stats["path"] == "forced_fallback"
    assert called["n"] == 1


def test_sp_prefill_not_implemented_takes_fallback_path():
    """Scaffold: _run_partial_target_prefill raises → fallback."""
    called = {"n": 0}

    def fallback_fn():
        called["n"] += 1
        return ["ok"]

    result = sp_prefill(
        target_model=None,
        input_ids=mx.zeros((1, 32), dtype=mx.uint32),
        projector=IdentityKVProjector(),
        gate=ConfidenceGate(),
        kv_shapes=[KVShape(n_kv_heads=2, head_dim=4)],
        early_layer=2,
        fallback_fn=fallback_fn,
    )
    assert result.fallback is True
    assert result.fallback_value == ["ok"]
    assert result.stats["path"] == "fallback"
    assert "NotImplemented" in result.stats.get("reason", "") or \
           "partial target prefill" in result.stats.get("reason", "").lower()
    assert called["n"] == 1


def test_sp_prefill_rejects_bad_args():
    with pytest.raises(ValueError):
        sp_prefill(
            target_model=None,
            input_ids=mx.zeros((1, 32), dtype=mx.uint32),
            projector=IdentityKVProjector(),
            gate=ConfidenceGate(),
            kv_shapes=[KVShape(n_kv_heads=2, head_dim=4)],
            early_layer=0,  # must be positive
            fallback_fn=lambda: None,
        )
    with pytest.raises(ValueError):
        sp_prefill(
            target_model=None,
            input_ids=mx.zeros((1, 32), dtype=mx.uint32),
            projector=IdentityKVProjector(),
            gate=ConfidenceGate(),
            kv_shapes=[],  # must be non-empty
            early_layer=4,
            fallback_fn=lambda: None,
        )
    with pytest.raises(ValueError):
        sp_prefill(
            target_model=None,
            input_ids=mx.zeros((32,), dtype=mx.uint32),  # must be (B, L)
            projector=IdentityKVProjector(),
            gate=ConfidenceGate(),
            kv_shapes=[KVShape(n_kv_heads=2, head_dim=4)],
            early_layer=4,
            fallback_fn=lambda: None,
        )


def test_sp_prefill_gate_reject_takes_fallback(monkeypatch):
    """When partial prefill returns low-norm hidden, gate rejects → fallback."""
    import mio.draft_kv.runtime as rt

    def fake_partial(*, target_model, input_ids, early_layer):
        # Return a zeroed hidden so the gate rejects.
        return mx.zeros((1, 16, 8)), []

    monkeypatch.setattr(rt, "_run_partial_target_prefill", fake_partial)

    called = {"n": 0}

    def fallback_fn():
        called["n"] += 1
        return "fb"

    result = sp_prefill(
        target_model=None,
        input_ids=mx.zeros((1, 16), dtype=mx.uint32),
        projector=IdentityKVProjector(),
        gate=ConfidenceGate(min_norm=1.0, max_norm=100.0),
        kv_shapes=[KVShape(n_kv_heads=2, head_dim=4)],
        early_layer=2,
        fallback_fn=fallback_fn,
    )
    assert result.fallback is True
    assert result.fallback_value == "fb"
    assert result.stats["path"] == "gate_reject"
    assert called["n"] == 1


def test_sp_prefill_gate_pass_invokes_projector(monkeypatch):
    """When hidden is in-distribution, projector runs for every late layer."""
    import mio.draft_kv.runtime as rt

    shape = KVShape(n_kv_heads=2, head_dim=4)
    hidden = mx.ones((1, 8, shape.channel_dim)) * 3.0  # per-token norm ~ OK

    def fake_partial(*, target_model, input_ids, early_layer):
        return hidden, ["cache0", "cache1"]  # 2 early caches as sentinels

    monkeypatch.setattr(rt, "_run_partial_target_prefill", fake_partial)

    projector_calls = {"n": 0, "seen_layers": []}

    class CountingProjector:
        def project(self, hidden, *, layer_idx, kv_shape):
            projector_calls["n"] += 1
            projector_calls["seen_layers"].append(layer_idx)
            B, L, _ = hidden.shape
            k = mx.zeros((B, kv_shape.n_kv_heads, L, kv_shape.head_dim))
            v = mx.zeros((B, kv_shape.n_kv_heads, L, kv_shape.head_dim))
            return k, v

    result = sp_prefill(
        target_model=None,
        input_ids=mx.zeros((1, 8), dtype=mx.uint32),
        projector=CountingProjector(),
        gate=ConfidenceGate(min_norm=1.0, max_norm=100.0),
        kv_shapes=[shape, shape, shape],  # 3 late layers
        early_layer=2,
        fallback_fn=lambda: None,
    )
    assert result.fallback is False
    assert result.gate is not None and result.gate.proceed is True
    assert projector_calls["n"] == 3
    # Late layers should be early_layer + [0, 1, 2] = [2, 3, 4]
    assert projector_calls["seen_layers"] == [2, 3, 4]
    # caches = early_partial (2) + late_projected (3) = 5
    assert result.caches is not None and len(result.caches) == 5
    assert result.stats["path"] == "sp"


# --- probe: ridge regression math -------------------------------------------


def _make_synthetic_shard(
    tmp_path: Path,
    *,
    early_layer: int,
    target_layers: list[int],
    n_samples: int,
    seq_len: int,
    d_in: int,
    n_heads: int,
    head_dim: int,
    relation: str,
) -> Path:
    """Build a HarvestRecorder shard where (hidden, KV) follows a known
    relation. `relation` is one of:
        "exact_linear": KV = hidden @ W_l for a deterministic W per layer.
        "random":       KV is independent random; expect R^2 near 0.
    """
    rec = HarvestRecorder(
        early_layer=early_layer,
        target_layers=target_layers,
        out_dir=tmp_path,
        shard_name=f"synth_{relation}_e{early_layer}_t{'-'.join(str(x) for x in target_layers)}",
    )
    d_out = n_heads * head_dim
    mx.random.seed(42)
    weights_per_layer = {
        layer: mx.random.normal((d_in, d_out)) for layer in target_layers
    }

    for s in range(n_samples):
        hidden = mx.random.normal((1, seq_len, d_in))
        kvs: list[tuple[mx.array, mx.array]] = []
        for layer in target_layers:
            if relation == "exact_linear":
                k_flat = hidden.reshape(seq_len, d_in) @ weights_per_layer[layer]
                v_flat = (
                    hidden.reshape(seq_len, d_in) @ weights_per_layer[layer] * 0.5
                )
            elif relation == "random":
                k_flat = mx.random.normal((seq_len, d_out))
                v_flat = mx.random.normal((seq_len, d_out))
            else:
                raise ValueError(relation)
            k = k_flat.reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            v = v_flat.reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            kvs.append((k, v))
        rec.record(sample_id=f"s{s}", hidden_at_early=hidden, kvs_per_layer=kvs)
    path = rec.flush()
    assert path is not None
    return path


def test_probe_perfect_linear_relation_gives_high_r2(tmp_path: Path):
    from mio.draft_kv.probe import run_probe
    path = _make_synthetic_shard(
        tmp_path,
        early_layer=3, target_layers=[4, 5],
        n_samples=8, seq_len=32, d_in=16, n_heads=2, head_dim=8,
        relation="exact_linear",
    )
    report = run_probe([path], ridge_lambda=1e-6)
    assert len(report.layers) == 2
    for layer in report.layers:
        # Tiny ridge → nearly exact fit on well-conditioned data.
        assert layer.k_r2 > 0.99, f"layer {layer.layer} K R² = {layer.k_r2}"
        assert layer.v_r2 > 0.99, f"layer {layer.layer} V R² = {layer.v_r2}"


def test_probe_random_targets_give_low_r2(tmp_path: Path):
    """When K/V is independent of hidden, R² on in-sample data is bounded
    by the ridge-regression over-fit. With enough tokens > d_in, the
    effective R^2 stays low (< 0.4) even without a held-out set."""
    from mio.draft_kv.probe import run_probe
    path = _make_synthetic_shard(
        tmp_path,
        early_layer=2, target_layers=[3],
        n_samples=16, seq_len=256, d_in=32, n_heads=4, head_dim=16,
        relation="random",
    )
    report = run_probe([path], ridge_lambda=1e-2)
    assert len(report.layers) == 1
    layer = report.layers[0]
    # In-sample R² grows with ratio d_in / n_tokens. With n_tokens=4096,
    # d_in=32, ridge λ=0.01: expected R² is very small. Give a generous
    # bound to avoid flakiness on different MLX builds.
    assert layer.k_r2 < 0.4, f"expected low R², got {layer.k_r2}"
    assert layer.v_r2 < 0.4, f"expected low R², got {layer.v_r2}"


def test_probe_report_summary_is_human_readable(tmp_path: Path):
    from mio.draft_kv.probe import run_probe
    path = _make_synthetic_shard(
        tmp_path,
        early_layer=1, target_layers=[2, 3, 4],
        n_samples=4, seq_len=16, d_in=8, n_heads=1, head_dim=4,
        relation="exact_linear",
    )
    report = run_probe([path])
    summary = report.summary()
    # Spot-check that summary contains expected headers + per-layer rows.
    assert "K R²" in summary and "V R²" in summary
    for l in report.layers:
        assert str(l.layer) in summary
    assert "mean" in summary


def test_probe_rejects_mixed_schemas(tmp_path: Path):
    from mio.draft_kv.probe import run_probe
    p1 = _make_synthetic_shard(
        tmp_path,
        early_layer=2, target_layers=[3, 4],
        n_samples=2, seq_len=8, d_in=4, n_heads=1, head_dim=4,
        relation="exact_linear",
    )
    p2 = _make_synthetic_shard(
        tmp_path,
        early_layer=5, target_layers=[6, 7],  # different early_layer
        n_samples=2, seq_len=8, d_in=4, n_heads=1, head_dim=4,
        relation="exact_linear",
    )
    with pytest.raises(ValueError):
        run_probe([p1, p2])


def test_probe_honors_layer_filter(tmp_path: Path):
    from mio.draft_kv.probe import run_probe
    path = _make_synthetic_shard(
        tmp_path,
        early_layer=1, target_layers=[2, 3, 4, 5],
        n_samples=2, seq_len=8, d_in=4, n_heads=1, head_dim=4,
        relation="exact_linear",
    )
    report = run_probe([path], layers=[3])
    assert [l.layer for l in report.layers] == [3]
