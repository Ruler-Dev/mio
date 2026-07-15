"""Transactional persistence regressions for the disk KV cache."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mio.cache_store import CacheStore


@pytest.fixture
def fake_mlx_io(monkeypatch):
    def save(path: str, **arrays) -> None:
        Path(path).write_text(json.dumps(arrays, sort_keys=True), encoding="utf-8")

    def load(path: str):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(mx, "savez", save)
    monkeypatch.setattr(mx, "load", load)
    return save


def _layers(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(keys=f"keys-{index}", values=f"values-{index}")
        for index in range(count)
    ]


def test_cache_save_publishes_verified_manifest_and_replaces_shorter_generation(
    tmp_path,
    fake_mlx_io,
):
    store = CacheStore(tmp_path, model_id="mlx-community/model-a")
    store.save("conversation", _layers(3), 128)

    key = store.make_key("conversation")
    first_record = dict(store._index[key])
    first_dir = tmp_path / first_record["generation"]
    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "cache_key_hash": key,
        "created": manifest["created"],
        "layer_count": 3,
        "layers": ["layer_000.npz", "layer_001.npz", "layer_002.npz"],
        "model": "mlx-community/model-a",
        "token_count": 128,
        "version": 1,
    }
    arrays, tokens = store.load("conversation")
    assert tokens == 128
    assert arrays is not None and len(arrays) == 3

    store.save("conversation", _layers(1), 64)
    second_record = store._index[key]
    second_dir = tmp_path / second_record["generation"]
    assert second_dir != first_dir
    assert not first_dir.exists()
    assert {path.name for path in second_dir.iterdir()} == {
        "manifest.json",
        "layer_000.npz",
    }
    arrays, tokens = store.load("conversation")
    assert tokens == 64
    assert arrays is not None and len(arrays) == 1


def test_failed_layer_write_preserves_previous_complete_generation(
    tmp_path,
    monkeypatch,
    fake_mlx_io,
):
    store = CacheStore(tmp_path, model_id="model-a")
    store.save("conversation", _layers(2), 20)
    key = store.make_key("conversation")
    previous_record = dict(store._index[key])
    previous_dir = tmp_path / previous_record["generation"]

    writes = 0

    def fail_second(path: str, **arrays) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        Path(path).write_text(json.dumps(arrays), encoding="utf-8")

    monkeypatch.setattr(mx, "savez", fail_second)
    store.save("conversation", _layers(3), 99)

    assert store._index[key] == previous_record
    assert previous_dir.is_dir()
    arrays, tokens = store.load("conversation")
    assert tokens == 20
    assert arrays is not None and len(arrays) == 2


def test_failed_index_publication_rolls_back_new_generation(
    tmp_path,
    monkeypatch,
    fake_mlx_io,
):
    store = CacheStore(tmp_path, model_id="model-a")
    store.save("conversation", _layers(2), 20)
    key = store.make_key("conversation")
    previous_record = dict(store._index[key])

    monkeypatch.setattr(
        store,
        "_save_index",
        lambda: (_ for _ in ()).throw(OSError("index unavailable")),
    )
    store.save("conversation", _layers(1), 40)

    assert store._index[key] == previous_record
    generations = [path.name for path in tmp_path.glob(f"{key}-*")]
    assert generations == [previous_record["generation"]]


def test_model_mismatch_and_stale_layer_are_never_restored(tmp_path, fake_mlx_io):
    store = CacheStore(tmp_path, model_id="model-a")
    store.save("conversation", _layers(2), 20)

    assert store.load("conversation", model_id="model-b") == (None, 0)
    arrays, _tokens = store.load("conversation", model_id="model-a")
    assert arrays is not None

    key = store.make_key("conversation")
    entry_dir = tmp_path / store._index[key]["generation"]
    (entry_dir / "layer_999.npz").write_text("stale", encoding="utf-8")

    assert not store.has("conversation")
    assert store.load("conversation") == (None, 0)


def test_concurrent_replacements_publish_one_complete_generation(tmp_path, fake_mlx_io):
    store = CacheStore(tmp_path, model_id="model-a")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda count: store.save(
                    "conversation",
                    _layers(count),
                    count * 10,
                ),
                [1, 2, 3, 4],
            )
        )

    key = store.make_key("conversation")
    record = store._index[key]
    entry_dir = tmp_path / record["generation"]
    manifest = json.loads((entry_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays, tokens = store.load("conversation")

    assert store.has("conversation")
    assert arrays is not None and len(arrays) == manifest["layer_count"]
    assert tokens == manifest["token_count"]
    assert [path.name for path in tmp_path.glob(f"{key}-*")] == [record["generation"]]
