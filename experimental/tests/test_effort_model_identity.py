from __future__ import annotations

from pathlib import Path

import pytest

from experimental.effort.model_identity import (
    LOCAL_MODEL_REVISION_PREFIX,
    ModelIdentityError,
    fingerprint_local_model,
    main,
    resolve_model_reference,
)


def _write_model(root: Path, *, reverse: bool = False) -> None:
    root.mkdir()
    rows = [
        ("config.json", b'{"model_type":"test"}\n'),
        ("model.safetensors", b"weights-v1"),
        ("tokenizer.json", b'{"version":"1"}\n'),
        ("tokenizer_config.json", b'{"chat_template":"test"}\n'),
        ("merges.txt", b"a b\n"),
        ("README.md", b"not loaded by MLX\n"),
    ]
    for relative_path, payload in reversed(rows) if reverse else rows:
        (root / relative_path).write_bytes(payload)


def test_local_fingerprint_is_path_and_creation_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_model(first)
    _write_model(second, reverse=True)

    left = fingerprint_local_model(first)
    right = fingerprint_local_model(second)

    assert left.digest == right.digest
    assert left.revision == f"{LOCAL_MODEL_REVISION_PREFIX}{left.digest}"
    assert [row.relative_path for row in left.files] == sorted(
        row.relative_path for row in left.files
    )
    assert "README.md" not in {row.relative_path for row in left.files}


def test_weight_or_tokenizer_mutation_changes_fingerprint(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_model(model)
    baseline = fingerprint_local_model(model)

    (model / "model.safetensors").write_bytes(b"weights-v2")
    weights_changed = fingerprint_local_model(model)
    assert weights_changed.digest != baseline.digest

    (model / "model.safetensors").write_bytes(b"weights-v1")
    (model / "tokenizer.json").write_bytes(b'{"version":"2"}\n')
    tokenizer_changed = fingerprint_local_model(model)
    assert tokenizer_changed.digest != baseline.digest


def test_local_resolution_requires_exact_content_digest(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_model(model)
    fingerprint = fingerprint_local_model(model)

    resolved = resolve_model_reference(model.as_posix(), fingerprint.revision)

    assert resolved.source_kind == "local"
    assert resolved.load_model_id == model.resolve().as_posix()
    assert resolved.load_revision is None
    assert resolved.canonical_model_id == f"local-mlx@{fingerprint.revision}"

    with pytest.raises(ModelIdentityError, match="does not match"):
        resolve_model_reference(
            model.as_posix(),
            f"{LOCAL_MODEL_REVISION_PREFIX}{'0' * 64}",
        )
    with pytest.raises(ModelIdentityError, match="must use"):
        resolve_model_reference(model.as_posix(), "main")
    with pytest.raises(ModelIdentityError, match="malformed"):
        resolve_model_reference(model.as_posix(), f"{LOCAL_MODEL_REVISION_PREFIX}bad")


def test_local_fingerprint_rejects_symlinks_and_missing_core_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_model(model)
    (model / "linked.json").symlink_to(model / "config.json")
    with pytest.raises(ModelIdentityError, match="symlinked file"):
        fingerprint_local_model(model)

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ModelIdentityError, match="config.json"):
        fingerprint_local_model(missing)

    with pytest.raises(ModelIdentityError, match="does not exist"):
        fingerprint_local_model(tmp_path / "not-there")


def test_remote_resolution_requires_full_commit_without_network() -> None:
    revision = "a" * 40
    resolved = resolve_model_reference("org/model", revision)
    assert resolved.source_kind == "huggingface"
    assert resolved.canonical_model_id == f"hf://org/model@{revision}"
    assert resolved.load_model_id == "org/model"
    assert resolved.load_revision == revision

    with pytest.raises(ModelIdentityError, match="immutable"):
        resolve_model_reference("org/model", "main")
    with pytest.raises(ModelIdentityError, match="org/repository"):
        resolve_model_reference("not-a-repository", revision)


def test_cli_prints_revision_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "model"
    _write_model(model)

    assert main(["--model", str(model)]) == 0

    output = capsys.readouterr().out.strip()
    assert output == fingerprint_local_model(model).revision
