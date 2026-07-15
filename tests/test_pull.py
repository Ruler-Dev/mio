"""Model-stack pull tests, including independent DSpark completeness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mio.models.registry import ModelEntry
from mio import pull


def _entry() -> ModelEntry:
    return ModelEntry(
        target_repo="org/target",
        target_local="target",
        draft_repo="org/dflash",
        draft_local="dflash",
        dspark_repo="org/dspark",
        dspark_local="dspark",
        adapter="qwen3_5",
        default_tier="large",
        context_window=4096,
        max_output_tokens=128,
    )


def _fake_pull(monkeypatch, tmp_path, *, initially_complete=()):
    complete = {str(path) for path in initially_complete}
    calls = []
    models = tmp_path / "models"
    drafts = tmp_path / "spd"
    monkeypatch.setattr(pull, "KNOWN_MODELS", {"stack": _entry()})
    monkeypatch.setattr(pull, "models_dir", lambda: models)
    monkeypatch.setattr(pull, "spd_dir", lambda: drafts)
    monkeypatch.setattr(
        pull,
        "_model_path_is_complete",
        lambda path, **_kwargs: str(path) in complete,
    )

    def snapshot_download(*, repo_id, local_dir):
        calls.append((repo_id, local_dir))
        complete.add(str(local_dir))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return models, drafts, calls


def test_pull_defaults_to_target_dspark_and_dflash(monkeypatch, tmp_path):
    models, drafts, calls = _fake_pull(monkeypatch, tmp_path)

    assert pull.pull_model("stack") is True

    assert calls == [
        ("org/target", str(models / "target")),
        ("org/dspark", str(drafts / "dspark")),
        ("org/dflash", str(drafts / "dflash")),
    ]


def test_pull_tracks_each_checkpoint_completeness_independently(monkeypatch, tmp_path):
    target = tmp_path / "models" / "target"
    dspark = tmp_path / "spd" / "dspark"
    _models, drafts, calls = _fake_pull(
        monkeypatch,
        tmp_path,
        initially_complete=(target, dspark),
    )

    assert pull.pull_model("stack") is True

    assert calls == [("org/dflash", str(drafts / "dflash"))]


def test_pull_can_skip_dspark_or_fallback(monkeypatch, tmp_path):
    models, drafts, calls = _fake_pull(monkeypatch, tmp_path)

    assert pull.pull_model("stack", include_dspark=False) is True
    assert calls == [
        ("org/target", str(models / "target")),
        ("org/dflash", str(drafts / "dflash")),
    ]

    calls.clear()
    # Start with a fresh completeness view for the strict-DSpark shape.
    models, drafts, calls = _fake_pull(monkeypatch, tmp_path / "strict")
    assert pull.pull_model("stack", include_fallback=False) is True
    assert calls == [
        ("org/target", str(models / "target")),
        ("org/dspark", str(drafts / "dspark")),
    ]


def test_pull_reports_failure_when_one_component_stays_incomplete(monkeypatch, tmp_path):
    complete = set()
    models = tmp_path / "models"
    drafts = tmp_path / "spd"
    monkeypatch.setattr(pull, "KNOWN_MODELS", {"stack": _entry()})
    monkeypatch.setattr(pull, "models_dir", lambda: models)
    monkeypatch.setattr(pull, "spd_dir", lambda: drafts)
    monkeypatch.setattr(
        pull,
        "_model_path_is_complete",
        lambda path, **_kwargs: str(path) in complete,
    )

    def snapshot_download(*, repo_id, local_dir):
        if repo_id != "org/dspark":
            complete.add(str(local_dir))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    assert pull.pull_model("stack") is False


def test_pull_cli_exits_nonzero_when_any_requested_component_fails(monkeypatch):
    from mio.main import _cmd_pull

    monkeypatch.setattr(pull, "pull_model", lambda *_args, **_kwargs: False)

    with pytest.raises(SystemExit) as error:
        _cmd_pull(
            SimpleNamespace(
                model_key="stack",
                no_dspark=False,
                no_fallback=False,
            )
        )

    assert error.value.code == 1


def test_snapshot_target_rejects_weights_without_tokenizer_assets(tmp_path):
    target = tmp_path / "target"

    def weights_only(*, repo_id, local_dir):
        assert repo_id == "org/target"
        path = Path(local_dir)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")

    assert (
        pull._snapshot_into(
            "org/target",
            target,
            "target",
            False,
            weights_only,
            require_tokenizer=True,
        )
        is False
    )

    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "chat_template.jinja").write_text(
        "{{ messages }}",
        encoding="utf-8",
    )
    assert (
        pull._snapshot_into(
            "org/target",
            target,
            "target",
            False,
            weights_only,
            require_tokenizer=True,
        )
        is True
    )
