"""Crash-safety regressions for Mio's JSON state writer."""

from __future__ import annotations

import json
import os

import pytest

from mio import persistence


def test_atomic_json_write_round_trips_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "state.json"
    persistence.atomic_write_json(path, {"name": "Mio", "items": [1, 2]})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "name": "Mio",
        "items": [1, 2],
    }
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_serialization_failure_preserves_previous_document(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"stable": true}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        persistence.atomic_write_json(path, {"invalid": object()})

    assert path.read_text(encoding="utf-8") == '{"stable": true}\n'


def test_publish_failure_preserves_previous_document_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"version": 1}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        persistence.atomic_write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_writer_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int):
        calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(persistence.os, "fsync", recording_fsync)
    persistence.atomic_write_json(tmp_path / "state.json", {"ok": True})

    assert len(calls) >= 2
