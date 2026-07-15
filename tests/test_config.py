"""Configuration persistence and cache-mode regression tests."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from types import SimpleNamespace

import huggingface_hub

from mio.config import MioConfig, TierConfig, default_config_path, load_config, save_config
from mio.configure import _estimate_kv_cache_gb, _resolve_tq_selection
from mio.main import _apply_tq4_flag, _cmd_serve
from mio.model_check import _check_hf_cache, _model_status
from mio.models.registry import DEFAULT_TIERS


def test_default_server_is_loopback_only():
    assert MioConfig.default().host == "127.0.0.1"


def test_config_path_and_directory_honor_mio_home(monkeypatch, tmp_path):
    home = tmp_path / "custom-mio-home"
    monkeypatch.setenv("MIO_HOME", str(home))

    assert default_config_path() == home / "config.json"
    assert MioConfig.default().config_dir == home


def test_default_configs_do_not_share_mutable_tiers():
    first = MioConfig.default()
    first.tiers["small"].tq_bits = 4

    second = MioConfig.default()
    assert second.tiers["small"].tq_bits == 16
    assert DEFAULT_TIERS["small"].tq_bits == 16


def test_config_round_trip_persists_tiers_and_top_level(tmp_path):
    path = tmp_path / "nested" / "config.json"
    config = MioConfig.default()
    config.active_tiers = ["small"]
    config.tandem = True
    config.port = 9191
    config.host = "127.0.0.1"
    config.tiers["small"].context_window = 65536
    config.tiers["small"].temperature = 0.7
    config.tiers["small"].drafter_backend = "dspark"
    config.tiers["small"].draft_fallback_model = "example/fallback"
    config.tiers["small"].drafter_strict = True
    config.tiers["small"].dspark_max_draft_tokens = 3
    config.tiers["small"].dspark_lookup_drafts = False
    config.tiers["small"].dspark_prefix_cache = False
    config.tiers["custom"] = TierConfig(
        name="custom",
        target_model="example/target",
        draft_model="example/draft",
        context_window=4096,
        max_output_tokens=1024,
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded.active_tiers == ["small"]
    assert loaded.tandem is True
    assert loaded.port == 9191
    assert loaded.host == "127.0.0.1"
    assert loaded.config_dir == path.parent
    assert loaded.tiers["small"].context_window == 65536
    assert loaded.tiers["small"].temperature == 0.7
    assert loaded.tiers["small"].drafter_backend == "dspark"
    assert loaded.tiers["small"].draft_fallback_model == "example/fallback"
    assert loaded.tiers["small"].drafter_strict is True
    assert loaded.tiers["small"].dspark_max_draft_tokens == 3
    assert loaded.tiers["small"].dspark_lookup_drafts is False
    assert loaded.tiers["small"].dspark_prefix_cache is False
    assert loaded.tiers["custom"].target_model == "example/target"


def test_load_config_reads_canonical_path_without_argument(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 9192}), encoding="utf-8")
    monkeypatch.setattr("mio.config.default_config_path", lambda: path)

    assert load_config().port == 9192


def test_load_config_migrates_legacy_conflicting_cache_modes(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"tiers": {"small": {"tq_bits": 4, "pq_bits": 4}}}),
        encoding="utf-8",
    )

    tier = load_config(path).tiers["small"]
    assert tier.tq_bits == 4
    assert tier.pq_bits == 16


def test_load_config_falls_back_on_malformed_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{broken", encoding="utf-8")

    config = load_config(path)
    assert config.port == 9090
    assert "large-moe" in config.tiers


def test_load_config_ignores_unknown_active_tiers(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"active_tiers": ["removed-tier"]}), encoding="utf-8")

    assert load_config(path).active_tiers == ["large-moe"]


def test_serve_uses_persisted_values_when_cli_does_not_override(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    config = MioConfig.default()
    config.port = 9193
    config.host = "127.0.0.1"
    config.active_tiers = ["small"]
    config.tiers["small"].context_window = 32768
    save_config(config, path)
    monkeypatch.setattr("mio.config.default_config_path", lambda: path)

    captured = {}

    class FakeManager:
        def __init__(self, loaded_config):
            captured["config"] = loaded_config

        def load_active_tiers(self):
            captured["loaded"] = True

    def fake_start_server(manager, **kwargs):
        captured["server"] = kwargs

    manager_module = ModuleType("mio.model_manager")
    manager_module.ModelManager = FakeManager
    server_module = ModuleType("mio.server")
    server_module.start_server = fake_start_server
    monkeypatch.setitem(sys.modules, "mio.model_manager", manager_module)
    monkeypatch.setitem(sys.modules, "mio.server", server_module)

    args = SimpleNamespace(
        port=None,
        host=None,
        tandem=False,
        tiers=None,
        tier=None,
        tq4=False,
        mpath=1,
        context=None,
        validate=False,
        caveman="full",
        compact_threshold=0.75,
        compact_target=0.5,
        no_compact_summarize=False,
        webui=False,
    )
    _cmd_serve(args)

    loaded = captured["config"]
    assert captured["loaded"] is True
    assert loaded.port == 9193
    assert loaded.host == "127.0.0.1"
    assert loaded.active_tiers == ["small"]
    assert loaded.tiers["small"].context_window == 32768
    assert captured["server"]["port"] == 9193
    assert captured["server"]["unsafe_remote_bind"] is False
    assert captured["server"]["replace_existing"] is False


def test_tq4_flag_disables_polarquant():
    config = MioConfig.default()
    tier = config.tiers["small"]
    assert tier.pq_bits == 4

    _apply_tq4_flag(config, ["small"], enabled=True)

    assert tier.tq_bits == 4
    assert tier.pq_bits == 16
    assert tier.tq_use_rotation is True
    assert tier.tq_use_normalization is True


def test_wizard_off_uses_canonical_uncompressed_cache():
    assert _resolve_tq_selection(16) == (16, 16, False)
    assert _resolve_tq_selection(0) == (16, 16, False)
    assert _resolve_tq_selection(4) == (4, 16, True)

    arch = {"num_hidden_layers": 4, "num_key_value_heads": 2, "head_dim": 64}
    assert _estimate_kv_cache_gb(arch, 8192, 16) == _estimate_kv_cache_gb(arch, 8192, 0)


def test_model_status_marks_config_only_directory_incomplete(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    status, display, ready = _model_status(str(tmp_path), "target")
    assert status == "[red]INCOMPLETE[/red]"
    assert display == str(tmp_path)
    assert ready is False

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    status, _, ready = _model_status(str(tmp_path), "target")
    assert status == "[red]INCOMPLETE[/red]"
    assert ready is False

    (tmp_path / "tokenizer.model").write_bytes(b"tokenizer")
    (tmp_path / "chat_template.jinja").write_text(
        "{{ messages }}",
        encoding="utf-8",
    )
    status, _, ready = _model_status(str(tmp_path), "target")
    assert status == "[green]LOCAL[/green]"
    assert ready is True


def test_hf_cache_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    cache = SimpleNamespace(
        repos=[
            SimpleNamespace(
                repo_id="example/model",
                revisions=[SimpleNamespace(snapshot_path=tmp_path)],
            )
        ]
    )
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: cache)

    assert _check_hf_cache("example/model") == (False, "")

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert _check_hf_cache("example/model") == (False, "")

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}),
        encoding="utf-8",
    )
    assert _check_hf_cache("example/model") == (True, str(tmp_path))
