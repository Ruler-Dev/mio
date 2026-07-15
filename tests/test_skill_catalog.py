from __future__ import annotations

import json
from pathlib import Path

import pytest

from mio.skill_catalog import (
    MANIFEST_NAME,
    MioSkillCatalog,
    SkillExecutionDisabled,
    SkillNotFoundError,
    SkillSource,
    SkillValidationError,
    install_skill_sources_from_checkouts,
    list_mio_skills,
    parse_skill,
    read_mio_skill,
    skills_root,
)


def _write_skill(
    directory: Path,
    name: str,
    *,
    description: str = "Useful test instructions",
    tags: tuple[str, ...] = (),
    script: str | None = None,
    body: str = "# Instructions\n\nDo the useful thing.",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = ["---", f"name: {name}", f"description: {description}"]
    if tags:
        frontmatter.append("tags: [" + ", ".join(tags) + "]")
    if script:
        frontmatter.append(f"script: {script}")
    frontmatter.extend(["---", "", body, ""])
    (directory / "SKILL.md").write_text("\n".join(frontmatter), encoding="utf-8")
    return directory


def _source(
    source_id: str,
    *,
    alias: str,
    revision: str,
    expected: int,
    exclude: tuple[str, ...] = (),
) -> SkillSource:
    return SkillSource(
        source_id=source_id,
        repository=f"https://example.test/{source_id}.git",
        revision=revision,
        include_prefixes=("skills/",),
        exclude_prefixes=exclude,
        alias_prefix=alias,
        expected_skills=expected,
    )


def test_mio_home_catalog_discovers_instruction_only_skills(monkeypatch, tmp_path):
    home = tmp_path / "mio-home"
    _write_skill(home / "skills" / "design-audit", "design-audit", tags=("design", "audit"))
    monkeypatch.setenv("MIO_HOME", str(home))

    assert skills_root() == home / "skills"
    result = list_mio_skills(query="audit", tag="design", limit=10)
    assert result["matched"] == 1
    assert result["skills"][0]["name"] == "design-audit"
    assert result["skills"][0]["kind"] == "instruction"

    document = read_mio_skill("design-audit", max_chars=24)
    assert document["content"].startswith("---\nname: design-audit")
    assert document["truncated"] is True


def test_catalog_keeps_legacy_local_snake_case_names(monkeypatch, tmp_path):
    home = tmp_path / "mio-home"
    _write_skill(home / "skills" / "hello-world", "hello_world")
    monkeypatch.setenv("MIO_HOME", str(home))

    result = list_mio_skills(query="hello_world")
    assert result["matched"] == 1
    assert result["skills"][0]["name"] == "hello-world"
    assert result["skills"][0]["canonical_name"] == "hello_world"
    assert read_mio_skill("hello_world")["skill"]["name"] == "hello-world"


def test_catalog_filters_source_and_never_accepts_path_traversal(tmp_path):
    root = tmp_path / "skills"
    skill = _write_skill(root / "secure-review", "secure-review", tags=("security",))
    manifest = {
        "schema_version": 1,
        "sources": [],
        "skills": [
            {
                "installed_name": "secure-review",
                "canonical_name": "secure-review",
                "source_id": "cyber",
                "source_url": "https://example.test/cyber.git",
                "source_revision": "a" * 40,
                "source_path": "skills/secure-review",
                "digest": "",
                "script": None,
                "execution_enabled": False,
            }
        ],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    catalog = MioSkillCatalog(root)
    assert [item.installed_name for item in catalog.search(source="cyber")] == ["secure-review"]
    assert catalog.search(source="game") == []
    with pytest.raises(SkillNotFoundError):
        catalog.read("../secure-review")
    assert skill.is_dir()


def test_frontmatter_validation_and_run_py_is_not_implicit(tmp_path):
    instruction = _write_skill(tmp_path / "instruction", "instruction")
    (instruction / "run.py").write_text("print('must not run')\n", encoding="utf-8")
    metadata, _ = parse_skill(instruction)
    assert metadata.script is None

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    with pytest.raises(SkillValidationError, match="frontmatter"):
        parse_skill(malformed)

    unsafe = _write_skill(tmp_path / "unsafe", "unsafe", script="../run.py")
    with pytest.raises(SkillValidationError, match="unsafe script path"):
        parse_skill(unsafe)


def test_installer_pins_provenance_excludes_inactive_and_aliases_collisions(tmp_path):
    checkout_a = tmp_path / "source-a"
    checkout_b = tmp_path / "source-b"
    _write_skill(checkout_a / "skills" / "common", "common", tags=("alpha",))
    _write_skill(checkout_a / "skills" / "only-a", "only-a")
    _write_skill(checkout_a / "skills" / "deprecated" / "old", "old")
    _write_skill(checkout_b / "skills" / "common", "common", tags=("beta",))

    source_a = _source(
        "source-a",
        alias="a",
        revision="a" * 40,
        expected=2,
        exclude=("skills/deprecated/",),
    )
    source_b = _source("source-b", alias="b", revision="b" * 40, expected=1)
    root = tmp_path / "mio" / "skills"
    _write_skill(root / "my-local-skill", "my-local-skill")

    report = install_skill_sources_from_checkouts(
        {"source-a": checkout_a, "source-b": checkout_b},
        root=root,
        sources=(source_a, source_b),
    )

    assert report.installed == 3
    assert report.preserved == 1
    assert report.aliases == {"source-b:common": "b-common"}
    assert {path.name for path in root.iterdir() if path.is_dir()} == {
        "common",
        "only-a",
        "b-common",
        "my-local-skill",
    }
    assert not (root / "old").exists()

    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    installed = {row["installed_name"]: row for row in manifest["skills"]}
    assert installed["common"]["source_revision"] == "a" * 40
    assert installed["b-common"]["canonical_name"] == "common"
    assert installed["b-common"]["source_revision"] == "b" * 40
    assert installed["common"]["execution_enabled"] is False


def test_installer_rejects_unexpected_pinned_skill_count(tmp_path):
    checkout = tmp_path / "checkout"
    _write_skill(checkout / "skills" / "one", "one")
    source = _source("counted", alias="counted", revision="c" * 40, expected=2)
    with pytest.raises(SkillValidationError, match="expected 2 skills"):
        install_skill_sources_from_checkouts(
            {"counted": checkout},
            root=tmp_path / "skills",
            sources=(source,),
        )


def test_runner_requires_declaration_persistent_policy_and_call_opt_in(tmp_path):
    root = tmp_path / "skills"
    runner = _write_skill(root / "runner", "runner", script="run.py")
    (runner / "run.py").write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'echo': payload.get('value')}))\n",
        encoding="utf-8",
    )
    catalog = MioSkillCatalog(root)

    with pytest.raises(SkillExecutionDisabled):
        catalog.execute("runner", {"value": 7}, allow_execution=True)
    enabled = catalog.set_execution_enabled("runner", True)
    assert enabled.execution_enabled is True
    with pytest.raises(SkillExecutionDisabled):
        catalog.execute("runner", {"value": 7})

    result = catalog.execute("runner", {"value": 7}, allow_execution=True, timeout_s=5)
    assert result["echo"] == 7
    assert result["returncode"] == 0

    instruction = _write_skill(root / "instruction", "instruction")
    (instruction / "run.py").write_text("print('{}')\n", encoding="utf-8")
    with pytest.raises(SkillExecutionDisabled, match="declares no executable"):
        catalog.set_execution_enabled("instruction", True)


def test_native_and_webui_register_only_catalog_list_and_read_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("MIO_HOME", str(tmp_path / "mio-home"))
    _write_skill(tmp_path / "mio-home" / "skills" / "instructions-only", "instructions-only")

    from mio.agent import AGENT_TOOLS, AGENT_TOOLS_SPEC
    from mio.webui.skills import SKILLS

    native_names = {item["function"]["name"] for item in AGENT_TOOLS_SPEC}
    assert {"list_mio_skills", "read_mio_skill"}.issubset(native_names)
    assert {"list_mio_skills", "read_mio_skill"}.issubset(AGENT_TOOLS)
    assert {"list_mio_skills", "read_mio_skill"}.issubset(SKILLS)
    assert "instructions-only" not in SKILLS
    assert SKILLS["list_mio_skills"]["function"](query="instructions")["matched"] == 1
