# 14 — Mio external skills

Mio owns its external skill catalog. It does not install instruction bundles
into Codex, Claude Code, or another agent home. The default layout is:

```text
~/.mio/
├── .skills.lock
└── skills/
    ├── .mio-skills.json
    ├── hallmark/
    ├── code-review/
    ├── game-code-review/   # deterministic collision alias
    └── …
```

Set `MIO_HOME` to relocate all Mio-owned state:

```bash
MIO_HOME=/Volumes/Fast/Mio python scripts/install_mio_skills.py
```

## Reviewed bundle

The installer uses full Git commit hashes recorded in
`mio/skill_catalog.py`. It never follows a moving branch during installation.
The installed manifest records the repository, revision, source path, content
digest, and execution policy for every skill.

| Source | Selection | Pinned count |
|---|---|---:|
| `Nutlope/hallmark` | `skills/` | 1 |
| `mattpocock/skills` | active skills; excludes `deprecated`, `in-progress`, and `personal` | 26 |
| `Ruler-Dev/Anthropic-Cybersecurity-Skills` | all skills | 817 |
| `Ruler-Dev/Claude-Code-Game-Studios` | all `.claude/skills` | 72 |

Install or atomically update the complete bundle:

```bash
python scripts/install_mio_skills.py
```

Existing unmanaged directories under `~/.mio/skills` are preserved. Use
`--replace-all` only when intentionally rebuilding the directory from the
reviewed sources. `--source SOURCE_ID` updates a subset while preserving the
other managed sources.

## How Mio uses instruction skills

Mio exposes two small built-in tools in both the native agent and Mio UI:

- `list_mio_skills` searches names, descriptions, tags, and source IDs;
- `read_mio_skill` retrieves one validated `SKILL.md`, with a caller-selected
  character limit and path confinement.

This keeps hundreds of tool schemas out of every inference prompt. A skill is
useful even when it contains instructions only; `run.py` is not required.

## Execution policy

Instruction retrieval never executes repository code. A runner is eligible
only when its YAML frontmatter explicitly declares a relative Python path:

```yaml
---
name: example
description: Example with an optional runner.
script: run.py
---
```

The presence of `run.py` alone has no effect. Execution additionally requires:

1. a persistent per-skill opt-in in the Mio manifest; and
2. `allow_execution=True` at the individual Python call site.

The default agent and Web UI expose no execute-catalog tool. If an integration
opts in, Mio runs the script without a shell and bounds input, output, wall
time, CPU time, open files, and address space. This subprocess policy reduces
accidents but is **not** a network or operating-system security sandbox; only
enable reviewed code.

## Validation and updates

Every managed source is validated before publication:

- UTF-8 `SKILL.md` with YAML frontmatter, `name`, `description`, and body;
- Agent-Skills-compatible lowercase names up to 64 characters (legacy local
  Mio skills may retain a snake_case canonical name);
- no source-tree symlinks, escaping runner paths, oversized files, or
  duplicate canonical names;
- expected skill count at each pinned revision;
- deterministic aliases for cross-source name collisions;
- digest changes reset executable opt-ins.

Installations are built in a staging directory under `MIO_HOME`, protected by
an inter-process lock, and published as a complete snapshot with rollback.
The manifest itself is written with `fsync` plus atomic rename.
