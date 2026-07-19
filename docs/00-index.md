# Mio documentation

The canonical documentation follows the current `codex/qwen36-mlx-engine`
development line. Numerical claims are either linked to raw benchmark JSON or
explicitly marked as historical/experimental.

## Start here

- [01 — Getting started](01-getting-started.md)
- [02 — CLI and API commands](02-commands.md)
- [03 — Deployment and local security](03-deployment.md)
- [04 — Configuration and prompt policies](04-customization.md)
- [05 — Model registry and Qwen 3.6](05-models.md)
- [06 — KV-cache modes](06-turboquant.md)
- [08 — BMP-DFlash](08-bmp-dflash.md)
- [09 — Prefix cache](09-prefix-cache.md)
- [10 — Context compaction](10-compaction.md)
- [11 — Mio UI](11-mio-ui.md)

## Architecture and integration

- [12 — Architecture](12-architecture.md)
- [13 — Development plan](13-development-plan.md)
- [14 — External skills inside Mio](14-external-skills.md)
- [15 — Mio-owned MCP](15-mcp.md)
- [16 — Reproducible benchmarks](16-benchmarks.md)

## Research

- [Mio on Qwen 3.6: local harnessing, prefill, and speculative decode](../papers/mio-qwen36-research.md)
- [22 — Repository-level Markov quality pilot](22-markov-quality-pilot.md) — preregistered exploratory protocol; no result yet
- [BMP-DFlash technical note](../papers/bmp-dflash.md) — historical Qwen 3/3.5 experiments
- [Prefill speedups technical note](../papers/prefill-speedups.md) — historical prefix/LM-head experiments
- [Raw benchmark artifacts](../benchmarks/results/)

## Current evidence boundary

The repository contains historical schema-v1 Qwen 3.6 results and current
working-tree matched R&D artifacts. The current 27B study finds exact cap-2 and
cap-3 DSpark runs with modest decode point estimates but material TTFT
regression; upstream DFlash has a larger direct decode gain but also regresses
TTFT and is not Mio's vendored production runtime. The fused cold-prefill
pilot is promising but short and single-threaded. None establishes a global
prefill breakthrough, long-context scaling, coding-task quality, or
multi-user throughput.

The active implementation also includes a Mio-owned managed skill snapshot
whose pinned revisions currently validate at 916 entries, Mio-local MCP
presets, prompt modes `none`/`caveman`/`ponytail`, Qwen 3.6
sliding-attention support, and loopback server defaults. User-managed skills
may make the live discovery count differ. Release gates and remaining work are
tracked in [13 — Development plan](13-development-plan.md).
