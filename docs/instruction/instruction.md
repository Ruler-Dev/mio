# Mio install and use

This former standalone guide duplicated the root README and had drifted from
the model registry, prompt-policy surface, MCP ownership, and benchmark data.
Use the canonical documents instead:

1. [Project README](../../README.md)
2. [Getting started](../01-getting-started.md)
3. [CLI and API](../02-commands.md)
4. [Model registry and Qwen 3.6](../05-models.md)
5. [External skills inside Mio](../14-external-skills.md)
6. [Mio-owned MCP](../15-mcp.md)
7. [Benchmark reproduction](../16-benchmarks.md)

Minimal measured-path setup:

```bash
python3 -m pip install -e .
mio pull large
python3 scripts/install_mio_skills.py
mio mcp install-tools
mio mcp doctor
mio serve --tier large --webui
```

This selects the complete local Qwen 3.6 27B stack when present: target,
preferred DSpark drafter, and a distinct DFlash fallback.
The service binds to `127.0.0.1` by default. It rejects a non-loopback bind
unless `--unsafe-remote-bind` is supplied; that flag adds no authentication,
so do not expose it without a trusted firewall and authenticated reverse proxy.
