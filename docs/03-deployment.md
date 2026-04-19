# Deployment Guide

## Local Development (Recommended)

```bash
cd mio
pip install -e .
mio download
mio                  # Start coding with 35B-A3B MoE, 128K context
```

## As a Coding Agent

The default `mio` launches a standalone coding agent:

```bash
mio                              # 35B-A3B MoE, 128K ctx, prefix cache on, caveman ultra
mio --tier medium                # 9B, faster responses
mio --tandem                     # All tiers, auto-routing by complexity
mio "refactor the auth module"   # Start with a task
```

Switch models live with `/tier medium`. Change context with `/context`. Both reload automatically.

## As API Backend (Ollama/LM Studio Replacement)

Mio's API is OpenAI-compatible. Drop-in swap:

```python
# Before (Ollama):
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Before (LM Studio):
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# After (Mio):
client = OpenAI(base_url="http://localhost:9090/v1", api_key="local")
```

Start the server:
```bash
mio serve                        # Default: large-moe
mio serve --tandem --validate    # All tiers + code validation
```

Works with Claude Code, Cursor, Aider, Continue, or any OpenAI SDK client.

## Code Validation

```bash
mio serve --validate
```

Or per-request: `"validate": true` in the request body. Auto-runs ruff + syntax check on generated Python code. If errors found, retries with error context (max 2 retries).

## Web Dashboard

Live monitoring at `http://localhost:9090/dashboard`:
- Active tiers and VRAM usage
- Per-request tok/s with bar visualization
- Acceptance rate and response time
- Real-time WebSocket updates

## Batch Inference

```bash
mio batch --input prompts.jsonl --output results.jsonl
```

Each line: `{"messages": [{"role": "user", "content": "..."}]}`

## Tandem Mode

Load multiple tiers, auto-route by complexity:

```bash
mio --tandem
# or
mio serve --tandem
```

Routing: short prompts -> small (187 tok/s), standard tasks -> medium (108 tok/s), complex tasks -> large-moe (204 tok/s).

## Network Deployment

```bash
mio serve --host 0.0.0.0 --port 9090    # Local network
mio serve --host 127.0.0.1              # Localhost only
```
