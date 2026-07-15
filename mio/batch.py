"""Batch inference: process multiple prompts concurrently."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from mio.model_manager import ModelManager


@dataclass
class BatchRequest:
    messages: list[dict]
    model: str = "mio-large"
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None


@dataclass(frozen=True)
class _SamplerKey:
    temperature: float
    top_p: float
    top_k: int
    seed: int | None


@dataclass
class BatchResult:
    index: int
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_tps: float | None
    time_s: float
    error: str | None = None
    backend: str = "mlx-continuous"
    metrics_scope: str = "request"
    batch_size: int = 1
    batch_generation_tps: float | None = None


def process_batch(
    requests: list[BatchRequest],
    manager: ModelManager,
    tier: str = "large",
) -> list[BatchResult]:
    """Process requests with shared weights and independent MLX KV caches.

    Requests are grouped by their complete sampler configuration because MLX
    applies one sampler/RNG stream to an active batch. A one-request group takes
    Mio's latency path; larger groups use continuous batching. Per-request stop
    strings remain independent inside either path.
    """
    engine = manager.get_engine(tier)
    if not requests:
        return []
    results: list[BatchResult | None] = [None] * len(requests)
    groups: dict[_SamplerKey, list[tuple[int, BatchRequest]]] = {}
    for index, request in enumerate(requests):
        key = _SamplerKey(
            temperature=(
                float(request.temperature)
                if request.temperature is not None
                else 0.0
            ),
            top_p=(
                float(request.top_p)
                if request.top_p is not None
                else float(engine.tier_config.top_p)
            ),
            top_k=(
                int(request.top_k)
                if request.top_k is not None
                else int(engine.tier_config.top_k)
            ),
            seed=None if request.seed is None else int(request.seed),
        )
        groups.setdefault(key, []).append((index, request))

    for sampler_key, group in groups.items():
        start = time.time()
        try:
            generated = engine.generate_batch(
                [request.messages for _, request in group],
                max_tokens=[
                    (
                        engine.tier_config.max_output_tokens
                        if request.max_tokens is None
                        else request.max_tokens
                    )
                    for _, request in group
                ],
                temperature=sampler_key.temperature,
                top_p=sampler_key.top_p,
                top_k=sampler_key.top_k,
                seed=sampler_key.seed,
                stop=[request.stop for _, request in group],
            )
            elapsed = time.time() - start
            for (index, _request), (text, metrics) in zip(group, generated, strict=True):
                backend = (
                    "mlx-target-sampling"
                    if sampler_key.seed is not None and sampler_key.temperature > 0.0
                    else "mlx-continuous"
                    if len(group) > 1
                    else (
                        "mlx-target-sampling"
                        if metrics.fallback_reason == "stochastic_sampling_requires_target_only"
                        else "dflash-latency"
                    )
                )
                results[index] = BatchResult(
                    index=index,
                    text=text,
                    prompt_tokens=metrics.prompt_tokens,
                    completion_tokens=metrics.completion_tokens,
                    generation_tps=(
                        metrics.generation_tps
                        if metrics.metrics_scope == "request"
                        else None
                    ),
                    time_s=elapsed,
                    backend=backend,
                    metrics_scope=metrics.metrics_scope,
                    batch_size=metrics.batch_size,
                    batch_generation_tps=(
                        metrics.generation_tps
                        if metrics.metrics_scope == "batch"
                        else None
                    ),
                )
        except Exception as e:
            elapsed = time.time() - start
            for index, _request in group:
                results[index] = BatchResult(
                    index=index,
                    text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    generation_tps=0,
                    time_s=elapsed,
                    error=str(e),
                    backend="error",
                )

    return [result for result in results if result is not None]


def load_batch_file(path: str) -> list[BatchRequest]:
    """Load batch requests from a JSONL file.

    Each line should be a JSON object with "messages" array.
    Optional: "model", "max_tokens", "temperature", "top_p", "top_k",
    "seed", and "stop" (a string or list of strings).
    """
    requests = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_no}: {e}")
                continue
            messages = data.get("messages", [])
            if not messages and "prompt" in data:
                messages = [{"role": "user", "content": data["prompt"]}]
            raw_stop = data.get("stop")
            stop = [raw_stop] if isinstance(raw_stop, str) else raw_stop
            requests.append(BatchRequest(
                messages=messages,
                model=data.get("model", "mio-large"),
                max_tokens=data.get("max_tokens"),
                temperature=data.get("temperature"),
                top_p=data.get("top_p"),
                top_k=data.get("top_k"),
                seed=data.get("seed"),
                stop=stop,
            ))
    return requests


def save_batch_results(results: list[BatchResult], path: str) -> None:
    """Save batch results to a JSONL file."""
    with open(path, "w") as f:
        for r in results:
            data = {
                "index": r.index,
                "text": r.text,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "generation_tps": r.generation_tps,
                "batch_generation_tps": r.batch_generation_tps,
                "metrics_scope": r.metrics_scope,
                "batch_size": r.batch_size,
                "time_s": r.time_s,
            }
            if r.error:
                data["error"] = r.error
            data["backend"] = r.backend
            f.write(json.dumps(data) + "\n")


def run_batch_cli(input_path: str, output_path: str, tier: str = "large") -> None:
    """Run batch from CLI."""
    from rich.console import Console
    from rich.progress import Progress

    from mio.config import MioConfig

    console = Console()
    config = MioConfig.default()
    config.active_tiers = [tier]

    from mio.model_manager import ModelManager

    manager = ModelManager(config)
    console.print(f"Loading {tier} tier...")
    manager.load_active_tiers()

    requests = load_batch_file(input_path)
    console.print(f"Processing {len(requests)} requests...")

    with Progress() as progress:
        task = progress.add_task("Batch inference", total=len(requests))
        batch_started = time.time()
        results = process_batch(requests, manager, tier=tier)
        batch_wall_s = time.time() - batch_started
        progress.advance(task, len(results))

    save_batch_results(results, output_path)

    # Summary
    ok = [r for r in results if not r.error]
    total_tokens = sum(r.completion_tokens for r in ok)
    avg_tps = total_tokens / max(batch_wall_s, 0.001)

    console.print(f"\n[green]Batch complete:[/green] {len(ok)}/{len(results)} succeeded")
    console.print(f"  Total tokens: {total_tokens}")
    console.print(f"  Total time:   {batch_wall_s:.1f}s")
    console.print(f"  Avg tok/s:    {avg_tps:.1f}")
    console.print(f"  Output:       {output_path}")

    manager.unload_all()
