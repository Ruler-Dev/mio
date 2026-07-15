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


@dataclass
class BatchResult:
    index: int
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_tps: float
    time_s: float
    error: str | None = None
    backend: str = "mlx-continuous"


def process_batch(
    requests: list[BatchRequest],
    manager: ModelManager,
    tier: str = "large",
) -> list[BatchResult]:
    """Process requests with shared weights and independent MLX KV caches.

    Requests are grouped by temperature because MLX applies one sampler to an
    active batch. A one-request group takes Mio's latency-oriented DFlash path;
    larger groups use :meth:`MioEngine.generate_batch` continuous batching.
    """
    engine = manager.get_engine(tier)
    if not requests:
        return []
    results: list[BatchResult | None] = [None] * len(requests)
    groups: dict[float, list[tuple[int, BatchRequest]]] = {}
    for index, request in enumerate(requests):
        temperature = (
            float(request.temperature)
            if request.temperature is not None
            else float(engine.tier_config.temperature)
        )
        groups.setdefault(temperature, []).append((index, request))

    for temperature, group in groups.items():
        start = time.time()
        try:
            generated = engine.generate_batch(
                [request.messages for _, request in group],
                max_tokens=[
                    request.max_tokens or engine.tier_config.max_output_tokens
                    for _, request in group
                ],
                temperature=temperature,
            )
            elapsed = time.time() - start
            backend = "mlx-continuous" if len(group) > 1 else "dflash-latency"
            for (index, _request), (text, metrics) in zip(group, generated, strict=True):
                results[index] = BatchResult(
                    index=index,
                    text=text,
                    prompt_tokens=metrics.prompt_tokens,
                    completion_tokens=metrics.completion_tokens,
                    generation_tps=metrics.generation_tps,
                    time_s=elapsed,
                    backend=backend,
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
    Optional: "model", "max_tokens", "temperature".
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
            requests.append(BatchRequest(
                messages=messages,
                model=data.get("model", "mio-large"),
                max_tokens=data.get("max_tokens"),
                temperature=data.get("temperature"),
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
