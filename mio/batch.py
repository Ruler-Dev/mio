"""Batch inference: process multiple prompts concurrently."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def process_batch(
    requests: list[BatchRequest],
    manager: ModelManager,
    tier: str = "large",
) -> list[BatchResult]:
    """Process a batch of requests sequentially.

    TODO: Implement true continuous batching with shared model weights
    and independent KV caches for concurrent processing.
    For now, sequential processing still benefits from DFlash + TQ.
    """
    engine = manager.get_engine(tier)
    results: list[BatchResult] = []

    for i, req in enumerate(requests):
        start = time.time()
        try:
            text, metrics = engine.generate(
                req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            elapsed = time.time() - start
            results.append(BatchResult(
                index=i,
                text=text,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                generation_tps=metrics.generation_tps,
                time_s=elapsed,
            ))
        except Exception as e:
            results.append(BatchResult(
                index=i, text="", prompt_tokens=0, completion_tokens=0,
                generation_tps=0, time_s=time.time() - start, error=str(e),
            ))

    return results


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
        results = []
        engine = manager.get_engine(tier)

        for i, req in enumerate(requests):
            start = time.time()
            try:
                text, metrics = engine.generate(
                    req.messages, max_tokens=req.max_tokens, temperature=req.temperature,
                )
                results.append(BatchResult(
                    index=i, text=text,
                    prompt_tokens=metrics.prompt_tokens,
                    completion_tokens=metrics.completion_tokens,
                    generation_tps=metrics.generation_tps,
                    time_s=time.time() - start,
                ))
            except Exception as e:
                results.append(BatchResult(
                    index=i, text="", prompt_tokens=0, completion_tokens=0,
                    generation_tps=0, time_s=time.time() - start, error=str(e),
                ))
            progress.advance(task)

    save_batch_results(results, output_path)

    # Summary
    ok = [r for r in results if not r.error]
    total_tokens = sum(r.completion_tokens for r in ok)
    total_time = sum(r.time_s for r in ok)
    avg_tps = total_tokens / max(total_time, 0.001)

    console.print(f"\n[green]Batch complete:[/green] {len(ok)}/{len(results)} succeeded")
    console.print(f"  Total tokens: {total_tokens}")
    console.print(f"  Total time:   {total_time:.1f}s")
    console.print(f"  Avg tok/s:    {avg_tps:.1f}")
    console.print(f"  Output:       {output_path}")

    manager.unload_all()
