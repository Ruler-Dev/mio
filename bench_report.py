"""Post-process bench_kv_experimentation.json into a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(num: float, denom: float) -> str:
    if denom <= 0:
        return "N/A"
    return f"{num/denom:+.1%}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--md", default=None, type=Path)
    args = p.parse_args()

    data = json.loads(args.json.read_text())
    runs = data["runs"]

    # Index runs by (config, prompt, ctx).
    idx: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        idx[(r["config"], r["prompt_id"], r["ctx_target"])] = r

    prompts = sorted({r["prompt_id"] for r in runs})
    ctxs = sorted({r["ctx_target"] for r in runs})
    configs = sorted({r["config"] for r in runs})

    lines: list[str] = []
    lines.append("# large-moe benchmark results\n")
    lines.append(f"Total runs: {len(runs)}\n")

    # Per-context decode-tps comparison.
    lines.append("## Decode throughput (tok/s)\n")
    for ctx in ctxs:
        lines.append(f"### context ≈ {ctx} tokens\n")
        header = "| prompt | " + " | ".join(configs) + " |"
        sep = "|" + "|".join(["---"] * (len(configs) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for prompt in prompts:
            row = [prompt]
            for c in configs:
                r = idx.get((c, prompt, ctx))
                row.append(f"{r['gen_tps']:.1f}" if r else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Prefill timings.
    lines.append("## Prefill wall-time (ms, min of repeats)\n")
    for ctx in ctxs:
        lines.append(f"### context ≈ {ctx} tokens\n")
        header = "| prompt | " + " | ".join(configs) + " |"
        sep = "|" + "|".join(["---"] * (len(configs) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for prompt in prompts:
            row = [prompt]
            for c in configs:
                r = idx.get((c, prompt, ctx))
                row.append(f"{r['prefill_ms']:.0f}" if r else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # DDTree vs baseline ratio.
    lines.append("## DDTree gain over baseline (gen tok/s)\n")
    lines.append("| prompt | ctx | baseline | ddtree | ratio |")
    lines.append("|---|---|---|---|---|")
    for prompt in prompts:
        for ctx in ctxs:
            base = idx.get(("baseline", prompt, ctx))
            dd = idx.get(("ddtree", prompt, ctx))
            if base and dd and base["gen_tps"] > 0:
                ratio = dd["gen_tps"] / base["gen_tps"]
                lines.append(
                    f"| {prompt} | {ctx} | {base['gen_tps']:.1f} | "
                    f"{dd['gen_tps']:.1f} | **{ratio:.2f}x** |"
                )
    lines.append("")

    # DDTree acceptance.
    lines.append("## Avg acceptance per cycle (DFlash 5-8 / DDTree ~4)\n")
    lines.append("| prompt | ctx | baseline | ddtree |")
    lines.append("|---|---|---|---|")
    for prompt in prompts:
        for ctx in ctxs:
            base = idx.get(("baseline", prompt, ctx))
            dd = idx.get(("ddtree", prompt, ctx))
            if base and dd:
                lines.append(
                    f"| {prompt} | {ctx} | {base['avg_accept']:.2f} | "
                    f"{dd['avg_accept']:.2f} |"
                )
    lines.append("")

    # Frozen KV — prefill delta.
    lines.append("## Frozen KV — prefill wall-time\n")
    lines.append("| ctx | cold prefill (ms) | warm prefill (ms) | speedup | delta |")
    lines.append("|---|---|---|---|---|")
    for ctx in ctxs:
        cold = idx.get(("frozen_cold", "sort_bug", ctx))
        warm = idx.get(("frozen_warm", "sort_bug", ctx))
        if cold and warm and cold["prefill_ms"] > 0:
            ratio = warm["prefill_ms"] / cold["prefill_ms"]
            saved = cold["prefill_ms"] - warm["prefill_ms"]
            lines.append(
                f"| {ctx} | {cold['prefill_ms']:.0f} | {warm['prefill_ms']:.0f} | "
                f"**{1/ratio:.1f}x** | -{saved:.0f} ms |"
            )
    lines.append("")

    # Output hash diffs.
    lines.append("## Output-hash match check\n")
    lines.append("| prompt | ctx | baseline | ddtree | frozen_cold | frozen_warm |")
    lines.append("|---|---|---|---|---|---|")
    for prompt in prompts:
        for ctx in ctxs:
            row = [prompt, str(ctx)]
            for c in ("baseline", "ddtree", "frozen_cold", "frozen_warm"):
                r = idx.get((c, prompt, ctx))
                row.append(r["output_sha256"] if r else "—")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    md = "\n".join(lines)
    print(md)
    if args.md:
        args.md.write_text(md)


if __name__ == "__main__":
    main()
