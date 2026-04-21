"""Analyze Phase 0 baseline profile results.

Reads experiments/phase0_baselines/results.json, validates against the
predictions in hypothesis.md, emits a structured finding doc.

This is an analysis tool, not a theory — output only summarizes measured
data, never invents numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


def _cells_by_ctx(samples: list[dict]) -> dict[int, dict[str, dict]]:
    by_ctx: dict[int, dict[str, dict]] = {}
    for s in samples:
        key = "cold" if s["cold"] else "warm"
        ent = by_ctx.setdefault(s["ctx"], {})
        # keep minimum-total as the canonical warm pick; cold is one-shot.
        if key not in ent or s["total_ms"] < ent[key]["total_ms"]:
            ent[key] = s
    return by_ctx


def _layer_breakdown(sample: dict) -> tuple[float, float, float]:
    """Return (linear_ms, attn_ms, other_ms)."""
    layers = sample.get("per_layer", [])
    linear_ms = sum(l["ms"] for l in layers if l["linear"])
    attn_ms = sum(l["ms"] for l in layers if not l["linear"])
    total = sample["total_ms"]
    other = max(0.0, total - linear_ms - attn_ms)
    return linear_ms, attn_ms, other


def _check_predictions(by_ctx: dict[int, dict[str, dict]]) -> list[str]:
    """Check predictions from hypothesis.md; return one pass/fail per prediction."""
    results: list[str] = []

    # (1) Linear share dominates below 8K
    small_ctxs = [c for c in by_ctx if c < 8192]
    linear_shares = [by_ctx[c]["warm"]["linear_share"] for c in small_ctxs if "warm" in by_ctx[c]]
    avg_linear_small = sum(linear_shares) / len(linear_shares) if linear_shares else 0
    p1 = avg_linear_small >= 0.55
    results.append(
        f"**P1 (linear share ≥55% below 8K):** {'PASS' if p1 else 'FAIL'} — "
        f"measured avg {avg_linear_small*100:.1f}% across "
        f"{sorted(small_ctxs)} warm runs."
    )

    # (2) Crossover: attention ≥40% at 32K OR attention > linear at 32K
    big = by_ctx.get(32768, {}).get("warm")
    if big:
        attn_share_32k = big["attention_share"]
        p2 = attn_share_32k >= 0.40
        results.append(
            f"**P2 (attention share ≥40% at 32K):** {'PASS' if p2 else 'FAIL'} — "
            f"measured {attn_share_32k*100:.1f}% warm at N=32K."
        )
    else:
        results.append("**P2:** no 32K sample; skipping.")

    # (3) Super-linear scaling
    ms_8k = by_ctx.get(8192, {}).get("warm", {}).get("total_ms", 0)
    ms_32k = by_ctx.get(32768, {}).get("warm", {}).get("total_ms", 0)
    if ms_8k > 0 and ms_32k > 0:
        ratio = ms_32k / ms_8k
        p3 = ratio > 4.0
        results.append(
            f"**P3 (ms(32K)/ms(8K) > 4 — super-linear):** "
            f"{'PASS' if p3 else 'FAIL'} — measured {ratio:.2f}× "
            f"(exact linear would be 4.00)."
        )
    else:
        results.append("**P3:** 8K and/or 32K missing; skipping.")

    # (4) Cold vs warm ≤20% at N≥8K
    big_deltas = []
    for c in [8192, 16384, 32768]:
        ent = by_ctx.get(c, {})
        if "cold" in ent and "warm" in ent:
            delta = (ent["cold"]["total_ms"] - ent["warm"]["total_ms"]) / ent["warm"]["total_ms"]
            big_deltas.append((c, delta))
    ok = all(abs(d) <= 0.20 for _, d in big_deltas)
    if big_deltas:
        details = ", ".join(
            f"N={c}: {d*100:+.1f}%" for c, d in big_deltas
        )
        results.append(
            f"**P4 (cold-warm gap ≤20% at N≥8K):** "
            f"{'PASS' if ok else 'FAIL'} — {details}."
        )
    return results


def _attack_vectors(by_ctx: dict[int, dict[str, dict]]) -> list[str]:
    """Data-driven attack-vector ranking from measured breakdowns."""
    lines: list[str] = []
    lines.append("")
    lines.append("## Data-driven attack vector priorities")
    lines.append("")
    lines.append("Ranked by absolute wall-clock (ms) each block consumes at "
                 "the contexts mio actually serves (4K, 16K, 32K). "
                 "Time spent is the upper bound on achievable savings.")
    lines.append("")
    lines.append("| ctx | linear (ms) | attention (ms) | other (ms) | top vector |")
    lines.append("|---|---:|---:|---:|:---|")
    for c in [4096, 16384, 32768]:
        warm = by_ctx.get(c, {}).get("warm")
        if not warm:
            continue
        lin, attn, other = _layer_breakdown(warm)
        pairs = [("linear", lin), ("attention", attn), ("other", other)]
        pairs.sort(key=lambda x: -x[1])
        top = pairs[0][0]
        lines.append(
            f"| {c} | {lin:.0f} | {attn:.0f} | {other:.0f} | {top} |"
        )
    return lines


def _per_layer_table(sample: dict, max_rows: int = 8) -> list[str]:
    """Top N slowest layers for a given sample."""
    layers = sorted(sample.get("per_layer", []), key=lambda l: -l["ms"])
    lines: list[str] = []
    lines.append("")
    lines.append(f"### Top {max_rows} slowest layers (ctx={sample['ctx']}, warm)")
    lines.append("")
    lines.append("| layer | type | ms |")
    lines.append("|---:|:---|---:|")
    for l in layers[:max_rows]:
        t = "linear (GatedDelta)" if l["linear"] else "full attention"
        lines.append(f"| {l['idx']} | {t} | {l['ms']:.2f} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="experiments/phase0_baselines/results.json")
    p.add_argument("--out", default="docs/theories/phase0_analysis.md")
    args = p.parse_args()

    data = json.loads(Path(args.json).read_text())
    samples = data["samples"]
    by_ctx = _cells_by_ctx(samples)

    lines = [
        "# Phase 0 analysis — measured vs predicted",
        "",
        f"- git SHA at run: `{data['git_sha']}`",
        f"- hardware: `{data['hardware']}`",
        f"- target: `{data['target_model']}`",
        f"- pq_bits={data['pq_bits']}, tq_bits={data['tq_bits']}",
        f"- timestamp (epoch): {data['timestamp_epoch']}",
        "",
        "## Prediction check",
        "",
    ]
    lines.extend(f"- {r}" for r in _check_predictions(by_ctx))
    lines.extend(_attack_vectors(by_ctx))

    # Per-layer slow list for the biggest context with a sample
    for c in sorted(by_ctx.keys(), reverse=True):
        if "warm" in by_ctx[c]:
            lines.extend(_per_layer_table(by_ctx[c]["warm"]))
            break

    Path(args.out).write_text("\n".join(lines))
    print(Path(args.out).read_text())


if __name__ == "__main__":
    main()
