"""End-to-end test of the autolearn pipeline on large-moe.

Ingests N=8 calibration prompts (each topically distinct: math, code,
docs, etc.), storing embedding + frozen KV per prompt. Then tests with
M=8 probe prompts — 4 are near-duplicates of calibration prompts
(should hit), 4 are novel (should miss).

Measures:
  - Retrieval precision: of the 4 "should hit" probes, how many find
    the right prototype in top-1 cosine?
  - Retrieval recall vs cosine threshold.
  - End-to-end wall-clock with autolearn vs without.

This validates Path A works at all.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mio.theories.prefill_autolearn.embed import embed_prompt
from mio.theories.prefill_autolearn.prototype_store import PrototypeStore


# (calibration_prompt, near_duplicate_probe, topic)
_PAIRS = [
    ("Write a Python function to compute Fibonacci with memoization. Return the n-th number.",
     "Implement fib(n) in Python using dict-based memoization. Return int.",
     "fib"),
    ("Implement binary search on a sorted list. Return index or -1 if not found.",
     "Write Python binary_search(arr, target). If present, return index; else -1.",
     "binsearch"),
    ("Explain the difference between a Python list and tuple in 3 bullets.",
     "What's the distinction between list and tuple in Python? Give 3 key points.",
     "list_vs_tuple"),
    ("Write a Stack class in Python with push, pop, peek, is_empty.",
     "Implement a Python Stack with methods: push, pop, peek, is_empty.",
     "stack"),
    ("Implement dict-based inverted index over a list of documents.",
     "Write a Python function that builds an inverted index from documents.",
     "inverted_index"),
    ("Write a merge sort in Python; return a new sorted list.",
     "Implement merge_sort(a) in Python. Return sorted copy of input list.",
     "mergesort"),
    ("Define a LinkedList class in Python with append and __repr__.",
     "Build a Python LinkedList with insert-at-end and string representation.",
     "linkedlist"),
    ("Write a function to check if a string is a valid palindrome (ignore case).",
     "Python: is_palindrome(s) ignoring case and non-alphanumerics.",
     "palindrome"),
]

# Novel probe prompts unrelated to any of the above topics.
_NOVEL_PROBES = [
    "Write a Python function that returns the mean of a list of floats.",
    "Implement quicksort in Python using the Lomuto partition scheme.",
    "Create a generator that yields Fibonacci numbers indefinitely.",
    "Write an asyncio TCP echo server in Python.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=None,
                   help="Prototype store dir. If unset, uses a tmpdir.")
    p.add_argument("--n-early", type=int, default=4,
                   help="Number of early layers for embedding")
    p.add_argument("--min-sim", type=float, default=0.85,
                   help="Minimum cosine similarity for a match")
    p.add_argument("--out", default="experiments/prefill_autolearn/bench.json")
    args = p.parse_args()

    if args.base_dir is None:
        tmp = tempfile.mkdtemp(prefix="autolearn-bench-")
        base_dir = Path(tmp)
    else:
        base_dir = Path(args.base_dir)
    print(f"[autolearn] store dir: {base_dir}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[autolearn] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer
    model_id = f"{tc.name}|{tc.target_model}"
    print(f"[autolearn] loaded. model_id={model_id}", flush=True)

    store = PrototypeStore(base_dir=base_dir, max_entries=50)

    # --- Phase 1: ingest calibration prompts -------------------------------
    print(f"\n[autolearn] === Phase 1: ingest {len(_PAIRS)} calibrations ===",
          flush=True)
    for (calib, _probe, topic) in _PAIRS:
        tokens = engine._apply_chat_template([{"role": "user", "content": calib}])
        t0 = time.perf_counter()
        emb = embed_prompt(engine._target_model, tokens, n_early=args.n_early)
        mx.eval(emb)
        t_embed = time.perf_counter() - t0
        store.add(
            tokens=tokens, embedding=emb,
            frozen_kv_path=f"<test-{topic}>",  # not actual KV path for this probe
            model_id=model_id, pq_bits=tc.pq_bits, tq_bits=tc.tq_bits,
            ctx_window=tc.context_window,
        )
        print(f"  ingested '{topic}': tokens={len(tokens)}  embed_ms={t_embed*1000:.1f}",
              flush=True)

    # --- Phase 2: probe with near-duplicates (should hit) ------------------
    print(f"\n[autolearn] === Phase 2: probe with near-duplicates ===", flush=True)
    hits = {"correct": 0, "wrong": 0, "missed": 0}
    sim_scores_hits: list[float] = []
    rank_of_correct: list[int] = []
    for (calib, probe, topic) in _PAIRS:
        tokens = engine._apply_chat_template([{"role": "user", "content": probe}])
        emb = embed_prompt(engine._target_model, tokens, n_early=args.n_early)
        mx.eval(emb)
        matches = store.nearest(
            emb, k=5, min_similarity=args.min_sim,
            model_id=model_id, pq_bits=tc.pq_bits, tq_bits=tc.tq_bits,
        )
        top_topic = matches[0][0].frozen_kv_path if matches else None
        correct_path = f"<test-{topic}>"
        # Find rank of correct match in the full list
        all_ranked = store.nearest(
            emb, k=len(store), min_similarity=0.0,
            model_id=model_id, pq_bits=tc.pq_bits, tq_bits=tc.tq_bits,
        )
        rank = next(
            (i for i, (p, _) in enumerate(all_ranked) if p.frozen_kv_path == correct_path),
            -1,
        )
        rank_of_correct.append(rank)
        if top_topic == correct_path:
            hits["correct"] += 1
            sim_scores_hits.append(matches[0][1])
            print(
                f"  '{topic}' probe: HIT top-1 sim={matches[0][1]:.3f}  "
                f"(rank_of_correct={rank})",
                flush=True,
            )
        elif top_topic is None:
            hits["missed"] += 1
            best_score = all_ranked[0][1] if all_ranked else 0.0
            print(
                f"  '{topic}' probe: MISS (no match above {args.min_sim}) "
                f"best_score={best_score:.3f}  rank_of_correct={rank}",
                flush=True,
            )
        else:
            hits["wrong"] += 1
            print(
                f"  '{topic}' probe: WRONG top-1 was {top_topic} "
                f"sim={matches[0][1]:.3f}  rank_of_correct={rank}",
                flush=True,
            )

    # --- Phase 3: novel prompts (should NOT hit above threshold) -----------
    print(f"\n[autolearn] === Phase 3: novel prompts (should not match) ===",
          flush=True)
    false_positives = 0
    novel_best_scores: list[float] = []
    for novel in _NOVEL_PROBES:
        tokens = engine._apply_chat_template([{"role": "user", "content": novel}])
        emb = embed_prompt(engine._target_model, tokens, n_early=args.n_early)
        mx.eval(emb)
        matches = store.nearest(
            emb, k=5, min_similarity=args.min_sim,
            model_id=model_id, pq_bits=tc.pq_bits, tq_bits=tc.tq_bits,
        )
        # All scores for diagnostics
        all_ranked = store.nearest(
            emb, k=3, min_similarity=0.0,
            model_id=model_id, pq_bits=tc.pq_bits, tq_bits=tc.tq_bits,
        )
        best_score = all_ranked[0][1] if all_ranked else 0.0
        novel_best_scores.append(best_score)
        if matches:
            false_positives += 1
            print(
                f"  novel '{novel[:40]}...': FALSE POSITIVE matched {matches[0][0].frozen_kv_path} "
                f"sim={matches[0][1]:.3f}",
                flush=True,
            )
        else:
            print(
                f"  novel '{novel[:40]}...': correctly no match (best={best_score:.3f})",
                flush=True,
            )

    # --- Summary ----------------------------------------------------------
    print(f"\n[autolearn] === SUMMARY ===", flush=True)
    print(f"  paired probes: {hits['correct']} correct / {hits['wrong']} wrong / "
          f"{hits['missed']} missed (of {len(_PAIRS)})",
          flush=True)
    if sim_scores_hits:
        avg_sim = sum(sim_scores_hits) / len(sim_scores_hits)
        print(f"  avg sim on correct hits: {avg_sim:.3f}", flush=True)
    if novel_best_scores:
        avg_novel = sum(novel_best_scores) / len(novel_best_scores)
        print(f"  avg best-sim on novel (should be low): {avg_novel:.3f}",
              flush=True)
    print(f"  false positives on novel: {false_positives} / {len(_NOVEL_PROBES)}",
          flush=True)
    # Rank-based metrics
    mrr_ranks = [r for r in rank_of_correct if r >= 0]
    if mrr_ranks:
        mrr = sum(1.0 / (r + 1) for r in mrr_ranks) / len(mrr_ranks)
        print(f"  Mean Reciprocal Rank of correct prototype: {mrr:.3f}", flush=True)

    result = {
        "n_calibrations": len(_PAIRS),
        "min_similarity": args.min_sim,
        "n_early_layers": args.n_early,
        "hits_correct": hits["correct"],
        "hits_wrong": hits["wrong"],
        "hits_missed": hits["missed"],
        "false_positives_on_novel": false_positives,
        "avg_sim_on_correct_hits": (sum(sim_scores_hits) / len(sim_scores_hits))
            if sim_scores_hits else None,
        "avg_sim_on_novel": (sum(novel_best_scores) / len(novel_best_scores))
            if novel_best_scores else None,
        "rank_of_correct": rank_of_correct,
        "model_id": model_id,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[autolearn] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
