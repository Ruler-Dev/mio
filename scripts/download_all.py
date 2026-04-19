#!/usr/bin/env python3
"""Download all required models for mio default tiers.

Downloads to HuggingFace cache (standard) and prints local paths.
Models are referenced by repo ID in the engine — no need to copy.

Target models (mlx-community quantized):
  - Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit  (large, ~14GB)
  - Qwen3.5-27B-8bit                                  (large alt, ~27GB)
  - Qwen3.5-9B-4bit                                   (medium, ~5GB)
  - Qwen3.5-4B-4bit                                   (small, ~2.5GB)

DFlash draft models (z-lab, small):
  - Qwen3.5-27B-DFlash                                (~2GB)
  - Qwen3.5-9B-DFlash                                 (~1GB)
  - Qwen3.5-4B-DFlash                                 (~0.5GB)
"""

import sys
import time

from huggingface_hub import snapshot_download


MODELS_TO_DOWNLOAD = [
    # (repo_id, description)
    # --- DFlash drafts (small, download first) ---
    ("z-lab/Qwen3.5-4B-DFlash", "DFlash draft for 4B"),
    ("z-lab/Qwen3.5-9B-DFlash", "DFlash draft for 9B"),
    ("z-lab/Qwen3.5-27B-DFlash", "DFlash draft for 27B"),
    # --- Target models ---
    ("mlx-community/Qwen3.5-4B-4bit", "Small tier target (4B 4-bit, ~2.5GB)"),
    ("mlx-community/Qwen3.5-9B-4bit", "Medium tier target (9B 4-bit, ~5GB)"),
    ("mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit", "Large tier target — Opus distilled (27B 4-bit, ~14GB)"),
    ("mlx-community/Qwen3.5-27B-8bit", "Large tier alt target (27B 8-bit, ~27GB)"),
]


def main():
    print("Mio Model Downloader")
    print("=" * 60)
    print(f"Downloading {len(MODELS_TO_DOWNLOAD)} models to HuggingFace cache\n")

    results = {}
    for i, (repo_id, desc) in enumerate(MODELS_TO_DOWNLOAD, 1):
        print(f"[{i}/{len(MODELS_TO_DOWNLOAD)}] {desc}")
        print(f"  Repo: {repo_id}")
        start = time.time()
        try:
            path = snapshot_download(repo_id)
            elapsed = time.time() - start
            print(f"  Path: {path}")
            print(f"  Time: {elapsed:.1f}s")
            results[repo_id] = path
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results[repo_id] = None
        print()

    # Summary
    print("=" * 60)
    print("Download Summary:")
    for repo_id, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  [{status}] {repo_id}")

    failed = sum(1 for v in results.values() if v is None)
    if failed:
        print(f"\n{failed} download(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} models downloaded successfully.")


if __name__ == "__main__":
    main()
