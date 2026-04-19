#!/usr/bin/env python3
"""Download all Qwen 3.5 + DFlash draft models for mio."""

from __future__ import annotations

import sys


def main():
    from huggingface_hub import snapshot_download
    from mio.models.registry import DEFAULT_TIERS

    models = []
    for tier in DEFAULT_TIERS.values():
        models.append(tier.target_model)
        models.append(tier.draft_model)

    # Deduplicate
    models = list(dict.fromkeys(models))

    print(f"Downloading {len(models)} models:")
    for m in models:
        print(f"  - {m}")
    print()

    for repo in models:
        print(f"Downloading {repo}...")
        try:
            path = snapshot_download(repo)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
