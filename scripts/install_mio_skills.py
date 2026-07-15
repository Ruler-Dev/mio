#!/usr/bin/env python3
"""Install Mio's reviewed third-party skill bundle under MIO_HOME."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mio.skill_catalog import PINNED_SOURCES, SkillCatalogError, install_pinned_sources, skills_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch pinned Hallmark, mattpocock, cybersecurity, and game-studio "
            "skills into Mio (never into Codex or Claude homes)."
        )
    )
    parser.add_argument(
        "--mio-home",
        type=Path,
        help="Mio application home (default: $MIO_HOME or ~/.mio)",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=[source.source_id for source in PINNED_SOURCES],
        help="Update only this source; repeat for more than one (default: all)",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="Discard existing unmanaged Mio skill directories instead of preserving them",
    )
    parser.add_argument("--json", action="store_true", help="Print only the final JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = skills_root(args.mio_home) if args.mio_home else None

    def progress(message: str) -> None:
        if not args.json:
            print(f"[mio-skills] {message}", file=sys.stderr)

    try:
        report = install_pinned_sources(
            root=root,
            source_ids=args.source,
            preserve_unmanaged=not args.replace_all,
            progress=progress,
        )
    except SkillCatalogError as exc:
        print(f"mio skill installation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
