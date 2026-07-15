#!/usr/bin/env python3
"""Compatibility wrapper for Mio's packaged MCP tool installer.

The implementation lives in :mod:`mio.mcp.tool_installer` so the same pinned
installer and its lock asset are available from source checkouts and wheels.
"""

import sys
from pathlib import Path


# Direct source-tree execution sets sys.path[0] to ``scripts/``.  Add the
# repository root without depending on an editable install; wheel users invoke
# the packaged ``mio mcp install-tools`` command instead.
repository_root = Path(__file__).resolve().parents[1]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from mio.mcp.tool_installer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
