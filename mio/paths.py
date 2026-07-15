"""Canonical filesystem roots for Mio-owned state."""

from __future__ import annotations

import os
from pathlib import Path


def mio_home(path: str | os.PathLike[str] | None = None) -> Path:
    """Return Mio's application home, honoring ``MIO_HOME``."""

    raw = path if path is not None else os.environ.get("MIO_HOME")
    if raw is None:
        return Path.home() / ".mio"
    return Path(raw).expanduser().absolute()
