#!/usr/bin/env python3
"""Backward-compatible entry point — delegates to benchmarks.cli."""

import sys
from pathlib import Path

# Ensure package is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli import app  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(app())
