#!/usr/bin/env python3
"""Legacy entry point that runs the Typer app from a source checkout.

Keeps ``python scripts/benchmark.py ...`` working when the package has not been
installed. New use should call the ``bench`` console script instead. It does
not re-implement any command.
"""

import sys
from pathlib import Path

# The package is not necessarily installed when this file is run directly, so
# the repository root is prepended before the import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli import app  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(app())
