"""Compatibility entry point for the isolated reference benchmark suite.

New commands should invoke ``python benchmarks/run.py`` directly. This wrapper keeps the
former example path working without treating the exploratory scripts in this directory as
comparable benchmark evidence.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    runpy.run_module("benchmarks.run", run_name="__main__")
