"""Shared pytest configuration for the rustmc test suite."""

from __future__ import annotations

import os
import sys

import pytest

# Allow the suite to run against a wheel that was built but not installed
# (``maturin build --release`` + extraction into ``target/extmod``), which is
# useful in sandboxes where ``maturin develop`` cannot write to the venv.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_EXT = os.path.join(_REPO_ROOT, "target", "extmod")
if os.path.isdir(_LOCAL_EXT) and _LOCAL_EXT not in sys.path:
    sys.path.insert(0, _LOCAL_EXT)


@pytest.fixture(scope="session")
def rustmc():
    """The compiled extension module."""
    return pytest.importorskip(
        "rustmc",
        reason="build the extension first: `maturin develop --release`",
    )
