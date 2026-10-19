"""Which build of rustmc a benchmark measured, for its saved record."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def build_provenance(module) -> dict:
    """The imported module's version and file, and the git revision and
    dirty state of the checkout that contains it. The build is assumed to be
    current with that checkout, as ``scripts/dev_pytest.sh`` leaves it."""
    location = Path(module.__file__).resolve().parent

    def git(*args):
        result = subprocess.run(["git", "-C", str(location), *args],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    # Saved records, which a run may be overwriting, do not count as changes.
    status = git("status", "--porcelain", "--untracked-files=no", "--", ".",
                 ":(exclude)benchmarks/results")
    home = str(Path.home())
    return {
        "version": module.__version__,
        # The home directory is written as `~`, so a record does not hold it.
        "file": str(Path(module.__file__).resolve()).replace(home, "~"),
        "revision": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "machine": f"{platform.machine()} {platform.system()}",
        "command": " ".join(sys.argv).replace(home, "~"),
    }
