#!/usr/bin/env python3
"""Run every example `examples/README.md` documents, under a time budget.

The README tables are the manifest. Listing a script there without it running,
or adding a runnable script without listing it, is itself a failure -- that is
what let two examples grow to ~57 and ~42 minutes and drift out of the docs
without anyone noticing.

Scripts under "Helpers, not examples" and "Exploratory comparisons" are
excluded by name, because they are modules or need third-party packages this
project does not depend on.

    python scripts/run_examples.py [--timeout SECONDS] [--jobs N]

Exits non-zero if any example fails, exceeds the budget, or if the README and
the directory disagree about which scripts exist.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
README = EXAMPLES / "README.md"

# examples/README.md states that every documented example runs on NumPy alone
# except the ones named here, and says which package each needs. That claim is
# what this map enforces: an example may skip only for the dependency it is
# documented to need. An example that silently acquires a new third-party
# import fails instead of quietly dropping out of the gate.
#
# Only an example that CANNOT run without the package belongs here. An example
# that merely prints more when the package is present must run either way, and
# skipping it would leave its NumPy-only path untested -- which is what happened
# to bayesian_local_level_forecasting.py, whose extra ArviZ diagnostics were
# treated as a prerequisite for running it at all.
OPTIONAL_BY_EXAMPLE = {
    "arviz_example.py": {"arviz", "matplotlib"},
}


def documented() -> tuple[set[str], set[str]]:
    """Return (documented examples, scripts the README excludes by name)."""
    text = README.read_text(encoding="utf-8")
    body, _, excluded_text = text.partition("## Helpers, not examples")
    if not excluded_text:
        sys.exit("examples/README.md has no 'Helpers, not examples' section")
    # Table rows look like: | `name.py` | Use it for ... |
    listed = set(re.findall(r"^\|\s*`([a-z0-9_]+\.py)`\s*\|", body, re.M))
    # Those sections also mention documented examples in passing ("the helper
    # that hierarchical_example.py imports"), so a table entry always wins.
    excluded = set(re.findall(r"`([a-z0-9_]+\.py)`", excluded_text)) - listed
    return listed, excluded


def check_manifest(listed: set[str], excluded: set[str]) -> list[str]:
    present = {p.name for p in EXAMPLES.glob("*.py")}
    problems = []
    for name in sorted(listed - present):
        problems.append(f"README lists {name}, which does not exist")
    for name in sorted(present - listed - excluded):
        problems.append(
            f"{name} exists but is in neither a README table nor an excluded section"
        )
    return problems


def absent_optional(name: str) -> str:
    """The documented optional dependency this example is missing, if any.

    Decided before running, by asking whether the module can be imported, rather
    than by reading the traceback afterwards. A well-behaved example catches its
    own ImportError and exits with a readable message -- `arviz_example.py` prints
    "Install ArviZ first" -- so there is no `ModuleNotFoundError` in stderr to
    match, and sniffing for one reported a documented, expected absence as a
    failure.
    """
    for module in sorted(OPTIONAL_BY_EXAMPLE.get(name, ())):
        if importlib.util.find_spec(module) is None:
            return module
    return ""


def run_one(name: str, timeout: int, env: dict[str, str]) -> tuple[str, str, float, str]:
    started = time.monotonic()
    absent = absent_optional(name)
    if absent:
        return name, "skipped", 0.0, f"needs {absent}"
    try:
        done = subprocess.run(
            [sys.executable, str(EXAMPLES / name)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return name, "TIMEOUT", time.monotonic() - started, f"exceeded {timeout}s"
    elapsed = time.monotonic() - started
    if done.returncode == 0:
        return name, "ok", elapsed, ""
    # Every documented optional dependency is present by the time we get here, so
    # anything still missing is one the README does not account for.
    missing = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", done.stderr)
    if missing:
        module = missing.group(1).split(".")[0]
        return name, "FAILED", elapsed, (
            f"imports {module}, which examples/README.md does not list as an optional\n"
            f"dependency of {name}. Either drop the import or document it and add it to\n"
            f"OPTIONAL_BY_EXAMPLE in this script."
        )
    tail = "\n".join((done.stderr or done.stdout).splitlines()[-12:])
    return name, "FAILED", elapsed, tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="per-example budget in seconds (default 120)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="examples to run at once")
    args = parser.parse_args()

    listed, excluded = documented()
    problems = check_manifest(listed, excluded)
    if problems:
        for problem in problems:
            print(f"MANIFEST: {problem}", file=sys.stderr)
        return 1

    # Chains inside an example should not fight the outer parallelism.
    env = dict(os.environ)
    if args.jobs > 1:
        env.setdefault("RAYON_NUM_THREADS", "2")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_one, name, args.timeout, env) for name in sorted(listed)
        ]
        for future in concurrent.futures.as_completed(futures):
            name, status, elapsed, detail = future.result()
            results.append((name, status, elapsed, detail))
            print(f"{status:>8}  {elapsed:6.1f}s  {name}" + (f"  ({detail})" if status in {"skipped", "TIMEOUT"} else ""))

    bad = [r for r in results if r[1] in {"FAILED", "TIMEOUT"}]
    skipped = [r for r in results if r[1] == "skipped"]
    for name, status, _, detail in bad:
        print(f"\n===== {status}: {name} =====\n{detail}", file=sys.stderr)
    slowest = max(results, key=lambda r: r[2])
    print(
        f"\n{len(results)} documented examples, {len(results) - len(bad) - len(skipped)} ran, "
        f"{len(skipped)} skipped for a documented optional dependency, {len(bad)} bad, "
        f"slowest {slowest[0]} at {slowest[2]:.1f}s (budget {args.timeout}s)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
