"""
Single entry point for the rustmc benchmark suite.

Runs every rustmc-vs-PyMC comparison script in this directory with a fixed
seed and captured environment, so results are reproducible run to run and
machine to machine (modulo actual hardware differences, which are recorded
alongside the numbers — see benchmarks/RESULTS_TEMPLATE.md).

Usage:
    python examples/run_benchmarks.py                 # run everything
    python examples/run_benchmarks.py --only compare_with_pymc.py
    python examples/run_benchmarks.py --list

Each script is run as a subprocess (so one script's failure/crash doesn't
take down the rest) and its combined stdout/stderr is written to
benchmarks/results/<script>.log. This script does not compute or claim any
numbers itself — read the per-script logs for the actual results.

Requires: pymc, nutpie, arviz, numpy installed in the active environment
(see README.md "Quick start"). If pymc/nutpie are not installed, the
comparison scripts still run the rustmc side and print a clear
"unavailable — skipping" message for the missing engine rather than
fabricating a number.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = EXAMPLES_DIR.parent / "benchmarks" / "results"

# Order matters only for readability of console output; each script is
# independent and self-contained.
BENCHMARK_SCRIPTS = [
    "compare_with_pymc.py",
    "benchmark_vs_pymc.py",
    "benchmark_multivariate.py",
    "batch_10k_skus.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=None,
                         help="Run only this script (basename). Repeatable.")
    parser.add_argument("--list", action="store_true",
                         help="List available benchmark scripts and exit.")
    args = parser.parse_args()

    if args.list:
        for name in BENCHMARK_SCRIPTS:
            print(name)
        return 0

    scripts = args.only if args.only else BENCHMARK_SCRIPTS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_ok = True
    for name in scripts:
        script_path = EXAMPLES_DIR / name
        if not script_path.exists():
            print(f"[skip] {name}: not found at {script_path}")
            overall_ok = False
            continue

        log_path = RESULTS_DIR / f"{name}.log"
        print(f"\n{'=' * 70}\nRunning {name}  ->  {log_path}\n{'=' * 70}")
        t0 = time.time()
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(EXAMPLES_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_file.write(proc.stdout)
        elapsed = time.time() - t0

        tail = "\n".join(proc.stdout.splitlines()[-15:])
        print(tail)
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[{status}] {name} finished in {elapsed:.1f}s — full log at {log_path}")
        if proc.returncode != 0:
            overall_ok = False

    print(f"\nAll logs written under {RESULTS_DIR}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
