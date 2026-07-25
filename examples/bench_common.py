"""
Shared utilities for rustmc benchmark/comparison scripts.

This module exists so every comparison script:
  - records the same environment fields (CPU, thread count, OS, library
    versions) alongside its numbers,
  - times model construction, compilation, sampling, and post-processing
    as separate phases instead of one conflated wall-clock number,
  - reports the same statistical-quality metrics (R-hat, bulk ESS,
    divergences, posterior error vs. known simulated truth) for every
    engine it compares, not just wall time.

Nothing in here changes the statistical model being fit; it only makes
timing and reporting consistent across scripts so results are comparable
and reproducible.
"""
from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------

def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _rustmc_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("rustmc")
    except Exception:
        pass
    try:
        import rustmc
        return getattr(rustmc, "__version__", "unknown")
    except Exception:
        return "unknown"


def _nutpie_version() -> str:
    try:
        import nutpie
        return getattr(nutpie, "__version__", "unknown")
    except ImportError:
        return "not installed"


def capture_environment() -> dict:
    """Everything a reader needs to judge whether a result is reproducible."""
    return {
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS", "(unset, defaults to logical cpus)"),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", "(unset)"),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "rustmc": _rustmc_version(),
        "pymc": _pkg_version("pymc"),
        "nutpie": _nutpie_version(),
        "arviz": _pkg_version("arviz"),
        "numpy": _pkg_version("numpy"),
    }


def print_environment(env: dict | None = None) -> None:
    env = env or capture_environment()
    print("Environment:")
    print(f"  CPU              : {env['cpu_model']}")
    print(f"  Logical CPUs     : {env['logical_cpus']}")
    print(f"  RAYON_NUM_THREADS: {env['rayon_num_threads_env']}")
    print(f"  OMP_NUM_THREADS  : {env['omp_num_threads_env']}")
    print(f"  OS               : {env['os']}")
    print(f"  Python           : {env['python']}")
    print(f"  rustmc           : {env['rustmc']}")
    print(f"  pymc             : {env['pymc']}")
    print(f"  nutpie           : {env['nutpie']}")
    print(f"  arviz            : {env['arviz']}")
    print(f"  numpy            : {env['numpy']}")


# --------------------------------------------------------------------------
# Phase timing
# --------------------------------------------------------------------------

@dataclass
class PhaseTimer:
    """Accumulates named phase durations so total wall time is never a
    single conflated number. Use as:

        pt = PhaseTimer()
        with pt.phase("build"):
            ...
        with pt.phase("sample"):
            ...
        pt.report()
    """
    phases: dict = field(default_factory=dict)

    @contextlib.contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t0)

    @property
    def total(self) -> float:
        return sum(self.phases.values())

    def report(self, label: str = "") -> None:
        if label:
            print(f"  [{label}] phase timing:")
        for name, dur in self.phases.items():
            print(f"    {name:<14}: {dur:.3f}s")
        print(f"    {'total':<14}: {self.total:.3f}s")


# --------------------------------------------------------------------------
# Posterior-quality metrics (engine-agnostic reporting helpers)
# --------------------------------------------------------------------------

@dataclass
class QualityReport:
    engine: str
    wall_time_total: float
    ess_bulk_mean: float
    ess_per_sec: float
    r_hat_max: float
    divergences: int
    posterior_error_rmse: float | None = None
    peak_rss_mb: float | None = None

    def print_row(self) -> None:
        rmse = f"{self.posterior_error_rmse:.4f}" if self.posterior_error_rmse is not None else "n/a"
        rss = f"{self.peak_rss_mb:.0f}" if self.peak_rss_mb is not None else "n/a"
        print(
            f"{self.engine:<16} time={self.wall_time_total:>8.2f}s  "
            f"ess_bulk={self.ess_bulk_mean:>8.0f}  ess/s={self.ess_per_sec:>9.1f}  "
            f"max_r_hat={self.r_hat_max:>6.3f}  divergences={self.divergences:>4}  "
            f"rmse_vs_truth={rmse:>8}  peak_rss_mb={rss:>8}"
        )


def peak_rss_mb() -> float | None:
    """Peak resident set size for this process, in MB. Linux-only; returns
    None elsewhere so callers can report 'n/a' rather than a wrong number."""
    try:
        import resource
        # ru_maxrss is KB on Linux, bytes on macOS.
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return val / (1024 * 1024)
        return val / 1024
    except Exception:
        return None


def rank_normalized_split_rhat_note() -> str:
    return (
        "R-hat/ESS use rank-normalized split R-hat and bulk/tail ESS "
        "(Vehtari et al. 2021) on both sides: rustmc's built-in diagnostics() "
        "and arviz.rhat/arviz.ess use the same estimator."
    )
