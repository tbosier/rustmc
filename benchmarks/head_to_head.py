"""rustmc against nutpie on matched models, one cell at a time.

A cell is one model at one size and one thread setting. For each cell, each engine
runs in its own worker subprocess, with the environment ``benchmarks.run`` gives
it, which builds and compiles its model once and then fits datasets on request.
The parent interleaves the engines dataset by dataset, rotating their order, so a
drift in the machine's speed during a cell falls on every engine alike. Every
engine gets the same priors, likelihood, chains, warmup, draws and datasets, and
its own defaults for everything else (target acceptance 0.8 and its own metric
and step-size adaptation). The parent holds an exclusive ``flock`` on ``--lock``
for the whole cell and, before each compile and each fit, waits for the one-minute
load average to fall below ``--max-load``. The lock only keeps out other
benchmarks that take it too; a wait that times out is recorded on that fit, and
the fit is then left out of every verdict and aggregate.

Per fit it records three wall times, without compilation, from bound data:

- ``seconds``, to an ArviZ InferenceData holding each engine's posterior draws
  and post-warmup sample statistics: rustmc's
  ``FitResult.to_arviz(include_log_likelihood=False)`` (which also carries a small
  ``observed_data`` group), and nutpie's arrow trace with the warmup rows sliced
  off and then converted as nutpie converts it. With the warmup gone, the
  conversion's only warmup work is building empty arrays. The work is close but
  not identical: each engine converts its own sample statistics, and nutpie has
  more of them. This is the end point the efficiency ratios use by default;
- ``sampling_seconds``, to the end of sampling, before any conversion;
- ``native_seconds``, sampling plus each engine's own default conversion, timed
  separately: rustmc's ``to_arviz()``, with pointwise log-likelihoods, and nutpie's
  conversion of the whole trace as ``nutpie.sample(save_warmup=False)`` does it,
  which still builds the warmup datasets before dropping them. It is what a user
  of each default API waits for, not matched work. The two conversions take turns
  going first by dataset, since the second finds caches warm, and each fit
  records which went first.

nutpie's spans read internals of nutpie 0.16.6: ``nutpie.sample(...,
blocking=False)`` returns a ``_BackgroundSampler``, whose ``_sampler.wait`` marks
the end of sampling, and the conversion is ``nutpie.sample._arrow_to_arviz`` as
``_BackgroundSampler._extract`` calls it. Each fit's record names the groups of the
matched InferenceData.

It also records the leapfrog steps of warmup and draws, the minimum and mean bulk
ESS and minimum tail ESS over the model's parameters, R-hat, post-warmup
divergences, each parameter's mean and Monte Carlo standard error, and the load
before the fit. Each fit's outcome is kept on its own, so a failure partway
through a cell keeps the fits before it. Compilation is timed once per cell and
engine, with the number of files in PyTensor's and Numba's compile caches
beforehand; an empty cache means a cold compile, and a non-empty one says only that
something was cached, not that this model was. Regression cells also run nutpie
with its own default warmup of 300 iterations, as ``nutpie_default_tune``, since
that is what a user gets without asking; the matched comparison keeps the same
warmup for both. One and four threads give the same draws in both engines, so a
cell's two thread settings time the same fits and are not extra replicates.

``benchmarks/head_to_head_summary.py`` turns a record into verdicts.

    python -m benchmarks.head_to_head --output results.json [--quick] [--model NAME]

The rustmc this measures is whichever ``import rustmc`` finds; point PYTHONPATH at
a fresh build (``scripts/dev_pytest.sh`` shows how). The record holds the checkout's
revision and dirty flag, and a SHA-256 of the extension the parent and each rustmc
worker loaded.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.protocol import BenchmarkConfig, environment_metadata, package_version
from benchmarks.run import _child_environment

RESULT_PREFIX = "RUSTMC_HEAD_TO_HEAD_RESULT="
ENGINES = ("rustmc", "nutpie", "nutpie_default_tune")
NUTPIE_DEFAULT_TUNE = 300
REGRESSIONS = ("normal", "normal_sigma", "logistic", "poisson")
LOAD_WAIT_SECONDS = 1800.0


@dataclass(frozen=True)
class Cell:
    """One model at one size, fitted on `datasets` datasets with `threads` threads.

    Six datasets let a paired sign test reach the 5% level unadjusted when all six
    ratios fall on one side of 1 (two-sided p = 0.031); with five it cannot."""

    model: str
    rows: int
    columns: int
    threads: int
    datasets: int = 6
    chains: int = 4
    warmup: int = 1000
    draws: int = 1000
    data_seed: int = 20261025

    @property
    def name(self) -> str:
        return f"{self.model} {self.rows}x{self.columns} threads={self.threads}"


# ── data ───────────────────────────────────────────────────────────────────

EIGHT_SCHOOLS_Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
EIGHT_SCHOOLS_SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def make_data(cell: Cell, index: int) -> dict[str, np.ndarray]:
    """Dataset `index` of `cell`, the same for every engine and thread setting.

    Regressions: `rows` x `columns` design with standardized columns, an
    intercept and coefficients scaled so the linear predictor has a standard
    deviation of about 1 (0.5 for Poisson, which keeps counts small). The
    varying-intercept model has `columns` groups. Eight schools has one dataset,
    so its datasets differ only in the sampler seed.
    """
    rng = np.random.default_rng([cell.data_seed, cell.rows, cell.columns, index])
    if cell.model == "eight_schools":
        return {"y": EIGHT_SCHOOLS_Y, "sigma": EIGHT_SCHOOLS_SIGMA}
    if cell.model == "varying_intercept":
        group = rng.integers(0, cell.columns, size=cell.rows)
        effects = 0.7 * rng.normal(size=cell.columns)
        y = 0.4 + effects[group] + rng.normal(size=cell.rows)
        return {"group": group, "y": y}
    x = rng.normal(size=(cell.rows, cell.columns))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    beta = rng.normal(size=cell.columns) / math.sqrt(cell.columns)
    if cell.model == "poisson":
        eta = 0.5 + 0.5 * (x @ beta)
        y = rng.poisson(np.exp(eta)).astype(float)
    elif cell.model == "logistic":
        eta = 0.3 + x @ beta
        y = (rng.random(cell.rows) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif cell.model == "normal":
        y = 1.0 + x @ beta + rng.normal(size=cell.rows)
    elif cell.model == "normal_sigma":
        y = 1.0 + x @ beta + 0.7 * rng.normal(size=cell.rows)
    else:
        raise ValueError(f"unknown model {cell.model!r}")
    return {"X": x, "y": y}


# The parameters each engine reports, as (name, length); length 0 is a scalar.
def parameters(cell: Cell) -> list[tuple[str, int]]:
    if cell.model == "eight_schools":
        return [("mu", 0), ("tau", 0), ("z", 8)]
    if cell.model == "varying_intercept":
        return [("mu", 0), ("tau", 0), ("z", cell.columns), ("sigma", 0)]
    names = [("alpha", 0), ("beta", cell.columns)]
    if cell.model == "normal_sigma":
        names.append(("sigma", 0))
    return names


def scalar_names(cell: Cell) -> list[str]:
    return [name if length == 0 else f"{name}[{k}]"
            for name, length in parameters(cell) for k in range(max(length, 1))]


# ── statistics, the same functions for both engines ───────────────────────

MAX_RHAT = 1.01
MIN_BULK_ESS = 400.0


def draw_statistics(draws: dict[str, np.ndarray], diverging: np.ndarray) -> dict[str, Any]:
    """ESS, R-hat and per-parameter means and MCSEs from `name -> (chains, draws)`.

    The minima and maxima propagate NaN, so a NaN diagnostic reaches the record
    rather than being skipped by a comparison."""
    import arviz as az

    dataset = az.convert_to_dataset({name: values for name, values in draws.items()})
    bulk = az.ess(dataset, method="bulk")
    tail = az.ess(dataset, method="tail")
    rhat = az.rhat(dataset)
    mcse = az.mcse(dataset, method="mean")
    names = list(draws)
    bulk_values = np.array([float(bulk[name]) for name in names])
    return {
        "min_bulk_ess": float(np.min(bulk_values)),
        "mean_bulk_ess": float(np.mean(bulk_values)),
        "min_tail_ess": float(np.min([float(tail[name]) for name in names])),
        "max_rhat": float(np.max([float(rhat[name]) for name in names])),
        "divergences": int(np.sum(diverging)),
        "means": {name: float(np.mean(values)) for name, values in draws.items()},
        "mcse": {name: float(mcse[name]) for name in names},
    }


def fit_problems(fit: dict[str, Any]) -> list[str]:
    """Why `fit` cannot enter a verdict; empty when it can.

    A fit fails when it did not finish, when its load wait timed out, when any
    diagnostic is not finite, when a recorded time is not finite and positive, when
    R-hat exceeds 1.01, when any post-warmup transition diverged, or when its
    smallest bulk ESS is under 400."""
    if fit.get("status") != "ok":
        return [f"{fit.get('status')}: {fit.get('reason')}"]
    problems = []
    if fit.get("load_wait_timed_out"):
        problems.append(f"load wait timed out at {fit.get('load_at_start', math.nan):.2f}")
    values = [fit["max_rhat"], fit["min_bulk_ess"], fit["min_tail_ess"],
              *fit["means"].values(), *fit["mcse"].values()]
    if not all(math.isfinite(value) for value in values):
        problems.append("non-finite diagnostic")
    times = [fit[key] for key in ("seconds", "sampling_seconds", "native_seconds") if key in fit]
    if not all(math.isfinite(value) and value > 0 for value in times):
        problems.append("time not finite and positive")
    if fit["max_rhat"] > MAX_RHAT:
        problems.append(f"R-hat {fit['max_rhat']:.3f}")
    if fit["divergences"]:
        problems.append(f"{fit['divergences']} divergences")
    if fit["min_bulk_ess"] < MIN_BULK_ESS:
        problems.append(f"min bulk ESS {fit['min_bulk_ess']:.0f}")
    return problems


def mean_difference_z(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Largest |difference of means| / combined MCSE over the parameters of two
    fits. Equal means with zero MCSE give 0, different ones infinity, and any
    non-finite input NaN."""
    worst = 0.0
    for name, mean in a["means"].items():
        values = (mean, b["means"][name], a["mcse"][name], b["mcse"][name])
        if not all(math.isfinite(value) for value in values):
            return math.nan
        spread = math.hypot(a["mcse"][name], b["mcse"][name])
        difference = abs(mean - b["means"][name])
        if spread == 0:
            z = 0.0 if difference == 0 else math.inf
        else:
            z = difference / spread
        worst = max(worst, z)
    return worst


def max_propagating_nan(values: list[float]) -> float:
    """The largest value, or NaN if any value is NaN, whatever the order."""
    if not values or any(math.isnan(value) for value in values):
        return math.nan
    return max(values)


def mean_difference_threshold(tests: int) -> float:
    """The two-sided normal quantile at a 0.1% family-wise level over `tests`."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - 0.001 / (2 * tests))


def sign_test(ratios: list[float]) -> float:
    """Two-sided exact sign test of whether rustmc wins (ratio below 1) more or fewer
    pairs than it loses (above 1). Ratios of exactly 1 are left out, so with ties it
    tests the split of the untied pairs, not the median of all of them."""
    below = sum(ratio < 1 for ratio in ratios)
    above = sum(ratio > 1 for ratio in ratios)
    n = below + above
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(below, above) + 1)) / 2**n
    return min(1.0, 2 * tail)


def winner(ratios: list[float]) -> str:
    """The engine that won more of the paired ratios, ties left out as the sign
    test leaves them out, so a significant test never names the engine that lost
    more pairs."""
    below = sum(ratio < 1 for ratio in ratios)
    above = sum(ratio > 1 for ratio in ratios)
    return "rustmc faster" if below > above else "nutpie faster"


def median_interval(values: list[float], level: float = 0.95) -> dict[str, float]:
    """The distribution-free interval for a median, between order statistics
    x_(k) and x_(n+1-k), the narrowest whose coverage, 1 - 2 P(Bin(n, 1/2) < k),
    is at least `level`; when none reaches it, [min, max] with its coverage. When no
    ratio is exactly 1 it excludes 1 exactly when a sign test with the same coverage
    would reject; ties at 1 can leave 1 inside an interval while the sign test, which
    drops them, rejects.
    With 6 pairs it is [min, max] at 96.9%; with 3, [min, max] at 75%."""
    x = sorted(values)
    n = len(x)

    def coverage(k: int) -> float:
        return 1 - 2 * sum(math.comb(n, j) for j in range(k)) / 2**n

    k = 1
    for candidate in range(1, n // 2 + 1):
        if coverage(candidate) >= level:
            k = candidate
    return {"low": x[k - 1], "high": x[n - k], "coverage": coverage(k)}


def paired_by_index(rustmc_fits: list[dict[str, Any]],
                    other_fits: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """The fits of the same dataset, matched by index, in rustmc's order. A repeated
    index would let one dataset count as several pairs, so it raises."""
    for fits in (rustmc_fits, other_fits):
        indices = [fit["index"] for fit in fits]
        if len(set(indices)) != len(indices):
            raise ValueError(f"repeated dataset index in {sorted(indices)}")
    by_index = {fit["index"]: fit for fit in other_fits}
    return [(a, by_index[a["index"]]) for a in rustmc_fits if a["index"] in by_index]


def compare(rustmc: dict[str, Any], other: dict[str, Any], parameters: int,
            time_key: str = "seconds", provenance_ok: bool = True) -> dict[str, Any]:
    """Paired comparison of rustmc with one nutpie run of the same cell.

    Efficiency ratios are rustmc over nutpie, per dataset. The verdict uses only
    pairs where both fits pass `fit_problems`. In order: a rustmc extension that
    differs from the record's gives "inconclusive (provenance)"; no pair where both
    fits finished (an engine that failed to compile has none) gives "inconclusive
    (no finished pairs)"; a mean-difference test across every parameter of every
    finished pair that crosses its threshold, or cannot be computed, gives
    "inconclusive (means differ)"; no passing pair
    gives "inconclusive (quality)"; otherwise a two-sided sign test at the 5% level,
    unadjusted for the cell being one of many, names a winner, and a non-significant
    one gives "no clear winner". `time_key` picks the span the per-ESS ratios use:
    "seconds", "sampling_seconds" or "native_seconds"."""
    pairs = paired_by_index(rustmc["fits"], other["fits"])
    finished = [(a, b) for a, b in pairs if a.get("status") == "ok" and b.get("status") == "ok"]
    passing = [(a, b) for a, b in finished if not fit_problems(a) and not fit_problems(b)]

    def per_ess(chosen, key=time_key):
        return [(a[key] / a["min_bulk_ess"]) / (b[key] / b["min_bulk_ess"]) for a, b in chosen]

    z = max_propagating_nan([mean_difference_z(a, b) for a, b in finished])
    tests = parameters * max(len(finished), 1)
    threshold = mean_difference_threshold(tests)
    # With no finished pair there are no means to compare, which is not a disagreement.
    means_flag = bool(finished) and not (z <= threshold)
    ratios = per_ess(passing)
    p_value = sign_test(ratios) if ratios else None
    if not provenance_ok:
        verdict = "inconclusive (provenance)"
    elif not finished:
        verdict = "inconclusive (no finished pairs)"
    elif means_flag:
        verdict = "inconclusive (means differ)"
    elif not ratios:
        verdict = "inconclusive (quality)"
    elif p_value < 0.05:
        verdict = winner(ratios)
    else:
        verdict = "no clear winner"
    return {
        "pairs": len(pairs),
        "finished_pairs": len(finished),
        "passing_pairs": len(passing),
        # Every finished pair, passing or not, so nothing measured is hidden.
        "all_ratio_seconds_per_ess": per_ess(finished),
        "all_ratio_seconds": [a[time_key] / b[time_key] for a, b in finished],
        # The pairs the verdict uses.
        "ratio_seconds_per_ess": ratios,
        "sign_test_p": p_value,
        "median_interval": median_interval(ratios) if ratios else None,
        "verdict": verdict,
        "time_key": time_key,
        "max_mean_difference_z": z,
        "mean_difference_threshold": threshold,
        "mean_difference_tests": tests,
        "mean_difference_flag": means_flag,
        "provenance_ok": provenance_ok,
    }


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm-adjusted p-values, valid under any dependence; None stays None."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


# ── engines ────────────────────────────────────────────────────────────────

def cache_files(path: str | None) -> int | None:
    if not path or not os.path.isdir(path):
        return None
    return sum(len(files) for _, _, files in os.walk(path))


def compiled_cache_files() -> int:
    """Files nutpie's compilers have cached: PyTensor's, under the `compiledir_*`
    directories of the `base_compiledir` in PYTENSOR_FLAGS, and Numba's index and
    data files in NUMBA_CACHE_DIR. Other files that share the directory, such as
    ArviZ's and Matplotlib's, do not count."""
    flags = dict(flag.split("=", 1) for flag in os.environ.get("PYTENSOR_FLAGS", "").split(",")
                 if "=" in flag)
    base = flags.get("base_compiledir")
    numba = os.environ.get("NUMBA_CACHE_DIR")
    total = 0
    if numba and os.path.isdir(numba):
        total = sum(1 for _, _, files in os.walk(numba) for name in files
                    if name.endswith((".nbi", ".nbc")))
    if base and os.path.isdir(base):
        total += sum(cache_files(str(entry)) or 0 for entry in Path(base).iterdir()
                     if entry.is_dir() and entry.name.startswith("compiledir_"))
    return total


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rustmc_model(cell: Cell):
    """The cell's model compiled by rustmc; it takes data from `rustmc_data`."""
    import rustmc as rmc

    b = rmc.ModelBuilder()
    if cell.model == "eight_schools":
        # The known per-school scale divides both sides: y / sigma ~ Normal(theta /
        # sigma, 1), whose log density differs from y ~ Normal(theta, sigma) only by
        # the constant sum of log sigma, so the posterior is the same.
        mu = b.normal_prior("mu", 0.0, 5.0)
        tau = b.half_normal_prior("tau", 5.0)
        z = b.vector_normal_prior("z", n=8, mu=0.0, sigma=1.0)
        b.normal_likelihood("obs", (mu + tau * z["school"]) * "inv_sigma", 1.0, "y_scaled")
    elif cell.model == "varying_intercept":
        mu = b.normal_prior("mu", 0.0, 5.0)
        tau = b.half_normal_prior("tau", 1.0)
        z = b.vector_normal_prior("z", n=cell.columns, mu=0.0, sigma=1.0)
        sigma = b.half_normal_prior("sigma", 1.0)
        b.normal_likelihood("obs", mu + tau * z["group"], sigma, "y")
    else:
        alpha = b.normal_prior("alpha", 0.0, 5.0)
        beta = b.vector_normal_prior("beta", n=cell.columns, mu=0.0, sigma=1.0)
        eta = alpha + beta @ "X"
        if cell.model == "normal":
            b.normal_likelihood("obs", eta, 1.0, "y")
        elif cell.model == "normal_sigma":
            b.normal_likelihood("obs", eta, b.half_normal_prior("sigma", 1.0), "y")
        elif cell.model == "logistic":
            b.bernoulli_logit_likelihood("obs", eta, "y")
        else:
            b.poisson_log_likelihood("obs", eta, "y")
    return b.compile()


def rustmc_data(cell: Cell, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Dataset `data` of `cell` as the rustmc model binds it."""
    if cell.model == "eight_schools":
        return {"school": np.arange(8.0), "inv_sigma": 1.0 / data["sigma"],
                "y_scaled": data["y"] / data["sigma"]}
    if cell.model == "varying_intercept":
        return {"group": data["group"].astype(float), "y": data["y"]}
    return data


def build_pymc_model(cell: Cell, first: dict[str, np.ndarray]):
    """The cell's PyMC model, holding dataset `first`; nutpie's `with_data` swaps
    in another."""
    import pymc as pm

    with pm.Model() as model:
        if cell.model == "eight_schools":
            mu = pm.Normal("mu", 0.0, 5.0)
            tau = pm.HalfNormal("tau", 5.0)
            z = pm.Normal("z", 0.0, 1.0, shape=8)
            pm.Normal("obs", mu + tau * z, first["sigma"], observed=first["y"])
        elif cell.model == "varying_intercept":
            group = pm.Data("group", first["group"])
            y = pm.Data("y", first["y"])
            mu = pm.Normal("mu", 0.0, 5.0)
            tau = pm.HalfNormal("tau", 1.0)
            z = pm.Normal("z", 0.0, 1.0, shape=cell.columns)
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal("obs", mu + tau * z[group], sigma, observed=y, shape=group.shape[0])
        else:
            x = pm.Data("X", first["X"])
            y = pm.Data("y", first["y"])
            alpha = pm.Normal("alpha", 0.0, 5.0)
            beta = pm.Normal("beta", 0.0, 1.0, shape=cell.columns)
            eta = alpha + pm.math.dot(x, beta)
            rows = x.shape[0]
            if cell.model == "normal":
                pm.Normal("obs", eta, 1.0, observed=y, shape=rows)
            elif cell.model == "normal_sigma":
                pm.Normal("obs", eta, pm.HalfNormal("sigma", 1.0), observed=y, shape=rows)
            elif cell.model == "logistic":
                pm.Bernoulli("obs", logit_p=eta, observed=y, shape=rows)
            else:
                pm.Poisson("obs", mu=pm.math.exp(eta), observed=y, shape=rows)
    return model


def prepare_rustmc(cell: Cell) -> tuple[dict[str, Any], Callable[[int, dict], dict]]:
    import rustmc as rmc
    import rustmc._rustmc as extension

    from benchmarks.build_provenance import build_provenance

    meta: dict[str, Any] = {
        "provenance": {"rustmc": build_provenance(rmc),
                       "extension_sha256": file_sha256(extension.__file__)},
        "compile_cache_files_before": None,
    }
    start = time.perf_counter()
    compiled = build_rustmc_model(cell)
    meta["compile_seconds"] = time.perf_counter() - start
    names = scalar_names(cell)

    def fit_one(index: int, data: dict) -> dict:
        bound = rustmc_data(cell, data)
        start = time.perf_counter()
        fit = compiled.sample(bound, chains=cell.chains, warmup=cell.warmup, draws=cell.draws,
                              seed=cell.data_seed + index, threads=cell.threads,
                              show_progress=False)
        sampled = time.perf_counter()
        # The matched export and the default one, for the native span, each
        # timed on its own; whichever runs second finds caches warm, so they
        # take turns going first.
        conversions: dict[str, float] = {}
        for kind in (("matched", "native") if index % 2 == 0 else ("native", "matched")):
            begin = time.perf_counter()
            if kind == "matched":
                idata = fit.to_arviz(include_log_likelihood=False)
            else:
                fit.to_arviz()
            conversions[kind] = time.perf_counter() - begin
        report = fit.transition_diagnostics()
        samples = fit.get_samples_2d()
        diverging = np.asarray(idata.sample_stats["diverging"].values)
        draws = {name: np.asarray(samples[name], dtype=float) for name in names}
        return {"seconds": (sampled - start) + conversions["matched"],
                "sampling_seconds": sampled - start,
                "native_seconds": (sampled - start) + conversions["native"],
                "matched_conversion_first": index % 2 == 0,
                "warmup": cell.warmup, "groups": list(idata.groups()),
                "leapfrog_steps": int(report["total_leapfrog_steps"]),
                **draw_statistics(draws, diverging)}

    return meta, fit_one


def prepare_nutpie(cell: Cell, tune: int | None = None) -> tuple[dict[str, Any], Callable[[int, dict], dict]]:
    import nutpie
    import pandas as pd
    import pymc  # noqa: F401 - imported before the compile timing starts, as before

    # `nutpie.sample` is the function; its module holds the conversion.
    sample_module = sys.modules["nutpie.sample"]
    first = make_data(cell, 0)
    meta: dict[str, Any] = {"provenance": {}, "compile_cache_files_before": compiled_cache_files()}
    start = time.perf_counter()
    model = build_pymc_model(cell, first)
    compiled = nutpie.compile_pymc_model(model)
    meta["compile_seconds"] = time.perf_counter() - start
    coords = {name: pd.Index(values) for name, values in compiled.coords.items()}

    def convert(sampler, draw_batches, stat_batches):
        # As `_BackgroundSampler._extract` converts an arrow trace.
        skip_vars = []
        for setting, names in {
            "store_gradient": ["gradient"],
            "store_unconstrained": ["unconstrained_draw"],
            "store_mass_matrix": ["mass_matrix_inv", "mass_matrix_eigvals", "mass_matrix_stds"],
            "store_divergences": ["divergence_start", "divergence_end", "divergence_momentum",
                                  "divergence_start_gradient"],
        }.items():
            if not getattr(sampler._settings, setting, False):
                skip_vars.extend(names)
        return sample_module._arrow_to_arviz(draw_batches, stat_batches, skip_vars=skip_vars,
                                             coords=coords, save_warmup=False)

    def fit_one(index: int, data: dict) -> dict:
        start = time.perf_counter()
        if cell.model == "eight_schools":
            bound = compiled
        elif cell.model == "varying_intercept":
            bound = compiled.with_data(group=data["group"], y=data["y"])
        else:
            bound = compiled.with_data(X=data["X"], y=data["y"])
        sampler = nutpie.sample(bound, draws=cell.draws, tune=tune or cell.warmup,
                                chains=cell.chains, cores=cell.threads,
                                seed=cell.data_seed + index, save_warmup=False,
                                progress_bar=False, blocking=False)
        sampler._sampler.wait(None)
        sampled = time.perf_counter()
        # The trace can be taken only once; the batches can be read again.
        draw_batches, stat_batches = sampler._sampler.take_results().get_arrow_trace()
        taken = time.perf_counter()
        # The matched conversion drops the warmup rows first, so it handles the
        # content rustmc's does; the native one converts the whole trace as
        # `nutpie.sample(save_warmup=False)` would. Each is timed on its own,
        # taking turns going first, as rustmc's are.
        conversions: dict[str, float] = {}
        for kind in (("matched", "native") if index % 2 == 0 else ("native", "matched")):
            begin = time.perf_counter()
            if kind == "matched":
                tuning = [int(batch.column("tuning").to_numpy().sum()) for batch in stat_batches]
                idata = convert(sampler,
                                [b.slice(t, b.num_rows - t) for b, t in zip(draw_batches, tuning)],
                                [b.slice(t, b.num_rows - t) for b, t in zip(stat_batches, tuning)])
            else:
                convert(sampler, draw_batches, stat_batches)
            conversions[kind] = time.perf_counter() - begin
        steps = sum(int(batch.column("n_steps").to_numpy().sum()) for batch in stat_batches)
        posterior = idata.posterior
        draws: dict[str, np.ndarray] = {}
        for name, length in parameters(cell):
            values = np.asarray(posterior[name].values, dtype=float)
            if length == 0:
                draws[name] = values
            else:
                for k in range(length):
                    draws[f"{name}[{k}]"] = values[:, :, k]
        return {"seconds": (taken - start) + conversions["matched"],
                "sampling_seconds": sampled - start,
                "native_seconds": (taken - start) + conversions["native"],
                "matched_conversion_first": index % 2 == 0,
                "warmup": tune or cell.warmup, "groups": list(idata.groups()),
                "leapfrog_steps": steps,
                **draw_statistics(draws, np.asarray(idata.sample_stats["diverging"].values))}

    return meta, fit_one


PREPARE: dict[str, Callable[[Cell], tuple[dict[str, Any], Callable[[int, dict], dict]]]] = {
    "rustmc": prepare_rustmc,
    "nutpie": prepare_nutpie,
    "nutpie_default_tune": lambda cell: prepare_nutpie(cell, NUTPIE_DEFAULT_TUNE),
}


def run_worker(engine: str, cell: Cell) -> int:
    """Compile, report, then fit the datasets the parent asks for, one per line
    of stdin, until it sends null."""
    def emit(payload: dict[str, Any]) -> None:
        print(RESULT_PREFIX + json.dumps(payload), flush=True)

    try:
        import arviz  # noqa: F401 - imported before any timing

        meta, fit_one = PREPARE[engine](cell)
        emit({"status": "ready", **meta})
    except Exception as error:  # noqa: BLE001 - failures become result data
        emit({"status": "error", "reason": f"{type(error).__name__}: {error}",
              "traceback": traceback.format_exc()[-8_000:]})
        return 0
    for line in sys.stdin:
        index = json.loads(line)
        if index is None:
            break
        try:
            emit({"index": index, "status": "ok", **fit_one(index, make_data(cell, index))})
        except Exception as error:  # noqa: BLE001 - a failed fit becomes result data
            emit({"index": index, "status": "error", "reason": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()[-4_000:]})
    return 0


# ── parent ─────────────────────────────────────────────────────────────────

def wait_for_quiet(max_load: float, timeout: float = LOAD_WAIT_SECONDS) -> tuple[float, bool]:
    """Wait until the one-minute load is below `max_load`. Returns the load and
    whether the wait timed out first."""
    deadline = time.monotonic() + timeout
    while True:
        load = os.getloadavg()[0]
        if load < max_load:
            return load, False
        if time.monotonic() > deadline:
            return load, True
        time.sleep(10)


class Worker:
    """One engine's worker subprocess for one cell."""

    def __init__(self, engine: str, cell: Cell, cache_dir: str):
        command = [sys.executable, "-m", "benchmarks.head_to_head", "--child-engine", engine,
                   "--child-cell", json.dumps(asdict(cell))]
        # Both nutpie runs take nutpie's environment, caches included.
        env_engine = "nutpie" if engine.startswith("nutpie") else engine
        env = _child_environment(env_engine, replace(BenchmarkConfig(), threads=cell.threads),
                                 cache_dir)
        self.engine = engine
        self.stderr = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent.parent),
                                     env=env, text=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr)

    def read(self) -> dict[str, Any] | None:
        for line in self.proc.stdout:
            if line.startswith(RESULT_PREFIX):
                return json.loads(line.removeprefix(RESULT_PREFIX))
        return None

    def stderr_tail(self) -> str:
        self.stderr.seek(0)
        return self.stderr.read()[-4_000:]

    def fit(self, index: int) -> dict[str, Any]:
        try:
            self.proc.stdin.write(json.dumps(index) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        result = self.read()
        if result is None:
            return {"index": index, "status": "error", "reason": "worker exited",
                    "returncode": self.proc.poll(), "stderr": self.stderr_tail()}
        return result

    def close(self) -> None:
        try:
            self.proc.stdin.write("null\n")
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.stderr.close()


def run_cell(cell: Cell, engines: list[str], position: int, cache_dir: str,
             max_load: float) -> dict[str, dict[str, Any]]:
    """Compile every engine, then fit dataset by dataset, interleaving engines and
    rotating their order, with a load check before each compile and fit."""
    results: dict[str, dict[str, Any]] = {}
    workers: dict[str, Worker] = {}
    try:
        for engine in engines:
            load, timed_out = wait_for_quiet(max_load)
            worker = Worker(engine, cell, cache_dir)
            ready = worker.read()
            result: dict[str, Any] = {"engine": engine, "fits": [], "compile_load": load,
                                      "compile_load_wait_timed_out": timed_out}
            if ready is None or ready.get("status") != "ready":
                result.update(status="error",
                              reason=(ready or {}).get("reason", "worker exited before compiling"),
                              traceback=(ready or {}).get("traceback"),
                              stderr=worker.stderr_tail())
                worker.close()
            else:
                result.update({key: value for key, value in ready.items() if key != "status"})
                result["status"] = "ok"
                workers[engine] = worker
            results[engine] = result
        for index in range(cell.datasets):
            shift = (position + index) % len(engines)
            for engine in engines[shift:] + engines[:shift]:
                if engine not in workers:
                    continue
                load, timed_out = wait_for_quiet(max_load)
                fit = workers[engine].fit(index)
                fit.update(load_at_start=load, load_wait_timed_out=timed_out,
                           order=(engines[shift:] + engines[:shift]).index(engine))
                results[engine]["fits"].append(fit)
                if fit.get("reason") == "worker exited":
                    workers.pop(engine).close()
    finally:
        for worker in workers.values():
            worker.close()
    for result in results.values():
        failed = sum(fit.get("status") != "ok" for fit in result["fits"])
        if result["status"] == "ok" and failed:
            result["status"] = "partial"
    return results


def summarize_cell(cell: Cell, results: dict[str, dict[str, Any]], extension_sha256: str) -> dict[str, Any]:
    """Per engine figures over its finished fits, and rustmc's paired comparison
    with each nutpie run."""
    out: dict[str, Any] = {}
    for engine, result in results.items():
        finished = [fit for fit in result.get("fits", []) if fit.get("status") == "ok"]
        out[engine] = {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "compile_seconds": result.get("compile_seconds"),
            "compile_cache_files_before": result.get("compile_cache_files_before"),
            "seconds": [fit["seconds"] for fit in finished],
            "sampling_seconds": [fit["sampling_seconds"] for fit in finished],
            "native_seconds": [fit["native_seconds"] for fit in finished],
            "seconds_per_min_bulk_ess": [fit["seconds"] / fit["min_bulk_ess"] for fit in finished],
            # Sampling wall time over leapfrog steps: amortized, so it includes
            # everything a sampler does besides gradients, and with several
            # threads it divides concurrent chains' work by their total steps.
            "amortized_us_per_step": [1e6 * fit["sampling_seconds"] / fit["leapfrog_steps"]
                                      for fit in finished],
            "leapfrog_steps": [fit["leapfrog_steps"] for fit in finished],
            "min_bulk_ess": [fit["min_bulk_ess"] for fit in finished],
            "failed_checks": [fit["index"] for fit in result.get("fits", []) if fit_problems(fit)],
        }
    sha = results.get("rustmc", {}).get("provenance", {}).get("extension_sha256")
    provenance_ok = sha is not None and sha == extension_sha256
    for other in ("nutpie", "nutpie_default_tune"):
        if other in results and results["rustmc"].get("fits") and results[other].get("fits"):
            out[f"vs_{other}"] = compare(results["rustmc"], results[other],
                                         len(scalar_names(cell)), "seconds", provenance_ok)
    return out


def default_cells(quick: bool) -> list[Cell]:
    sizes = [(100, 4), (1000, 16), (10000, 16), (1000, 64), (1000, 128)]
    models = []
    for model in REGRESSIONS:
        models += [(model, rows, columns) for rows, columns in sizes]
    models += [("varying_intercept", 500, 20), ("varying_intercept", 5000, 100),
               ("eight_schools", 8, 8)]
    if quick:
        models = [("normal", 100, 4), ("poisson", 1000, 16), ("varying_intercept", 500, 20),
                  ("eight_schools", 8, 8)]
    cells = []
    for model, rows, columns in models:
        for threads in (1, 4):
            cell = Cell(model, rows, columns, threads)
            if quick:
                cell = replace(cell, datasets=2, warmup=200, draws=200)
            cells.append(cell)
    return cells


def _revision() -> dict[str, Any]:
    from benchmarks.repeated import _revision as revision

    return revision()


def number(values: list[float], digits: int = 3) -> str:
    """Median and observed range, as `median [low, high]`; not an interval."""
    if not values:
        return "-"
    return (f"{np.median(values):.{digits}g} [{min(values):.{digits}g}, "
            f"{max(values):.{digits}g}]")


def print_table(rows: list[dict[str, Any]]) -> None:
    """`fit s` runs to the InferenceData described above; `sampling s` stops when
    sampling does. Values are the median and observed range over datasets. Ratio
    rows are rustmc over nutpie per ESS, over pairs that pass both engines' checks,
    with an unadjusted per-cell sign test."""
    print(f"{'cell':<36} {'engine':<9} {'fit s':<26} {'sampling s':<26} "
          f"{'s per min ESS (ms)':<26} {'min ESS':<22} {'compile s':>9} {'failed':>6}")
    for row in rows:
        summary = row["summary"]
        for engine in ENGINES:
            s = summary.get(engine)
            if s is None:
                continue
            label = {"nutpie_default_tune": "nutpie300"}.get(engine, engine)
            if s["status"] == "error":
                print(f"{row['cell']:<36} {label:<9} error: {s['reason']}")
                continue
            per_ess_ms = [1e3 * v for v in s["seconds_per_min_bulk_ess"]]
            compile_s = "-" if s["compile_seconds"] is None else f"{s['compile_seconds']:.2f}"
            print(f"{row['cell']:<36} {label:<9} {number(s['seconds']):<26} "
                  f"{number(s['sampling_seconds']):<26} {number(per_ess_ms):<26} "
                  f"{number(s['min_bulk_ess'], 4):<22} {compile_s:>9} "
                  f"{len(s['failed_checks']):>6}")
        for other, label in (("nutpie", "r/n"), ("nutpie_default_tune", "r/n300")):
            c = summary.get(f"vs_{other}")
            if c:
                p = "-" if c["sign_test_p"] is None else f"{c['sign_test_p']:.3f}"
                print(f"{'':<36} {label:<9} per ESS {number(c['ratio_seconds_per_ess'])} over "
                      f"{c['passing_pairs']}/{c['pairs']} passing pairs, unadjusted sign test p {p}: "
                      f"{c['verdict']}; max mean-difference z {c['max_mean_difference_z']:.1f} "
                      f"(threshold {c['mean_difference_threshold']:.1f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the complete JSON record here")
    parser.add_argument("--quick", action="store_true", help="a few small cells, short chains")
    parser.add_argument("--model", action="append", help="only cells of this model; repeatable")
    parser.add_argument("--threads", type=int, action="append", help="only these thread counts")
    parser.add_argument("--datasets", type=int, help="datasets per cell, overriding the default")
    parser.add_argument("--lock", type=Path, default=Path(tempfile.gettempdir()) / "rustmc-bench.lock")
    parser.add_argument("--max-load", type=float, default=1.5)
    parser.add_argument("--cache-dir", type=Path,
                        help="PyTensor and Numba caches, shared by every cell; a new temporary "
                             "directory by default")
    parser.add_argument("--child-engine", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--child-cell", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_engine:
        return run_worker(args.child_engine, Cell(**json.loads(args.child_cell)))

    cells = default_cells(args.quick)
    if args.model:
        cells = [cell for cell in cells if cell.model in args.model]
    if args.threads:
        cells = [cell for cell in cells if cell.threads in args.threads]
    if args.datasets:
        cells = [replace(cell, datasets=args.datasets) for cell in cells]
    cache_dir = str(args.cache_dir or tempfile.mkdtemp(prefix="rustmc-head-to-head-"))
    import rustmc._rustmc as extension

    record: dict[str, Any] = {
        "schema": 2,
        "revision": _revision(),
        "extension_sha256": file_sha256(extension.__file__),
        "versions": {name: package_version(name) for name in ("rustmc", "nutpie", "pymc", "arviz")},
        "environment": environment_metadata(BenchmarkConfig()),
        "command": " ".join(sys.argv).replace(str(Path.home()), "~"),
        "compile_cache": {"directory_existed": os.path.isdir(cache_dir),
                          "files_at_start": cache_files(cache_dir) or 0},
        "max_load": args.max_load,
        "cells": [],
    }
    args.lock.touch(exist_ok=True)
    for position, cell in enumerate(cells):
        engines = [engine for engine in ENGINES
                   if engine != "nutpie_default_tune" or cell.model in REGRESSIONS]
        with args.lock.open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            started = time.time()
            results = run_cell(cell, engines, position, cache_dir, args.max_load)
        row = {"cell": cell.name, "config": asdict(cell), "engines": engines,
               "seconds": time.time() - started, "results": results,
               "summary": summarize_cell(cell, results, record["extension_sha256"])}
        record["cells"].append(row)
        print_table([row])
        sys.stdout.flush()
        if args.output:
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
