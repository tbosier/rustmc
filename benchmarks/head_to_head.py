"""rustmc against nutpie on matched models, one cell at a time.

A cell is one model at one size and one thread setting. For each cell, each engine
runs in a fresh subprocess with the environment ``benchmarks.run`` gives it, builds
and compiles its model once, then fits the same datasets in turn, with the same
priors, likelihood, chains, warmup and draws, and each engine's own defaults for
everything else (target acceptance 0.8 and its own metric and step-size
adaptation). The engines' order alternates from cell to cell. Before each cell the
parent takes an exclusive ``flock`` on ``--lock`` and waits for the one-minute load
average to fall below ``--max-load``, so other benchmarks on the machine neither
overlap a cell nor are overlapped by it.

Per fit it records two wall times, without compilation: from bound data to the
end of sampling, and from bound data to an ArviZ InferenceData, which nutpie
builds inside ``nutpie.sample`` and rustmc builds in ``FitResult.to_arviz``. The
second span is the comparable end point and the one the ratios use; nutpie's
sampling-only span reads ``_BackgroundSampler`` internals of nutpie 0.16.6. It
also records the leapfrog steps of warmup and draws, the
minimum and mean bulk ESS and minimum tail ESS over the model's parameters, R-hat,
post-warmup divergences, and each parameter's mean and Monte Carlo standard error,
from which the parent checks that the engines agree. Compilation is timed once
per cell and engine. Regression cells also run nutpie with its own default
warmup of 300 iterations, as ``nutpie_default_tune``, since that is what a user
gets without asking; the matched comparison keeps the same warmup for both.

    python -m benchmarks.head_to_head --output results.json [--quick] [--model NAME]

The rustmc this measures is whichever ``import rustmc`` finds; point PYTHONPATH at
a fresh build (``scripts/dev_pytest.sh`` shows how), and the record says which
revision it was.
"""

from __future__ import annotations

import argparse
import fcntl
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
# The matched comparison; `nutpie_default_tune` is an extra column.
MATCHED = ("rustmc", "nutpie")
NUTPIE_DEFAULT_TUNE = 300
REGRESSIONS = ("normal", "normal_sigma", "logistic", "poisson")


@dataclass(frozen=True)
class Cell:
    """One model at one size, fitted on `datasets` datasets with `threads` threads."""

    model: str
    rows: int
    columns: int
    threads: int
    datasets: int = 3
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


# ── statistics, the same function for both engines ────────────────────────

def draw_statistics(draws: dict[str, np.ndarray], diverging: np.ndarray) -> dict[str, Any]:
    """ESS, R-hat and per-parameter means and MCSEs from `name -> (chains, draws)`."""
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
        "min_tail_ess": float(min(float(tail[name]) for name in names)),
        "max_rhat": float(max(float(rhat[name]) for name in names)),
        "divergences": int(np.sum(diverging)),
        "means": {name: float(np.mean(values)) for name, values in draws.items()},
        "mcse": {name: float(mcse[name]) for name in names},
    }


def scalar_names(cell: Cell) -> list[str]:
    return [name if length == 0 else f"{name}[{k}]"
            for name, length in parameters(cell) for k in range(max(length, 1))]


# ── engines ────────────────────────────────────────────────────────────────

def rustmc_engine(cell: Cell, datasets: list[dict[str, np.ndarray]]) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    import rustmc as rmc

    from benchmarks.build_provenance import build_provenance

    start = time.perf_counter()
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
    compiled = b.compile()
    compile_seconds = time.perf_counter() - start

    names = scalar_names(cell)
    fits = []
    for index, data in enumerate(datasets):
        if cell.model == "eight_schools":
            bound = {"school": np.arange(8.0), "inv_sigma": 1.0 / data["sigma"],
                     "y_scaled": data["y"] / data["sigma"]}
        elif cell.model == "varying_intercept":
            bound = {"group": data["group"].astype(float), "y": data["y"]}
        else:
            bound = data
        start = time.perf_counter()
        fit = compiled.sample(bound, chains=cell.chains, warmup=cell.warmup, draws=cell.draws,
                              seed=cell.data_seed + index, threads=cell.threads,
                              show_progress=False)
        sampled = time.perf_counter()
        idata = fit.to_arviz()
        seconds = time.perf_counter() - start
        report = fit.transition_diagnostics()
        samples = fit.get_samples_2d()
        diverging = np.asarray(idata.sample_stats["diverging"].values)
        draws = {name: np.asarray(samples[name], dtype=float) for name in names}
        fits.append({"index": index, "seconds": seconds, "sampling_seconds": sampled - start,
                     "leapfrog_steps": int(report["total_leapfrog_steps"]),
                     **draw_statistics(draws, diverging)})
    return compile_seconds, fits, {"rustmc": build_provenance(rmc)}


def nutpie_engine(cell: Cell, datasets: list[dict[str, np.ndarray]],
                  tune: int | None = None) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    import nutpie
    import pymc as pm

    first = datasets[0]
    start = time.perf_counter()
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
    compiled = nutpie.compile_pymc_model(model)
    compile_seconds = time.perf_counter() - start

    fits = []
    for index, data in enumerate(datasets):
        start = time.perf_counter()
        if cell.model == "eight_schools":
            bound = compiled
        elif cell.model == "varying_intercept":
            bound = compiled.with_data(group=data["group"], y=data["y"])
        else:
            bound = compiled.with_data(X=data["X"], y=data["y"])
        # `blocking=False` returns the running sampler, so the end of sampling
        # and the end of the InferenceData conversion that `nutpie.sample` would
        # do next can be timed apart; `wait()` is that wait plus `_extract`.
        sampler = nutpie.sample(bound, draws=cell.draws, tune=tune or cell.warmup,
                                chains=cell.chains, cores=cell.threads,
                                seed=cell.data_seed + index, save_warmup=True,
                                progress_bar=False, blocking=False)
        sampler._sampler.wait(None)
        sampled = time.perf_counter()
        idata = sampler._extract(sampler._sampler.take_results())
        seconds = time.perf_counter() - start
        stats = idata.sample_stats
        steps = int(stats["n_steps"].values.sum() + idata.warmup_sample_stats["n_steps"].values.sum())
        posterior = idata.posterior
        draws: dict[str, np.ndarray] = {}
        for name, length in parameters(cell):
            values = np.asarray(posterior[name].values, dtype=float)
            if length == 0:
                draws[name] = values
            else:
                for k in range(length):
                    draws[f"{name}[{k}]"] = values[:, :, k]
        fits.append({"index": index, "seconds": seconds, "sampling_seconds": sampled - start,
                     "warmup": tune or cell.warmup, "leapfrog_steps": steps,
                     **draw_statistics(draws, np.asarray(stats["diverging"].values))})
    return compile_seconds, fits, {}


ADAPTERS: dict[str, Callable[..., Any]] = {
    "rustmc": rustmc_engine,
    "nutpie": nutpie_engine,
    "nutpie_default_tune": lambda cell, datasets: nutpie_engine(cell, datasets, NUTPIE_DEFAULT_TUNE),
}


def run_child(engine: str, cell: Cell) -> dict[str, Any]:
    payload: dict[str, Any] = {"engine": engine, "cell": asdict(cell)}
    try:
        import arviz  # noqa: F401 - imported before any timing

        datasets = [make_data(cell, index) for index in range(cell.datasets)]
        compile_seconds, fits, provenance = ADAPTERS[engine](cell, datasets)
        payload.update(status="ok", compile_seconds=compile_seconds, fits=fits, provenance=provenance)
    except Exception as error:  # noqa: BLE001 - failures become result data
        payload.update(status="error", reason=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc()[-8_000:])
    return payload


# ── parent ─────────────────────────────────────────────────────────────────

def wait_for_quiet(max_load: float, timeout: float = 1800.0) -> float:
    """Wait until the one-minute load is below `max_load`; return it."""
    deadline = time.monotonic() + timeout
    while True:
        load = os.getloadavg()[0]
        if load < max_load or time.monotonic() > deadline:
            return load
        time.sleep(10)


def run_isolated(engine: str, cell: Cell, cache_dir: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "benchmarks.head_to_head", "--child-engine", engine,
               "--child-cell", json.dumps(asdict(cell))]
    # Both nutpie runs take nutpie's environment, caches included.
    env_engine = "nutpie" if engine.startswith("nutpie") else engine
    env = _child_environment(env_engine, replace(BenchmarkConfig(), threads=cell.threads), cache_dir)
    proc = subprocess.run(command, cwd=str(Path(__file__).resolve().parent.parent), env=env,
                          text=True, capture_output=True, check=False)
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith(RESULT_PREFIX)), None)
    if line is None:
        return {"engine": engine, "cell": asdict(cell), "status": "error",
                "reason": "child produced no machine-readable result",
                "returncode": proc.returncode, "stderr": proc.stderr[-4_000:]}
    return json.loads(line.removeprefix(RESULT_PREFIX))


def agreement(rustmc: dict[str, Any], nutpie: dict[str, Any]) -> float:
    """Largest |difference of means| / combined MCSE over the parameters."""
    worst = 0.0
    for name, mean in rustmc["means"].items():
        spread = math.hypot(rustmc["mcse"][name], nutpie["mcse"][name])
        if spread > 0:
            worst = max(worst, abs(mean - nutpie["means"][name]) / spread)
    return worst


def summarize_cell(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per engine medians and ranges, and the paired rustmc / nutpie ratios."""
    out: dict[str, Any] = {}
    for engine, result in results.items():
        if result.get("status") != "ok":
            out[engine] = {"status": result.get("status"), "reason": result.get("reason")}
            continue
        fits = result["fits"]
        per_ess = [fit["seconds"] / fit["min_bulk_ess"] for fit in fits]
        out[engine] = {
            "status": "ok",
            "compile_seconds": result["compile_seconds"],
            "seconds": [fit["seconds"] for fit in fits],
            "sampling_seconds": [fit["sampling_seconds"] for fit in fits],
            "seconds_per_min_bulk_ess": per_ess,
            "leapfrog_steps": [fit["leapfrog_steps"] for fit in fits],
            "min_bulk_ess": [fit["min_bulk_ess"] for fit in fits],
            "min_tail_ess": [fit["min_tail_ess"] for fit in fits],
            "max_rhat": max(fit["max_rhat"] for fit in fits),
            "divergences": sum(fit["divergences"] for fit in fits),
        }
    # Paired by dataset: rustmc over each nutpie run that finished.
    for other in ("nutpie", "nutpie_default_tune"):
        if out.get("rustmc", {}).get("status") != "ok" or out.get(other, {}).get("status") != "ok":
            continue
        r, n = results["rustmc"]["fits"], results[other]["fits"]
        out[f"vs_{other}"] = {
            "ratio_seconds": [a["seconds"] / b["seconds"] for a, b in zip(r, n)],
            "ratio_sampling_seconds": [
                a["sampling_seconds"] / b["sampling_seconds"] for a, b in zip(r, n)
            ],
            "ratio_seconds_per_ess": [
                (a["seconds"] / a["min_bulk_ess"]) / (b["seconds"] / b["min_bulk_ess"])
                for a, b in zip(r, n)
            ],
            "max_agreement_z": max(agreement(a, b) for a, b in zip(r, n)),
        }
    return out


def default_cells(quick: bool) -> list[Cell]:
    sizes = [(100, 4), (1000, 16), (10000, 16), (1000, 64), (1000, 128)]
    models = []
    for model in ("normal", "normal_sigma", "logistic", "poisson"):
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
    """Median and range, as `median [low, high]`."""
    if not values:
        return "-"
    return (f"{np.median(values):.{digits}g} [{min(values):.{digits}g}, "
            f"{max(values):.{digits}g}]")


def print_table(rows: list[dict[str, Any]]) -> None:
    """Fit seconds run to an InferenceData; `sampling` stops when sampling does.
    Ratio rows are rustmc over nutpie, paired by dataset: median [low, high]."""
    print(f"{'cell':<36} {'engine':<9} {'fit s':<26} {'sampling s':<26} "
          f"{'s per min ESS (ms)':<26} {'min ESS':<22} {'compile s':>9} {'rhat':>6} {'div':>4}")
    for row in rows:
        summary = row["summary"]
        for engine in ENGINES:
            s = summary.get(engine)
            label = {"nutpie_default_tune": "nutpie300"}.get(engine, engine)
            if s is None:
                continue
            if s.get("status") != "ok":
                print(f"{row['cell']:<36} {label:<9} {s.get('status')}: {s.get('reason')}")
                continue
            per_ess_ms = [1e3 * v for v in s["seconds_per_min_bulk_ess"]]
            print(f"{row['cell']:<36} {label:<9} {number(s['seconds']):<26} "
                  f"{number(s['sampling_seconds']):<26} {number(per_ess_ms):<26} "
                  f"{number(s['min_bulk_ess'], 4):<22} "
                  f"{s['compile_seconds']:>9.2f} {s['max_rhat']:>6.3f} {s['divergences']:>4}")
        for other, label in (("nutpie", "r/n"), ("nutpie_default_tune", "r/n300")):
            ratios = summary.get(f"vs_{other}")
            if ratios:
                print(f"{'':<36} {label:<9} {number(ratios['ratio_seconds']):<26} "
                      f"{number(ratios['ratio_sampling_seconds']):<26} "
                      f"{number(ratios['ratio_seconds_per_ess']):<26} "
                      f"agreement z {ratios['max_agreement_z']:.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the complete JSON record here")
    parser.add_argument("--quick", action="store_true", help="a few small cells, short chains")
    parser.add_argument("--model", action="append", help="only cells of this model; repeatable")
    parser.add_argument("--threads", type=int, action="append", help="only these thread counts")
    parser.add_argument("--lock", type=Path, default=Path(tempfile.gettempdir()) / "rustmc-bench.lock")
    parser.add_argument("--max-load", type=float, default=1.5)
    parser.add_argument("--cache-dir", type=Path,
                        help="PyTensor and Numba caches, shared by every cell; a new temporary "
                             "directory by default")
    parser.add_argument("--child-engine", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--child-cell", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_engine:
        cell = Cell(**json.loads(args.child_cell))
        print(RESULT_PREFIX + json.dumps(run_child(args.child_engine, cell)))
        return 0

    cells = default_cells(args.quick)
    if args.model:
        cells = [cell for cell in cells if cell.model in args.model]
    if args.threads:
        cells = [cell for cell in cells if cell.threads in args.threads]
    cache_dir = str(args.cache_dir or tempfile.mkdtemp(prefix="rustmc-head-to-head-"))
    record: dict[str, Any] = {
        "revision": _revision(),
        "versions": {name: package_version(name) for name in ("rustmc", "nutpie", "pymc", "arviz")},
        "environment": environment_metadata(BenchmarkConfig()),
        "cells": [],
    }
    args.lock.touch(exist_ok=True)
    for position, cell in enumerate(cells):
        engines = [engine for engine in ENGINES
                   if engine != "nutpie_default_tune" or cell.model in REGRESSIONS]
        shift = position % len(engines)
        engines = engines[shift:] + engines[:shift]
        with args.lock.open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            load = wait_for_quiet(args.max_load)
            started = time.time()
            results = {engine: run_isolated(engine, cell, cache_dir) for engine in engines}
        row = {"cell": cell.name, "config": asdict(cell), "engine_order": engines,
               "load_at_start": load, "seconds": time.time() - started,
               "results": results, "summary": summarize_cell(results)}
        record["cells"].append(row)
        print_table([row])
        sys.stdout.flush()
        if args.output:
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
