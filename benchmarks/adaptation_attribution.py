"""Why rustmc takes more leapfrog steps than nutpie: adaptation against tree building.

The 2026-10-27 head-to-head study found rustmc taking 2.66 times nutpie's leapfrog
steps on varying_intercept 500x20, 2.24 times on 5000x100 and 1.08 to 1.40 times
on the regressions, counting warmup and draws. This script splits that gap on
three of its cells (varying_intercept 500x20 and 5000x100, normal 1000x16),
datasets 0 to 2, with the study's models, data, seeds, 4 chains, 1000 warmup and
1000 draws, one thread.

For each dataset it runs:

- nutpie 0.16.6 with its defaults, storing its per-draw inverse metric
  (``store_mass_matrix``, the variances its diagonal metric uses, the same
  convention as rustmc) and unconstrained draws; and nutpie again with
  ``step_size_jitter=None``, since by default nutpie draws each transition's
  step size uniformly from 0.9 to 1.1 times its adapted one (nuts-rs 0.17.4,
  ``stepsize/adapt.rs``), which rustmc does not;
- rustmc with its own adaptation, through the private
  ``CompiledModel._experimental_nuts`` hook, which runs the production NUTS
  kernel and also returns each chain's final inverse metric and per-transition
  telemetry;
- rustmc with a frozen metric and step size (no adaptation of either; warmup
  only moves the chain), chain c starting from the last draw of rustmc's
  adapted chain c:

  - ``R/R``: rustmc's own final metric and step (checks the hook: it should
    take the steps rustmc's adapted draws took);
  - ``N/N``: nutpie chain c's final metric and adapted step (``step_size_bar``);
  - ``R/N``: rustmc's metric at nutpie's step; ``N/R``: nutpie's metric at
    rustmc's step;
  - ``N0/N0``: the final metric and step of the unjittered nutpie run;
  - ``X@0.8`` and ``X@a``: metric X at the one step size (shared by the
    chains) whose mean acceptance statistic is 0.8, the target both engines
    adapt to, or is what rustmc's adapted draws actually achieved, found by
    bisection on short frozen runs. X is ``R`` (rustmc's metric), ``Rdiag``
    (the metric of rustmc adapting with ``metric="diag"``), ``N`` (nutpie's), or
    ``dense``, the posterior covariance of the pooled unconstrained draws of
    both engines' adapted runs, which no adapted metric of either engine can
    represent.

It also runs rustmc's adaptation with ``metric="diag"``, since under the
default ``"auto"`` a vector parameter can get a dense block.

The draws gap splits exactly along two paths at matched acceptance. Path A:
rustmc's adapted run over ``R@0.8`` (where its step adaptation lands relative
to its own target), ``R@0.8`` over ``N@0.8`` (the metric), and ``N@0.8`` over
nutpie's draws (the kernel: tree building, mainly the U-turn test, and
nutpie's step jitter). Path B swaps the order of metric and step. The crossed
runs ``R/N`` and ``N/R`` are kept as data, but a metric run at a step it cannot
sustain diverges, so they do not split the gap. Warmup steps are reported
separately, by iteration bin, since the study's ratio counts them.

    python -m benchmarks.adaptation_attribution --output results.json

Run it under ``flock`` on a quiet machine; it waits for the one-minute load to
fall below ``--max-load`` before each dataset. Step counts do not depend on
load; no wall time is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.head_to_head import (Cell, build_pymc_model, build_rustmc_model, draw_statistics,
                                     make_data, parameters, rustmc_data, scalar_names,
                                     wait_for_quiet)

CELLS = (Cell("varying_intercept", 500, 20, 1), Cell("varying_intercept", 5000, 100, 1),
         Cell("normal", 1000, 16, 1))
DATASETS = (0, 1, 2)
BURN_IN = 300
CALIBRATION_WARMUP = 100
CALIBRATION_DRAWS = 300
CALIBRATION_ROUNDS = 8
# Parameters whose unconstrained coordinate is the log of the reported value.
LOG_SCALE = {"tau", "sigma"}


def unconstrained_names(cell: Cell) -> list[str]:
    names = []
    for name, length in parameters(cell):
        label = f"log {name}" if name in LOG_SCALE else name
        names += [label] if length == 0 else [f"{label}[{k}]" for k in range(length)]
    return names


def to_unconstrained(cell: Cell, draws: dict[str, np.ndarray]) -> np.ndarray:
    """(chains, draws, dim) unconstrained draws from `name -> (chains, draws)`."""
    columns = []
    for name, length in parameters(cell):
        for label in ([name] if length == 0 else [f"{name}[{k}]" for k in range(length)]):
            values = np.asarray(draws[label], dtype=float)
            columns.append(np.log(values) if name in LOG_SCALE else values)
    return np.stack(columns, axis=-1)


# ── coordinate check ───────────────────────────────────────────────────────

def check_coordinates(cell: Cell, compiled, model, data: dict) -> dict[str, Any]:
    """rustmc's log density and gradient against PyMC's, Jacobians included, at a
    few unconstrained points. The densities may differ by a constant (rustmc
    drops none, PyMC none either, but normalizing constants can differ); the
    gradients must agree, which pins the order and transform of every
    coordinate the two engines' metrics are compared on."""
    logp = model.compile_logp(jacobian=True)
    dlogp = model.compile_dlogp(jacobian=True)
    point = model.initial_point()
    names = [var.name for var in model.value_vars]
    shapes = [np.shape(point[name]) for name in names]
    sizes = [int(np.prod(shape)) for shape in shapes]
    rng = np.random.default_rng(1)
    differences, gradient_errors = [], []
    for _ in range(4):
        q = 0.5 * rng.normal(size=sum(sizes))
        pieces = np.split(q, np.cumsum(sizes)[:-1])
        values = {name: piece.reshape(shape) for name, piece, shape in zip(names, pieces, shapes)}
        ours, gradient = compiled.log_density(rustmc_data(cell, data), q.tolist())
        differences.append(ours - float(logp(values)))
        gradient_errors.append(float(np.max(np.abs(np.asarray(gradient) - dlogp(values)))))
    return {"pymc_value_vars": names, "logp_difference_spread": float(np.ptp(differences)),
            "max_gradient_difference": float(max(gradient_errors))}


# ── runs ───────────────────────────────────────────────────────────────────

def histogram(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return {str(k): int(c) for k, c in zip(keys, counts)}


# Iteration bins for warmup work: rustmc's initial buffer, its five metric
# windows and its terminal buffer at 1000 warmup iterations. nutpie's phases
# differ; the same bins show where each engine's warmup steps go.
WARMUP_BINS = (0, 75, 100, 150, 250, 450, 950, 1000)
TERMINAL = 50


def summarize(cell: Cell, draws: dict[str, np.ndarray], diverging: np.ndarray,
              steps: np.ndarray, warmup_steps: np.ndarray, depth: np.ndarray,
              accept: np.ndarray, step_sizes: list[float], warmup_accept: np.ndarray,
              warmup_step_used: np.ndarray) -> dict[str, Any]:
    """`steps`, `depth`, `accept`, `diverging`: (chains, draws), post-warmup;
    `warmup_steps`, `warmup_accept`, `warmup_step_used`: (chains, warmup)."""
    stats = draw_statistics(draws, diverging)
    total = int(steps.sum())
    warmup = warmup_steps.shape[1]
    edges = [e for e in WARMUP_BINS if e < warmup] + [warmup]
    phases = {f"{a}-{b}": int(warmup_steps[:, a:b].sum()) for a, b in zip(edges, edges[1:])}
    terminal = slice(max(warmup - TERMINAL, 0), warmup)
    return {
        "warmup_phase_steps": phases,
        # The last 50 warmup transitions: mean acceptance, and the final step
        # over the median step they used, chain by chain.
        "terminal_accept": float(np.mean(warmup_accept[:, terminal])) if warmup else math.nan,
        "final_over_terminal_step": ([float(s / np.median(u[terminal]))
                                      for s, u in zip(step_sizes, warmup_step_used)]
                                     if warmup else []),
        "steps_per_draw": float(steps.mean()),
        "steps_per_draw_by_chain": steps.mean(axis=1).tolist(),
        "warmup_steps": int(warmup_steps.sum()),
        "draw_steps": total,
        "mean_accept": float(np.mean(accept)),
        "divergences": int(np.sum(diverging)),
        "min_bulk_ess": stats["min_bulk_ess"],
        "min_tail_ess": stats["min_tail_ess"],
        "max_rhat": stats["max_rhat"],
        "ess_per_step": stats["min_bulk_ess"] / total,
        "depth_histogram": histogram(depth),
        # floor(log2(steps)): the doubling a trajectory reached, free of how
        # each engine counts a final subtree that turned and was discarded.
        "doubling_histogram": histogram(np.floor(np.log2(np.maximum(steps, 1)))),
        "step_sizes": [float(s) for s in step_sizes],
        "means": stats["means"],
    }


def run_nutpie(cell: Cell, compiled, data: dict, seed: int, jitter: bool) -> dict[str, Any]:
    import nutpie

    bound = compiled.with_data(**({"group": data["group"], "y": data["y"]}
                                  if cell.model == "varying_intercept"
                                  else {"X": data["X"], "y": data["y"]}))
    options = {} if jitter else {"step_size_jitter": None}
    idata = nutpie.sample(bound, draws=cell.draws, tune=cell.warmup, chains=cell.chains, cores=1,
                          seed=seed, save_warmup=True, progress_bar=False,
                          store_mass_matrix=True, store_unconstrained=True, **options)
    stats, warm = idata.sample_stats, idata.warmup_sample_stats
    inverse = stats["mass_matrix_inv"].values
    step_bar = stats["step_size_bar"].values
    if not (np.all(inverse == inverse[:, :1]) and np.all(step_bar == step_bar[:, :1])):
        raise RuntimeError("nutpie's metric or adapted step changed after warmup")
    draws: dict[str, np.ndarray] = {}
    for name, length in parameters(cell):
        values = np.asarray(idata.posterior[name].values, dtype=float)
        if length == 0:
            draws[name] = values
        else:
            for k in range(length):
                draws[f"{name}[{k}]"] = values[:, :, k]
    summary = summarize(cell, draws, stats["diverging"].values, stats["n_steps"].values,
                        warm["n_steps"].values, stats["depth"].values,
                        stats["mean_tree_accept"].values, list(step_bar[:, -1]),
                        warm["mean_tree_accept"].values, warm["step_size"].values)
    summary["mean_accept_sym"] = float(stats["mean_tree_accept_sym"].values.mean())
    summary["used_step_spread"] = [float(stats["step_size"].values.min()),
                                   float(stats["step_size"].values.max())]
    return {"summary": summary,
            "inverse_metric": inverse[:, -1, :],
            "step_size": step_bar[:, -1],
            "unconstrained_labels": [str(v) for v in stats.coords["unconstrained_parameter"].values],
            "unconstrained": stats["unconstrained_draw"].values}


def metric_vector(blocks: list[dict], dim: int) -> tuple[np.ndarray, list[str]]:
    """A chain's reported inverse metric as `dim` variances when every block is
    diagonal, or else a flattened block-diagonal `dim * dim` covariance."""
    kinds = [f"{b['start']}+{b['len']}:{'dense' if b['dense'] else 'diag'}" for b in blocks]
    if not any(b["dense"] for b in blocks):
        out = np.empty(dim)
        for b in blocks:
            out[b["start"]:b["start"] + b["len"]] = b["values"]
        return out, kinds
    full = np.zeros((dim, dim))
    for b in blocks:
        s, n = b["start"], b["len"]
        full[s:s + n, s:s + n] = (np.asarray(b["values"]).reshape(n, n) if b["dense"]
                                  else np.diag(b["values"]))
    return full.ravel(), kinds


def metric_diagonal(metric: np.ndarray, dim: int) -> np.ndarray:
    return metric if metric.size == dim else metric.reshape(dim, dim).diagonal().copy()


def run_rustmc(cell: Cell, compiled, data: dict, seed: int, *, warmup: int, draws: int,
               metrics=None, steps=None, init=None, metric: str = "auto") -> dict[str, Any]:
    out = compiled._experimental_nuts(
        rustmc_data(cell, data), chains=cell.chains, draws=draws, warmup=warmup, seed=seed,
        threads=1, metric=metric, init=None if init is None else [list(map(float, q)) for q in init],
        inverse_metrics=None if metrics is None else [list(map(float, m)) for m in metrics],
        step_sizes=None if steps is None else [float(s) for s in steps])
    fit, chains = out["fit"], out["chains"]

    def post(key):
        return np.stack([c[key][c["is_warmup"] == 0] for c in chains])

    samples = fit.get_samples_2d()
    sample_draws = {name: np.asarray(samples[name], dtype=float) for name in scalar_names(cell)}
    def warm(key):
        return np.stack([c[key][c["is_warmup"] == 1] for c in chains])

    summary = summarize(cell, sample_draws, post("divergent").astype(bool), post("n_leapfrog"),
                        warm("n_leapfrog"), post("tree_depth"), post("accept_prob"),
                        [c["step_size"] for c in chains], warm("accept_prob"),
                        warm("step_size_used"))
    dim = len(unconstrained_names(cell))
    metric_list = [metric_vector(c["inverse_metric"], dim) for c in chains]
    return {"summary": summary, "fit": fit,
            "inverse_metric": [m for m, _ in metric_list],
            "metric_blocks": [kinds for _, kinds in metric_list],
            "step_size": np.array([c["step_size"] for c in chains]),
            "unconstrained": to_unconstrained(cell, sample_draws)}


def calibrate(cell: Cell, compiled, data: dict, seed: int, metrics, init, target: float,
              start: float) -> dict[str, Any]:
    """The step size, shared by the chains, at which frozen `metrics` give mean
    acceptance `target`: bisection on log step over [start / 8, start * 8],
    assuming acceptance falls as the step grows."""
    low, high = math.log(start / 8), math.log(start * 8)
    history = []
    for _ in range(CALIBRATION_ROUNDS):
        middle = 0.5 * (low + high)
        trial = run_rustmc(cell, compiled, data, seed, warmup=CALIBRATION_WARMUP,
                           draws=CALIBRATION_DRAWS, metrics=metrics,
                           steps=[math.exp(middle)] * cell.chains, init=init)
        accept = trial["summary"]["mean_accept"]
        history.append((math.exp(middle), accept))
        if accept > target:
            low = middle
        else:
            high = middle
    return {"step": math.exp(0.5 * (low + high)), "history": history}


def preconditioned_spectrum(posterior: np.ndarray, metric: np.ndarray) -> dict[str, float]:
    """Extreme eigenvalues of the posterior covariance preconditioned by the
    inverse metric (the eigenvalues of `metric^-1 posterior`) and their ratio.
    For a Gaussian target, leapfrog is stable only for steps below
    2 sqrt(lambda_min), and a U-turn takes about pi sqrt(lambda_max) / step."""
    from scipy.linalg import eigh

    dim = posterior.shape[0]
    matrix = np.diag(metric) if metric.size == dim else metric.reshape(dim, dim)
    values = eigh(posterior, matrix, eigvals_only=True)
    return {"min": float(values.min()), "max": float(values.max()),
            "condition": float(values.max() / values.min())}


def run_dataset(cell: Cell, index: int, compiled_rustmc, compiled_nutpie,
                max_load: float) -> dict[str, Any]:
    load, timed_out = wait_for_quiet(max_load)
    data = make_data(cell, index)
    seed = cell.data_seed + index
    dim = len(unconstrained_names(cell))
    record: dict[str, Any] = {"index": index, "seed": seed, "load_at_start": load,
                              "load_wait_timed_out": timed_out}
    nutpie = run_nutpie(cell, compiled_nutpie, data, seed, jitter=True)
    nutpie0 = run_nutpie(cell, compiled_nutpie, data, seed, jitter=False)
    adapted = run_rustmc(cell, compiled_rustmc, data, seed, warmup=cell.warmup, draws=cell.draws)
    diagonal = run_rustmc(cell, compiled_rustmc, data, seed, warmup=cell.warmup,
                          draws=cell.draws, metric="diag")
    if index == 0:
        # The hook's adapted run is the production one: same seed, same steps.
        production = compiled_rustmc.sample(rustmc_data(cell, data), chains=cell.chains,
                                            warmup=cell.warmup, draws=cell.draws, seed=seed,
                                            threads=1, show_progress=False)
        record["production_total_steps"] = int(
            production.transition_diagnostics()["total_leapfrog_steps"])
        record["hook_total_steps"] = int(
            adapted["fit"].transition_diagnostics()["total_leapfrog_steps"])
        record["production_step_sizes"] = list(production.step_sizes())
    record["nutpie_unconstrained_labels"] = nutpie["unconstrained_labels"]
    init = adapted["unconstrained"][:, -1, :]
    r_metric, r_step = adapted["inverse_metric"], adapted["step_size"]
    n_metric, n_step = nutpie["inverse_metric"], nutpie["step_size"]

    def frozen(metrics, steps):
        return run_rustmc(cell, compiled_rustmc, data, seed, warmup=BURN_IN, draws=cell.draws,
                          metrics=metrics, steps=steps, init=init)

    runs = {"nutpie": nutpie["summary"], "nutpie no jitter": nutpie0["summary"],
            "rustmc adapted": adapted["summary"], "rustmc diag adapted": diagonal["summary"]}
    runs["R/R"] = frozen(r_metric, r_step)["summary"]
    runs["N/N"] = frozen(n_metric, n_step)["summary"]
    runs["R/N"] = frozen(r_metric, n_step)["summary"]
    runs["N/R"] = frozen(n_metric, r_step)["summary"]
    runs["N0/N0"] = frozen(nutpie0["inverse_metric"], nutpie0["step_size"])["summary"]
    target = runs["rustmc adapted"]["mean_accept"]
    pooled = np.concatenate([nutpie["unconstrained"].reshape(-1, dim),
                             adapted["unconstrained"].reshape(-1, dim)])
    posterior = np.cov(pooled, rowvar=False)
    posterior = 0.5 * (posterior + posterior.T)
    calibrations = {}
    dense = [posterior.ravel()] * cell.chains
    d_metric = diagonal["inverse_metric"]
    # "@a" runs match the acceptance rustmc's adapted draws achieved; "@0.8"
    # runs match the 0.8 both engines target.
    for label, metrics, goal in (("N@a", n_metric, target), ("dense@a", dense, target),
                                 ("R@0.8", r_metric, 0.8), ("Rdiag@0.8", d_metric, 0.8),
                                 ("N@0.8", n_metric, 0.8), ("dense@0.8", dense, 0.8)):
        found = calibrate(cell, compiled_rustmc, data, seed, metrics, init, goal,
                          float(np.median(n_step)))
        found["target"] = goal
        calibrations[label] = found
        runs[label] = frozen(metrics, [found["step"]] * cell.chains)["summary"]
    record["runs"] = runs
    record["calibrations"] = calibrations
    record["calibration_target_accept"] = target
    record["metrics"] = {
        "rustmc": [metric_diagonal(m, dim).tolist() for m in r_metric],
        "rustmc_full": [m.tolist() for m in r_metric],
        "rustmc_blocks": adapted["metric_blocks"],
        "rustmc_diag": [metric_diagonal(m, dim).tolist() for m in d_metric],
        "nutpie": n_metric.tolist(),
        "nutpie_no_jitter": nutpie0["inverse_metric"].tolist(),
        "posterior_variance": np.diag(posterior).tolist(),
    }
    record["spectra"] = {
        "rustmc": [preconditioned_spectrum(posterior, m) for m in r_metric],
        "rustmc diag": [preconditioned_spectrum(posterior, m) for m in d_metric],
        "nutpie": [preconditioned_spectrum(posterior, m) for m in n_metric],
        "unit": [preconditioned_spectrum(posterior, np.ones(dim))],
    }
    corr = np.corrcoef(pooled, rowvar=False)
    record["posterior_correlation"] = corr.tolist()
    record["load_at_end"] = os.getloadavg()[0]
    return record


# ── report ─────────────────────────────────────────────────────────────────

def log_ratio(a: float, b: float) -> float:
    return math.log(a / b)


def attribution(record: dict[str, Any]) -> dict[str, float]:
    """Log ratios of post-warmup steps per draw; every run but nutpie's own uses
    rustmc's kernel. The two "path" groups each sum exactly to the draws gap."""
    s = {label: run["steps_per_draw"] for label, run in record["runs"].items()}
    r, n = record["runs"]["rustmc adapted"], record["runs"]["nutpie"]
    return {
        "total steps with warmup": log_ratio(r["warmup_steps"] + r["draw_steps"],
                                             n["warmup_steps"] + n["draw_steps"]),
        "warmup steps": log_ratio(r["warmup_steps"], n["warmup_steps"]),
        "draws: steps per draw": log_ratio(s["rustmc adapted"], s["nutpie"]),
        "path A step: adapted over R@0.8": log_ratio(s["rustmc adapted"], s["R@0.8"]),
        "path A metric: R@0.8 over N@0.8": log_ratio(s["R@0.8"], s["N@0.8"]),
        "path A kernel: N@0.8 over nutpie": log_ratio(s["N@0.8"], s["nutpie"]),
        "path B metric: adapted over N@a": log_ratio(s["rustmc adapted"], s["N@a"]),
        "path B step: N@a over N@0.8": log_ratio(s["N@a"], s["N@0.8"]),
        "path B kernel: N@0.8 over nutpie": log_ratio(s["N@0.8"], s["nutpie"]),
        "hook: R/R over adapted": log_ratio(s["R/R"], s["rustmc adapted"]),
        "kernel at nutpie's step: N/N over nutpie": log_ratio(s["N/N"], s["nutpie"]),
        "kernel unjittered: N0/N0 over nutpie no jitter": log_ratio(s["N0/N0"],
                                                                     s["nutpie no jitter"]),
        "jitter: nutpie over nutpie no jitter": log_ratio(s["nutpie"], s["nutpie no jitter"]),
        "auto vs diag metric: R@0.8 over Rdiag@0.8": log_ratio(s["R@0.8"], s["Rdiag@0.8"]),
        "diag adaptation: rustmc diag adapted over adapted": log_ratio(s["rustmc diag adapted"],
                                                                        s["rustmc adapted"]),
        "correlation: N@0.8 over dense@0.8": log_ratio(s["N@0.8"], s["dense@0.8"]),
    }


def fmt(values: list[float], digits: int = 3) -> str:
    return " ".join(f"{v:.{digits}g}" for v in values)


def report(cell_records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    out = lines.append
    for entry in cell_records:
        cell = Cell(**entry["config"])
        names = unconstrained_names(cell)
        dim = len(names)
        out("")
        out(f"=== {entry['cell']} ===")
        check = entry["coordinates"]
        out(f"coordinates: PyMC value vars {check['pymc_value_vars']}; nutpie labels "
            f"{entry['datasets'][0]['nutpie_unconstrained_labels'][:3]}...; rustmc log density "
            f"minus PyMC's varies by {check['logp_difference_spread']:.2e} over 4 points, "
            f"gradients differ by at most {check['max_gradient_difference']:.2e}")
        first = entry["datasets"][0]
        out(f"hook check: production sample() took {first['production_total_steps']} leapfrog "
            f"steps with step sizes {fmt(first['production_step_sizes'], 6)}; the hook's adapted "
            f"run took {first['hook_total_steps']} with "
            f"{fmt(first['runs']['rustmc adapted']['step_sizes'], 6)}")
        out("")
        out("Post-warmup runs, per dataset: steps per draw, mean acceptance (each engine's own "
            "statistic), divergences, min bulk ESS, min bulk ESS per post-warmup leapfrog step "
            "(x1000), step sizes by chain.")
        labels = list(entry["datasets"][0]["runs"])
        for record in entry["datasets"]:
            out(f"  dataset {record['index']} (seed {record['seed']}, load "
                f"{record['load_at_start']:.2f} to {record['load_at_end']:.2f})")
            for label in labels:
                run = record["runs"][label]
                out(f"    {label:<20} {run['steps_per_draw']:7.2f} {run['mean_accept']:6.3f} "
                    f"{run['divergences']:4d} {run['min_bulk_ess']:7.0f} "
                    f"{1000 * run['ess_per_step']:7.2f}   step {fmt(run['step_sizes'])}")
            for label, found in record["calibrations"].items():
                out(f"    {label} bisection to accept {found['target']:.3f}: "
                    + ", ".join(f"{s:.4g}->{a:.3f}" for s, a in found["history"]))
        out("")
        out("R/N is rustmc's metric at nutpie's step; where it diverges its step counts are "
            "cut short by divergences and say nothing about efficiency.")
        out("")
        out("Warmup steps by iteration bin (all chains), and the last 50 warmup transitions: "
            "mean acceptance and final step over the median step they used, by chain.")
        for record in entry["datasets"]:
            for label in ("rustmc adapted", "rustmc diag adapted", "nutpie"):
                run = record["runs"][label]
                phases = " ".join(f"{k}:{v}" for k, v in run["warmup_phase_steps"].items())
                out(f"  dataset {record['index']} {label:<20} {phases}; terminal accept "
                    f"{run['terminal_accept']:.3f}; final/terminal step "
                    f"{fmt(run['final_over_terminal_step'])}")
        out("")
        out("Warmup and total steps (all chains): warmup steps rustmc / nutpie; total ratio.")
        for record in entry["datasets"]:
            r, n = record["runs"]["rustmc adapted"], record["runs"]["nutpie"]
            out(f"  dataset {record['index']}: warmup {r['warmup_steps']} / {n['warmup_steps']} "
                f"= {r['warmup_steps'] / n['warmup_steps']:.2f}; draws {r['draw_steps']} / "
                f"{n['draw_steps']} = {r['draw_steps'] / n['draw_steps']:.2f}; total "
                f"{(r['warmup_steps'] + r['draw_steps']) / (n['warmup_steps'] + n['draw_steps']):.2f}")
        out("")
        out("Attribution: log ratios of post-warmup steps per draw (positive means rustmc's "
            "side takes more), per dataset, then as factors exp(median).")
        parts = [attribution(record) for record in entry["datasets"]]
        for key in parts[0]:
            values = [p[key] for p in parts]
            out(f"  {key:<40} {' '.join(f'{v:+.3f}' for v in values)}   factor "
                f"{math.exp(float(np.median(values))):.3f}")
        out("")
        out("Tree depth, post-warmup, all datasets pooled (each engine's own convention: rustmc "
            "counts a final subtree that turned and was discarded, nuts-rs does not), and the "
            "doubling class floor(log2(steps)), comparable across engines.")
        for label in ("rustmc adapted", "rustmc diag adapted", "nutpie", "N/N", "R@0.8",
                      "N@0.8", "dense@0.8"):
            for key in ("depth_histogram", "doubling_histogram"):
                pooled: dict[str, int] = {}
                for record in entry["datasets"]:
                    for k, c in record["runs"][label][key].items():
                        pooled[k] = pooled.get(k, 0) + c
                total = sum(pooled.values())
                text = " ".join(f"{k}:{100 * c / total:.1f}%"
                                for k, c in sorted(pooled.items(), key=lambda kv: int(kv[0])))
                out(f"  {label:<18} {key.split('_')[0]:<8} {text}")
        out("")
        out("Final inverse metrics (variances), dataset 0: median over chains; posterior is the "
            "pooled draws' variance; ratios are per chain, then median.")
        record = entry["datasets"][0]
        m = record["metrics"]
        rust, nut = np.array(m["rustmc"]), np.array(m["nutpie"])
        post = np.array(m["posterior_variance"])
        out(f"  rustmc blocks by chain: {record['metrics']['rustmc_blocks']}")
        out(f"  {'element':<14} {'rustmc':>10} {'nutpie':>10} {'posterior':>10} {'R/N':>7} "
            f"{'R/post':>7} {'N/post':>7}")
        for i, name in enumerate(names):
            if dim > 30 and "[" in name and int(name.split("[")[1][:-1]) >= 5:
                continue
            out(f"  {name:<14} {np.median(rust[:, i]):10.4g} {np.median(nut[:, i]):10.4g} "
                f"{post[i]:10.4g} {np.median(rust[:, i] / nut[:, i]):7.3f} "
                f"{np.median(rust[:, i] / post[i]):7.3f} {np.median(nut[:, i] / post[i]):7.3f}")
        out("  Ratio summaries over every chain and dataset (median [min, max]):")
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            groups.setdefault(name.split("[")[0], []).append(i)
        for group, idx in groups.items():
            for label, (a, b) in {"R/N": ("rustmc", "nutpie"), "R/post": ("rustmc", None),
                                  "N/post": ("nutpie", None)}.items():
                ratios = []
                for rec in entry["datasets"]:
                    mm = rec["metrics"]
                    top = np.array(mm[a])[:, idx]
                    bottom = (np.array(mm[b])[:, idx] if b
                              else np.array(mm["posterior_variance"])[idx][None, :])
                    ratios.extend((top / bottom).ravel().tolist())
                out(f"    {group:<10} {label:<7} {np.median(ratios):.3f} "
                    f"[{min(ratios):.3f}, {max(ratios):.3f}]")
        out("")
        out("Posterior covariance preconditioned by each metric (median over chains): smallest "
            "eigenvalue, largest, and their ratio, by dataset.")
        for rec in entry["datasets"]:
            parts = []
            for label, spectra in rec["spectra"].items():
                parts.append(f"{label} " + " ".join(
                    f"{np.median([x[k] for x in spectra]):.3g}" for k in ("min", "max", "condition")))
            out(f"  dataset {rec['index']}: " + "; ".join(parts))
        if cell.model == "varying_intercept":
            out("")
            out("Posterior correlations, pooled draws, by dataset: log tau with each z (median "
                "|r|, max |r|), mu with each z (median |r|), log tau with mu.")
            for rec in entry["datasets"]:
                corr = np.array(rec["posterior_correlation"])
                z = [i for i, name in enumerate(names) if name.startswith("z[")]
                t, mu = names.index("log tau"), names.index("mu")
                out(f"  dataset {rec['index']}: {np.median(np.abs(corr[t, z])):.3f} "
                    f"{np.max(np.abs(corr[t, z])):.3f}; {np.median(np.abs(corr[mu, z])):.3f}; "
                    f"{corr[t, mu]:+.3f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="JSON record")
    parser.add_argument("--max-load", type=float, default=1.5)
    parser.add_argument("--cell", type=int, action="append", help="index into CELLS; repeatable")
    parser.add_argument("--datasets", type=int, default=len(DATASETS))
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 200 warmup and draws, two calibration rounds")
    args = parser.parse_args(argv)
    global CALIBRATION_ROUNDS
    if args.quick:
        CALIBRATION_ROUNDS = 2

    import nutpie
    import rustmc

    from benchmarks.head_to_head import _revision
    from benchmarks.protocol import BenchmarkConfig, environment_metadata, package_version

    record: dict[str, Any] = {
        "revision": _revision(),
        "rustmc_file": rustmc.__file__,
        "versions": {name: package_version(name)
                     for name in ("rustmc", "nutpie", "pymc", "arviz", "numpy")},
        "environment": environment_metadata(BenchmarkConfig()),
        "command": " ".join(sys.argv).replace(str(Path.home()), "~"),
        "load_at_start": os.getloadavg()[0],
        "quick": args.quick,
        "settings": {"burn_in": BURN_IN, "calibration_warmup": CALIBRATION_WARMUP,
                     "calibration_draws": CALIBRATION_DRAWS,
                     "calibration_rounds": CALIBRATION_ROUNDS},
        "cells": [],
    }
    cells = [CELLS[i] for i in args.cell] if args.cell else list(CELLS)
    if args.quick:
        from dataclasses import replace

        cells = [replace(cell, warmup=200, draws=200) for cell in cells]
    started = time.time()
    for cell in cells:
        compiled_rustmc = build_rustmc_model(cell)
        model = build_pymc_model(cell, make_data(cell, 0))
        compiled_nutpie = nutpie.compile_pymc_model(model)
        entry = {"cell": cell.name, "config": asdict(cell),
                 "coordinates": check_coordinates(cell, compiled_rustmc, model, make_data(cell, 0)),
                 "datasets": []}
        for index in DATASETS[:args.datasets]:
            entry["datasets"].append(run_dataset(cell, index, compiled_rustmc, compiled_nutpie,
                                                 args.max_load))
            print(f"{cell.name} dataset {index} done after {time.time() - started:.0f} s",
                  file=sys.stderr, flush=True)
        record["cells"].append(entry)
        args.output.write_text(json.dumps(record, indent=1) + "\n")
    record["load_at_end"] = os.getloadavg()[0]
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(f"revision: {json.dumps(record['revision'])}")
    print(f"command: {record['command']}")
    print(f"rustmc: {record['rustmc_file']}; versions {json.dumps(record['versions'])}")
    print(f"machine: {json.dumps(record['environment'])}")
    print(f"one-minute load: {record['load_at_start']:.2f} at start, "
          f"{record['load_at_end']:.2f} at end; settings {json.dumps(record['settings'])}; "
          f"quick {record['quick']}")
    for line in report(record["cells"]):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
