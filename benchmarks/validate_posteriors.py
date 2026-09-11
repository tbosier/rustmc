"""Release checks against analytic posteriors and independently simulated regressions.

Run: python -m benchmarks.validate_posteriors --output /tmp/posteriors.json
A nonzero exit status means a fit, precision check, or calibration gate failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def reference_cases():
    """Small fixed datasets; moments follow conjugate formulas, independent of rustmc."""
    import rustmc as mc
    m = mc.ModelBuilder()
    mu = m.normal_prior("mu", 0., 2.)
    m.normal_likelihood("obs", mu, 1., "y")
    y = np.array([-1., .5, .25, 1., 2.])
    variance = 1 / (.25 + len(y))
    yield "normal_location", m.compile(), {"y": y}, ["mu"], np.array([variance*y.sum()]), np.array([[variance]])

    m = mc.ModelBuilder()
    beta = m.vector_normal_prior("beta", 2, 0., 1.)
    m.normal_likelihood("obs", beta @ "X", .7, "y")
    x = np.array([[1., -1.], [1., -.2], [1., .5], [1., 1.], [1., 2.]])
    y = np.array([-.8, -.1, .7, 1.2, 1.9])
    covariance = np.linalg.solve(np.eye(2) + x.T@x/.7**2, np.eye(2))
    yield "correlated_regression", m.compile(), {"X": x, "y": y}, ["beta[0]", "beta[1]"], covariance@(x.T@y/.7**2), covariance

    m = mc.ModelBuilder()
    p = m.beta_prior("p", 2., 3.)
    m.bernoulli_logit_likelihood("obs", (p+0.).log() - (1.-p).log(), "y")
    y = np.array([0., 1., 1., 1., 0., 1.])
    a, b = 2+y.sum(), 3+len(y)-y.sum()
    yield "beta_bernoulli", m.compile(), {"y": y}, ["p"], np.array([a/(a+b)]), np.array([[a*b/((a+b)**2*(a+b+1))]])

    m = mc.ModelBuilder()
    rate = m.gamma_prior("rate", 2., 1.5)
    m.poisson_log_likelihood("obs", (rate+0.).log(), "y")
    y = np.array([0., 2., 1., 4., 1.])
    a, b = 2+y.sum(), 1.5+len(y)
    yield "gamma_poisson", m.compile(), {"y": y}, ["rate"], np.array([a/b]), np.array([[a/b**2]])


def assess_fit(fit, names, reference_mean, reference_covariance):
    """Require convergence and marginal/joint posterior accuracy; keep failed metrics."""
    samples = np.stack([fit.get_samples_2d()[name] for name in names], axis=-1)
    flat = samples.reshape(-1, len(names))
    reference_sd = np.sqrt(np.diag(reference_covariance))
    covariance = np.atleast_2d(np.cov(flat, rowvar=False))
    diagnostics = fit.diagnostics()
    metrics = {
        "max_rhat": max(d["r_hat"] for d in diagnostics),
        "min_ess_bulk": min(d["ess_bulk"] for d in diagnostics),
        "min_ess_tail": min(d["ess_tail"] for d in diagnostics),
        "divergences": sum(fit.divergences()),
        "max_mean_error_sd": float(np.max(np.abs(flat.mean(axis=0)-reference_mean)/reference_sd)),
        "max_covariance_error_sd": float(np.max(np.abs(covariance-reference_covariance)/np.outer(reference_sd, reference_sd))),
    }
    failures = [name for name, value in metrics.items() if not math.isfinite(value)]
    limits = {"max_rhat": 1.01, "max_mean_error_sd": .12, "max_covariance_error_sd": .15}
    failures += [name for name, limit in limits.items() if metrics[name] > limit]
    failures += [name for name in ("min_ess_bulk", "min_ess_tail") if metrics[name] < 400]
    if metrics["divergences"] != 0:
        failures.append("divergences")
    return {"passed": not failures, "failed_metrics": sorted(set(failures)), "metrics": metrics,
            "reference_mean": reference_mean.tolist(), "reference_covariance": reference_covariance.tolist(),
            "posterior_mean": flat.mean(axis=0).tolist(), "diagnostics": diagnostics}


def run(*, replicates=64, seed=20260911, draws=2000, warmup=1000):
    import rustmc as mc
    import rustmc._rustmc as native
    started = time.perf_counter()
    kwargs = dict(chains=4, draws=draws, warmup=warmup, target_accept=.95, show_progress=False)
    records = []
    for index, (name, model, data, names, mean, covariance) in enumerate(reference_cases()):
        for repeat in range(3):
            fit_seed = seed + index*100 + repeat
            record = {"case": name, "seed": fit_seed, "kind": "fixed_reference"}
            try:
                fit = model.sample(data, seed=fit_seed, **kwargs)
                record.update(assess_fit(fit, names, mean, covariance))
            except Exception as error:
                record.update(passed=False, error=f"{type(error).__name__}: {error}")
            records.append(record)
    rng = np.random.default_rng(seed)
    m = mc.ModelBuilder()
    beta = m.vector_normal_prior("beta", 2, 0., 1.)
    m.normal_likelihood("obs", beta @ "X", .7, "y")
    model = m.compile()
    coverage, quantiles = [], []
    for replicate in range(replicates):
        theta = rng.normal(size=2)
        x = np.column_stack((np.ones(30), rng.normal(size=30)))
        y = x@theta + rng.normal(0., .7, len(x))
        covariance = np.linalg.solve(np.eye(2) + x.T@x/.7**2, np.eye(2))
        mean = covariance@(x.T@y/.7**2)
        fit_seed = seed + 1000 + replicate
        record = {"case": "prior_simulated_regression", "replicate": replicate, "seed": fit_seed,
                  "kind": "calibration", "generating_beta": theta.tolist(),
                  "data_sha256": hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()}
        try:
            fit = model.sample({"X": x, "y": y}, seed=fit_seed, **kwargs)
            record.update(assess_fit(fit, ["beta[0]", "beta[1]"], mean, covariance))
            values = np.stack([fit.get_samples_2d()[f"beta[{i}]"] for i in range(2)], axis=-1).reshape(-1, 2)
            interval = np.quantile(values, [.05, .95], axis=0)
            covered = (interval[0] <= theta) & (theta <= interval[1])
            ranks = (values < theta).mean(axis=0)
            record.update(covered_90=covered.tolist(), posterior_quantile=ranks.tolist())
            coverage.append(covered)
            quantiles.append(ranks)
        except Exception as error:
            record.update(passed=False, error=f"{type(error).__name__}: {error}")
        records.append(record)
    # Hoeffding bound with a union bound over two parameters. Replicates, not MCMC
    # draws, are the independent units. A small run has deliberately low power.
    tolerance = math.sqrt(math.log(4/.001)/(2*replicates))
    rate = np.mean(coverage, axis=0) if coverage else np.full(2, np.nan)
    calibration = {"nominal": .9, "coverage": rate.tolist(), "replicates": replicates,
                   "family_error_bound": .001, "coverage_tolerance": tolerance,
                   "passed": len(coverage) == replicates and bool(np.all(np.abs(rate-.9) <= tolerance)),
                   "rank_histograms": [np.histogram(np.array(quantiles)[:, i], bins=np.linspace(0, 1, 11))[0].tolist() for i in range(2)] if quantiles else [],
                   "rank_note": "Exploratory ranks use autocorrelated posterior draws; no iid uniform-rank test is applied."}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    native_path = Path(native.__file__)
    return {"format": "rustmc.posterior-validation", "version": 1, "revision": revision,
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "rustmc": mc.__version__, "native_path": str(native_path),
            "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "command": sys.argv, "seed": seed, "sampling": kwargs, "records": records,
            "calibration": calibration, "seconds": time.perf_counter()-started,
            "passed": all(r["passed"] for r in records) and calibration["passed"]}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1 or args.draws < 1 or args.warmup < 1:
        parser.error("replicates, draws, and warmup must be positive")
    report = run(replicates=args.replicates, seed=args.seed, draws=args.draws, warmup=args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False)+"\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['records'])} attempts; {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
