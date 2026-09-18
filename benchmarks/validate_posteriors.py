"""Release checks against analytic posteriors and independently simulated regressions.

Run: python -m benchmarks.validate_posteriors --output /tmp/posteriors.json
A nonzero exit status means a fit, precision check, or calibration gate failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


#: The fixed-reference cases the gate must run, and how many seeds each runs under.
#: This manifest is the gate's contract, not a description of it: ``run`` fails when a
#: listed case does not produce its records and equally when ``reference_cases`` yields
#: a case that is not listed, so neither dropping a case nor adding an unvetted one can
#: slip through. Keep it in step with docs/statistical-validation.md.
REFERENCE_CASES = ("normal_location", "correlated_regression", "beta_bernoulli", "gamma_poisson")
REFERENCE_REPEATS = 3


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


#: Per-parameter convergence diagnostic -> reported metric and its aggregation.
CONVERGENCE_METRICS = (("r_hat", "max_rhat", max), ("ess_bulk", "min_ess_bulk", min),
                       ("ess_tail", "min_ess_tail", min))
#: Slack on the derived R-hat floor, far above the estimator's rounding error and far
#: below the gap to any value a broken payload would carry.
RHAT_FLOOR_SLACK = 1e-6
#: Relative slack on the derived ESS ceiling, which is exact up to rounding.
ESS_CEILING_SLACK = 1e-9


def rhat_floor(draws):
    """Smallest R-hat the estimator can return for chains of ``draws`` draws.

    ``r_hat_chains`` in rust_core/src/diagnostics.rs splits every chain in half, so its
    split length is ``draws // 2``, and ``basic_r_hat`` returns ``sqrt(var_hat / W)``
    with ``var_hat = (n-1)/n * W + B/n``. B is a sum of squares and cannot be negative,
    so the ratio bottoms out at ``(n-1)/n``. Anything below that is not a slightly
    unlucky R-hat, it is a payload that does not come from the estimator - and only a
    lower bound catches it, because ``max`` keeps the largest value and hides the rest.
    """
    split = max(int(draws) // 2, 2)
    return math.sqrt((split - 1) / split)


def ess_ceiling(chains, draws):
    """Largest ESS the estimator can return for ``chains`` chains of ``draws`` draws.

    ``ess_raw`` in rust_core/src/diagnostics.rs splits every chain in half and returns
    ``total / tau`` with ``tau = (...).max(1.0 / total.log10())``, so the quotient
    cannot exceed ``total * log10(total)``, where ``total`` is the split draw count.
    ``min`` hides an impossibly large ESS behind a healthy neighbour exactly as ``max``
    hides an impossibly small R-hat, so this bound is needed for the same reason.
    """
    total = 2 * max(int(chains), 1) * max(int(draws) // 2, 1)
    return total * math.log10(total) if total > 1 else math.inf


def divergence_total(counts, chains):
    """Total divergences, or NaN when the telemetry is not one count per chain.

    ``sum`` cancels and does not notice gaps: ``[1, -1, 0, 0]`` totals zero and reports
    a clean run, and a single count for a four-chain fit accepts three missing chains.
    Neither can be trusted before the counts themselves are checked.
    """
    failures = []
    if len(counts) != chains:
        failures.append(f"divergences[{len(counts)} counts for {chains} chains]")
    for index, count in enumerate(counts):
        value = _as_float(count)
        if not math.isfinite(value) or value < 0 or value != int(value):
            failures.append(f"divergences[chain {index}]")
    return failures, math.nan if failures else sum(counts)


def _as_float(value):
    # float() silently drops the imaginary part of a complex value, which would turn a
    # non-finite diagnostic into a plausible one, and raises OverflowError on a huge int.
    # numbers.Complex rather than complex: np.complex64 and np.clongdouble are not
    # subclasses of the builtin, so `isinstance(value, complex)` let them straight past.
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def convergence_metrics(diagnostics, names, floor, ceiling):
    """Aggregate per-parameter diagnostics, naming every invalid entry as a failure.

    Builtin ``max``/``min`` return the non-NaN operand unless the NaN comes first, and
    both silently keep a signed infinity that is not the extremum, so a single bad
    parameter could otherwise pass the gate. Each parameter is screened before
    aggregation, and an invalid entry is carried into the reported metric so the
    failure survives into the JSON report rather than being replaced by a plausible
    value from a neighbouring parameter. A parameter in ``names`` with no diagnostic
    row at all is the same hole and fails too, since an absent row is never screened.

    Screening is against each diagnostic's full domain, not only against finiteness.
    A value outside what the estimators in rust_core/src/diagnostics.rs can return is a
    payload that did not come from them, and each aggregation hides exactly the half of
    the domain it does not select for: ``max`` hides a too-small R-hat, so r_hat
    [1.0, -100.0, 1.0] reports a healthy 1.0, and ``min`` hides a too-large ESS, so
    ess_bulk [10000.0, 1e100] reports a healthy 10000.0. Both ends are therefore bound.
    """
    failures, metrics = [], {}
    labels = [diagnostic.get("name", index) for index, diagnostic in enumerate(diagnostics)]
    if not diagnostics:
        failures.append("diagnostics_empty")
    failures += [f"diagnostics_missing[{name}]" for name in names if name not in labels]
    domains = {"r_hat": (floor - RHAT_FLOOR_SLACK, math.inf),
               "ess_bulk": (0., ceiling), "ess_tail": (0., ceiling)}
    for key, metric, reduce in CONVERGENCE_METRICS:
        low, high = domains[key]
        values, invalid = [], []
        for label, diagnostic in zip(labels, diagnostics):
            value = _as_float(diagnostic.get(key))
            # NaN fails every comparison, so finiteness has to be tested first.
            if not math.isfinite(value) or value < low or value > high:
                failures += [metric, f"{metric}[{label}]"]
                invalid.append(value)
            values.append(value)
        metrics[metric] = invalid[0] if invalid else (reduce(values) if values else math.nan)
    return failures, metrics


def assess_fit(fit, names, reference_mean, reference_covariance):
    """Require convergence and marginal/joint posterior accuracy; keep failed metrics."""
    samples = np.stack([fit.get_samples_2d()[name] for name in names], axis=-1)
    flat = samples.reshape(-1, len(names))
    reference_sd = np.sqrt(np.diag(reference_covariance))
    covariance = np.atleast_2d(np.cov(flat, rowvar=False))
    chains, draws = samples.shape[0], samples.shape[-2]
    diagnostics = fit.diagnostics()
    convergence_failures, convergence = convergence_metrics(
        diagnostics, names, rhat_floor(draws), ess_ceiling(chains, draws) * (1 + ESS_CEILING_SLACK))
    divergence_failures, divergences = divergence_total(list(fit.divergences()), chains)
    metrics = {
        **convergence,
        # The error metrics take np.abs first, so every element is non-negative and
        # np.max keeps both NaN and infinity for the finiteness sweep below; builtin
        # max would not, so do not swap it in. divergence_total does the same job for
        # the divergence counts, whose sum can cancel and cannot see a missing chain.
        "divergences": divergences,
        "max_mean_error_sd": float(np.max(np.abs(flat.mean(axis=0)-reference_mean)/reference_sd)),
        "max_covariance_error_sd": float(np.max(np.abs(covariance-reference_covariance)/np.outer(reference_sd, reference_sd))),
    }
    failures = convergence_failures + divergence_failures
    failures += [name for name, value in metrics.items() if not math.isfinite(value)]
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
    records, attempted = [], {}
    # reference_cases() compiles its models as it is advanced. Advancing it inside the
    # for-statement put that work outside the per-fit try, so a construction failure
    # propagated out of run() and main() never reached write_text: the report promised
    # "even if a gate fails" was never written, losing the attempts already completed.
    cases, index = reference_cases(), -1
    while True:
        index += 1
        try:
            name, model, data, names, mean, covariance = next(cases)
        except StopIteration:
            break
        except Exception as error:
            # Covers a malformed yield as well as a failed compile: either way the case
            # is unusable, and case_coverage reports the ones that never ran.
            records.append({"case": "reference_case_construction", "kind": "fixed_reference",
                            "index": index, "passed": False,
                            "error": f"{type(error).__name__}: {error}"})
            break
        for repeat in range(REFERENCE_REPEATS):
            fit_seed = seed + index*100 + repeat
            record = {"case": name, "seed": fit_seed, "kind": "fixed_reference"}
            try:
                fit = model.sample(data, seed=fit_seed, **kwargs)
                record.update(assess_fit(fit, names, mean, covariance))
            except Exception as error:
                record.update(passed=False, error=f"{type(error).__name__}: {error}")
            records.append(record)
            attempted[name] = attempted.get(name, 0) + 1
    # A gate that runs nothing passes everything: all() over no fixed-reference records
    # is vacuously true, so the promised cases are checked against the manifest by name.
    case_coverage = {"required": {name: REFERENCE_REPEATS for name in REFERENCE_CASES},
                     "attempted": attempted,
                     "missing": sorted(name for name in REFERENCE_CASES
                                       if attempted.get(name, 0) != REFERENCE_REPEATS),
                     "unlisted": sorted(set(attempted) - set(REFERENCE_CASES))}
    case_coverage["passed"] = not case_coverage["missing"] and not case_coverage["unlisted"]
    rng = np.random.default_rng(seed)
    # Same shape as the reference cases: this compile ran outside every try.
    try:
        m = mc.ModelBuilder()
        beta = m.vector_normal_prior("beta", 2, 0., 1.)
        m.normal_likelihood("obs", beta @ "X", .7, "y")
        model = m.compile()
    except Exception as error:
        model = None
        records.append({"case": "prior_simulated_regression", "kind": "calibration",
                        "passed": False, "error": f"{type(error).__name__}: {error}"})
    coverage, quantiles = [], []
    for replicate in range(replicates if model is not None else 0):
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
            "calibration": calibration, "case_coverage": case_coverage,
            "seconds": time.perf_counter()-started,
            "passed": all(r["passed"] for r in records) and calibration["passed"]
                      and case_coverage["passed"]}


def json_safe(value):
    """Normalize a report for ``json.dumps(..., allow_nan=False)``, keeping failures.

    A failed metric is kept as a null rather than dropped, so the report still records
    that the gate looked at it. The retained raw ``diagnostics`` are whatever the
    bindings handed over, and today that is Python floats; a NumPy scalar among them
    would be neither caught by the ``float`` test nor serializable, and the strict dump
    in ``main`` would raise after every fit had run, writing no report at all. Unwrap
    NumPy scalars and arrays to their Python equivalents first so the finiteness test
    sees them. ``np.float64`` already subclasses ``float`` and round-trips unchanged.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        # tolist() yields nested lists, or a scalar for a 0-d array; both recurse.
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    # Integral values, bool among them, are always finite, and math.isfinite raises
    # OverflowError on an int too large to convert, so do not ask it about them.
    if (isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral)
            and not math.isfinite(value)):
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
