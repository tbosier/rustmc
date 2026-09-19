"""Engine-independent configuration, data, timing, and quality metrics."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

ENGINE_NAMES = ("rustmc", "pymc", "nutpie", "numpyro")


@dataclass(frozen=True)
class BenchmarkConfig:
    """A complete, serializable inference workload."""

    name: str = "linear-regression-standard"
    observations: int = 2_000
    parameters: int = 8
    chains: int = 4
    warmup: int = 500
    draws: int = 1_000
    threads: int = 4
    data_seed: int = 20_260_802
    sampler_seed: int = 314_159
    observation_sigma: float = 1.0
    prior_sigma: float = 1.0
    target_accept: float = 0.8
    max_tree_depth: int = 10
    quality_max_rhat: float = 1.01
    quality_max_divergences: int = 0
    quality_min_ess_bulk: float = 400.0
    quality_max_mean_error_sd_units: float = 0.2
    quality_max_sd_relative_rmse: float = 0.15

    def validate(self) -> None:
        for name in ("observations", "parameters", "chains", "warmup", "draws", "threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("data_seed", "sampler_seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isfinite(self.observation_sigma) or self.observation_sigma <= 0:
            raise ValueError("observation_sigma must be finite and positive")
        if not math.isfinite(self.prior_sigma) or self.prior_sigma <= 0:
            raise ValueError("prior_sigma must be finite and positive")
        if not 0 < self.target_accept < 1:
            raise ValueError("target_accept must be between zero and one")
        if self.max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be positive")
        if self.quality_max_rhat < 1:
            raise ValueError("quality_max_rhat must be at least one")
        if self.quality_max_divergences < 0:
            raise ValueError("quality_max_divergences must be non-negative")
        if self.quality_min_ess_bulk <= 0:
            raise ValueError("quality_min_ess_bulk must be positive")
        if self.quality_max_mean_error_sd_units <= 0:
            raise ValueError("quality_max_mean_error_sd_units must be positive")
        if self.quality_max_sd_relative_rmse <= 0:
            raise ValueError("quality_max_sd_relative_rmse must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkConfig:
        values = json.loads(Path(path).read_text())
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class LinearRegressionProblem:
    x: np.ndarray
    y: np.ndarray
    generating_beta: np.ndarray
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    digest: str


def make_linear_regression(config: BenchmarkConfig) -> LinearRegressionProblem:
    """Generate common float64 data and its analytic Gaussian posterior."""

    config.validate()
    rng = np.random.default_rng(config.data_seed)
    generating_beta = rng.normal(0.0, 0.5, size=config.parameters)
    x = np.ascontiguousarray(
        rng.normal(size=(config.observations, config.parameters)), dtype=np.float64
    )
    y = np.ascontiguousarray(
        x @ generating_beta
        + rng.normal(0.0, config.observation_sigma, size=config.observations),
        dtype=np.float64,
    )

    prior_precision = 1.0 / config.prior_sigma**2
    likelihood_precision = 1.0 / config.observation_sigma**2
    precision = likelihood_precision * (x.T @ x)
    precision.flat[:: config.parameters + 1] += prior_precision
    covariance = np.linalg.inv(precision)
    posterior_mean = covariance @ (likelihood_precision * x.T @ y)
    posterior_sd = np.sqrt(np.diag(covariance))

    hasher = hashlib.sha256()
    hasher.update(x.tobytes(order="C"))
    hasher.update(y.tobytes(order="C"))
    hasher.update(generating_beta.tobytes(order="C"))
    return LinearRegressionProblem(
        x=x,
        y=y,
        generating_beta=generating_beta,
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
        digest=hasher.hexdigest(),
    )


@dataclass
class PhaseTimer:
    phases: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (
                time.perf_counter() - started
            )


INFERENCE_PHASES = (
    "build",
    "compile",
    "bind",
    "warmup",
    "sample",
    "warmup_sample",
    "compile_warmup",
    "compile_warmup_sample",
)


def timing_summary(phases: dict[str, float]) -> dict[str, float | None]:
    """Summarize phases without double-counting combined engine-native phases."""

    fit_seconds = sum(float(phases.get(name, 0.0)) for name in INFERENCE_PHASES)
    cold_fit_seconds = fit_seconds + float(phases.get("import", 0.0))
    total_seconds = cold_fit_seconds + float(phases.get("postprocess", 0.0))
    sample_seconds = phases.get("sample")
    return {
        "fit_seconds": fit_seconds,
        "cold_fit_seconds": cold_fit_seconds,
        "total_seconds": total_seconds,
        "sample_seconds": float(sample_seconds) if sample_seconds is not None else None,
    }


# --------------------------------------------------------------------------------
# Diagnostic screening, shared with benchmarks.validate_posteriors.
#
# Both release gates reduce a set of diagnostics to a pass/fail verdict, and both are
# defeated the same way: every comparison against NaN is false, so an unscreened gate
# appends nothing to its failure list and reports success on a fit whose diagnostics
# are not numbers. validate_posteriors was hardened against that; this module was a
# second copy of the same gate and was missed. The predicates below are the single
# implementation of the rule. They live here rather than in validate_posteriors
# because this module has no rustmc dependency.
#
# Sharing the predicate is necessary but not sufficient: each gate must also apply it
# at the same point in its pipeline. `posterior_quality` screens per parameter before
# aggregating for exactly that reason -- calling the same function on an already
# aggregated value screens nothing the aggregation has thrown away.
# --------------------------------------------------------------------------------

#: Slack on the derived R-hat floor, far above the estimator's rounding error and far
#: below the gap to any value a broken payload would carry.
RHAT_FLOOR_SLACK = 1e-6
#: Relative slack on the derived ESS ceiling, which is exact up to rounding.
ESS_CEILING_SLACK = 1e-9


def as_float(value: Any) -> float:
    """Convert a diagnostic to a float, or to NaN when it is not a real number."""

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


def rhat_floor(draws: int) -> float:
    """Smallest R-hat either estimator can return for chains of ``draws`` draws.

    ``_rhat_rank`` in arviz.stats.diagnostics, which this module's ``posterior_quality``
    calls for every engine, splits each chain in half and returns
    ``sqrt((B/W + n - 1) / n)`` over the split length ``n = draws // 2``; B is a
    variance of chain means and cannot be negative, so the value bottoms out at
    ``sqrt((n-1)/n)``. ``r_hat_chains`` in rust_core/src/diagnostics.rs splits the same
    way and returns ``sqrt(var_hat / W)`` with ``var_hat = (n-1)/n * W + B/n``, which
    bottoms out at the same place, so one bound serves both gates.

    Anything below that is not a slightly unlucky R-hat, it is a payload that does not
    come from the estimator - and only a lower bound catches it, because the reported
    maximum keeps the largest value and hides the rest.
    """
    split = max(int(draws) // 2, 2)
    return math.sqrt((split - 1) / split)


def ess_ceiling(chains: int, draws: int) -> float:
    """Largest ESS either estimator can return for ``chains`` chains of ``draws`` draws.

    ``_ess`` in arviz.stats.diagnostics returns ``total / tau_hat`` after clamping
    ``tau_hat`` to at least ``1 / log10(total)``, and ``_ess_bulk`` splits every chain
    in half first, so the quotient cannot exceed ``total * log10(total)`` for the split
    draw count ``total``. ``ess_raw`` in rust_core/src/diagnostics.rs applies the same
    clamp to the same split count.

    The reported minimum hides an impossibly large ESS behind a healthy neighbour
    exactly as the reported maximum hides an impossibly small R-hat, so this bound is
    needed for the same reason.

    One exception, in both estimators: a chain whose draws are all equal short-circuits
    to the split draw count itself, which exceeds ``total * log10(total)`` whenever
    ``total`` is below ten. Such a fit is degenerate and its R-hat is NaN, so the gate
    fails it either way, but the reported reason will be the ESS bound rather than the
    constant chain. The bound is exact for every chain length a benchmark actually runs.
    """
    total = 2 * max(int(chains), 1) * max(int(draws) // 2, 1)
    return total * math.log10(total) if total > 1 else math.inf


def metric_failure(value: Any, low: float, high: float) -> str | None:
    """Name why ``value`` is not a diagnostic the estimator could have produced.

    Screening is against the diagnostic's full domain, not only against finiteness: a
    value outside what the estimators can return is a payload that did not come from
    them, and each aggregation hides exactly the half of the domain it does not select
    for. Returns ``None`` when the value is usable.
    """
    number = as_float(value)
    # NaN fails every comparison, so finiteness has to be tested first. as_float maps
    # None, complex values, objects and unparseable strings to NaN, so this arm covers
    # them too. It does NOT reject a value that is merely the wrong *type* for a number
    # it can parse: float("1.0") and bool are accepted as 1.0. That is deliberate --
    # benchmarks.validate_posteriors has always done the same, and diverging here is
    # what let the two gates drift apart in the first place.
    if not math.isfinite(number):
        return "non-finite"
    if number < low or number > high:
        return "out-of-domain"
    return None


def count_failure(value: Any) -> str | None:
    """Name why ``value`` is not a whole, non-negative event count."""

    number = as_float(value)
    if not math.isfinite(number):
        return "non-finite"
    if number < 0:
        return "negative"
    if number != int(number):
        return "fractional"
    return None


def quality_metric_domains(config: BenchmarkConfig) -> dict[str, tuple[float, float]]:
    """The interval each gated quality metric must lie in to have come from a fit.

    ``ess_bulk_mean`` carries no published threshold but is the numerator of the
    ``ess_per_fit_second`` figure the comparison is written around, so an unscreened
    NaN or fabrication there corrupts the headline number just as directly.

    These are the values the estimators can produce *for a fit that was actually run at
    this config*. They are derived from ``config.chains`` and ``config.draws``, so a
    hand-written quality payload paired with a config it did not come from is screened
    against the wrong interval. Screening cannot authenticate a payload, only reject one
    that contradicts itself.
    """
    config.validate()
    ceiling = ess_ceiling(config.chains, config.draws) * (1 + ESS_CEILING_SLACK)
    return {
        "rhat_rank_max": (rhat_floor(config.draws) - RHAT_FLOOR_SLACK, math.inf),
        "ess_bulk_mean": (0.0, ceiling),
        "ess_bulk_min": (0.0, ceiling),
        # Both are a square root of a mean of squares and cannot be negative.
        "mean_rmse_exact_posterior_sd_units": (0.0, math.inf),
        "sd_relative_rmse_vs_exact_posterior": (0.0, math.inf),
    }


def posterior_quality(
    samples: np.ndarray,
    problem: LinearRegressionProblem,
    config: BenchmarkConfig,
    divergences: int,
    arviz_module: Any,
) -> dict[str, float | int]:
    """Compute the same ArviZ diagnostics and analytic checks for every engine.

    The per-parameter diagnostics are screened here, before they are aggregated.
    ``max`` over the R-hats keeps the largest and ``min`` over the ESS keeps the
    smallest, so aggregating first hides exactly the half of each domain the gate
    cares about: R-hats of [-100, 1] reported a healthy maximum of 1, and ESS values
    of [20000, 500] reported a healthy minimum of 500, and the gate never saw either
    impossible value. ``convergence_metrics`` in benchmarks.validate_posteriors screens
    its rows one at a time for this reason; doing the same here is what actually makes
    the two gates enforce one rule, rather than merely calling one predicate.

    An invalid entry is carried into the reported metric rather than dropped, again as
    validate_posteriors does, so the failure survives into the JSON report instead of
    being replaced by a plausible value from a neighbouring parameter.
    """

    draws = np.asarray(samples, dtype=np.float64)
    expected_shape = (config.chains, config.draws, problem.posterior_mean.size)
    if draws.ndim != 3 or draws.shape != expected_shape:
        raise ValueError(
            "samples must have shape (chain, draw, parameter); "
            f"received {draws.shape}"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError("samples contain non-finite values")
    # int() truncates, so a count of -0.5 used to reach the gate as a clean zero.
    divergence_reason = count_failure(divergences)
    if divergence_reason is not None:
        raise ValueError(f"divergence count is {divergence_reason}: {divergences!r}")

    ess = np.asarray(
        [
            float(arviz_module.ess(draws[:, :, i], method="bulk"))
            for i in range(draws.shape[2])
        ]
    )
    rhat = np.asarray(
        [
            float(arviz_module.rhat(draws[:, :, i], method="rank"))
            for i in range(draws.shape[2])
        ]
    )
    domains = quality_metric_domains(config)

    def aggregate(values: np.ndarray, metric: str, reduce: Any) -> float:
        low, high = domains[metric]
        invalid = [v for v in values if metric_failure(v, low, high) is not None]
        return float(invalid[0] if invalid else reduce(values))

    posterior_mean = draws.mean(axis=(0, 1))
    posterior_sd = draws.reshape(-1, draws.shape[2]).std(axis=0, ddof=1)
    return {
        "ess_bulk_mean": aggregate(ess, "ess_bulk_mean", np.mean),
        "ess_bulk_min": aggregate(ess, "ess_bulk_min", np.min),
        "rhat_rank_max": aggregate(rhat, "rhat_rank_max", np.max),
        "divergences": int(divergences),
        "mean_rmse_vs_exact_posterior": float(
            np.sqrt(np.mean((posterior_mean - problem.posterior_mean) ** 2))
        ),
        "mean_rmse_exact_posterior_sd_units": float(
            np.sqrt(
                np.mean(
                    ((posterior_mean - problem.posterior_mean) / problem.posterior_sd)
                    ** 2
                )
            )
        ),
        "mean_rmse_vs_generating_beta": float(
            np.sqrt(np.mean((posterior_mean - problem.generating_beta) ** 2))
        ),
        "sd_relative_rmse_vs_exact_posterior": float(
            np.sqrt(np.mean(((posterior_sd / problem.posterior_sd) - 1.0) ** 2))
        ),
    }


def evaluate_quality_gate(
    quality: dict[str, float | int], config: BenchmarkConfig
) -> dict[str, Any]:
    """Return an explicit necessary-quality gate for interpreting timing.

    Every gated metric is screened before it is compared. Comparing with a bare ``>``
    or ``<`` let a NaN through: the comparison is false, nothing was appended to
    ``failures``, and the gate that authorises publishing a speed claim reported
    success on a fit whose diagnostics were not numbers. A missing key raised a
    KeyError out of the benchmark row instead, and ``int(divergences)`` truncated a
    fractional count and read a negative one as clean.

    Each failure is reported twice: once under the bare metric name, which is what
    benchmarks/README.md documents and what downstream consumers match on, and once as
    ``metric[reason]`` so the report says why the value was rejected.
    """

    thresholds = {
        "rhat_rank_max": config.quality_max_rhat,
        "divergences": config.quality_max_divergences,
        "ess_bulk_min": config.quality_min_ess_bulk,
        "mean_rmse_exact_posterior_sd_units": config.quality_max_mean_error_sd_units,
        "sd_relative_rmse_vs_exact_posterior": config.quality_max_sd_relative_rmse,
    }
    domains = quality_metric_domains(config)
    failures: list[str] = []
    screened: dict[str, float] = {}

    def screen(name: str, reason: str | None) -> None:
        if reason is not None:
            failures.extend((name, f"{name}[{reason}]"))
        else:
            value = quality[name]
            # A count is kept as an exact integer. as_float rounds 2**53 + 1 down to
            # 2**53, which made a count one above a threshold of 2**53 compare as
            # equal to it and pass, where the old int() comparison rejected it.
            screened[name] = (
                value if isinstance(value, numbers.Integral) else as_float(value)
            )

    for name, (low, high) in domains.items():
        if name not in quality:
            failures.extend((name, f"{name}[missing]"))
            continue
        screen(name, metric_failure(quality[name], low, high))
    if "divergences" not in quality:
        failures.extend(("divergences", "divergences[missing]"))
    else:
        screen("divergences", count_failure(quality["divergences"]))

    # Only values that survived screening are compared, so a threshold is never read
    # as met by a value that is not a number.
    for name in ("rhat_rank_max", "divergences", "mean_rmse_exact_posterior_sd_units",
                 "sd_relative_rmse_vs_exact_posterior"):
        if name in screened and screened[name] > thresholds[name]:
            failures.append(name)
    if "ess_bulk_min" in screened and screened["ess_bulk_min"] < thresholds["ess_bulk_min"]:
        failures.append("ess_bulk_min")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "thresholds": thresholds,
        # An unbounded side is reported as null: json.dumps would otherwise emit
        # `Infinity`, which Python reads back but is not valid JSON for anyone else.
        "domains": {
            name: [None if math.isinf(bound) else bound for bound in bounds]
            for name, bounds in domains.items()
        },
        "interpretation": (
            "necessary but not sufficient for publishing a performance comparison"
        ),
    }


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "threads_requested": config.threads,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "versions": {
            name: package_version(name)
            for name in ("rustmc", "pymc", "nutpie", "numpyro", "jax", "arviz", "numpy")
        },
    }


def peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def result_payload(
    *,
    engine: str,
    config: BenchmarkConfig,
    problem: LinearRegressionProblem,
    phases: dict[str, float],
    quality: dict[str, float | int],
    notes: list[str],
) -> dict[str, Any]:
    timing = timing_summary(phases)
    fit_seconds = float(timing["fit_seconds"] or 0.0)
    sample_seconds = timing["sample_seconds"]
    mean_ess = float(quality["ess_bulk_mean"])
    quality_gate = evaluate_quality_gate(quality, config)
    return {
        "schema_version": 1,
        "status": "ok",
        "engine": engine,
        "config": asdict(config),
        "data_sha256": problem.digest,
        "environment": environment_metadata(config),
        "phases_seconds": phases,
        "timing": timing,
        "quality": quality,
        "quality_gate": quality_gate,
        "ess_per_fit_second": mean_ess / fit_seconds if fit_seconds > 0 else None,
        "ess_per_sample_second": (
            mean_ess / float(sample_seconds) if sample_seconds is not None else None
        ),
        "peak_rss_mb": peak_rss_mb(),
        "notes": notes,
    }
