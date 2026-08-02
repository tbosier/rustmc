"""Engine-independent configuration, data, timing, and quality metrics."""

from __future__ import annotations

import hashlib
import json
import math
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


def posterior_quality(
    samples: np.ndarray,
    problem: LinearRegressionProblem,
    config: BenchmarkConfig,
    divergences: int,
    arviz_module: Any,
) -> dict[str, float | int]:
    """Compute the same ArviZ diagnostics and analytic checks for every engine."""

    draws = np.asarray(samples, dtype=np.float64)
    expected_shape = (config.chains, config.draws, problem.posterior_mean.size)
    if draws.ndim != 3 or draws.shape != expected_shape:
        raise ValueError(
            "samples must have shape (chain, draw, parameter); "
            f"received {draws.shape}"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError("samples contain non-finite values")

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
    posterior_mean = draws.mean(axis=(0, 1))
    posterior_sd = draws.reshape(-1, draws.shape[2]).std(axis=0, ddof=1)
    return {
        "ess_bulk_mean": float(ess.mean()),
        "ess_bulk_min": float(ess.min()),
        "rhat_rank_max": float(rhat.max()),
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
    """Return an explicit necessary-quality gate for interpreting timing."""

    thresholds = {
        "rhat_rank_max": config.quality_max_rhat,
        "divergences": config.quality_max_divergences,
        "ess_bulk_min": config.quality_min_ess_bulk,
        "mean_rmse_exact_posterior_sd_units": config.quality_max_mean_error_sd_units,
        "sd_relative_rmse_vs_exact_posterior": config.quality_max_sd_relative_rmse,
    }
    failures = []
    if float(quality["rhat_rank_max"]) > config.quality_max_rhat:
        failures.append("rhat_rank_max")
    if int(quality["divergences"]) > config.quality_max_divergences:
        failures.append("divergences")
    if float(quality["ess_bulk_min"]) < config.quality_min_ess_bulk:
        failures.append("ess_bulk_min")
    if (
        float(quality["mean_rmse_exact_posterior_sd_units"])
        > config.quality_max_mean_error_sd_units
    ):
        failures.append("mean_rmse_exact_posterior_sd_units")
    if (
        float(quality["sd_relative_rmse_vs_exact_posterior"])
        > config.quality_max_sd_relative_rmse
    ):
        failures.append("sd_relative_rmse_vs_exact_posterior")
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": thresholds,
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
