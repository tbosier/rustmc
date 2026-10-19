"""Repeated linear-regression inference: many datasets through one compiled model.

Each engine builds and compiles its model once, then fits every dataset in turn,
fixed-size or ragged, and predicts at new inputs after each fit. Every fit is
checked against its exact conjugate posterior with the gate of
``benchmarks/protocol.py``. A fit that fails the gate, or raises, is kept in the
output with its reason, and its time still counts against the engine: the
headline ``useful_ess_per_second`` counts the bulk ESS of passing fits only,
divided by setup plus every fit's time.

Each engine runs in a fresh subprocess with the environment ``benchmarks.run``
gives it. See benchmarks/README.md, "Repeated inference".
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.protocol import (
    BenchmarkConfig,
    LinearRegressionProblem,
    environment_metadata,
    evaluate_quality_gate,
    make_linear_regression,
    peak_rss_mb,
    posterior_quality,
)
from benchmarks.run import _child_environment

RESULT_PREFIX = "RUSTMC_REPEATED_RESULT="
ENGINES = ("rustmc", "rustmc_batch", "nutpie", "numpyro", "pymc")


@dataclass(frozen=True)
class RepeatedConfig:
    """A repeated workload: `datasets` regressions sharing one model."""

    base: BenchmarkConfig
    datasets: int = 20
    # (low, high) inclusive row counts drawn per dataset, or None for every
    # dataset to have `base.observations` rows.
    ragged: tuple[int, int] | None = None
    predict_rows: int = 10

    def validate(self) -> None:
        self.base.validate()
        if self.datasets <= 0 or self.predict_rows <= 0:
            raise ValueError("datasets and predict_rows must be positive")
        if self.ragged is not None:
            low, high = self.ragged
            if not 0 < low <= high:
                raise ValueError("ragged must be (low, high) with 0 < low <= high")

    @classmethod
    def from_json(cls, path: str | Path) -> RepeatedConfig:
        raw = json.loads(Path(path).read_text())
        base = BenchmarkConfig(**raw.pop("base"))
        ragged = raw.pop("ragged", None)
        config = cls(base=base, ragged=tuple(ragged) if ragged else None, **raw)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": asdict(self.base),
            "datasets": self.datasets,
            "ragged": list(self.ragged) if self.ragged else None,
            "predict_rows": self.predict_rows,
        }


@dataclass
class Dataset:
    index: int
    config: BenchmarkConfig
    problem: LinearRegressionProblem
    x_new: np.ndarray
    # Exact posterior mean of the expected response at `x_new`.
    expected_new: np.ndarray


def make_datasets(config: RepeatedConfig) -> list[Dataset]:
    """The same datasets for every engine, from the base seeds."""
    config.validate()
    rows = np.random.default_rng(config.base.data_seed - 1)
    datasets = []
    for index in range(config.datasets):
        observations = (
            int(rows.integers(config.ragged[0], config.ragged[1] + 1))
            if config.ragged
            else config.base.observations
        )
        cell = replace(
            config.base,
            observations=observations,
            data_seed=config.base.data_seed + index,
            sampler_seed=config.base.sampler_seed + index,
        )
        problem = make_linear_regression(cell)
        x_new = np.random.default_rng(cell.data_seed + 10_000).normal(
            size=(config.predict_rows, config.base.parameters)
        )
        datasets.append(
            Dataset(index, cell, problem, x_new, x_new @ problem.posterior_mean)
        )
    return datasets


def current_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class EngineRun:
    """What one engine measured, phase by phase and fit by fit."""

    setup_phases: dict[str, float] = field(default_factory=dict)
    fits: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def setup(self, name: str, work: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        value = work()
        self.setup_phases[name] = time.perf_counter() - start
        return value


def record_fit(
    run: EngineRun,
    dataset: Dataset,
    fit: Callable[[], tuple[np.ndarray, int, Any]],
    predict: Callable[[Any], np.ndarray],
) -> None:
    """Fit one dataset, predict at its new inputs and gate the result.

    `fit` returns `(draws shaped (chain, draw, parameter), divergences, handle)`
    and `predict(handle)` the expected response at `dataset.x_new`, shaped
    `(chain, draw, row)`. Any exception becomes the row's error. ArviZ, which
    computes the diagnostics, is imported by `run_child` before any engine, so
    no engine's setup time includes it.
    """
    import arviz

    row: dict[str, Any] = {
        "index": dataset.index,
        "observations": dataset.config.observations,
        "data_sha256": dataset.problem.digest,
    }
    try:
        start = time.perf_counter()
        draws, divergences, handle = fit()
        row["fit_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        predicted = np.asarray(predict(handle), dtype=np.float64)
        row["predict_seconds"] = time.perf_counter() - start
        quality = posterior_quality(draws, dataset.problem, dataset.config, divergences, arviz)
        mean_prediction = predicted.reshape(-1, predicted.shape[-1]).mean(axis=0)
        quality["prediction_max_abs_error"] = float(
            np.max(np.abs(mean_prediction - dataset.expected_new))
        )
        row["quality"] = quality
        row["quality_gate"] = evaluate_quality_gate(quality, dataset.config)
        row["status"] = "passed" if row["quality_gate"]["passed"] else "failed"
    except Exception as error:  # noqa: BLE001 - every failure is a result row
        row["status"] = "error"
        row["reason"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc()[-4_000:]
    row["rss_mb_after"] = current_rss_mb()
    run.fits.append(row)


def _beta_draws(mapping: dict[str, np.ndarray], parameters: int) -> np.ndarray:
    names = [f"beta[{i}]" for i in range(parameters)]
    return np.stack([np.asarray(mapping[name]) for name in names], axis=2)


def _rustmc(config: RepeatedConfig, datasets: list[Dataset], run: EngineRun) -> None:
    base = config.base
    rmc = run.setup("import", lambda: __import__("rustmc"))

    def build() -> Any:
        builder = rmc.ModelBuilder()
        beta = builder.vector_normal_prior("beta", n=base.parameters, mu=0.0, sigma=base.prior_sigma)
        builder.normal_likelihood("obs", mu_expr=beta @ "X", sigma=base.observation_sigma, observed_key="y")
        return builder

    builder = run.setup("build", build)
    compiled = run.setup("compile", builder.compile)
    for dataset in datasets:
        def fit(dataset: Dataset = dataset) -> tuple[np.ndarray, int, Any]:
            result = compiled.sample(
                {"X": dataset.problem.x, "y": dataset.problem.y},
                chains=base.chains, warmup=base.warmup, draws=base.draws,
                seed=dataset.config.sampler_seed, threads=base.threads,
                target_accept=base.target_accept, max_tree_depth=base.max_tree_depth,
                show_progress=False,
            )
            return _beta_draws(result.get_samples_2d(), base.parameters), sum(result.divergences()), result

        def predict(result: Any, dataset: Dataset = dataset) -> np.ndarray:
            empty = np.zeros(config.predict_rows)
            return result.predict({"X": dataset.x_new, "y": empty}, expected=True)["obs"]

        record_fit(run, dataset, fit, predict)
    run.notes.append("compiled once; each dataset is bound and sampled by CompiledModel.sample")
    run.notes.append("prediction is FitResult.predict(expected=True) at the new inputs")


def _rustmc_batch(config: RepeatedConfig, datasets: list[Dataset], run: EngineRun) -> None:
    base = config.base
    rmc = run.setup("import", lambda: __import__("rustmc"))

    def build() -> Any:
        builder = rmc.ModelBuilder()
        beta = builder.vector_normal_prior("beta", n=base.parameters, mu=0.0, sigma=base.prior_sigma)
        builder.normal_likelihood("obs", mu_expr=beta @ "X", sigma=base.observation_sigma, observed_key="y")
        return builder

    builder = run.setup("build", build)
    compiled = run.setup("compile", builder.compile)
    start = time.perf_counter()
    try:
        batch = compiled.sample_batch(
            [{"X": d.problem.x, "y": d.problem.y} for d in datasets],
            ids=[str(d.index) for d in datasets],
            chains=base.chains, warmup=base.warmup, draws=base.draws,
            seed=base.sampler_seed, threads=base.threads,
            target_accept=base.target_accept, max_tree_depth=base.max_tree_depth,
            show_progress=False, errors="collect",
        )
        batch_error = None
    except Exception as error:  # noqa: BLE001
        batch, batch_error = None, f"{type(error).__name__}: {error}"
    batch_seconds = time.perf_counter() - start
    run.setup_phases["batch_wall_seconds"] = batch_seconds
    for dataset in datasets:
        def fit(dataset: Dataset = dataset) -> tuple[np.ndarray, int, Any]:
            if batch is None:
                raise RuntimeError(f"sample_batch raised: {batch_error}")
            key = str(dataset.index)
            if key in batch.errors:
                raise RuntimeError(batch.errors[key])
            result = batch.get(key)
            return _beta_draws(result.get_samples_2d(), base.parameters), result.divergences, result

        def predict(result: Any, dataset: Dataset = dataset) -> np.ndarray:
            empty = np.zeros(config.predict_rows)
            return result.predict({"X": dataset.x_new, "y": empty}, expected=True)["obs"]

        record_fit(run, dataset, fit, predict)
    # The cells ran together, so a per-fit time is the batch's wall time shared out.
    for row in run.fits:
        if "fit_seconds" in row:
            row["fit_seconds"] = batch_seconds / len(datasets)
    run.notes.append(
        "one sample_batch call over every dataset with threads running cells in parallel; "
        "seeds come from cell IDs, so draws differ from the per-dataset rustmc run; "
        "fit_seconds per row is the batch wall time divided by the dataset count"
    )


def _nutpie(config: RepeatedConfig, datasets: list[Dataset], run: EngineRun) -> None:
    base = config.base

    def imports() -> tuple[Any, Any]:
        import nutpie
        import pymc

        return nutpie, pymc

    nutpie, pm = run.setup("import", imports)
    first = datasets[0].problem

    def build() -> Any:
        with pm.Model() as model:
            x = pm.Data("X", first.x)
            y = pm.Data("y", first.y)
            beta = pm.Normal("beta", mu=0.0, sigma=base.prior_sigma, shape=base.parameters)
            pm.Normal("obs", mu=pm.math.dot(x, beta), sigma=base.observation_sigma, observed=y, shape=x.shape[0])
        return model

    model = run.setup("build", build)
    compiled = run.setup("compile", lambda: nutpie.compile_pymc_model(model))
    for dataset in datasets:
        def fit(dataset: Dataset = dataset) -> tuple[np.ndarray, int, Any]:
            idata = nutpie.sample(
                compiled.with_data(X=dataset.problem.x, y=dataset.problem.y),
                draws=base.draws, tune=base.warmup, chains=base.chains,
                cores=min(base.threads, base.chains), seed=dataset.config.sampler_seed,
                target_accept=base.target_accept, maxdepth=base.max_tree_depth,
                save_warmup=False, progress_bar=False,
            )
            draws = np.asarray(idata.posterior["beta"].values)
            return draws, int(idata.sample_stats["diverging"].values.sum()), draws

        record_fit(run, dataset, fit, lambda draws, d=dataset: draws @ d.x_new.T)
    run.notes.append("compiled once with pm.Data; each dataset is set with CompiledModel.with_data")
    run.notes.append("prediction is the NumPy product of the beta draws with the new inputs")


def _numpyro(config: RepeatedConfig, datasets: list[Dataset], run: EngineRun) -> None:
    base = config.base

    def imports() -> tuple[Any, ...]:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        numpyro.enable_x64()
        return jax, jnp, numpyro, dist, MCMC, NUTS

    jax, jnp, numpyro, dist, MCMC, NUTS = run.setup("import", imports)

    def build() -> Any:
        def model(x_data: Any, y_data: Any) -> None:
            beta = numpyro.sample("beta", dist.Normal(0.0, base.prior_sigma).expand([base.parameters]))
            numpyro.sample("obs", dist.Normal(x_data @ beta, base.observation_sigma), obs=y_data)

        kernel = NUTS(model, target_accept_prob=base.target_accept, max_tree_depth=base.max_tree_depth)
        return MCMC(
            kernel, num_warmup=base.warmup, num_samples=base.draws, num_chains=base.chains,
            chain_method="parallel", progress_bar=False, jit_model_args=True,
        )

    mcmc = run.setup("build", build)
    for dataset in datasets:
        def fit(dataset: Dataset = dataset) -> tuple[np.ndarray, int, Any]:
            mcmc.run(
                jax.random.key(dataset.config.sampler_seed),
                jnp.asarray(dataset.problem.x), jnp.asarray(dataset.problem.y),
                extra_fields=("diverging",),
            )
            beta = mcmc.get_samples(group_by_chain=True)["beta"]
            jax.block_until_ready(beta)
            divergences = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
            draws = np.asarray(beta)
            return draws, divergences, draws

        record_fit(run, dataset, fit, lambda draws, d=dataset: draws @ d.x_new.T)
    run.notes.append(
        "one MCMC object reused with jit_model_args=True; JAX compiles again for each new "
        "data shape, so a ragged workload recompiles and that time is in fit_seconds"
    )
    run.notes.append("prediction is the NumPy product of the beta draws with the new inputs")


def _pymc(config: RepeatedConfig, datasets: list[Dataset], run: EngineRun) -> None:
    base = config.base

    pm = run.setup("import", lambda: __import__("pymc"))
    first = datasets[0].problem

    def build() -> Any:
        with pm.Model() as model:
            x = pm.Data("X", first.x)
            y = pm.Data("y", first.y)
            beta = pm.Normal("beta", mu=0.0, sigma=base.prior_sigma, shape=base.parameters)
            pm.Normal("obs", mu=pm.math.dot(x, beta), sigma=base.observation_sigma, observed=y, shape=x.shape[0])
        return model

    model = run.setup("build", build)
    for dataset in datasets:
        def fit(dataset: Dataset = dataset) -> tuple[np.ndarray, int, Any]:
            with model:
                pm.set_data({"X": dataset.problem.x, "y": dataset.problem.y})
                idata = pm.sample(
                    draws=base.draws, tune=base.warmup, chains=base.chains,
                    cores=min(base.threads, base.chains), random_seed=dataset.config.sampler_seed,
                    nuts_sampler="pymc", target_accept=base.target_accept,
                    nuts={"max_treedepth": base.max_tree_depth}, progressbar=False,
                    compute_convergence_checks=False, idata_kwargs={"log_likelihood": False},
                )
            draws = np.asarray(idata.posterior["beta"].values)
            return draws, int(idata.sample_stats["diverging"].values.sum()), draws

        record_fit(run, dataset, fit, lambda draws, d=dataset: draws @ d.x_new.T)
    run.notes.append(
        "one pm.Model with pm.Data, updated by pm.set_data; pm.sample compiles its "
        "functions on every call, and that time is in fit_seconds"
    )
    run.notes.append("prediction is the NumPy product of the beta draws with the new inputs")


ADAPTERS = {
    "rustmc": _rustmc,
    "rustmc_batch": _rustmc_batch,
    "nutpie": _nutpie,
    "numpyro": _numpyro,
    "pymc": _pymc,
}


def summarize(run: EngineRun) -> dict[str, Any]:
    """Totals over every fit, with throughput counted from passing fits only."""
    setup = sum(v for k, v in run.setup_phases.items() if k != "batch_wall_seconds")
    fit_times = [row["fit_seconds"] for row in run.fits if "fit_seconds" in row]
    predict_times = [row["predict_seconds"] for row in run.fits if "predict_seconds" in row]
    passed = [row for row in run.fits if row["status"] == "passed"]
    total_fit = float(sum(fit_times))
    useful_ess = sum(row["quality"]["ess_bulk_mean"] for row in passed)
    rss = [row["rss_mb_after"] for row in run.fits if row.get("rss_mb_after") is not None]
    return {
        "datasets": len(run.fits),
        "passed": len(passed),
        "failed": sum(row["status"] == "failed" for row in run.fits),
        "errors": sum(row["status"] == "error" for row in run.fits),
        "setup_seconds": setup,
        "first_fit_seconds": fit_times[0] if fit_times else None,
        "median_fit_seconds": float(np.median(fit_times)) if fit_times else None,
        "total_fit_seconds": total_fit,
        "total_predict_seconds": float(sum(predict_times)),
        "useful_ess_per_second": useful_ess / (setup + total_fit) if setup + total_fit > 0 else None,
        "peak_rss_mb": peak_rss_mb(),
        "rss_growth_after_first_fit_mb": (rss[-1] - rss[0]) if len(rss) > 1 else None,
    }


def run_child(engine: str, config: RepeatedConfig) -> dict[str, Any]:
    datasets = make_datasets(config)
    run = EngineRun()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "engine": engine,
        "config": config.to_dict(),
        "environment": environment_metadata(config.base),
    }
    try:
        import arviz  # noqa: F401 - imported untimed, before any engine

        ADAPTERS[engine](config, datasets, run)
        payload["status"] = "ok"
    except ModuleNotFoundError as error:
        payload.update(status="unavailable", reason=f"optional dependency is not installed: {error.name}")
    except Exception as error:  # noqa: BLE001 - setup failures become result data
        payload.update(
            status="error", reason=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc()[-8_000:],
        )
    payload.update(
        setup_phases_seconds=run.setup_phases,
        fits=run.fits,
        notes=run.notes,
        summary=summarize(run),
    )
    return payload


def run_isolated(engine: str, config_path: Path, config: RepeatedConfig) -> dict[str, Any]:
    command = [sys.executable, "-m", "benchmarks.repeated", "--config", str(config_path), "--child-engine", engine]
    # Both rustmc runs take the rustmc environment of benchmarks.run.
    env_engine = "rustmc" if engine.startswith("rustmc") else engine
    with tempfile.TemporaryDirectory(prefix=f"rustmc-repeated-{engine}-") as cache_dir:
        env = _child_environment(env_engine, config.base, cache_dir)
        proc = subprocess.run(
            command, cwd=str(Path(__file__).resolve().parent.parent), env=env,
            text=True, capture_output=True, check=False,
        )
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith(RESULT_PREFIX)), None)
    if line is None:
        return {
            "schema_version": 1, "status": "error", "engine": engine, "returncode": proc.returncode,
            "reason": "child produced no machine-readable result",
            "stdout": proc.stdout[-4_000:], "stderr": proc.stderr[-4_000:],
        }
    result = json.loads(line.removeprefix(RESULT_PREFIX))
    if proc.stderr:
        result["stderr_tail"] = proc.stderr[-4_000:]
    if proc.returncode != 0:
        result["status"] = "error"
        result["returncode"] = proc.returncode
    return result


def _without_home(value: Any) -> Any:
    """`value` with the user's home directory written as `~`, so a stored
    traceback or stderr tail does not record it."""
    home = str(Path.home())
    if isinstance(value, str):
        return value.replace(home, "~")
    if isinstance(value, list):
        return [_without_home(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_home(item) for key, item in value.items()}
    return value


def _revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        ).stdout.strip()

    # Untracked files, such as the redirected output of this run, do not count.
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": git("rev-parse", "HEAD"), "dirty": dirty}


def print_summary(results: list[dict[str, Any]]) -> None:
    header = f"{'engine':<13} {'status':<11} {'pass':>4} {'fail':>4} {'err':>3} {'setup s':>8} {'fit s':>8} {'1st fit':>8} {'ESS/s':>9} {'RSS MB':>7}"
    print(header)
    for result in results:
        s = result.get("summary", {})

        def number(key: str, digits: int = 2) -> str:
            value = s.get(key)
            return "-" if value is None else f"{value:.{digits}f}"

        print(
            f"{result['engine']:<13} {result['status']:<11} {s.get('passed', 0):>4} {s.get('failed', 0):>4} "
            f"{s.get('errors', 0):>3} {number('setup_seconds'):>8} {number('total_fit_seconds'):>8} "
            f"{number('first_fit_seconds'):>8} {number('useful_ess_per_second', 0):>9} {number('peak_rss_mb', 0):>7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", action="append", choices=ENGINES, help="repeat to select several")
    parser.add_argument("--output", type=Path, help="write complete JSON result here")
    parser.add_argument("--order-seed", type=int, default=20261019, help="shuffles the engine order")
    parser.add_argument("--child-engine", choices=ENGINES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = RepeatedConfig.from_json(args.config)
    if args.child_engine:
        print(RESULT_PREFIX + json.dumps(run_child(args.child_engine, config)))
        return 0
    engines = list(args.engine or ENGINES)
    random.Random(args.order_seed).shuffle(engines)
    revision = _revision()
    results = [_without_home(run_isolated(engine, args.config, config)) for engine in engines]
    print_summary(results)
    if args.output:
        record = {"revision": revision, "engine_order": engines, "results": results}
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
