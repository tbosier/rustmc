"""Run isolated, matched linear-regression benchmarks across Bayesian engines."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.protocol import (
    ENGINE_NAMES,
    BenchmarkConfig,
    PhaseTimer,
    make_linear_regression,
    posterior_quality,
    result_payload,
)

RESULT_PREFIX = "RUSTMC_BENCHMARK_RESULT="


def _rustmc(config: BenchmarkConfig, problem: Any) -> dict[str, Any]:
    timer = PhaseTimer()
    with timer.phase("import"):
        import arviz as az
        import rustmc as rmc

    with timer.phase("build"):
        builder = rmc.ModelBuilder()
        beta = builder.vector_normal_prior(
            "beta", n=config.parameters, mu=0.0, sigma=config.prior_sigma
        )
        builder.normal_likelihood(
            "obs",
            mu_expr=beta @ "X",
            sigma=config.observation_sigma,
            observed_key="y",
        )
    with timer.phase("compile"):
        compiled = builder.compile()
    with timer.phase("bind"):
        bound = compiled.bind({"X": problem.x, "y": problem.y}, id="benchmark")
    with timer.phase("warmup_sample"):
        fit = compiled.sample(
            bound,
            chains=config.chains,
            warmup=config.warmup,
            draws=config.draws,
            seed=config.sampler_seed,
            threads=config.threads,
            target_accept=config.target_accept,
            max_tree_depth=config.max_tree_depth,
            show_progress=False,
        )
    with timer.phase("postprocess"):
        raw = fit.get_samples_2d()
        names = sorted(
            (name for name in raw if name.startswith("beta[")),
            key=lambda name: int(name.removeprefix("beta[").removesuffix("]")),
        )
        samples = np.stack([np.asarray(raw[name]) for name in names], axis=2)
        quality = posterior_quality(
            samples, problem, config, sum(fit.divergences()), az
        )
    payload = result_payload(
        engine="rustmc",
        config=config,
        problem=problem,
        phases=timer.phases,
        quality=quality,
        notes=[
            "compile is rustmc graph compilation, not native-code generation",
            "the public rustmc API reports adaptation and retained sampling as warmup_sample",
        ],
    )
    payload["sampler_telemetry"] = {
        "adapted_step_sizes": list(fit.step_sizes()),
        "transition_diagnostics": fit.transition_diagnostics(),
    }
    return payload


def _build_pymc_model(pm: Any, config: BenchmarkConfig, problem: Any) -> Any:
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=config.prior_sigma, shape=config.parameters)
        pm.Normal(
            "obs",
            mu=pm.math.dot(problem.x, beta),
            sigma=config.observation_sigma,
            observed=problem.y,
        )
    return model


def _pymc(config: BenchmarkConfig, problem: Any) -> dict[str, Any]:
    timer = PhaseTimer()
    with timer.phase("import"):
        import arviz as az
        import pymc as pm

    with timer.phase("build"):
        model = _build_pymc_model(pm, config, problem)
    with timer.phase("compile_warmup_sample"), model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.warmup,
            chains=config.chains,
            cores=min(config.threads, config.chains),
            random_seed=config.sampler_seed,
            nuts_sampler="pymc",
            target_accept=config.target_accept,
            nuts={"max_treedepth": config.max_tree_depth},
            progressbar=False,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": False},
        )
    with timer.phase("postprocess"):
        samples = np.asarray(idata.posterior["beta"].values)
        divergences = int(idata.sample_stats["diverging"].values.sum())
        quality = posterior_quality(samples, problem, config, divergences, az)
    return result_payload(
        engine="pymc",
        config=config,
        problem=problem,
        phases=timer.phases,
        quality=quality,
        notes=[
            "PyMC's public pm.sample path combines sampler compilation, warmup, and retained sampling",
            "compute_convergence_checks is disabled because common ArviZ diagnostics are timed separately",
        ],
    )


def _nutpie(config: BenchmarkConfig, problem: Any) -> dict[str, Any]:
    timer = PhaseTimer()
    with timer.phase("import"):
        import arviz as az
        import nutpie
        import pymc as pm

    with timer.phase("build"):
        model = _build_pymc_model(pm, config, problem)
    with timer.phase("compile"):
        compiled = nutpie.compile_pymc_model(model)
    with timer.phase("warmup_sample"):
        idata = nutpie.sample(
            compiled,
            draws=config.draws,
            tune=config.warmup,
            chains=config.chains,
            cores=min(config.threads, config.chains),
            seed=config.sampler_seed,
            target_accept=config.target_accept,
            maxdepth=config.max_tree_depth,
            save_warmup=False,
            progress_bar=False,
        )
    with timer.phase("postprocess"):
        samples = np.asarray(idata.posterior["beta"].values)
        divergences = int(idata.sample_stats["diverging"].values.sum())
        quality = posterior_quality(samples, problem, config, divergences, az)
    return result_payload(
        engine="nutpie",
        config=config,
        problem=problem,
        phases=timer.phases,
        quality=quality,
        notes=[
            "PyMC model conversion and nutpie compilation are measured separately",
            "nutpie's public API combines warmup and retained sampling",
        ],
    )


def _numpyro(config: BenchmarkConfig, problem: Any) -> dict[str, Any]:
    timer = PhaseTimer()
    with timer.phase("import"):
        import arviz as az
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        numpyro.enable_x64()

    with timer.phase("build"):
        x = jnp.asarray(problem.x)
        y = jnp.asarray(problem.y)

        def model(x_data: Any, y_data: Any) -> None:
            beta = numpyro.sample(
                "beta",
                dist.Normal(0.0, config.prior_sigma).expand([config.parameters]),
            )
            numpyro.sample(
                "obs",
                dist.Normal(x_data @ beta, config.observation_sigma),
                obs=y_data,
            )

        kernel = NUTS(
            model,
            target_accept_prob=config.target_accept,
            max_tree_depth=config.max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=config.warmup,
            num_samples=config.draws,
            num_chains=config.chains,
            chain_method="parallel",
            progress_bar=False,
        )
        warmup_key, sample_key = jax.random.split(jax.random.key(config.sampler_seed))

    with timer.phase("compile_warmup"):
        mcmc.warmup(warmup_key, x, y, collect_warmup=False)
        jax.block_until_ready(mcmc.post_warmup_state)
    with timer.phase("sample"):
        mcmc.run(sample_key, x, y, extra_fields=("diverging",))
        sample_mapping = mcmc.get_samples(group_by_chain=True)
        jax.block_until_ready(sample_mapping["beta"])
    with timer.phase("postprocess"):
        samples = np.asarray(sample_mapping["beta"])
        divergences = int(
            np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum()
        )
        quality = posterior_quality(samples, problem, config, divergences, az)
    return result_payload(
        engine="numpyro",
        config=config,
        problem=problem,
        phases=timer.phases,
        quality=quality,
        notes=[
            "NumPyro is forced to float64 to match the other engines",
            "compile_warmup includes JAX compilation and adaptation; asynchronous work is blocked before timing stops",
            "sample begins from NumPyro's documented post_warmup_state",
        ],
    )


ADAPTERS = {
    "rustmc": _rustmc,
    "pymc": _pymc,
    "nutpie": _nutpie,
    "numpyro": _numpyro,
}


def run_child(engine: str, config: BenchmarkConfig) -> dict[str, Any]:
    problem = make_linear_regression(config)
    try:
        return ADAPTERS[engine](config, problem)
    except ModuleNotFoundError as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "engine": engine,
            "config": asdict(config),
            "data_sha256": problem.digest,
            "reason": f"optional dependency is not installed: {exc.name}",
        }
    except ImportError as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "engine": engine,
            "config": asdict(config),
            "data_sha256": problem.digest,
            "reason": f"optional dependency could not be imported: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 - child failures must become result data
        return {
            "schema_version": 1,
            "status": "error",
            "engine": engine,
            "config": asdict(config),
            "data_sha256": problem.digest,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-8_000:],
        }


def _child_environment(
    engine: str, config: BenchmarkConfig, cache_dir: str
) -> dict[str, str]:
    env = os.environ.copy()
    thread_value = str(config.threads)
    env.update(
        {
            "RAYON_NUM_THREADS": thread_value,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "XDG_CACHE_HOME": cache_dir,
            "MPLCONFIGDIR": cache_dir,
            "NUMBA_CACHE_DIR": cache_dir,
        }
    )
    pytensor_flags = [
        flag
        for flag in env.get("PYTENSOR_FLAGS", "").split(",")
        if flag and not flag.strip().startswith("base_compiledir=")
    ]
    if engine in {"pymc", "nutpie"}:
        pytensor_flags.append(f"base_compiledir={cache_dir}")
    if pytensor_flags:
        env["PYTENSOR_FLAGS"] = ",".join(pytensor_flags)
    else:
        env.pop("PYTENSOR_FLAGS", None)

    xla_flags = [
        flag
        for flag in env.get("XLA_FLAGS", "").split()
        if not flag.startswith("--xla_force_host_platform_device_count=")
    ]
    if engine == "numpyro":
        xla_flags.append(f"--xla_force_host_platform_device_count={config.threads}")
    if xla_flags:
        env["XLA_FLAGS"] = " ".join(xla_flags)
    else:
        env.pop("XLA_FLAGS", None)
    return env


def run_isolated(engine: str, config_path: Path) -> dict[str, Any]:
    config = BenchmarkConfig.from_json(config_path)
    command = [
        sys.executable,
        "-m",
        "benchmarks.run",
        "--config",
        str(config_path),
        "--child-engine",
        engine,
    ]
    with tempfile.TemporaryDirectory(prefix=f"rustmc-benchmark-{engine}-") as cache_dir:
        proc = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=_child_environment(engine, config, cache_dir),
            text=True,
            capture_output=True,
            check=False,
        )
    result_line = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        return {
            "schema_version": 1,
            "status": "error",
            "engine": engine,
            "returncode": proc.returncode,
            "reason": "child produced no machine-readable result",
            "stdout": proc.stdout[-4_000:],
            "stderr": proc.stderr[-4_000:],
        }
    result = json.loads(result_line.removeprefix(RESULT_PREFIX))
    if proc.stderr:
        result["stderr_tail"] = proc.stderr[-4_000:]
    if proc.returncode != 0:
        result["status"] = "error"
        result["returncode"] = proc.returncode
    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    print(
        f"{'engine/run':<14} {'status':<12} {'gate':<6} {'fit_s':>10} {'ESS':>10} "
        f"{'ESS/fit_s':>12} {'R-hat':>9} {'div':>6} {'mean RMSE':>12}"
    )
    print("-" * 104)
    for result in results:
        label = f"{result['engine']}/{result.get('repetition', 1)}"
        if result.get("status") != "ok":
            print(
                f"{label:<14} {result['status']:<12} {'-':<6} "
                f"{result.get('reason', '')}"
            )
            continue
        quality = result["quality"]
        gate = "PASS" if result["quality_gate"]["passed"] else "FAIL"
        print(
            f"{label:<14} {'ok':<12} {gate:<6} "
            f"{result['timing']['fit_seconds']:>10.3f} "
            f"{quality['ess_bulk_mean']:>10.1f} "
            f"{result['ess_per_fit_second']:>12.1f} "
            f"{quality['rhat_rank_max']:>9.4f} "
            f"{quality['divergences']:>6d} "
            f"{quality['mean_rmse_vs_exact_posterior']:>12.5g}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "standard.json",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=ENGINE_NAMES,
        help="engine to run; repeatable (default: all)",
    )
    parser.add_argument("--output", type=Path, help="write complete JSON result here")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--randomize-order",
        action="store_true",
        help="shuffle engine order independently for each repetition",
    )
    parser.add_argument("--order-seed", type=int, default=20260802)
    parser.add_argument("--dry-run", action="store_true", help="validate config/data only")
    parser.add_argument("--list-engines", action="store_true")
    parser.add_argument("--child-engine", choices=ENGINE_NAMES, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_engines:
        print("\n".join(ENGINE_NAMES))
        return 0
    config = BenchmarkConfig.from_json(args.config)
    problem = make_linear_regression(config)
    if args.dry_run:
        print(json.dumps({"config": asdict(config), "data_sha256": problem.digest}, indent=2))
        return 0
    if args.child_engine:
        payload = run_child(args.child_engine, config)
        print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] in {"ok", "unavailable"} else 1

    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    engines = args.engine or list(ENGINE_NAMES)
    order_rng = random.Random(args.order_seed)
    results = []
    execution_order = []
    for repetition in range(1, args.repetitions + 1):
        ordered_engines = list(engines)
        if args.randomize_order:
            order_rng.shuffle(ordered_engines)
        execution_order.append(ordered_engines)
        for engine in ordered_engines:
            result = run_isolated(engine, args.config.resolve())
            result["repetition"] = repetition
            results.append(result)
    digests = {result.get("data_sha256") for result in results if result.get("data_sha256")}
    if len(digests) > 1:
        raise RuntimeError(f"engine subprocesses did not use identical data: {digests}")
    report = {
        "schema_version": 1,
        "config": asdict(config),
        "data_sha256": problem.digest,
        "execution": {
            "repetitions": args.repetitions,
            "randomize_order": args.randomize_order,
            "order_seed": args.order_seed,
            "engine_order": execution_order,
        },
        "results": results,
    }
    print_summary(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nFull machine-readable result: {args.output}")
    return 1 if any(result.get("status") == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
