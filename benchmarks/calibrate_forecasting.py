#!/usr/bin/env python3
"""Seeded prior-simulation forecast calibration; every attempted fit is retained.

Run against a built rustmc extension, e.g.:
  python benchmarks/calibrate_forecasting.py --output benchmarks/results/2026-09-09-calibration

Generators use NumPy distributions and explicit recursions, not rustmc predictive
methods. Drawn variance/parameter priors match the fitted specifications, so the
experiment measures prior-averaged Bayesian calibration. It does not establish
calibration for arbitrary fixed parameters, misspecified data, or real portfolios.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import rustmc as mc

MODELS = (
    "structural_gaussian", "structural_student_t", "dynamic_poisson",
    "hierarchical_dynamic_gaussian",
)


def seed_for(seed: int, model: str, replicate: int, phase: str) -> int:
    payload = f"rustmc-calibration-v1:{seed}:{model}:{replicate}:{phase}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def generate(model: str, seed: int, history: int, horizon: int):
    rng = np.random.default_rng(seed)
    length = history + horizon
    V, C = mc.VarianceParameter, mc.StructuralComponent
    if model.startswith("structural_"):
        # IG(a,b) = 1/Gamma(a, scale=1/b); independent initial level prior.
        q = 1.0 / rng.gamma(4.0, 1.0 / 0.12)
        r = 1.0 / rng.gamma(4.0, 1.0 / 0.75)
        state = rng.normal(0.0, 1.0)
        states = []
        for _ in range(length):
            state += rng.normal(0.0, np.sqrt(q))
            states.append(state)
        df = 5.0 if model == "structural_student_t" else None
        noise = rng.normal(size=length) if df is None else rng.standard_t(df, size=length)
        values = np.asarray(states) + np.sqrt(r) * noise
        training = values[:history].copy()
        training[[history // 3, 2 * history // 3]] = np.nan
        specification = mc.StructuralModel([
            C.level("level", V.inverse_gamma(4.0, 0.12), 0.0, 1.0),
        ], V.inverse_gamma(4.0, 0.75), student_df=df)
        return specification, training, values[history:], {}, {}, {
            "process_variance": q, "observation_scale_squared": r, "student_df": df,
        }

    x = np.sin(np.arange(length) * 0.7)[:, None]
    if model == "dynamic_poisson":
        coefficient_sd, process_sd = 0.6, 0.12
        beta = rng.normal(0.0, coefficient_sd, size=2)
        state = np.cumsum(rng.normal(0.0, process_sd, size=length))
        exposure = 0.5 + 0.5 * (np.arange(length) % 4)
        exposure[history // 4] = 0.0
        rate = exposure * np.exp(beta[0] + x[:, 0] * beta[1] + state)
        values = rng.poisson(rate).astype(float)
        training = values[:history].copy()
        training[history // 2] = np.nan
        specification = mc.BayesianDynamicPoisson(
            coefficient_sd=coefficient_sd, group_sd=0.0, process_sd=process_sd,
        )
        fit_data = {"exog": x[None, :history], "exposure": exposure[None, :history]}
        future_data = {"exog": x[None, history:], "exposure": exposure[None, history:]}
        return specification, training[None], values[None, history:], fit_data, future_data, {
            "coefficients": beta.tolist(), "process_sd": process_sd,
        }

    groups = 3
    coefficient_sd, group_sd = 0.5, 0.3
    process_sd, shared_process_sd, observation_sd = 0.1, 0.05, 0.6
    population = rng.normal(0.0, coefficient_sd, size=2)
    coefficients = population + rng.normal(0.0, group_sd, size=(groups, 2))
    shared = rng.normal(0.0, shared_process_sd, size=length)
    innovation = shared + rng.normal(0.0, process_sd, size=(groups, length))
    states = np.cumsum(innovation, axis=1)
    values = (coefficients[:, 0, None] + coefficients[:, 1, None] * x[:, 0]
              + states + rng.normal(0.0, observation_sd, size=(groups, length)))
    training = values[:, :history].copy()
    training[1, 1::2] = np.nan
    training[2, np.arange(history) % 4 != 0] = np.nan
    design = np.broadcast_to(x, (groups, length, 1)).copy()
    specification = mc.BayesianHierarchicalDynamicRegression(
        coefficient_sd=coefficient_sd, group_sd=group_sd, process_sd=process_sd,
        shared_process_sd=shared_process_sd, observation_sd=observation_sd,
    )
    return specification, training, values[:, history:], {"exog": design[:, :history]}, {
        "exog": design[:, history:],
    }, {
        "population_coefficients": population.tolist(), "group_coefficients": coefficients.tolist(),
        "process_sd": process_sd, "shared_process_sd": shared_process_sd,
        "observation_sd": observation_sd,
    }


def score(samples: np.ndarray, actual: np.ndarray) -> dict:
    """Empirical CRPS, central coverage, and width; retain group/horizon axes."""
    samples = np.asarray(samples, dtype=float)
    actual = np.asarray(actual, dtype=float)
    flat = samples.reshape((-1,) + actual.shape)
    if not np.isfinite(flat).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite forecasts or outcomes")
    lower, upper = np.quantile(flat, [0.05, 0.95], axis=0)
    count = len(flat)
    ordered = np.sort(flat, axis=0)
    weights = (2 * np.arange(1, count + 1) - count - 1).reshape(
        (count,) + (1,) * actual.ndim
    )
    crps = np.abs(flat - actual).mean(axis=0) - (weights * ordered).sum(axis=0) / count**2
    return {
        "coverage90": ((lower <= actual) & (actual <= upper)).astype(float).tolist(),
        "width90": (upper - lower).tolist(), "crps": crps.tolist(),
    }


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        # Preserve detected divergence of R-hat separately from undefined values.
        return "infinity" if np.isposinf(value) else None
    return value


def summarize(records: list[dict], horizon: int) -> dict:
    successful = [r for r in records if r["status"] == "ok"]
    output = {"attempted": len(records), "successful": len(successful),
              "failed": len(records) - len(successful)}
    for target in ("scores", "aggregate_scores"):
        selected = [r for r in successful if target in r]
        if not selected:
            continue
        target_result = {}
        for key in ("coverage90", "width90", "crps"):
            # The replicate is the independent MC unit. Within-panel groups share
            # parameters/shocks, so average groups before estimating MC standard error.
            per_replicate = np.stack([
                np.asarray(r[target][key]).reshape((-1, horizon)).mean(axis=0)
                for r in selected
            ])
            target_result[key] = {
                "mean": per_replicate.mean(axis=0).tolist(),
                "monte_carlo_se": (per_replicate.std(axis=0, ddof=1) / np.sqrt(len(selected))).tolist()
                if len(selected) > 1 else [None] * horizon,
            }
        output[target] = target_result
    panel_records = [r for r in successful if "aggregate_scores" in r]
    if panel_records:
        output["series_scores"] = {}
        for group, label in enumerate(panel_records[0]["series_labels"]):
            group_scores = {}
            for key in ("coverage90", "width90", "crps"):
                values = np.stack([np.asarray(r["scores"][key])[group] for r in panel_records])
                group_scores[key] = {
                    "mean": values.mean(axis=0).tolist(),
                    "monte_carlo_se": (values.std(axis=0, ddof=1)/np.sqrt(len(values))).tolist()
                    if len(values) > 1 else [None]*horizon,
                }
            output["series_scores"][label] = group_scores
    if successful:
        maximum_rhat = [r["max_rhat"] for r in successful]
        bulk = [r["min_ess_bulk"] for r in successful]
        tail = [r["min_ess_tail"] for r in successful]
        output["diagnostics"] = {
            "max_rhat_across_fits": max(maximum_rhat),
            "median_fit_max_rhat": float(np.median(maximum_rhat)),
            "min_ess_bulk_across_fits": min(bulk),
            "median_fit_min_ess_bulk": float(np.median(bulk)),
            "min_ess_tail_across_fits": min(tail),
            "median_fit_min_ess_tail": float(np.median(tail)),
            "fits_with_rhat_above_1_01": sum(x > 1.01 for x in maximum_rhat),
            "fits_with_min_bulk_ess_below_100": sum(x < 100 for x in bulk),
            "fits_with_min_tail_ess_below_100": sum(x < 100 for x in tail),
            "fits_with_min_bulk_ess_below_400": sum(x < 400 for x in bulk),
            "fits_with_min_tail_ess_below_400": sum(x < 400 for x in tail),
            "unavailable_parameter_diagnostics": sum(r["unavailable_parameter_diagnostics"] for r in successful),
        }
    output["fit_forecast_seconds"] = sum(r["elapsed_seconds"] for r in records)
    return output


def markdown(artifact: dict) -> str:
    settings = artifact["settings"]
    lines = [
        "# Repeated forecasting calibration — 2026-09-09", "",
        f"Each model has {settings['replicates']} independently generated replicates, "
        f"{settings['history']} training periods, and {settings['horizon']} forecast periods. "
        f"Fits use {settings['chains']} chains, {settings['warmup']} warmup sweeps and "
        f"{settings['draws']} retained draws per chain. Seed: {settings['seed']}.", "",
        "Parameters are drawn independently from the stated fitted priors. NumPy generates "
        "the latent recursions and observations; rustmc prior/predictive methods are not "
        "used to generate data. These are prior-averaged calibration checks under correct "
        "specification, not evidence of calibration under arbitrary fixed parameters or "
        "misspecification. Every attempted fit, exception, seed, generating parameter, "
        "and parameter diagnostic is retained in the JSON companion.", "",
        "Coverage uses pointwise central 90% predictive intervals. CRPS scores the empirical "
        "predictive distribution; lower is better, but scales differ across models. Reported "
        "Monte Carlo standard errors use independent replicates. Panel series are averaged "
        "within each replicate before computing standard errors. Discrete Poisson intervals "
        "can overcover nominal levels. Values are estimates without a calibration pass/fail gate.", "",
        "| Model | Successful / attempted | Maximum R-hat | Minimum bulk / tail ESS | Fits R-hat > 1.01 | Fits bulk / tail ESS < 400 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in artifact["summaries"].items():
        d = result.get("diagnostics", {})
        lines.append(f"| {name} | {result['successful']} / {result['attempted']} | "
                     f"{d.get('max_rhat_across_fits', float('nan')):.4f} | "
                     f"{d.get('min_ess_bulk_across_fits', float('nan')):.1f} / "
                     f"{d.get('min_ess_tail_across_fits', float('nan')):.1f} | "
                     f"{d.get('fits_with_rhat_above_1_01', 'N/A')} | "
                     f"{d.get('fits_with_min_bulk_ess_below_400', 'N/A')} / "
                     f"{d.get('fits_with_min_tail_ess_below_400', 'N/A')} |")
    lines.extend(["", "Diagnostics cover inferred variances and terminal states for structural "
                  "models, and population/group coefficients and terminal states for dynamic "
                  "GLMs. Deterministically constant parameters are excluded from aggregate "
                  "diagnostic extrema, with their names retained per fit. Historical latent "
                  "states and Student-t precision variables are not exhaustively diagnosed. "
                  "The R-hat/ESS flag counts describe this exact schedule; flagged fits remain "
                  "in the calibration tables and may need longer sampling. The stated "
                  "diagnostic targets are R-hat <= 1.01 and both bulk/tail ESS >= 400 for "
                  "every monitored nonconstant parameter; these are precision/convergence "
                  "screening criteria, not a calibration success test.", ""])
    for name, result in artifact["summaries"].items():
        lines.extend([f"## {name}", ""])
        for target in ("scores", "aggregate_scores"):
            if target not in result:
                continue
            lines.extend(["Joint panel sum:" if target == "aggregate_scores" else "Observation forecasts (groups averaged for the panel):", "",
                          "| Horizon | Coverage ± MC SE | CRPS ± MC SE | Mean interval width |",
                          "|---:|---:|---:|---:|"])
            scores = result[target]
            for h in range(settings["horizon"]):
                coverage, crps, width = (scores[k] for k in ("coverage90", "crps", "width90"))
                def se(item):
                    value = item["monte_carlo_se"][h]
                    return "N/A" if value is None else f"{value:.3f}"
                lines.append(f"| {h+1} | {coverage['mean'][h]:.3f} ± {se(coverage)} | "
                             f"{crps['mean'][h]:.3f} ± {se(crps)} | {width['mean'][h]:.3f} |")
            lines.append("")
        if "series_scores" in result:
            lines.extend(["Series-level forecasts, with missingness labels:", "",
                          "| Series | Horizon | Coverage ± MC SE | CRPS |",
                          "|---|---:|---:|---:|"])
            for label, scores in result["series_scores"].items():
                for h in range(settings["horizon"]):
                    error = scores["coverage90"]["monte_carlo_se"][h]
                    error_text = "N/A" if error is None else f"{error:.3f}"
                    lines.append(f"| {label} | {h+1} | {scores['coverage90']['mean'][h]:.3f} ± {error_text} | "
                                 f"{scores['crps']['mean'][h]:.3f} |")
            lines.append("")
        failures = [r for r in artifact["records"] if r["model"] == name and r["status"] != "ok"]
        for failure in failures:
            lines.append(f"Failed replicate {failure['replicate']}: `{failure['error']}`")
    lines.extend([
        "## Interpretation and reproducibility", "",
        f"With {settings['replicates']} independent scalar outcomes per horizon, nominal 90% "
        f"coverage has approximate binomial MC SE {np.sqrt(.9*.1/settings['replicates']):.3f}; "
        "a rough two-standard-error band is wide. Horizons share training data and future "
        "paths, so their coverage estimates are correlated. No multiple-testing adjustment "
        "or universal calibration claim is made. Per-horizon zero estimated SE can occur "
        "when this small experiment covers every outcome; it is not certainty of perfect "
        "population coverage. CRPS uncertainty also reflects occasional high-count/tail outcomes.", "",
        "The hierarchical experiment has three series, uncertain shared/group regression "
        "coefficients, group state innovations, and shared dynamic shocks. Two groups have "
        "50% and 75% missing training values. Its aggregate intervals sum groups within "
        "aligned joint posterior draws. Structural models include two missing training periods; "
        "the count model includes one missing period and a zero-exposure training period.", "",
        "Reproduce against the recorded source revision and rebuilt extension:", "",
        "```bash",
        "python benchmarks/calibrate_forecasting.py "
        f"--replicates {settings['replicates']} --chains {settings['chains']} "
        f"--draws {settings['draws']} --warmup {settings['warmup']} "
        f"--history {settings['history']} --horizon {settings['horizon']} --seed {settings['seed']} "
        f"--output {settings['output']}",
        "```", "",
        f"Recorded source revision: `{artifact['environment']['source_revision']}`. "
        f"Backend version: `{artifact['environment']['rustmc_version']}`; "
        f"NumPy: `{artifact['environment']['numpy_version']}`. "
        "Runtime figures include scoring/diagnostics and are not throughput benchmarks.",
        "The JSON companion records the imported native module location and checksum. "
        "Use the matching package and source revision when reproducing the run.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("replicates", 32), ("chains", 2), ("draws", 1000),
                          ("warmup", 1000), ("history", 24), ("horizon", 4), ("seed", 20260909)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output", default="benchmarks/results/2026-09-09-calibration")
    args = parser.parse_args()
    if min(args.replicates, args.chains, args.draws, args.horizon) < 1 or args.warmup < 0 or args.history < 8:
        parser.error("positive replicate/chain/draw/horizon counts, nonnegative warmup, and history>=8 required")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    import importlib
    native = importlib.import_module("rustmc._rustmc")
    native_path = Path(native.__file__)
    artifact = {
        "format": "rustmc.forecast_calibration", "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(), "settings": vars(args),
        "environment": {"source_revision": revision, "rustmc_version": getattr(mc, "__version__", "unknown"),
                        "numpy_version": np.__version__, "python_version": platform.python_version(),
                        "platform": platform.platform(), "extension_path": str(native_path),
                        "extension_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                        "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "records": [], "summaries": {},
    }
    for name in args.models:
        for replicate in range(args.replicates):
            started = time.perf_counter()
            record = {"model": name, "replicate": replicate, "status": "error"}
            seeds = {phase: seed_for(args.seed, name, replicate, phase) for phase in ("generate", "fit", "forecast")}
            record["seeds"] = seeds
            try:
                model, training, actual, fit_data, future_data, truth = generate(name, seeds["generate"], args.history, args.horizon)
                record["generating_parameters"] = truth
                if name == "hierarchical_dynamic_gaussian":
                    record["series_labels"] = ["group_0_dense", "group_1_half_observed", "group_2_quarter_observed"]
                elif name == "dynamic_poisson":
                    record["series_labels"] = ["count_series"]
                record["observed_training_count"] = int(np.isfinite(training).sum())
                fit = model.fit(training, **fit_data, chains=args.chains, draws=args.draws,
                                warmup=args.warmup, seed=seeds["fit"])
                forecast = fit.forecast(args.horizon, **future_data, seed=seeds["forecast"])
                samples = np.asarray(forecast.observation_samples)
                record["scores"] = score(samples, actual)
                record["actual"] = actual.tolist()
                if actual.ndim == 2 and actual.shape[0] > 1:
                    record["aggregate_scores"] = score(samples.sum(axis=2), actual.sum(axis=0))
                diagnostics = fit.diagnostics()
                parameter_samples = fit.get_samples_2d()
                constants = {key for key, value in parameter_samples.items() if np.ptp(value) == 0.0}
                monitored = [p for p in diagnostics if p["name"] not in constants]
                record["constant_parameters"] = sorted(constants)
                record["diagnostics"] = diagnostics
                record["unavailable_parameter_diagnostics"] = sum(
                    any(p[key] is None for key in ("r_hat", "ess_bulk", "ess_tail")) for p in monitored
                )
                rhats = [p["r_hat"] for p in monitored if p["r_hat"] is not None]
                bulk = [p["ess_bulk"] for p in monitored if p["ess_bulk"] is not None]
                tail = [p["ess_tail"] for p in monitored if p["ess_tail"] is not None]
                record["max_rhat"] = max(rhats, default=float("inf"))
                record["min_ess_bulk"] = min(bulk, default=0.0)
                record["min_ess_tail"] = min(tail, default=0.0)
                record["status"] = "ok"
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()
            record["elapsed_seconds"] = time.perf_counter() - started
            artifact["records"].append(record)
            print(f"{name} {replicate+1}/{args.replicates}: {record['status']} "
                  f"rhat={record.get('max_rhat', float('nan')):.4f} "
                  f"bulk_ess={record.get('min_ess_bulk', float('nan')):.1f} "
                  f"{record['elapsed_seconds']:.2f}s", flush=True)
        artifact["summaries"][name] = summarize([r for r in artifact["records"] if r["model"] == name], args.horizon)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".md").write_text(markdown(artifact))
    output.with_suffix(".json").write_text(json.dumps(finite_or_none(artifact), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output.with_suffix('.json')} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
