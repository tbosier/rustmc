#!/usr/bin/env python3
"""
Verify an installed rustmc wheel actually works, run with the *target*
interpreter of a clean environment (not this repo's dev venv, and not with
the repo on sys.path).

Usage:
    /path/to/clean/venv/bin/python scripts/verify_wheel_install.py

Intended usage pattern (both locally and in CI): copy this file to a
directory outside the repo checkout, then run it with a python whose only
`rustmc` on sys.path is the one just `pip install`-ed from a built wheel.
Running it from outside the repo means sys.path[0] (the script's own
directory) cannot possibly resolve to a package named `rustmc`, which is
the concrete guard against silently importing repo source instead of the
installed wheel.

Exits 0 on success, 1 with a diagnostic message on any failure.
"""
import sys
from importlib.metadata import version


def main() -> int:
    import numpy as np

    import rustmc as rmc

    mod_file = getattr(rmc, "__file__", "") or ""
    print(f"Python:          {sys.version.split()[0]}")
    print(f"rustmc.__file__: {mod_file}")

    installed_version = version("rustmc")
    if rmc.__version__ != installed_version:
        print(
            "FAIL: module/distribution version mismatch: "
            f"{rmc.__version__!r} != {installed_version!r}",
            file=sys.stderr,
        )
        return 1
    print(f"rustmc version:  {installed_version}")

    if "site-packages" not in mod_file.replace("\\", "/").split("/"):
        print(
            f"FAIL: rustmc was not loaded from a site-packages directory: {mod_file}",
            file=sys.stderr,
        )
        return 1

    expected_api = {
        "ModelBuilder", "ModelSpec", "ParamRef", "VectorParamRef", "Expr",
        "FitResult", "BatchResult", "CompiledModel", "BoundModel", "BatchFit",
        "LinearGaussianStateSpace", "KalmanFilterResult", "KalmanSmootherResult",
        "ForecastResult", "InverseGammaPrior", "BayesianLocalLevel",
        "BayesianLocalLevelFit", "BayesianForecastResult",
        "BayesianLocalLinearTrend", "BayesianLocalLinearTrendFit",
        "BayesianTrendForecast", "NormalInverseGammaPrior",
        "BayesianAutoRegression", "BayesianAR", "BayesianARFit",
        "BayesianARForecast", "ParameterError", "StateSpaceError", "sample",
        "batch_sample", "sample_prior_predictive",
    }
    missing = {name for name in expected_api if not hasattr(rmc, name)}
    if missing:
        print(f"FAIL: missing API members: {sorted(missing)}", file=sys.stderr)
        return 1
    print(f"API surface OK: {sorted(expected_api)}")

    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    alpha_true, beta_true, sigma_true = 1.5, 2.5, 1.0
    y = alpha_true + beta_true * x + rng.standard_normal(n) * sigma_true

    builder = rmc.ModelBuilder(data={"x": x, "y": y})
    alpha = builder.normal_prior("alpha", mu=0.0, sigma=10.0)
    beta = builder.normal_prior("beta", mu=0.0, sigma=10.0)
    sigma = builder.half_normal_prior("sigma", sigma=2.0)
    builder.normal_likelihood("obs", mu_expr=alpha + beta * "x", sigma=sigma, observed_key="y")
    model = builder.build()

    fit = rmc.sample(model_spec=model, chains=2, draws=300, warmup=300, seed=42)
    means = fit.mean()
    print(f"Posterior means: {means}")

    if abs(means["alpha"] - alpha_true) > 1.5:
        print(f"FAIL: alpha mean {means['alpha']} too far from {alpha_true}", file=sys.stderr)
        return 1
    if abs(means["beta"] - beta_true) > 1.5:
        print(f"FAIL: beta mean {means['beta']} too far from {beta_true}", file=sys.stderr)
        return 1
    if means["sigma"] <= 0:
        print(f"FAIL: sigma mean {means['sigma']} not positive", file=sys.stderr)
        return 1

    samples = fit.get_samples()
    alpha_arr = samples["alpha"]
    if not isinstance(alpha_arr, np.ndarray) or alpha_arr.dtype != np.float64:
        print(f"FAIL: expected float64 ndarray, got {type(alpha_arr)} {getattr(alpha_arr, 'dtype', None)}", file=sys.stderr)
        return 1

    print("OK: wheel install verified (import, API surface, numpy interop, end-to-end sampling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
