"""
Import / API-surface / end-to-end smoke tests for whatever `rustmc` is
importable in the current interpreter.

These run against a `maturin develop` editable install during local
development (`.venv/bin/python -m pytest tests/ -q`), and against an
installed wheel in CI's wheel-install job (see
.github/workflows/ci.yml). Set RUSTMC_REQUIRE_SITE_PACKAGES=1 to also
assert the module was loaded from site-packages rather than the repo --
CI's wheel job sets this; local dev runs do not need to.
"""
import os
from importlib.metadata import version

import numpy as np
import pytest


EXPECTED_API = {
    "ModelBuilder",
    "ModelSpec",
    "ParamRef",
    "VectorParamRef",
    "Expr",
    "FitResult",
    "BatchResult",
    "CompiledModel",
    "BoundModel",
    "BatchFit",
    "LinearGaussianStateSpace",
    "KalmanFilterResult",
    "KalmanSmootherResult",
    "ForecastResult",
    "ParameterError",
    "StateSpaceError",
    "sample",
    "batch_sample",
    "sample_prior_predictive",
}


def test_import(rustmc_module):
    assert rustmc_module is not None
    assert rustmc_module.__version__ == version("rustmc")


def test_site_packages_when_required(rustmc_module, assert_installed_from_site_packages):
    if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") != "1":
        pytest.skip("RUSTMC_REQUIRE_SITE_PACKAGES not set; skipping strict location check")
    mod_file = assert_installed_from_site_packages(rustmc_module)
    print(f"rustmc loaded from: {mod_file}")


def test_public_api_surface(rustmc_module):
    present = {name for name in EXPECTED_API if hasattr(rustmc_module, name)}
    missing = EXPECTED_API - present
    assert not missing, f"rustmc is missing expected public API members: {sorted(missing)}"


def test_numpy_interop_and_end_to_end_sampling(rustmc_module, linreg_data):
    rmc = rustmc_module
    data = linreg_data

    builder = rmc.ModelBuilder(data={"x": data["x"], "y": data["y"]})
    alpha = builder.normal_prior("alpha", mu=0.0, sigma=10.0)
    beta = builder.normal_prior("beta", mu=0.0, sigma=10.0)
    sigma = builder.half_normal_prior("sigma", sigma=2.0)
    builder.normal_likelihood("obs", mu_expr=alpha + beta * "x", sigma=sigma, observed_key="y")
    model = builder.build()

    fit = rmc.sample(model_spec=model, chains=2, draws=200, warmup=200, seed=42)

    means = fit.mean()
    assert set(means) == {"alpha", "beta", "sigma"}
    assert abs(means["alpha"] - data["alpha_true"]) < 1.5
    assert abs(means["beta"] - data["beta_true"]) < 1.5
    assert means["sigma"] > 0

    samples = fit.get_samples()
    alpha_samples = samples["alpha"]
    assert isinstance(alpha_samples, np.ndarray)
    assert alpha_samples.dtype == np.float64
    assert alpha_samples.shape == (2 * 200,)

    summary = fit.summary()
    assert "alpha" in summary and "beta" in summary and "sigma" in summary
