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
import math
import os
from pathlib import Path
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
    "InverseGammaPrior",
    "BayesianHierarchicalMean",
    "BayesianHierarchicalMeanFit",
    "BayesianHierarchicalForecast",
    "BayesianLocalLevel",
    "BayesianLocalLevelFit",
    "BayesianSeasonalLocalLevel",
    "BayesianSeasonalLocalLevelFit",
    "BayesianSeasonalForecast",
    "BayesianForecastResult",
    "BayesianLocalLinearTrend",
    "BayesianLocalLinearTrendFit",
    "BayesianTrendForecast",
    "NormalInverseGammaPrior",
    "BayesianAutoRegression",
    "BayesianAR",
    "BayesianARFit",
    "BayesianARForecast",
    "ParameterError",
    "InferenceError",
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

    # Declared once so the negative control below reads the priors the model is
    # actually built from. Hard-coded control constants would go on passing if someone
    # moved a prior onto the truth, which is the failure the control exists to catch.
    location_prior = {"mu": 0.0, "sigma": 10.0}
    scale_prior = {"sigma": 2.0}
    builder = rmc.ModelBuilder(data={"x": data["x"], "y": data["y"]})
    alpha = builder.normal_prior("alpha", **location_prior)
    beta = builder.normal_prior("beta", **location_prior)
    sigma = builder.half_normal_prior("sigma", **scale_prior)
    builder.normal_likelihood("obs", mu_expr=alpha + beta * "x", sigma=sigma, observed_key="y")
    model = builder.build()

    fit = rmc.sample(model_spec=model, chains=2, draws=200, warmup=200, seed=42)

    means = fit.mean()
    assert set(means) == {"alpha", "beta", "sigma"}
    # These windows used to be +/-1.5 on alpha and beta, and `sigma > 0`. The posterior
    # SD of alpha here is about 0.072, so +/-1.5 was twenty-one posterior SDs wide and
    # its lower edge sat exactly on the Normal(0, 10) prior mean of zero: a prior-only
    # answer, whose mean over 400 draws is 0 +/- 0.5, landed inside it about half the
    # time. `sigma > 0` is a support check, not an accuracy one, and every prior draw
    # satisfies it.
    tolerances = {"alpha": 0.3, "beta": 0.3, "sigma": 0.2}
    prior_means = {
        "alpha": location_prior["mu"],
        "beta": location_prior["mu"],
        # A half-normal's mean is sigma * sqrt(2/pi).
        "sigma": scale_prior["sigma"] * math.sqrt(2 / math.pi),
    }
    for name, tolerance in tolerances.items():
        truth = data[f"{name}_true"]
        assert means[name] == pytest.approx(truth, abs=tolerance), name
        # Negative control: an answer that ignored the data would report the prior
        # mean, which must therefore be outside the window just asserted.
        assert abs(prior_means[name] - truth) > tolerance, (
            f"the {name} prior mean {prior_means[name]} is inside its accepted window"
        )

    samples = fit.get_samples()
    alpha_samples = samples["alpha"]
    assert isinstance(alpha_samples, np.ndarray)
    assert alpha_samples.dtype == np.float64
    assert alpha_samples.shape == (2 * 200,)

    summary = fit.summary()
    assert "alpha" in summary and "beta" in summary and "sigma" in summary
def test_editable_source_provenance_when_required(rustmc_module):
    root = os.environ.get("RUSTMC_REQUIRE_SOURCE_ROOT")
    if root is None:
        pytest.skip("editable source provenance not requested")
    expected = Path(root).resolve() / "python" / "rustmc"
    assert Path(rustmc_module.__file__).resolve().parent == expected
    from rustmc import _rustmc
    assert Path(_rustmc.__file__).resolve().parent == expected
