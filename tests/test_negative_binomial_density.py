"""Large-count fitted diagnostics retain negative-binomial likelihood curvature."""

import numpy as np
import pytest

import rustmc


@pytest.mark.parametrize("alpha", [1e14, 1e16])
def test_large_negative_binomial_target_pointwise_and_artifact_replay(alpha):
    count = 1e14
    builder = rustmc.ModelBuilder({"y": np.array([count])})
    z = builder.normal_prior("z", 0.0, 10.0)
    eta = np.log(count) + z * np.sqrt(1 / count + 1 / alpha)
    builder.negative_binomial_likelihood("obs", eta, alpha, "y")
    model = builder.compile()
    # At these counts the independent local Gaussian limit has O(1e-7)
    # corrections; its variance is mu + mu**2/alpha.
    mode = -0.5 * np.log(2 * np.pi * count * (1 + count / alpha))
    prior_normalizer = -0.5 * np.log(2 * np.pi) - np.log(10.0)
    for value in [-1.0, 0.0, 1.0]:
        lp, grad = model.log_density({}, [value])
        assert lp == pytest.approx(mode + prior_normalizer - 0.505 * value**2, abs=3e-6)
        assert grad[0] == pytest.approx(-1.01 * value, abs=3e-6)
    fit = model.sample({}, chains=1, draws=20, warmup=30, seed=73, show_progress=False)
    z_draws = fit.get_samples_2d()["z"]
    np.testing.assert_allclose(
        fit.log_likelihood()["obs"][..., 0], mode - 0.5 * z_draws**2,
        atol=3e-5, rtol=0.0,
    )
    restored = rustmc.FitResult.from_json(fit.to_json())
    np.testing.assert_array_equal(restored.log_likelihood()["obs"], fit.log_likelihood()["obs"])
