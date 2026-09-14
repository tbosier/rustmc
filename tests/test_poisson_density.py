"""Fitted Poisson diagnostics preserve local likelihood shape at large rates."""

import numpy as np
import pytest

import rustmc


@pytest.mark.parametrize("count", [1e14, 1e15, 8e15])
def test_poisson_target_pointwise_likelihood_and_fit_replay_at_large_counts(count):
    builder = rustmc.ModelBuilder({"y": np.array([count])})
    z = builder.normal_prior("z", 0.0, 10.0)
    eta = np.log(count) + z / np.sqrt(count)
    builder.poisson_log_likelihood("obs", eta, "y")
    model = builder.compile()
    mode_logp = -0.5 * np.log(2 * np.pi * count)
    prior_normalizer = -0.5 * np.log(2 * np.pi) - np.log(10.0)
    for value in [-1.0, 0.0, 1.0]:
        lp, grad = model.log_density({}, [value])
        expected = mode_logp + prior_normalizer - 0.5 * 1.01 * value**2
        assert lp == pytest.approx(expected, abs=3e-6)
        assert grad[0] == pytest.approx(-1.01 * value, abs=3e-6)
    fit = model.sample({}, chains=1, draws=20, warmup=30, seed=72, show_progress=False)
    z_draws = fit.get_samples_2d()["z"]
    pointwise = fit.log_likelihood()["obs"][..., 0]
    np.testing.assert_allclose(pointwise, mode_logp - 0.5 * z_draws**2, atol=3e-5, rtol=0.0)
    restored = rustmc.FitResult.from_json(fit.to_json())
    np.testing.assert_array_equal(restored.log_likelihood()["obs"], fit.log_likelihood()["obs"])
