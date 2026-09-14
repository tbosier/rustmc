"""Prior simulation and pointwise likelihoods preserve the specified distributions."""
import math

import numpy as np
import pytest

import rustmc as r


@pytest.mark.parametrize("family,expected", [
    ("half_normal", math.sqrt(2/math.pi) * 1e-15),
    ("exponential", 1e-15),
    ("gamma", 2e-15),
])
def test_small_positive_prior_draws_are_not_floored(family, expected):
    m = r.ModelBuilder()
    if family == "half_normal":
        m.half_normal_prior("x", 1e-15)
    elif family == "exponential":
        m.exponential_prior("x", 1e15)
    else:
        m.gamma_prior("x", 2., 1e15)
    x = r.sample_prior_predictive(m.build(), n_samples=30000, seed=110)["x"]
    assert np.isfinite(x).all() and (x > 0).all()
    assert x.mean() == pytest.approx(expected, rel=.025)
    assert (x < 1e-12).all()


def test_gamma_and_beta_prior_draws_retain_near_zero_mass():
    m = r.ModelBuilder()
    m.gamma_prior("g", .01, 1.)
    m.beta_prior("p", .01, .01)
    samples = r.sample_prior_predictive(m.build(), n_samples=50000, seed=111)
    threshold = math.exp(-100.)
    # At this tiny threshold the remaining CDF series terms are negligible.
    expected_gamma = math.exp(-1. - math.lgamma(1.01))
    expected_beta = math.exp(-1. - math.log(.01) - 2*math.lgamma(.01) + math.lgamma(.02))
    assert np.mean(samples["g"] < threshold) == pytest.approx(expected_gamma, abs=.012)
    assert np.mean(samples["p"] < threshold) == pytest.approx(expected_beta, abs=.012)


@pytest.mark.parametrize("sigma", [1e-200, 1e-15, 1., 1e200])
def test_pointwise_normal_likelihood_keeps_actual_scale(sigma):
    m = r.ModelBuilder({"y": np.array([0., .5*sigma])})
    a = m.normal_prior("a", 0., 1.)
    m.normal_likelihood("obs", a*0, sigma, "y")
    fit = m.compile().sample({}, chains=1, draws=4, warmup=5, seed=112, show_progress=False)
    expected = -.5*np.log(2*np.pi) - np.log(sigma) - .5*np.array([0., .5])**2
    np.testing.assert_allclose(fit.log_likelihood()["obs"], np.tile(expected, (1, 4, 1)), atol=1e-12)


def test_lognormal_tiny_observations_are_not_clipped():
    y = np.array([1e-310, 2e-310])
    mu = math.log(1e-310)
    m = r.ModelBuilder({"y": y})
    a = m.normal_prior("a", 0., 1.)
    m.log_normal_likelihood("obs", a*0 + mu, 1., "y")
    compiled = m.compile()
    expected = -.5*np.log(2*np.pi) - np.log(y) - .5*(np.log(y)-mu)**2
    target, gradient = compiled.log_density({}, [0.])
    assert target == pytest.approx(expected.sum() - .5*np.log(2*np.pi), abs=1e-10)
    assert np.isfinite(gradient).all()
    fit = compiled.sample({}, chains=1, draws=4, warmup=5, seed=113, show_progress=False)
    np.testing.assert_allclose(fit.log_likelihood()["obs"], np.tile(expected, (1, 4, 1)), atol=1e-10)


def test_negative_binomial_pointwise_dispersion_is_not_clipped():
    alpha = 1e-15
    m = r.ModelBuilder({"y": np.array([0., 1.])})
    a = m.normal_prior("a", 0., 1.)
    m.negative_binomial_likelihood("obs", a*0, alpha, "y")
    fit = m.compile().sample({}, chains=1, draws=4, warmup=5, seed=114, show_progress=False)
    zero = -alpha*np.log1p(1/alpha)
    one = zero + np.log(alpha) - np.log1p(alpha)
    np.testing.assert_allclose(fit.log_likelihood()["obs"], np.tile([zero, one], (1, 4, 1)), atol=1e-12)


def test_prior_prediction_rejects_unrepresentable_output():
    m = r.ModelBuilder()
    m.log_normal_prior("x", 800., 1.)
    with pytest.raises(ValueError, match="representable"):
        r.sample_prior_predictive(m.build(), n_samples=2)
