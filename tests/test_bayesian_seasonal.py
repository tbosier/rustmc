"""Bayesian structural seasonal local-level inference and forecasting."""

import math

import numpy as np
import pytest


def make_model(rmc, level=(3.0, 0.16), seasonal=(3.0, 0.08), observation=(3.0, 0.36)):
    return rmc.BayesianSeasonalLocalLevel(
        period=4,
        level_variance_prior=rmc.InverseGammaPrior(*level),
        seasonal_variance_prior=rmc.InverseGammaPrior(*seasonal),
        observation_variance_prior=rmc.InverseGammaPrior(*observation),
        initial_level=5.0,
        initial_seasonal_effects=[1.0, -0.5, -0.25, -0.25],
        initial_level_variance=4.0,
        initial_seasonal_variance=2.0,
    )


def seasonal_observations(
    seed=123,
    count=40,
    level_variance=0.04,
    seasonal_variance=0.02,
    observation_variance=0.09,
):
    rng = np.random.default_rng(seed)
    state = np.array([5.0, -0.25, -0.25, -0.5])
    values = []
    for _ in range(count):
        next_level = state[0] + rng.normal(0.0, np.sqrt(level_variance))
        next_seasonal = -state[1:].sum() + rng.normal(0.0, np.sqrt(seasonal_variance))
        state[2:] = state[1:-1]
        state[0] = next_level
        state[1] = next_seasonal
        values.append(next_level + next_seasonal + rng.normal(0.0, np.sqrt(observation_variance)))
    return np.asarray(values)


def test_seeded_fit_infers_all_variances_and_handles_missing(rustmc_module):
    observations = seasonal_observations()
    observations[[7, 22]] = np.nan
    model = make_model(rustmc_module)
    first = model.fit(observations, chains=2, draws=60, warmup=40, seed=17)
    second = model.fit(observations, chains=2, draws=60, warmup=40, seed=17)

    assert model.period == 4
    np.testing.assert_allclose(model.initial_seasonal_effects.sum(), 0.0)
    assert first.period == 4
    assert first.chains == 2
    assert first.draws == 60
    assert first.time_count == 40
    assert first.observed_count == 38
    samples = first.get_samples_2d()
    repeated = second.get_samples_2d()
    assert set(samples) == {
        "level_variance",
        "seasonal_variance",
        "observation_variance",
        "level_sd",
        "seasonal_sd",
        "observation_sd",
        "terminal_level",
        "terminal_seasonal",
    }
    for name, values in samples.items():
        assert values.shape == (2, 60)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, repeated[name])
    for component in ("level", "seasonal", "observation"):
        assert np.all(samples[f"{component}_variance"] > 0.0)
        assert np.std(samples[f"{component}_variance"]) > 0.0
        np.testing.assert_allclose(
            samples[f"{component}_sd"] ** 2,
            samples[f"{component}_variance"],
        )


def test_forecast_paths_and_cumulative_intervals_are_drawwise(rustmc_module):
    fit = make_model(rustmc_module).fit(
        seasonal_observations(), chains=2, draws=80, warmup=50, seed=19
    )
    forecast = fit.forecast(steps=8, seed=23)
    repeated = fit.forecast(steps=8, seed=23)

    assert forecast.chains == 2
    assert forecast.draws == 80
    assert forecast.steps == 8
    for name in ("level", "seasonal", "observation"):
        paths = getattr(forecast, f"{name}_samples")
        assert paths.shape == (2, 80, 8)
        np.testing.assert_array_equal(paths, getattr(repeated, f"{name}_samples"))
        np.testing.assert_allclose(
            getattr(forecast, f"{name}_mean"), paths.mean(axis=(0, 1))
        )
    cumulative = forecast.cumulative_observation_samples
    np.testing.assert_array_equal(cumulative, np.cumsum(forecast.observation_samples, axis=2))
    np.testing.assert_allclose(
        forecast.cumulative_observation_mean, cumulative.mean(axis=(0, 1))
    )

    lower, upper = forecast.interval()
    cumulative_lower, cumulative_upper = forecast.cumulative_interval()
    np.testing.assert_allclose(
        lower, np.quantile(forecast.observation_samples, 0.025, axis=(0, 1))
    )
    np.testing.assert_allclose(
        upper, np.quantile(forecast.observation_samples, 0.975, axis=(0, 1))
    )
    np.testing.assert_allclose(
        cumulative_lower, np.quantile(cumulative, 0.025, axis=(0, 1))
    )
    np.testing.assert_allclose(
        cumulative_upper, np.quantile(cumulative, 0.975, axis=(0, 1))
    )
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


#: Variances the recovery test generates from, and the single deliberately wrong prior
#: it fits all three with. The earlier version of this test used the model's own
#: priors, InverseGamma(3, 0.16 / 0.08 / 0.36), against truths 0.04 / 0.02 / 0.09 and
#: asserted only ``truth / 2 < median < truth * 2``. An InverseGamma(3, b) has median
#: b / 2.67406, so those priors' medians are 0.0598, 0.0299 and 0.1346 with no data
#: involved at all, against windows of (0.02, 0.08), (0.01, 0.04) and (0.045, 0.18).
#: Every one of them is inside its window, so the test passed against a sampler that
#: discarded the observations and returned prior draws. It tested nothing.
RECOVERY_TRUTHS = {"level_variance": 0.50, "seasonal_variance": 0.20,
                   "observation_variance": 1.00}
#: InverseGamma(3, 0.08): mean 0.04, median 0.0299. That is 17x, 7x and 33x below the
#: three truths, so the likelihood has to do all the work, and a prior-only sampler
#: lands nowhere near the window below.
RECOVERY_PRIOR = (3.0, 0.08)
#: Relative half-width of the accepted window. Across ten generating seeds at this
#: series length the worst observed deviation of the posterior median from the truth
#: was 0.188, so this leaves real headroom; the prior-only medians sit 0.85 to 0.97
#: away, so the window excludes them by a factor of three or more.
RECOVERY_TOLERANCE = 0.30
#: Long enough that the conjugate update's data term (count/2 added to the shape)
#: outweighs the prior shape of 3 by a factor of 250.
RECOVERY_COUNT = 1500


def inverse_gamma_upper_tail(shape, scale, threshold):
    """P(X > threshold) for X ~ InverseGamma(shape=3, scale), in closed form.

    ``X > t`` exactly when ``Gamma(3, 1) < scale / t``, and the Gamma(3, 1) CDF is
    ``1 - e**-g * (1 + g + g**2 / 2)``. Closed form rather than sampled so the negative
    control below is exact and carries no Monte Carlo noise of its own.
    """
    assert shape == 3.0, "closed form is specialised to shape 3"
    g = scale / threshold
    return 1.0 - math.exp(-g) * (1.0 + g + g * g / 2.0)


def test_the_recovery_prior_cannot_by_itself_reach_the_window_recovery_asserts():
    """Negative control for the test below, so the window can never be widened onto it.

    Under InverseGamma(3, 0.08) alone the probability of a single draw even reaching
    the bottom of each accepted window is 0.0017, 0.0204 and 0.0002. The median of 600
    such draws therefore lands inside a window only if a majority clear a threshold at
    most one draw in forty-nine reaches, which is not something that happens by chance.
    Whatever the recovery test below observes, the observations put it there.
    """
    shape, scale = RECOVERY_PRIOR
    tails = {}
    for name, truth in RECOVERY_TRUTHS.items():
        lower = truth * (1.0 - RECOVERY_TOLERANCE)
        tails[name] = inverse_gamma_upper_tail(shape, scale, lower)
        assert tails[name] < 0.05, (name, tails[name])
        # The prior mean is outside the window on the same side, by a wide margin.
        assert scale / (shape - 1.0) < lower
    assert tails["level_variance"] == pytest.approx(0.0017, abs=5e-4)
    assert tails["seasonal_variance"] == pytest.approx(0.0204, abs=5e-4)
    assert tails["observation_variance"] == pytest.approx(0.0002, abs=5e-4)


def test_synthetic_variance_recovery_overrides_a_deliberately_wrong_prior(rustmc_module):
    """The posterior median must be pulled from the prior onto the generating variance.

    Prior-only medians would be 0.0299 for all three parameters against truths of
    0.50, 0.20 and 1.00 - ratios of 0.06, 0.15 and 0.03, every one of them far outside
    the +/-30% window asserted here. See the negative control above for the exact
    prior tail masses.
    """
    model = make_model(
        rustmc_module,
        level=RECOVERY_PRIOR,
        seasonal=RECOVERY_PRIOR,
        observation=RECOVERY_PRIOR,
    )
    fit = model.fit(
        seasonal_observations(seed=321, count=RECOVERY_COUNT, **RECOVERY_TRUTHS),
        chains=2,
        draws=300,
        warmup=200,
        seed=5,
    )
    samples = fit.get_samples_2d()
    for name, truth in RECOVERY_TRUTHS.items():
        posterior_median = float(np.median(samples[name]))
        assert posterior_median == pytest.approx(truth, rel=RECOVERY_TOLERANCE), (
            f"{name}: {posterior_median} is not within "
            f"{RECOVERY_TOLERANCE:.0%} of {truth}"
        )


def test_validation_rejects_insufficient_or_invalid_seasonal_fits(rustmc_module):
    rmc = rustmc_module
    with pytest.raises(ValueError, match="sum to zero"):
        rmc.BayesianSeasonalLocalLevel(
            4,
            rmc.InverseGammaPrior(3.0, 0.16),
            rmc.InverseGammaPrior(3.0, 0.08),
            rmc.InverseGammaPrior(3.0, 0.36),
            initial_seasonal_effects=[1.0, 0.0, 0.0, 0.0],
        )
    model = make_model(rmc)
    for observations in (
        np.zeros(1),
        # Three variances need three finite observations; two used to pass.
        np.array([0.0, np.nan, 0.0, np.nan]),
        np.array([0.0] + [np.nan] * 3),
        np.array([0.0] * 7 + [np.inf]),
    ):
        with pytest.raises(ValueError):
            model.fit(observations, chains=1, draws=2, warmup=1)
    fit = model.fit(np.zeros(8), chains=1, draws=4, warmup=2)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    forecast = fit.forecast(2)
    for level in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(level)
        with pytest.raises(ValueError, match="strictly between"):
            forecast.cumulative_interval(level)
