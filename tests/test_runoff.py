"""Native integer-event runoff: censoring, exact references, and coherent paths."""
import numpy as np
import pytest
import rustmc


def test_complete_conjugate_posterior_and_diagnostics():
    fit = rustmc.DirichletMultinomialRunoff([2, 3, 1]).fit(
        np.array([[8., 3., 1.], [4., 5., 3.]]), [0, 1], 5,
        totals=[12, 12], draws=4000, chains=2, seed=42,
    )
    p = fit.lag_probability_samples
    np.testing.assert_allclose(p.mean(axis=(0, 1)), np.array([14, 11, 5]) / 30, atol=.004)
    np.testing.assert_allclose(p.sum(axis=-1), 1)
    assert fit.sampler == "independent_conjugate"
    assert fit.allocation_samples.dtype == np.uint64
    assert fit.allocation_samples.shape == (2, 4000, 2, 3)
    assert fit.observed_mask.all()
    assert not fit.tail_samples.any()
    stats = fit.sampler_stats
    assert stats["divergences"] is None
    assert stats["acceptance_rate"] is None
    assert len(fit.diagnostics()) == 3
    assert "independent_conjugate" in fit.summary()


def test_future_calendar_and_tail_conserve_known_totals():
    model = rustmc.DirichletMultinomialRunoff([3, 2, 1])
    counts = np.array([[4., np.nan, np.nan], [np.nan, np.nan, np.nan]])
    fit = model.fit(counts, [0, 1], 0, totals=[9, 6], draws=80, chains=2, seed=19)
    repeat = model.fit(counts, [0, 1], 0, totals=[9, 6], draws=80, chains=2, seed=19)
    np.testing.assert_array_equal(fit.allocation_samples, repeat.allocation_samples)
    np.testing.assert_array_equal(fit.lag_probability_samples, repeat.lag_probability_samples)
    allocation = fit.allocation_samples
    np.testing.assert_array_equal(allocation.sum(axis=-1), fit.ultimate_samples)
    assert (allocation[:, :, 0, 0] == 4).all()
    calendar = fit.calendar_samples(2)
    np.testing.assert_array_equal(calendar[:, :, 0], allocation[:, :, 0, 1] + allocation[:, :, 1, 0])
    np.testing.assert_array_equal(calendar[:, :, 1], allocation[:, :, 1, 1])
    assert (calendar.sum(axis=-1) + fit.tail_samples.sum(axis=-1) + 4 == 15).all()
    assert not fit.calendar_samples(4)[:, :, 2:].any()
    np.testing.assert_array_equal(fit.observed_mask, np.isfinite(counts))
    assert np.isnan(fit.intensity_samples).all()


def test_unknown_ultimate_matches_independent_quadrature():
    # x0=3 observed, tail unknown. Integrate the marginal posterior for p0:
    # p0^(alpha0+x0-1) (1-p0)^(alpha1-1) / (rate+p0)^(shape+x0).
    nodes, weights = np.polynomial.legendre.leggauss(256)
    p = (nodes + 1) / 2
    weight = weights * p**4 * (1-p)**2 / (.2+p)**7
    weight /= weight.sum()
    expected_p = weight @ p
    expected_intensity = weight @ (7 / (.2+p))
    expected_remaining = weight @ (7 * (1-p) / (.2+p))
    fit = rustmc.DirichletMultinomialRunoff([2, 3], total_shape=4, total_rate=.2).fit(
        np.array([[3., np.nan]]), [0], 0, draws=6000, warmup=1200, chains=2, seed=61,
    )
    assert fit.sampler == "latent_count_gibbs"
    assert abs(fit.lag_probability_samples[:, :, 0].mean() - expected_p) < .018
    assert abs(fit.intensity_samples.mean() - expected_intensity) < .5
    assert abs(fit.tail_samples.mean() - expected_remaining) < .5
    assert (fit.ultimate_samples >= 3).all()
    np.testing.assert_array_equal(fit.ultimate_samples, fit.tail_samples + 3)
    assert {r["name"] for r in fit.diagnostics()} == {
        "lag_probability[0]", "lag_probability[1]", "intensity[0]", "ultimate[0]",
    }


def test_zero_observation_has_information_and_prior_future_cohort_is_usable():
    model = rustmc.DirichletMultinomialRunoff([2, 3], total_shape=4, total_rate=.2)
    fit = model.fit(np.array([[0., np.nan], [np.nan, np.nan]]), [0, 1], 0,
                    draws=3000, warmup=500, chains=2, seed=3)
    assert (fit.allocation_samples[:, :, 0, 0] == 0).all()
    assert fit.ultimate_samples[:, :, 0].mean() > 0
    # No observed events in the future cohort: its ultimate marginal is its prior.
    assert abs(fit.ultimate_samples[:, :, 1].mean() - 20) < .7
    assert fit.ultimate_samples[:, :, 0].mean() < 20


@pytest.mark.parametrize("counts, origins, valuation, totals", [
    ([[1., 0., np.nan]], [0], 0, [5]),  # future zero is observed data, not a mask
    ([[np.nan, np.nan, np.nan]], [0], 0, [5]),  # past missing not supported
    ([[2., np.nan, np.nan]], [0], 0, [1]),
    ([[1., 2., 3.]], [0], 3, [7]),  # closed total mismatch
    ([[1.5, np.nan, np.nan]], [0], 0, [5]),
    ([[-1., np.nan, np.nan]], [0], 0, [5]),
    ([[np.inf, np.nan, np.nan]], [0], 0, [5]),
    ([[1., np.nan, 0.]], [0], 0, [5]),  # tail closure before tail begins
    ([[1., np.nan, np.nan]], [], 0, [5]),
])
def test_invalid_triangle_is_rejected(counts, origins, valuation, totals):
    with pytest.raises(ValueError):
        rustmc.DirichletMultinomialRunoff([1, 1, 1]).fit(
            np.array(counts), origins, valuation, totals, draws=2, chains=1,
        )


def test_short_diagnostics_and_invalid_prior():
    fit = rustmc.DirichletMultinomialRunoff([1, 1]).fit(
        np.array([[0., np.nan]]), [0], 0, [2], draws=2, chains=1,
    )
    assert all(row["r_hat"] is None for row in fit.diagnostics())
    with pytest.raises(ValueError):
        fit.calendar_samples(0)
    for alpha in ([1], [1, 0], [1, np.inf]):
        with pytest.raises(ValueError):
            rustmc.DirichletMultinomialRunoff(alpha)


def test_large_sparse_totals_and_oversized_calendar_are_safe():
    fit = rustmc.DirichletMultinomialRunoff([1, 1]).fit(
        np.array([[0., np.nan], [np.nan, np.nan]]), [0, 1], 0,
        [10**10, 10**10], draws=10, chains=1,
    )
    assert (fit.ultimate_samples == 10**10).all()
    with pytest.raises(ValueError, match="allocation"):
        fit.calendar_samples(2**61)
    with pytest.raises(ValueError, match="allocation"):
        rustmc.DirichletMultinomialRunoff([1, 1]).fit(
            np.array([[0., np.nan]]), [0], 0, [1], draws=10**9,
        )


def test_rolling_valuation_holdout_on_known_total_cohorts():
    rng = np.random.default_rng(203)
    truth = np.array([.5, .3, .15, .05])
    origins = np.repeat(np.arange(8), 5)
    full = np.array([rng.multinomial(120, truth) for _ in origins], dtype=float)
    covered = []
    errors, uniform_errors = [], []
    for valuation in [4, 5, 6]:
        observed = origins[:, None] + np.arange(4) <= valuation
        counts = np.where(observed, full, np.nan)
        fit = rustmc.DirichletMultinomialRunoff([1, 1, 1, 1]).fit(
            counts, origins.tolist(), valuation, [120] * len(origins),
            draws=1200, chains=2, seed=valuation,
        )
        actual = sum(full[i, lag] for i in range(len(origins)) for lag in range(3)
                     if origins[i] + lag == valuation + 1)
        paths = fit.calendar_samples(1).reshape(-1)
        low, high = np.quantile(paths, [.025, .975])
        covered.append(low <= actual <= high)
        errors.append(abs(paths.mean() - actual))
        baseline = 0.
        for i, row in enumerate(counts):
            for lag in range(3):
                if origins[i] + lag == valuation + 1:
                    baseline += (120 - np.nansum(row)) / np.isnan(row).sum()
        uniform_errors.append(abs(baseline - actual))
    assert sum(covered) >= 2
    assert np.mean(errors) < np.mean(uniform_errors)
