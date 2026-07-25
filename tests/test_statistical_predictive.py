"""Prior-predictive and posterior-predictive calibration through the Python API.

``VALIDATION.md`` workstreams 1 (coverage) and 2 (calibration).

Tolerances
----------
* Prior-predictive moment checks compare against exact closed forms.  The
  Monte Carlo standard error of each estimate is computed from the *known*
  variance of the estimator (accounting for the fact that observations sharing
  a parameter draw are correlated), and the assertion allows 4 of them.
* Posterior-predictive coverage is averaged over independent replicates whose
  true parameters are drawn from the prior.  Under that design the nominal
  coverage is exact, so the tolerance is 4 empirical standard errors of the
  replicate-level coverage — measured from the run itself, not assumed.
"""

import numpy as np
import pytest

rustmc = pytest.importorskip("rustmc")

K = 4.0


def _fit(spec, seed, chains=2, draws=600, warmup=600):
    return rustmc.sample(
        spec,
        chains=chains,
        draws=draws,
        warmup=warmup,
        seed=seed,
        threads=1,
        show_progress=False,
    )


# ── Prior predictive ──────────────────────────────────────────────────────


def test_prior_predictive_parameter_draws_match_closed_form_moments():
    """``sample_prior_raw`` is a second, independent implementation of every
    prior family (it draws directly rather than sampling the log density).
    Its marginals must match the same closed forms the sampler reproduces.
    """
    n_samples = 40_000
    y = np.zeros(5)

    cases = [
        # (builder call, exact mean, exact sd)
        (lambda b: b.normal_prior("p", 1.5, 2.0), 1.5, 2.0),
        (
            lambda b: b.half_normal_prior("p", 1.3),
            1.3 * np.sqrt(2 / np.pi),
            1.3 * np.sqrt(1 - 2 / np.pi),
        ),
        (lambda b: b.exponential_prior("p", 2.0), 0.5, 0.5),
        (
            lambda b: b.log_normal_prior("p", -0.3, 0.5),
            np.exp(-0.3 + 0.125),
            np.sqrt((np.exp(0.25) - 1) * np.exp(-0.6 + 0.25)),
        ),
        (lambda b: b.gamma_prior("p", 3.0, 1.5), 3.0 / 1.5, np.sqrt(3.0) / 1.5),
        (
            lambda b: b.beta_prior("p", 2.0, 5.0),
            2 / 7,
            np.sqrt(2 * 5 / (49 * 8)),
        ),
        (lambda b: b.uniform_prior("p", -1.0, 3.0), 1.0, 4 / np.sqrt(12)),
        (
            lambda b: b.student_t_prior("p", 8.0, 0.4, 1.2),
            0.4,
            1.2 * np.sqrt(8 / 6),
        ),
    ]

    for make, exact_mean, exact_sd in cases:
        b = rustmc.ModelBuilder({"y": y})
        p = make(b)
        b.normal_likelihood("y_obs", p, 1.0, "y")
        out = rustmc.sample_prior_predictive(b.build(), n_samples=n_samples, seed=7)
        draws = np.asarray(out["p"])
        assert draws.shape == (n_samples,)

        se_mean = exact_sd / np.sqrt(n_samples)
        assert abs(draws.mean() - exact_mean) <= K * se_mean, (
            f"prior draws for {make}: mean {draws.mean():.5f} vs exact "
            f"{exact_mean:.5f}, tolerance {K} * {se_mean:.5f}"
        )
        # sd of a sample sd: sigma * sqrt((kurtosis - 1) / (4 n)); we do not
        # know the kurtosis for every family here, so bound it generously at
        # 12 (heavier than any family in this list except LogNormal(0.5),
        # whose kurtosis is 8.9).
        se_sd = exact_sd * np.sqrt((12.0 - 1.0) / (4.0 * n_samples))
        assert abs(draws.std(ddof=1) - exact_sd) <= K * se_sd, (
            f"prior draws for {make}: sd {draws.std(ddof=1):.5f} vs exact "
            f"{exact_sd:.5f}, tolerance {K} * {se_sd:.5f}"
        )


def test_prior_predictive_observations_match_the_marginal_distribution():
    """For ``mu ~ N(m0, s0)`` and ``y_i ~ N(mu, sigma)`` the prior predictive
    of ``y`` is exactly ``N(m0, sqrt(s0^2 + sigma^2))``."""
    m0, s0, sigma = 0.5, 1.5, 0.8
    n_obs, n_samples = 6, 20_000
    b = rustmc.ModelBuilder({"y": np.zeros(n_obs)})
    mu = b.normal_prior("mu", m0, s0)
    b.normal_likelihood("y_obs", mu, sigma, "y")
    out = rustmc.sample_prior_predictive(b.build(), n_samples=n_samples, seed=17)

    yy = np.asarray(out["y_obs"])
    assert yy.shape == (n_samples, n_obs)

    exact_sd = np.sqrt(s0**2 + sigma**2)
    # Observations within a draw share mu, so Var(overall mean) is
    # s0^2 / n_samples + sigma^2 / (n_samples * n_obs), not exact_sd^2 / N.
    se_mean = np.sqrt(s0**2 / n_samples + sigma**2 / (n_samples * n_obs))
    assert abs(yy.mean() - m0) <= K * se_mean

    # The per-column sd uses one observation per draw, so the draws are
    # independent and the Gaussian sd standard error applies.
    col_sd = yy[:, 0].std(ddof=1)
    se_sd = exact_sd / np.sqrt(2 * n_samples)
    assert abs(col_sd - exact_sd) <= K * se_sd, (
        f"prior predictive sd {col_sd:.5f} vs exact {exact_sd:.5f}"
    )


def test_prior_predictive_respects_the_likelihood_support():
    """Count and binary likelihoods must produce draws inside their support."""
    n_obs, n_samples = 5, 2_000

    b = rustmc.ModelBuilder({"y": np.zeros(n_obs)})
    eta = b.normal_prior("eta", 0.0, 1.0)
    b.bernoulli_logit_likelihood("y_obs", eta, "y")
    out = rustmc.sample_prior_predictive(b.build(), n_samples=n_samples, seed=3)
    vals = np.unique(np.asarray(out["y_obs"]))
    assert set(vals.tolist()) <= {0.0, 1.0}

    b = rustmc.ModelBuilder({"y": np.zeros(n_obs)})
    eta = b.normal_prior("eta", 0.0, 0.5)
    b.poisson_log_likelihood("y_obs", eta, "y")
    out = rustmc.sample_prior_predictive(b.build(), n_samples=n_samples, seed=3)
    yy = np.asarray(out["y_obs"])
    assert (yy >= 0).all() and np.allclose(yy, np.round(yy))

    b = rustmc.ModelBuilder({"y": np.ones(n_obs)})
    eta = b.normal_prior("eta", 0.0, 0.5)
    b.exponential_likelihood("y_obs", eta, "y")
    out = rustmc.sample_prior_predictive(b.build(), n_samples=n_samples, seed=3)
    assert (np.asarray(out["y_obs"]) > 0).all()


# ── Posterior predictive ──────────────────────────────────────────────────


@pytest.mark.parametrize("nominal", [0.5, 0.9])
def test_posterior_predictive_interval_coverage_matches_nominal(nominal):
    """Replicated coverage check.

    Each replicate draws the true coefficients from the model's own prior and
    simulates data, so the nominal coverage of a central posterior-predictive
    interval is *exactly* ``nominal``.  We average the per-replicate coverage
    and compare against ``nominal`` using the empirical standard error of the
    replicate coverages, so the tolerance is measured rather than assumed.
    """
    reps = 16
    n, sigma, prior_sd = 60, 0.8, 1.5
    lo_q, hi_q = (1 - nominal) / 2, 1 - (1 - nominal) / 2

    coverages = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        a_true, b_true = rng.normal(0, prior_sd, size=2)
        x = rng.normal(size=n)
        y = a_true + b_true * x + rng.normal(0, sigma, size=n)

        builder = rustmc.ModelBuilder({"x": x, "y": y})
        a = builder.normal_prior("a", 0.0, prior_sd)
        bb = builder.normal_prior("b", 0.0, prior_sd)
        builder.normal_likelihood("y_obs", a + bb * "x", sigma, "y")
        fit = _fit(builder.build(), 2000 + r, chains=2, draws=500, warmup=500)

        ppc = np.asarray(fit.posterior_predictive(seed=r)["y_obs"])
        assert ppc.shape[1] == n
        lo = np.quantile(ppc, lo_q, axis=0)
        hi = np.quantile(ppc, hi_q, axis=0)
        coverages.append(float(np.mean((y >= lo) & (y <= hi))))

    coverages = np.array(coverages)
    emp = coverages.mean()
    se = coverages.std(ddof=1) / np.sqrt(reps)
    assert abs(emp - nominal) <= K * se, (
        f"posterior-predictive {nominal:.0%} interval covered {emp:.1%} of "
        f"observations across {reps} replicates; |diff| {abs(emp - nominal):.4f} "
        f"> {K} * empirical se ({se:.4f}); per-replicate coverages {coverages}"
    )


def test_posterior_predictive_mean_and_spread_match_the_analytic_predictive():
    """For a known-sigma Gaussian linear model the posterior predictive of the
    i-th observation is exactly ``N(x_i' m, sigma^2 + x_i' C x_i)`` where
    ``(m, C)`` is the exact Gaussian posterior."""
    rng = np.random.default_rng(55)
    n, sigma, prior_sd = 70, 0.7, 2.0
    x = rng.normal(size=n)
    y = 0.4 + 0.9 * x + rng.normal(0, sigma, size=n)
    X = np.column_stack([np.ones(n), x])

    prior_prec = np.eye(2) / prior_sd**2
    lam = prior_prec + X.T @ X / sigma**2
    cov = np.linalg.inv(lam)
    mean = cov @ (X.T @ y / sigma**2)

    b = rustmc.ModelBuilder({"x": x, "y": y})
    a = b.normal_prior("a", 0.0, prior_sd)
    bb = b.normal_prior("b", 0.0, prior_sd)
    b.normal_likelihood("y_obs", a + bb * "x", sigma, "y")
    fit = _fit(b.build(), 303, chains=4, draws=1500, warmup=1000)

    ppc = np.asarray(fit.posterior_predictive(seed=1)["y_obs"])
    n_draws = ppc.shape[0]

    exact_mean = X @ mean
    exact_sd = np.sqrt(sigma**2 + np.einsum("ij,jk,ik->i", X, cov, X))

    emp_mean = ppc.mean(axis=0)
    emp_sd = ppc.std(axis=0, ddof=1)

    # Draws are autocorrelated; take an effective sample size of n_draws / 10
    # (conservative for a 2-parameter Gaussian target where the reported
    # ess_bulk is typically > n_draws / 2).
    ess = n_draws / 10.0
    se_mean = exact_sd / np.sqrt(ess)
    se_sd = exact_sd / np.sqrt(2 * ess)

    assert np.all(np.abs(emp_mean - exact_mean) <= K * se_mean), (
        "posterior predictive means deviate from the analytic predictive by "
        f"up to {np.max(np.abs(emp_mean - exact_mean) / se_mean):.2f} standard errors"
    )
    assert np.all(np.abs(emp_sd - exact_sd) <= K * se_sd), (
        "posterior predictive sds deviate from the analytic predictive by "
        f"up to {np.max(np.abs(emp_sd - exact_sd) / se_sd):.2f} standard errors"
    )


def test_posterior_predictive_is_reproducible_and_respects_n_samples():
    rng = np.random.default_rng(66)
    n = 30
    y = rng.normal(size=n)
    b = rustmc.ModelBuilder({"y": y})
    mu = b.normal_prior("mu", 0.0, 2.0)
    b.normal_likelihood("y_obs", mu, 1.0, "y")
    fit = _fit(b.build(), 404, chains=2, draws=200, warmup=200)

    first = np.asarray(fit.posterior_predictive(n_samples=50, seed=9)["y_obs"])
    second = np.asarray(fit.posterior_predictive(n_samples=50, seed=9)["y_obs"])
    np.testing.assert_array_equal(first, second)
    assert first.shape == (50, n)

    other = np.asarray(fit.posterior_predictive(n_samples=50, seed=10)["y_obs"])
    assert not np.array_equal(first, other), "the ppc seed has no effect"
