"""Analytic-posterior tests driven through the public Python DSL.

The Rust suite (``rust_core/tests/analytic_posterior.rs``) proves the *engine*
targets the right posterior. This file proves the *bindings* build the model
the user described: a mis-wired prior, a dropped intercept, or a transposed
design matrix would leave the Rust tests green and these red.

Tolerances
----------
Every assertion compares against a closed-form value and is sized from the
reported Monte Carlo standard error:

* posterior mean: ``|mean_hat - exact| <= 4 * mcse_mean``.  Under a normal CLT
  approximation a single such assertion fires spuriously with probability
  ~6e-5.
* posterior sd: ``|sd_hat - exact| <= 5 * exact / sqrt(2 * ess_bulk)``, from
  the delta-method standard error of a sample sd for a Gaussian target.

Nothing here was widened after the fact; the constants 4 and 5 are fixed in
``K_MEAN``/``K_SD`` and reused everywhere.
"""

import numpy as np
import pytest

rustmc = pytest.importorskip("rustmc")

K_MEAN = 4.0
K_SD = 5.0

CHAINS = 4
DRAWS = 1500
WARMUP = 1000


def _diag(fit, name):
    for d in fit.diagnostics():
        if d["name"] == name:
            return d
    raise AssertionError(f"no diagnostic named {name!r}; have {[d['name'] for d in fit.diagnostics()]}")


def assert_mean(fit, name, exact, min_ess=400.0):
    d = _diag(fit, name)
    assert d["ess_bulk"] >= min_ess, (
        f"{name}: ess_bulk {d['ess_bulk']:.0f} < {min_ess}; an MCSE-based "
        "tolerance would be meaningless"
    )
    tol = K_MEAN * d["mcse_mean"]
    assert abs(d["mean"] - exact) <= tol, (
        f"{name}: posterior mean {d['mean']:.6f} vs exact {exact:.6f} "
        f"(|diff| {abs(d['mean'] - exact):.6f} > {K_MEAN} * mcse {d['mcse_mean']:.6f} "
        f"= {tol:.6f}, ess_bulk {d['ess_bulk']:.0f})"
    )


def assert_sd(fit, name, exact, min_ess=400.0):
    d = _diag(fit, name)
    assert d["ess_bulk"] >= min_ess
    rel = K_SD / np.sqrt(2.0 * d["ess_bulk"])
    tol = rel * abs(exact)
    assert abs(d["std"] - exact) <= tol, (
        f"{name}: posterior sd {d['std']:.6f} vs exact {exact:.6f} "
        f"({100 * abs(d['std'] - exact) / exact:.2f}% relative) exceeds "
        f"{100 * rel:.2f}% tolerance (ess_bulk {d['ess_bulk']:.0f})"
    )


def assert_rhat(fit, limit=1.05):
    for d in fit.diagnostics():
        assert np.isfinite(d["r_hat"]) and d["r_hat"] < limit, (
            f"{d['name']}: r_hat {d['r_hat']}"
        )


def linear_gaussian_posterior(X, y, sigma, prior_mean, prior_sd):
    """Exact posterior of ``y ~ N(X beta, sigma^2 I)`` with independent
    ``beta_k ~ N(prior_mean[k], prior_sd[k])``.  Returns (mean, cov)."""
    X = np.asarray(X, dtype=float)
    prior_prec = np.diag(1.0 / np.asarray(prior_sd, dtype=float) ** 2)
    lam = prior_prec + X.T @ X / sigma**2
    h = prior_prec @ np.asarray(prior_mean, dtype=float) + X.T @ np.asarray(y) / sigma**2
    cov = np.linalg.inv(lam)
    return cov @ h, cov


def _fit(spec, seed, chains=CHAINS, draws=DRAWS, warmup=WARMUP):
    return rustmc.sample(
        spec,
        chains=chains,
        draws=draws,
        warmup=warmup,
        seed=seed,
        threads=1,
        show_progress=False,
    )


# ── Conjugate Normal-Normal ───────────────────────────────────────────────


def test_conjugate_normal_normal_matches_analytic_posterior():
    rng = np.random.default_rng(11)
    n, sigma, m0, s0 = 50, 0.9, 0.0, 2.5
    y = rng.normal(1.4, sigma, size=n)

    prec = 1.0 / s0**2 + n / sigma**2
    exact_mean = (m0 / s0**2 + y.sum() / sigma**2) / prec
    exact_sd = np.sqrt(1.0 / prec)

    b = rustmc.ModelBuilder({"y": y})
    mu = b.normal_prior("mu", m0, s0)
    b.normal_likelihood("y_obs", mu, sigma, "y")
    fit = _fit(b.build(), 101)

    assert_rhat(fit)
    assert_mean(fit, "mu", exact_mean)
    assert_sd(fit, "mu", exact_sd)


# ── Linear regression through ``param * "x"`` ─────────────────────────────


def test_linear_regression_matches_analytic_posterior():
    rng = np.random.default_rng(12)
    n, sigma = 80, 0.7
    x = rng.normal(size=n)
    y = 0.8 - 1.3 * x + rng.normal(0, sigma, size=n)

    X = np.column_stack([np.ones(n), x])
    exact_mean, exact_cov = linear_gaussian_posterior(X, y, sigma, [0.0, 0.0], [3.0, 3.0])

    b = rustmc.ModelBuilder({"x": x, "y": y})
    a = b.normal_prior("a", 0.0, 3.0)
    bb = b.normal_prior("b", 0.0, 3.0)
    b.normal_likelihood("y_obs", a + bb * "x", sigma, "y")
    fit = _fit(b.build(), 102)

    assert_rhat(fit)
    for k, name in enumerate(["a", "b"]):
        assert_mean(fit, name, exact_mean[k])
        assert_sd(fit, name, np.sqrt(exact_cov[k, k]))


# ── Matrix-vector regression through the ``@`` operator ───────────────────


def test_matvec_regression_matches_analytic_posterior():
    rng = np.random.default_rng(13)
    n, p, sigma, prior_sd = 100, 4, 0.6, 2.0
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    beta_true = np.array([0.5, -1.0, 0.3, 0.7])
    y = X @ beta_true + rng.normal(0, sigma, size=n)

    exact_mean, exact_cov = linear_gaussian_posterior(
        X, y, sigma, np.zeros(p), np.full(p, prior_sd)
    )

    b = rustmc.ModelBuilder({"X": X, "y": y})
    beta = b.vector_normal_prior("beta", p, 0.0, prior_sd)
    b.normal_likelihood("y_obs", beta @ "X", sigma, "y")
    fit = _fit(b.build(), 103)

    assert_rhat(fit)
    for k in range(p):
        assert_mean(fit, f"beta[{k}]", exact_mean[k])
        assert_sd(fit, f"beta[{k}]", np.sqrt(exact_cov[k, k]))


def test_matvec_auto_promotion_matches_explicit_vector_prior():
    """``normal_prior`` + ``@`` must build the same model as
    ``vector_normal_prior`` + ``@``.  Both are checked against the same exact
    posterior, so a discrepancy localises to the auto-promotion path."""
    rng = np.random.default_rng(14)
    n, p, sigma, prior_sd = 90, 3, 0.5, 1.5
    X = rng.normal(size=(n, p))
    y = X @ np.array([0.4, -0.8, 0.2]) + rng.normal(0, sigma, size=n)
    exact_mean, exact_cov = linear_gaussian_posterior(
        X, y, sigma, np.zeros(p), np.full(p, prior_sd)
    )

    b = rustmc.ModelBuilder({"X": X, "y": y})
    beta = b.normal_prior("beta", 0.0, prior_sd)  # scalar prior, auto-promoted
    b.normal_likelihood("y_obs", beta @ "X", sigma, "y")
    fit = _fit(b.build(), 104)

    assert_rhat(fit)
    for k in range(p):
        assert_mean(fit, f"beta[{k}]", exact_mean[k])
        assert_sd(fit, f"beta[{k}]", np.sqrt(exact_cov[k, k]))


# ── Hierarchical model with a ParamRef hyperparameter ─────────────────────


@pytest.mark.xfail(
    reason=(
        "DEFECT 1 (see tests/test_statistical_engine_bugs.py): a prior with a "
        "ParamRef hyperparameter is auto-non-centered into '<name>__raw' and "
        "its derived value node is never registered under the user-facing "
        "name, so referencing it in a likelihood raises "
        "ValueError('Unknown param: ...'). No hierarchical model can be fitted "
        "through the Python DSL today. This test is the analytic reference the "
        "fix should be validated against."
    ),
    raises=ValueError,
    strict=False,
)
def test_hierarchical_known_variances_matches_analytic_posterior():
    """Two-level linear-Gaussian model with *fixed* variance components, so
    the joint posterior over (mu, theta_1..theta_J) is exactly Gaussian.

    ``normal_prior(name, mu=ParamRef, sigma=const)`` triggers the automatic
    non-centering path in the bindings; the resulting posterior must be the
    same distribution regardless of parameterisation.
    """
    J = 5
    s0, tau = 4.0, 1.1
    sigma = np.array([1.0, 1.3, 0.9, 1.6, 1.1])
    y = np.array([1.5, -0.6, 2.2, 0.4, 1.0])

    # Joint precision over [mu, theta_0..theta_{J-1}].
    dim = J + 1
    lam = np.zeros((dim, dim))
    h = np.zeros(dim)
    lam[0, 0] = 1.0 / s0**2 + J / tau**2
    for g in range(J):
        lam[0, g + 1] = lam[g + 1, 0] = -1.0 / tau**2
        lam[g + 1, g + 1] = 1.0 / tau**2 + 1.0 / sigma[g] ** 2
        h[g + 1] = y[g] / sigma[g] ** 2
    cov = np.linalg.inv(lam)
    mean = cov @ h

    # The DSL exposes one scalar `sigma` per likelihood, but the groups have
    # different observation sds. Dividing both the response and the one-hot
    # indicator column by sigma_g rescales every row to unit sd, which leaves
    # the posterior unchanged.
    scaled = {"y": y / sigma}
    for g in range(J):
        col = np.zeros(J)
        col[g] = 1.0 / sigma[g]
        scaled[f"g{g}"] = col

    b = rustmc.ModelBuilder(scaled)
    mu = b.normal_prior("mu", 0.0, s0)
    thetas = [b.normal_prior(f"theta_{g}", mu, tau) for g in range(J)]
    expr = thetas[0] * "g0"
    for g in range(1, J):
        expr = expr + thetas[g] * f"g{g}"
    b.normal_likelihood("y_obs", expr, 1.0, "y")
    fit = _fit(b.build(), 105)

    assert_rhat(fit)
    assert_mean(fit, "mu", mean[0])
    assert_sd(fit, "mu", np.sqrt(cov[0, 0]))
    for g in range(J):
        assert_mean(fit, f"theta_{g}", mean[g + 1])
        assert_sd(fit, f"theta_{g}", np.sqrt(cov[g + 1, g + 1]))


def test_sampling_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(15)
    y = rng.normal(size=40)
    b = rustmc.ModelBuilder({"y": y})
    mu = b.normal_prior("mu", 0.0, 2.0)
    b.normal_likelihood("y_obs", mu, 1.0, "y")
    spec = b.build()
    first = _fit(spec, 999, chains=2, draws=100, warmup=100)
    second = _fit(spec, 999, chains=2, draws=100, warmup=100)
    np.testing.assert_array_equal(
        first.get_samples()["mu"], second.get_samples()["mu"]
    )
