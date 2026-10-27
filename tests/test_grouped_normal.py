"""Normal likelihoods on a grouped mean, ``mu + tau * z["group"]``, run as one
fused forward and one fused reverse loop in the graph evaluator. These tests
go through the real builder, compiler, evaluator and sampler, show that the
fused loops ran, and check them against oracles that know nothing of fusion:
finite differences of the log density, and the exact posterior of a model
whose scales are known, where the posterior is Gaussian in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rustmc(rustmc_module):
    return rustmc_module


def _passes(rustmc):
    from rustmc import _rustmc

    return _rustmc._fused_grouped_normal_passes()


def _data(rows=60, groups=4, seed=11):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, groups, size=rows)
    y = 0.4 + 0.8 * rng.normal(size=groups)[group] + rng.normal(scale=0.6, size=rows)
    return {"group": group.astype(float), "y": y}


def test_grouped_gradient_matches_finite_differences(rustmc):
    data = _data()
    builder = rustmc.ModelBuilder()
    mu = builder.normal_prior("mu", 0.0, 5.0)
    tau = builder.half_normal_prior("tau", 1.0)
    z = builder.vector_normal_prior("z", n=4, mu=0.0, sigma=1.0)
    sigma = builder.half_normal_prior("sigma", 1.0)
    builder.normal_likelihood("obs", mu + tau * z["group"], sigma, "y")
    compiled = builder.compile()

    rng = np.random.default_rng(3)
    before = _passes(rustmc)
    for _ in range(4):
        position = rng.normal(scale=0.7, size=len(compiled.param_names))
        _, analytic = compiled.log_density(data, position)
        numeric = np.empty_like(position)
        for k in range(len(position)):
            step = np.zeros_like(position)
            step[k] = 1e-6
            up, _ = compiled.log_density(data, position + step)
            down, _ = compiled.log_density(data, position - step)
            numeric[k] = (up - down) / 2e-6
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)
    after = _passes(rustmc)
    # Each log_density call builds and drops one evaluator that ran fused.
    assert after[0] - before[0] >= 4 * (1 + 2 * len(compiled.param_names))
    assert after[1] - before[1] >= 4


def test_known_scales_posterior_matches_the_exact_gaussian(rustmc):
    """With tau and sigma fixed, (mu, z) has a Gaussian posterior.

    y = mu + tau * z[group] + e, e ~ N(0, sigma^2), mu ~ N(0, 5^2),
    z ~ N(0, 1): the posterior precision is A'A / sigma^2 + diag(1/25, 1, ..., 1)
    with A the rows' design for (mu, z), and its mean solves that system.
    """
    groups, tau, sigma = 4, 0.8, 0.6
    data = _data(rows=80, groups=groups, seed=5)
    builder = rustmc.ModelBuilder()
    mu = builder.normal_prior("mu", 0.0, 5.0)
    z = builder.vector_normal_prior("z", n=groups, mu=0.0, sigma=1.0)
    builder.normal_likelihood("obs", mu + tau * z["group"], sigma, "y")
    before = _passes(rustmc)
    fit = builder.compile().sample(
        data, chains=4, warmup=500, draws=1500, seed=17, show_progress=False
    )
    after = _passes(rustmc)
    steps = fit.transition_diagnostics()["total_leapfrog_steps"]
    assert after[1] - before[1] >= steps

    rows = len(data["y"])
    design = np.zeros((rows, 1 + groups))
    design[:, 0] = 1.0
    design[np.arange(rows), 1 + data["group"].astype(int)] = tau
    precision = design.T @ design / sigma**2 + np.diag([1 / 25] + [1.0] * groups)
    covariance = np.linalg.inv(precision)
    mean = covariance @ (design.T @ data["y"] / sigma**2)

    samples = fit.get_samples_2d()
    draws = np.column_stack(
        [np.ravel(samples["mu"])] + [np.ravel(samples[f"z[{g}]"]) for g in range(groups)]
    )
    idata = fit.to_arviz()
    import arviz as az

    ess = az.ess(idata, method="mean")
    effective = np.array([float(ess["mu"])] + [float(ess[f"z[{g}]"]) for g in range(groups)])
    assert effective.min() > 400
    sd = np.sqrt(np.diag(covariance))
    z_scores = (draws.mean(axis=0) - mean) / (sd / np.sqrt(effective))
    # Five means: a family-wise two-sided 0.001 bound.
    assert np.abs(z_scores).max() < 4.0, z_scores
    np.testing.assert_allclose(draws.std(axis=0), sd, rtol=0.1)
    correlation = np.corrcoef(draws.T)
    exact = covariance / np.outer(sd, sd)
    np.testing.assert_allclose(correlation, exact, atol=0.08)
