"""Normal likelihoods on a grouped mean, ``mu + tau * z["group"]``, which the
graph evaluator runs as one pass over per-group statistics of the data. These
tests go through the real builder, compiler, evaluator and sampler, show that
the fused pass ran, and check it against oracles that know nothing of fusion:
finite differences of the log density, the exact posterior of a model whose
scales are known, where the posterior is Gaussian in closed form, and every
output that reads node values computed again with fusion turned off.
"""

from __future__ import annotations

from statistics import NormalDist

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


@pytest.mark.parametrize("rows,groups", [(80, 4), (7, 10)])
def test_known_scales_posterior_matches_the_exact_gaussian(rustmc, rows, groups):
    """With tau and sigma fixed, (mu, z) has a Gaussian posterior.

    y = mu + tau * z[group] + e, e ~ N(0, sigma^2), mu ~ N(0, 5^2),
    z ~ N(0, 1): the posterior precision is A'A / sigma^2 + diag(1/25, 1, ..., 1)
    with A the rows' design for (mu, z), and its mean solves that system. The
    second case has more groups than rows, so at least three groups have no
    rows and keep their prior.
    """
    tau, sigma = 0.8, 0.6
    data = _data(rows=rows, groups=groups, seed=5)
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
    names = ["mu"] + [f"z[{g}]" for g in range(groups)]
    diagnostics = {row["name"]: row for row in fit.diagnostics()}
    # Bulk ESS, from rank-normalized draws, gates that each chain set is long
    # enough. The z-scores need the standard error of the plain mean of the
    # draws, sd / sqrt(ESS) with the ESS of the draws themselves rather than of
    # their ranks; rustmc reports exactly that as mcse_mean.
    assert min(diagnostics[name]["ess_bulk"] for name in names) > 400
    mcse = np.array([diagnostics[name]["mcse_mean"] for name in names])
    assert np.isfinite(mcse).all() and (mcse > 0).all()
    sd = np.sqrt(np.diag(covariance))
    z_scores = (draws.mean(axis=0) - mean) / mcse
    # A family-wise two-sided 0.001 bound over the means.
    bound = NormalDist().inv_cdf(1 - 0.0005 / len(names))
    assert np.abs(z_scores).max() < bound, z_scores
    np.testing.assert_allclose(draws.std(axis=0), sd, rtol=0.1)
    correlation = np.corrcoef(draws.T)
    exact = covariance / np.outer(sd, sd)
    np.testing.assert_allclose(correlation, exact, atol=0.08)


def _grouped_builder(rustmc, groups):
    builder = rustmc.ModelBuilder()
    mu = builder.normal_prior("mu", 0.0, 5.0)
    tau = builder.half_normal_prior("tau", 1.0)
    z = builder.vector_normal_prior("z", n=groups, mu=0.0, sigma=1.0)
    sigma = builder.half_normal_prior("sigma", 1.0)
    mean = mu + tau * z["group"]
    builder.normal_likelihood("obs", mean, sigma, "y")
    builder.deterministic("eta", mean)
    return builder


def _readers(rustmc, fit, model, data, future):
    """Every output of a fit or model that reads node values, by name."""
    out = {}
    for expected in (False, True):
        out[f"predict {expected}"] = fit.predict(seed=5, expected=expected)["obs"]
        out[f"predict new {expected}"] = fit.predict(future, seed=6, expected=expected)["obs"]
        out[f"posterior predictive {expected}"] = fit.posterior_predictive(
            n_samples=17, seed=7, expected=expected
        )["obs"]
    out["log likelihood"] = fit.log_likelihood()["obs"]
    out["deterministics"] = fit.deterministics()["eta"]
    out["deterministics new"] = fit.deterministics(future)["eta"]
    prior = rustmc.sample_prior_predictive(model, data, n_samples=13, seed=8)
    out["prior predictive"] = prior["obs"]
    out["prior deterministics"] = prior["eta"]
    return out


def test_every_reader_matches_the_unfused_path(rustmc):
    """The fused pass does not write the rows' means; outputs that read them
    get them computed on demand. With fusion turned off, the same fit gives
    the same outputs to the bit, and the deterministic equals mu + tau * z
    computed from the draws."""
    from rustmc import _rustmc

    groups = 6
    data = _data(rows=50, groups=groups - 1, seed=9)
    future = {"group": np.array([5.0, 0.0, 3.0, 3.0])}
    builder = _grouped_builder(rustmc, groups)
    model = builder.build()
    fit = builder.compile().sample(
        data, chains=2, warmup=100, draws=40, seed=4, show_progress=False
    )
    assert _rustmc._set_grouped_normal_fusion(True)
    try:
        before = _passes(rustmc)
        fused = _readers(rustmc, fit, model, data, future)
        middle = _passes(rustmc)
        assert _rustmc._set_grouped_normal_fusion(False)
        separate = _readers(rustmc, fit, model, data, future)
        after = _passes(rustmc)
    finally:
        _rustmc._set_grouped_normal_fusion(True)
    # One fused forward pass per draw at least, for the log-likelihood alone.
    assert middle[0] - before[0] >= 80
    assert after == middle
    assert fused.keys() == separate.keys()
    for name in fused:
        assert fused[name].size > 0, name
        np.testing.assert_array_equal(fused[name], separate[name], err_msg=name)

    samples = fit.get_samples_2d()
    z = np.stack([samples[f"z[{g}]"] for g in range(groups)], axis=-1)
    index = data["group"].astype(int)
    expected = samples["mu"][..., None] + samples["tau"][..., None] * z[..., index]
    np.testing.assert_allclose(fused["deterministics"], expected, rtol=1e-14, atol=1e-14)


def test_seeded_draws_repeat_and_do_not_depend_on_threads(rustmc):
    data = _data(rows=200, groups=8, seed=12)
    compiled = _grouped_builder(rustmc, 8).compile()

    def draws(threads):
        fit = compiled.sample(
            data, chains=4, warmup=150, draws=100, seed=21, threads=threads, show_progress=False
        )
        return fit.get_samples()

    first = draws(1)
    for name, other in (("again", draws(1)), ("four threads", draws(4))):
        assert first.keys() == other.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], other[key], err_msg=f"{name}: {key}")
