"""The ArviZ export must preserve posterior-predictive (chain, draw) identity.

A predictive draw is only interpretable next to the parameter draw that
produced it: LOO/PSIS reweighting and every per-chain predictive diagnostic
pair the two through the exported `(chain, draw)` coordinates. Collapsing the
predictive group onto a single fabricated chain destroys that pairing, and a
subsample that does not record which draws it kept destroys it irrecoverably.
"""

import numpy as np
import pytest

import rustmc

arviz = pytest.importorskip("arviz")


def fit_multichain(chains=4, draws=25, seed=11):
    builder = rustmc.ModelBuilder()
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("obs", beta * "x", 1.0, "y")
    compiled = builder.compile()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=8)
    data = {"x": x, "y": 0.7 * x + rng.normal(size=8)}
    fit = compiled.sample(
        data,
        chains=chains,
        draws=draws,
        warmup=30,
        seed=seed,
        show_progress=False,
    )
    return fit, data


def group(idata, name):
    node = getattr(idata, name)
    return getattr(node, "dataset", node)


def test_posterior_predictive_keeps_posterior_chain_draw_axes():
    fit, data = fit_multichain(chains=4, draws=25)
    idata = fit.to_arviz(include_ppc=True)

    posterior = group(idata, "posterior")
    ppc = group(idata, "posterior_predictive")

    # Before the fix this was (1, 100, 8): all four chains flattened into one.
    assert ppc["obs"].shape == (4, 25, len(data["y"]))
    assert ppc["obs"].dims[:2] == posterior["beta"].dims[:2] == ("chain", "draw")
    np.testing.assert_array_equal(ppc.chain.values, posterior.chain.values)
    np.testing.assert_array_equal(ppc.draw.values, posterior.draw.values)

    # log_likelihood is the other group LOO reads; it must agree too.
    log_likelihood = group(idata, "log_likelihood")
    assert log_likelihood["obs"].shape == (4, 25, len(data["y"]))
    np.testing.assert_array_equal(
        log_likelihood.draw.values, posterior.draw.values
    )


def test_predictive_draw_is_generated_from_the_colocated_posterior_draw():
    """The (chain, draw) label must be the truth, not a relabelling.

    Checked exactly rather than statistically. A sampled predictive draw is
    parameter plus noise, so a residual test cannot tell a correct pairing from
    one shifted by a draw -- adjacent MCMC draws are too similar. The predictive
    *mean* is a deterministic function of the parameter draw, so it pins the
    pairing down to floating-point equality.
    """
    fit, data = fit_multichain(chains=4, draws=25)
    idata = fit.to_arviz(include_ppc=True)
    posterior = group(idata, "posterior")
    ppc = group(idata, "posterior_predictive")

    beta = posterior["beta"].values
    x = data["x"]

    # mu[k] == beta.flat[k] * x, for the flat API's documented chain-major order.
    expected = fit.posterior_predictive(expected=True)["obs"]
    exact = beta.reshape(-1, 1) * x[None, :]
    np.testing.assert_allclose(expected, exact, atol=1e-12)
    # A one-draw shift must break that, or the equality above proves nothing.
    assert not np.allclose(expected, np.roll(exact, 1, axis=0), atol=1e-12)

    # ...and the ArviZ grid is that same chain-major series laid out on
    # (chain, draw), value for value.
    sampled = fit.posterior_predictive(seed=42)["obs"]
    assert ppc["obs"].values.shape == (4, 25, len(x))
    np.testing.assert_array_equal(ppc["obs"].values.reshape(sampled.shape), sampled)


def test_ppc_samples_thins_each_chain_and_records_the_retained_draws():
    fit, data = fit_multichain(chains=4, draws=25)
    idata = fit.to_arviz(include_ppc=True, ppc_samples=20, ppc_seed=7)

    posterior = group(idata, "posterior")
    ppc = group(idata, "posterior_predictive")

    # Chain-stratified: 20 // 4 = 5 draws retained from every chain, so the
    # chain axis survives instead of being flattened away.
    assert ppc["obs"].shape == (4, 5, len(data["y"]))
    np.testing.assert_array_equal(ppc.chain.values, posterior.chain.values)

    retained = ppc.draw.values.tolist()
    assert len(retained) == 5
    assert len(set(retained)) == 5
    assert set(retained) <= set(posterior.draw.values.tolist())
    assert retained == sorted(retained)
    # Thinning is a random subsample, not truncation to the first k draws. The
    # seed is fixed, so this is deterministic rather than merely improbable.
    assert retained != list(range(5)), retained

    # The recorded coordinates are what makes the subsample recoverable:
    # selecting the posterior down to them lines the groups back up.
    matched = posterior.sel(draw=ppc.draw)
    assert matched["beta"].shape == (4, 5)

    # Shape alone would also pass for coordinates that are plausible but wrong,
    # so pin the values. A predictive draw carries observation noise and cannot
    # identify its own parameter draw, but the log likelihood is a deterministic
    # function of that draw, so it can. Selecting log_likelihood down to the
    # retained coordinates must reproduce the closed-form density evaluated at
    # the posterior draws those same coordinates select.
    log_lik = group(idata, "log_likelihood").sel(draw=ppc.draw)["obs"].values
    closed_form = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * (data["y"] - matched["beta"].values[..., None] * data["x"]) ** 2
    )
    np.testing.assert_allclose(log_lik, closed_form, atol=1e-12)
    # Negative control: the same comparison under a one-draw roll must fail, or
    # the assertion above is not actually testing the pairing.
    assert not np.allclose(log_lik, np.roll(closed_form, 1, axis=1), atol=1e-12)

    # Same seed, same retained draws; a different seed picks a different set.
    again = group(fit.to_arviz(include_ppc=True, ppc_samples=20, ppc_seed=7),
                  "posterior_predictive")
    assert again.draw.values.tolist() == retained
    other = group(fit.to_arviz(include_ppc=True, ppc_samples=20, ppc_seed=99),
                  "posterior_predictive")
    assert other.draw.values.tolist() != retained


def test_ppc_samples_below_chain_count_keeps_one_draw_per_chain():
    fit, _ = fit_multichain(chains=4, draws=25)
    ppc = group(
        fit.to_arviz(include_ppc=True, ppc_samples=2, ppc_seed=3),
        "posterior_predictive",
    )
    assert ppc["obs"].shape[:2] == (4, 1)


def test_unthinned_export_leaves_the_draw_coordinate_untouched():
    fit, _ = fit_multichain(chains=2, draws=12)
    idata = fit.to_arviz(include_ppc=True, ppc_samples=1000)
    posterior = group(idata, "posterior")
    ppc = group(idata, "posterior_predictive")
    assert ppc["obs"].shape[:2] == (2, 12)
    np.testing.assert_array_equal(ppc.draw.values, posterior.draw.values)


def test_retained_coordinates_follow_arviz_index_origin():
    """The labels come from the posterior group, not from bare 0-based indices.

    ArviZ's `data.index_origin` decides where the `draw` coordinate starts;
    writing raw indices would offset the two groups by one wherever it is set.
    """
    fit, _ = fit_multichain(chains=2, draws=12)
    original = arviz.rcParams["data.index_origin"]
    arviz.rcParams["data.index_origin"] = 1
    try:
        idata = fit.to_arviz(include_ppc=True, ppc_samples=6, ppc_seed=5)
        posterior = group(idata, "posterior")
        ppc = group(idata, "posterior_predictive")
        assert posterior.draw.values[0] == 1
        assert set(ppc.draw.values.tolist()) <= set(posterior.draw.values.tolist())
        assert posterior.sel(draw=ppc.draw)["beta"].shape == (2, 3)
    finally:
        arviz.rcParams["data.index_origin"] = original


def test_flat_posterior_predictive_api_is_unchanged():
    """`fit.posterior_predictive()` keeps its documented flat (n, n_obs) shape."""
    fit, data = fit_multichain(chains=4, draws=25)
    flat = fit.posterior_predictive()
    assert flat["obs"].shape == (100, len(data["y"]))
    subsampled = fit.posterior_predictive(n_samples=20, seed=7)
    assert subsampled["obs"].shape == (20, len(data["y"]))
