"""Pinned defects found while building the statistical validation suite.

Each test here documents a *reproducible* engine defect. They are marked
``xfail`` so the suite stays green, and each one names the fix that should make
it pass. When a defect is fixed, delete the marker (or the whole test if it is
subsumed by a positive test elsewhere).

None of these are fixed in this branch: this worktree owns tests only.
See ``VALIDATION_RESULTS.md`` for the full write-up.
"""

import numpy as np
import pytest

rustmc = pytest.importorskip("rustmc")


@pytest.mark.xfail(
    reason=(
        "DEFECT 1: a parameter whose prior takes a ParamRef hyperparameter "
        "(any hierarchical model) cannot be referenced in a likelihood. "
        "build_prior_into_graph's auto-non-centering path registers the raw "
        "parameter as '<name>__raw' and stores the derived value node only in "
        "value_node_map, but build_mu_expr resolves names via "
        "Graph::node_by_name, which never sees it. Every hierarchical model "
        "expressible in the DSL fails with ValueError('Unknown param: ...'), "
        "including examples/hierarchical_example.py."
    ),
    raises=ValueError,
    strict=False,
)
def test_hierarchical_parameter_can_be_used_in_a_likelihood():
    rng = np.random.default_rng(0)
    data = {f"y_{j}": rng.normal(size=10) for j in range(3)}
    b = rustmc.ModelBuilder(data)
    mu_global = b.normal_prior("mu_global", 0.0, 10.0)
    sigma_group = b.half_normal_prior("sigma_group", 5.0)
    for j in range(3):
        group = b.normal_prior(f"mu_{j}", mu_global, sigma_group)
        b.normal_likelihood(f"obs_{j}", group, 2.0, f"y_{j}")

    fit = rustmc.sample(
        b.build(), chains=2, draws=200, warmup=200, seed=1, show_progress=False
    )
    means = fit.mean()
    assert "mu_global" in means and np.isfinite(means["mu_global"])


@pytest.mark.xfail(
    reason=(
        "DEFECT 2: sample_prior_predictive panics (a Rust PanicException, not "
        "a Python error) when a scalar prior is auto-promoted to a vector "
        "parameter by the '@' operator. sample_prior_raw pushes one raw value "
        "for PriorSpec::Normal / HalfNormal / StudentT / Uniform / Gamma / "
        "Beta regardless of auto_vector_params, so the raw vector is n-1 "
        "values short and derive_display_draw indexes out of bounds. Only the "
        "Exponential and LogNormal branches handle the vector case; explicit "
        "vector_normal_prior works because PriorSpec::VectorNormal loops over n."
    ),
    strict=False,
)
def test_prior_predictive_supports_auto_promoted_vector_parameters():
    rng = np.random.default_rng(0)
    n, p = 20, 3
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    b = rustmc.ModelBuilder({"X": X, "y": y})
    beta = b.normal_prior("beta", 0.0, 1.0)  # auto-promoted to length p by @
    b.normal_likelihood("y_obs", beta @ "X", 1.0, "y")

    out = rustmc.sample_prior_predictive(b.build(), n_samples=25, seed=1)
    assert np.asarray(out["y_obs"]).shape == (25, n)


def test_prior_predictive_works_with_an_explicit_vector_prior():
    """Positive control for DEFECT 2: the explicit vector prior path is fine,
    which is what localises the bug to the auto-promotion branch."""
    rng = np.random.default_rng(0)
    n, p = 20, 3
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    b = rustmc.ModelBuilder({"X": X, "y": y})
    beta = b.vector_normal_prior("beta", p, 0.0, 1.0)
    b.normal_likelihood("y_obs", beta @ "X", 1.0, "y")

    out = rustmc.sample_prior_predictive(b.build(), n_samples=25, seed=1)
    assert np.asarray(out["y_obs"]).shape == (25, n)


@pytest.mark.xfail(
    reason=(
        "DEFECT 3: FitResult.divergences() reports divergences accumulated "
        "over warmup *and* sampling. Stan and PyMC report post-warmup "
        "divergences only. On a well-behaved 1-parameter Gaussian target this "
        "reports ~20 'divergences' where the true post-warmup count is 0, so "
        "the number cannot be used as the health signal the docs describe. The "
        "per-draw flags needed to split it are already stored in "
        "SampleResult::transitions."
    ),
    strict=False,
)
def test_divergence_count_excludes_warmup():
    rng = np.random.default_rng(3)
    y = rng.normal(1.25, 0.8, size=40)
    b = rustmc.ModelBuilder({"y": y})
    mu = b.normal_prior("mu", 0.0, 2.0)
    b.normal_likelihood("y_obs", mu, 0.8, "y")
    fit = rustmc.sample(
        b.build(), chains=4, draws=750, warmup=750, seed=20001,
        threads=1, show_progress=False,
    )
    # The Rust-side post-warmup count for this exact model and seed is 0
    # (see rust_core/tests/analytic_posterior.rs).
    assert sum(fit.divergences()) == 0


@pytest.mark.xfail(
    reason=(
        "DEFECT 4: to_arviz() omits sample_stats['diverging'], warning that "
        "'exact per-draw divergence flags are not stored'. They are: "
        "SampleResult::transitions carries `divergent` and `is_warmup` for "
        "every transition. As a result az.plot_pair(divergences=True) and "
        "every ArviZ divergence diagnostic are unavailable."
    ),
    strict=False,
)
def test_arviz_export_includes_per_draw_divergence_flags():
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(4)
    y = rng.normal(size=30)
    b = rustmc.ModelBuilder({"y": y})
    mu = b.normal_prior("mu", 0.0, 2.0)
    b.normal_likelihood("y_obs", mu, 1.0, "y")
    fit = rustmc.sample(
        b.build(), chains=2, draws=200, warmup=200, seed=5, show_progress=False
    )
    idata = fit.to_arviz()
    assert "diverging" in idata.sample_stats
    assert idata.sample_stats["diverging"].shape == (2, 200)
    _ = az


def test_unknown_data_key_fails_with_an_actionable_error():
    """Positive failure-mode test: bad keys must raise a clear Python error,
    not panic and not silently produce a wrong model."""
    y = np.zeros(5)
    b = rustmc.ModelBuilder({"y": y})
    a = b.normal_prior("a", 0.0, 1.0)
    with pytest.raises(ValueError) as exc:
        b.normal_likelihood("y_obs", a * "not_a_key", 1.0, "y")
    assert "not_a_key" in str(exc.value)


def test_mismatched_observation_length_fails_with_an_actionable_error():
    b = rustmc.ModelBuilder({"x": np.zeros(10), "y": np.zeros(7)})
    a = b.normal_prior("a", 0.0, 1.0)
    b.normal_likelihood("y_obs", a * "x", 1.0, "y")
    with pytest.raises(ValueError) as exc:
        rustmc.sample(
            b.build(), chains=1, draws=10, warmup=10, seed=1, show_progress=False
        )
    msg = str(exc.value).lower()
    assert "length" in msg or "expected" in msg


def test_binary_likelihood_rejects_non_binary_observations():
    """Support validation happens at sample() time, not at builder time — the
    builder accepts the model and only `build_likelihood_into_graph` checks.
    That is a usability wart, not a correctness one, so it is asserted as-is."""
    b = rustmc.ModelBuilder({"y": np.array([0.0, 1.0, 2.0])})
    eta = b.normal_prior("eta", 0.0, 1.0)
    b.bernoulli_logit_likelihood("y_obs", eta, "y")  # accepted here
    with pytest.raises(ValueError) as exc:
        rustmc.sample(
            b.build(), chains=1, draws=10, warmup=10, seed=1, show_progress=False
        )
    assert "y_obs" in str(exc.value)


def test_count_likelihood_rejects_negative_observations():
    b = rustmc.ModelBuilder({"y": np.array([0.0, 1.0, -3.0])})
    eta = b.normal_prior("eta", 0.0, 1.0)
    b.poisson_log_likelihood("y_obs", eta, "y")
    with pytest.raises(ValueError) as exc:
        rustmc.sample(
            b.build(), chains=1, draws=10, warmup=10, seed=1, show_progress=False
        )
    assert "y_obs" in str(exc.value)


def test_discrete_prior_cannot_be_auto_promoted_by_matmul():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(10, 2))
    b = rustmc.ModelBuilder({"X": X, "y": rng.normal(size=10)})
    p = b.bernoulli_prior("p", 0.5)
    b.normal_likelihood("y_obs", p @ "X", 1.0, "y")
    with pytest.raises(ValueError) as exc:
        rustmc.sample(
            b.build(), chains=1, draws=10, warmup=10, seed=1, show_progress=False
        )
    assert "Bernoulli" in str(exc.value)
