"""Focused safety checks for Python model construction and binding."""

import numpy as np
import pytest


def _model(rustmc, data=None):
    data = data or {"x": np.ones(4), "y": np.zeros(4)}
    builder = rustmc.ModelBuilder(data)
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("y", beta * "x", 1.0, "y")
    return builder.build()


def test_builder_context_returns_self_and_never_suppresses(rustmc_module):
    builder = rustmc_module.ModelBuilder()
    with builder as entered:
        assert entered is builder

    with pytest.raises(RuntimeError, match="sentinel"):
        with builder:
            raise RuntimeError("sentinel")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_vector_data_is_rejected(rustmc_module, bad):
    with pytest.raises(ValueError, match="non-finite"):
        rustmc_module.ModelBuilder({"x": np.array([0.0, bad])})


def test_non_finite_matrix_data_is_rejected(rustmc_module):
    with pytest.raises(ValueError, match="non-finite"):
        rustmc_module.ModelBuilder({"X": np.array([[0.0, np.nan]])})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chains": 0}, "chains"),
        ({"draws": 0}, "draws"),
        ({"warmup": 0}, "warmup"),
        ({"step_size": float("nan")}, "step_size"),
        ({"step_size": -0.1}, "step_size"),
        ({"max_tree_depth": 0}, "max_tree_depth"),
        ({"max_tree_depth": 64}, "max_tree_depth"),
        ({"num_leapfrog_steps": 0}, "num_leapfrog_steps"),
    ],
)
def test_invalid_sample_configuration_fails_before_sampling(
    rustmc_module, kwargs, message
):
    config = {
        "chains": 1,
        "draws": 1,
        "warmup": 1,
        "show_progress": False,
    }
    config.update(kwargs)
    with pytest.raises(ValueError, match=message):
        rustmc_module.sample(_model(rustmc_module), **config)


@pytest.mark.parametrize(
    "make_prior",
    [
        lambda b: b.normal_prior("beta", 0.0, 1.0),
        lambda b: b.half_normal_prior("beta", 1.0),
        lambda b: b.exponential_prior("beta", 1.0),
        lambda b: b.log_normal_prior("beta", 0.0, 1.0),
        lambda b: b.student_t_prior("beta", 4.0),
        lambda b: b.uniform_prior("beta", -2.0, 2.0),
        lambda b: b.gamma_prior("beta", 2.0, 1.0),
        lambda b: b.beta_prior("beta", 2.0, 2.0),
    ],
)
def test_prior_predictive_auto_vector_priors_have_compiled_width(
    rustmc_module, make_prior
):
    n, width, draws = 5, 3, 7
    builder = rustmc_module.ModelBuilder(
        {"X": np.arange(n * width, dtype=float).reshape(n, width), "y": np.zeros(n)}
    )
    beta = make_prior(builder)
    builder.normal_likelihood("y", beta @ "X", 1.0, "y")
    result = rustmc_module.sample_prior_predictive(
        builder.build(), n_samples=draws, seed=10
    )
    assert result["y"].shape == (draws, n)
    assert np.isfinite(result["y"]).all()


@pytest.mark.parametrize(
    "make_prior",
    [
        lambda b: b.half_normal_prior("beta", 1.0),
        lambda b: b.exponential_prior("beta", 1.0),
        lambda b: b.log_normal_prior("beta", 0.0, 0.5),
        lambda b: b.uniform_prior("beta", -2.0, 3.0),
        lambda b: b.gamma_prior("beta", 2.0, 1.0),
        lambda b: b.beta_prior("beta", 2.0, 3.0),
    ],
)
def test_constrained_auto_vector_draws_are_the_values_used_by_matvec(
    rustmc_module, make_prior
):
    width, draws = 3, 40
    builder = rustmc_module.ModelBuilder(
        {"X": np.eye(width), "y": np.zeros(width)}
    )
    beta = make_prior(builder)
    builder.normal_likelihood("y", beta @ "X", 1e-10, "y")
    result = rustmc_module.sample_prior_predictive(
        builder.build(), n_samples=draws, seed=99
    )

    displayed = np.column_stack([result[f"beta[{j}]"] for j in range(width)])
    np.testing.assert_allclose(result["y"], displayed, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize(
    ("method", "args", "message"),
    [
        ("normal_prior", ("x", 0.0, 0.0), "sigma"),
        ("half_normal_prior", ("x", -1.0), "sigma"),
        ("exponential_prior", ("x", float("nan")), "rate"),
        ("log_normal_prior", ("x", 0.0, float("inf")), "sigma"),
        ("student_t_prior", ("x", 0.0), "nu"),
        ("uniform_prior", ("x", 2.0, 1.0), "lower"),
        ("bernoulli_prior", ("x", 1.1), "p"),
        ("poisson_prior", ("x", 0.0), "lam"),
        ("gamma_prior", ("x", -1.0, 2.0), "alpha"),
        ("beta_prior", ("x", 1.0, 0.0), "beta"),
        ("vector_normal_prior", ("x", 0), "n"),
    ],
)
def test_invalid_prior_domains_raise_value_error(
    rustmc_module, method, args, message
):
    builder = rustmc_module.ModelBuilder()
    with pytest.raises(ValueError, match=message):
        getattr(builder, method)(*args)


def test_invalid_constant_likelihood_scale_is_rejected(rustmc_module):
    builder = rustmc_module.ModelBuilder()
    beta = builder.normal_prior("beta", 0.0, 1.0)
    with pytest.raises(ValueError, match="sigma"):
        builder.normal_likelihood("y", beta * "x", 0.0, "y")


def test_empty_data_is_rejected(rustmc_module):
    with pytest.raises(ValueError, match="at least one"):
        rustmc_module.ModelBuilder({"x": np.array([], dtype=float)})


@pytest.mark.parametrize("entrypoint", ["sample", "batch", "prior_predictive"])
def test_cross_dimensional_override_removes_stale_bound_value(
    rustmc_module, entrypoint
):
    builder = rustmc_module.ModelBuilder(
        {"X": np.eye(2), "y": np.zeros(2)}
    )
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("y", beta @ "X", 1.0, "y")
    spec = builder.build()
    wrong_kind = {"X": np.ones(2)}

    with pytest.raises(ValueError, match="matrix key 'X'"):
        if entrypoint == "sample":
            rustmc_module.sample(
                spec,
                data=wrong_kind,
                chains=1,
                draws=1,
                warmup=1,
                show_progress=False,
            )
        elif entrypoint == "batch":
            rustmc_module.batch_sample(
                [(spec, wrong_kind)],
                chains=1,
                draws=1,
                warmup=1,
                show_progress=False,
            )
        else:
            rustmc_module.sample_prior_predictive(
                spec, data=wrong_kind, n_samples=1
            )


def test_zero_prior_predictive_draws_is_rejected(rustmc_module):
    with pytest.raises(ValueError, match="n_samples"):
        rustmc_module.sample_prior_predictive(
            _model(rustmc_module), n_samples=0
        )
