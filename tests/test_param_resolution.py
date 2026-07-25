"""Parameter references must resolve, or fail loudly.

The failure mode these tests guard against is a *silent* one: a parameter
reference that cannot be resolved being replaced by a default value, producing
a posterior (or prior predictive draw) that looks plausible but is wrong.

Every assertion below therefore checks two things: that an error is raised, and
that the message names the offending parameter.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(20240725)
    x = rng.normal(size=200)
    y = 1.5 + 2.0 * x + rng.normal(scale=0.3, size=200)
    return {"x": x, "y": y}


def _simple_model(rustmc, data):
    """alpha + beta * x, sigma ~ HalfNormal — the canonical valid model."""
    b = rustmc.ModelBuilder(data)
    alpha = b.normal_prior("alpha", 0.0, 10.0)
    beta = b.normal_prior("beta", 0.0, 10.0)
    sigma = b.half_normal_prior("sigma", 1.0)
    b.normal_likelihood("y", alpha + beta * "x", sigma, "y")
    return b


# ── the exception type ────────────────────────────────────────────────────


def test_parameter_error_is_exported_and_subclasses_value_error(rustmc):
    assert issubclass(rustmc.ParameterError, ValueError)


def test_valid_model_builds_and_samples(rustmc, linear_data):
    spec = _simple_model(rustmc, linear_data).build()
    fit = rustmc.sample(spec, chains=2, draws=400, warmup=400, show_progress=False)
    means = fit.mean()
    assert means["alpha"] == pytest.approx(1.5, abs=0.15)
    assert means["beta"] == pytest.approx(2.0, abs=0.15)


# ── references belonging to a different model ─────────────────────────────


def test_hyperparameter_from_another_model_is_rejected(rustmc):
    """A ParamRef minted by one builder must not resolve inside another."""
    other = rustmc.ModelBuilder()
    foreign_sigma = other.half_normal_prior("sigma_pop", 1.0)

    b = rustmc.ModelBuilder()
    with pytest.raises(rustmc.ParameterError) as exc:
        b.normal_prior("theta", 0.0, foreign_sigma)

    message = str(exc.value)
    assert "sigma_pop" in message
    assert "different model" in message


def test_same_named_parameter_from_another_model_is_still_rejected(rustmc, linear_data):
    """The dangerous case: the name exists in both models, so name-based
    resolution would silently succeed against the wrong declaration."""
    other = rustmc.ModelBuilder()
    foreign_sigma = other.half_normal_prior("sigma", 1.0)

    b = rustmc.ModelBuilder(linear_data)
    b.normal_prior("alpha", 0.0, 10.0)
    b.half_normal_prior("sigma", 1.0)  # same name, different model

    with pytest.raises(rustmc.ParameterError) as exc:
        b.normal_prior("theta", 0.0, foreign_sigma)
    assert "sigma" in str(exc.value)


def test_likelihood_scale_from_another_model_is_rejected(rustmc, linear_data):
    other = rustmc.ModelBuilder()
    foreign_sigma = other.half_normal_prior("sigma_other", 1.0)

    b = rustmc.ModelBuilder(linear_data)
    alpha = b.normal_prior("alpha", 0.0, 10.0)
    with pytest.raises(rustmc.ParameterError) as exc:
        b.normal_likelihood("y", alpha, foreign_sigma, "y")

    message = str(exc.value)
    assert "sigma_other" in message
    assert "different model" in message


def test_predictor_from_another_model_is_rejected(rustmc, linear_data):
    other = rustmc.ModelBuilder()
    foreign_beta = other.normal_prior("beta_other", 0.0, 1.0)

    b = rustmc.ModelBuilder(linear_data)
    sigma = b.half_normal_prior("sigma", 1.0)
    with pytest.raises(rustmc.ParameterError) as exc:
        b.normal_likelihood("y", foreign_beta * "x", sigma, "y")

    message = str(exc.value)
    assert "beta_other" in message
    assert "different model" in message


def test_expression_mixing_two_models_is_rejected(rustmc):
    a = rustmc.ModelBuilder()
    beta_a = a.normal_prior("beta_a", 0.0, 1.0)

    b = rustmc.ModelBuilder()
    beta_b = b.normal_prior("beta_b", 0.0, 1.0)

    with pytest.raises(rustmc.ParameterError) as exc:
        _ = beta_a * "x" + beta_b * "x"

    message = str(exc.value)
    assert "two different models" in message
    assert "beta_a" in message or "beta_b" in message


def test_reverse_add_also_rejects_mixed_models(rustmc):
    a = rustmc.ModelBuilder()
    alpha_a = a.normal_prior("alpha_a", 0.0, 1.0)

    b = rustmc.ModelBuilder()
    beta_b = b.normal_prior("beta_b", 0.0, 1.0)

    with pytest.raises(rustmc.ParameterError) as exc:
        _ = alpha_a + beta_b * "x"
    assert "alpha_a" in str(exc.value) or "beta_b" in str(exc.value)


# ── unresolvable references inside one model ──────────────────────────────


def test_duplicate_parameter_declaration_is_rejected(rustmc, linear_data):
    """Two priors with the same name make every reference to it ambiguous."""
    b = rustmc.ModelBuilder(linear_data)
    alpha = b.normal_prior("alpha", 0.0, 10.0)
    b.normal_prior("alpha", 0.0, 1.0)
    sigma = b.half_normal_prior("sigma", 1.0)
    b.normal_likelihood("y", alpha, sigma, "y")

    with pytest.raises(rustmc.ParameterError) as exc:
        b.build()
    message = str(exc.value)
    assert "alpha" in message
    assert "declared twice" in message


def test_scalar_use_of_a_vector_promoted_parameter_is_rejected(rustmc):
    """`beta @ 'X'` promotes beta to a vector parameter; using the same name as
    a scalar coefficient can no longer resolve, and must not fall back."""
    rng = np.random.default_rng(0)
    data = {
        "X": rng.normal(size=(50, 3)),
        "x": rng.normal(size=50),
        "y": rng.normal(size=50),
    }
    b = rustmc.ModelBuilder(data)
    beta = b.normal_prior("beta", 0.0, 1.0)
    sigma = b.half_normal_prior("sigma", 1.0)
    b.normal_likelihood("y", beta @ "X" + beta * "x", sigma, "y")
    spec = b.build()

    with pytest.raises(rustmc.ParameterError) as exc:
        rustmc.sample(spec, chains=1, draws=10, warmup=10, show_progress=False)
    assert "beta" in str(exc.value)


# ── resolution must be correct, not merely present ────────────────────────


def test_constrained_parameter_in_predictor_uses_its_constrained_value(rustmc):
    """A positive-support parameter used as a coefficient must contribute its
    constrained value, not the unconstrained (log-scale) sampling variable.

    Regression test: resolving through `Graph::node_by_name` returned the raw
    node for transformed priors, so a HalfNormal coefficient silently entered
    the linear predictor as log(beta) and the posterior was wrong.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    beta_true = 2.5
    y = beta_true * x + rng.normal(scale=0.2, size=300)

    b = rustmc.ModelBuilder({"x": x, "y": y})
    beta = b.half_normal_prior("beta", 5.0)
    sigma = b.half_normal_prior("sigma", 1.0)
    b.normal_likelihood("y", beta * "x", sigma, "y")
    fit = rustmc.sample(b.build(), chains=2, draws=500, warmup=500, show_progress=False)

    assert fit.mean()["beta"] == pytest.approx(beta_true, abs=0.15)


def test_hierarchical_prior_predictive_resolves_its_hyperparameter(rustmc):
    """Prior predictive draws must use the sampled hyperparameter value.

    The removed defect substituted 1.0 for an unresolved hyperparameter, which
    would place `theta` near 1 instead of near `mu_pop`.
    """
    n = 40
    data = {"x": np.zeros(n), "y": np.zeros(n)}

    b = rustmc.ModelBuilder(data)
    mu_pop = b.normal_prior("mu_pop", 100.0, 0.5)
    theta = b.normal_prior("theta", mu_pop, 1.0)
    sigma = b.half_normal_prior("sigma", 1.0)
    b.normal_likelihood("y", theta, sigma, "y")

    draws = rustmc.sample_prior_predictive(b.build(), n_samples=400, seed=1)
    assert draws["theta"].mean() == pytest.approx(100.0, abs=1.0)
    assert abs(draws["theta"].mean() - 1.0) > 50.0


def test_hierarchical_model_still_samples(rustmc):
    """Ordering-sensitive hierarchical models keep working end to end."""
    rng = np.random.default_rng(11)
    y = rng.normal(loc=3.0, scale=1.0, size=200)
    data = {"y": y}

    b = rustmc.ModelBuilder(data)
    mu_pop = b.normal_prior("mu_pop", 0.0, 10.0)
    sigma_pop = b.half_normal_prior("sigma_pop", 5.0)
    theta = b.normal_prior("theta", mu_pop, sigma_pop)
    sigma = b.half_normal_prior("sigma", 5.0)
    b.normal_likelihood("y", theta, sigma, "y")

    fit = rustmc.sample(b.build(), chains=2, draws=400, warmup=400, show_progress=False)
    assert fit.mean()["theta"] == pytest.approx(3.0, abs=0.3)
