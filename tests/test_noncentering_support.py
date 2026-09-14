"""Noncentering preserves the conditional Normal's scale support."""

import numpy as np
import pytest

import rustmc


def test_referenced_normal_scale_retains_positive_support_and_artifact_roundtrip():
    builder = rustmc.ModelBuilder()
    scale = builder.normal_prior("scale", 0.0, 1.0)
    builder.normal_prior("effect", 0.0, scale)
    compiled = builder.compile()
    restored = rustmc.CompiledModel.from_json(compiled.to_json())
    for model in [compiled, restored]:
        for sigma in [-1.0, 0.0]:
            lp, _ = model.log_density({}, [sigma, 0.3])
            assert lp == -np.inf
        for sigma in [1e-12, 0.5, 1.0, 10.0]:
            lp, grad = model.log_density({}, [sigma, 0.3])
            assert lp == pytest.approx(-np.log(2 * np.pi) - 0.5 * (sigma**2 + 0.3**2))
            np.testing.assert_allclose(grad, [-sigma, -0.3], atol=1e-12)


@pytest.mark.parametrize("kind", ["half_normal", "exponential", "log_normal", "gamma", "beta", "uniform"])
def test_positive_scale_declarations_remain_supported(kind):
    builder = rustmc.ModelBuilder()
    if kind == "half_normal":
        scale = builder.half_normal_prior("scale", 1.0)
    elif kind == "exponential":
        scale = builder.exponential_prior("scale", 1.0)
    elif kind == "log_normal":
        scale = builder.log_normal_prior("scale", 0.0, 1.0)
    elif kind == "gamma":
        scale = builder.gamma_prior("scale", 2.0, 1.0)
    elif kind == "beta":
        scale = builder.beta_prior("scale", 2.0, 2.0)
    else:
        scale = builder.uniform_prior("scale", 0.0, 2.0)
    scale_only = builder.compile()
    # A large mean must not erase raw effects in the transformed density.
    builder.normal_prior("effect", 1e20, scale)
    model = builder.compile()
    for raw_scale in [-15.0, 0.0, 10.0]:
        base_lp, base_grad = scale_only.log_density({}, [raw_scale])
        lp, grad = model.log_density({}, [raw_scale, 0.3])
        assert lp == pytest.approx(base_lp - 0.5 * (np.log(2 * np.pi) + 0.3**2))
        np.testing.assert_allclose(grad, [base_grad[0], -0.3], atol=1e-12)


def test_potential_names_match_artifact_validation():
    builder = rustmc.ModelBuilder()
    value = builder.normal_prior("value", 0.0, 1.0)
    with pytest.raises(ValueError, match="potential name must not be empty"):
        builder.potential("", -value * value)
    builder.potential("penalty", -value * value)
    original = builder.compile()
    restored = rustmc.CompiledModel.from_json(original.to_json())
    assert original.log_density({}, [0.3])[0] == restored.log_density({}, [0.3])[0]


def test_prior_predictive_rejects_sampled_nonpositive_referenced_scale():
    builder = rustmc.ModelBuilder()
    scale = builder.normal_prior("scale", -100.0, 1.0)
    builder.normal_prior("effect", 0.0, scale)
    with pytest.raises(ValueError, match="sigma must be > 0"):
        rustmc.sample_prior_predictive(builder.build(), n_samples=2, seed=42)
