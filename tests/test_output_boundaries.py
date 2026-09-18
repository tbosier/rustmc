"""Invalid ranges and overflowing displayed draws never become successful fits."""

import json

import numpy as np
import pytest

import rustmc


def test_uniform_overflowing_width_is_rejected_by_builder_and_artifact():
    builder = rustmc.ModelBuilder()
    with pytest.raises(ValueError, match="uniform width"):
        builder.uniform_prior("x", -1e308, 1e308)
    builder.uniform_prior("x", -1.0, 1.0)
    artifact = json.loads(builder.compile().to_json())
    artifact["definition"]["priors"][0]["Uniform"].update(lower=-1e308, upper=1e308)
    with pytest.raises(ValueError, match="uniform width"):
        rustmc.CompiledModel.from_json(json.dumps(artifact))


@pytest.mark.parametrize(
    "path,field",
    [
        ([], "provenance"),
        (["schema"], "scalars"),
        (["schema", "observations", 0], "stride"),
        (["definition"], "constraints"),
        (["definition", "priors", 0, "Uniform"], "tau"),
        (["definition", "likelihoods", 0], "weights"),
    ],
)
def test_artifact_fields_this_version_cannot_carry_are_rejected(path, field):
    """A field we would silently drop must fail the load, not survive as a lie.

    Round-tripping such an artifact used to return one with the field gone,
    so a newer or corrupted writer lost data with no error anywhere.
    """
    builder = rustmc.ModelBuilder()
    x = builder.uniform_prior("x", -1.0, 1.0)
    builder.normal_likelihood("obs", x, 1.0, "y")
    artifact = json.loads(builder.compile().to_json())
    target = artifact
    for step in path:
        target = target[step]
    target[field] = "this version cannot carry me"
    with pytest.raises(ValueError, match=f"unknown field `{field}`"):
        rustmc.CompiledModel.from_json(json.dumps(artifact))


@pytest.mark.parametrize("entrypoint", ["sample", "batch", "legacy_batch"])
def test_nonfinite_constrained_draws_are_rejected(entrypoint):
    builder = rustmc.ModelBuilder()
    builder.log_normal_prior("x", 800.0, 1.0)
    compiled = builder.compile()
    options = dict(chains=1, draws=3, warmup=1, sampler="hmc", step_size=1.0,
                   num_leapfrog_steps=2, seed=42, show_progress=False)
    with pytest.raises(ValueError, match="nonfinite after transformation"):
        if entrypoint == "sample":
            compiled.sample({}, init=[[800.0]], **options)
        elif entrypoint == "batch":
            compiled.sample_batch([{}], ids=["cell"], init={"cell": [[800.0]]}, **options)
        else:
            rustmc.batch_sample([(builder.build(), {})], **options)


def test_nonfinite_noncentered_display_draws_are_rejected():
    builder = rustmc.ModelBuilder()
    scale = builder.normal_prior("scale", 1e308, 1.0)
    builder.normal_prior("effect", 1e308, scale)
    with pytest.raises(ValueError, match="nonfinite after display transformation"):
        builder.compile().sample({}, init=[[1e308, 2.0]], chains=1, draws=2, warmup=1,
                                 sampler="hmc", step_size=1e-6, num_leapfrog_steps=1,
                                 seed=42, show_progress=False)


def test_valid_large_uniform_range_still_produces_finite_draws():
    builder = rustmc.ModelBuilder()
    builder.uniform_prior("x", -1e307, 1e307)
    fit = builder.compile().sample({}, chains=1, draws=4, warmup=4, show_progress=False)
    assert np.isfinite(fit.get_samples()["x"]).all()
    restored = rustmc.FitResult.from_json(fit.to_json())
    np.testing.assert_array_equal(restored.get_samples()["x"], fit.get_samples()["x"])


def _blowup_data(n=5):
    return {"x": np.linspace(-1.0, 1.0, n), "y": np.zeros(n)}


def test_nonfinite_prior_predictive_deterministic_is_rejected():
    """A deterministic that overflows must fail, not ride along in the result.

    The representability check covered the raw and display draws only, so a
    `deterministic` evaluating to infinity was handed back inside an otherwise
    successful prior predictive -- the one sampled output this library did not
    guard.
    """
    builder = rustmc.ModelBuilder(_blowup_data())
    alpha = builder.normal_prior("alpha", 800.0, 1.0)
    builder.deterministic("blown", alpha.exp())
    builder.normal_likelihood("obs", alpha * "x", 1.0, "y")
    with pytest.raises(ValueError, match="deterministic 'blown' is nonfinite"):
        rustmc.sample_prior_predictive(builder.build(), n_samples=4, seed=1)


def test_finite_prior_predictive_deterministics_still_come_back():
    """The guard must not cost a model whose deterministics are representable."""
    builder = rustmc.ModelBuilder(_blowup_data())
    alpha = builder.normal_prior("alpha", 0.0, 1.0)
    builder.deterministic("scaled", alpha * 2.0)
    builder.deterministic("per_obs", alpha * "x")
    builder.normal_likelihood("obs", alpha * "x", 1.0, "y")
    draws = rustmc.sample_prior_predictive(builder.build(), n_samples=8, seed=1)
    assert np.isfinite(draws["scaled"]).all()
    assert draws["per_obs"].shape == (8, 5)
    assert np.isfinite(draws["per_obs"]).all()


def test_nonfinite_posterior_deterministic_is_rejected():
    """The posterior accessor had the same hole, and is closed the same way."""
    builder = rustmc.ModelBuilder(_blowup_data())
    alpha = builder.normal_prior("alpha", 0.0, 1.0)
    builder.deterministic("blown", (alpha * 5000.0).exp())
    builder.normal_likelihood("obs", alpha * "x", 1.0, "y")
    fit = rustmc.sample(
        builder.build(), chains=2, draws=200, warmup=200, seed=3, show_progress=False
    )
    with pytest.raises(ValueError, match="deterministic 'blown' is nonfinite"):
        fit.deterministics()


def test_finite_posterior_deterministics_still_come_back():
    builder = rustmc.ModelBuilder(_blowup_data())
    alpha = builder.normal_prior("alpha", 0.0, 1.0)
    builder.deterministic("scaled", alpha * 2.0)
    builder.normal_likelihood("obs", alpha * "x", 1.0, "y")
    fit = rustmc.sample(
        builder.build(), chains=1, draws=20, warmup=20, seed=3, show_progress=False
    )
    assert np.isfinite(fit.deterministics()["scaled"]).all()
