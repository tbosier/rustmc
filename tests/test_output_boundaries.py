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
