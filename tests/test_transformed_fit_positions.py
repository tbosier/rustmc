"""Exact sampler positions survive rounded transforms, batch paths, and JSON."""

import copy
import json

import numpy as np
import pytest

import rustmc


def tail_builder(kind):
    data = {"y": np.ones(2)}
    if kind == "vector_beta":
        data["X"] = np.eye(2)
    builder = rustmc.ModelBuilder(data)
    if kind == "uniform":
        value = builder.uniform_prior("value", 0.0, 1.0)
    else:
        value = builder.beta_prior("value", 0.01, 0.01)
    predictor = value @ "X" if kind == "vector_beta" else value
    builder.normal_likelihood("obs", predictor, 1.0, "y")
    builder.deterministic("mean", predictor)
    return builder, data


@pytest.mark.parametrize("kind", ["beta", "vector_beta", "uniform"])
def test_tail_fit_artifact_retains_finite_positions_and_replays(kind):
    builder, data = tail_builder(kind)
    width = 2 if kind == "vector_beta" else 1
    compiled = builder.compile()
    fit = compiled.sample(data, init=[[40.0] * width], sampler="hmc", step_size=1e-6,
                          num_leapfrog_steps=1, chains=1, draws=3, warmup=1,
                          seed=17, show_progress=False)
    artifact = json.loads(fit.to_json())
    assert artifact["version"] == 2
    raw = np.asarray(artifact["posterior"]["unconstrained_samples"])
    assert raw.shape == (1, 3, width)
    assert np.all(np.isfinite(raw))
    assert np.all(raw > 39.0)
    np.testing.assert_array_equal(artifact["posterior"]["samples"], np.ones((1, 3, width)))
    restored = rustmc.FitResult.from_json(json.dumps(artifact))
    for result in [fit, restored]:
        np.testing.assert_array_equal(result.predict(expected=True)["obs"], np.ones((1, 3, 2)))
        np.testing.assert_allclose(result.log_likelihood()["obs"], -0.5 * np.log(2 * np.pi))
        assert np.isfinite(result.deterministics()["mean"]).all()
    np.testing.assert_array_equal(fit.predict(seed=31)["obs"], restored.predict(seed=31)["obs"])
    assert artifact == json.loads(restored.to_json())

    # A v1 artifact has no way to recover the finite position from rounded one.
    legacy = copy.deepcopy(artifact)
    legacy["version"] = 1
    del legacy["posterior"]["unconstrained_samples"]
    with pytest.raises(ValueError, match="transform support"):
        rustmc.FitResult.from_json(json.dumps(legacy))


def test_legacy_fit_reading_reconstructs_positions_and_upgrades_to_version_two():
    builder = rustmc.ModelBuilder({"y": np.ones(2)})
    scale = builder.half_normal_prior("scale", 1.0)
    builder.normal_likelihood("obs", scale, 1.0, "y")
    fit = builder.compile().sample({}, chains=1, draws=3, warmup=3, seed=12, show_progress=False)
    legacy = json.loads(fit.to_json())
    legacy["version"] = 1
    del legacy["posterior"]["unconstrained_samples"]
    restored = rustmc.FitResult.from_json(json.dumps(legacy))
    np.testing.assert_array_equal(restored.get_samples()["scale"], fit.get_samples()["scale"])
    np.testing.assert_allclose(restored.predict(expected=True)["obs"], fit.predict(expected=True)["obs"])
    upgraded = json.loads(restored.to_json())
    assert upgraded["version"] == 2
    assert "unconstrained_samples" in upgraded["posterior"]


@pytest.mark.parametrize("corruption", [
    lambda a: a["posterior"].pop("unconstrained_samples"),
    lambda a: a["posterior"]["unconstrained_samples"].pop(),
    lambda a: a["posterior"]["unconstrained_samples"][0].pop(),
    lambda a: a["posterior"]["unconstrained_samples"][0][0].pop(),
    lambda a: a["posterior"]["unconstrained_samples"][0][0].__setitem__(0, float("nan")),
    lambda a: a["posterior"]["unconstrained_samples"][0][0].__setitem__(0, 0.0),
    lambda a: a.update(version=1),
])
def test_corrupt_unconstrained_positions_are_rejected(corruption):
    builder, data = tail_builder("beta")
    fit = builder.compile().sample(data, init=[[40.0]], sampler="hmc", step_size=1e-6,
                                   num_leapfrog_steps=1, chains=1, draws=2, warmup=1,
                                   seed=17, show_progress=False)
    artifact = json.loads(fit.to_json())
    corruption(artifact)
    with pytest.raises(ValueError):
        rustmc.FitResult.from_json(json.dumps(artifact))


def test_global_and_batch_entrypoints_retain_raw_graph_coordinates():
    builder = rustmc.ModelBuilder({"y": np.ones(2)})
    scale = builder.half_normal_prior("scale", 1.0)
    builder.normal_likelihood("obs", scale, 1.0, "y")
    options = dict(chains=2, draws=3, warmup=3, seed=43, show_progress=False)
    compiled = builder.compile()
    results = [
        rustmc.sample(builder.build(), **options),
        compiled.sample_batch([{}], ids=["one"], **options).get("one").fit,
        rustmc.batch_sample([(builder.build(), {})], **options)[0].fit,
    ]
    for fit in results:
        artifact = json.loads(fit.to_json())
        raw = np.asarray(artifact["posterior"]["unconstrained_samples"])
        assert raw.shape == (2, 3, 1)
        np.testing.assert_allclose(np.exp(raw), artifact["posterior"]["samples"])
        restored = rustmc.FitResult.from_json(json.dumps(artifact))
        np.testing.assert_array_equal(fit.predict(seed=99)["obs"], restored.predict(seed=99)["obs"])
