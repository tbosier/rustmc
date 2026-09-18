"""Exact sampler positions survive rounded transforms, batch paths, and JSON."""

import copy
import json
import math
import pathlib

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


# ── an artifact written by the PREVIOUS bounded transform must still load ──
#
# 0.13 re-associated the bounded sigmoid from `lower + span * s(raw)` to an
# endpoint-anchored form, so that it evaluates the density at the point it
# reports. The two forms round differently in the last bits. The position
# agreement check budgeted that difference relative to the *draw*, but the
# difference scales with the *interval*, so any interval straddling zero could
# push a legitimate draw near zero outside the budget and make a saved fit
# unloadable.
#
# This fixture is a real artifact written by the 0.12 extension, committed
# verbatim. It must not be regenerated with the current code: an artifact the
# current writer produces agrees with the current reader by construction, which
# is exactly why the existing round-trip tests missed this.
OLD_WRITER_UNIFORM = pathlib.Path(__file__).parent / "fixtures" / "graph_fit_v2_uniform.json"


def _previous_bounded_sigmoid(raw, lower, upper):
    """The 0.12 formula, mirrored here so the fixture's provenance is checkable.

    The association matters: 0.12 formed the sigmoid first and then scaled it,
    so `span * s` and `span / d` are not interchangeable at the last bit --
    which is the whole subject of this section.
    """
    s = 1.0 / (1.0 + math.exp(-raw))
    return lower + (upper - lower) * s


def test_the_committed_fixture_really_exercises_the_old_writer_regression():
    """Guard the guard: prove the fixture would fail a value-relative budget.

    If this ever stops holding, the fixture has been regenerated with the
    current writer and the regression test below has quietly become vacuous.
    """
    stored = json.loads(OLD_WRITER_UNIFORM.read_text())
    assert (stored["format"], stored["version"]) == ("rustmc.graph-fit", 2)
    lower, upper = -2.0, 3.0
    prior = stored["model"]["definition"]["priors"][0]["Uniform"]
    assert (prior["lower"], prior["upper"]) == (lower, upper)

    # Every stored draw is exactly what the PREVIOUS formula produces...
    pairs = [
        (displayed, raw)
        for cs, rs in zip(stored["posterior"]["samples"],
                          stored["posterior"]["unconstrained_samples"])
        for cd, rd in zip(cs, rs)
        for displayed, raw in zip(cd, rd)
    ]
    assert pairs
    for displayed, raw in pairs:
        assert _previous_bounded_sigmoid(raw, lower, upper) == displayed

    # ...and at least one of them lands near zero, where a budget of
    # 8 * eps * |draw| is too small to absorb the re-association.
    closest = min(abs(displayed) for displayed, _ in pairs)
    assert closest < 0.5 * (upper - lower), closest


def test_a_fit_written_by_the_previous_bounded_transform_still_loads():
    artifact = OLD_WRITER_UNIFORM.read_text()
    stored = json.loads(artifact)

    restored = rustmc.FitResult.from_json(artifact)

    # The old writer's draws are returned unchanged. The loader accepts the
    # earlier rounding; it does not quietly recompute the posterior into
    # something the user never sampled.
    np.testing.assert_array_equal(
        restored.get_samples()["u"],
        np.asarray(stored["posterior"]["samples"]).reshape(-1),
    )
    assert np.isfinite(restored.get_samples()["u"]).all()
    assert (restored.get_samples()["u"] > -2.0).all()
    assert (restored.get_samples()["u"] < 3.0).all()

    # It is a working fit, not just a document that parsed: it re-saves, and
    # re-saving preserves the old writer's draws rather than rewriting them.
    resaved = json.loads(restored.to_json())
    assert resaved["posterior"]["samples"] == stored["posterior"]["samples"]
    assert len(restored.diagnostics()) == 1  # the one parameter, `u`


@pytest.mark.parametrize("corruption", [
    # A raw position that decodes to a genuinely different draw misses by a
    # fraction of the interval, not by an ULP of it, so the widened budget
    # still refuses every one of these.
    lambda a: a["posterior"]["unconstrained_samples"][0][0].__setitem__(0, 0.0),
    lambda a: a["posterior"]["unconstrained_samples"][0][0].__setitem__(0, 1.0),
    lambda a: a["posterior"]["unconstrained_samples"][0][0].__setitem__(0, -40.0),
    lambda a: a["posterior"]["unconstrained_samples"][0][0].__setitem__(0, float("nan")),
    lambda a: a["posterior"]["samples"][0][0].__setitem__(0, 2.9),
    lambda a: a["posterior"]["unconstrained_samples"].pop(),
])
def test_a_corrupted_old_writer_artifact_is_still_rejected(corruption):
    artifact = json.loads(OLD_WRITER_UNIFORM.read_text())
    corruption(artifact)
    with pytest.raises(ValueError):
        rustmc.FitResult.from_json(json.dumps(artifact))


def test_the_widened_budget_is_far_tighter_than_a_wrong_draw():
    """The smallest corruption the check must still catch is orders of magnitude
    larger than the largest rounding it must now tolerate."""
    artifact = json.loads(OLD_WRITER_UNIFORM.read_text())
    lower, upper = -2.0, 3.0
    raw = artifact["posterior"]["unconstrained_samples"][0][0][0]
    displayed = artifact["posterior"]["samples"][0][0][0]

    budget = 8.0 * np.finfo(float).eps * ((upper - lower) + abs(displayed))
    # Nudging the raw position by one ULP is rounding, and is tolerated.
    nudged = np.nextafter(raw, math.inf)
    assert abs(_previous_bounded_sigmoid(nudged, lower, upper) - displayed) < budget
    # Moving it enough to change the draw at all is not.
    moved = raw + 1e-9
    assert abs(_previous_bounded_sigmoid(moved, lower, upper) - displayed) > budget
