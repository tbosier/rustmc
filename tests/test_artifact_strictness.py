"""Every artifact loader must reject an unknown field rather than drop it.

Serde ignores unknown fields by default, so an artifact written by a newer or a
corrupted writer used to load, lose the field, and re-save as a quietly
different document -- with no error anywhere to attribute the loss to.

`rust_core/tests/artifact_strictness.rs` covers the Rust loaders directly.
This module covers the five Python entry points that reach them, because each
`from_json` is its own promise to the caller and they were not all keeping it.
"""
import json

import numpy as np
import pytest

import rustmc


NAMES = ["CompiledModel", "FitResult", "StructuralModel", "StructuralFit", "DynamicGLMFit"]


def _graph_model_and_fit():
    data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([1.0, 2.0, 3.0])}
    builder = rustmc.ModelBuilder(data)
    alpha = builder.normal_prior("alpha", 0.0, 1.0)
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("obs", alpha + beta * builder.data("x"), 0.8, "y")
    compiled = builder.compile()
    fit = compiled.sample(data, chains=1, draws=5, warmup=5, seed=7, show_progress=False)
    return compiled, fit


def _structural_model_and_fit():
    V, C = rustmc.VarianceParameter, rustmc.StructuralComponent
    # Inverse-gamma on both variances so the struct-variant arm of
    # VarianceParameter -- the only arm the attribute can act on -- is present.
    model = rustmc.StructuralModel(
        [C.level("lvl", V.inverse_gamma(3.0, 0.1), 0.0, 1.0)],
        V.inverse_gamma(3.0, 0.2),
    )
    fit = model.fit(np.array([0.1, 0.2, 0.15, 0.3]), chains=1, draws=5, warmup=5,
                    seed=1, store_states=True)
    return model, fit


def _dynamic_glm_fit():
    model = rustmc.BayesianDynamicPoisson()
    return model.fit([[0.0, 1.0, np.nan, 2.0]], chains=1, warmup=10, draws=10, seed=11)


@pytest.fixture(scope="module")
def loaders():
    """name -> (artifact text, loader, path to a nested object inside it)."""
    compiled, graph_fit = _graph_model_and_fit()
    structural_model, structural_fit = _structural_model_and_fit()
    glm_fit = _dynamic_glm_fit()
    return {
        "CompiledModel": (compiled.to_json(), rustmc.CompiledModel.from_json, ["definition"]),
        "FitResult": (graph_fit.to_json(), rustmc.FitResult.from_json, ["posterior"]),
        "StructuralModel": (structural_model.to_json(), rustmc.StructuralModel.from_json, ["model"]),
        "StructuralFit": (structural_fit.to_json(), rustmc.StructuralFit.from_json, ["posterior"]),
        "DynamicGLMFit": (glm_fit.to_json(), rustmc.DynamicGLMFit.from_json, ["posterior"]),
    }


@pytest.mark.parametrize("name", NAMES)
def test_an_unknown_top_level_field_is_rejected(name, loaders):
    text, load, _ = loaders[name]
    artifact = json.loads(text)
    artifact["totally_unknown_field"] = 123
    with pytest.raises(ValueError, match="unknown field `totally_unknown_field`"):
        load(json.dumps(artifact))


@pytest.mark.parametrize("name", NAMES)
def test_an_unknown_field_in_a_nested_object_is_rejected(name, loaders):
    text, load, path = loaders[name]
    artifact = json.loads(text)
    target = artifact
    for step in path:
        target = target[step]
    assert isinstance(target, dict)
    target["nested_unknown"] = 1
    with pytest.raises(ValueError, match="unknown field `nested_unknown`"):
        load(json.dumps(artifact))


@pytest.mark.parametrize("name", NAMES)
def test_an_unknown_field_deep_inside_the_artifact_is_rejected(name, loaders):
    """A rule applied near the top of a document is not the claim being made.

    Each path below ends at the innermost named-field type that loader can
    reach -- a prior's parameters, a component's per-coordinate variance prior,
    a single posterior draw -- so a nested payload cannot be truncated either.
    """
    deepest = {
        "CompiledModel": (["definition", "priors", 0, "Normal"], "tau"),
        "FitResult": (["posterior", "transitions", 0, 0], "momentum"),
        "StructuralModel": (
            ["model", "components", 0, "innovations", 0, "InverseGamma"], "rate"),
        "StructuralFit": (["posterior", "chains", 0, 0], "log_density"),
        "DynamicGLMFit": (["posterior", "chains", 0, 0], "log_density"),
    }
    text, load, _ = loaders[name]
    path, field = deepest[name]
    artifact = json.loads(text)
    target = artifact
    for step in path:
        target = target[step]
    assert isinstance(target, dict)
    target[field] = 1.0
    with pytest.raises(ValueError, match=f"unknown field `{field}`"):
        load(json.dumps(artifact))


@pytest.mark.parametrize("name", ["CompiledModel", "FitResult"])
def test_the_embedded_graph_model_is_strict_from_both_loaders(name, loaders):
    """`FitResult` embeds a whole model artifact, and must not relax it."""
    prefix = {"CompiledModel": [], "FitResult": ["model"]}[name]
    text, load, _ = loaders[name]
    artifact = json.loads(text)
    target = artifact
    for step in prefix + ["definition"]:
        target = target[step]
    target["constraints"] = {}
    with pytest.raises(ValueError, match="unknown field `constraints`"):
        load(json.dumps(artifact))


# The two graph artifacts embed unordered `HashMap` payloads -- the model's
# `dimensions`, and the fit's `training` vectors and matrices -- so their JSON
# key order is not stable from one decode to the next and their round-trip is
# checked for content rather than for bytes. The structural and dynamic-GLM
# artifacts are ordered throughout, so those are checked byte for byte.
BYTE_STABLE = {"StructuralModel", "StructuralFit", "DynamicGLMFit"}


@pytest.mark.parametrize("name", NAMES)
def test_the_artifact_still_round_trips(name, loaders):
    """Strictness must not have cost any real writer its own reader.

    If this fails, something this library emits is a field it now refuses --
    which is a bug in the format, not in the test.
    """
    text, load, _ = loaders[name]
    resaved = load(text).to_json()
    assert json.loads(resaved) == json.loads(text)
    if name in BYTE_STABLE:
        assert resaved == text


def test_a_training_entry_in_the_wrong_namespace_is_rejected():
    """A key is checked against the namespace it was supplied in.

    `DataSchema::required_keys` flattens observations, vectors and matrices into
    one set. Checking inputs against that union let a *matrix* keyed with the name
    of a required *vector* pass as a known key; the binding then dropped it on the
    next save, so a round trip silently lost data the artifact claimed to carry.
    A genuinely unknown key was always rejected, which is why this hid.
    """
    import json

    import numpy as np

    import rustmc

    matrix_builder = rustmc.ModelBuilder()
    beta = matrix_builder.vector_normal_prior("beta", 2, 0.0, 1.0)
    matrix_builder.normal_likelihood("obs", beta @ "X", 1.0, "y")
    matrix_model = matrix_builder.compile()
    design = np.column_stack([np.ones(6), np.linspace(-1.0, 1.0, 6)])
    matrix_fit = matrix_model.sample(
        {"X": design, "y": np.linspace(0.0, 1.0, 6)},
        chains=1, warmup=50, draws=50, seed=1, show_progress=False,
    )
    serialized_matrix = json.loads(matrix_fit.to_json())["training"]["matrices"]["X"]

    builder = rustmc.ModelBuilder()
    slope = builder.normal_prior("a", 0.0, 1.0)
    builder.normal_likelihood("obs", slope * "x", 1.0, "y")
    compiled = builder.compile()
    predictor = np.linspace(-1.0, 1.0, 6)
    fit = compiled.sample(
        {"x": predictor, "y": 0.5 * predictor + 0.1},
        chains=1, warmup=50, draws=50, seed=1, show_progress=False,
    )

    # "x" is a required vector and "y" a required observation; neither is a matrix.
    for stolen in ("x", "y"):
        artifact = json.loads(fit.to_json())
        artifact["training"]["matrices"][stolen] = serialized_matrix
        with pytest.raises(ValueError, match=stolen):
            rustmc.FitResult.from_json(json.dumps(artifact))

    # The unchanged artifact still round-trips, so the check is not merely strict.
    assert rustmc.FitResult.from_json(fit.to_json()).mean() == fit.mean()
