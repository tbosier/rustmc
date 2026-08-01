import numpy as np
import pytest

import rustmc


def _compiled_model():
    builder = rustmc.ModelBuilder()
    beta = builder.normal_prior("beta", 0.0, 1.0)
    builder.normal_likelihood("obs", beta * "x", 1.0, "y")
    return builder, builder.compile()


def _batch_fit(compiled):
    return compiled.sample_batch(
        [{"x": np.ones(2), "y": np.ones(2)}],
        ids=["series-a"],
        chains=1,
        draws=1,
        warmup=1,
        sampler="hmc",
        num_leapfrog_steps=1,
        show_progress=False,
    )


def test_context_managers_return_the_same_object_and_remain_reusable():
    builder, compiled = _compiled_model()
    bound = compiled.bind({"x": np.ones(2), "y": np.ones(2)}, id="series-a")
    batch_fit = _batch_fit(compiled)

    for managed in (builder, compiled, bound, batch_fit):
        with managed as entered:
            assert entered is managed

    # A context block is optional lexical scoping, not a close/dispose lifecycle.
    assert builder.compile().param_names == ["beta"]
    assert compiled.param_names == ["beta"]
    assert bound.id == "series-a"
    assert batch_fit.ids == ["series-a"]
    assert len(batch_fit) == 1


def test_context_managers_never_suppress_exceptions():
    builder, compiled = _compiled_model()
    bound = compiled.bind({"x": np.ones(2), "y": np.ones(2)})
    batch_fit = _batch_fit(compiled)

    for managed in (builder, compiled, bound, batch_fit):
        with pytest.raises(RuntimeError, match="sentinel"):
            with managed:
                raise RuntimeError("sentinel")


def test_builder_context_is_not_an_ambient_current_model():
    outer = rustmc.ModelBuilder()
    with outer as explicit_outer:
        explicit_outer.normal_prior("outer", 0.0, 1.0)
        with rustmc.ModelBuilder() as explicit_inner:
            explicit_inner.normal_prior("inner", 0.0, 1.0)
        explicit_outer.normal_prior("outer_two", 0.0, 1.0)

    assert outer.compile().param_names == ["outer", "outer_two"]
    assert explicit_inner.compile().param_names == ["inner"]
