"""A bare ``"key"`` string means "the data column with this key", everywhere.

``ParamRef.__mul__`` has always special-cased a string right-hand side, so
``beta * "x"`` is the idiom used throughout the README.  These tests pin down
that the same shorthand is accepted by *every* expression operand slot, on both
sides of every operator, so the idiom does not die the moment an expression is
one node deep (``beta["group"] * "x"``, ``(a + b) * "x"``, ``a.sqrt() * "x"``).

They also pin down the two things that must not regress while that shorthand
spreads:

* ``beta * "x"`` must still compile to the fused ``ParamTimesData`` term that
  ``model.rs::try_extract_linear`` turns into a single ``FusedLinearMu`` op;
* an unknown data key must still be rejected, with the key named.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def rustmc(rustmc_module):
    return rustmc_module


# ── data ──────────────────────────────────────────────────────────────────


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(20260918)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    group = rng.integers(0, 3, size=n).astype(float)
    y = 0.5 + 1.25 * x + rng.normal(scale=0.2, size=n)
    return {"x": x, "z": z, "group": group, "y": y}


def only(data, *keys):
    """The subset of ``data`` a model actually reads.

    ``ModelBuilder(data)`` keeps every bound column, and binding later rejects
    columns the compiled model has no slot for, so tests that call
    ``log_density`` must hand the builder exactly what the model uses.
    """
    return {key: data[key] for key in keys}


def _mu_expr_json(builder):
    """The serialised ``mu_expr`` of the builder's single likelihood."""
    document = json.loads(builder.compile().to_json())
    return document["definition"]["likelihoods"][0]["mu_expr"]


def _finite_difference_gradient(compiled, position, step=1e-6):
    gradient = np.empty_like(position)
    for index in range(position.size):
        up, down = position.copy(), position.copy()
        up[index] += step
        down[index] -= step
        gradient[index] = (
            compiled.log_density({}, up)[0] - compiled.log_density({}, down)[0]
        ) / (2 * step)
    return gradient


# ── the fused fast path must survive ──────────────────────────────────────


def test_param_times_string_still_compiles_to_the_fused_linear_term(rustmc, toy_data):
    """``beta * "x"`` keeps producing ``ParamTimesData``, the fusible form.

    ``try_extract_linear`` (rust_core/src/model.rs) fuses a sum of
    ``ParamTimesData`` terms into one ``FusedLinearMu`` op and refuses to fuse
    anything else -- notably ``Binary(Mul, Param, Data)``.  ``to_json``
    serialises the *model definition*, not the compiled graph, so what this
    pins is the input to that decision: the expression handed to
    ``build_mu_expr`` is still in the only shape that fuses.  Whether fusion
    then fires is ``try_extract_linear``'s own contract, covered on the Rust
    side.  The regression this guards is the realistic one -- the string
    shorthand quietly degrading to the unfused ``Binary`` form.
    """
    builder = rustmc.ModelBuilder(toy_data)
    alpha = builder.normal_prior("alpha", 0.0, 10.0)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    builder.normal_likelihood("obs", alpha + beta * "x", 1.0, "y")

    assert _mu_expr_json(builder) == {
        "Add": [{"Param": "alpha"}, {"ParamTimesData": {"param_name": "beta", "data_key": "x"}}]
    }


def test_string_times_param_is_also_fused(rustmc, toy_data):
    """The reflected form ``"x" * beta`` fuses identically."""
    builder = rustmc.ModelBuilder(toy_data)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    builder.normal_likelihood("obs", "x" * beta, 1.0, "y")

    assert _mu_expr_json(builder) == {
        "ParamTimesData": {"param_name": "beta", "data_key": "x"}
    }


def test_multi_term_linear_predictor_stays_entirely_fusible(rustmc, toy_data):
    builder = rustmc.ModelBuilder(toy_data)
    alpha = builder.normal_prior("alpha", 0.0, 10.0)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    gamma = builder.normal_prior("gamma", 0.0, 10.0)
    builder.normal_likelihood("obs", alpha + beta * "x" + gamma * "z", 1.0, "y")

    serialised = json.dumps(_mu_expr_json(builder))
    assert "ParamTimesData" in serialised
    assert "Binary" not in serialised
    assert '"Data"' not in serialised


def test_explicit_builder_data_is_the_unfused_form(rustmc, toy_data):
    """Contrast case: ``builder.data("x")`` deliberately does *not* fuse.

    This is the behaviour that existed before bare strings were accepted in
    nested positions, and it still holds -- which is what makes the assertion
    above meaningful rather than vacuous.
    """
    builder = rustmc.ModelBuilder(toy_data)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    builder.normal_likelihood("obs", beta * builder.data("x"), 1.0, "y")

    assert _mu_expr_json(builder) == {
        "Binary": ["Mul", {"Param": "beta"}, {"Data": "x"}]
    }


def test_bare_string_and_builder_data_are_the_same_expression(rustmc, toy_data):
    """A bare ``"x"`` in a nested slot is exactly ``builder.data("x")``."""
    bare = rustmc.ModelBuilder(toy_data)
    alpha = bare.normal_prior("alpha", 0.0, 10.0)
    bare.normal_likelihood("obs", (alpha + 1.0) * "x", 1.0, "y")

    explicit = rustmc.ModelBuilder(toy_data)
    alpha = explicit.normal_prior("alpha", 0.0, 10.0)
    explicit.normal_likelihood("obs", (alpha + 1.0) * explicit.data("x"), 1.0, "y")

    assert _mu_expr_json(bare) == _mu_expr_json(explicit)
    assert bare.compile().dimensions == explicit.compile().dimensions


# ── the case that used to die: beta["group"] * "x" ────────────────────────


def test_gather_times_string_builds_and_compiles(rustmc, toy_data):
    builder = rustmc.ModelBuilder(toy_data)
    builder.normal_prior("slope", 0.0, 10.0)
    slope = builder.vector_normal_prior("slope_vec", 3, 0.0, 10.0)
    builder.normal_likelihood("obs", slope["group"] * "x", 1.0, "y")

    assert _mu_expr_json(builder) == {
        "Binary": [
            "Mul",
            {"Gather": {"param_name": "slope_vec", "data_key": "group"}},
            {"Data": "x"},
        ]
    }
    compiled = builder.compile()
    assert set(compiled.param_names) == {
        "slope",
        "slope_vec[0]",
        "slope_vec[1]",
        "slope_vec[2]",
    }
    assert "x" in compiled.required_keys and "group" in compiled.required_keys


def test_gather_times_string_gradient_matches_finite_differences(rustmc, toy_data):
    builder = rustmc.ModelBuilder(only(toy_data, "x", "group", "y"))
    slope = builder.vector_normal_prior("slope_vec", 3, 0.0, 10.0)
    intercept = builder.normal_prior("intercept", 0.0, 10.0)
    builder.normal_likelihood("obs", intercept + slope["group"] * "x", 0.5, "y")
    compiled = builder.compile()

    rng = np.random.default_rng(7)
    for _ in range(3):
        position = rng.normal(size=len(compiled.param_names))
        _, analytic = compiled.log_density({}, position)
        np.testing.assert_allclose(
            analytic, _finite_difference_gradient(compiled, position), rtol=2e-5, atol=2e-5
        )


def test_unary_result_times_string(rustmc, toy_data):
    builder = rustmc.ModelBuilder(only(toy_data, "x", "y"))
    scale = builder.half_normal_prior("scale", 1.0)
    builder.normal_likelihood("obs", scale.sqrt() * "x", 1.0, "y")
    compiled = builder.compile()
    density, gradient = compiled.log_density({}, np.array([0.3]))
    assert np.isfinite(density)
    assert np.isfinite(gradient).all()


def test_sum_of_expressions_times_string(rustmc, toy_data):
    builder = rustmc.ModelBuilder(only(toy_data, "x", "y"))
    a = builder.normal_prior("a", 0.0, 10.0)
    b = builder.normal_prior("b", 0.0, 10.0)
    builder.normal_likelihood("obs", (a + b) * "x", 1.0, "y")
    compiled = builder.compile()
    _, gradient = compiled.log_density({}, np.array([0.2, -0.1]))
    np.testing.assert_allclose(
        gradient,
        _finite_difference_gradient(compiled, np.array([0.2, -0.1])),
        rtol=2e-5,
        atol=2e-5,
    )


# ── the full operator / side matrix ───────────────────────────────────────

_OPERATORS = {
    "mul": lambda left, right: left * right,
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "truediv": lambda left, right: left / right,
    "pow": lambda left, right: left**right,
}


def _operands(builder):
    """One of each expression-carrying type, plus the bare string."""
    param = builder.normal_prior("p", 0.0, 10.0)
    return {
        "ParamRef": param,
        "Expr": param + 0.0,
        "Gather": builder.vector_normal_prior("v", 3, 0.0, 10.0)["group"],
    }


_SERIALISED_OP = {"mul": "Mul", "add": None, "sub": "Sub", "truediv": "Div", "pow": "Pow"}


@pytest.mark.parametrize("operator", sorted(_OPERATORS))
@pytest.mark.parametrize("kind", ["ParamRef", "Expr", "Gather"])
@pytest.mark.parametrize("string_on_left", [False, True])
def test_every_operator_accepts_a_bare_data_key_on_either_side(
    rustmc, toy_data, operator, kind, string_on_left
):
    builder = rustmc.ModelBuilder(toy_data)
    operand = _operands(builder)[kind]
    apply = _OPERATORS[operator]
    expression = apply("x", operand) if string_on_left else apply(operand, "x")

    assert type(expression).__name__ == "Expr"
    builder.normal_likelihood("obs", expression, 1.0, "y")
    compiled = builder.compile()
    assert "x" in compiled.required_keys

    tree = _mu_expr_json(builder)
    if operator == "mul" and kind == "ParamRef":
        # The fused form carries no operand order to inspect.
        assert tree == {"ParamTimesData": {"param_name": "p", "data_key": "x"}}
        return
    key = "Add" if operator == "add" else "Binary"
    assert key in tree, tree
    operands = tree[key] if operator == "add" else tree[key][1:]
    if operator != "add":
        assert tree["Binary"][0] == _SERIALISED_OP[operator]

    if operator in {"add", "mul"}:
        # `+` and `*` commute, and the reflected dunders deliberately do not
        # swap their operands, so only presence is meaningful here.
        assert {"Data": "x"} in operands, tree
        return
    # `-`, `/` and `**` do not commute: a reflected dunder that forgot to swap
    # would still produce a compiling Expr, so pin the side the column lands on.
    data_side = 0 if string_on_left else 1
    assert operands[data_side] == {"Data": "x"}, (
        f"data column landed on the wrong side for {operator} "
        f"with the string on the {'left' if string_on_left else 'right'}: {tree}"
    )
    assert operands[1 - data_side] != {"Data": "x"}


def test_bare_string_is_accepted_by_deterministic(rustmc, toy_data):
    builder = rustmc.ModelBuilder(toy_data)
    alpha = builder.normal_prior("alpha", 0.0, 10.0)
    builder.deterministic("raw", "z")
    builder.deterministic("scaled", alpha * "z" + 1.0)
    builder.potential("penalty", (alpha * "x").sum())
    builder.normal_likelihood("obs", alpha, 1.0, "y")
    required = builder.compile().required_keys
    assert "x" in required and "z" in required


def test_bare_string_potential_is_rejected_as_non_scalar(rustmc, toy_data):
    """A data column is a vector, and `potential` takes a scalar.

    This is the one operand slot where a bare string is accepted by
    `extract_expr` and then refused -- and the refusal names the real reason
    rather than claiming a string is not an expression.
    """
    builder = rustmc.ModelBuilder(toy_data)
    with pytest.raises(ValueError, match="scalar"):
        builder.potential("penalty", "x")


# ── errors stay loud and name the real problem ────────────────────────────


def test_unknown_data_key_in_a_nested_expression_is_rejected_by_name(rustmc, toy_data):
    builder = rustmc.ModelBuilder(toy_data)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    slope = builder.vector_normal_prior("slope_vec", 3, 0.0, 10.0)
    with pytest.raises(ValueError, match="no_such_column"):
        builder.normal_likelihood("obs", slope["group"] * "no_such_column", 1.0, "y")
    with pytest.raises(ValueError, match="no_such_column"):
        builder.normal_likelihood("obs2", (beta + 1.0) * "no_such_column", 1.0, "y")


def test_unknown_data_key_is_rejected_at_compile_time_without_bound_data(rustmc, toy_data):
    builder = rustmc.ModelBuilder()
    beta = builder.normal_prior("beta", 0.0, 10.0)
    builder.normal_likelihood("obs", (beta + 1.0) * "no_such_column", 1.0, "y")
    with pytest.raises(ValueError, match="no_such_column"):
        builder.compile().bind(toy_data)


def test_any_string_is_a_key_including_odd_ones(rustmc):
    """No key is special-cased on the Python side.

    ``parse_data_dict`` accepts whatever string a dict uses, so the DSL must
    not reject keys of its own accord; an unknown key is caught by name later,
    which is the check that actually protects the user.
    """
    n = 6
    data = {"": np.arange(n, dtype=float), "y": np.zeros(n)}
    builder = rustmc.ModelBuilder(data)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    builder.normal_likelihood("obs", (beta + 1.0) * "", 1.0, "y")
    assert "" in builder.compile().required_keys


def test_non_expression_operand_still_reports_what_is_accepted(rustmc, toy_data):
    builder = rustmc.ModelBuilder(toy_data)
    beta = builder.normal_prior("beta", 0.0, 10.0)
    with pytest.raises(ValueError) as caught:
        _ = (beta + 1.0) * {"not": "an expression"}
    message = str(caught.value)
    assert "data key" in message and "expression" in message


def test_vector_parameter_arithmetic_names_the_actual_problem(rustmc, toy_data):
    """A whole vector parameter has no scalar meaning -- say so, don't guess."""
    builder = rustmc.ModelBuilder(toy_data)
    slope = builder.vector_normal_prior("slope_vec", 3, 0.0, 10.0)
    for bad in (
        lambda: slope * "x",
        lambda: slope + 1.0,
        lambda: 1.0 - slope,
        lambda: slope / 2.0,
        lambda: slope**2.0,
        lambda: "x" * slope,
        lambda: -slope,
    ):
        with pytest.raises(rustmc.ParameterError) as caught:
            bad()
        message = str(caught.value)
        assert "slope_vec" in message
        assert "[" in message and "@" in message  # points at __getitem__ / __matmul__


def test_vector_parameter_defers_to_foreign_types(rustmc, toy_data):
    """Rejecting a DSL operand must not break Python's operator protocol.

    The rejecting dunders return ``NotImplemented`` for anything that is not a
    DSL operand, so a third-party type still gets its reflected method and
    behaves as it did before those dunders existed.
    """
    builder = rustmc.ModelBuilder(toy_data)
    slope = builder.vector_normal_prior("slope_vec", 3, 0.0, 10.0)

    class Absorbing:
        def __radd__(self, other):
            return "absorbed"

        def __add__(self, other):
            return "absorbed"

    assert slope + Absorbing() == "absorbed"
    assert Absorbing() + slope == "absorbed"
    # NumPy still gets to apply its own object-array semantics: with nothing to
    # broadcast against there is no element to reject, and the result is an
    # array rather than an exception raised out of the vector parameter.
    assert isinstance(slope + np.array([]), np.ndarray)


def test_mixing_models_is_still_caught_through_a_bare_string(rustmc, toy_data):
    first = rustmc.ModelBuilder(toy_data)
    second = rustmc.ModelBuilder(toy_data)
    a = first.normal_prior("a", 0.0, 1.0)
    b = second.normal_prior("b", 0.0, 1.0)
    with pytest.raises(ValueError, match="different models"):
        _ = (a * "x") + (b * "x")
    with pytest.raises(ValueError):
        first.normal_likelihood("obs", b * "x", 1.0, "y")


# ── end-to-end: random slopes recover their generating values ─────────────


@pytest.mark.slow
def test_random_slopes_recover_known_per_group_slopes(rustmc):
    rng = np.random.default_rng(11)
    true_slopes = np.array([-1.5, 0.75, 2.25])
    n_per_group = 400
    group = np.repeat(np.arange(3), n_per_group).astype(float)
    x = rng.normal(size=group.size)
    y = 0.5 + true_slopes[group.astype(int)] * x + rng.normal(scale=0.3, size=group.size)

    builder = rustmc.ModelBuilder({"x": x, "group": group, "y": y})
    intercept = builder.normal_prior("intercept", 0.0, 5.0)
    slope = builder.vector_normal_prior("slope", 3, 0.0, 5.0)
    sigma = builder.half_normal_prior("sigma", 1.0)
    builder.normal_likelihood("obs", intercept + slope["group"] * "x", sigma, "y")

    fit = rustmc.sample(
        builder.build(), chains=2, draws=600, warmup=600, seed=3, show_progress=False
    )
    means = fit.mean()
    for index, expected in enumerate(true_slopes):
        assert means[f"slope[{index}]"] == pytest.approx(expected, abs=0.05)
    assert means["intercept"] == pytest.approx(0.5, abs=0.05)
