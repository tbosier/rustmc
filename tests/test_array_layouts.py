"""The core's row-major matrix contract must not depend on NumPy storage."""

import numpy as np
import pytest

import rustmc


MATRIX = np.arange(1.0, 7.0).reshape(3, 2)
OBSERVED = np.array([0.0, 1.0, -1.0])


def layout(values, kind):
    if kind == "c":
        return values.copy(order="C")
    if kind == "fortran":
        return values.copy(order="F")
    if kind == "transpose":
        return values.T.copy(order="C").T
    if kind == "strided":
        storage = np.zeros(tuple(2 * size for size in values.shape))
        view = storage[tuple(slice(None, None, 2) for _ in values.shape)]
        view[...] = values
        return view
    if kind == "reversed":
        return values[::-1].copy()[::-1]
    if kind == "integer":
        return values.astype(np.int64)
    return values.tolist()


def builder(data):
    model = rustmc.ModelBuilder(data)
    beta = model.vector_normal_prior("beta", 2, 0.0, 1.0)
    model.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return model


@pytest.mark.parametrize("kind", ["c", "fortran", "transpose", "strided", "reversed", "integer", "list"])
def test_layout_preserves_construction_binding_prediction_and_batch(kind):
    reference_data = {"X": MATRIX, "y": OBSERVED}
    data = {"X": layout(MATRIX, kind), "y": layout(OBSERVED, kind)}
    model = builder(data)
    compiled = model.compile()
    q = np.array([0.1, 0.2])
    residual = OBSERVED - MATRIX @ q
    expected_lp = -0.5 * (q @ q + residual @ residual + 5 * np.log(2 * np.pi))
    expected_grad = -q + MATRIX.T @ residual
    reference = builder(reference_data).compile()
    for candidate, binding in [(compiled, {}), (reference, data), (reference, reference.bind(data))]:
        lp, grad = candidate.log_density(binding, q)
        assert lp == pytest.approx(expected_lp, abs=1e-12)
        np.testing.assert_allclose(grad, expected_grad, atol=1e-12)

    # The ModelSpec / prior-predictive entrypoint also shares the parser.
    prior = rustmc.sample_prior_predictive(model.build(), n_samples=3, seed=41)
    prior_reference = rustmc.sample_prior_predictive(builder(reference_data).build(), n_samples=3, seed=41)
    np.testing.assert_array_equal(prior["obs"], prior_reference["obs"])

    options = dict(chains=1, draws=4, warmup=5, seed=17, show_progress=False)
    fit = reference.sample(reference_data, **options)
    prediction = fit.predict({"X": layout(MATRIX, kind)}, expected=True)["obs"]
    samples = fit.get_samples_2d()
    coefficients = np.stack([samples["beta[0]"], samples["beta[1]"]], axis=-1)
    np.testing.assert_allclose(prediction, coefficients @ MATRIX.T, atol=1e-12)

    batch_reference = reference.sample_batch([reference_data], ids=["cell"], **options)
    for batch in [
        reference.sample_batch([data], ids=["cell"], **options),
        reference.sample_batch([{"y": data["y"]}], shared={"X": data["X"]}, ids=["cell"], **options),
    ]:
        for name, expected in batch_reference.get("cell").get_samples_2d().items():
            np.testing.assert_array_equal(batch.get("cell").get_samples_2d()[name], expected)
