"""Data dictionaries are converted to float64 only when that is exact.

Every graph-model entry point (ModelBuilder(data=...), sample, bind,
sample_batch, predict, sample_prior_predictive) reads data through one
converter. It used to coerce anything NumPy could cast: a scalar became a
length-1 vector, booleans became 0/1, strings such as '1.5' parsed as numbers,
complex values lost their imaginary part, and int64 values above 2**53 were
rounded.
"""
import re

import numpy as np
import pytest
import rustmc


def compiled_regression():
    b = rustmc.ModelBuilder()
    a = b.normal_prior("a", 0.0, 1.0)
    beta = b.normal_prior("beta", 0.0, 1.0)
    b.normal_likelihood("obs", a + beta * "x", 1.0, "y")
    return b.compile()


def compiled_matrix():
    b = rustmc.ModelBuilder()
    beta = b.vector_normal_prior("beta", 2)
    b.normal_likelihood("obs", beta @ "X", 1.0, "y")
    return b.compile()


X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Y = np.array([0.5, -1.0, 2.0])


@pytest.mark.parametrize(
    "value, message",
    [
        (3.0, "scalar"),
        (np.float64(3.0), "scalar"),
        (np.array(3.0), "scalar"),
        (np.zeros((2, 2, 2)), "3 dimensions"),
        (np.array([True, False, True]), "bool"),
        ([True, False, True], "bool"),
        (["1.5", "2.5", "3.5"], "str"),
        (np.array([b"1", b"2", b"3"]), "bytes"),
        (np.array([1.0, 2.0, 3.0], dtype=object), "object"),
        (np.array([1 + 1j, 2.0, 3.0]), "complex"),
        (np.array([2**53 + 1, 0, 1], dtype=np.int64), "2**53"),
        (np.array([2**64 - 1, 0, 1], dtype=np.uint64), "2**53"),
    ],
)
def test_inexact_or_non_numeric_data_is_refused_naming_the_key(value, message):
    message = re.escape(message)
    with pytest.raises(ValueError, match=rf"'x'.*{message}"):
        rustmc.ModelBuilder(data={"x": value, "y": Y})
    compiled = compiled_regression()
    with pytest.raises(ValueError, match="'x'"):
        compiled.bind({"x": value, "y": Y})


def test_complex_data_is_refused_without_a_numpy_warning(recwarn):
    with pytest.raises(ValueError, match="complex"):
        compiled_regression().bind({"x": np.array([1j, 2.0, 3.0]), "y": Y})
    assert not recwarn.list


@pytest.mark.parametrize(
    "x",
    [
        [1.0, 2.0, 3.0],
        [1, 2, 3],
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 0.0, 2.0, 0.0, 3.0])[::2],
        np.array([2**53, 2, 3], dtype=np.int64) - np.array([2**53 - 1, 0, 0]),
    ],
)
def test_real_numeric_forms_bind_as_their_exact_values(x):
    compiled = compiled_regression()
    reference = compiled.log_density({"x": np.array([1.0, 2.0, 3.0]), "y": Y}, [0.3, -0.2])
    value = compiled.log_density({"x": x, "y": Y}, [0.3, -0.2])
    assert value[0] == reference[0]
    np.testing.assert_array_equal(value[1], reference[1])


def test_integers_up_to_2_53_are_exact():
    compiled = compiled_regression()
    big = np.array([2**53, -(2**53), 1], dtype=np.int64)
    bound = compiled.bind({"x": big, "y": Y})
    assert bound.n_obs == 3


def test_fortran_ordered_and_strided_matrices_keep_their_rows():
    compiled = compiled_matrix()
    reference = compiled.log_density({"X": X, "y": Y}, [0.4, -0.7])
    for layout in (np.asfortranarray(X), np.repeat(X, 2, axis=1)[:, ::2], X.tolist()):
        value = compiled.log_density({"X": layout, "y": Y}, [0.4, -0.7])
        assert value[0] == reference[0]


def test_scalar_prediction_data_is_refused():
    compiled = compiled_regression()
    fit = compiled.sample({"x": np.array([1.0, 2.0, 3.0]), "y": Y}, chains=1, draws=5,
                          warmup=5, seed=1, show_progress=False)
    with pytest.raises(ValueError, match="'x'.*scalar"):
        fit.predict({"x": 2.0})
