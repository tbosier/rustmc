"""Forecasting models take any real numeric array-like and refuse the rest by name."""
import numpy as np
import pytest

import rustmc as rmc

SAMPLING = dict(chains=2, draws=12, warmup=6)


def _prior():
    return rmc.InverseGammaPrior(3.0, 0.5)


def _fits():
    """(name, callable taking observations) for every single-series family."""
    prior = _prior()
    ar_prior = rmc.NormalInverseGammaPrior(np.zeros(2), np.eye(2), 3.0, 0.5)
    return [
        ("local level", lambda y: rmc.BayesianLocalLevel(prior, prior).fit(y, **SAMPLING)),
        ("seasonal", lambda y: rmc.BayesianSeasonalLocalLevel(2, prior, prior, prior).fit(y, **SAMPLING)),
        ("trend", lambda y: rmc.BayesianLocalLinearTrend(prior, prior, prior).fit(y, **SAMPLING)),
        ("AR", lambda y: rmc.BayesianAR(1, ar_prior).fit(y, chains=2, draws=12)),
        ("hurdle", lambda y: rmc.BayesianHurdleLogNormal(prior, prior).fit(y, **SAMPLING)),
    ]


def _layouts(values):
    """The same series as float, integer, list, strided and float32 inputs."""
    base = np.asarray(values)
    doubled = np.repeat(base, 2)
    return {
        "float64": base.astype(float),
        "int64": base.astype(np.int64),
        "uint8": base.astype(np.uint8),
        "list": [int(value) for value in base],
        "strided": doubled[::2].astype(float),
        "float32": base.astype(np.float32),
    }


@pytest.mark.parametrize("name, fit", _fits(), ids=[name for name, _ in _fits()])
def test_numeric_layouts_give_the_same_fit_as_float64(name, fit):
    integers = [1, 2, 1, 3, 2, 3, 3, 4]
    reference = fit(np.asarray(integers, dtype=float))
    samples = reference.get_samples() if hasattr(reference, "get_samples") else reference.get_samples_2d()
    for layout, observations in _layouts(integers).items():
        other = fit(observations)
        other_samples = other.get_samples() if hasattr(other, "get_samples") else other.get_samples_2d()
        for key, values in samples.items():
            np.testing.assert_array_equal(values, other_samples[key], err_msg=f"{name} {layout} {key}")


@pytest.mark.parametrize("name, fit", _fits(), ids=[name for name, _ in _fits()])
@pytest.mark.parametrize("bad, reason", [
    (np.array([True, False, True, True, False, True, True, False]), "real numbers"),
    (np.array(["1", "2", "3", "4", "5", "6", "7", "8"]), "real numbers"),
    (np.array([1 + 1j] * 8), "real numbers"),
    (np.array([1.0, None] * 4, dtype=object), "real numbers"),
    (np.float64(3.0), "one-dimensional"),
    (np.ones((2, 4)), "one-dimensional"),
    ("12345678", "real numbers"),
])
def test_non_real_or_misshaped_observations_are_refused_by_name(name, fit, bad, reason):
    with pytest.raises(ValueError, match=f"observations must .*{reason}") as error:
        fit(bad)
    assert "PyArray" not in str(error.value)


@pytest.mark.parametrize("bad, reason", [
    ([True, 1, 2, 1, 3, 2, 3, 4], "bool"),
    ([True, 0.5, 2, 1, 3, 2, 3, 4], "bool"),
    (np.ma.masked_array([1.0, 1e6, 2, 1, 3, 2, 3, 4], mask=[0, 1, 0, 0, 0, 0, 0, 0]), "masked"),
    (np.arange(1, 9, dtype=np.longdouble) / 3, "long double"),
    (np.array([2**53 + 1, 2, 1, 3, 2, 3, 3, 4], dtype=np.int64), "2\\*\\*53"),
    ([0.5, 2**53 + 1, 1, 3, 2, 3, 3, 4], "2\\*\\*53"),
    (np.arange(8.0).astype(object), "astype\\(float\\)"),
])
def test_values_float64_would_change_are_refused_as_for_model_data(bad, reason):
    # Forecasting arrays and ModelBuilder data share one exact conversion.
    local_level = rmc.LinearGaussianStateSpace(np.eye(1), [1.0], np.eye(1), 1.0, [0.0], [[1.0]])
    with pytest.raises(ValueError, match=f"observations .*{reason}"):
        local_level.filter(bad)
    with pytest.raises(ValueError, match=f"observations .*{reason}"):
        rmc.BayesianLocalLevel(_prior(), _prior()).fit(bad, **SAMPLING)
    with pytest.raises(ValueError, match=f"'y'.*{reason}"):
        rmc.ModelBuilder(data={"y": bad})


def test_values_float64_holds_exactly_are_accepted_in_any_container():
    local_level = rmc.LinearGaussianStateSpace(np.eye(1), [1.0], np.eye(1), 1.0, [0.0], [[1.0]])
    reference = local_level.filter(np.array([1.0, np.nan, 3.0])).filtered_means
    for same in (
        np.ma.masked_array([1.0, np.nan, 3.0], mask=False),
        np.array([1.0, np.nan, 3.0], dtype=np.longdouble),
        (1, np.nan, np.int8(3)),
    ):
        np.testing.assert_array_equal(local_level.filter(same).filtered_means, reference)
    big = local_level.filter(np.array([2**53, 2**60], dtype=np.int64)).filtered_means
    np.testing.assert_array_equal(big, local_level.filter([2.0**53, 2.0**60]).filtered_means)


def test_missing_values_survive_conversion():
    prior = _prior()
    observations = [1.0, np.nan, 2.0, 3.0, np.nan, 4.0]
    fit = rmc.BayesianLocalLevel(prior, prior).fit(observations, **SAMPLING)
    assert (fit.time_count, fit.observed_count) == (6, 4)
    batch = rmc.BayesianLocalLevel(prior, prior).fit_batch(
        [observations, np.array([1, 2, 3, 4])], ["gaps", "ints"], **SAMPLING
    )
    assert batch["gaps"].observed_count == 4
    assert batch["ints"].observed_count == 4


def test_batch_cells_convert_each_row_and_report_bad_rows_per_cell():
    prior = _prior()
    model = rmc.BayesianLocalLevel(prior, prior)
    batch = model.fit_batch(
        [np.arange(8), [True, False, True, False], np.ones((2, 2))],
        ["ints", "bools", "matrix"],
        errors="collect",
        **SAMPLING,
    )
    assert batch["ints"].time_count == 8
    assert "real numbers" in batch.errors["bools"]
    assert "one-dimensional" in batch.errors["matrix"]


def test_state_space_hierarchical_and_priors_accept_integer_arrays():
    model = rmc.LinearGaussianStateSpace(
        np.eye(1, dtype=int), [1], np.eye(1, dtype=int), 1.0, [0], [[1]]
    )
    np.testing.assert_array_equal(
        model.filter(np.arange(5)).filtered_means,
        model.filter(np.arange(5.0)).filtered_means,
    )
    with pytest.raises(ValueError, match="observations must hold real numbers"):
        model.filter(np.array([True, False]))
    with pytest.raises(ValueError, match="transition must be two-dimensional"):
        rmc.LinearGaussianStateSpace([1.0], [1.0], [[1.0]], 1.0, [0.0], [[1.0]])

    hierarchical = rmc.BayesianHierarchicalMean(_prior(), _prior(), _prior())
    fit = hierarchical.fit([[1, 2, 3], np.array([4, 5])], [0, 0], **SAMPLING)
    assert fit.time_counts == [3, 2]
    with pytest.raises(ValueError, match=r"series\[1\] must hold real numbers"):
        hierarchical.fit([[1.0, 2.0], ["a", "b"]], [0, 0], **SAMPLING)

    ar_prior = rmc.NormalInverseGammaPrior([0, 0], [[1, 0], [0, 1]], 3.0, 0.5)
    np.testing.assert_array_equal(ar_prior.coefficient_precision, np.eye(2))
    coefficient_prior = rmc.GaussianCoefficientPrior([0], [[1]])
    np.testing.assert_array_equal(coefficient_prior.covariance, [[1.0]])
    with pytest.raises(ValueError, match="covariance must hold real numbers"):
        rmc.GaussianCoefficientPrior([0.0], [[True]])


def test_structural_glm_and_runoff_accept_integer_arrays():
    level = rmc.StructuralComponent.level("level", rmc.VarianceParameter.inverse_gamma(3.0, 0.5), 0.0, 1.0)
    model = rmc.StructuralModel([level], rmc.VarianceParameter.inverse_gamma(3.0, 0.5))
    integers = model.fit(np.arange(8), **SAMPLING)
    floats = model.fit(np.arange(8.0), **SAMPLING)
    np.testing.assert_array_equal(integers.variance_draws, floats.variance_draws)
    with pytest.raises(ValueError, match="observations must hold real numbers"):
        model.fit(np.ones(8, dtype=bool), **SAMPLING)

    glm = rmc.BayesianDynamicPoisson()
    counts = np.array([[0, 1, 2, 1]])
    from_integers = glm.fit(counts, chains=1, warmup=5, draws=5, seed=3).get_samples()
    from_floats = glm.fit(counts.astype(float), chains=1, warmup=5, draws=5, seed=3).get_samples()
    for name, values in from_floats.items():
        np.testing.assert_array_equal(from_integers[name], values)

    runoff = rmc.DirichletMultinomialRunoff([2, 3, 1])
    triangle = np.array([[8, 3, 1], [4, 5, 3]])
    fit = runoff.fit(triangle, [0, 1], 5, totals=[12, 12], draws=10, chains=2, seed=1)
    assert fit.allocation_samples.shape == (2, 10, 2, 3)
    with pytest.raises(ValueError, match="counts must hold real numbers"):
        runoff.fit(triangle.astype(bool), [0, 1], 5, draws=10, chains=2)
