"""Joint hierarchical mean fitting, ragged series, and coherent rollups."""

import numpy as np
import pytest


def make_model(rmc):
    return rmc.BayesianHierarchicalMean(
        group_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
        program_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
        observation_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
        population_mean_prior=0.0,
        population_variance_prior=100.0,
    )


def ragged_data():
    return [
        np.array([9.5]),
        np.array([9.8, 10.1, np.nan]),
        np.array([10.0, 10.2, 9.9, 10.1]),
        np.array([29.8, 30.2]),
        np.array([30.1, 29.9, 30.0, 30.2, 29.8]),
    ]


def test_joint_ragged_fit_shapes_metadata_and_seed(rustmc_module):
    model = make_model(rustmc_module)
    kwargs = dict(
        series=ragged_data(),
        group_index=[0, 0, 0, 1, 1],
        program_names=["a", "b", "c", "d", "e"],
        group_names=["division-a", "division-b"],
        chains=2,
        draws=80,
        warmup=60,
        seed=17,
    )
    first = model.fit(**kwargs)
    second = model.fit(**kwargs)

    assert first.chains == 2
    assert first.draws == 80
    assert first.program_count == 5
    assert first.group_count == 2
    assert first.time_counts == [1, 3, 4, 2, 5]
    assert first.observed_counts == [1, 2, 4, 2, 5]
    assert first.total_observed_count == 14
    assert first.group_index == [0, 0, 0, 1, 1]
    assert first.program_names == ["a", "b", "c", "d", "e"]
    assert first.group_names == ["division-a", "division-b"]
    assert first.inference_method == "conjugate_gibbs"

    samples = first.get_samples()
    repeated = second.get_samples()
    assert set(samples) == {
        "population_mean",
        "group_variance",
        "group_sd",
        "program_variance",
        "program_sd",
        "observation_variance",
        "observation_sd",
        "group_mean",
        "program_mean",
    }
    for name in (
        "population_mean",
        "group_variance",
        "group_sd",
        "program_variance",
        "program_sd",
        "observation_variance",
        "observation_sd",
    ):
        assert samples[name].shape == (2, 80)
        assert np.isfinite(samples[name]).all()
        np.testing.assert_array_equal(samples[name], repeated[name])
    assert samples["group_mean"].shape == (2, 80, 2)
    assert samples["program_mean"].shape == (2, 80, 5)
    np.testing.assert_array_equal(samples["group_mean"], repeated["group_mean"])
    np.testing.assert_array_equal(samples["program_mean"], repeated["program_mean"])
    np.testing.assert_allclose(samples["group_sd"] ** 2, samples["group_variance"])
    np.testing.assert_allclose(samples["program_sd"] ** 2, samples["program_variance"])
    np.testing.assert_allclose(
        samples["observation_sd"] ** 2, samples["observation_variance"]
    )
    assert not np.array_equal(samples["population_mean"][0], samples["population_mean"][1])
    diagnostics = first.diagnostics()
    by_name = {item["name"]: item for item in diagnostics}
    assert {"population_mean", "group_variance", "program_variance"} <= set(by_name)
    assert np.isfinite(by_name["program_variance"]["r_hat"])
    assert by_name["program_variance"]["ess_bulk"] > 0
    assert "ess_bulk" in first.summary()


def test_diagnostics_warn_when_seeded_chains_do_not_converge(rustmc_module):
    model = rustmc_module.BayesianHierarchicalMean(
        group_variance_prior=rustmc_module.InverseGammaPrior(3.0, 1e-6),
        program_variance_prior=rustmc_module.InverseGammaPrior(3.0, 1e-6),
        observation_variance_prior=rustmc_module.InverseGammaPrior(3.0, 1e-6),
        population_variance_prior=1.0,
    )
    fit = model.fit(
        [
            np.array([10.0]),
            np.array([10.0, 10.0001, 9.9999]),
            np.array([10.0, 10.0002]),
            np.array([10.0]),
            np.array([10.0, 9.9998, 10.0002]),
            np.array([10.0, 10.0001]),
        ],
        [0, 0, 0, 1, 1, 1],
        chains=4,
        draws=250,
        warmup=100,
        seed=101,
    )
    diagnostics = fit.diagnostics()
    assert any(
        item["r_hat"] > 1.01
        or item["ess_bulk"] < 400
        or item["ess_tail"] < 400
        for item in diagnostics
    )
    assert "WARNING:" in fit.summary()


def test_sparse_program_adaptively_shrinks_more(rustmc_module):
    rng = np.random.default_rng(3)
    series = [
        np.array([20.0]),
        rng.normal(20.0, 2.0, 30),
        rng.normal(10.0, 2.0, 30),
        rng.normal(10.0, 2.0, 30),
        rng.normal(10.0, 2.0, 30),
        rng.normal(30.0, 2.0, 30),
        rng.normal(30.0, 2.0, 30),
        rng.normal(30.0, 2.0, 30),
    ]
    fit = make_model(rustmc_module).fit(
        series,
        [0, 0, 0, 0, 0, 1, 1, 1],
        chains=2,
        draws=250,
        warmup=200,
        seed=91,
    )
    posterior_means = fit.get_samples()["program_mean"].mean(axis=(0, 1))
    sparse_shift = 20.0 - posterior_means[0]
    dense_shift = 20.0 - posterior_means[1]
    assert 10.0 < posterior_means[0] < 20.0
    assert abs(sparse_shift) > abs(dense_shift) + 0.15


def test_forecast_alignment_intervals_and_rollups(rustmc_module):
    fit = make_model(rustmc_module).fit(
        ragged_data(),
        [0, 0, 0, 1, 1],
        chains=2,
        draws=100,
        warmup=75,
        seed=19,
    )
    forecast = fit.forecast(steps=4, seed=23)
    repeated = fit.forecast(steps=4, seed=23)
    state = forecast.state_samples
    observations = forecast.observation_samples

    assert state.shape == (2, 100, 5, 4)
    assert observations.shape == (2, 100, 5, 4)
    assert observations[:, :, 0, :].shape == (2, 100, 4)
    np.testing.assert_array_equal(observations, repeated.observation_samples)
    np.testing.assert_array_equal(state[..., 0], fit.get_samples()["program_mean"])
    for step in range(1, 4):
        np.testing.assert_array_equal(state[..., step], state[..., 0])
    np.testing.assert_allclose(forecast.state_mean, state.mean(axis=(0, 1)))
    np.testing.assert_allclose(
        forecast.observation_mean, observations.mean(axis=(0, 1))
    )
    np.testing.assert_allclose(
        forecast.observation_quantile(0.5),
        np.quantile(observations, 0.5, axis=(0, 1)),
    )
    lower, upper = forecast.interval(0.95)
    np.testing.assert_allclose(lower, np.quantile(observations, 0.025, axis=(0, 1)))
    np.testing.assert_allclose(upper, np.quantile(observations, 0.975, axis=(0, 1)))

    company = observations.sum(axis=2)
    division_a = observations[:, :, np.array(fit.group_index) == 0, :].sum(axis=2)
    division_b = observations[:, :, np.array(fit.group_index) == 1, :].sum(axis=2)
    assert company.shape == (2, 100, 4)
    np.testing.assert_allclose(company, division_a + division_b)
    np.testing.assert_allclose(forecast.total_observation_samples, company)
    np.testing.assert_allclose(forecast.group_observation_samples[:, :, 0, :], division_a)
    np.testing.assert_allclose(forecast.group_observation_samples[:, :, 1, :], division_b)
    np.testing.assert_allclose(company.mean(axis=(0, 1)), forecast.observation_mean.sum(axis=0))
    assert forecast.uncertainty_kind == "parameter_integrated_posterior_predictive"
    assert forecast.interval_kind == "pointwise_equal_tailed"


def test_hierarchical_validation(rustmc_module):
    model = make_model(rustmc_module)
    with pytest.raises(rustmc_module.InferenceError):
        model.fit([np.array([np.nan])], [0], chains=1, draws=2, warmup=1)
    invalid = [
        ([], []),
        ([np.array([])], [0]),
        ([np.array([np.nan])], [0]),
        ([np.array([1.0, np.inf])], [0]),
        ([np.array([1.0])], []),
        ([np.array([1.0]), np.array([2.0])], [0, 2]),
    ]
    for series, groups in invalid:
        with pytest.raises(ValueError):
            model.fit(series, groups, chains=1, draws=2, warmup=1)

    with pytest.raises(rustmc_module.InferenceError, match="program_names"):
        model.fit([np.array([1.0])], [0], program_names=[])
    with pytest.raises(rustmc_module.InferenceError, match="unique"):
        model.fit(
            [np.array([1.0]), np.array([2.0])],
            [0, 0],
            program_names=["same", "same"],
        )
    with pytest.raises(ValueError):
        model.fit([np.array([1.0])], [0], chains=0, draws=2, warmup=1)
    fit = model.fit([np.array([1.0])], [0], chains=1, draws=2, warmup=1)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    with pytest.raises(rustmc_module.InferenceError, match="safety limit"):
        fit.forecast(30_000_000)
    forecast = fit.forecast(2)
    for level in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            forecast.interval(level)
