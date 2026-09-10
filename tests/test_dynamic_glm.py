import json

import numpy as np
import pytest

import rustmc


@pytest.mark.parametrize("name", [
    "BayesianDynamicPoisson", "BayesianDynamicNegativeBinomial",
    "BayesianDynamicHurdleLogNormal", "BayesianHierarchicalDynamicRegression",
])
def test_joint_forecast_protocol_and_artifact(name):
    model = getattr(rustmc, name)(process_sd=.1, shared_process_sd=.05)
    y = [[0., 1., np.nan, 2.], [2., 1., 0., 3.]]
    fit = model.fit(y, chains=2, warmup=30, draws=30, seed=11)
    f = fit.forecast(3, seed=92)
    assert f.observation_samples.shape == (2, 30, 2, 3)
    assert (f.chains, f.draws, f.groups, f.steps) == (2, 30, 2, 3)
    np.testing.assert_allclose(f.mean, f.mean_samples.mean(axis=(0, 1)))
    np.testing.assert_allclose(f.observation_mean, f.observation_samples.mean(axis=(0, 1)))
    np.testing.assert_allclose(f.interval(), np.quantile(f.observation_samples, [.025, .975], axis=(0, 1)))
    assert fit.state_samples().shape == (2, 30, 2, 4)
    np.testing.assert_array_equal(f.aggregate_observation_samples,
                                  f.observation_samples.sum(axis=2))
    assert len(fit.diagnostics()) == len(fit.param_names)
    assert all(x.shape == (60,) for x in fit.get_samples().values())
    assert fit.sampler_stats["sampler"] == "block_elliptical_slice"
    assert fit.sampler_stats["fixed_parameters"]["process_sd"] == .1
    restored = rustmc.DynamicGLMFit.from_json(fit.to_json())
    np.testing.assert_array_equal(f.observation_samples,
                                  restored.forecast(3, seed=92).observation_samples)
    prior = model.prior_predictive(2, groups=2, draws=10, seed=12)
    assert prior.observation_samples.shape == (1, 10, 2, 2)
    if name == "BayesianDynamicHurdleLogNormal":
        assert fit.state_samples(component=1).shape == (2, 30, 2, 4)
        np.testing.assert_allclose(f.mean_samples,
                                  f.occurrence_samples * f.positive_mean_samples)
    else:
        assert f.occurrence_samples is None


def test_exposure_design_and_missing_validation():
    model = rustmc.BayesianDynamicPoisson()
    fit = model.fit([[0., 1., np.nan]], exog=[[[1.], [2.], [3.]]],
                    exposure=[[0., 1., 1.]], chains=1, warmup=10, draws=20)
    with pytest.raises(ValueError):
        fit.forecast(2)
    with pytest.raises(ValueError):
        fit.forecast(2, exog=[[[1., 2.], [3., 4.]]])
    future = fit.forecast(2, exog=[[[4.], [5.]]], exposure=[[0., 0.]])
    assert np.all(future.observation_samples == 0)
    with pytest.raises(ValueError):
        model.fit([[1.]], exposure=[[0.]])
    with pytest.raises(ValueError):
        rustmc.BayesianDynamicHurdleLogNormal().fit([[0.]], exposure=[[1.]])
    artifact = json.loads(fit.to_json())
    artifact["posterior"]["groups"] = 2
    with pytest.raises(ValueError):
        rustmc.DynamicGLMFit.from_json(json.dumps(artifact))


def test_single_series_requires_explicit_group_axis():
    with pytest.raises(TypeError):
        rustmc.BayesianDynamicPoisson().fit([0., 1., 2.])


def test_arviz_export_when_available():
    pytest.importorskip("arviz")
    fit = rustmc.BayesianDynamicPoisson().fit([[0., 1., 2.]], chains=2, warmup=10, draws=20)
    exported = fit.to_arviz()
    assert "posterior" in exported
