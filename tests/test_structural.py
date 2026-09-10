"""Composable forecasting integration and public-array contracts."""
import json
import numpy as np
import pytest
import rustmc as mc


def test_composition_joint_paths_and_persistence():
    V, C = mc.VarianceParameter, mc.StructuralComponent
    model = mc.StructuralModel([
        C.trend("trend", V.inverse_gamma(3, .1), V.fixed(.002), [1., 0.], [[1., 0.], [0., .1]], damping=.9),
        C.seasonal("weekly", 7.25, 2, V.fixed(0), .2),
        C.regression("promotion", [0.], [[2.]]),
        C.regression("price", [0.], [[2.]], innovations=[V.fixed(.01)]),
        C.ar("residual", [.3], V.fixed(.03), [0.], [[.1]]),
    ], V.inverse_gamma(3, .2), student_df=5.)
    x = np.column_stack([np.linspace(0, 1, 12), np.linspace(1, 0, 12)])
    fit = model.fit(np.linspace(1, 2, 12), exog=x, chains=2, draws=8, warmup=5, store_states=True)
    assert fit.states.shape == (2, 8, 13, 9)
    assert fit.historical_components.shape == (2, 8, 12, 5)
    assert fit.variance_draws.shape == (2, 8, 10)
    assert (fit.chains, fit.draws) == (2, 8)
    assert len(fit.diagnostics()) == len(fit.param_names) == 19
    assert all(x.shape == (2, 8) for x in fit.get_samples_2d().values())
    assert all(x.shape == (16,) for x in fit.get_samples().values())
    assert fit.sampler_stats["sampler"] == "gibbs_ffbs_student_t"
    assert "Gamma precision" in fit.summary()
    future = np.ones((3, 2))
    paths = fit.forecast(3, exog=future, seed=5)
    assert paths.state_paths.shape == (2, 8, 3, 9)
    assert (paths.chains, paths.draws, paths.steps) == (2, 8, 3)
    np.testing.assert_array_equal(paths.observation_samples, paths.observation_paths)
    np.testing.assert_array_equal(paths.mean_samples, paths.mean_paths)
    np.testing.assert_allclose(paths.component_paths.sum(-1), paths.mean_paths)
    np.testing.assert_allclose(paths.observation_paths.cumsum(-1), paths.cumulative_observation_paths)
    for restored in [mc.StructuralFit.from_json(fit.to_json())]:
        np.testing.assert_array_equal(paths.observation_paths, restored.forecast(3, exog=future, seed=5).observation_paths)
    loaded_model = mc.StructuralModel.from_json(model.to_json())
    np.testing.assert_array_equal(model.prior_predict(3, exog=future, draws=7).observation_paths,
                                  loaded_model.prior_predict(3, exog=future, draws=7).observation_paths)
    broken = json.loads(fit.to_json())
    broken["version"] = 999
    with pytest.raises(ValueError):
        mc.StructuralFit.from_json(json.dumps(broken))


def test_structural_validates_priors_dimensions_and_optional_history():
    V, C = mc.VarianceParameter, mc.StructuralComponent
    with pytest.raises(ValueError):
        V.inverse_gamma(0, 1)
    with pytest.raises(ValueError):
        C.trend("trend", V.fixed(0), V.fixed(0), [0], [[1]])
    with pytest.raises(ValueError):
        C.ar("ar", [1.1], V.fixed(1), [0], [[1]])
    with pytest.raises(ValueError):
        C.seasonal("aliased", 4., 2, V.fixed(0), 1.)
    model = mc.StructuralModel([C.level("level", V.fixed(.1), 0, 1)], V.fixed(.2))
    fit = model.fit([0., np.nan, 1.], chains=1, draws=4, warmup=0)
    with pytest.raises(ValueError, match="store_states"):
        _ = fit.states
    with pytest.raises(ValueError):
        fit.forecast(2, exog=[[1.], [1.]])
    assert fit.terminal_states.shape == (1, 4, 1)
    for df in [0.5, 1.0]:
        with pytest.raises(ValueError, match="greater than one"):
            mc.StructuralModel([C.level("level", V.fixed(.1), 0, 1)], V.fixed(.2), student_df=df)
    with pytest.raises(ValueError, match="25 million"):
        C.seasonal("oversized", 1e20, 2**20, V.fixed(.1), 1.)
