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


@pytest.mark.parametrize('coefficients', [[0.], [.5, 0.]])
def test_zero_innovation_ar_singular_states_fit_forecast_and_replay(coefficients):
    """A deterministic stable AR can lose rank without invalidating its prior."""
    n = len(coefficients)
    V, C = mc.VarianceParameter, mc.StructuralComponent
    model = mc.StructuralModel([
        C.ar('ar', coefficients, V.fixed(0), [0.] * n, np.eye(n).tolist())
    ], V.fixed(1))
    fit = model.fit([.2, np.nan, -.1, .3], chains=2, draws=20, warmup=0,
                    store_states=True, seed=751)
    states = fit.states
    transition = np.zeros((n, n))
    transition[0] = coefficients
    if n > 1:
        transition[1:, :-1] = np.eye(n - 1)
    np.testing.assert_allclose(states[:, :, 1:], states[:, :, :-1] @ transition.T, atol=1e-7)
    forecast = fit.forecast(3, seed=752)
    expected = fit.terminal_states.copy()
    for t in range(3):
        expected = expected @ transition.T
        np.testing.assert_allclose(forecast.state_paths[:, :, t], expected, atol=1e-7)
    restored = mc.StructuralFit.from_json(fit.to_json())
    np.testing.assert_array_equal(restored.forecast(3, seed=752).observation_paths,
                                  forecast.observation_paths)


def test_forecast_reusing_the_fit_seed_stays_independent_of_the_terminal_state():
    """Forecasting with the fit's own seed must not replay the fit's RNG stream.

    Fitting and forecasting draw from separate domains of the seed, so a caller
    who passes one seed to both still gets predictive innovations that are
    independent of the sampled terminal state. When the two shared a stream, the
    forecast's first transition innovation was literally the same standard normal
    pair that built the terminal state it was applied to, inflating the
    predictive variance by ~60% with no error reported.
    """
    V, C = mc.VarianceParameter, mc.StructuralComponent
    q, r, p0, seed, draws = .5, 1., 1., 42, 30000
    model = mc.StructuralModel([C.level("a", V.fixed(q), 0., p0),
                                C.level("b", V.fixed(q), 0., p0)], V.fixed(r))
    # One observation of the summed levels keeps the posterior exactly Gaussian,
    # and fixed variances make every Gibbs sweep an independent FFBS draw.
    fit = model.fit([1.], chains=1, draws=draws, warmup=0, seed=seed)
    terminal = fit.terminal_states[0]
    total = terminal.sum(-1)

    # Kalman update for state_1 given the single observation y_1 = [1 1] state_1.
    prior, design = np.diag([p0 + q, p0 + q]), np.ones((1, 2))
    innovation_variance = design @ prior @ design.T + r
    gain = prior @ design.T / innovation_variance
    posterior = prior - gain @ innovation_variance @ gain.T
    analytic_state = float((design @ posterior @ design.T)[0, 0])
    analytic_observation = analytic_state + 2 * q + r
    assert abs(total.var(ddof=1) / analytic_state - 1) < .04, "fitted terminal state is off"

    # Several horizons: the two streams advance at different rates, so only some
    # step counts line them up, and which ones is an implementation detail.
    for steps in (1, 2, 3):
        paths = fit.forecast(steps, seed=seed)
        first = paths.observation_paths[0][:, 0]
        transition = paths.state_paths[0][:, 0, :].sum(-1) - total
        correlation = float(np.corrcoef(total, transition)[0, 1])
        assert abs(correlation) < .05, (
            f"steps={steps}: terminal state and first forecast innovation correlate "
            f"at {correlation:+.4f}; the forecast is replaying the fit's RNG stream")
        assert abs(first.var(ddof=1) / analytic_observation - 1) < .04, (
            f"steps={steps}: step-1 predictive variance {first.var(ddof=1):.4f} "
            f"does not match the analytic {analytic_observation:.4f}")
