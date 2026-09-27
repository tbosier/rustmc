"""Intervention, changepoint and Fourier designs: exact rows, aligned futures,
named refusals, and known-DGP recovery through the forecasting models."""
import numpy as np
import pytest

import rustmc as rmc

D = rmc.DesignTerm


def test_intervention_columns_are_exact_in_training_and_future():
    design = rmc.TimeDesign([
        D.step("shift", at=3), D.pulse("outage", at=3),
        D.ramp("slope", at=3), D.decay("launch", at=3, rate=0.5),
    ], length=6)
    assert design.names == ["shift", "outage", "slope", "launch"]
    assert (design.start, design.length, design.width) == (0, 6, 4)
    np.testing.assert_array_equal(design.training(), [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0.5],
        [1, 0, 2, 0.25],
    ])
    np.testing.assert_array_equal(design.future(3), [
        [1, 0, 3, 0.125],
        [1, 0, 4, 0.0625],
        [1, 0, 5, 0.03125],
    ])
    assert design.future(0).shape == (0, 4)


def test_edge_positions_before_the_start_at_zero_and_in_the_horizon():
    # Training covers calendar positions 0..4; the forecast starts at 5.
    design = rmc.TimeDesign([
        D.step("before", at=-2), D.ramp("ramp_before", at=-2), D.pulse("pulse_before", at=-1),
        D.decay("decay_before", at=-1, rate=0.5),
        D.step("at_zero", at=0), D.ramp("ramp_at_zero", at=0), D.pulse("pulse_at_zero", at=0),
        D.step("last", at=4),
        D.step("horizon", at=6), D.ramp("ramp_horizon", at=6), D.pulse("pulse_horizon", at=5),
        D.decay("decay_horizon", at=6, rate=0.5),
    ], length=5)
    expected_training = {
        "before": [1, 1, 1, 1, 1],
        "ramp_before": [2, 3, 4, 5, 6],
        "pulse_before": [0, 0, 0, 0, 0],
        "decay_before": [0.5, 0.25, 0.125, 0.0625, 0.03125],
        "at_zero": [1, 1, 1, 1, 1],
        "ramp_at_zero": [0, 1, 2, 3, 4],
        "pulse_at_zero": [1, 0, 0, 0, 0],
        "last": [0, 0, 0, 0, 1],
        "horizon": [0, 0, 0, 0, 0],
        "ramp_horizon": [0, 0, 0, 0, 0],
        "pulse_horizon": [0, 0, 0, 0, 0],
        "decay_horizon": [0, 0, 0, 0, 0],
    }
    expected_future = {
        "before": [1, 1, 1],
        "ramp_before": [7, 8, 9],
        "pulse_before": [0, 0, 0],
        "decay_before": [0.015625, 0.0078125, 0.00390625],
        "at_zero": [1, 1, 1],
        "ramp_at_zero": [5, 6, 7],
        "pulse_at_zero": [0, 0, 0],
        "last": [1, 1, 1],
        "horizon": [0, 1, 1],
        "ramp_horizon": [0, 0, 1],
        "pulse_horizon": [1, 0, 0],
        "decay_horizon": [0, 1, 0.5],
    }
    training, future = design.training(), design.future(3)
    for index, name in enumerate(design.names):
        np.testing.assert_array_equal(training[:, index], expected_training[name], err_msg=name)
        np.testing.assert_array_equal(future[:, index], expected_future[name], err_msg=name)


def test_start_shifts_the_calendar_but_not_the_definitions():
    shifted = rmc.TimeDesign([D.step("s", at=103), D.ramp("r", at=101)], length=4, start=100)
    np.testing.assert_array_equal(shifted.training(), [[0, 0], [0, 0], [0, 1], [1, 2]])
    np.testing.assert_array_equal(shifted.future(2), [[1, 3], [1, 4]])
    negative = rmc.TimeDesign([D.pulse("p", at=-1)], length=3, start=-3)
    np.testing.assert_array_equal(negative.training()[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(negative.future(2)[:, 0], [0, 0])


@pytest.mark.parametrize("length", [1, 7, 12, 25])
def test_future_rows_continue_the_training_rows_exactly(length):
    total = 36
    rng = np.random.default_rng(length)
    promo = rng.normal(size=total)
    terms = [
        D.step("shift", at=13), D.pulse("outage", at=20), D.ramp("slope", at=5),
        D.decay("launch", at=11, rate=0.7), D.fourier("annual", period=12, harmonics=6),
        D.changepoints("trend", at=[4, 9, 16, 30]), D.regressor("promo"),
    ]
    whole = rmc.TimeDesign(terms, length=total, start=-3).training(regressors={"promo": promo})
    design = rmc.TimeDesign(terms, length=length, start=-3)
    np.testing.assert_array_equal(design.training(regressors={"promo": promo[:length]}), whole[:length])
    np.testing.assert_array_equal(
        design.future(total - length, regressors={"promo": promo[length:]}), whole[length:])


def test_fourier_term_is_fourier_design_on_the_same_calendar():
    design = rmc.TimeDesign([D.fourier("annual", period=12, harmonics=6)], length=18, start=5)
    np.testing.assert_array_equal(design.training(), rmc.fourier_design(18, 12, 6, start=5))
    np.testing.assert_array_equal(design.future(12), rmc.fourier_design(12, 12, 6, start=23))
    assert design.names == [
        "annual_sin1", "annual_cos1", "annual_sin2", "annual_cos2", "annual_sin3", "annual_cos3",
        "annual_sin4", "annual_cos4", "annual_sin5", "annual_cos5", "annual_cos6",
    ]


def test_changepoint_grid_window_fraction_and_hinge_rows():
    # Defaults: the training window 100..149 and Prophet's fraction 0.8, so the
    # grid ends at 100 + floor(0.8 * 49) = 139.
    design = rmc.TimeDesign([D.changepoints("trend", count=5)], length=50, start=100)
    assert design.changepoints == {"trend": [108, 116, 123, 131, 139]}
    assert design.names == ["trend@108", "trend@116", "trend@123", "trend@131", "trend@139"]
    t = np.arange(100, 150)
    expected = np.maximum(0, t[:, None] - np.array([108, 116, 123, 131, 139])[None, :])
    np.testing.assert_array_equal(design.training(), expected)
    np.testing.assert_array_equal(design.future(2), [[42, 34, 27, 19, 11], [43, 35, 28, 20, 12]])

    whole = rmc.TimeDesign([D.changepoints("cp", count=4, fraction=1.0)], length=11)
    assert whole.changepoints["cp"] == [3, 5, 8, 10]  # 2.5, 5, 7.5, 10 rounded half up
    explicit = rmc.TimeDesign([D.changepoints("cp", count=2, first=20, last=30, fraction=0.5)], length=5)
    assert explicit.changepoints["cp"] == [23, 25]  # candidates may lie past the training window
    np.testing.assert_array_equal(explicit.training(), np.zeros((5, 2)))
    listed = rmc.TimeDesign([D.changepoints("cp", at=[-5, 2])], length=3)
    np.testing.assert_array_equal(listed.training(), [[5, 0], [6, 0], [7, 0]])


def test_term_slices_names_and_representations():
    design = rmc.TimeDesign([
        D.fourier("weekly", period=7, harmonics=2), D.step("shift", at=4), D.regressor("price"),
    ], length=10)
    assert design.term_slice("weekly") == slice(0, 4, 1)
    assert design.term_slice("shift") == slice(4, 5, 1)
    assert design.term_slice("price") == slice(5, 6, 1)
    assert design.regressor_names == ["price"]
    assert [term.name for term in design.terms] == ["weekly", "shift", "price"]
    assert repr(D.decay("launch", at=3, rate=0.5)) == 'DesignTerm.decay("launch", at=3, rate=0.5)'
    assert repr(D.changepoints("t", count=3)) == (
        'DesignTerm.changepoints("t", count=3, first=None, last=None, fraction=0.8)')
    assert "names=" in repr(design)
    with pytest.raises(ValueError, match="no term named 'missing'"):
        design.term_slice("missing")


REFUSALS = {
    "decay rate of one": (lambda: D.decay("fade", at=0, rate=1.0), "design term 'fade': decay rate"),
    "decay rate of zero": (lambda: D.decay("fade", at=0, rate=0.0), "decay rate must lie strictly"),
    "decay rate NaN": (lambda: D.decay("fade", at=0, rate=float("nan")), "decay rate"),
    "empty name": (lambda: D.step("", at=0), "nonempty name"),
    "fourier harmonics": (lambda: D.fourier("annual", period=12, harmonics=7), "design term 'annual'"),
    "unsorted changepoints": (lambda: D.changepoints("trend", at=[5, 5]), "strictly increasing"),
    "empty changepoints": (lambda: D.changepoints("trend", at=[]), "at least one changepoint"),
    "grid and positions": (lambda: D.changepoints("trend", at=[3], count=2), "exactly one of"),
    "neither grid nor positions": (lambda: D.changepoints("trend"), "exactly one of"),
    "grid options with positions": (lambda: D.changepoints("trend", at=[3], fraction=0.5), "leave them unset"),
    "zero grid count": (lambda: D.changepoints("trend", count=0), "count must be at least one"),
    "fraction above one": (lambda: D.changepoints("trend", count=2, fraction=1.5), "fraction must lie in"),
    "first after last": (lambda: D.changepoints("trend", count=2, first=9, last=3), "first must be before last"),
    "grid too dense": (lambda: rmc.TimeDesign([D.changepoints("trend", count=9)], length=10),
                       "9 changepoints need at least 9 positions"),
    "duplicate term": (lambda: rmc.TimeDesign([D.step("a", at=1), D.pulse("a", at=2)], length=5),
                       "'a' is used twice"),
    "duplicate column": (lambda: rmc.TimeDesign([D.fourier("annual", 12, 1), D.regressor("annual_sin1")], length=5),
                         "'annual_sin1' is used twice"),
    "no terms": (lambda: rmc.TimeDesign([], length=5), "at least one term"),
    "zero length": (lambda: rmc.TimeDesign([D.step("a", at=1)], length=0), "at least one"),
    "calendar overflow": (lambda: rmc.TimeDesign([D.step("a", at=1)], length=2, start=2**63 - 1),
                          "64-bit calendar"),
    "oversized horizon": (lambda: rmc.TimeDesign([D.step("a", at=1)], length=2).future(2**40), "safety limit"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_invalid_terms_and_designs_raise_plain_value_errors_naming_the_problem(case):
    build, message = REFUSALS[case]
    with pytest.raises(ValueError, match=message) as error:
        build()
    assert type(error.value) is ValueError


def test_regressor_values_are_named_sized_finite_and_exact():
    design = rmc.TimeDesign([D.regressor("promo"), D.step("shift", at=2)], length=3)
    np.testing.assert_array_equal(design.training(regressors={"promo": [1, 0, 1]}),
                                  [[1, 0], [0, 0], [1, 1]])
    refusals = [
        (lambda: design.training(), "regressor 'promo': no training values"),
        (lambda: design.future(2), "regressor 'promo': no future values"),
        (lambda: design.future(2, regressors={"promo": [1.0]}), "needs 2 future values"),
        (lambda: design.training(regressors={"promo": [1.0, np.nan, 1.0]}), "row 1 is not finite"),
        (lambda: design.training(regressors={"promo": [1.0] * 3, "price": [1.0] * 3}),
         "regressor 'price': is not a regressor term"),
        (lambda: design.training(regressors={"promo": np.array([True, False, True])}), "bool"),
        (lambda: design.training(regressors={"promo": np.ones((3, 1))}), "one-dimensional"),
    ]
    for build, message in refusals:
        with pytest.raises(ValueError, match=message):
            build()


# Known data-generating process: a local level plus a linear trend, with a
# level shift of 3 and a slope change of +0.08 per step at position 60.
N, HORIZON, BREAK = 100, 24, 60
SHIFT, SLOPE_CHANGE = 3.0, 0.08


def _break_series(seed, shift=SHIFT, slope_change=SLOPE_CHANGE):
    rng = np.random.default_rng(seed)
    t = np.arange(N + HORIZON)
    walk = np.cumsum(rng.normal(0.0, 0.03, N + HORIZON))
    return (10 + 0.02 * t + shift * (t >= BREAK) + slope_change * np.maximum(0, t - BREAK)
            + walk + rng.normal(0.0, 0.3, N + HORIZON))


def _level_model():
    ig = rmc.InverseGammaPrior
    return rmc.BayesianLocalLevel(ig(3, 0.01), ig(3, 0.2), initial_mean=10.0, initial_variance=4.0)


def _prior(variances):
    return rmc.GaussianCoefficientPrior(np.zeros(len(variances)), np.diag(variances))


def _designs(length=N):
    with_break = rmc.TimeDesign([
        D.ramp("trend", at=0), D.step("shift", at=BREAK), D.ramp("slope_change", at=BREAK),
    ], length=length)
    without = rmc.TimeDesign([D.ramp("trend", at=0)], length=length)
    return (with_break, [0.01, 25.0, 0.01]), (without, [0.01])


def test_level_shift_and_slope_change_are_recovered_and_absent_when_there_is_none():
    (design, variances), _ = _designs()
    model = _level_model()
    for seed in range(4):
        fit = model.fit(_break_series(seed)[:N], exog=design.training(), coefficient_prior=_prior(variances),
                        chains=4, draws=500, warmup=500, seed=seed)
        beta = fit.get_samples_2d()["coefficients"].reshape(-1, design.width)
        lower, upper = np.quantile(beta, [0.025, 0.975], axis=0)
        for index, truth in ((1, SHIFT), (2, SLOPE_CHANGE)):
            assert lower[index] < truth < upper[index], (seed, index, lower, upper)
            assert lower[index] > 0, (seed, index, lower)
        r_hat = {d["name"]: d["r_hat"] for d in fit.diagnostics()}
        assert max(r_hat[f"coefficient[{i}]"] for i in range(3)) < 1.01, r_hat

        # The same columns on the same series without a break: both intervals
        # cover zero, so the regressors do not manufacture an effect.
        control = model.fit(_break_series(seed, 0.0, 0.0)[:N], exog=design.training(),
                            coefficient_prior=_prior(variances), chains=4, draws=500, warmup=500, seed=seed)
        beta = control.get_samples_2d()["coefficients"].reshape(-1, design.width)
        lower, upper = np.quantile(beta, [0.025, 0.975], axis=0)
        assert (lower[1:] < 0).all() and (upper[1:] > 0).all(), (seed, lower, upper)


def test_intervention_columns_improve_the_post_break_forecast():
    """Over 8 seeded series, the break columns beat the same model without them."""
    model = _level_model()
    scores = {"with": [], "without": []}
    for seed in range(8):
        y = _break_series(seed)
        for label, (design, variances) in zip(scores, _designs()):
            fit = model.fit(y[:N], exog=design.training(), coefficient_prior=_prior(variances),
                            chains=4, draws=500, warmup=500, seed=seed)
            forecast = fit.forecast(HORIZON, exog=design.future(HORIZON), seed=seed)
            scores[label].append(rmc.crps(forecast.observation_samples, y[N:]).mean())
    with_break, without = np.array(scores["with"]), np.array(scores["without"])
    # Measured: mean CRPS 0.20 against 0.35, and lower on every seed.
    assert (with_break < without).all(), (with_break, without)
    assert with_break.mean() < 0.75 * without.mean(), (with_break.mean(), without.mean())


def test_rolling_origin_backtest_after_the_break():
    y = _break_series(11)
    results = {}
    for label, (design, variances) in zip(("with", "without"), _designs(length=len(y))):
        # The design covers every observation; backtest slices it at each origin.
        results[label] = rmc.backtest(
            _level_model(), y, horizon=12, origins=[76, 88, 100, 112], exog=design.training(),
            fit_kwargs={"coefficient_prior": _prior(variances), "chains": 2, "draws": 400, "warmup": 400},
            seed=5)
    with_break = results["with"].summary(by_horizon=False)["crps"]
    without = results["without"].summary(by_horizon=False)["crps"]
    naive = results["with"].summary(baseline=True, by_horizon=False)["crps"]
    assert with_break < 0.8 * without, (with_break, without)
    assert with_break < naive, (with_break, naive)


def test_changepoint_basis_recovers_a_piecewise_linear_trend():
    n, horizon = 100, 20
    t = np.arange(n + horizon)
    slope = np.where(t < 35, 0.05, np.where(t < 65, 0.20, -0.05))
    trend = 5 + np.concatenate([[0.0], np.cumsum(slope[:-1])])
    ig = rmc.InverseGammaPrior
    model = rmc.BayesianLocalLevel(ig(3, 0.002), ig(3, 0.2), initial_mean=5.0, initial_variance=4.0)
    with_changepoints = rmc.TimeDesign([D.ramp("trend", at=0), D.changepoints("cp", count=10)], length=n)
    # Neither true change (35, 65) is a candidate: the grid is 8, 16, ..., 79.
    positions = np.array(with_changepoints.changepoints["cp"])
    assert 35 not in positions and 65 not in positions
    without = rmc.TimeDesign([D.ramp("trend", at=0)], length=n)
    scores = {"with": [], "without": []}
    for seed in range(4):
        y = trend + np.random.default_rng(seed).normal(0.0, 0.3, n + horizon)
        for label, design in (("with", with_changepoints), ("without", without)):
            fit = model.fit(y[:n], exog=design.training(), coefficient_prior=_prior([0.01] * design.width),
                            chains=4, draws=300, warmup=300, seed=seed)
            forecast = fit.forecast(horizon, exog=design.future(horizon), seed=seed)
            scores[label].append(rmc.crps(forecast.observation_samples, y[n:]).mean())
            if label == "without":
                continue
            # The coefficients mix well. The level variance does not: its R-hat
            # reached 1.1-1.4 at these settings, the local-level mixing problem
            # ROADMAP.md describes, so it is not asserted here.
            r_hat = {d["name"]: d["r_hat"] for d in fit.diagnostics()}
            assert max(r_hat[f"coefficient[{i}]"] for i in range(design.width)) < 1.02, r_hat
            beta = fit.get_samples_2d()["coefficients"].reshape(-1, design.width)
            # Slope from t to t + 1: the base slope plus every hinge already passed.
            passed = (np.arange(n)[:, None] >= positions[None, :]).astype(float)
            fitted = (beta[:, :1] + beta[:, 1:] @ passed.T).mean(axis=0)
            for (first, last), truth in (((5, 30), 0.05), ((40, 60), 0.20), ((70, 95), -0.05)):
                assert abs(fitted[first:last].mean() - truth) < 0.03, (seed, first, fitted[first:last].mean())
    # Measured: mean CRPS 0.21 against 0.75.
    assert (np.array(scores["with"]) < 0.5 * np.array(scores["without"])).all(), scores


def test_structural_regression_components_take_term_slices():
    n, horizon, shift_at = 80, 12, 50
    rng = np.random.default_rng(3)
    t = np.arange(n + horizon)
    promo = rng.binomial(1, 0.3, n + horizon).astype(float)
    y = 10 + 2.0 * (t >= shift_at) + 1.5 * promo + rng.normal(0, 0.3, n + horizon)
    design = rmc.TimeDesign([D.step("shift", at=shift_at), D.regressor("promo")], length=n)
    C, V = rmc.StructuralComponent, rmc.VarianceParameter
    components = [C.level("level", V.inverse_gamma(3, 0.01), initial_mean=10.0, initial_variance=4.0)]
    for term in design.terms:
        width = len(design.names[design.term_slice(term.name)])
        components.append(C.regression(term.name, initial_mean=np.zeros(width),
                                       initial_covariance=9.0 * np.eye(width)))
    model = rmc.StructuralModel(components, observation_variance=V.inverse_gamma(3, 0.2))
    fit = model.fit(y[:n], exog=design.training(regressors={"promo": promo[:n]}),
                    chains=4, draws=500, warmup=500, seed=1)
    future = design.future(horizon, regressors={"promo": promo[n:]})
    forecast = fit.forecast(horizon, exog=future, seed=2)
    # State order: level, then one coefficient per regression component, in
    # design column order; each component path is its column times its state.
    states = forecast.state_paths
    for index, name in enumerate(("shift", "promo"), start=1):
        column = future[:, design.term_slice(name)][:, 0]
        np.testing.assert_allclose(forecast.component_paths[..., index], states[..., index] * column)
    lower, upper = np.quantile(fit.terminal_states[..., 1:].reshape(-1, 2), [0.025, 0.975], axis=0)
    assert lower[0] < 2.0 < upper[0] and lower[0] > 0, (lower, upper)
    assert lower[1] < 1.5 < upper[1], (lower, upper)


def test_dynamic_glm_takes_a_design_broadcast_over_groups():
    groups, n, horizon, shift_at = 3, 50, 10, 35
    rng = np.random.default_rng(3)
    rate = np.exp(1.0 + 0.7 * (np.arange(n) >= shift_at))
    counts = rng.poisson(np.broadcast_to(rate, (groups, n))).astype(float)
    design = rmc.TimeDesign([D.step("shift", at=shift_at)], length=n)
    exog = np.broadcast_to(design.training(), (groups, n, design.width)).copy()
    model = rmc.BayesianDynamicPoisson(initial_mean=1.0, coefficient_sd=1.0, group_sd=0.1, process_sd=0.05)
    fit = model.fit(counts, exog=exog, chains=2, draws=300, warmup=300, seed=3)
    # Column 0 of population_beta is the automatic intercept.
    effect = fit.get_samples_2d()["population_beta[0,1]"]
    lower, upper = np.quantile(effect, [0.025, 0.975])
    assert lower < 0.7 < upper and lower > 0, (lower, upper)
    future = np.broadcast_to(design.future(horizon), (groups, horizon, design.width)).copy()
    assert fit.forecast(horizon, exog=future, seed=4).observation_samples.shape == (2, 300, groups, horizon)
