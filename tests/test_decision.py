"""Decisions from predictive draws: optima, expected losses, horizon totals, MCSE."""
from statistics import NormalDist

import numpy as np
import pytest

import rustmc as r


def normal_draws(seed, shape=(4, 500), mean=7.5, sd=1.25):
    return np.random.default_rng(seed).normal(mean, sd, size=shape)


def ar1_chains(seed, chains, draws, phi):
    """Stationary AR(1) chains with unit marginal variance."""
    rng = np.random.default_rng(seed)
    values = np.empty((chains, draws))
    current = rng.standard_normal(chains)
    for index in range(draws):
        current = phi * current + np.sqrt(1 - phi**2) * rng.standard_normal(chains)
        values[:, index] = current
    return values


def pinball(decision, outcomes, under, over):
    shortfall = outcomes - decision
    return np.where(shortfall > 0, under * shortfall, over * (decision - outcomes))


def test_pinball_optimum_is_the_librarys_empirical_quantile():
    draws = normal_draws(1)
    decision = r.decide(draws, r.AsymmetricLinearLoss(under=5, over=1))
    assert decision.method == "quantile"
    assert decision.quantile == pytest.approx(5 / 6)
    # The library's quantile rule reproduces NumPy's default linear method.
    assert decision.decision == np.quantile(draws, 5 / 6)


def test_pinball_optimum_approaches_the_normal_quantile():
    mean, sd, probability = 7.5, 1.25, 5 / 6
    draws = normal_draws(2, shape=(4, 100_000), mean=mean, sd=sd)
    decision = r.decide(draws, r.AsymmetricLinearLoss(under=5, over=1))
    analytic = NormalDist(mean, sd).inv_cdf(probability)
    density = NormalDist(mean, sd).pdf(analytic)
    standard_error = np.sqrt(probability * (1 - probability) / draws.size) / density
    assert abs(decision.decision - analytic) < 4 * standard_error
    # The batch-means MCSE is of the same order as the asymptotic error.
    assert 0.5 * standard_error < decision.decision_mcse < 2 * standard_error


def test_squared_loss_gives_the_mean_and_absolute_loss_the_median():
    draws = normal_draws(3, shape=(3, 101), mean=2.0, sd=3.0)
    squared = r.decide(draws, r.SquaredLoss())
    assert squared.method == "mean" and squared.quantile is None
    assert squared.decision == pytest.approx(draws.mean(), rel=1e-14)
    absolute = r.decide(draws, r.AbsoluteLoss())
    assert absolute.method == "median"
    assert absolute.decision == np.median(draws)


@pytest.mark.parametrize(
    "loss, reference",
    [
        (r.AsymmetricLinearLoss(under=5, over=1), lambda a, y: pinball(a, y, 5, 1)),
        (r.SquaredLoss(), lambda a, y: (y - a) ** 2),
        (r.AbsoluteLoss(), lambda a, y: np.abs(y - a)),
        (
            r.PiecewiseLinearLoss([-1.0, 0.0, 2.0], [2.0, 0.0, 10.0]),
            lambda a, y: np.interp(y - a, [-1.0, 0.0, 2.0], [2.0, 0.0, 10.0], left=None, right=None)
            + np.where(y - a < -1, 2 * (-1 - (y - a)), 0)
            + np.where(y - a > 2, 5 * ((y - a) - 2), 0),
        ),
    ],
)
def test_expected_loss_matches_numpy(loss, reference):
    draws = normal_draws(4, shape=(2, 300), mean=0.0, sd=1.5)
    for decision in (-2.5, 0.3, 4.0):
        estimate = r.expected_loss(draws, decision, loss)
        assert estimate.decision == decision
        assert estimate.value == pytest.approx(reference(decision, draws).mean(), rel=1e-12)
        assert estimate.mcse > 0 and estimate.ess > 50


def test_grid_search_agrees_with_the_closed_form():
    draws = normal_draws(5)
    ordered = np.sort(draws.ravel())
    widest_gap = np.diff(ordered).max()
    for under, over in [(5, 1), (1, 3), (1, 1)]:
        loss = r.AsymmetricLinearLoss(under=under, over=over)
        closed = r.decide(draws, loss)
        # Every draw is a candidate, so the grid holds the exact minimiser of
        # the average pinball loss.
        grid = r.decide(draws, loss, candidates=ordered, method="grid")
        assert grid.method == "grid"
        assert grid.expected_loss <= closed.expected_loss + 1e-12
        assert closed.expected_loss - grid.expected_loss <= (under + over) / draws.size * widest_gap
        assert abs(closed.decision - grid.decision) <= widest_gap
    fine = np.linspace(ordered[0], ordered[-1], 4001)
    squared = r.decide(draws, r.SquaredLoss(), candidates=fine, method="grid")
    assert abs(squared.decision - draws.mean()) <= fine[1] - fine[0]
    np.testing.assert_allclose(squared.curve, ((draws.ravel()[:, None] - fine) ** 2).mean(axis=0), rtol=1e-10)


def test_piecewise_and_callable_losses_are_searched_on_the_grid():
    draws = normal_draws(6)
    candidates = np.linspace(5, 11, 601)
    piecewise = r.decide(draws, r.PiecewiseLinearLoss([-1, 0, 1], [1, 0, 5]), candidates=candidates)
    assert piecewise.method == "grid" and not piecewise.on_grid_edge
    reference = r.decide(draws, r.AsymmetricLinearLoss(under=5, over=1), candidates=candidates)
    np.testing.assert_allclose(piecewise.curve, reference.curve, rtol=1e-12)
    assert abs(piecewise.decision - reference.decision) <= 2 * (candidates[1] - candidates[0])

    seen = []

    def custom(decision, outcomes):
        seen.append((type(decision), outcomes.shape, outcomes.flags.writeable))
        return pinball(decision, outcomes, 5, 1)

    called = r.decide(draws, custom, candidates=candidates)
    assert called.method == "grid"
    np.testing.assert_allclose(called.curve, reference.curve, rtol=1e-12)
    assert called.decision == piecewise.decision
    assert seen[0] == (float, (4, 500), False)
    estimate = r.expected_loss(draws, 9.0, custom)
    assert estimate.value == pytest.approx(pinball(9.0, draws, 5, 1).mean(), rel=1e-12)


def test_grid_edge_is_flagged():
    draws = normal_draws(7)
    decision = r.decide(draws, r.SquaredLoss(), candidates=[10.0, 11.0, 12.0], method="grid")
    assert decision.decision == 10.0 and decision.on_grid_edge
    # Constant draws give a one-point default grid, which is its own edge.
    constant = r.decide(np.full(5, 2.0), lambda a, y: (y + 1 - a) ** 2)
    assert constant.candidates.tolist() == [2.0] and constant.on_grid_edge


def test_callable_is_called_once_per_candidate_and_grid_decisions_have_no_mcse():
    calls = []

    def squared(decision, outcomes):
        calls.append(decision)
        return (outcomes - decision) ** 2

    draws = normal_draws(12, shape=(4, 500), mean=0.0, sd=1.0)
    decision = r.decide(draws, squared, candidates=[-1.0, 1.0])
    assert calls == [-1.0, 1.0]
    assert decision.expected_loss == r.expected_loss(draws, decision.decision, r.SquaredLoss()).value
    # Batch means does not describe a decision that jumps between grid points.
    assert np.isnan(decision.decision_mcse)
    assert decision.batch_decisions.shape == (4, 2)
    assert np.isnan(r.decide(draws, r.AsymmetricLinearLoss(under=0, over=1)).decision_mcse)


def test_horizon_totals_match_summing_draws_by_hand():
    paths = np.random.default_rng(8).gamma(4.0, 2.0, size=(3, 200, 6))
    loss = r.AsymmetricLinearLoss(under=5, over=1)
    by_hand = paths[..., 0] + paths[..., 1] + paths[..., 2]
    total = r.decide(paths, loss, steps=range(3))
    assert total.decision == r.decide(by_hand, loss).decision
    assert total.expected_loss == r.decide(by_hand, loss).expected_loss
    cumulative = r.ForecastDraws(paths).cumulative_observation_samples[..., 2]
    np.testing.assert_array_equal(by_hand, cumulative)
    # Order does not matter; any subset of steps can be totalled.
    assert r.decide(paths, loss, steps=[2, 0, 1]).decision == total.decision
    later = paths[..., 3] + paths[..., 5]
    assert r.decide(paths, loss, steps=np.array([5, 3])).decision == r.decide(later, loss).decision
    assert r.expected_loss(paths, 20.0, loss, steps=[3, 5]).value == pytest.approx(
        pinball(20.0, later, 5, 1).mean(), rel=1e-12
    )
    # A horizon of one needs no steps.
    assert r.decide(paths[..., :1], loss).decision == np.quantile(paths[..., 0], 5 / 6)


def test_native_forecasts_and_forecast_draws_are_accepted():
    model = r.BayesianLocalLevel(
        process_variance_prior=r.InverseGammaPrior(shape=2.5, scale=0.3),
        observation_variance_prior=r.InverseGammaPrior(shape=2.5, scale=0.6),
        initial_mean=0.0,
        initial_variance=4.0,
    )
    fit = model.fit(np.array([0.2, -0.1, np.nan, 0.4, 0.3, 0.6]), chains=2, draws=100, warmup=50, seed=19)
    forecast = fit.forecast(steps=3, seed=23)
    loss = r.AsymmetricLinearLoss(under=5, over=1)
    direct = r.decide(forecast.observation_samples.sum(axis=2), loss)
    native = r.decide(forecast, loss, steps=range(3))
    draws = r.ForecastDraws.from_result(forecast)
    method = draws.optimal_decision(loss, steps=range(3))
    assert native.decision == direct.decision == method.decision
    assert method.batch_decisions.shape == (2, 4)
    assert draws.expected_loss(1.0, loss, steps=[0]).value == pytest.approx(
        pinball(1.0, forecast.observation_samples[..., 0], 5, 1).mean(), rel=1e-12
    )


def test_single_chain_draws_and_batch_decisions():
    draws = normal_draws(9, shape=(1000,))
    decision = r.decide(draws, r.AbsoluteLoss())
    assert decision.batch_decisions.shape == (1, 8)
    blocks = np.array_split(draws, 8)
    np.testing.assert_allclose(decision.batch_decisions[0], [np.median(b) for b in blocks], rtol=1e-15)
    assert decision.decision_mcse == pytest.approx(
        np.std(decision.batch_decisions, ddof=1) / np.sqrt(8), rel=1e-12
    )
    assert "Decision(decision=" in repr(decision)


def test_expected_loss_mcse_matches_repeated_simulation():
    """The spread of the estimate over seeds matches the reported MCSE, with and
    without autocorrelation; 300 replicates know the spread to about 4%."""
    loss = r.AsymmetricLinearLoss(under=5, over=1)
    for phi in (0.0, 0.8):
        results = [r.expected_loss(ar1_chains(seed, 4, 500, phi), 1.0, loss) for seed in range(300)]
        spread = np.std([result.value for result in results], ddof=1)
        reported = np.mean([result.mcse for result in results])
        assert 0.85 < reported / spread < 1.15, (phi, reported, spread)
    # Ignoring the autocorrelation would understate the error.
    draws = ar1_chains(1, 4, 1000, 0.9)
    estimate = r.expected_loss(draws, 0.0, r.SquaredLoss())
    naive = np.std(draws**2, ddof=1) / np.sqrt(draws.size)
    assert estimate.mcse > 2 * naive


def test_decision_mcse_matches_repeated_simulation():
    loss = r.AsymmetricLinearLoss(under=5, over=1)
    decisions = [r.decide(ar1_chains(seed, 4, 500, 0.5), loss) for seed in range(300)]
    spread = np.std([decision.decision for decision in decisions], ddof=1)
    reported = np.mean([decision.decision_mcse for decision in decisions])
    assert 0.8 < reported / spread < 1.2, (reported, spread)


def test_too_few_draws_leave_the_mcse_unavailable():
    decision = r.decide([[1.0], [2.0], [4.0]], r.SquaredLoss())
    assert np.isnan(decision.decision_mcse) and np.isnan(decision.expected_loss_mcse)
    assert decision.batch_decisions.shape == (3, 0)
    constant = r.expected_loss(np.full((2, 10), 3.0), 3.0, r.AbsoluteLoss())
    assert constant.value == 0.0 and constant.mcse == 0.0


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: r.decide([[1.0, np.nan]], r.SquaredLoss()), "draws must be finite; chain 0, draw 1"),
        (lambda: r.decide([], r.SquaredLoss()), "draws must hold at least one draw"),
        (lambda: r.decide(3.0, r.SquaredLoss()), "not a single number"),
        (lambda: r.decide([True, False], r.SquaredLoss()), "draws must hold real numbers"),
        (lambda: r.decide(np.ones((2, 3, 4, 5)), r.SquaredLoss()), "series axis"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss()), "pass steps"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=[4]), "steps holds step 4"),
        (lambda: r.decide([[[1.0, np.nan]]], r.SquaredLoss(), steps=[0]), "draw 0, step 1 is NaN"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=[1, 1]), "more than once"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=[-1]), "numbered from 0"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=[True]), "steps must hold integers"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=[1.0]), "steps must hold integers"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=2), "sequence of step indices"),
        (lambda: r.decide(np.ones((2, 3, 4)), r.SquaredLoss(), steps=[]), "at least one step"),
        (lambda: r.decide(np.ones((2, 3)), r.SquaredLoss(), steps=[0]), "steps applies only"),
        (lambda: r.decide(np.ones(4), r.SquaredLoss(), candidates=[2.0, 1.0]), "candidates must be strictly"),
        (lambda: r.decide(np.ones(4), r.SquaredLoss(), candidates=[]), "candidates must hold at least one"),
        (lambda: r.decide(np.ones(4), r.SquaredLoss(), candidates=[1.0, np.inf]), "candidates must be finite"),
        (lambda: r.decide(np.ones(4), r.SquaredLoss(), method="exact"), "method must be 'auto' or 'grid'"),
        (lambda: r.decide(np.ones(4), "squared"), "loss must be an AsymmetricLinearLoss"),
        (lambda: r.expected_loss(np.ones(4), True, r.SquaredLoss()), "decision must hold real numbers"),
        (lambda: r.expected_loss(np.ones(4), np.nan, r.SquaredLoss()), "decision must be finite"),
        (lambda: r.expected_loss(np.ones(4), [1.0], r.SquaredLoss()), "decision must be a single number"),
        (lambda: r.AsymmetricLinearLoss(under=-1, over=1), "under must be finite and nonnegative"),
        (lambda: r.AsymmetricLinearLoss(under=1, over=True), "over must hold real numbers"),
        (lambda: r.AsymmetricLinearLoss(under=0, over=0), "cannot both be zero"),
        (lambda: r.AsymmetricLinearLoss(under=2**53 + 1, over=1), "under contains integer"),
        (lambda: r.PiecewiseLinearLoss([0.0], [0.0]), "at least two knots"),
        (lambda: r.PiecewiseLinearLoss([1.0, 0.0], [0.0, 1.0]), "shortfall must be strictly increasing"),
        (lambda: r.PiecewiseLinearLoss([0.0, 1.0], [0.0]), "one value per shortfall knot"),
        (lambda: r.decide(np.ones((2, 5)), lambda a, y: y[0]), "one loss per draw"),
    ],
)
def test_invalid_inputs_raise_value_errors_naming_the_argument(call, message):
    with pytest.raises(ValueError, match=message) as raised:
        call()
    # Refused arguments are plain ValueErrors; InferenceError is for numerical failures.
    assert not isinstance(raised.value, r.InferenceError)


def test_numerical_failures_and_callable_exceptions():
    with pytest.raises(r.InferenceError, match="losses must be finite"):
        r.expected_loss([1e200, -1e200], 0.0, r.SquaredLoss())
    with pytest.raises(r.InferenceError, match="not finite"):
        r.decide(np.full((1, 2, 2), 1e308), r.SquaredLoss(), steps=[0, 1])

    class Failure(Exception):
        pass

    def failing(decision, outcomes):
        raise Failure("from the loss")

    with pytest.raises(Failure, match="from the loss"):
        r.decide(np.ones(4), failing)
    with pytest.raises(r.InferenceError, match="losses must be finite"):
        r.expected_loss(np.ones(4), 0.0, lambda a, y: np.full_like(y, np.inf))


def test_loss_objects_report_their_parameters():
    loss = r.AsymmetricLinearLoss(under=5, over=1)
    assert (loss.under, loss.over, loss.quantile) == (5.0, 1.0, pytest.approx(5 / 6))
    assert repr(loss) == "AsymmetricLinearLoss(under=5, over=1)"
    assert r.AsymmetricLinearLoss(under=0, over=1).quantile == 0.0
    piecewise = r.PiecewiseLinearLoss(np.array([-1, 0, 1]), [1, 0, 5])
    np.testing.assert_array_equal(piecewise.shortfall, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(piecewise.loss, [1.0, 0.0, 5.0])
