"""Forecast reconciliation: hierarchies, the textbook formula, coherence, and a known process."""

import numpy as np
import pytest

LABELS = [
    ("BU1", "IDN1", "c1"),
    ("BU1", "IDN1", "c2"),
    ("BU1", "IDN2", "c3"),
    ("BU2", "IDN3", "c4"),
    ("BU2", "IDN3", "c5"),
]
NODES = [
    "total", "BU1", "BU2", "BU1/IDN1", "BU1/IDN2", "BU2/IDN3",
    "BU1/IDN1/c1", "BU1/IDN1/c2", "BU1/IDN2/c3", "BU2/IDN3/c4", "BU2/IDN3/c5",
]


@pytest.fixture
def rmc(rustmc_module):
    return rustmc_module


@pytest.fixture
def hierarchy(rmc):
    return rmc.Hierarchy.from_labels(LABELS)


def base_draws(n, chains=2, draws=30, steps=3, seed=0):
    rng = np.random.default_rng(seed)
    return 20.0 + 5.0 * rng.standard_normal((chains, draws, n, steps))


def residual_matrix(n, times=40, seed=1):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((times, 1))
    scale = 0.5 + 0.4 * np.arange(n)
    return scale * (0.6 * common + rng.standard_normal((times, n)))


def textbook_map(s, w):
    """S (S' W^-1 S)^-1 S' W^-1, formed with dense inverses."""
    w_inverse = np.linalg.inv(w)
    p = np.linalg.inv(s.T @ w_inverse @ s) @ s.T @ w_inverse
    return s @ p


def shrunk_covariance(e):
    """Schafer-Strimmer shrinkage of the uncentred residual covariance, as in hts."""
    times, n = e.shape
    covariance = e.T @ e / times
    x = e / np.sqrt(np.diag(covariance))
    products = x[:, :, None] * x[:, None, :]
    correlation = products.mean(axis=0)
    variance = products.var(axis=0, ddof=1) / times
    off = ~np.eye(n, dtype=bool)
    lam = np.clip(variance[off].sum() / (correlation[off] ** 2).sum(), 0.0, 1.0)
    return lam * np.diag(np.diag(covariance)) + (1 - lam) * covariance, lam


def apply_map(matrix, draws):
    return np.einsum("nk,cdkh->cdnh", matrix, draws)


def incoherence(hierarchy, draws):
    """Largest |aggregate - sum of its bottoms| relative to the sum."""
    s = hierarchy.summing_matrix
    r = hierarchy.aggregate_count
    sums = np.einsum("ak,cdkh->cdah", s[:r], draws[:, :, r:, :])
    return np.max(np.abs(draws[:, :, :r, :] - sums) / np.maximum(np.abs(sums), 1.0))


def test_labels_give_documented_node_order_names_and_structure(hierarchy):
    assert hierarchy.node_names == NODES
    assert hierarchy.aggregate_names == NODES[:6]
    assert hierarchy.bottom_names == NODES[6:]
    assert (hierarchy.node_count, hierarchy.aggregate_count, hierarchy.bottom_count) == (11, 6, 5)
    assert hierarchy.is_nested
    assert hierarchy.levels == [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    assert hierarchy.parents[:4] == [None, "total", "total", "BU1"]
    assert hierarchy.parents[-1] == "BU2/IDN3"
    s = hierarchy.summing_matrix
    assert s.shape == (11, 5)
    np.testing.assert_array_equal(s[0], np.ones(5))
    np.testing.assert_array_equal(s[1], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(s[5], [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(s[6:], np.eye(5))


def test_summing_matrix_constructor_and_grouped_structures(rmc, hierarchy):
    again = rmc.Hierarchy(hierarchy.summing_matrix, node_names=hierarchy.node_names)
    assert again == hierarchy
    assert rmc.Hierarchy(hierarchy.summing_matrix).node_names[:2] == ["node_0", "node_1"]
    grid = np.array([
        [1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1],
        *np.eye(4, dtype=int),
    ])
    grouped = rmc.Hierarchy(grid)
    assert not grouped.is_nested
    assert grouped.levels is None and grouped.parents is None


@pytest.mark.parametrize(
    "matrix, message",
    [
        ([[1.0, 0.0], [0.0, 1.0]], "at least one aggregate"),
        ([[1.0, 2.0], [1.0, 0.0], [0.0, 1.0]], "only 0 and 1"),
        ([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], "identity"),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "sums no bottom"),
        ([[True, True], [True, False], [False, True]], "summing_matrix must hold real numbers"),
    ],
)
def test_malformed_summing_matrices_are_refused(rmc, matrix, message):
    with pytest.raises(ValueError, match=message):
        rmc.Hierarchy(matrix)


@pytest.mark.parametrize(
    "labels, message",
    [
        ("BU1", "got a string"),
        (["BU1"], r"labels\[0\] must be a sequence of strings"),
        ([("BU1", 3)], r"labels\[0\] must be a sequence of strings"),
        ([("A", "x"), ("A",)], "same length"),
        ([("A", "x"), ("A", "x")], "repeats"),
        ([("A", "x/y")], "separator"),
        ([("total", "x")], "occurs more than once"),
    ],
)
def test_malformed_labels_are_refused(rmc, labels, message):
    with pytest.raises(ValueError, match=message):
        rmc.Hierarchy.from_labels(labels)


def test_projections_match_the_numpy_textbook_formula(rmc, hierarchy):
    n = hierarchy.node_count
    s = hierarchy.summing_matrix
    draws = base_draws(n)
    residuals = residual_matrix(n)
    shrunk, lam = shrunk_covariance(residuals)
    assert 0 < lam < 1
    weights = {
        "ols": np.eye(n),
        "wls_struct": np.diag(s.sum(axis=1)),
        "wls_var": np.diag((residuals ** 2).mean(axis=0)),
        "mint_shrink": shrunk,
    }
    for method, w in weights.items():
        options = {"residuals": residuals} if method in ("wls_var", "mint_shrink") else {}
        result = rmc.reconcile(draws, hierarchy, method=method, **options)
        expected = apply_map(textbook_map(s, w), draws)
        np.testing.assert_allclose(result.observation_samples, expected, rtol=1e-10, atol=1e-10,
                                   err_msg=method)
        assert result.method == method
    mint = rmc.reconcile(draws, hierarchy, method="mint_shrink", residuals=residuals)
    assert mint.shrinkage_intensity == pytest.approx(lam, rel=1e-12)


def test_bottom_up_is_the_sum_of_the_bottom_draws(rmc, hierarchy):
    draws = base_draws(hierarchy.node_count)
    result = rmc.reconcile(draws, hierarchy, method="bottom_up")
    reconciled = result.observation_samples
    np.testing.assert_array_equal(reconciled[:, :, 6:, :], draws[:, :, 6:, :])
    np.testing.assert_allclose(reconciled, apply_map(hierarchy.summing_matrix, draws[:, :, 6:, :]),
                               rtol=1e-15)
    assert result.shrinkage_intensity is None


def test_every_method_makes_every_draw_coherent(rmc, hierarchy):
    n = hierarchy.node_count
    draws = np.abs(base_draws(n, chains=3, draws=50, steps=4, seed=3))
    residuals = residual_matrix(n)
    history = np.random.default_rng(4).uniform(1, 10, (24, hierarchy.bottom_count))
    options = {
        "bottom_up": {}, "ols": {}, "wls_struct": {},
        "top_down_average_proportions": {"history": history},
        "top_down_forecast_proportions": {},
        "wls_var": {"residuals": residuals}, "mint_shrink": {"residuals": residuals},
    }
    assert set(options) == set(rmc.reconciliation.METHODS)
    for method, extra in options.items():
        result = rmc.reconcile(draws, hierarchy, method=method, **extra)
        assert result.observation_samples.shape == draws.shape
        assert incoherence(hierarchy, result.observation_samples) <= 1e-10, method
    assert incoherence(hierarchy, draws) > 0.1


def test_projections_leave_coherent_draws_unchanged(rmc, hierarchy):
    s = hierarchy.summing_matrix
    rng = np.random.default_rng(8)
    bottom = rng.normal(10, 2, (2, 25, hierarchy.bottom_count, 3))
    coherent = apply_map(s, bottom)
    residuals = residual_matrix(hierarchy.node_count)
    for method, extra in {
        "bottom_up": {}, "ols": {}, "wls_struct": {},
        "wls_var": {"residuals": residuals}, "mint_shrink": {"residuals": residuals},
    }.items():
        result = rmc.reconcile(coherent, hierarchy, method=method, **extra)
        np.testing.assert_allclose(result.observation_samples, coherent, rtol=1e-12, err_msg=method)


def test_top_down_methods_use_the_documented_proportions(rmc, hierarchy):
    n, m = hierarchy.node_count, hierarchy.bottom_count
    history = np.array([[1.0, 1.0, 2.0, 4.0, 2.0], [2.0, 0.0, 2.0, 3.0, 3.0]])
    draws = base_draws(n, steps=2)
    result = rmc.reconcile(draws, hierarchy, method="top_down_average_proportions", history=history)
    shares = (history / history.sum(axis=1, keepdims=True)).mean(axis=0)
    np.testing.assert_allclose(result.observation_samples[:, :, 6:, :],
                               shares[None, None, :, None] * draws[:, :, :1, :], rtol=1e-14)
    # Forecast proportions: each node's share of its siblings' base means,
    # multiplied down the path from the total.
    result = rmc.reconcile(draws, hierarchy, method="top_down_forecast_proportions")
    mean = draws.mean(axis=(0, 1))
    index = {name: i for i, name in enumerate(NODES)}
    parents = dict(zip(NODES, hierarchy.parents))
    children = {}
    for name, parent in parents.items():
        if parent is not None:
            children.setdefault(parent, []).append(name)
    expected = np.ones((m, draws.shape[-1]))
    for j, name in enumerate(NODES[6:]):
        node = name
        while parents[node] is not None:
            siblings = children[parents[node]]
            expected[j] *= mean[index[node]] / sum(mean[index[s]] for s in siblings)
            node = parents[node]
    np.testing.assert_allclose(result.observation_samples[:, :, 6:, :],
                               expected[None, None] * draws[:, :, :1, :], rtol=1e-12)


def error_process(n, r):
    """One shared factor, one factor per region, node noise scaled thirtyfold."""
    region = np.array([-1] + [0, 1, 2] + [j // 4 for j in range(n - r)])
    loadings = np.zeros((n, 4))
    loadings[:, 0] = 0.6
    for node, k in enumerate(region):
        if k >= 0:
            loadings[node, k + 1] = 0.6
    scales = np.array([6.0, 3.0, 3.0, 3.0] + [[0.2, 0.5, 1.5, 6.0][j % 4] for j in range(n - r)])
    covariance = scales[:, None] * scales[None, :] * (loadings @ loadings.T + np.eye(n))
    return np.linalg.cholesky(covariance), covariance


def test_mint_and_wls_beat_incoherent_base_forecasts_on_a_known_process(rmc):
    labels = [(f"R{j // 4}", f"s{j}") for j in range(12)]
    hierarchy = rmc.Hierarchy.from_labels(labels)
    n, r = hierarchy.node_count, hierarchy.aggregate_count
    s = hierarchy.summing_matrix
    factor, covariance = error_process(n, r)
    oracle = textbook_map(s, covariance)
    oracle_mse = np.trace(oracle @ covariance @ oracle.T) / n
    rng = np.random.default_rng(20260927)
    truth = s @ (20.0 + np.arange(12.0))
    methods = ["base", "ols", "wls_var", "mint_shrink"]
    squared = dict.fromkeys(methods, 0.0)
    scores = dict.fromkeys(methods, 0.0)
    replications, ensemble = 400, 200
    for _ in range(replications):
        center = truth + factor @ rng.standard_normal(n)
        draws = (center + rng.standard_normal((ensemble, n)) @ factor.T)[None, :, :, None]
        residuals = rng.standard_normal((60, n)) @ factor.T
        for method in methods:
            if method == "base":
                reconciled = draws
            else:
                extra = {"residuals": residuals} if method in ("wls_var", "mint_shrink") else {}
                reconciled = rmc.reconcile(draws, hierarchy, method=method, **extra).observation_samples
            squared[method] += np.sum((reconciled.mean(axis=(0, 1))[:, 0] - truth) ** 2)
            scores[method] += np.sum(rmc.crps(reconciled[..., 0], truth))
    mse = {m: v / (replications * n) for m, v in squared.items()}
    crps = {m: v / (replications * n) for m, v in scores.items()}
    print({m: round(mse[m], 3) for m in methods}, {m: round(crps[m], 3) for m in methods},
          round(oracle_mse, 3))
    assert abs(mse["base"] / (np.trace(covariance) / n) - 1) < 0.1
    assert mse["mint_shrink"] < 1.2 * oracle_mse < mse["wls_var"]
    assert mse["ols"] > 1.5 * oracle_mse
    assert mse["mint_shrink"] < 0.45 * mse["base"]
    assert mse["wls_var"] < 0.55 * mse["base"]
    assert crps["mint_shrink"] < 0.72 * crps["base"]
    assert crps["wls_var"] < 0.76 * crps["base"]
    assert crps["mint_shrink"] < 0.9 * crps["ols"]


def test_summaries_follow_the_shared_quantile_rule(rmc, hierarchy):
    draws = base_draws(hierarchy.node_count, draws=41)
    result = rmc.reconcile(draws, hierarchy, method="ols")
    samples = result.observation_samples
    pooled = samples.reshape((-1,) + samples.shape[2:])
    np.testing.assert_allclose(result.observation_mean, pooled.mean(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(result.observation_quantile(0.3), np.quantile(pooled, 0.3, axis=0))
    lower, upper = result.interval(0.9)
    # The tail probabilities are (1 - level) / 2 and (1 + level) / 2, as for
    # every other forecast interval.
    np.testing.assert_array_equal(lower, np.quantile(pooled, (1 - 0.9) / 2, axis=0))
    np.testing.assert_array_equal(upper, np.quantile(pooled, (1 + 0.9) / 2, axis=0))
    np.testing.assert_array_equal(result.node_samples("BU2"), samples[:, :, 2, :])
    assert (result.chains, result.draws, result.node_count, result.steps) == (2, 41, 11, 3)
    assert result.node_names == NODES and result.hierarchy == hierarchy
    assert result.interval_kind == "pointwise_equal_tailed"
    forecast = rmc.ForecastDraws.from_result(result)
    assert forecast.series == tuple(NODES)
    with pytest.raises(ValueError, match="not a node"):
        result.node_samples("BU3")
    with pytest.raises(ValueError, match="level"):
        result.interval(1.0)


def test_name_keyed_inputs_match_node_ordered_arrays(rmc, hierarchy):
    draws = base_draws(hierarchy.node_count)
    residuals = residual_matrix(hierarchy.node_count)
    expected = rmc.reconcile(draws, hierarchy, method="mint_shrink", residuals=residuals)
    order = list(reversed(range(len(NODES))))
    by_name = {NODES[i]: draws[:, :, i, :] for i in order}
    residuals_by_name = {NODES[i]: residuals[:, i] for i in order}
    result = rmc.reconcile(by_name, hierarchy, method="mint_shrink", residuals=residuals_by_name)
    np.testing.assert_array_equal(result.observation_samples, expected.observation_samples)
    shuffled = rmc.ForecastDraws(draws[:, :, order, :], series=[NODES[i] for i in order])
    result = rmc.reconcile(shuffled, hierarchy, method="mint_shrink", residuals=residuals)
    np.testing.assert_array_equal(result.observation_samples, expected.observation_samples)
    history = {name: np.arange(1.0, 6.0) + j for j, name in enumerate(hierarchy.bottom_names)}
    by_history = rmc.reconcile(draws, hierarchy, method="top_down_average_proportions", history=history)
    as_array = rmc.reconcile(draws, hierarchy, method="top_down_average_proportions",
                             history=np.column_stack(list(history.values())))
    np.testing.assert_array_equal(by_history.observation_samples, as_array.observation_samples)


def test_hierarchical_forecast_adapter_builds_the_tree_and_is_already_coherent(rmc):
    model = rmc.BayesianHierarchicalMean(
        group_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
        program_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
        observation_variance_prior=rmc.InverseGammaPrior(3.0, 2.0),
    )
    series = [np.array([9.5, 10.1]), np.array([30.2, 29.8, 30.0]), np.array([10.3, 9.7])]
    fit = model.fit(series, group_index=[1, 0, 1], program_names=["p", "q", "r"],
                    group_names=["west", "east"], chains=2, draws=40, warmup=40, seed=3)
    forecast = fit.forecast(4, seed=5)
    result = rmc.reconcile(forecast, method="ols")
    assert result.node_names == ["total", "east", "west", "east/p", "west/q", "east/r"]
    samples = result.observation_samples
    np.testing.assert_allclose(samples[:, :, 0, :], forecast.total_observation_samples, rtol=1e-12)
    np.testing.assert_allclose(samples[:, :, 1, :], forecast.group_observation_samples[:, :, 1, :],
                               rtol=1e-12)
    np.testing.assert_allclose(samples[:, :, 3:, :], forecast.observation_samples, rtol=1e-12)
    with pytest.raises(ValueError, match="must have the forecast's structure"):
        rmc.reconcile(forecast, rmc.Hierarchy.from_labels(LABELS), method="ols")
    # Names the default naming cannot express: pass a hierarchy built with
    # another total name and separator, matched by structure.
    awkward = model.fit(series, group_index=[1, 0, 1], program_names=["p/1", "q", "r"],
                        group_names=["total", "east"], chains=2, draws=40, warmup=40, seed=3)
    forecast = awkward.forecast(4, seed=5)
    with pytest.raises(ValueError, match="cannot name this forecast's nodes by default"):
        rmc.reconcile(forecast, method="ols")
    custom = rmc.Hierarchy.from_labels([("east", "p/1"), ("total", "q"), ("east", "r")],
                                       total="all", separator="|")
    result = rmc.reconcile(forecast, custom, method="ols")
    assert result.node_names == ["all", "east", "total", "east|p/1", "total|q", "east|r"]
    np.testing.assert_allclose(result.node_samples("total"),
                               forecast.group_observation_samples[:, :, 0, :], rtol=1e-12)


def test_batch_forecast_adapter_places_cells_by_id(rmc):
    hierarchy = rmc.Hierarchy.from_labels([("a",), ("b",)])
    prior = rmc.InverseGammaPrior(3.0, 0.4)
    model = rmc.BayesianLocalLevel(process_variance_prior=prior, observation_variance_prior=prior)
    rng = np.random.default_rng(2)
    a, b = 5 + rng.normal(0, 0.3, 20), 7 + rng.normal(0, 0.3, 20)
    batch = model.fit_batch([b, a + b + 0.5, a], ids=["b", "total", "a"],
                            chains=2, draws=50, warmup=50, seed=1)
    forecast = batch.forecast(3, seed=2)
    result = rmc.reconcile(forecast, hierarchy, method="bottom_up")
    np.testing.assert_array_equal(result.node_samples("a"), forecast["a"].observation_samples)
    np.testing.assert_allclose(
        result.node_samples("total"),
        forecast["a"].observation_samples + forecast["b"].observation_samples, rtol=1e-15)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"method": "mint"}, "method must be"),
        ({"method": "mint_shrink"}, "needs residuals"),
        ({"method": "ols", "residuals": np.ones((5, 11))}, "residuals are used only by"),
        ({"method": "ols", "history": np.ones((5, 5))}, "history is used only by"),
        ({"method": "top_down_average_proportions"}, "needs history"),
        ({"method": "mint_shrink", "residuals": np.ones((5, 10))}, "one column per node"),
        ({"method": "mint_shrink", "residuals": np.ones((1, 11))}, "at least 2 rows"),
        ({"method": "wls_var", "residuals": np.ones(11)}, "residuals must be two-dimensional"),
        ({"method": "wls_var", "residuals": np.full((3, 11), np.nan)},
         r"residuals\[0, 0\] \(node \"total\"\) is not finite"),
        ({"method": "wls_var", "residuals": np.zeros((3, 11))}, "mean square 0"),
        ({"method": "top_down_average_proportions", "history": -np.ones((2, 5))},
         r"history\[0, 0\]"),
        ({"method": "wls_var", "residuals": {"total": np.ones(3)}}, "missing nodes"),
        ({"method": "wls_var", "residuals": dict.fromkeys(NODES, 1.0)},
         "residuals for node 'total' must be one series"),
        ({"method": "wls_var", "residuals": {n: np.ones(4 if n == "BU2" else 3) for n in NODES}},
         "residuals for node 'BU2' has 4 values"),
    ],
)
def test_invalid_arguments_are_refused_by_name(rmc, hierarchy, arguments, message):
    draws = base_draws(hierarchy.node_count)
    with pytest.raises(ValueError, match=message):
        rmc.reconcile(draws, hierarchy, **arguments)


def test_invalid_draws_are_refused_by_name(rmc, hierarchy):
    draws = base_draws(hierarchy.node_count)
    with pytest.raises(ValueError, match="four-dimensional"):
        rmc.reconcile(draws[0], hierarchy, method="ols")
    with pytest.raises(ValueError, match="draws have 10 nodes"):
        rmc.reconcile(draws[:, :, 1:, :], hierarchy, method="ols")
    bad = draws.copy()
    bad[1, 2, 3, 0] = np.nan
    with pytest.raises(ValueError, match=r"draws\[1, 2, 3, 0\] is not finite"):
        rmc.reconcile(bad, hierarchy, method="ols")
    with pytest.raises(ValueError, match="bool"):
        rmc.reconcile(draws > 0, hierarchy, method="ols")
    by_name = {name: draws[:, :, i, :] for i, name in enumerate(NODES)}
    by_name["BU1"] = by_name["BU1"][:, :-1]
    with pytest.raises(ValueError, match="node 'BU1' have shape"):
        rmc.reconcile(by_name, hierarchy, method="ols")
    by_name["BU1"] = draws[:, :, 1, :] > 0
    with pytest.raises(ValueError, match="draws for node 'BU1' must hold real numbers"):
        rmc.reconcile(by_name, hierarchy, method="ols")
    with pytest.raises(ValueError, match="hierarchy is required"):
        rmc.reconcile(draws, method="ols")
    unlabeled = rmc.ForecastDraws(draws[:, :, :, 0])
    with pytest.raises(ValueError, match="needs a series axis"):
        rmc.reconcile(unlabeled, hierarchy, method="ols")


def test_top_down_refusals_name_the_reason(rmc):
    no_total = rmc.Hierarchy([[1, 1, 0, 0], [0, 0, 1, 1], *np.eye(4)])
    draws = base_draws(6)
    with pytest.raises(ValueError, match="no node of this hierarchy sums every bottom"):
        rmc.reconcile(draws, no_total, method="top_down_forecast_proportions")
    grid = rmc.Hierarchy([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], *np.eye(4)])
    with pytest.raises(ValueError, match="nested"):
        rmc.reconcile(base_draws(7), grid, method="top_down_forecast_proportions")
    negative = -np.abs(base_draws(11))
    with pytest.raises(ValueError, match="non-negative base means"):
        rmc.reconcile(negative, rmc.Hierarchy.from_labels(LABELS),
                      method="top_down_forecast_proportions")
