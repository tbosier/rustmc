"""The private `CompiledModel._experimental_nuts` hook that
`benchmarks/adaptation_attribution.py` uses to freeze a metric and step size.

It is not a supported API. These tests check that the metric and step size
handed to each chain are the ones that chain samples with, against an
independent oracle: NUTS is affine invariant, so on N(0, S) with inverse metric
S it takes exactly the trajectories it takes on N(0, I) with the unit metric,
given the same random numbers. The scales are powers of two, so the scaling is
exact in floating point.
"""
import numpy as np
import pytest

import rustmc


def normals(*sds):
    builder = rustmc.ModelBuilder()
    for k, sd in enumerate(sds):
        builder.normal_prior(f"x{k}", 0.0, sd)
    return builder.compile()


def run(compiled, metrics, steps, init, seed=3):
    return compiled._experimental_nuts({}, chains=len(metrics), draws=400, warmup=50,
                                       seed=seed, inverse_metrics=metrics, step_sizes=steps,
                                       init=init)


def test_each_chain_samples_with_its_own_frozen_metric_and_step():
    sds = (8.0, 0.25)
    start = np.array([[0.3, -1.1], [-0.7, 0.4]])
    # Chain 0 gets the whitening metric, chain 1 the unit metric, each at its
    # own step; the reference runs both chains on the standard normal.
    scaled = run(normals(*sds), [[64.0, 0.0625], [1.0, 1.0]], [0.9, 0.2],
                 (start * sds).tolist())
    reference = run(normals(1.0, 1.0), [[1.0, 1.0], [1.0, 1.0]], [0.9, 0.2], start.tolist())
    whitened, unit = scaled["chains"]
    np.testing.assert_array_equal(whitened["n_leapfrog"], reference["chains"][0]["n_leapfrog"])
    draws = scaled["fit"].get_samples_2d()
    reference_draws = reference["fit"].get_samples_2d()
    for k, sd in enumerate(sds):
        np.testing.assert_array_equal(draws[f"x{k}"][0], sd * reference_draws[f"x{k}"][0])
    # The unit metric on the scaled target needs far longer trajectories, so
    # a chain given another chain's metric would fail the equality above.
    assert unit["n_leapfrog"].mean() > 4 * whitened["n_leapfrog"].mean()
    # Nothing adapted: every transition used the chain's step, and each chain
    # reports the metric it was given.
    for chain, step, metric in zip(scaled["chains"], [0.9, 0.2], [[64.0, 0.0625], [1.0, 1.0]]):
        assert np.all(chain["step_size_used"] == step) and chain["step_size"] == step
        (block,) = chain["inverse_metric"]
        assert (block["start"], block["len"], block["dense"]) == (0, 2, False)
        np.testing.assert_array_equal(block["values"], metric)
    assert scaled["fit"].step_sizes() == [0.9, 0.2]


def test_a_dense_metric_is_reported_as_its_covariance():
    out = normals(1.0, 1.0)._experimental_nuts({}, chains=1, draws=20, warmup=5,
                                                 inverse_metrics=[[4.0, 3.0, 3.0, 2.5]],
                                                 step_sizes=[0.5], init=[[0.0, 0.0]])
    (block,) = out["chains"][0]["inverse_metric"]
    assert block["dense"]
    np.testing.assert_allclose(block["values"], [4.0, 3.0, 3.0, 2.5], rtol=1e-12)


def test_without_a_frozen_metric_the_hook_adapts_and_reports_telemetry():
    out = normals(8.0, 0.25)._experimental_nuts({}, chains=2, draws=300, warmup=300, seed=5)
    fit = out["fit"]
    report = fit.transition_diagnostics()
    total = sum(chain["n_leapfrog"].sum() for chain in out["chains"])
    assert total == report["total_leapfrog_steps"]
    for chain, step in zip(out["chains"], fit.step_sizes()):
        assert chain["step_size"] == step
        assert chain["is_warmup"].sum() == 300 and len(chain["is_warmup"]) == 600
        # The adapted variances are near the target's, 64 and 1/16.
        variances = [block["values"][0] for block in chain["inverse_metric"]]
        assert 20 < variances[0] < 200 and 0.02 < variances[1] < 0.2, variances


def test_unusable_frozen_settings_are_refused():
    compiled = normals(1.0, 1.0)
    with pytest.raises(ValueError, match="go together"):
        compiled._experimental_nuts({}, chains=1, inverse_metrics=[[1.0, 1.0]])
    with pytest.raises(ValueError, match="positive"):
        compiled._experimental_nuts({}, chains=1, inverse_metrics=[[1.0, -1.0]], step_sizes=[0.5])
    with pytest.raises(ValueError, match="one metric of dimension 2 per chain"):
        compiled._experimental_nuts({}, chains=2, inverse_metrics=[[1.0, 1.0]], step_sizes=[0.5])
