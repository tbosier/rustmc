"""The repeated-inference harness keeps every failure and counts only passing ESS."""
import os
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("RUSTMC_REQUIRE_SITE_PACKAGES") == "1":
    pytest.skip(
        "the source-only benchmark harness is not part of the runtime wheel",
        allow_module_level=True,
    )

from benchmarks import repeated
from benchmarks.protocol import BenchmarkConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def config(**overrides):
    base = BenchmarkConfig(observations=40, parameters=2, chains=2, warmup=50, draws=400)
    return repeated.RepeatedConfig(base=base, **overrides)


def test_datasets_are_reproducible_and_ragged_rows_stay_in_range():
    ragged = config(datasets=12, ragged=(5, 9))
    first, second = repeated.make_datasets(ragged), repeated.make_datasets(ragged)
    assert [d.problem.digest for d in first] == [d.problem.digest for d in second]
    rows = [d.config.observations for d in first]
    assert all(5 <= n <= 9 for n in rows) and len(set(rows)) > 1
    assert len({d.problem.digest for d in first}) == 12
    fixed = repeated.make_datasets(config(datasets=3))
    assert [d.problem.x.shape for d in fixed] == [(40, 2)] * 3
    for d in fixed:
        np.testing.assert_allclose(d.expected_new, d.x_new @ d.problem.posterior_mean)


@pytest.mark.parametrize("bad", [dict(datasets=0), dict(predict_rows=0), dict(ragged=(0, 3)), dict(ragged=(4, 3))])
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        config(**bad).validate()


def exact_draws(dataset, chains=2, draws=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(
        dataset.problem.posterior_mean, dataset.posterior_covariance, size=(chains, draws))


def test_failures_and_errors_are_kept_and_cost_time_but_add_no_ess():
    run = repeated.EngineRun(setup_phases={"import": 1.0})
    datasets = repeated.make_datasets(config(datasets=3))
    good, biased, broken = datasets

    repeated.record_fit(run, good, lambda: (exact_draws(good), 0, exact_draws(good)), lambda d: d @ good.x_new.T)
    shifted = exact_draws(biased) + 10 * biased.problem.posterior_sd
    repeated.record_fit(run, biased, lambda: (shifted, 0, shifted), lambda d: d @ biased.x_new.T)

    def raises():
        raise RuntimeError("sampler fell over")

    repeated.record_fit(run, broken, raises, lambda d: d)
    assert [row["status"] for row in run.fits] == ["passed", "failed", "error"]
    assert "mean_rmse_exact_posterior_sd_units" in run.fits[1]["quality_gate"]["failures"]
    assert "sampler fell over" in run.fits[2]["reason"]

    assert "fit_seconds" in run.fits[2]
    for row in run.fits:
        row["fit_seconds"] = 2.0
    summary = repeated.summarize(run)
    assert (summary["passed"], summary["failed"], summary["errors"]) == (1, 1, 1)
    assert summary["total_fit_seconds"] == 6.0
    passing_ess = run.fits[0]["quality"]["ess_bulk_mean"]
    assert summary["useful_ess_per_second"] == pytest.approx(passing_ess / 7.0)


def test_the_batch_wall_time_is_not_counted_as_setup():
    run = repeated.EngineRun(setup_phases={"import": 1.0, "compile": 0.5, "batch_wall_seconds": 9.0})
    assert repeated.summarize(run)["setup_seconds"] == 1.5


def test_the_quick_config_loads():
    loaded = repeated.RepeatedConfig.from_json(REPO_ROOT / "benchmarks/configs/repeated-quick.json")
    assert loaded.ragged == (50, 400)
    assert loaded.to_dict()["base"]["parameters"] == 3


def test_stored_text_does_not_record_the_home_directory():
    home = str(Path.home())
    cleaned = repeated._without_home({"stderr": [f"{home}/x.py:1", 3], "n": 2})
    assert cleaned == {"stderr": ["~/x.py:1", 3], "n": 2}


def test_draws_with_the_right_marginals_but_wrong_correlations_fail():
    dataset = repeated.make_datasets(config(datasets=1))[0]
    p = dataset.problem
    shared = np.random.default_rng(1).normal(size=(2, 400, 1))
    rank_one = p.posterior_mean + p.posterior_sd * shared
    run = repeated.EngineRun()
    repeated.record_fit(run, dataset, lambda: (rank_one, 0, rank_one), lambda d: d @ dataset.x_new.T)
    assert run.fits[0]["status"] == "failed"
    assert "covariance_eigenvalue_error" in run.fits[0]["quality_gate"]["joint_failures"]


@pytest.mark.parametrize("bad", [np.nan, 1e12])
def test_unusable_predictions_fail(bad):
    dataset = repeated.make_datasets(config(datasets=1))[0]
    good = np.random.default_rng(2).multivariate_normal(
        dataset.problem.posterior_mean, dataset.posterior_covariance, size=(2, 400))
    run = repeated.EngineRun()
    repeated.record_fit(run, dataset, lambda: (good, 0, good), lambda d: np.full((2, 400, 10), bad))
    assert run.fits[0]["status"] == "failed"
    assert "prediction_error_sd_units" in run.fits[0]["quality_gate"]["joint_failures"]


def test_predictions_without_their_spread_or_shape_fail():
    dataset = repeated.make_datasets(config(datasets=1))[0]
    good = exact_draws(dataset)
    exact_mean = np.broadcast_to(dataset.expected_new, (2, 400, 10))
    for predict, failure in [
        (lambda d: exact_mean, "prediction_sd_relative_error"),
        (lambda d: dataset.expected_new, "prediction_shape"),
    ]:
        run = repeated.EngineRun()
        repeated.record_fit(run, dataset, lambda: (good, 0, good), predict)
        assert run.fits[0]["status"] == "failed"
        assert failure in run.fits[0]["quality_gate"]["joint_failures"]


def test_a_lost_direction_fails_in_fifty_dimensions():
    base = BenchmarkConfig(observations=400, parameters=50, chains=2, warmup=50, draws=2000)
    dataset = repeated.make_datasets(repeated.RepeatedConfig(base=base, datasets=1))[0]
    rng = np.random.default_rng(3)
    lower = np.linalg.cholesky(dataset.posterior_covariance)
    z = rng.normal(size=(2, 2000, 50))
    z[..., 0] = 0.0  # one whitened direction never explored
    draws = dataset.problem.posterior_mean + z @ lower.T
    run = repeated.EngineRun()
    repeated.record_fit(run, dataset, lambda: (draws, 0, draws), lambda d: d @ dataset.x_new.T)
    assert "covariance_eigenvalue_error" in run.fits[0]["quality_gate"]["joint_failures"]
    whole = dataset.problem.posterior_mean + rng.normal(size=(2, 2000, 50)) @ lower.T
    run = repeated.EngineRun()
    repeated.record_fit(run, dataset, lambda: (whole, 0, whole), lambda d: d @ dataset.x_new.T)
    assert run.fits[0]["quality_gate"]["joint_failures"] == []


def test_one_coefficient_is_checked_not_an_error():
    base = BenchmarkConfig(observations=40, parameters=1, chains=2, warmup=50, draws=400)
    dataset = repeated.make_datasets(repeated.RepeatedConfig(base=base, datasets=1))[0]
    draws = exact_draws(dataset)
    run = repeated.EngineRun()
    repeated.record_fit(run, dataset, lambda: (draws, 0, draws), lambda d: d @ dataset.x_new.T)
    assert run.fits[0]["status"] == "passed"


def test_a_fit_that_raises_keeps_its_time():
    dataset = repeated.make_datasets(config(datasets=1))[0]

    def slow_failure():
        import time
        time.sleep(0.05)
        raise RuntimeError("diverged badly")

    run = repeated.EngineRun()
    repeated.record_fit(run, dataset, slow_failure, lambda d: d)
    assert run.fits[0]["status"] == "error"
    assert run.fits[0]["fit_seconds"] >= 0.05
    assert repeated.summarize(run)["total_fit_seconds"] >= 0.05
