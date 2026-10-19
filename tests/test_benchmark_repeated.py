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
    p = dataset.problem
    return p.posterior_mean + p.posterior_sd * rng.normal(size=(chains, draws, p.posterior_mean.size))


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

    for row in run.fits[:2]:
        row["fit_seconds"] = 2.0
    summary = repeated.summarize(run)
    assert (summary["passed"], summary["failed"], summary["errors"]) == (1, 1, 1)
    assert summary["total_fit_seconds"] == 4.0
    passing_ess = run.fits[0]["quality"]["ess_bulk_mean"]
    assert summary["useful_ess_per_second"] == pytest.approx(passing_ess / 5.0)


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
