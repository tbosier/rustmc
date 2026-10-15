"""Per-fit triage: E-BFMI, fit warnings and batch flagging.

The reference pair is the eight schools model in its centred form (Rubin 1981;
Gelman et al., Bayesian Data Analysis, section 5.5). With the published
standard errors the posterior has a funnel between the group scale and the
school effects that NUTS cannot follow at the shipped target acceptance of
0.8; with standard errors of 1 the data pin every school down and the same
model samples cleanly. The model is written with a potential because the
builder non-centres scalar hierarchies automatically and has no vector one.
"""
import numpy as np
import pytest

import rustmc

SEEDS = [1, 2, 3]
Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
HAMILTONIAN_ONLY = {"divergences", "e_bfmi", "tree_depth_saturation"}
# Single-fit defaults, written out for the batch API, whose defaults are
# smaller (one chain, 500 draws, tree depth 8).
SHIPPED = dict(chains=4, draws=1000, warmup=500, max_tree_depth=10)


def centred_eight_schools():
    m = rustmc.ModelBuilder()
    mu = m.normal_prior("mu", 0.0, 5.0)
    tau = m.half_normal_prior("tau", 5.0)
    # A flat-in-practice prior; the hierarchy is the potential below.
    theta = m.vector_normal_prior("theta", 8, 0.0, 1000.0)
    school = theta["school"]
    m.potential("theta_prior", (-0.5 * ((school - mu) / tau) ** 2 - tau.log()).sum())
    # y / sigma ~ Normal(theta / sigma, 1) is y ~ Normal(theta, sigma) up to a
    # constant.
    m.normal_likelihood("y", school * "inv_sigma", 1.0, "y_scaled")
    return m.compile()


def schools_data(y, sigma):
    return {"school": np.arange(8.0), "inv_sigma": 1.0 / sigma, "y_scaled": y / sigma}


FUNNEL = schools_data(Y, SIGMA)
INFORMATIVE = schools_data(np.linspace(-15.0, 15.0, 8), np.ones(8))


def codes(warnings, status="fail"):
    return {warning["code"] for warning in warnings if warning["status"] == status}


def assert_messages_state_thresholds(warnings):
    """Each message states its threshold in one of the rule's phrasings; a bare
    number is not enough, since "7 of 1000" contains the 0 of a count check."""
    for warning in warnings:
        t = format(warning["threshold"], "g")
        phrasings = [f"the {t} threshold", f"threshold of {t}", f"fails above {t}",
                     f"fails below {t}", f"the {t} they need", f"at least {t} "]
        assert any(p in warning["message"] for p in phrasings), warning


def assert_hamiltonian_checks_unavailable_once(warnings):
    for code in HAMILTONIAN_ONLY:
        records = [w for w in warnings if w["code"] == code]
        assert len(records) == 1, (code, warnings)
        assert records[0]["status"] == "unavailable"
        assert records[0]["value"] is None


def test_e_bfmi_equals_arviz_bfmi_on_the_exported_energies():
    az = pytest.importorskip("arviz")
    compiled = centred_eight_schools()
    for sampler in ["nuts", "hmc"]:
        fit = compiled.sample(FUNNEL, seed=4, sampler=sampler, show_progress=False)
        idata = fit.to_arviz(include_log_likelihood=False)
        stats = getattr(idata.sample_stats, "dataset", idata.sample_stats)
        assert stats["energy"].shape == (4, 1000)
        assert ("tree_depth" in stats) == (sampler == "nuts")
        ours = [chain["e_bfmi"] for chain in fit.transition_diagnostics()["chains"]]
        np.testing.assert_allclose(ours, az.bfmi(idata), rtol=1e-12, atol=0)


def test_transition_diagnostics_report_e_bfmi_and_saturation():
    fit = centred_eight_schools().sample(INFORMATIVE, seed=1, show_progress=False)
    telemetry = fit.transition_diagnostics()
    assert telemetry["max_tree_depth_limit"] == 10
    assert telemetry["total_tree_depth_saturation"] == sum(
        chain["tree_depth_saturation"] for chain in telemetry["chains"]
    )
    for chain in telemetry["chains"]:
        assert chain["e_bfmi"] > 0.3
    hmc = centred_eight_schools().sample(
        INFORMATIVE, seed=1, sampler="hmc", num_leapfrog_steps=10, show_progress=False
    )
    telemetry = hmc.transition_diagnostics()
    assert telemetry["max_tree_depth_limit"] is None
    assert telemetry["total_tree_depth_saturation"] is None
    assert all(chain["tree_depth_saturation"] is None for chain in telemetry["chains"])


@pytest.mark.parametrize("seed", SEEDS)
def test_centred_eight_schools_is_flagged_and_informative_data_is_not(seed):
    compiled = centred_eight_schools()
    funnel = compiled.sample(FUNNEL, seed=seed, show_progress=False).warnings()
    assert "divergences" in codes(funnel), funnel
    assert_messages_state_thresholds(funnel)
    clean = compiled.sample(INFORMATIVE, seed=seed, show_progress=False).warnings()
    assert clean == []


@pytest.mark.parametrize("seed", SEEDS)
def test_independent_normals_raise_nothing(seed):
    m = rustmc.ModelBuilder({"y": np.zeros(1)})
    z = m.vector_normal_prior("z", 10, 0.0, 1.0)
    # A likelihood that does not involve z leaves its posterior the prior.
    m.normal_likelihood("y", m.normal_prior("c", 0.0, 1.0), 1.0, "y")
    del z
    fit = m.compile().sample({}, seed=seed, show_progress=False)
    assert fit.warnings() == []


def test_batch_flagged_returns_exactly_the_injected_bad_cells():
    compiled = centred_eight_schools()
    broken = {"school": np.arange(8.0), "inv_sigma": np.ones(8)}  # no response
    datasets = [INFORMATIVE, FUNNEL, INFORMATIVE, broken, FUNNEL, INFORMATIVE]
    ids = ["clean-1", "funnel-1", "clean-2", "broken", "funnel-2", "clean-3"]
    batch = compiled.sample_batch(
        datasets, ids=ids, seed=11, errors="collect", show_progress=False, **SHIPPED
    )
    assert list(batch.errors) == ["broken"]
    assert batch.flagged() == ["funnel-1", "funnel-2"]
    for id in ["clean-1", "clean-2", "clean-3"]:
        assert batch.get(id).warnings() == []
    # Ignoring every code the funnel cells fail clears them.
    failed = set().union(*(codes(batch.get(id).warnings()) for id in ["funnel-1", "funnel-2"]))
    assert batch.flagged(ignore=sorted(failed)) == []
    assert batch.flagged(ignore=["known_limitation"]) == ["funnel-1", "funnel-2"]


def test_flagged_refuses_an_unknown_code():
    batch = centred_eight_schools().sample_batch(
        [INFORMATIVE], ids=["a"], show_progress=False
    )
    with pytest.raises(ValueError, match='unknown warning code "rhat"'):
        batch.flagged(ignore=["rhat"])


def local_level():
    prior = rustmc.InverseGammaPrior(3.0, 0.4)
    return rustmc.BayesianLocalLevel(
        process_variance_prior=prior, observation_variance_prior=prior
    )


def local_linear_trend():
    return rustmc.BayesianLocalLinearTrend(
        level_variance_prior=rustmc.InverseGammaPrior(3.0, 0.24),
        slope_variance_prior=rustmc.InverseGammaPrior(3.0, 0.04),
        observation_variance_prior=rustmc.InverseGammaPrior(3.0, 0.70),
        initial_level=0.0,
        initial_slope=0.0,
        initial_level_variance=4.0,
        initial_slope_variance=1.0,
    )


def series(seed, n=80):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, 0.3, n)) + rng.normal(0.0, 0.5, n)


def test_gibbs_fits_report_hamiltonian_checks_as_unavailable_never_as_passes():
    fit = local_level().fit(series(0), chains=4, draws=300, warmup=200, seed=3)
    warnings = fit.warnings()
    assert_hamiltonian_checks_unavailable_once(warnings)
    assert_messages_state_thresholds(warnings)
    assert all(warning["status"] in {"fail", "unavailable"} for warning in warnings)


def test_every_forecasting_fit_class_reports_the_unavailable_checks():
    ar = rustmc.BayesianAutoRegression(
        1, rustmc.NormalInverseGammaPrior([0.0, 0.0], np.eye(2), 3.0, 1.0)
    )
    fits = [
        local_level().fit(series(1), chains=2, draws=50, warmup=50, seed=1),
        local_linear_trend().fit(series(1), chains=2, draws=50, warmup=50, seed=1),
        ar.fit(series(1), chains=2, draws=50, seed=1),
    ]
    for fit in fits:
        assert_hamiltonian_checks_unavailable_once(fit.warnings())


def test_local_linear_trend_carries_its_known_limitation_into_triage():
    fit = local_linear_trend().fit(series(2), chains=2, draws=100, warmup=100, seed=2)
    limitation = [w for w in fit.warnings() if w["code"] == "known_limitation"]
    assert len(limitation) == 1
    assert limitation[0]["status"] == "fail"
    assert limitation[0]["threshold"] == 1.01
    assert "1.01" in limitation[0]["message"]
    assert "docs/reference.md" in limitation[0]["message"]

    # Autoregressions are exact independent draws, so their cells are clean;
    # the trend cells are flagged by the limitation until it is ignored.
    ar = rustmc.BayesianAutoRegression(
        1, rustmc.NormalInverseGammaPrior([0.0, 0.0], np.eye(2), 3.0, 1.0)
    )
    ids = ["ar-1", "trend-1", "ar-2", "trend-2"]
    batch = ar.fit_batch(
        [series(seed) for seed in range(4)],
        ids=ids,
        models=[None, local_linear_trend(), None, local_linear_trend()],
        chains=4,
        draws=1000,
        warmup=500,
        seed=7,
    )
    warnings = batch.warnings()
    assert list(warnings) == ids
    for id in ["ar-1", "ar-2"]:
        assert codes(warnings[id]) == set(), warnings[id]
        assert_hamiltonian_checks_unavailable_once(warnings[id])
    assert batch.flagged() == ["trend-1", "trend-2"]
    still_failing = [
        id for id in ["trend-1", "trend-2"] if codes(warnings[id]) - {"known_limitation"}
    ]
    assert batch.flagged(ignore=["known_limitation"]) == still_failing
    for cell in warnings.values():
        assert_messages_state_thresholds(cell)


def test_forecast_batch_warnings_mirror_diagnostics_for_failed_cells():
    batch = local_level().fit_batch(
        [series(0), np.array([1.0])],
        ids=["ok", "short"],
        chains=2,
        draws=40,
        warmup=40,
        errors="collect",
    )
    assert batch.warnings()["short"] is None
    assert "short" not in batch.flagged()
    assert "short" in batch.errors
