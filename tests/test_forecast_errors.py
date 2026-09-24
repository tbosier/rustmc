"""One exception rule across the forecasting models.

InferenceError: a Bayesian forecasting model refused its priors, settings or
data, or failed numerically. StateSpaceError: the fixed-parameter state-space
layer and structural specifications. Plain ValueError: argument checks every
model shares. All three stay ValueErrors.
"""
import numpy as np
import pytest

import rustmc as rmc

Y = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 3.5])
QUICK = dict(chains=1, draws=6, warmup=3)


def test_exception_classes_keep_their_bases():
    assert issubclass(rmc.StateSpaceError, ValueError)
    assert issubclass(rmc.InferenceError, ValueError)
    assert issubclass(rmc.ParameterError, ValueError)
    assert not issubclass(rmc.StateSpaceError, rmc.InferenceError)
    assert not issubclass(rmc.InferenceError, rmc.StateSpaceError)


def _prior():
    return rmc.InverseGammaPrior(3.0, 0.5)


def _level_component():
    return rmc.StructuralComponent.level("level", rmc.VarianceParameter.inverse_gamma(3.0, 0.5), 0.0, 1.0)


BAYESIAN_REFUSALS = {
    "inverse gamma prior": lambda: rmc.InverseGammaPrior(-1.0, 1.0),
    "local level prior": lambda: rmc.BayesianLocalLevel(_prior(), _prior(), initial_variance=-1.0),
    "local level data": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit([np.nan, np.nan], **QUICK),
    "local level settings": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit(Y, chains=0),
    "local level exog": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit(
        Y, coefficient_prior=rmc.GaussianCoefficientPrior([0.0], [[1.0]])),
    "regression horizon": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit(
        Y, exog=np.ones((6, 1)), coefficient_prior=rmc.GaussianCoefficientPrior([0.0], [[1.0]]), **QUICK,
    ).forecast(2, exog=np.ones((3, 1))),
    "seasonal effects": lambda: rmc.BayesianSeasonalLocalLevel(3, _prior(), _prior(), _prior(),
                                                              initial_seasonal_effects=[1.0, 1.0, 1.0]),
    "trend covariance": lambda: rmc.BayesianLocalLinearTrend(_prior(), _prior(), _prior(),
                                                            initial_level_slope_covariance=100.0),
    "AR order": lambda: rmc.BayesianAR(0, rmc.NormalInverseGammaPrior(np.zeros(1), np.eye(1), 3.0, 0.5)),
    "AR prior": lambda: rmc.NormalInverseGammaPrior(np.zeros(2), np.diag([1.0, 0.0]), 3.0, 0.5),
    "AR horizon": lambda: rmc.BayesianAR(1, rmc.NormalInverseGammaPrior(np.zeros(2), np.eye(2), 3.0, 0.5))
        .fit(Y, chains=1, draws=4).forecast(0),
    "coefficient prior": lambda: rmc.GaussianCoefficientPrior([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]]),
    "hierarchical groups": lambda: rmc.BayesianHierarchicalMean(_prior(), _prior(), _prior()).fit(
        [Y, Y], [0, 2], **QUICK),
    "hurdle data": lambda: rmc.BayesianHurdleLogNormal(_prior(), _prior()).fit([-1.0, 2.0], **QUICK),
    "structural fit": lambda: rmc.StructuralModel([_level_component()], rmc.VarianceParameter.fixed(1.0))
        .fit([np.nan], **QUICK),
    "structural states": lambda: rmc.StructuralModel([_level_component()], rmc.VarianceParameter.fixed(1.0))
        .fit(Y, **QUICK).states,
    "dynamic GLM family": lambda: rmc.BayesianDynamicGLM("binomial"),
    "dynamic GLM data": lambda: rmc.BayesianDynamicPoisson().fit([[-1.0, 2.0]], chains=1, warmup=2, draws=2),
    "runoff alpha": lambda: rmc.DirichletMultinomialRunoff([1.0]),
    "runoff counts": lambda: rmc.DirichletMultinomialRunoff([1.0, 1.0]).fit([[0.5, 1.0]], [0], 1),
    "batch cell": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit_batch([[]], ["empty"]),
}


@pytest.mark.parametrize("case", sorted(BAYESIAN_REFUSALS))
def test_bayesian_models_raise_inference_error(case):
    with pytest.raises(rmc.InferenceError):
        BAYESIAN_REFUSALS[case]()


STATE_SPACE_REFUSALS = {
    "non-square transition": lambda: rmc.LinearGaussianStateSpace(
        np.ones((1, 2)), [1.0], [[1.0]], 1.0, [0.0], [[1.0]]),
    "stationarity": lambda: rmc.LinearGaussianStateSpace.stationary_ar1(1.0, 0.36, 0.25),
    "filter data": lambda: rmc.LinearGaussianStateSpace.local_level(1.0, 1.0).filter([np.inf]),
    "future rows": lambda: rmc.LinearGaussianStateSpace.local_level(1.0, 1.0).forecast(
        [1.0], 2, future_observation_rows=np.ones((3, 1))),
    "variance parameter": lambda: rmc.VarianceParameter.fixed(-1.0),
    "structural component": lambda: rmc.StructuralComponent.trend(
        "trend", rmc.VarianceParameter.fixed(0.1), rmc.VarianceParameter.fixed(0.1), [0.0, 0.0], [[1.0]]),
    "structural model json": lambda: rmc.StructuralModel.from_json("{}"),
}


@pytest.mark.parametrize("case", sorted(STATE_SPACE_REFUSALS))
def test_state_space_layer_raises_state_space_error(case):
    with pytest.raises(rmc.StateSpaceError):
        STATE_SPACE_REFUSALS[case]()


def _forecast():
    return rmc.BayesianLocalLevel(_prior(), _prior()).fit(Y, **QUICK).forecast(2)


SHARED_ARGUMENT_CHECKS = {
    "interval level": lambda: _forecast().interval(1.5),
    "quantile probability": lambda: _forecast().observation_quantile(-0.1),
    "hierarchical probability": lambda: rmc.BayesianHierarchicalMean(_prior(), _prior(), _prior())
        .fit([Y], [0], **QUICK).forecast(2).observation_quantile(2.0),
    "state-space level": lambda: rmc.LinearGaussianStateSpace.local_level(1.0, 1.0).forecast([1.0], 2).interval(0.0),
    "array conversion": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit(["a", "b"]),
    "batch errors option": lambda: rmc.BayesianLocalLevel(_prior(), _prior()).fit_batch([Y], ["a"], errors="skip"),
    "fourier arguments": lambda: rmc.fourier_design(5, 12, 7),
}


@pytest.mark.parametrize("case", sorted(SHARED_ARGUMENT_CHECKS))
def test_shared_argument_checks_raise_plain_value_error(case):
    with pytest.raises(ValueError) as error:
        SHARED_ARGUMENT_CHECKS[case]()
    assert type(error.value) is ValueError
