// Graph-model surface: builder, compiled model, fits and batches.
mod arviz;
mod batch;
mod builder;
mod compiled;
mod data_input;
mod errors;
mod expressions;
mod fit_artifact;
mod fit_result;
mod generic_results;
mod model_artifact;
mod prediction_binding;
mod sampling;
// Shared with the forecasting modules, which name them from the crate root.
#[allow(unused_imports)]
pub(crate) use arviz::{
    arviz_api_generation, arviz_from_groups, arviz_from_groups_versioned, arviz_group,
    assign_posterior_predictive_draw_coords,
};
#[allow(unused_imports)]
pub(crate) use errors::{
    model_error, param_error, InferenceError, ParameterError, StateSpaceError,
};

mod dynamic_glm;
mod forecast_batch;
mod forecast_diagnostics;
mod hurdle;
mod regression;
mod runoff;
mod structural;
use ndarray::{Array2, Array3, Array4};
use numpy::PyUntypedArrayMethods;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_ar::{
    fit_bayesian_ar, BayesianArConfig as CoreBayesianArConfig,
    BayesianArForecast as CoreBayesianArForecast, BayesianArPosterior as CoreBayesianArPosterior,
    BayesianArPosteriorDraw as CoreBayesianArPosteriorDraw,
    NormalInverseGammaPrior as CoreNormalInverseGammaPrior,
};
use rustmc_core::bayesian_forecast::{
    fit_bayesian_local_level, BayesianForecastError as CoreBayesianForecastError,
    BayesianLocalLevelConfig as CoreBayesianLocalLevelConfig,
    InverseGammaPrior as CoreInverseGammaPrior, LocalLevelPosterior as CoreLocalLevelPosterior,
    LocalLevelPosteriorDraw as CoreLocalLevelPosteriorDraw,
    PosteriorPredictiveForecast as CorePosteriorPredictiveForecast,
};
use rustmc_core::bayesian_seasonal::{
    fit_bayesian_seasonal_local_level,
    BayesianSeasonalLocalLevelConfig as CoreBayesianSeasonalLocalLevelConfig,
    SeasonalLocalLevelPosterior as CoreSeasonalLocalLevelPosterior,
    SeasonalLocalLevelPosteriorDraw as CoreSeasonalLocalLevelPosteriorDraw,
    SeasonalPosteriorPredictiveForecast as CoreSeasonalPosteriorPredictiveForecast,
};
use rustmc_core::bayesian_trend::{
    fit_bayesian_local_linear_trend,
    BayesianLocalLinearTrendConfig as CoreBayesianLocalLinearTrendConfig,
    LocalLinearTrendPosterior as CoreLocalLinearTrendPosterior,
    LocalLinearTrendPosteriorDraw as CoreLocalLinearTrendPosteriorDraw,
    TrendPosteriorPredictiveForecast as CoreTrendPosteriorPredictiveForecast,
};
use rustmc_core::diagnostics::inv_normal_cdf;
use rustmc_core::hierarchical::{
    fit_hierarchical_mean, HierarchicalMeanConfig as CoreHierarchicalMeanConfig,
    HierarchicalMeanForecast as CoreHierarchicalMeanForecast,
    HierarchicalMeanPosterior as CoreHierarchicalMeanPosterior,
    HierarchicalMeanPosteriorDraw as CoreHierarchicalMeanPosteriorDraw,
};
use rustmc_core::state_space::{
    ForecastResult as CoreForecastResult, KalmanFilterResult as CoreKalmanFilterResult,
    KalmanSmootherResult as CoreKalmanSmootherResult,
    LinearGaussianStateSpace as CoreLinearGaussianStateSpace,
    StateSpaceError as CoreStateSpaceError,
};

type PyIntervalArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type PyIntervalMatrices<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

fn state_space_error(error: CoreStateSpaceError) -> PyErr {
    StateSpaceError::new_err(error.to_string())
}

fn bayesian_forecast_error(error: CoreBayesianForecastError) -> PyErr {
    StateSpaceError::new_err(error.to_string())
}

fn hierarchical_error(error: CoreBayesianForecastError) -> PyErr {
    InferenceError::new_err(error.to_string())
}

fn state_space_matrix(name: &str, value: PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize)> {
    let shape = value.shape();
    if shape[0] != shape[1] {
        return Err(StateSpaceError::new_err(format!(
            "invalid dimension: {name} must be a square matrix"
        )));
    }
    Ok((value.as_array().iter().copied().collect(), shape[0]))
}

fn state_space_vector(value: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    value.as_array().iter().copied().collect()
}

fn state_means_array<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
    dimension: usize,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((values.len(), dimension), |(time, state)| {
        values[time][state]
    })
    .into_pyarray(py)
}

fn state_covariances_array<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
    dimension: usize,
) -> Bound<'py, PyArray3<f64>> {
    Array3::from_shape_fn(
        (values.len(), dimension, dimension),
        |(time, row, column)| values[time][row * dimension + column],
    )
    .into_pyarray(py)
}

/// A linear Gaussian state-space model with scalar observations and constant
/// transition and process matrices. Observation rows may vary by time.
/// Initial moments describe the state immediately before the first observation;
/// filtering performs one prediction before updating on observations[0].
#[pyclass(name = "LinearGaussianStateSpace", module = "rustmc")]
#[derive(Clone)]
struct PyLinearGaussianStateSpace {
    inner: CoreLinearGaussianStateSpace,
}

#[pymethods]
impl PyLinearGaussianStateSpace {
    #[new]
    #[pyo3(signature = (transition, observation, process_covariance, observation_variance, initial_mean, initial_covariance))]
    fn new(
        transition: PyReadonlyArray2<'_, f64>,
        observation: PyReadonlyArray1<'_, f64>,
        process_covariance: PyReadonlyArray2<'_, f64>,
        observation_variance: f64,
        initial_mean: PyReadonlyArray1<'_, f64>,
        initial_covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let (transition, dimension) = state_space_matrix("transition", transition)?;
        let (process_covariance, process_dimension) =
            state_space_matrix("process_covariance", process_covariance)?;
        let (initial_covariance, initial_dimension) =
            state_space_matrix("initial_covariance", initial_covariance)?;
        if process_dimension != dimension || initial_dimension != dimension {
            return Err(StateSpaceError::new_err(
                "invalid dimension: covariance matrices must match the transition matrix",
            ));
        }
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::new(
                dimension,
                transition,
                state_space_vector(observation),
                process_covariance,
                observation_variance,
                state_space_vector(initial_mean),
                initial_covariance,
            )
            .map_err(state_space_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (process_variance, observation_variance, initial_mean=0.0, initial_variance=1.0))]
    fn local_level(
        process_variance: f64,
        observation_variance: f64,
        initial_mean: f64,
        initial_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::local_level(
                process_variance,
                observation_variance,
                initial_mean,
                initial_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (level_variance, trend_variance, observation_variance, initial_level=0.0, initial_trend=0.0, initial_level_variance=1.0, initial_trend_variance=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn local_linear_trend(
        level_variance: f64,
        trend_variance: f64,
        observation_variance: f64,
        initial_level: f64,
        initial_trend: f64,
        initial_level_variance: f64,
        initial_trend_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::local_linear_trend(
                level_variance,
                trend_variance,
                observation_variance,
                initial_level,
                initial_trend,
                initial_level_variance,
                initial_trend_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    /// Construct a local-level model with sum-to-zero dummy seasonality.
    /// `initial_seasonal_effects`, when supplied, is one complete cycle in
    /// forecast order and must sum to zero.
    #[staticmethod]
    #[pyo3(signature = (period, level_variance, seasonal_variance, observation_variance, initial_level=0.0, initial_seasonal_effects=None, initial_level_variance=1.0, initial_seasonal_variance=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn seasonal_local_level(
        period: usize,
        level_variance: f64,
        seasonal_variance: f64,
        observation_variance: f64,
        initial_level: f64,
        initial_seasonal_effects: Option<Vec<f64>>,
        initial_level_variance: f64,
        initial_seasonal_variance: f64,
    ) -> PyResult<Self> {
        let effects = initial_seasonal_effects.unwrap_or_else(|| vec![0.0; period]);
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::seasonal_local_level(
                period,
                level_variance,
                seasonal_variance,
                observation_variance,
                initial_level,
                effects,
                initial_level_variance,
                initial_seasonal_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    /// Construct a zero-mean stationary AR(1) latent process observed with
    /// independent Gaussian noise.
    #[staticmethod]
    fn stationary_ar1(
        coefficient: f64,
        process_variance: f64,
        observation_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::stationary_ar1(
                coefficient,
                process_variance,
                observation_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Return a model with a finite observation row for each training time.
    fn with_observation_rows(&self, observation_rows: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_observation_rows(regression::rows(observation_rows))
                .map_err(state_space_error)?,
        })
    }

    fn filter(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyKalmanFilterResult> {
        let observations = state_space_vector(observations);
        let result = py
            .allow_threads(|| self.inner.filter(&observations))
            .map_err(state_space_error)?;
        Ok(PyKalmanFilterResult::new(result, self.inner.dimension()))
    }

    fn smooth(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyKalmanSmootherResult> {
        let observations = state_space_vector(observations);
        let result = py
            .allow_threads(|| self.inner.smooth(&observations))
            .map_err(state_space_error)?;
        Ok(PyKalmanSmootherResult::new(result, self.inner.dimension()))
    }

    #[pyo3(signature=(observations, steps, *, future_observation_rows=None))]
    fn forecast(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        steps: usize,
        future_observation_rows: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<PyForecastResult> {
        let observations = state_space_vector(observations);
        let future_rows = future_observation_rows.map(regression::rows);
        if future_rows.as_ref().is_some_and(|rows| rows.len() != steps) {
            return Err(StateSpaceError::new_err(
                "future observation row count must equal steps",
            ));
        }
        let result = py
            .allow_threads(|| match future_rows {
                Some(rows) => self
                    .inner
                    .forecast_with_observation_rows(&observations, &rows),
                None => self.inner.forecast(&observations, steps),
            })
            .map_err(state_space_error)?;
        Ok(PyForecastResult::new(result, self.inner.dimension()))
    }
}

#[pyclass(name = "KalmanFilterResult", module = "rustmc")]
struct PyKalmanFilterResult {
    inner: CoreKalmanFilterResult,
    dimension: usize,
}

impl PyKalmanFilterResult {
    fn new(inner: CoreKalmanFilterResult, dimension: usize) -> Self {
        Self { inner, dimension }
    }
}

#[pymethods]
impl PyKalmanFilterResult {
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    #[getter]
    fn predicted_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.predicted_means, self.dimension)
    }

    #[getter]
    fn predicted_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.predicted_covariances, self.dimension)
    }

    #[getter]
    fn filtered_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.filtered_means, self.dimension)
    }

    #[getter]
    fn filtered_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.filtered_covariances, self.dimension)
    }
}

#[pyclass(name = "KalmanSmootherResult", module = "rustmc")]
struct PyKalmanSmootherResult {
    inner: CoreKalmanSmootherResult,
    dimension: usize,
}

impl PyKalmanSmootherResult {
    fn new(inner: CoreKalmanSmootherResult, dimension: usize) -> Self {
        Self { inner, dimension }
    }
}

#[pymethods]
impl PyKalmanSmootherResult {
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.filter.log_likelihood
    }

    #[getter]
    fn filtered_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.filter.filtered_means, self.dimension)
    }

    #[getter]
    fn filtered_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.filter.filtered_covariances, self.dimension)
    }

    #[getter]
    fn smoothed_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.smoothed_means, self.dimension)
    }

    #[getter]
    fn smoothed_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.smoothed_covariances, self.dimension)
    }
}

#[pyclass(name = "ForecastResult", module = "rustmc")]
struct PyForecastResult {
    inner: CoreForecastResult,
    dimension: usize,
}

impl PyForecastResult {
    fn new(inner: CoreForecastResult, dimension: usize) -> Self {
        Self { inner, dimension }
    }
}

#[pymethods]
impl PyForecastResult {
    #[getter]
    fn state_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.state_means, self.dimension)
    }

    #[getter]
    fn state_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.state_covariances, self.dimension)
    }

    #[getter]
    fn observation_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.observation_means.clone().into_pyarray(py)
    }

    #[getter]
    fn observation_variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.observation_variances.clone().into_pyarray(py)
    }

    /// Joint covariance matrix across forecast observations. Off-diagonal
    /// entries retain the dependence needed for aggregate forecast intervals.
    #[getter]
    fn observation_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let steps = self.inner.observation_means.len();
        Array2::from_shape_fn((steps, steps), |(row, column)| {
            self.inner.observation_covariance[row * steps + column]
        })
        .into_pyarray(py)
    }

    /// Prefix-sum forecast means. Entry h-1 summarizes observations 1..h.
    #[getter]
    fn cumulative_observation_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .cumulative_observation_means
            .clone()
            .into_pyarray(py)
    }

    /// Prefix-sum forecast variances including cross-horizon covariance.
    #[getter]
    fn cumulative_observation_variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .cumulative_observation_variances
            .clone()
            .into_pyarray(py)
    }

    /// Pointwise Gaussian predictive interval conditional on the fixed model
    /// parameters. This does not include parameter-estimation uncertainty.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(PyValueError::new_err(
                "level must be finite and strictly between 0 and 1",
            ));
        }
        let critical = inv_normal_cdf(0.5 + level / 2.0);
        let mut lower = Vec::with_capacity(self.inner.observation_means.len());
        let mut upper = Vec::with_capacity(self.inner.observation_means.len());
        for (&mean, &variance) in self
            .inner
            .observation_means
            .iter()
            .zip(&self.inner.observation_variances)
        {
            let half_width = critical * variance.sqrt();
            lower.push(mean - half_width);
            upper.push(mean + half_width);
        }
        Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
    }

    /// Gaussian predictive intervals for cumulative observations 1..h,
    /// conditional on the fixed model parameters.
    #[pyo3(signature = (level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(PyValueError::new_err(
                "level must be finite and strictly between 0 and 1",
            ));
        }
        let critical = inv_normal_cdf(0.5 + level / 2.0);
        let mut lower = Vec::with_capacity(self.inner.cumulative_observation_means.len());
        let mut upper = Vec::with_capacity(self.inner.cumulative_observation_means.len());
        for (&mean, &variance) in self
            .inner
            .cumulative_observation_means
            .iter()
            .zip(&self.inner.cumulative_observation_variances)
        {
            let half_width = critical * variance.sqrt();
            lower.push(mean - half_width);
            upper.push(mean + half_width);
        }
        Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "conditional_fixed_parameters"
    }
}

fn local_level_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreLocalLevelPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreLocalLevelPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

fn local_level_path_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
) -> Bound<'py, PyArray3<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    let horizon = paths
        .first()
        .and_then(|chain| chain.first())
        .map_or(0, Vec::len);
    Array3::from_shape_fn((chains, draws, horizon), |(chain, draw, step)| {
        paths[chain][draw][step]
    })
    .into_pyarray(py)
}

fn hierarchical_scalar_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreHierarchicalMeanPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreHierarchicalMeanPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

fn hierarchical_vector_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreHierarchicalMeanPosterior,
    width: usize,
    value: F,
) -> Bound<'py, PyArray3<f64>>
where
    F: Fn(&CoreHierarchicalMeanPosteriorDraw, usize) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array3::from_shape_fn((chains, draws, width), |(chain, draw, index)| {
        value(&posterior.chains[chain][draw], index)
    })
    .into_pyarray(py)
}

fn hierarchical_path_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    Array4::from_shape_fn(
        (chains, draws, programs, steps),
        |(chain, draw, program, step)| paths[chain][draw][program * steps + step],
    )
    .into_pyarray(py)
}

fn hierarchical_path_summary(
    paths: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
    probability: Option<f64>,
) -> Array2<f64> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    Array2::from_shape_fn((programs, steps), |(program, step)| {
        let flat_index = program * steps + step;
        if let Some(probability) = probability {
            let mut values = Vec::with_capacity(chains * draws);
            for chain in paths {
                for draw in chain {
                    values.push(draw[flat_index]);
                }
            }
            values.sort_by(f64::total_cmp);
            let index = probability * (values.len() - 1) as f64;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            let weight = index - lower as f64;
            values[lower] * (1.0 - weight) + values[upper] * weight
        } else {
            paths
                .iter()
                .flat_map(|chain| chain.iter())
                .map(|draw| draw[flat_index])
                .sum::<f64>()
                / (chains * draws) as f64
        }
    })
}

fn hierarchical_state_array<'py>(
    py: Python<'py>,
    states: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let chains = states.len();
    let draws = states.first().map_or(0, Vec::len);
    Array4::from_shape_fn(
        (chains, draws, programs, steps),
        |(chain, draw, program, _step)| states[chain][draw][program],
    )
    .into_pyarray(py)
}

fn hierarchical_state_summary(
    states: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
    probability: Option<f64>,
) -> Array2<f64> {
    let chains = states.len();
    let draws = states.first().map_or(0, Vec::len);
    let by_program = (0..programs)
        .map(|program| {
            if let Some(probability) = probability {
                let mut values = states
                    .iter()
                    .flat_map(|chain| chain.iter())
                    .map(|draw| draw[program])
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let index = probability * (values.len() - 1) as f64;
                let lower = index.floor() as usize;
                let upper = index.ceil() as usize;
                let weight = index - lower as f64;
                values[lower] * (1.0 - weight) + values[upper] * weight
            } else {
                states
                    .iter()
                    .flat_map(|chain| chain.iter())
                    .map(|draw| draw[program])
                    .sum::<f64>()
                    / (chains * draws) as f64
            }
        })
        .collect::<Vec<_>>();
    Array2::from_shape_fn((programs, steps), |(program, _step)| by_program[program])
}

fn hierarchical_group_rollup_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    group_index: &[usize],
    group_count: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    let mut rollups = Array4::zeros((chains, draws, group_count, steps));
    for chain in 0..chains {
        for draw in 0..draws {
            for (program, &group) in group_index.iter().enumerate() {
                for step in 0..steps {
                    rollups[(chain, draw, group, step)] +=
                        paths[chain][draw][program * steps + step];
                }
            }
        }
    }
    rollups.into_pyarray(py)
}

fn hierarchical_total_rollup_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
) -> Bound<'py, PyArray3<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    Array3::from_shape_fn((chains, draws, steps), |(chain, draw, step)| {
        (0..programs)
            .map(|program| paths[chain][draw][program * steps + step])
            .sum()
    })
    .into_pyarray(py)
}

/// Joint population -> group -> program Gaussian partial-pooling model.
///
/// Ragged program series are fitted in one conjugate Gibbs posterior. This
/// structure-aware sampler draws exact full conditionals and therefore avoids
/// requiring NUTS to traverse a hierarchical funnel.
#[pyclass(name = "BayesianHierarchicalMean", frozen, module = "rustmc")]
#[derive(Clone)]
struct PyBayesianHierarchicalMean {
    population_mean_prior: f64,
    population_variance_prior: f64,
    group_variance_prior: CoreInverseGammaPrior,
    program_variance_prior: CoreInverseGammaPrior,
    observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianHierarchicalMean {
    #[new]
    #[pyo3(signature = (group_variance_prior, program_variance_prior, observation_variance_prior, population_mean_prior=0.0, population_variance_prior=100.0))]
    fn new(
        group_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        program_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        population_mean_prior: f64,
        population_variance_prior: f64,
    ) -> PyResult<Self> {
        if !population_mean_prior.is_finite() {
            return Err(InferenceError::new_err(
                "invalid configuration: population mean prior must be finite",
            ));
        }
        if !population_variance_prior.is_finite() || population_variance_prior <= 0.0 {
            return Err(InferenceError::new_err(
                "invalid configuration: population variance prior must be finite and strictly positive",
            ));
        }
        Ok(Self {
            population_mean_prior,
            population_variance_prior,
            group_variance_prior: group_variance_prior.inner,
            program_variance_prior: program_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn population_mean_prior(&self) -> f64 {
        self.population_mean_prior
    }

    #[getter]
    fn population_variance_prior(&self) -> f64 {
        self.population_variance_prior
    }

    #[getter]
    fn group_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.group_variance_prior,
        }
    }

    #[getter]
    fn program_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.program_variance_prior,
        }
    }

    #[getter]
    fn observation_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.observation_variance_prior,
        }
    }

    #[pyo3(signature = (series, group_index, program_names=None, group_names=None, chains=4, draws=1000, warmup=500, thin=1, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        series: Vec<PyReadonlyArray1<'_, f64>>,
        group_index: Vec<usize>,
        program_names: Option<Vec<String>>,
        group_names: Option<Vec<String>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
    ) -> PyResult<PyBayesianHierarchicalMeanFit> {
        let series = series
            .into_iter()
            .map(state_space_vector)
            .collect::<Vec<_>>();
        let program_count = series.len();
        if group_index.len() != program_count {
            return Err(InferenceError::new_err(
                "series and group_index must have the same length",
            ));
        }
        let mut present_groups = vec![false; program_count];
        for &group in &group_index {
            if group >= program_count {
                return Err(InferenceError::new_err(
                    "group indices must be contiguous from zero with no empty groups",
                ));
            }
            present_groups[group] = true;
        }
        let inferred_group_count = group_index
            .iter()
            .copied()
            .max()
            .map_or(0, |value| value + 1);
        if present_groups[..inferred_group_count].contains(&false) {
            return Err(InferenceError::new_err(
                "group indices must be contiguous from zero with no empty groups",
            ));
        }
        let program_names = program_names.unwrap_or_else(|| {
            (0..program_count)
                .map(|index| format!("program_{index}"))
                .collect()
        });
        let group_names = group_names.unwrap_or_else(|| {
            (0..inferred_group_count)
                .map(|index| format!("group_{index}"))
                .collect()
        });
        validate_unique_names(&program_names, program_count, "program_names")?;
        validate_unique_names(&group_names, inferred_group_count, "group_names")?;
        let time_counts = series.iter().map(Vec::len).collect::<Vec<_>>();
        let config = CoreHierarchicalMeanConfig {
            population_mean_prior: self.population_mean_prior,
            population_variance_prior: self.population_variance_prior,
            group_variance_prior: self.group_variance_prior,
            program_variance_prior: self.program_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_hierarchical_mean(&series, &group_index, &config))
            .map_err(hierarchical_error)?;
        Ok(PyBayesianHierarchicalMeanFit {
            posterior,
            time_counts,
            program_names,
            group_names,
            config,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianHierarchicalMean(population_mean_prior={}, population_variance_prior={}, group_variance_prior=({}, {}), program_variance_prior=({}, {}), observation_variance_prior=({}, {}))",
            self.population_mean_prior,
            self.population_variance_prior,
            self.group_variance_prior.shape,
            self.group_variance_prior.scale,
            self.program_variance_prior.shape,
            self.program_variance_prior.scale,
            self.observation_variance_prior.shape,
            self.observation_variance_prior.scale,
        )
    }
}

fn validate_unique_names(names: &[String], expected: usize, field: &str) -> PyResult<()> {
    if names.len() != expected {
        return Err(InferenceError::new_err(format!(
            "{field} must contain exactly {expected} entries"
        )));
    }
    let unique = names.iter().collect::<std::collections::HashSet<_>>();
    if unique.len() != names.len() {
        return Err(InferenceError::new_err(format!(
            "{field} entries must be unique"
        )));
    }
    Ok(())
}

#[pyclass(name = "BayesianHierarchicalMeanFit", module = "rustmc")]
struct PyBayesianHierarchicalMeanFit {
    posterior: CoreHierarchicalMeanPosterior,
    time_counts: Vec<usize>,
    program_names: Vec<String>,
    group_names: Vec<String>,
    config: CoreHierarchicalMeanConfig,
}

#[pymethods]
impl PyBayesianHierarchicalMeanFit {
    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
    }

    #[getter]
    fn program_count(&self) -> usize {
        self.posterior.program_count()
    }

    #[getter]
    fn group_count(&self) -> usize {
        self.posterior.group_count
    }

    #[getter]
    fn time_counts(&self) -> Vec<usize> {
        self.time_counts.clone()
    }

    #[getter]
    fn observed_counts(&self) -> Vec<usize> {
        self.posterior.observed_counts.clone()
    }

    #[getter]
    fn total_observed_count(&self) -> usize {
        self.posterior.observed_counts.iter().sum()
    }

    #[getter]
    fn group_index(&self) -> Vec<usize> {
        self.posterior.group_index.clone()
    }

    #[getter]
    fn program_names(&self) -> Vec<String> {
        self.program_names.clone()
    }

    #[getter]
    fn group_names(&self) -> Vec<String> {
        self.group_names.clone()
    }

    #[getter]
    fn warmup(&self) -> usize {
        self.config.num_warmup
    }

    #[getter]
    fn thin(&self) -> usize {
        self.config.thinning
    }

    #[getter]
    fn inference_method(&self) -> &'static str {
        "conjugate_gibbs"
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        samples.set_item(
            "population_mean",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.population_mean),
        )?;
        samples.set_item(
            "group_variance",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.group_variance),
        )?;
        samples.set_item(
            "program_variance",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.program_variance),
        )?;
        samples.set_item(
            "observation_variance",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.observation_variance),
        )?;
        samples.set_item(
            "group_sd",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.group_variance.sqrt()),
        )?;
        samples.set_item(
            "program_sd",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.program_variance.sqrt()),
        )?;
        samples.set_item(
            "observation_sd",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.observation_variance.sqrt()),
        )?;
        samples.set_item(
            "group_mean",
            hierarchical_vector_array(py, &self.posterior, self.group_count(), |draw, index| {
                draw.group_means[index]
            }),
        )?;
        samples.set_item(
            "program_mean",
            hierarchical_vector_array(py, &self.posterior, self.program_count(), |draw, index| {
                draw.program_means[index]
            }),
        )?;
        Ok(samples)
    }

    /// Formatted rank-normalized R-hat, ESS, MCSE, and HDI diagnostics.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: conjugate Gibbs (acceptance and divergences unavailable)",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "conjugate Gibbs",
            self.chains(),
            self.draws(),
            "all retained hierarchical parameters",
        )
    }

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
    ) -> PyResult<PyBayesianHierarchicalForecast> {
        let inner = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(hierarchical_error)?;
        Ok(PyBayesianHierarchicalForecast {
            inner,
            program_names: self.program_names.clone(),
            group_names: self.group_names.clone(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianHierarchicalMeanFit(chains={}, draws={}, programs={}, groups={}, observations={})",
            self.chains(), self.draws(), self.program_count(), self.group_count(), self.total_observed_count()
        )
    }
}

#[pyclass(name = "BayesianHierarchicalForecast", module = "rustmc")]
struct PyBayesianHierarchicalForecast {
    inner: CoreHierarchicalMeanForecast,
    program_names: Vec<String>,
    group_names: Vec<String>,
}

#[pymethods]
impl PyBayesianHierarchicalForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.chain_count()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draw_count()
    }

    #[getter]
    fn program_count(&self) -> usize {
        self.inner.program_count()
    }

    #[getter]
    fn group_count(&self) -> usize {
        self.inner.group_count
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    fn group_index(&self) -> Vec<usize> {
        self.inner.group_index.clone()
    }

    #[getter]
    fn program_names(&self) -> Vec<String> {
        self.program_names.clone()
    }

    #[getter]
    fn group_names(&self) -> Vec<String> {
        self.group_names.clone()
    }

    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        hierarchical_state_array(
            py,
            &self.inner.state_means,
            self.program_count(),
            self.steps(),
        )
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        hierarchical_path_array(
            py,
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
        )
    }

    /// Draw-wise group totals indexed `(chain, draw, group, step)`.
    #[getter]
    fn group_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        hierarchical_group_rollup_array(
            py,
            &self.inner.observation_paths,
            &self.inner.group_index,
            self.inner.group_count,
            self.steps(),
        )
    }

    /// Draw-wise total across all programs, shaped `(chain, draw, step)`.
    #[getter]
    fn total_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        hierarchical_total_rollup_array(
            py,
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
        )
    }

    #[getter]
    fn state_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        hierarchical_state_summary(
            &self.inner.state_means,
            self.program_count(),
            self.steps(),
            None,
        )
        .into_pyarray(py)
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        hierarchical_path_summary(
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
            None,
        )
        .into_pyarray(py)
    }

    fn state_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_probability(probability)?;
        Ok(hierarchical_state_summary(
            &self.inner.state_means,
            self.program_count(),
            self.steps(),
            Some(probability),
        )
        .into_pyarray(py))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_probability(probability)?;
        Ok(hierarchical_path_summary(
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
            Some(probability),
        )
        .into_pyarray(py))
    }

    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalMatrices<'py>> {
        validate_interval_level(level)?;
        let tail = (1.0 - level) / 2.0;
        Ok((
            hierarchical_path_summary(
                &self.inner.observation_paths,
                self.program_count(),
                self.steps(),
                Some(tail),
            )
            .into_pyarray(py),
            hierarchical_path_summary(
                &self.inner.observation_paths,
                self.program_count(),
                self.steps(),
                Some(1.0 - tail),
            )
            .into_pyarray(py),
        ))
    }

    fn state_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalMatrices<'py>> {
        validate_interval_level(level)?;
        let tail = (1.0 - level) / 2.0;
        Ok((
            hierarchical_state_summary(
                &self.inner.state_means,
                self.program_count(),
                self.steps(),
                Some(tail),
            )
            .into_pyarray(py),
            hierarchical_state_summary(
                &self.inner.state_means,
                self.program_count(),
                self.steps(),
                Some(1.0 - tail),
            )
            .into_pyarray(py),
        ))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }

    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianHierarchicalForecast(chains={}, draws={}, programs={}, groups={}, steps={})",
            self.chains(),
            self.draws(),
            self.program_count(),
            self.group_count(),
            self.steps()
        )
    }
}

fn validate_probability(probability: f64) -> PyResult<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "probability must be finite and between zero and one",
        ))
    }
}

/// Inverse-gamma prior for a variance, parameterized by shape and scale.
/// The density is proportional to x^(-shape-1) exp(-scale/x).
#[pyclass(name = "InverseGammaPrior", frozen, module = "rustmc")]
#[derive(Clone, Copy)]
struct PyInverseGammaPrior {
    inner: CoreInverseGammaPrior,
}

#[pymethods]
impl PyInverseGammaPrior {
    #[new]
    fn new(shape: f64, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CoreInverseGammaPrior::new(shape, scale).map_err(bayesian_forecast_error)?,
        })
    }

    #[getter]
    fn shape(&self) -> f64 {
        self.inner.shape
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn __repr__(&self) -> String {
        format!(
            "InverseGammaPrior(shape={}, scale={})",
            self.inner.shape, self.inner.scale
        )
    }
}

/// Bayesian scalar Gaussian local-level model fitted with conjugate
/// forward-filtering/backward-sampling Gibbs updates.
#[pyclass(name = "BayesianLocalLevel", frozen, module = "rustmc")]
#[derive(Clone)]
struct PyBayesianLocalLevel {
    initial_mean: f64,
    initial_variance: f64,
    process_variance_prior: CoreInverseGammaPrior,
    observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianLocalLevel {
    /// Fit independent ragged cells on one bounded native worker pool.
    #[pyo3(signature = (observations, ids, *, models=None, exog=None, coefficient_priors=None, chains=4, draws=1000, warmup=500, thin=1, seed=42, threads=1, chunk_size=64, errors="raise"))]
    #[allow(clippy::too_many_arguments)]
    fn fit_batch(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        ids: Vec<String>,
        models: Option<&Bound<'_, PyAny>>,
        exog: Option<&Bound<'_, PyAny>>,
        coefficient_priors: Option<&Bound<'_, PyAny>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<forecast_batch::PyForecastBatchFit> {
        forecast_batch::fit_batch(
            py,
            observations,
            ids,
            models,
            exog,
            coefficient_priors,
            self.batch_config(chains, draws, warmup, thin),
            chains,
            draws,
            warmup,
            thin,
            seed,
            threads,
            chunk_size,
            errors,
        )
    }

    #[new]
    #[pyo3(signature = (process_variance_prior, observation_variance_prior, initial_mean=0.0, initial_variance=100.0))]
    fn new(
        process_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        initial_mean: f64,
        initial_variance: f64,
    ) -> PyResult<Self> {
        if !initial_mean.is_finite() {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial mean must be finite",
            ));
        }
        if !initial_variance.is_finite() || initial_variance <= 0.0 {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial variance must be finite and strictly positive",
            ));
        }
        Ok(Self {
            initial_mean,
            initial_variance,
            process_variance_prior: process_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn initial_mean(&self) -> f64 {
        self.initial_mean
    }

    #[getter]
    fn initial_variance(&self) -> f64 {
        self.initial_variance
    }

    #[getter]
    fn process_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.process_variance_prior,
        }
    }

    #[getter]
    fn observation_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.observation_variance_prior,
        }
    }

    #[pyo3(signature = (observations, chains=4, draws=1000, warmup=500, thin=1, seed=42, *, exog=None, coefficient_prior=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        exog: Option<PyReadonlyArray2<'_, f64>>,
        coefficient_prior: Option<PyRef<'_, regression::PyGaussianCoefficientPrior>>,
    ) -> PyResult<PyObject> {
        let observations = state_space_vector(observations);
        if let Some(exog) = exog {
            let config = regression::config(
                CoreLinearGaussianStateSpace::local_level(
                    1.0,
                    1.0,
                    self.initial_mean,
                    self.initial_variance,
                )
                .map_err(state_space_error)?,
                vec![self.process_variance_prior],
                vec!["process_variance"],
                self.observation_variance_prior,
                false,
                (chains, draws, warmup, thin, seed),
            );
            return regression::fit(py, observations, exog, coefficient_prior, config);
        }
        if coefficient_prior.is_some() {
            return Err(StateSpaceError::new_err("coefficient_prior requires exog"));
        }
        let config = CoreBayesianLocalLevelConfig {
            initial_mean: self.initial_mean,
            initial_variance: self.initial_variance,
            process_variance_prior: self.process_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_bayesian_local_level(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(Py::new(
            py,
            PyBayesianLocalLevelFit {
                posterior,
                observations,
                config,
            },
        )?
        .into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianLocalLevel(initial_mean={}, initial_variance={}, process_variance_prior=({}, {}), observation_variance_prior=({}, {}))",
            self.initial_mean,
            self.initial_variance,
            self.process_variance_prior.shape,
            self.process_variance_prior.scale,
            self.observation_variance_prior.shape,
            self.observation_variance_prior.scale,
        )
    }
}

#[pyclass(name = "BayesianLocalLevelFit", module = "rustmc")]
#[derive(Clone)]
struct PyBayesianLocalLevelFit {
    posterior: CoreLocalLevelPosterior,
    observations: Vec<f64>,
    config: CoreBayesianLocalLevelConfig,
}

#[pymethods]
impl PyBayesianLocalLevelFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: conjugate Gibbs/FFBS; acceptance and divergences unavailable",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "conjugate Gibbs/FFBS",
            self.chains(),
            self.draws(),
            "variance parameters and terminal level; historical states are not retained",
        )
    }

    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
    }

    #[getter]
    fn observed_count(&self) -> usize {
        self.observations
            .iter()
            .filter(|value| !value.is_nan())
            .count()
    }

    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }

    #[getter]
    fn warmup(&self) -> usize {
        self.config.num_warmup
    }

    #[getter]
    fn thin(&self) -> usize {
        self.config.thinning
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        samples.set_item(
            "process_variance",
            local_level_parameter_array(py, &self.posterior, |draw| draw.process_variance),
        )?;
        samples.set_item(
            "observation_variance",
            local_level_parameter_array(py, &self.posterior, |draw| draw.observation_variance),
        )?;
        samples.set_item(
            "process_sd",
            local_level_parameter_array(py, &self.posterior, |draw| draw.process_variance.sqrt()),
        )?;
        samples.set_item(
            "observation_sd",
            local_level_parameter_array(py, &self.posterior, |draw| {
                draw.observation_variance.sqrt()
            }),
        )?;
        samples.set_item(
            "terminal_level",
            local_level_parameter_array(py, &self.posterior, |draw| draw.terminal_level),
        )?;
        Ok(samples)
    }

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
    ) -> PyResult<PyBayesianForecastResult> {
        let forecast = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianForecastResult { inner: forecast })
    }

    /// Export parameter draws and the fitted observations to ArviZ.
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", PyArray1::from_vec(py, self.observations.clone()))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianLocalLevelFit(chains={}, draws={}, time_count={}, observed_count={})",
            self.chains(),
            self.draws(),
            self.time_count(),
            self.observed_count(),
        )
    }
}

#[pyclass(name = "BayesianForecastResult", module = "rustmc")]
struct PyBayesianForecastResult {
    inner: CorePosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianForecastResult {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observation_paths.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths.first().map_or(0, Vec::len)
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.state_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn state_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .state_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    fn state_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .state_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    /// Equal-tailed pointwise posterior interval for the latent state.
    #[pyo3(signature = (level=0.95))]
    fn state_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .state_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    /// Equal-tailed pointwise posterior-predictive interval for observations.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .observation_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }

    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianForecastResult(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

fn seasonal_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreSeasonalLocalLevelPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreSeasonalLocalLevelPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

/// Bayesian structural seasonal local-level model using conjugate Gibbs/FFBS.
#[pyclass(name = "BayesianSeasonalLocalLevel", frozen, module = "rustmc")]
#[derive(Clone)]
struct PyBayesianSeasonalLocalLevel {
    period: usize,
    initial_level: f64,
    initial_seasonal_effects: Vec<f64>,
    initial_level_variance: f64,
    initial_seasonal_variance: f64,
    level_variance_prior: CoreInverseGammaPrior,
    seasonal_variance_prior: CoreInverseGammaPrior,
    observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianSeasonalLocalLevel {
    /// Fit independent ragged cells on one bounded native worker pool.
    #[pyo3(signature = (observations, ids, *, models=None, exog=None, coefficient_priors=None, chains=4, draws=1000, warmup=500, thin=1, seed=42, threads=1, chunk_size=64, errors="raise"))]
    #[allow(clippy::too_many_arguments)]
    fn fit_batch(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        ids: Vec<String>,
        models: Option<&Bound<'_, PyAny>>,
        exog: Option<&Bound<'_, PyAny>>,
        coefficient_priors: Option<&Bound<'_, PyAny>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<forecast_batch::PyForecastBatchFit> {
        forecast_batch::fit_batch(
            py,
            observations,
            ids,
            models,
            exog,
            coefficient_priors,
            self.batch_config(chains, draws, warmup, thin),
            chains,
            draws,
            warmup,
            thin,
            seed,
            threads,
            chunk_size,
            errors,
        )
    }

    #[new]
    #[pyo3(signature = (period, level_variance_prior, seasonal_variance_prior, observation_variance_prior, initial_level=0.0, initial_seasonal_effects=None, initial_level_variance=100.0, initial_seasonal_variance=10.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        period: usize,
        level_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        seasonal_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        initial_level: f64,
        initial_seasonal_effects: Option<Vec<f64>>,
        initial_level_variance: f64,
        initial_seasonal_variance: f64,
    ) -> PyResult<Self> {
        let effects = initial_seasonal_effects.unwrap_or_else(|| vec![0.0; period]);
        // Reuse the fixed structural constructor for immediate shape,
        // sum-to-zero, and covariance validation.
        CoreLinearGaussianStateSpace::seasonal_local_level(
            period,
            level_variance_prior.inner.scale / (level_variance_prior.inner.shape + 1.0),
            seasonal_variance_prior.inner.scale / (seasonal_variance_prior.inner.shape + 1.0),
            observation_variance_prior.inner.scale / (observation_variance_prior.inner.shape + 1.0),
            initial_level,
            effects.clone(),
            initial_level_variance,
            initial_seasonal_variance,
        )
        .map_err(state_space_error)?;
        Ok(Self {
            period,
            initial_level,
            initial_seasonal_effects: effects,
            initial_level_variance,
            initial_seasonal_variance,
            level_variance_prior: level_variance_prior.inner,
            seasonal_variance_prior: seasonal_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn period(&self) -> usize {
        self.period
    }

    #[getter]
    fn initial_seasonal_effects<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.initial_seasonal_effects.clone().into_pyarray(py)
    }

    #[pyo3(signature = (observations, chains=4, draws=1000, warmup=500, thin=1, seed=42, *, exog=None, coefficient_prior=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        exog: Option<PyReadonlyArray2<'_, f64>>,
        coefficient_prior: Option<PyRef<'_, regression::PyGaussianCoefficientPrior>>,
    ) -> PyResult<PyObject> {
        let observations = state_space_vector(observations);
        if let Some(exog) = exog {
            let config = regression::config(
                CoreLinearGaussianStateSpace::seasonal_local_level(
                    self.period,
                    1.0,
                    1.0,
                    1.0,
                    self.initial_level,
                    self.initial_seasonal_effects.clone(),
                    self.initial_level_variance,
                    self.initial_seasonal_variance,
                )
                .map_err(state_space_error)?,
                vec![self.level_variance_prior, self.seasonal_variance_prior],
                vec!["level_variance", "seasonal_variance"],
                self.observation_variance_prior,
                true,
                (chains, draws, warmup, thin, seed),
            );
            return regression::fit(py, observations, exog, coefficient_prior, config);
        }
        if coefficient_prior.is_some() {
            return Err(StateSpaceError::new_err("coefficient_prior requires exog"));
        }
        let config = CoreBayesianSeasonalLocalLevelConfig {
            period: self.period,
            initial_level: self.initial_level,
            initial_seasonal_effects: self.initial_seasonal_effects.clone(),
            initial_level_variance: self.initial_level_variance,
            initial_seasonal_variance: self.initial_seasonal_variance,
            level_variance_prior: self.level_variance_prior,
            seasonal_variance_prior: self.seasonal_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_bayesian_seasonal_local_level(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(Py::new(
            py,
            PyBayesianSeasonalLocalLevelFit {
                posterior,
                observations,
                config,
            },
        )?
        .into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianSeasonalLocalLevel(period={}, initial_level={})",
            self.period, self.initial_level
        )
    }
}

#[pyclass(name = "BayesianSeasonalLocalLevelFit", module = "rustmc")]
#[derive(Clone)]
struct PyBayesianSeasonalLocalLevelFit {
    posterior: CoreSeasonalLocalLevelPosterior,
    observations: Vec<f64>,
    config: CoreBayesianSeasonalLocalLevelConfig,
}

#[pymethods]
impl PyBayesianSeasonalLocalLevelFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: conjugate Gibbs/FFBS; acceptance and divergences unavailable",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(py, "conjugate Gibbs/FFBS", self.chains(), self.draws(), "variance parameters and all terminal seasonal state coordinates; historical states are not retained")
    }

    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
    }
    #[getter]
    fn period(&self) -> usize {
        self.posterior.period
    }
    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }
    #[getter]
    fn observed_count(&self) -> usize {
        self.observations
            .iter()
            .filter(|value| !value.is_nan())
            .count()
    }
    #[getter]
    fn warmup(&self) -> usize {
        self.config.num_warmup
    }
    #[getter]
    fn thin(&self) -> usize {
        self.config.thinning
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        for (name, values) in [
            (
                "level_variance",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.level_variance),
            ),
            (
                "seasonal_variance",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.seasonal_variance),
            ),
            (
                "observation_variance",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.observation_variance),
            ),
            (
                "level_sd",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.level_variance.sqrt()),
            ),
            (
                "seasonal_sd",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.seasonal_variance.sqrt()),
            ),
            (
                "observation_sd",
                seasonal_parameter_array(py, &self.posterior, |draw| {
                    draw.observation_variance.sqrt()
                }),
            ),
            (
                "terminal_level",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.terminal_state[0]),
            ),
            (
                "terminal_seasonal",
                seasonal_parameter_array(py, &self.posterior, |draw| draw.terminal_state[1]),
            ),
        ] {
            samples.set_item(name, values)?;
        }
        Ok(samples)
    }

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
    ) -> PyResult<PyBayesianSeasonalForecast> {
        let inner = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianSeasonalForecast { inner })
    }

    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", PyArray1::from_vec(py, self.observations.clone()))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianSeasonalLocalLevelFit(period={}, chains={}, draws={}, time_count={}, observed_count={})",
            self.period(), self.chains(), self.draws(), self.time_count(), self.observed_count()
        )
    }
}

#[pyclass(name = "BayesianSeasonalForecast", module = "rustmc")]
struct PyBayesianSeasonalForecast {
    inner: CoreSeasonalPosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianSeasonalForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observation_paths.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths.first().map_or(0, Vec::len)
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }
    #[getter]
    fn level_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.level_paths)
    }
    #[getter]
    fn seasonal_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.seasonal_paths)
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }
    #[getter]
    fn cumulative_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.cumulative_observation_paths)
    }
    #[getter]
    fn level_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .level_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn seasonal_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .seasonal_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn cumulative_observation_mean<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .cumulative_observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    fn level_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .level_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }
    fn seasonal_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .seasonal_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }
    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }
    fn cumulative_observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .cumulative_observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?[0]
            .values
            .clone()
            .into_pyarray(py))
    }

    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        seasonal_interval(py, level, |probabilities| {
            self.inner.observation_quantiles(probabilities)
        })
    }
    #[pyo3(signature = (level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        seasonal_interval(py, level, |probabilities| {
            self.inner.cumulative_observation_quantiles(probabilities)
        })
    }
    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }
    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }
    fn __repr__(&self) -> String {
        format!(
            "BayesianSeasonalForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps()
        )
    }
}

fn seasonal_interval<'py, F>(
    py: Python<'py>,
    level: f64,
    quantiles: F,
) -> PyResult<PyIntervalArrays<'py>>
where
    F: FnOnce(
        &[f64],
    ) -> Result<
        Vec<rustmc_core::bayesian_forecast::ForecastQuantile>,
        rustmc_core::bayesian_forecast::BayesianForecastError,
    >,
{
    validate_interval_level(level)?;
    let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
    let values = quantiles(&probabilities).map_err(bayesian_forecast_error)?;
    Ok((
        values[0].values.clone().into_pyarray(py),
        values[1].values.clone().into_pyarray(py),
    ))
}

fn trend_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreLocalLinearTrendPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreLocalLinearTrendPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

/// Bayesian local-linear-trend model with stochastic level and slope.
#[pyclass(name = "BayesianLocalLinearTrend", frozen, module = "rustmc")]
#[derive(Clone)]
struct PyBayesianLocalLinearTrend {
    initial_mean: [f64; 2],
    initial_covariance: [f64; 4],
    level_variance_prior: CoreInverseGammaPrior,
    slope_variance_prior: CoreInverseGammaPrior,
    observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianLocalLinearTrend {
    /// Fit independent ragged cells on one bounded native worker pool.
    #[pyo3(signature = (observations, ids, *, models=None, exog=None, coefficient_priors=None, chains=4, draws=1000, warmup=500, thin=1, seed=42, threads=1, chunk_size=64, errors="raise"))]
    #[allow(clippy::too_many_arguments)]
    fn fit_batch(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        ids: Vec<String>,
        models: Option<&Bound<'_, PyAny>>,
        exog: Option<&Bound<'_, PyAny>>,
        coefficient_priors: Option<&Bound<'_, PyAny>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<forecast_batch::PyForecastBatchFit> {
        forecast_batch::fit_batch(
            py,
            observations,
            ids,
            models,
            exog,
            coefficient_priors,
            self.batch_config(chains, draws, warmup, thin),
            chains,
            draws,
            warmup,
            thin,
            seed,
            threads,
            chunk_size,
            errors,
        )
    }

    #[new]
    #[pyo3(signature = (
        level_variance_prior,
        slope_variance_prior,
        observation_variance_prior,
        initial_level=0.0,
        initial_slope=0.0,
        initial_level_variance=100.0,
        initial_slope_variance=10.0,
        initial_level_slope_covariance=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        level_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        slope_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        initial_level: f64,
        initial_slope: f64,
        initial_level_variance: f64,
        initial_slope_variance: f64,
        initial_level_slope_covariance: f64,
    ) -> PyResult<Self> {
        if !initial_level.is_finite() || !initial_slope.is_finite() {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial level and slope must be finite",
            ));
        }
        if !initial_level_variance.is_finite()
            || initial_level_variance <= 0.0
            || !initial_slope_variance.is_finite()
            || initial_slope_variance <= 0.0
            || !initial_level_slope_covariance.is_finite()
        {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial variances must be finite and positive and covariance must be finite",
            ));
        }
        let level_scale = initial_level_variance.sqrt();
        let scaled_covariance = initial_level_slope_covariance / level_scale;
        let slope_remainder = initial_slope_variance - scaled_covariance * scaled_covariance;
        if !slope_remainder.is_finite() || slope_remainder <= 0.0 {
            return Err(StateSpaceError::new_err(
                "invalid configuration: initial state covariance must be positive definite",
            ));
        }
        Ok(Self {
            initial_mean: [initial_level, initial_slope],
            initial_covariance: [
                initial_level_variance,
                initial_level_slope_covariance,
                initial_level_slope_covariance,
                initial_slope_variance,
            ],
            level_variance_prior: level_variance_prior.inner,
            slope_variance_prior: slope_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn initial_level(&self) -> f64 {
        self.initial_mean[0]
    }

    #[getter]
    fn initial_slope(&self) -> f64 {
        self.initial_mean[1]
    }

    #[getter]
    fn initial_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_fn((2, 2), |(row, column)| {
            self.initial_covariance[row * 2 + column]
        })
        .into_pyarray(py)
    }

    #[getter]
    fn level_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.level_variance_prior,
        }
    }

    #[getter]
    fn slope_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.slope_variance_prior,
        }
    }

    #[getter]
    fn observation_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.observation_variance_prior,
        }
    }

    #[pyo3(signature = (observations, chains=4, draws=1000, warmup=500, thin=1, seed=42, *, exog=None, coefficient_prior=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        exog: Option<PyReadonlyArray2<'_, f64>>,
        coefficient_prior: Option<PyRef<'_, regression::PyGaussianCoefficientPrior>>,
    ) -> PyResult<PyObject> {
        let observations = state_space_vector(observations);
        if let Some(exog) = exog {
            let config = regression::config(
                CoreLinearGaussianStateSpace::new(
                    2,
                    vec![1.0, 1.0, 0.0, 1.0],
                    vec![1.0, 0.0],
                    vec![1.0, 0.0, 0.0, 1.0],
                    1.0,
                    self.initial_mean.to_vec(),
                    self.initial_covariance.to_vec(),
                )
                .map_err(state_space_error)?,
                vec![self.level_variance_prior, self.slope_variance_prior],
                vec!["level_variance", "slope_variance"],
                self.observation_variance_prior,
                false,
                (chains, draws, warmup, thin, seed),
            );
            return regression::fit(py, observations, exog, coefficient_prior, config);
        }
        if coefficient_prior.is_some() {
            return Err(StateSpaceError::new_err("coefficient_prior requires exog"));
        }
        let config = CoreBayesianLocalLinearTrendConfig {
            initial_mean: self.initial_mean,
            initial_covariance: self.initial_covariance,
            level_variance_prior: self.level_variance_prior,
            slope_variance_prior: self.slope_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_bayesian_local_linear_trend(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(Py::new(
            py,
            PyBayesianLocalLinearTrendFit {
                posterior,
                observations,
                config,
            },
        )?
        .into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianLocalLinearTrend(initial_level={}, initial_slope={})",
            self.initial_mean[0], self.initial_mean[1],
        )
    }
}

#[pyclass(name = "BayesianLocalLinearTrendFit", module = "rustmc")]
#[derive(Clone)]
struct PyBayesianLocalLinearTrendFit {
    posterior: CoreLocalLinearTrendPosterior,
    observations: Vec<f64>,
    config: CoreBayesianLocalLinearTrendConfig,
}

#[pymethods]
impl PyBayesianLocalLinearTrendFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: conjugate Gibbs/FFBS; acceptance and divergences unavailable",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "conjugate Gibbs/FFBS",
            self.chains(),
            self.draws(),
            "variance parameters, terminal level and slope; historical states are not retained",
        )
    }

    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
    }

    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }

    #[getter]
    fn observed_count(&self) -> usize {
        self.observations
            .iter()
            .filter(|value| !value.is_nan())
            .count()
    }

    #[getter]
    fn warmup(&self) -> usize {
        self.config.num_warmup
    }

    #[getter]
    fn thin(&self) -> usize {
        self.config.thinning
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        for (name, values) in [
            (
                "level_variance",
                trend_parameter_array(py, &self.posterior, |draw| draw.level_variance),
            ),
            (
                "slope_variance",
                trend_parameter_array(py, &self.posterior, |draw| draw.slope_variance),
            ),
            (
                "observation_variance",
                trend_parameter_array(py, &self.posterior, |draw| draw.observation_variance),
            ),
            (
                "level_sd",
                trend_parameter_array(py, &self.posterior, |draw| draw.level_variance.sqrt()),
            ),
            (
                "slope_sd",
                trend_parameter_array(py, &self.posterior, |draw| draw.slope_variance.sqrt()),
            ),
            (
                "observation_sd",
                trend_parameter_array(py, &self.posterior, |draw| draw.observation_variance.sqrt()),
            ),
            (
                "terminal_level",
                trend_parameter_array(py, &self.posterior, |draw| draw.terminal_level),
            ),
            (
                "terminal_slope",
                trend_parameter_array(py, &self.posterior, |draw| draw.terminal_slope),
            ),
        ] {
            samples.set_item(name, values)?;
        }
        Ok(samples)
    }

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
    ) -> PyResult<PyBayesianTrendForecast> {
        let forecast = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianTrendForecast { inner: forecast })
    }

    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", PyArray1::from_vec(py, self.observations.clone()))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianLocalLinearTrendFit(chains={}, draws={}, time_count={}, observed_count={})",
            self.chains(),
            self.draws(),
            self.time_count(),
            self.observed_count(),
        )
    }
}

#[pyclass(name = "BayesianTrendForecast", module = "rustmc")]
struct PyBayesianTrendForecast {
    inner: CoreTrendPosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianTrendForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observation_paths.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths.first().map_or(0, Vec::len)
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    fn level_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.level_paths)
    }

    #[getter]
    fn slope_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.slope_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn level_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .level_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    #[getter]
    fn slope_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .slope_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    fn level_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .level_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    fn slope_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .slope_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    #[pyo3(signature = (level=0.95))]
    fn level_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .level_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[pyo3(signature = (level=0.95))]
    fn slope_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .slope_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .observation_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }

    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianTrendForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

fn ar_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreBayesianArPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreBayesianArPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

fn ar_coefficient_array<'py>(
    py: Python<'py>,
    posterior: &CoreBayesianArPosterior,
) -> Bound<'py, PyArray3<f64>> {
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    let coefficient_count = posterior.order + 1;
    Array3::from_shape_fn(
        (chains, draws, coefficient_count),
        |(chain, draw, coefficient)| posterior.chains[chain][draw].coefficients[coefficient],
    )
    .into_pyarray(py)
}

/// Conjugate prior for a Gaussian autoregression.
///
/// If beta contains ``[intercept, lag_1, ..., lag_p]``, then
/// ``beta | sigma2 ~ Normal(mean, sigma2 * precision^-1)`` and
/// ``sigma2 ~ InverseGamma(variance_shape, variance_scale)``.
#[pyclass(name = "NormalInverseGammaPrior", frozen, module = "rustmc")]
#[derive(Clone)]
struct PyNormalInverseGammaPrior {
    inner: CoreNormalInverseGammaPrior,
}

#[pymethods]
impl PyNormalInverseGammaPrior {
    #[new]
    fn new(
        coefficient_mean: PyReadonlyArray1<'_, f64>,
        coefficient_precision: PyReadonlyArray2<'_, f64>,
        variance_shape: f64,
        variance_scale: f64,
    ) -> PyResult<Self> {
        let mean = coefficient_mean.as_array().to_vec();
        let precision = coefficient_precision
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        Ok(Self {
            inner: CoreNormalInverseGammaPrior::new(
                mean,
                precision,
                variance_shape,
                variance_scale,
            )
            .map_err(bayesian_forecast_error)?,
        })
    }

    #[getter]
    fn coefficient_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.coefficient_mean.clone())
    }

    #[getter]
    fn coefficient_precision<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let dimension = self.inner.coefficient_mean.len();
        Array2::from_shape_fn((dimension, dimension), |(row, column)| {
            self.inner.coefficient_precision[row][column]
        })
        .into_pyarray(py)
    }

    #[getter]
    fn variance_shape(&self) -> f64 {
        self.inner.variance_shape
    }

    #[getter]
    fn variance_scale(&self) -> f64 {
        self.inner.variance_scale
    }

    #[getter]
    fn coefficient_count(&self) -> usize {
        self.inner.coefficient_mean.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "NormalInverseGammaPrior(coefficient_count={}, variance_shape={}, variance_scale={})",
            self.coefficient_count(),
            self.inner.variance_shape,
            self.inner.variance_scale,
        )
    }
}

/// Directly observed Gaussian Bayesian AR(p) model.
///
/// This is distinct from ``LinearGaussianStateSpace.stationary_ar1``: the
/// latter is a latent AR(1) observed with separate measurement noise.
#[pyclass(name = "BayesianAutoRegression", frozen, module = "rustmc")]
#[derive(Clone)]
struct PyBayesianAutoRegression {
    order: usize,
    prior: CoreNormalInverseGammaPrior,
}

#[pymethods]
impl PyBayesianAutoRegression {
    /// Fit independent ragged cells on one bounded native worker pool.
    #[pyo3(signature = (observations, ids, *, models=None, exog=None, coefficient_priors=None, chains=4, draws=1000, warmup=500, thin=1, seed=42, threads=1, chunk_size=64, errors="raise"))]
    #[allow(clippy::too_many_arguments)]
    fn fit_batch(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        ids: Vec<String>,
        models: Option<&Bound<'_, PyAny>>,
        exog: Option<&Bound<'_, PyAny>>,
        coefficient_priors: Option<&Bound<'_, PyAny>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        threads: usize,
        chunk_size: usize,
        errors: &str,
    ) -> PyResult<forecast_batch::PyForecastBatchFit> {
        forecast_batch::fit_batch(
            py,
            observations,
            ids,
            models,
            exog,
            coefficient_priors,
            self.batch_config(chains, draws, warmup, thin),
            chains,
            draws,
            warmup,
            thin,
            seed,
            threads,
            chunk_size,
            errors,
        )
    }

    #[new]
    fn new(order: usize, prior: PyRef<'_, PyNormalInverseGammaPrior>) -> PyResult<Self> {
        if order == 0 {
            return Err(StateSpaceError::new_err(
                "invalid configuration: AR order must be at least one",
            ));
        }
        let expected = order.checked_add(1).ok_or_else(|| {
            StateSpaceError::new_err(
                "invalid configuration: AR order is too large to represent its coefficients",
            )
        })?;
        if prior.inner.coefficient_mean.len() != expected {
            return Err(StateSpaceError::new_err(format!(
                "invalid configuration: AR({order}) requires {expected} coefficient prior entries (intercept plus {order} lags)"
            )));
        }
        Ok(Self {
            order,
            prior: prior.inner.clone(),
        })
    }

    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    #[getter]
    fn prior(&self) -> PyNormalInverseGammaPrior {
        PyNormalInverseGammaPrior {
            inner: self.prior.clone(),
        }
    }

    #[pyo3(signature = (observations, chains=4, draws=1000, seed=42))]
    fn fit(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        seed: u64,
    ) -> PyResult<PyBayesianArFit> {
        let observations = state_space_vector(observations);
        let config = CoreBayesianArConfig {
            order: self.order,
            prior: self.prior.clone(),
            num_chains: chains,
            num_draws: draws,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_bayesian_ar(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianArFit {
            posterior,
            observations,
            config,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianAutoRegression(order={}, coefficient_count={})",
            self.order,
            self.prior.coefficient_mean.len(),
        )
    }
}

#[pyclass(name = "BayesianARFit", module = "rustmc")]
#[derive(Clone)]
struct PyBayesianArFit {
    posterior: CoreBayesianArPosterior,
    observations: Vec<f64>,
    config: CoreBayesianArConfig,
}

#[pymethods]
impl PyBayesianArFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: exact conjugate independent draws; acceptance and divergences unavailable",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "exact conjugate independent draws",
            self.chains(),
            self.draws(),
            "coefficients and innovation variance",
        )
    }

    #[getter]
    fn order(&self) -> usize {
        self.posterior.order
    }

    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
    }

    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }

    #[getter]
    fn regression_count(&self) -> usize {
        self.observations.len() - self.posterior.order
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.config.seed
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        samples.set_item("coefficient", ar_coefficient_array(py, &self.posterior))?;
        samples.set_item(
            "innovation_variance",
            ar_parameter_array(py, &self.posterior, |draw| draw.innovation_variance),
        )?;
        samples.set_item(
            "innovation_sd",
            ar_parameter_array(py, &self.posterior, |draw| draw.innovation_variance.sqrt()),
        )?;
        Ok(samples)
    }

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(&self, py: Python<'_>, steps: usize, seed: u64) -> PyResult<PyBayesianArForecast> {
        let forecast = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianArForecast { inner: forecast })
    }

    /// Export coefficient and innovation-variance draws to ArviZ.
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", PyArray1::from_vec(py, self.observations.clone()))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianARFit(order={}, chains={}, draws={}, time_count={})",
            self.order(),
            self.chains(),
            self.draws(),
            self.time_count(),
        )
    }
}

#[pyclass(name = "BayesianARForecast", module = "rustmc")]
struct PyBayesianArForecast {
    inner: CoreBayesianArForecast,
}

#[pymethods]
impl PyBayesianArForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.observation_paths.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths.first().map_or(0, Vec::len)
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    fn conditional_mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.conditional_mean_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn conditional_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .conditional_mean_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }

    fn conditional_mean_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .conditional_mean_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .observation_quantiles(&[probability])
            .map_err(bayesian_forecast_error)?;
        Ok(quantiles[0].values.clone().into_pyarray(py))
    }

    /// Pointwise equal-tailed interval for the recursive conditional mean.
    #[pyo3(signature = (level=0.95))]
    fn conditional_mean_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .conditional_mean_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    /// Pointwise equal-tailed posterior-predictive interval for future observations.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .observation_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "parameter_integrated_posterior_predictive"
    }

    #[getter]
    fn interval_kind(&self) -> &'static str {
        "pointwise_equal_tailed"
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianARForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

fn validate_interval_level(level: f64) -> PyResult<()> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(PyValueError::new_err(
            "level must be finite and strictly between 0 and 1",
        ));
    }
    Ok(())
}

#[pymodule]
fn _rustmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dynamic_glm::register(m)?;
    hurdle::register(m)?;
    regression::register(m)?;
    structural::register(m)?;
    runoff::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<builder::ModelBuilder>()?;
    m.add_class::<builder::ModelSpec>()?;
    m.add_class::<expressions::ParamRef>()?;
    m.add_class::<expressions::VectorParamRef>()?;
    m.add_class::<expressions::Expr>()?;
    m.add_class::<fit_result::FitResult>()?;
    m.add_class::<batch::BatchResult>()?;
    m.add_class::<compiled::PyCompiledModel>()?;
    m.add_class::<compiled::PyBoundModel>()?;
    m.add_class::<batch::PyBatchFit>()?;
    m.add_function(wrap_pyfunction!(forecast_batch::forecast_cell_seed, m)?)?;
    m.add_class::<forecast_batch::PyForecastBatchFit>()?;
    m.add_class::<forecast_batch::PyForecastBatchForecast>()?;
    m.add_class::<PyLinearGaussianStateSpace>()?;
    m.add_class::<PyKalmanFilterResult>()?;
    m.add_class::<PyKalmanSmootherResult>()?;
    m.add_class::<PyForecastResult>()?;
    m.add_class::<PyInverseGammaPrior>()?;
    m.add_class::<PyBayesianHierarchicalMean>()?;
    m.add_class::<PyBayesianHierarchicalMeanFit>()?;
    m.add_class::<PyBayesianHierarchicalForecast>()?;
    m.add_class::<PyBayesianLocalLevel>()?;
    m.add_class::<PyBayesianLocalLevelFit>()?;
    m.add_class::<PyBayesianForecastResult>()?;
    m.add_class::<PyBayesianSeasonalLocalLevel>()?;
    m.add_class::<PyBayesianSeasonalLocalLevelFit>()?;
    m.add_class::<PyBayesianSeasonalForecast>()?;
    m.add_class::<PyBayesianLocalLinearTrend>()?;
    m.add_class::<PyBayesianLocalLinearTrendFit>()?;
    m.add_class::<PyBayesianTrendForecast>()?;
    m.add_class::<PyNormalInverseGammaPrior>()?;
    m.add_class::<PyBayesianAutoRegression>()?;
    m.add("BayesianAR", m.getattr("BayesianAutoRegression")?)?;
    m.add_class::<PyBayesianArFit>()?;
    m.add_class::<PyBayesianArForecast>()?;
    m.add("ParameterError", m.py().get_type::<ParameterError>())?;
    m.add("StateSpaceError", m.py().get_type::<StateSpaceError>())?;
    m.add("InferenceError", m.py().get_type::<InferenceError>())?;
    m.add_function(wrap_pyfunction!(sampling::sample, m)?)?;
    m.add_function(wrap_pyfunction!(batch::batch_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::sample_prior_predictive, m)?)?;
    Ok(())
}

impl PyBayesianLocalLevel {
    fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;

        forecast_batch::Config::Local(CoreBayesianLocalLevelConfig {
            initial_mean: self.initial_mean,
            initial_variance: self.initial_variance,
            process_variance_prior: self.process_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        })
    }
}

impl PyBayesianSeasonalLocalLevel {
    fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;

        forecast_batch::Config::Seasonal(CoreBayesianSeasonalLocalLevelConfig {
            period: self.period,
            initial_level: self.initial_level,
            initial_seasonal_effects: self.initial_seasonal_effects.clone(),
            initial_level_variance: self.initial_level_variance,
            initial_seasonal_variance: self.initial_seasonal_variance,
            level_variance_prior: self.level_variance_prior,
            seasonal_variance_prior: self.seasonal_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        })
    }
}

impl PyBayesianLocalLinearTrend {
    fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;

        forecast_batch::Config::Trend(CoreBayesianLocalLinearTrendConfig {
            initial_mean: self.initial_mean,
            initial_covariance: self.initial_covariance,
            level_variance_prior: self.level_variance_prior,
            slope_variance_prior: self.slope_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        })
    }
}

impl PyBayesianAutoRegression {
    fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;
        let _ = (warmup, thin);
        forecast_batch::Config::Ar(CoreBayesianArConfig {
            order: self.order,
            prior: self.prior.clone(),
            num_chains: chains,
            num_draws: draws,
            seed,
        })
    }
}
