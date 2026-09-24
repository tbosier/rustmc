//! Fixed-parameter linear Gaussian state-space bindings.
use crate::forecast_support::*;
use crate::StateSpaceError;
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rustmc_core::state_space::{
    ForecastResult as CoreForecastResult, KalmanFilterResult as CoreKalmanFilterResult,
    KalmanSmootherResult as CoreKalmanSmootherResult,
    LinearGaussianStateSpace as CoreLinearGaussianStateSpace,
};

pub(crate) fn state_means_array<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
    dimension: usize,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((values.len(), dimension), |(time, state)| {
        values[time][state]
    })
    .into_pyarray(py)
}

pub(crate) fn state_covariances_array<'py>(
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
pub(crate) struct PyLinearGaussianStateSpace {
    pub(crate) inner: CoreLinearGaussianStateSpace,
}

/// A square real matrix, row-major, with its dimension.
fn square_matrix(value: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<f64>, usize)> {
    let rows = real_matrix(value, name)?;
    let dimension = rows.len();
    if rows.iter().any(|row| row.len() != dimension) {
        return Err(StateSpaceError::new_err(format!(
            "invalid dimension: {name} must be a square matrix"
        )));
    }
    Ok((rows.concat(), dimension))
}

#[pymethods]
impl PyLinearGaussianStateSpace {
    #[new]
    #[pyo3(signature = (transition, observation, process_covariance, observation_variance, initial_mean, initial_covariance))]
    fn new(
        transition: &Bound<'_, PyAny>,
        observation: &Bound<'_, PyAny>,
        process_covariance: &Bound<'_, PyAny>,
        observation_variance: f64,
        initial_mean: &Bound<'_, PyAny>,
        initial_covariance: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let (transition, dimension) = square_matrix(transition, "transition")?;
        let (process_covariance, process_dimension) =
            square_matrix(process_covariance, "process_covariance")?;
        let (initial_covariance, initial_dimension) =
            square_matrix(initial_covariance, "initial_covariance")?;
        if process_dimension != dimension || initial_dimension != dimension {
            return Err(StateSpaceError::new_err(
                "invalid dimension: covariance matrices must match the transition matrix",
            ));
        }
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::new(
                dimension,
                transition,
                real_vector(observation, "observation")?,
                process_covariance,
                observation_variance,
                real_vector(initial_mean, "initial_mean")?,
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
        initial_seasonal_effects: Option<&Bound<'_, PyAny>>,
        initial_level_variance: f64,
        initial_seasonal_variance: f64,
    ) -> PyResult<Self> {
        let effects = match initial_seasonal_effects {
            Some(effects) => real_vector(effects, "initial_seasonal_effects")?,
            None => vec![0.0; period],
        };
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
    fn with_observation_rows(&self, observation_rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_observation_rows(real_matrix(observation_rows, "observation_rows")?)
                .map_err(state_space_error)?,
        })
    }

    fn filter(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
    ) -> PyResult<PyKalmanFilterResult> {
        let observations = real_vector(observations, "observations")?;
        let result = py
            .allow_threads(|| self.inner.filter(&observations))
            .map_err(state_space_error)?;
        Ok(PyKalmanFilterResult::new(result, self.inner.dimension()))
    }

    fn smooth(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
    ) -> PyResult<PyKalmanSmootherResult> {
        let observations = real_vector(observations, "observations")?;
        let result = py
            .allow_threads(|| self.inner.smooth(&observations))
            .map_err(state_space_error)?;
        Ok(PyKalmanSmootherResult::new(result, self.inner.dimension()))
    }

    #[pyo3(signature=(observations, steps, *, future_observation_rows=None))]
    fn forecast(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        steps: usize,
        future_observation_rows: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyForecastResult> {
        let observations = real_vector(observations, "observations")?;
        let future_rows = future_observation_rows
            .map(|rows| real_matrix(rows, "future_observation_rows"))
            .transpose()?;
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
pub(crate) struct PyKalmanFilterResult {
    pub(crate) inner: CoreKalmanFilterResult,
    pub(crate) dimension: usize,
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
pub(crate) struct PyKalmanSmootherResult {
    pub(crate) inner: CoreKalmanSmootherResult,
    pub(crate) dimension: usize,
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
pub(crate) struct PyForecastResult {
    pub(crate) inner: CoreForecastResult,
    pub(crate) dimension: usize,
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
        validate_interval_level(level)?;
        let bounds = self
            .inner
            .observation_interval(level)
            .map_err(state_space_error)?;
        Ok(interval_arrays(py, bounds))
    }

    /// Gaussian predictive intervals for cumulative observations 1..h,
    /// conditional on the fixed model parameters.
    #[pyo3(signature = (level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let bounds = self
            .inner
            .cumulative_observation_interval(level)
            .map_err(state_space_error)?;
        Ok(interval_arrays(py, bounds))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "conditional_fixed_parameters"
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearGaussianStateSpace>()?;
    m.add_class::<PyKalmanFilterResult>()?;
    m.add_class::<PyKalmanSmootherResult>()?;
    m.add_class::<PyForecastResult>()?;
    Ok(())
}
