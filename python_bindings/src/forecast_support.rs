//! Helpers shared by the Gaussian forecasting bindings.
use crate::{arviz_from_groups, forecast_diagnostics, InferenceError, StateSpaceError};
use ndarray::{Array2, Array3};
use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::{
    BayesianForecastError as CoreBayesianForecastError, ForecastQuantile,
    InverseGammaPrior as CoreInverseGammaPrior,
};
pub(crate) use rustmc_core::diagnostics::DiagnosticsReport;
use rustmc_core::state_space::StateSpaceError as CoreStateSpaceError;

pub(crate) type PyIntervalArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
pub(crate) type PyIntervalMatrices<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

pub(crate) fn state_space_error(error: CoreStateSpaceError) -> PyErr {
    StateSpaceError::new_err(error.to_string())
}

pub(crate) fn bayesian_forecast_error(error: CoreBayesianForecastError) -> PyErr {
    StateSpaceError::new_err(error.to_string())
}

pub(crate) fn hierarchical_error(error: CoreBayesianForecastError) -> PyErr {
    InferenceError::new_err(error.to_string())
}

pub(crate) fn state_space_matrix(
    name: &str,
    value: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Vec<f64>, usize)> {
    let shape = value.shape();
    if shape[0] != shape[1] {
        return Err(StateSpaceError::new_err(format!(
            "invalid dimension: {name} must be a square matrix"
        )));
    }
    Ok((value.as_array().iter().copied().collect(), shape[0]))
}

pub(crate) fn state_space_vector(value: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    value.as_array().iter().copied().collect()
}

/// `(chains, draws)` of chain-major draws; chains all hold the same count.
pub(crate) fn chain_shape<T>(chains: &[Vec<T>]) -> (usize, usize) {
    (chains.len(), chains.first().map_or(0, Vec::len))
}

/// A scalar of every posterior draw, shaped `(chain, draw)`.
pub(crate) fn draw_array<'py, D>(
    py: Python<'py>,
    chains: &[Vec<D>],
    value: impl Fn(&D) -> f64,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn(chain_shape(chains), |(chain, draw)| {
        value(&chains[chain][draw])
    })
    .into_pyarray(py)
}

/// A `width`-vector of every posterior draw, shaped `(chain, draw, width)`.
pub(crate) fn draw_vector_array<'py, D>(
    py: Python<'py>,
    chains: &[Vec<D>],
    width: usize,
    value: impl Fn(&D, usize) -> f64,
) -> Bound<'py, PyArray3<f64>> {
    let (chain_count, draw_count) = chain_shape(chains);
    Array3::from_shape_fn((chain_count, draw_count, width), |(chain, draw, index)| {
        value(&chains[chain][draw], index)
    })
    .into_pyarray(py)
}

/// Forecast paths indexed `[chain][draw][step]`, shaped `(chain, draw, step)`.
pub(crate) fn path_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
) -> Bound<'py, PyArray3<f64>> {
    let horizon = paths
        .first()
        .and_then(|chain| chain.first())
        .map_or(0, Vec::len);
    draw_vector_array(py, paths, horizon, |path, step| path[step])
}

pub(crate) fn validate_probability(probability: f64) -> PyResult<()> {
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
pub(crate) struct PyInverseGammaPrior {
    pub(crate) inner: CoreInverseGammaPrior,
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

pub(crate) fn validate_interval_level(level: f64) -> PyResult<()> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(PyValueError::new_err(
            "level must be finite and strictly between 0 and 1",
        ));
    }
    Ok(())
}

/// The tail probabilities of a central interval at `level`.
pub(crate) fn interval_probabilities(level: f64) -> PyResult<[f64; 2]> {
    validate_interval_level(level)?;
    Ok([(1.0 - level) / 2.0, (1.0 + level) / 2.0])
}

/// Validate `probabilities`, then return the values of each core quantile.
pub(crate) fn quantile_values<const N: usize>(
    probabilities: [f64; N],
    quantiles: impl FnOnce(&[f64]) -> Result<Vec<ForecastQuantile>, CoreBayesianForecastError>,
) -> PyResult<[Vec<f64>; N]> {
    for probability in probabilities {
        validate_probability(probability)?;
    }
    let values: Vec<Vec<f64>> = quantiles(&probabilities)
        .map_err(bayesian_forecast_error)?
        .into_iter()
        .map(|quantile| quantile.values)
        .collect();
    values
        .try_into()
        .map_err(|_| InferenceError::new_err("core returned the wrong number of quantiles"))
}

/// One quantile across the horizon as a NumPy array.
pub(crate) fn quantile_array<'py>(
    py: Python<'py>,
    probability: f64,
    quantiles: impl FnOnce(&[f64]) -> Result<Vec<ForecastQuantile>, CoreBayesianForecastError>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let [values] = quantile_values([probability], quantiles)?;
    Ok(values.into_pyarray(py))
}

/// A central equal-tailed interval across the horizon from core quantiles.
pub(crate) fn quantile_interval<'py>(
    py: Python<'py>,
    level: f64,
    quantiles: impl FnOnce(&[f64]) -> Result<Vec<ForecastQuantile>, CoreBayesianForecastError>,
) -> PyResult<PyIntervalArrays<'py>> {
    let [lower, upper] = quantile_values(interval_probabilities(level)?, quantiles)?;
    Ok(interval_arrays(py, (lower, upper)))
}

/// A core summary across the horizon as a NumPy array.
pub(crate) fn summary_array<'py>(
    py: Python<'py>,
    summary: Result<Vec<f64>, CoreBayesianForecastError>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(summary.map_err(bayesian_forecast_error)?.into_pyarray(py))
}

/// What the `summary`, `diagnostics`, `sampler_stats` and `to_arviz` methods
/// every forecasting fit shares need to know about it. PyO3 cannot inherit
/// methods between `#[pyclass]`es, so each class keeps one-line wrappers
/// around the `fit_*` functions below.
pub(crate) trait ForecastFit {
    /// The sampler named in `sampler_stats`, e.g. `"conjugate Gibbs/FFBS"`.
    fn sampler(&self) -> &'static str;
    /// Which retained quantities the diagnostics cover.
    fn coverage(&self) -> &'static str;
    fn report(&self) -> DiagnosticsReport;
    /// `(chains, draws)` of the retained posterior.
    fn shape(&self) -> (usize, usize);
    /// Parameter draws keyed by name, each with leading `(chain, draw)` axes.
    fn posterior<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;

    /// The sampler line printed under the summary table.
    fn summary_line(&self) -> String {
        format!(
            "Sampler: {}; acceptance and divergences unavailable",
            self.sampler()
        )
    }
}

pub(crate) fn fit_summary(fit: &impl ForecastFit) -> String {
    fit.report()
        .to_table_with_sampler(Some(&fit.summary_line()))
}

pub(crate) fn fit_diagnostics<'py>(
    py: Python<'py>,
    fit: &impl ForecastFit,
) -> PyResult<Bound<'py, PyList>> {
    forecast_diagnostics::diagnostics_list(py, &fit.report())
}

pub(crate) fn fit_sampler_stats<'py>(
    py: Python<'py>,
    fit: &impl ForecastFit,
) -> PyResult<Bound<'py, PyDict>> {
    let (chains, draws) = fit.shape();
    forecast_diagnostics::sampler_stats(py, fit.sampler(), chains, draws, fit.coverage())
}

/// Export the parameter draws and the fitted series to ArviZ.
pub(crate) fn fit_to_arviz<'py>(
    py: Python<'py>,
    fit: &impl ForecastFit,
    observations: &[f64],
) -> PyResult<Bound<'py, PyAny>> {
    let az = py.import("arviz")?;
    let groups = PyDict::new(py);
    groups.set_item("posterior", fit.posterior(py)?)?;
    let observed = PyDict::new(py);
    observed.set_item("y", PyArray1::from_slice(py, observations))?;
    groups.set_item("observed_data", observed)?;
    arviz_from_groups(&az, groups)
}

/// Count of non-missing observations.
pub(crate) fn observed_count(observations: &[f64]) -> usize {
    observations.iter().filter(|value| !value.is_nan()).count()
}

/// Lower and upper bounds as a pair of NumPy arrays.
pub(crate) fn interval_arrays<'py>(
    py: Python<'py>,
    (lower, upper): (Vec<f64>, Vec<f64>),
) -> PyIntervalArrays<'py> {
    (lower.into_pyarray(py), upper.into_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInverseGammaPrior>()?;
    Ok(())
}
