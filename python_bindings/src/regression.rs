use crate::forecast_support::*;
use crate::InferenceError;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::InverseGammaPrior as CoreInverseGammaPrior;
use rustmc_core::bayesian_regression::{
    self as core, GaussianCoefficientPrior, RegressionConfig, RegressionForecast,
    RegressionPosterior,
};
use rustmc_core::forecast_common::{path_means, path_quantiles};
use rustmc_core::state_space::LinearGaussianStateSpace as CoreLinearGaussianStateSpace;

#[pyclass(name = "GaussianCoefficientPrior", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyGaussianCoefficientPrior {
    pub(crate) inner: GaussianCoefficientPrior,
}
#[pymethods]
impl PyGaussianCoefficientPrior {
    #[new]
    fn new(mean: &Bound<'_, PyAny>, covariance: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mean = real_vector(mean, "mean")?;
        let covariance = real_matrix(covariance, "covariance")?.concat();
        Ok(Self {
            inner: GaussianCoefficientPrior::new(mean, covariance).map_err(inference_error)?,
        })
    }
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.mean.clone().into_pyarray(py)
    }
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let p = self.inner.mean.len();
        Array2::from_shape_fn((p, p), |(i, j)| self.inner.covariance[i * p + j]).into_pyarray(py)
    }
}

#[pyfunction]
#[pyo3(signature=(count, period, harmonics, start=0))]
fn fourier_design<'py>(
    py: Python<'py>,
    count: usize,
    period: usize,
    harmonics: usize,
    start: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let width = core::fourier_width(period, harmonics)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let rows = core::fourier_design(count, period, harmonics, start)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Array2::from_shape_vec((count, width), rows.concat())
        .map(|design| design.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn fit(
    py: Python<'_>,
    observations: Vec<f64>,
    exog: &Bound<'_, PyAny>,
    prior: Option<PyRef<'_, PyGaussianCoefficientPrior>>,
    mut config: RegressionConfig,
) -> PyResult<PyObject> {
    config.coefficient_prior = prior
        .ok_or_else(|| {
            InferenceError::new_err(
                "exog requires an explicit GaussianCoefficientPrior via coefficient_prior",
            )
        })?
        .inner
        .clone();
    let design = real_matrix(exog, "exog")?;
    let posterior = py
        .allow_threads(|| core::fit_regression(&observations, &design, &config))
        .map_err(inference_error)?;
    Ok(Py::new(
        py,
        PyBayesianRegressionFit {
            posterior,
            observations,
        },
    )?
    .into_any())
}
pub(crate) fn config(
    structural_model: CoreLinearGaussianStateSpace,
    variance_priors: Vec<CoreInverseGammaPrior>,
    variance_names: Vec<&str>,
    observation_variance_prior: CoreInverseGammaPrior,
    seasonal: bool,
    sampling: (usize, usize, usize, usize, u64),
) -> RegressionConfig {
    let (num_chains, num_draws, num_warmup, thinning, seed) = sampling;
    RegressionConfig {
        innovation_indices: (0..variance_priors.len()).collect(),
        structural_model,
        variance_priors,
        variance_names: variance_names.into_iter().map(str::to_string).collect(),
        observation_variance_prior,
        coefficient_prior: GaussianCoefficientPrior {
            mean: vec![],
            covariance: vec![],
        },
        num_chains,
        num_draws,
        num_warmup,
        thinning,
        seed,
        seasonal,
    }
}

#[pyclass(name = "BayesianRegressionFit", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianRegressionFit {
    pub(crate) posterior: RegressionPosterior,
    pub(crate) observations: Vec<f64>,
}
impl ForecastFit for PyBayesianRegressionFit {
    fn sampler(&self) -> &'static str {
        "joint conjugate Gibbs/FFBS"
    }
    fn coverage(&self) -> &'static str {
        "all variance parameters, regression coefficients and terminal structural states; historical states are not retained"
    }
    fn report(&self) -> DiagnosticsReport {
        self.posterior.diagnostics()
    }
    fn shape(&self) -> (usize, usize) {
        chain_shape(&self.posterior.chains)
    }
    fn posterior<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let chains = &self.posterior.chains;
        let config = &self.posterior.config;
        let result = PyDict::new(py);
        for (index, name) in config.variance_names.iter().enumerate() {
            result.set_item(name, draw_array(py, chains, |draw| draw.variances[index]))?;
        }
        result.set_item(
            "observation_variance",
            draw_array(py, chains, |draw| draw.observation_variance),
        )?;
        result.set_item(
            "coefficients",
            draw_vector_array(
                py,
                chains,
                config.coefficient_prior.mean.len(),
                |draw, index| draw.coefficients[index],
            ),
        )?;
        result.set_item(
            "terminal_state",
            draw_vector_array(
                py,
                chains,
                config.structural_model.dimension(),
                |draw, index| draw.terminal_state[index],
            ),
        )?;
        Ok(result)
    }
}

#[pymethods]
impl PyBayesianRegressionFit {
    fn summary(&self) -> String {
        fit_summary(self)
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        fit_diagnostics(py, self)
    }
    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        fit_sampler_stats(py, self)
    }
    #[getter]
    fn chains(&self) -> usize {
        self.shape().0
    }
    #[getter]
    fn draws(&self) -> usize {
        self.shape().1
    }
    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }
    #[getter]
    fn observed_count(&self) -> usize {
        observed_count(&self.observations)
    }
    #[getter]
    fn warmup(&self) -> usize {
        self.posterior.config.num_warmup
    }
    #[getter]
    fn thin(&self) -> usize {
        self.posterior.config.thinning
    }
    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.posterior(py)
    }
    #[pyo3(signature=(steps, seed=43, *, exog=None))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
        exog: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBayesianRegressionForecast> {
        let exog = exog.ok_or_else(|| {
            InferenceError::new_err("future exog is required for regression forecasts")
        })?;
        let design = real_matrix(exog, "exog")?;
        if design.len() != steps {
            return Err(InferenceError::new_err(
                "future exog row count must equal steps",
            ));
        }
        let inner = py
            .allow_threads(|| self.posterior.forecast(&design, seed))
            .map_err(inference_error)?;
        Ok(PyBayesianRegressionForecast {
            inner,
            seasonal: self.posterior.config.seasonal,
            dimension: self.posterior.config.structural_model.dimension(),
        })
    }
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fit_to_arviz(py, self, &self.observations)
    }
}

#[pyclass(name = "BayesianRegressionForecast", module = "rustmc")]
pub(crate) struct PyBayesianRegressionForecast {
    pub(crate) inner: RegressionForecast,
    pub(crate) seasonal: bool,
    pub(crate) dimension: usize,
}
#[pymethods]
impl PyBayesianRegressionForecast {
    #[getter]
    fn chains(&self) -> usize {
        chain_shape(&self.inner.observation_paths).0
    }
    #[getter]
    fn draws(&self) -> usize {
        chain_shape(&self.inner.observation_paths).1
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner
            .observation_paths
            .first()
            .and_then(|chain| chain.first())
            .map_or(0, Vec::len)
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.observation_paths)
    }
    #[getter]
    fn cumulative_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.cumulative_observation_paths)
    }
    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.level_paths)
    }
    #[getter]
    fn level_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.state_samples(py)
    }
    #[getter]
    fn seasonal_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if !self.seasonal {
            return Err(InferenceError::new_err(
                "model has no stochastic seasonal component",
            ));
        }
        Ok(path_array(py, &self.inner.secondary_paths))
    }
    #[getter]
    fn slope_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if self.seasonal || self.dimension != 2 {
            return Err(InferenceError::new_err("model has no slope component"));
        }
        Ok(path_array(py, &self.inner.secondary_paths))
    }
    #[getter]
    fn regression_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.regression_paths)
    }
    #[getter]
    fn mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.mean_paths)
    }
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, path_means(&self.inner.observation_paths))
    }
    #[getter]
    fn cumulative_observation_mean<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, path_means(&self.inner.cumulative_observation_paths))
    }
    /// Equal-tailed interval holding `probability` of the draws; unlike
    /// `interval`, zero and one are accepted.
    #[pyo3(signature=(probability=0.9))]
    fn observation_interval<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        central_interval(py, &self.inner.observation_paths, probability)
    }
    #[pyo3(signature=(probability=0.9))]
    fn cumulative_observation_interval<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        central_interval(py, &self.inner.cumulative_observation_paths, probability)
    }
    #[pyo3(signature=(level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| {
            path_quantiles(&self.inner.observation_paths, p)
        })
    }
    #[pyo3(signature=(level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| {
            path_quantiles(&self.inner.cumulative_observation_paths, p)
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
}
/// The central interval holding `probability` of the draws, which may be
/// zero (the median twice) or one (the range).
fn central_interval<'py>(
    py: Python<'py>,
    paths: &core::Paths,
    probability: f64,
) -> PyResult<PyIntervalArrays<'py>> {
    validate_probability(probability)?;
    let [lower, upper] = quantile_values(
        [(1.0 - probability) / 2.0, (1.0 + probability) / 2.0],
        |p| path_quantiles(paths, p),
    )?;
    Ok(interval_arrays(py, (lower, upper)))
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianCoefficientPrior>()?;
    m.add_class::<PyBayesianRegressionFit>()?;
    m.add_class::<PyBayesianRegressionForecast>()?;
    m.add_function(wrap_pyfunction!(fourier_design, m)?)?;
    Ok(())
}
