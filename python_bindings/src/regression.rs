use crate::forecast_support::*;
use crate::{arviz_from_groups, forecast_diagnostics, StateSpaceError};
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
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
    fn new(
        mean: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let mean = state_space_vector(mean);
        let (covariance, _) = state_space_matrix("coefficient covariance", covariance)?;
        Ok(Self {
            inner: GaussianCoefficientPrior::new(mean, covariance).map_err(state_space_error)?,
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
pub(crate) fn rows(array: PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    array
        .as_array()
        .rows()
        .into_iter()
        .map(|r| r.iter().copied().collect())
        .collect()
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
    let width = core::fourier_width(period, harmonics).map_err(state_space_error)?;
    let rows = core::fourier_design(count, period, harmonics, start).map_err(state_space_error)?;
    Array2::from_shape_vec((count, width), rows.concat())
        .map(|design| design.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn fit(
    py: Python<'_>,
    observations: Vec<f64>,
    exog: PyReadonlyArray2<'_, f64>,
    prior: Option<PyRef<'_, PyGaussianCoefficientPrior>>,
    mut config: RegressionConfig,
) -> PyResult<PyObject> {
    config.coefficient_prior = prior
        .ok_or_else(|| {
            StateSpaceError::new_err(
                "exog requires an explicit GaussianCoefficientPrior via coefficient_prior",
            )
        })?
        .inner
        .clone();
    let design = rows(exog);
    let posterior = py
        .allow_threads(|| core::fit_regression(&observations, &design, &config))
        .map_err(state_space_error)?;
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
#[pymethods]
impl PyBayesianRegressionFit {
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: joint conjugate Gibbs/FFBS; acceptance and divergences unavailable",
        ))
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }
    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(py, "joint conjugate Gibbs/FFBS", self.chains(), self.draws(),
            "all variance parameters, regression coefficients and terminal structural states; historical states are not retained")
    }
    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains[0].len()
    }
    #[getter]
    fn time_count(&self) -> usize {
        self.observations.len()
    }
    #[getter]
    fn observed_count(&self) -> usize {
        self.observations.iter().filter(|v| v.is_finite()).count()
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
        let result = PyDict::new(py);
        for (i, name) in self.posterior.config.variance_names.iter().enumerate() {
            result.set_item(
                name,
                Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| {
                    self.posterior.chains[c][d].variances[i]
                })
                .into_pyarray(py),
            )?;
        }
        result.set_item(
            "observation_variance",
            Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| {
                self.posterior.chains[c][d].observation_variance
            })
            .into_pyarray(py),
        )?;
        let p = self.posterior.config.coefficient_prior.mean.len();
        result.set_item(
            "coefficients",
            Array3::from_shape_fn((self.chains(), self.draws(), p), |(c, d, p)| {
                self.posterior.chains[c][d].coefficients[p]
            })
            .into_pyarray(py),
        )?;
        let n = self.posterior.config.structural_model.dimension();
        result.set_item(
            "terminal_state",
            Array3::from_shape_fn((self.chains(), self.draws(), n), |(c, d, p)| {
                self.posterior.chains[c][d].terminal_state[p]
            })
            .into_pyarray(py),
        )?;
        Ok(result)
    }
    #[pyo3(signature=(steps, seed=43, *, exog=None))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
        exog: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<PyBayesianRegressionForecast> {
        let design = rows(exog.ok_or_else(|| {
            StateSpaceError::new_err("future exog is required for regression forecasts")
        })?);
        if design.len() != steps {
            return Err(StateSpaceError::new_err(
                "future exog row count must equal steps",
            ));
        }
        let inner = py
            .allow_threads(|| self.posterior.forecast(&design, seed))
            .map_err(state_space_error)?;
        Ok(PyBayesianRegressionForecast {
            inner,
            seasonal: self.posterior.config.seasonal,
            dimension: self.posterior.config.structural_model.dimension(),
        })
    }
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", self.observations.clone().into_pyarray(py))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
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
        self.inner.observation_paths.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.observation_paths[0].len()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.observation_paths[0][0].len()
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
            return Err(PyValueError::new_err(
                "model has no stochastic seasonal component",
            ));
        }
        Ok(path_array(py, &self.inner.secondary_paths))
    }
    #[getter]
    fn slope_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if self.seasonal || self.dimension != 2 {
            return Err(PyValueError::new_err("model has no slope component"));
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
