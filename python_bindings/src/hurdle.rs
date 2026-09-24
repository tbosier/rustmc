//! Python bindings for sparse nonnegative amount forecasting.
use crate::forecast_batch;
use crate::forecast_support::*;
use numpy::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::forecast_common::{cumulative_paths, path_quantiles, Paths};
use rustmc_core::hurdle::{
    fit_hurdle_lognormal, HurdleLogNormalConfig, HurdleLogNormalForecast, HurdleLogNormalPosterior,
};

#[pyclass(name = "BayesianHurdleLogNormal", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyHurdleLogNormal {
    pub(crate) config: HurdleLogNormalConfig,
}

impl PyHurdleLogNormal {
    pub(crate) fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let mut config = self.config.clone();
        config.num_chains = chains;
        config.num_draws = draws;
        config.num_warmup = warmup;
        config.thinning = thin;
        forecast_batch::Config::Hurdle(config)
    }
}

#[pymethods]
impl PyHurdleLogNormal {
    /// Fit independent sparse-amount cells with stable IDs on one native worker pool.
    /// Optional models permit per-cell priors and mixed forecasting model families.
    /// Hurdle cells do not support exog; regression cells in a mixed batch may use it.
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

    /// Independent Beta occurrence and dynamic lognormal positive amounts.
    /// Log-variance inverse-gamma priors are truncated at the explicit upper bounds.
    #[new]
    #[pyo3(signature = (process_variance_prior, observation_variance_prior, occurrence_alpha=1.0, occurrence_beta=1.0, initial_log_level=0.0, initial_variance=1.0, process_variance_upper=1.0, observation_variance_upper=4.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        process_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        occurrence_alpha: f64,
        occurrence_beta: f64,
        initial_log_level: f64,
        initial_variance: f64,
        process_variance_upper: f64,
        observation_variance_upper: f64,
    ) -> PyResult<Self> {
        let config = HurdleLogNormalConfig {
            process_variance_prior: process_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
            occurrence_alpha,
            occurrence_beta,
            initial_log_level,
            initial_variance,
            process_variance_upper,
            observation_variance_upper,
            num_chains: 4,
            num_draws: 1000,
            num_warmup: 500,
            thinning: 1,
            seed: 42,
        };
        config.validate().map_err(bayesian_forecast_error)?;
        Ok(Self { config })
    }

    #[pyo3(signature = (observations, chains=4, draws=1000, warmup=500, thin=1, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
    ) -> PyResult<PyHurdleFit> {
        let observations = real_vector(observations, "observations")?;
        let mut config = self.config.clone();
        config.num_chains = chains;
        config.num_draws = draws;
        config.num_warmup = warmup;
        config.thinning = thin;
        config.seed = seed;
        let posterior = py
            .allow_threads(|| fit_hurdle_lognormal(&observations, &config))
            .map_err(bayesian_forecast_error)?;
        Ok(PyHurdleFit {
            posterior,
            observations,
            config,
        })
    }

    #[getter]
    fn process_variance_upper(&self) -> f64 {
        self.config.process_variance_upper
    }
    #[getter]
    fn observation_variance_upper(&self) -> f64 {
        self.config.observation_variance_upper
    }
    fn __repr__(&self) -> String {
        format!("BayesianHurdleLogNormal(occurrence_alpha={}, occurrence_beta={}, initial_log_level={}, process_variance_upper={}, observation_variance_upper={})",
            self.config.occurrence_alpha, self.config.occurrence_beta, self.config.initial_log_level,
            self.config.process_variance_upper, self.config.observation_variance_upper)
    }
}

#[pyclass(name = "BayesianHurdleLogNormalFit", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyHurdleFit {
    pub(crate) posterior: HurdleLogNormalPosterior,
    pub(crate) observations: Vec<f64>,
    pub(crate) config: HurdleLogNormalConfig,
}

impl ForecastFit for PyHurdleFit {
    fn sampler(&self) -> &'static str {
        if self.posterior.positive_count == 0 {
            "independent_prior_and_beta"
        } else {
            "gibbs_ffbs_hurdle_lognormal"
        }
    }
    fn summary_line(&self) -> String {
        if self.posterior.positive_count == 0 {
            "Sampler: independent Beta and truncated-prior severity draws (no positive observations)"
        } else {
            "Sampler: Beta occurrence and conjugate truncated-variance Gibbs/FFBS severity"
        }
        .into()
    }
    fn coverage(&self) -> &'static str {
        "payment probability, variance parameters, and terminal log level"
    }
    fn report(&self) -> DiagnosticsReport {
        self.posterior.diagnostics()
    }
    fn shape(&self) -> (usize, usize) {
        chain_shape(&self.posterior.chains)
    }
    fn posterior<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = PyDict::new(py);
        let values = self.posterior.parameter_samples();
        for (index, name) in HurdleLogNormalPosterior::parameter_names()
            .iter()
            .enumerate()
        {
            samples.set_item(name, draw_array(py, &values, |draw| draw[index]))?;
        }
        Ok(samples)
    }
}

#[pymethods]
impl PyHurdleFit {
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
        self.posterior.time_count
    }
    #[getter]
    fn observed_count(&self) -> usize {
        self.posterior.observed_count
    }
    #[getter]
    fn positive_count(&self) -> usize {
        self.posterior.positive_count
    }
    #[getter]
    fn severity_informed_by_data(&self) -> bool {
        self.positive_count() > 0
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
        self.posterior(py)
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        fit_diagnostics(py, self)
    }
    fn summary(&self) -> String {
        fit_summary(self)
    }
    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = fit_sampler_stats(py, self)?;
        stats.set_item(
            "severity_informed_by_data",
            self.severity_informed_by_data(),
        )?;
        stats.set_item("process_variance_upper", self.config.process_variance_upper)?;
        stats.set_item(
            "observation_variance_upper",
            self.config.observation_variance_upper,
        )?;
        Ok(stats)
    }
    #[pyo3(signature = (steps, seed=43))]
    fn forecast(&self, py: Python<'_>, steps: usize, seed: u64) -> PyResult<PyHurdleForecast> {
        let inner = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyHurdleForecast { inner })
    }
    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fit_to_arviz(py, self, &self.observations)
    }
}

#[pyclass(name = "BayesianHurdleLogNormalForecast", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyHurdleForecast {
    pub(crate) inner: HurdleLogNormalForecast,
}

#[pymethods]
impl PyHurdleForecast {
    #[getter]
    fn chains(&self) -> usize {
        chain_shape(&self.inner.paths.observation_paths).0
    }
    #[getter]
    fn draws(&self) -> usize {
        chain_shape(&self.inner.paths.observation_paths).1
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.paths.horizon()
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.paths.observation_paths)
    }
    /// Conditional arithmetic means including probability of no payment.
    #[getter]
    fn mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, self.inner.expected_value_paths())
    }
    #[getter]
    fn positive_mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.positive_mean_paths)
    }
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.paths.observation_means())
    }
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.expected_value_means())
    }
    #[getter]
    fn cumulative_observation_samples<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        Ok(path_array(py, &self.cumulative_paths()?))
    }
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.paths.observation_quantiles(p))
    }
    #[pyo3(signature = (level=0.95))]
    fn mean_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.expected_value_quantiles(p))
    }
    #[pyo3(signature = (level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        let paths = self.cumulative_paths()?;
        quantile_interval(py, level, |p| path_quantiles(&paths, p))
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

impl PyHurdleForecast {
    fn cumulative_paths(&self) -> PyResult<Paths> {
        cumulative_paths(&self.inner.paths.observation_paths).map_err(bayesian_forecast_error)
    }
}

#[pyfunction(name = "hurdle_lognormal_logp")]
fn logp(y: f64, payment_probability: f64, log_level: f64, log_variance: f64) -> PyResult<f64> {
    rustmc_core::hurdle::hurdle_lognormal_logp(y, payment_probability, log_level, log_variance)
        .map_err(bayesian_forecast_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHurdleLogNormal>()?;
    m.add_class::<PyHurdleFit>()?;
    m.add_class::<PyHurdleForecast>()?;
    m.add_function(wrap_pyfunction!(logp, m)?)?;
    Ok(())
}
