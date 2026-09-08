//! Python bindings for sparse nonnegative amount forecasting.
use super::*;
use rustmc_core::forecast_diagnostics::parameter_diagnostics;
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
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
    ) -> PyResult<PyHurdleFit> {
        let observations = state_space_vector(observations);
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

impl PyHurdleFit {
    pub(crate) fn report(&self) -> rustmc_core::diagnostics::DiagnosticsReport {
        parameter_diagnostics(
            &self.posterior.parameter_samples(),
            &HurdleLogNormalPosterior::parameter_names(),
        )
    }
}

#[pymethods]
impl PyHurdleFit {
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
        let samples = PyDict::new(py);
        let values = self.posterior.parameter_samples();
        for (i, name) in HurdleLogNormalPosterior::parameter_names()
            .iter()
            .enumerate()
        {
            let array =
                Array2::from_shape_fn((self.chains(), self.draws()), |(c, d)| values[c][d][i]);
            samples.set_item(name, array.into_pyarray(py))?;
        }
        Ok(samples)
    }
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.report())
    }
    fn summary(&self) -> String {
        let sampler = if self.positive_count() == 0 {
            "Sampler: independent Beta and truncated-prior severity draws (no positive observations)"
        } else {
            "Sampler: Beta occurrence and conjugate truncated-variance Gibbs/FFBS severity"
        };
        self.report().to_table_with_sampler(Some(sampler))
    }
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let name = if self.positive_count() == 0 {
            "independent_prior_and_beta"
        } else {
            "gibbs_ffbs_hurdle_lognormal"
        };
        let stats = forecast_diagnostics::sampler_stats(
            py,
            name,
            self.chains(),
            self.draws(),
            "payment probability, variance parameters, and terminal log level",
        )?;
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
        let kwargs = PyDict::new(py);
        kwargs.set_item("posterior", self.get_samples_2d(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", self.observations.clone().into_pyarray(py))?;
        kwargs.set_item("observed_data", observed)?;
        py.import("arviz")?
            .getattr("from_dict")?
            .call((), Some(&kwargs))
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
        self.inner.paths.observation_paths.len()
    }
    #[getter]
    fn draws(&self) -> usize {
        self.inner.paths.observation_paths[0].len()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.paths.horizon()
    }
    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.paths.observation_paths)
    }
    /// Conditional arithmetic means including probability of no payment.
    #[getter]
    fn mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.paths.state_paths)
    }
    #[getter]
    fn positive_mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.positive_mean_paths)
    }
    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .paths
            .observation_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .paths
            .state_means()
            .map_err(bayesian_forecast_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn cumulative_observation_samples<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        Ok(local_level_path_array(py, &self.cumulative_paths()?))
    }
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        self.quantile_interval(py, &self.inner.paths, level, false)
    }
    #[pyo3(signature = (level=0.95))]
    fn mean_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        self.quantile_interval(py, &self.inner.paths, level, true)
    }
    #[pyo3(signature = (level=0.95))]
    fn cumulative_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        let paths = CorePosteriorPredictiveForecast {
            state_paths: Vec::new(),
            observation_paths: self.cumulative_paths()?,
        };
        self.quantile_interval(py, &paths, level, false)
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
    fn cumulative_paths(&self) -> PyResult<Vec<Vec<Vec<f64>>>> {
        self.inner
            .paths
            .observation_paths
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|path| {
                        let mut sum = 0.0;
                        path.iter()
                            .map(|x| {
                                sum += x;
                                if sum.is_finite() {
                                    Ok(sum)
                                } else {
                                    Err(InferenceError::new_err("cumulative payment overflowed"))
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
    fn quantile_interval<'py>(
        &self,
        py: Python<'py>,
        paths: &CorePosteriorPredictiveForecast,
        level: f64,
        mean: bool,
    ) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probs = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let q = if mean {
            paths.state_quantiles(&probs)
        } else {
            paths.observation_quantiles(&probs)
        }
        .map_err(bayesian_forecast_error)?;
        Ok((
            q[0].values.clone().into_pyarray(py),
            q[1].values.clone().into_pyarray(py),
        ))
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
