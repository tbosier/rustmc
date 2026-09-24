//! Bayesian local-level bindings.
use crate::forecast_support::*;
use crate::StateSpaceError;
use crate::{forecast_batch, regression};
use numpy::{PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::{
    fit_bayesian_local_level, BayesianLocalLevelConfig as CoreBayesianLocalLevelConfig,
    InverseGammaPrior as CoreInverseGammaPrior, LocalLevelPosterior as CoreLocalLevelPosterior,
    PosteriorPredictiveForecast as CorePosteriorPredictiveForecast,
};
use rustmc_core::state_space::LinearGaussianStateSpace as CoreLinearGaussianStateSpace;

/// Bayesian scalar Gaussian local-level model fitted with conjugate
/// forward-filtering/backward-sampling Gibbs updates.
#[pyclass(name = "BayesianLocalLevel", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianLocalLevel {
    pub(crate) initial_mean: f64,
    pub(crate) initial_variance: f64,
    pub(crate) process_variance_prior: CoreInverseGammaPrior,
    pub(crate) observation_variance_prior: CoreInverseGammaPrior,
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
pub(crate) struct PyBayesianLocalLevelFit {
    pub(crate) posterior: CoreLocalLevelPosterior,
    pub(crate) observations: Vec<f64>,
    pub(crate) config: CoreBayesianLocalLevelConfig,
}

impl ForecastFit for PyBayesianLocalLevelFit {
    fn sampler(&self) -> &'static str {
        "conjugate Gibbs/FFBS"
    }
    fn coverage(&self) -> &'static str {
        "variance parameters and terminal level; historical states are not retained"
    }
    fn report(&self) -> DiagnosticsReport {
        self.posterior.diagnostics()
    }
    fn shape(&self) -> (usize, usize) {
        chain_shape(&self.posterior.chains)
    }
    fn posterior<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let chains = &self.posterior.chains;
        let samples = PyDict::new(py);
        samples.set_item(
            "process_variance",
            draw_array(py, chains, |draw| draw.process_variance),
        )?;
        samples.set_item(
            "observation_variance",
            draw_array(py, chains, |draw| draw.observation_variance),
        )?;
        samples.set_item(
            "process_sd",
            draw_array(py, chains, |draw| draw.process_variance.sqrt()),
        )?;
        samples.set_item(
            "observation_sd",
            draw_array(py, chains, |draw| draw.observation_variance.sqrt()),
        )?;
        samples.set_item(
            "terminal_level",
            draw_array(py, chains, |draw| draw.terminal_level),
        )?;
        Ok(samples)
    }
}

#[pymethods]
impl PyBayesianLocalLevelFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
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
    fn observed_count(&self) -> usize {
        observed_count(&self.observations)
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
        self.posterior(py)
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
        fit_to_arviz(py, self, &self.observations)
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
pub(crate) struct PyBayesianForecastResult {
    pub(crate) inner: CorePosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianForecastResult {
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
        self.inner.horizon()
    }

    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.state_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn state_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.state_means())
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.observation_means())
    }

    fn state_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| self.inner.state_quantiles(p))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| self.inner.observation_quantiles(p))
    }

    /// Equal-tailed pointwise posterior interval for the latent state.
    #[pyo3(signature = (level=0.95))]
    fn state_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.state_quantiles(p))
    }

    /// Equal-tailed pointwise posterior-predictive interval for observations.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.observation_quantiles(p))
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

impl PyBayesianLocalLevel {
    pub(crate) fn batch_config(
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBayesianLocalLevel>()?;
    m.add_class::<PyBayesianLocalLevelFit>()?;
    m.add_class::<PyBayesianForecastResult>()?;
    Ok(())
}
