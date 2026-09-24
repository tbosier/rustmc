//! Bayesian local-level bindings.
use crate::forecast_support::*;
use crate::StateSpaceError;
use crate::{arviz_from_groups, forecast_batch, forecast_diagnostics, regression};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::{
    fit_bayesian_local_level, BayesianLocalLevelConfig as CoreBayesianLocalLevelConfig,
    InverseGammaPrior as CoreInverseGammaPrior, LocalLevelPosterior as CoreLocalLevelPosterior,
    LocalLevelPosteriorDraw as CoreLocalLevelPosteriorDraw,
    PosteriorPredictiveForecast as CorePosteriorPredictiveForecast,
};
use rustmc_core::state_space::LinearGaussianStateSpace as CoreLinearGaussianStateSpace;

pub(crate) fn local_level_parameter_array<'py, F>(
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
    pub(crate) fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    pub(crate) fn draws(&self) -> usize {
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
pub(crate) struct PyBayesianForecastResult {
    pub(crate) inner: CorePosteriorPredictiveForecast,
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
