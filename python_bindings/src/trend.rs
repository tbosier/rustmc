//! Bayesian local-linear-trend bindings.
use crate::forecast_support::*;
use crate::InferenceError;
use crate::{forecast_batch, regression};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::InverseGammaPrior as CoreInverseGammaPrior;
use rustmc_core::bayesian_trend::{
    fit_bayesian_local_linear_trend,
    BayesianLocalLinearTrendConfig as CoreBayesianLocalLinearTrendConfig,
    LocalLinearTrendPosterior as CoreLocalLinearTrendPosterior,
    TrendPosteriorPredictiveForecast as CoreTrendPosteriorPredictiveForecast,
};
use rustmc_core::state_space::LinearGaussianStateSpace as CoreLinearGaussianStateSpace;

/// Bayesian local-linear-trend model with stochastic level and slope.
#[pyclass(name = "BayesianLocalLinearTrend", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianLocalLinearTrend {
    pub(crate) initial_mean: [f64; 2],
    pub(crate) initial_covariance: [f64; 4],
    pub(crate) level_variance_prior: CoreInverseGammaPrior,
    pub(crate) slope_variance_prior: CoreInverseGammaPrior,
    pub(crate) observation_variance_prior: CoreInverseGammaPrior,
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
            return Err(InferenceError::new_err(
                "invalid configuration: initial level and slope must be finite",
            ));
        }
        if !initial_level_variance.is_finite()
            || initial_level_variance <= 0.0
            || !initial_slope_variance.is_finite()
            || initial_slope_variance <= 0.0
            || !initial_level_slope_covariance.is_finite()
        {
            return Err(InferenceError::new_err(
                "invalid configuration: initial variances must be finite and positive and covariance must be finite",
            ));
        }
        let level_scale = initial_level_variance.sqrt();
        let scaled_covariance = initial_level_slope_covariance / level_scale;
        let slope_remainder = initial_slope_variance - scaled_covariance * scaled_covariance;
        if !slope_remainder.is_finite() || slope_remainder <= 0.0 {
            return Err(InferenceError::new_err(
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
        observations: &Bound<'_, PyAny>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
        exog: Option<&Bound<'_, PyAny>>,
        coefficient_prior: Option<PyRef<'_, regression::PyGaussianCoefficientPrior>>,
    ) -> PyResult<PyObject> {
        let observations = real_vector(observations, "observations")?;
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
                .map_err(inference_error)?,
                vec![self.level_variance_prior, self.slope_variance_prior],
                vec!["level_variance", "slope_variance"],
                self.observation_variance_prior,
                false,
                (chains, draws, warmup, thin, seed),
            );
            return regression::fit(py, observations, exog, coefficient_prior, config);
        }
        if coefficient_prior.is_some() {
            return Err(InferenceError::new_err("coefficient_prior requires exog"));
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
pub(crate) struct PyBayesianLocalLinearTrendFit {
    pub(crate) posterior: CoreLocalLinearTrendPosterior,
    pub(crate) observations: Vec<f64>,
    pub(crate) config: CoreBayesianLocalLinearTrendConfig,
}

impl ForecastFit for PyBayesianLocalLinearTrendFit {
    fn sampler(&self) -> &'static str {
        "conjugate Gibbs/FFBS"
    }
    fn coverage(&self) -> &'static str {
        "variance parameters, terminal level and slope; historical states are not retained"
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
        for (name, values) in [
            (
                "level_variance",
                draw_array(py, chains, |draw| draw.level_variance),
            ),
            (
                "slope_variance",
                draw_array(py, chains, |draw| draw.slope_variance),
            ),
            (
                "observation_variance",
                draw_array(py, chains, |draw| draw.observation_variance),
            ),
            (
                "level_sd",
                draw_array(py, chains, |draw| draw.level_variance.sqrt()),
            ),
            (
                "slope_sd",
                draw_array(py, chains, |draw| draw.slope_variance.sqrt()),
            ),
            (
                "observation_sd",
                draw_array(py, chains, |draw| draw.observation_variance.sqrt()),
            ),
            (
                "terminal_level",
                draw_array(py, chains, |draw| draw.terminal_level),
            ),
            (
                "terminal_slope",
                draw_array(py, chains, |draw| draw.terminal_slope),
            ),
        ] {
            samples.set_item(name, values)?;
        }
        Ok(samples)
    }
}

#[pymethods]
impl PyBayesianLocalLinearTrendFit {
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
    fn time_count(&self) -> usize {
        self.observations.len()
    }

    #[getter]
    fn observed_count(&self) -> usize {
        observed_count(&self.observations)
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
    ) -> PyResult<PyBayesianTrendForecast> {
        let forecast = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(bayesian_forecast_error)?;
        Ok(PyBayesianTrendForecast { inner: forecast })
    }

    fn to_arviz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fit_to_arviz(py, self, &self.observations)
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
pub(crate) struct PyBayesianTrendForecast {
    pub(crate) inner: CoreTrendPosteriorPredictiveForecast,
}

#[pymethods]
impl PyBayesianTrendForecast {
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
    fn level_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.level_paths)
    }

    #[getter]
    fn slope_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.slope_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn level_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.level_means())
    }

    #[getter]
    fn slope_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.slope_means())
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.observation_means())
    }

    fn level_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| self.inner.level_quantiles(p))
    }

    fn slope_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| self.inner.slope_quantiles(p))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| self.inner.observation_quantiles(p))
    }

    #[pyo3(signature = (level=0.95))]
    fn level_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.level_quantiles(p))
    }

    #[pyo3(signature = (level=0.95))]
    fn slope_interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.slope_quantiles(p))
    }

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
            "BayesianTrendForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

impl PyBayesianLocalLinearTrend {
    pub(crate) fn batch_config(
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBayesianLocalLinearTrend>()?;
    m.add_class::<PyBayesianLocalLinearTrendFit>()?;
    m.add_class::<PyBayesianTrendForecast>()?;
    Ok(())
}
