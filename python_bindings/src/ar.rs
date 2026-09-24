//! Bayesian autoregression bindings.
use crate::forecast_batch;
use crate::forecast_support::*;
use crate::InferenceError;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_ar::{
    fit_bayesian_ar, BayesianArConfig as CoreBayesianArConfig,
    BayesianArForecast as CoreBayesianArForecast, BayesianArPosterior as CoreBayesianArPosterior,
    NormalInverseGammaPrior as CoreNormalInverseGammaPrior,
};

/// Conjugate prior for a Gaussian autoregression.
///
/// If beta contains ``[intercept, lag_1, ..., lag_p]``, then
/// ``beta | sigma2 ~ Normal(mean, sigma2 * precision^-1)`` and
/// ``sigma2 ~ InverseGamma(variance_shape, variance_scale)``.
#[pyclass(name = "NormalInverseGammaPrior", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyNormalInverseGammaPrior {
    pub(crate) inner: CoreNormalInverseGammaPrior,
}

#[pymethods]
impl PyNormalInverseGammaPrior {
    #[new]
    fn new(
        coefficient_mean: &Bound<'_, PyAny>,
        coefficient_precision: &Bound<'_, PyAny>,
        variance_shape: f64,
        variance_scale: f64,
    ) -> PyResult<Self> {
        let mean = real_vector(coefficient_mean, "coefficient_mean")?;
        let precision = real_matrix(coefficient_precision, "coefficient_precision")?;
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
pub(crate) struct PyBayesianAutoRegression {
    pub(crate) order: usize,
    pub(crate) prior: CoreNormalInverseGammaPrior,
}

#[pymethods]
impl PyBayesianAutoRegression {
    /// Fit independent ragged cells on one bounded native worker pool.
    ///
    /// AR posterior draws are exact and independent, so `warmup` and `thin`
    /// only reach Gibbs-sampled cells named in `models`; a batch with no such
    /// cell refuses values other than the defaults instead of ignoring them.
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
            self.batch_config(chains, draws),
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
            return Err(InferenceError::new_err(
                "invalid configuration: AR order must be at least one",
            ));
        }
        let expected = order.checked_add(1).ok_or_else(|| {
            InferenceError::new_err(
                "invalid configuration: AR order is too large to represent its coefficients",
            )
        })?;
        if prior.inner.coefficient_mean.len() != expected {
            return Err(InferenceError::new_err(format!(
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
        observations: &Bound<'_, PyAny>,
        chains: usize,
        draws: usize,
        seed: u64,
    ) -> PyResult<PyBayesianArFit> {
        let observations = real_vector(observations, "observations")?;
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
pub(crate) struct PyBayesianArFit {
    pub(crate) posterior: CoreBayesianArPosterior,
    pub(crate) observations: Vec<f64>,
    pub(crate) config: CoreBayesianArConfig,
}

impl ForecastFit for PyBayesianArFit {
    fn sampler(&self) -> &'static str {
        "exact conjugate independent draws"
    }
    fn coverage(&self) -> &'static str {
        "coefficients and innovation variance"
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
            "coefficient",
            draw_vector_array(py, chains, self.posterior.order + 1, |draw, index| {
                draw.coefficients[index]
            }),
        )?;
        samples.set_item(
            "innovation_variance",
            draw_array(py, chains, |draw| draw.innovation_variance),
        )?;
        samples.set_item(
            "innovation_sd",
            draw_array(py, chains, |draw| draw.innovation_variance.sqrt()),
        )?;
        Ok(samples)
    }
}

#[pymethods]
impl PyBayesianArFit {
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
    fn order(&self) -> usize {
        self.posterior.order
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
    fn regression_count(&self) -> usize {
        self.observations.len() - self.posterior.order
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.config.seed
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.posterior(py)
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
        fit_to_arviz(py, self, &self.observations)
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
pub(crate) struct PyBayesianArForecast {
    pub(crate) inner: CoreBayesianArForecast,
}

#[pymethods]
impl PyBayesianArForecast {
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
    fn conditional_mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.conditional_mean_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn conditional_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.conditional_mean_means())
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        summary_array(py, self.inner.observation_means())
    }

    fn conditional_mean_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| {
            self.inner.conditional_mean_quantiles(p)
        })
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        quantile_array(py, probability, |p| self.inner.observation_quantiles(p))
    }

    /// Pointwise equal-tailed interval for the recursive conditional mean.
    #[pyo3(signature = (level=0.95))]
    fn conditional_mean_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        quantile_interval(py, level, |p| self.inner.conditional_mean_quantiles(p))
    }

    /// Pointwise equal-tailed posterior-predictive interval for future observations.
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
            "BayesianARForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

impl PyBayesianAutoRegression {
    /// AR draws are exact and independent, so there is no warmup or thinning.
    pub(crate) fn batch_config(&self, chains: usize, draws: usize) -> forecast_batch::Config {
        let seed = 0;
        forecast_batch::Config::Ar(CoreBayesianArConfig {
            order: self.order,
            prior: self.prior.clone(),
            num_chains: chains,
            num_draws: draws,
            seed,
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNormalInverseGammaPrior>()?;
    m.add_class::<PyBayesianAutoRegression>()?;
    m.add("BayesianAR", m.getattr("BayesianAutoRegression")?)?;
    m.add_class::<PyBayesianArFit>()?;
    m.add_class::<PyBayesianArForecast>()?;
    Ok(())
}
