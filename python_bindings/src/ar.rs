//! Bayesian autoregression bindings.
use crate::forecast_support::*;
use crate::StateSpaceError;
use crate::{arviz_from_groups, forecast_batch, forecast_diagnostics};
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_ar::{
    fit_bayesian_ar, BayesianArConfig as CoreBayesianArConfig,
    BayesianArForecast as CoreBayesianArForecast, BayesianArPosterior as CoreBayesianArPosterior,
    BayesianArPosteriorDraw as CoreBayesianArPosteriorDraw,
    NormalInverseGammaPrior as CoreNormalInverseGammaPrior,
};

pub(crate) fn ar_parameter_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreBayesianArPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreBayesianArPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

pub(crate) fn ar_coefficient_array<'py>(
    py: Python<'py>,
    posterior: &CoreBayesianArPosterior,
) -> Bound<'py, PyArray3<f64>> {
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    let coefficient_count = posterior.order + 1;
    Array3::from_shape_fn(
        (chains, draws, coefficient_count),
        |(chain, draw, coefficient)| posterior.chains[chain][draw].coefficients[coefficient],
    )
    .into_pyarray(py)
}

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
        coefficient_mean: PyReadonlyArray1<'_, f64>,
        coefficient_precision: PyReadonlyArray2<'_, f64>,
        variance_shape: f64,
        variance_scale: f64,
    ) -> PyResult<Self> {
        let mean = coefficient_mean.as_array().to_vec();
        let precision = coefficient_precision
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
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
    fn new(order: usize, prior: PyRef<'_, PyNormalInverseGammaPrior>) -> PyResult<Self> {
        if order == 0 {
            return Err(StateSpaceError::new_err(
                "invalid configuration: AR order must be at least one",
            ));
        }
        let expected = order.checked_add(1).ok_or_else(|| {
            StateSpaceError::new_err(
                "invalid configuration: AR order is too large to represent its coefficients",
            )
        })?;
        if prior.inner.coefficient_mean.len() != expected {
            return Err(StateSpaceError::new_err(format!(
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
        observations: PyReadonlyArray1<'_, f64>,
        chains: usize,
        draws: usize,
        seed: u64,
    ) -> PyResult<PyBayesianArFit> {
        let observations = state_space_vector(observations);
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

#[pymethods]
impl PyBayesianArFit {
    /// Rank-normalized folded split R-hat, bulk/tail ESS, MCSE and HDIs.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: exact conjugate independent draws; acceptance and divergences unavailable",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "exact conjugate independent draws",
            self.chains(),
            self.draws(),
            "coefficients and innovation variance",
        )
    }

    #[getter]
    fn order(&self) -> usize {
        self.posterior.order
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
        let samples = PyDict::new(py);
        samples.set_item("coefficient", ar_coefficient_array(py, &self.posterior))?;
        samples.set_item(
            "innovation_variance",
            ar_parameter_array(py, &self.posterior, |draw| draw.innovation_variance),
        )?;
        samples.set_item(
            "innovation_sd",
            ar_parameter_array(py, &self.posterior, |draw| draw.innovation_variance.sqrt()),
        )?;
        Ok(samples)
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
        let az = py.import("arviz")?;
        let groups = PyDict::new(py);
        groups.set_item("posterior", self.get_samples(py)?)?;
        let observed = PyDict::new(py);
        observed.set_item("y", PyArray1::from_vec(py, self.observations.clone()))?;
        groups.set_item("observed_data", observed)?;
        arviz_from_groups(&az, groups)
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
    fn conditional_mean_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.conditional_mean_paths)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        local_level_path_array(py, &self.inner.observation_paths)
    }

    #[getter]
    fn conditional_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .conditional_mean_means()
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

    fn conditional_mean_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let quantiles = self
            .inner
            .conditional_mean_quantiles(&[probability])
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

    /// Pointwise equal-tailed interval for the recursive conditional mean.
    #[pyo3(signature = (level=0.95))]
    fn conditional_mean_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalArrays<'py>> {
        validate_interval_level(level)?;
        let probabilities = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
        let quantiles = self
            .inner
            .conditional_mean_quantiles(&probabilities)
            .map_err(bayesian_forecast_error)?;
        Ok((
            quantiles[0].values.clone().into_pyarray(py),
            quantiles[1].values.clone().into_pyarray(py),
        ))
    }

    /// Pointwise equal-tailed posterior-predictive interval for future observations.
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
            "BayesianARForecast(chains={}, draws={}, steps={})",
            self.chains(),
            self.draws(),
            self.steps(),
        )
    }
}

impl PyBayesianAutoRegression {
    pub(crate) fn batch_config(
        &self,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
    ) -> forecast_batch::Config {
        let seed = 0;
        let _ = (warmup, thin);
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
