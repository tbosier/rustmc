//! Hierarchical partial-pooling mean bindings.
use crate::forecast_support::*;
use crate::InferenceError;
use ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::InverseGammaPrior as CoreInverseGammaPrior;
use rustmc_core::hierarchical::{
    fit_hierarchical_mean, HierarchicalMeanConfig as CoreHierarchicalMeanConfig,
    HierarchicalMeanForecast as CoreHierarchicalMeanForecast,
    HierarchicalMeanPosterior as CoreHierarchicalMeanPosterior,
};

/// Values indexed `[chain][draw][row * steps + step]`, shaped
/// `(chain, draw, row, step)`.
fn row_major_path_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    rows: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let (chains, draws) = chain_shape(paths);
    Array4::from_shape_fn((chains, draws, rows, steps), |(chain, draw, row, step)| {
        paths[chain][draw][row * steps + step]
    })
    .into_pyarray(py)
}

/// A per-program summary repeated across forecast steps: the expected level
/// is static, so every step shares it.
fn repeat_over_steps<'py>(
    py: Python<'py>,
    by_program: &[f64],
    steps: usize,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((by_program.len(), steps), |(program, _)| {
        by_program[program]
    })
    .into_pyarray(py)
}

/// A program-major flat summary, shaped `(program, step)`.
fn program_step_array<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    programs: usize,
    steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Array2::from_shape_vec((programs, steps), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|error| InferenceError::new_err(error.to_string()))
}

/// Joint population -> group -> program Gaussian partial-pooling model.
///
/// Ragged program series are fitted in one conjugate Gibbs posterior. This
/// structure-aware sampler draws exact full conditionals and therefore avoids
/// requiring NUTS to traverse a hierarchical funnel.
#[pyclass(name = "BayesianHierarchicalMean", frozen, module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBayesianHierarchicalMean {
    pub(crate) population_mean_prior: f64,
    pub(crate) population_variance_prior: f64,
    pub(crate) group_variance_prior: CoreInverseGammaPrior,
    pub(crate) program_variance_prior: CoreInverseGammaPrior,
    pub(crate) observation_variance_prior: CoreInverseGammaPrior,
}

#[pymethods]
impl PyBayesianHierarchicalMean {
    #[new]
    #[pyo3(signature = (group_variance_prior, program_variance_prior, observation_variance_prior, population_mean_prior=0.0, population_variance_prior=100.0))]
    fn new(
        group_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        program_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        observation_variance_prior: PyRef<'_, PyInverseGammaPrior>,
        population_mean_prior: f64,
        population_variance_prior: f64,
    ) -> PyResult<Self> {
        if !population_mean_prior.is_finite() {
            return Err(InferenceError::new_err(
                "invalid configuration: population mean prior must be finite",
            ));
        }
        if !population_variance_prior.is_finite() || population_variance_prior <= 0.0 {
            return Err(InferenceError::new_err(
                "invalid configuration: population variance prior must be finite and strictly positive",
            ));
        }
        Ok(Self {
            population_mean_prior,
            population_variance_prior,
            group_variance_prior: group_variance_prior.inner,
            program_variance_prior: program_variance_prior.inner,
            observation_variance_prior: observation_variance_prior.inner,
        })
    }

    #[getter]
    fn population_mean_prior(&self) -> f64 {
        self.population_mean_prior
    }

    #[getter]
    fn population_variance_prior(&self) -> f64 {
        self.population_variance_prior
    }

    #[getter]
    fn group_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.group_variance_prior,
        }
    }

    #[getter]
    fn program_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.program_variance_prior,
        }
    }

    #[getter]
    fn observation_variance_prior(&self) -> PyInverseGammaPrior {
        PyInverseGammaPrior {
            inner: self.observation_variance_prior,
        }
    }

    #[pyo3(signature = (series, group_index, program_names=None, group_names=None, chains=4, draws=1000, warmup=500, thin=1, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        series: Vec<PyReadonlyArray1<'_, f64>>,
        group_index: Vec<usize>,
        program_names: Option<Vec<String>>,
        group_names: Option<Vec<String>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        thin: usize,
        seed: u64,
    ) -> PyResult<PyBayesianHierarchicalMeanFit> {
        let series = series
            .into_iter()
            .map(state_space_vector)
            .collect::<Vec<_>>();
        let program_count = series.len();
        if group_index.len() != program_count {
            return Err(InferenceError::new_err(
                "series and group_index must have the same length",
            ));
        }
        let mut present_groups = vec![false; program_count];
        for &group in &group_index {
            if group >= program_count {
                return Err(InferenceError::new_err(
                    "group indices must be contiguous from zero with no empty groups",
                ));
            }
            present_groups[group] = true;
        }
        let inferred_group_count = group_index
            .iter()
            .copied()
            .max()
            .map_or(0, |value| value + 1);
        if present_groups[..inferred_group_count].contains(&false) {
            return Err(InferenceError::new_err(
                "group indices must be contiguous from zero with no empty groups",
            ));
        }
        let program_names = program_names.unwrap_or_else(|| {
            (0..program_count)
                .map(|index| format!("program_{index}"))
                .collect()
        });
        let group_names = group_names.unwrap_or_else(|| {
            (0..inferred_group_count)
                .map(|index| format!("group_{index}"))
                .collect()
        });
        validate_unique_names(&program_names, program_count, "program_names")?;
        validate_unique_names(&group_names, inferred_group_count, "group_names")?;
        let time_counts = series.iter().map(Vec::len).collect::<Vec<_>>();
        let config = CoreHierarchicalMeanConfig {
            population_mean_prior: self.population_mean_prior,
            population_variance_prior: self.population_variance_prior,
            group_variance_prior: self.group_variance_prior,
            program_variance_prior: self.program_variance_prior,
            observation_variance_prior: self.observation_variance_prior,
            num_chains: chains,
            num_warmup: warmup,
            num_draws: draws,
            thinning: thin,
            seed,
        };
        let posterior = py
            .allow_threads(|| fit_hierarchical_mean(&series, &group_index, &config))
            .map_err(hierarchical_error)?;
        Ok(PyBayesianHierarchicalMeanFit {
            posterior,
            time_counts,
            program_names,
            group_names,
            config,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianHierarchicalMean(population_mean_prior={}, population_variance_prior={}, group_variance_prior=({}, {}), program_variance_prior=({}, {}), observation_variance_prior=({}, {}))",
            self.population_mean_prior,
            self.population_variance_prior,
            self.group_variance_prior.shape,
            self.group_variance_prior.scale,
            self.program_variance_prior.shape,
            self.program_variance_prior.scale,
            self.observation_variance_prior.shape,
            self.observation_variance_prior.scale,
        )
    }
}

pub(crate) fn validate_unique_names(
    names: &[String],
    expected: usize,
    field: &str,
) -> PyResult<()> {
    if names.len() != expected {
        return Err(InferenceError::new_err(format!(
            "{field} must contain exactly {expected} entries"
        )));
    }
    let unique = names.iter().collect::<std::collections::HashSet<_>>();
    if unique.len() != names.len() {
        return Err(InferenceError::new_err(format!(
            "{field} entries must be unique"
        )));
    }
    Ok(())
}

#[pyclass(name = "BayesianHierarchicalMeanFit", module = "rustmc")]
pub(crate) struct PyBayesianHierarchicalMeanFit {
    pub(crate) posterior: CoreHierarchicalMeanPosterior,
    pub(crate) time_counts: Vec<usize>,
    pub(crate) program_names: Vec<String>,
    pub(crate) group_names: Vec<String>,
    pub(crate) config: CoreHierarchicalMeanConfig,
}

impl ForecastFit for PyBayesianHierarchicalMeanFit {
    fn sampler(&self) -> &'static str {
        "conjugate Gibbs"
    }
    fn summary_line(&self) -> String {
        "Sampler: conjugate Gibbs (acceptance and divergences unavailable)".into()
    }
    fn coverage(&self) -> &'static str {
        "all retained hierarchical parameters"
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
                "population_mean",
                draw_array(py, chains, |draw| draw.population_mean),
            ),
            (
                "group_variance",
                draw_array(py, chains, |draw| draw.group_variance),
            ),
            (
                "program_variance",
                draw_array(py, chains, |draw| draw.program_variance),
            ),
            (
                "observation_variance",
                draw_array(py, chains, |draw| draw.observation_variance),
            ),
            (
                "group_sd",
                draw_array(py, chains, |draw| draw.group_variance.sqrt()),
            ),
            (
                "program_sd",
                draw_array(py, chains, |draw| draw.program_variance.sqrt()),
            ),
            (
                "observation_sd",
                draw_array(py, chains, |draw| draw.observation_variance.sqrt()),
            ),
        ] {
            samples.set_item(name, values)?;
        }
        samples.set_item(
            "group_mean",
            draw_vector_array(py, chains, self.posterior.group_count, |draw, index| {
                draw.group_means[index]
            }),
        )?;
        samples.set_item(
            "program_mean",
            draw_vector_array(py, chains, self.posterior.program_count(), |draw, index| {
                draw.program_means[index]
            }),
        )?;
        Ok(samples)
    }
}

#[pymethods]
impl PyBayesianHierarchicalMeanFit {
    #[getter]
    fn chains(&self) -> usize {
        self.shape().0
    }

    #[getter]
    fn draws(&self) -> usize {
        self.shape().1
    }

    #[getter]
    fn program_count(&self) -> usize {
        self.posterior.program_count()
    }

    #[getter]
    fn group_count(&self) -> usize {
        self.posterior.group_count
    }

    #[getter]
    fn time_counts(&self) -> Vec<usize> {
        self.time_counts.clone()
    }

    #[getter]
    fn observed_counts(&self) -> Vec<usize> {
        self.posterior.observed_counts.clone()
    }

    #[getter]
    fn total_observed_count(&self) -> usize {
        self.posterior.observed_counts.iter().sum()
    }

    #[getter]
    fn group_index(&self) -> Vec<usize> {
        self.posterior.group_index.clone()
    }

    #[getter]
    fn program_names(&self) -> Vec<String> {
        self.program_names.clone()
    }

    #[getter]
    fn group_names(&self) -> Vec<String> {
        self.group_names.clone()
    }

    #[getter]
    fn warmup(&self) -> usize {
        self.config.num_warmup
    }

    #[getter]
    fn thin(&self) -> usize {
        self.config.thinning
    }

    #[getter]
    fn inference_method(&self) -> &'static str {
        "conjugate_gibbs"
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.posterior(py)
    }

    /// Formatted rank-normalized R-hat, ESS, MCSE, and HDI diagnostics.
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

    #[pyo3(signature = (steps, seed=43))]
    fn forecast(
        &self,
        py: Python<'_>,
        steps: usize,
        seed: u64,
    ) -> PyResult<PyBayesianHierarchicalForecast> {
        let inner = py
            .allow_threads(|| self.posterior.forecast(steps, seed))
            .map_err(hierarchical_error)?;
        Ok(PyBayesianHierarchicalForecast {
            inner,
            program_names: self.program_names.clone(),
            group_names: self.group_names.clone(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesianHierarchicalMeanFit(chains={}, draws={}, programs={}, groups={}, observations={})",
            self.chains(), self.draws(), self.program_count(), self.group_count(), self.total_observed_count()
        )
    }
}

impl PyBayesianHierarchicalForecast {
    fn observation_quantiles<const N: usize>(
        &self,
        probabilities: [f64; N],
    ) -> PyResult<[Vec<f64>; N]> {
        quantile_values(probabilities, |p| self.inner.observation_quantiles(p))
    }

    fn state_quantiles<const N: usize>(&self, probabilities: [f64; N]) -> PyResult<[Vec<f64>; N]> {
        quantile_values(probabilities, |p| self.inner.state_quantiles(p))
    }
}

#[pyclass(name = "BayesianHierarchicalForecast", module = "rustmc")]
pub(crate) struct PyBayesianHierarchicalForecast {
    pub(crate) inner: CoreHierarchicalMeanForecast,
    pub(crate) program_names: Vec<String>,
    pub(crate) group_names: Vec<String>,
}

#[pymethods]
impl PyBayesianHierarchicalForecast {
    #[getter]
    fn chains(&self) -> usize {
        self.inner.chain_count()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draw_count()
    }

    #[getter]
    fn program_count(&self) -> usize {
        self.inner.program_count()
    }

    #[getter]
    fn group_count(&self) -> usize {
        self.inner.group_count
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    fn group_index(&self) -> Vec<usize> {
        self.inner.group_index.clone()
    }

    #[getter]
    fn program_names(&self) -> Vec<String> {
        self.program_names.clone()
    }

    #[getter]
    fn group_names(&self) -> Vec<String> {
        self.group_names.clone()
    }

    /// The static expected level of each program repeated over steps,
    /// shaped `(chain, draw, program, step)`.
    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        let (chains, draws) = chain_shape(&self.inner.state_means);
        let states = &self.inner.state_means;
        Array4::from_shape_fn(
            (chains, draws, self.program_count(), self.steps()),
            |(chain, draw, program, _)| states[chain][draw][program],
        )
        .into_pyarray(py)
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        row_major_path_array(
            py,
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
        )
    }

    /// Draw-wise group totals indexed `(chain, draw, group, step)`.
    #[getter]
    fn group_observation_samples<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let paths = self
            .inner
            .group_observation_paths()
            .map_err(hierarchical_error)?;
        Ok(row_major_path_array(
            py,
            &paths,
            self.group_count(),
            self.steps(),
        ))
    }

    /// Draw-wise total across all programs, shaped `(chain, draw, step)`.
    #[getter]
    fn total_observation_samples<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let paths = self
            .inner
            .total_observation_paths()
            .map_err(hierarchical_error)?;
        Ok(path_array(py, &paths))
    }

    #[getter]
    fn state_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let means = self
            .inner
            .state_means_by_program()
            .map_err(hierarchical_error)?;
        Ok(repeat_over_steps(py, &means, self.steps()))
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let means = self.inner.observation_means().map_err(hierarchical_error)?;
        program_step_array(py, means, self.program_count(), self.steps())
    }

    fn state_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let [values] = self.state_quantiles([probability])?;
        Ok(repeat_over_steps(py, &values, self.steps()))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let [values] = self.observation_quantiles([probability])?;
        program_step_array(py, values, self.program_count(), self.steps())
    }

    /// Pointwise equal-tailed posterior-predictive interval, each bound shaped
    /// `(program, step)`.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalMatrices<'py>> {
        let [lower, upper] = self.observation_quantiles(interval_probabilities(level)?)?;
        Ok((
            program_step_array(py, lower, self.program_count(), self.steps())?,
            program_step_array(py, upper, self.program_count(), self.steps())?,
        ))
    }

    /// Pointwise equal-tailed interval for each program's expected level,
    /// repeated over steps.
    #[pyo3(signature = (level=0.95))]
    fn state_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalMatrices<'py>> {
        let [lower, upper] = self.state_quantiles(interval_probabilities(level)?)?;
        Ok((
            repeat_over_steps(py, &lower, self.steps()),
            repeat_over_steps(py, &upper, self.steps()),
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
            "BayesianHierarchicalForecast(chains={}, draws={}, programs={}, groups={}, steps={})",
            self.chains(),
            self.draws(),
            self.program_count(),
            self.group_count(),
            self.steps()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBayesianHierarchicalMean>()?;
    m.add_class::<PyBayesianHierarchicalMeanFit>()?;
    m.add_class::<PyBayesianHierarchicalForecast>()?;
    Ok(())
}
