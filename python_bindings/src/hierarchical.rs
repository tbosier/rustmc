//! Hierarchical partial-pooling mean bindings.
use crate::forecast_diagnostics;
use crate::forecast_support::*;
use crate::InferenceError;
use ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::bayesian_forecast::InverseGammaPrior as CoreInverseGammaPrior;
use rustmc_core::hierarchical::{
    fit_hierarchical_mean, HierarchicalMeanConfig as CoreHierarchicalMeanConfig,
    HierarchicalMeanForecast as CoreHierarchicalMeanForecast,
    HierarchicalMeanPosterior as CoreHierarchicalMeanPosterior,
    HierarchicalMeanPosteriorDraw as CoreHierarchicalMeanPosteriorDraw,
};

pub(crate) fn hierarchical_scalar_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreHierarchicalMeanPosterior,
    value: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&CoreHierarchicalMeanPosteriorDraw) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array2::from_shape_fn((chains, draws), |(chain, draw)| {
        value(&posterior.chains[chain][draw])
    })
    .into_pyarray(py)
}

pub(crate) fn hierarchical_vector_array<'py, F>(
    py: Python<'py>,
    posterior: &CoreHierarchicalMeanPosterior,
    width: usize,
    value: F,
) -> Bound<'py, PyArray3<f64>>
where
    F: Fn(&CoreHierarchicalMeanPosteriorDraw, usize) -> f64,
{
    let chains = posterior.chains.len();
    let draws = posterior.chains.first().map_or(0, Vec::len);
    Array3::from_shape_fn((chains, draws, width), |(chain, draw, index)| {
        value(&posterior.chains[chain][draw], index)
    })
    .into_pyarray(py)
}

pub(crate) fn hierarchical_path_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    Array4::from_shape_fn(
        (chains, draws, programs, steps),
        |(chain, draw, program, step)| paths[chain][draw][program * steps + step],
    )
    .into_pyarray(py)
}

pub(crate) fn hierarchical_path_summary(
    paths: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
    probability: Option<f64>,
) -> Array2<f64> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    Array2::from_shape_fn((programs, steps), |(program, step)| {
        let flat_index = program * steps + step;
        if let Some(probability) = probability {
            let mut values = Vec::with_capacity(chains * draws);
            for chain in paths {
                for draw in chain {
                    values.push(draw[flat_index]);
                }
            }
            values.sort_by(f64::total_cmp);
            let index = probability * (values.len() - 1) as f64;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            let weight = index - lower as f64;
            values[lower] * (1.0 - weight) + values[upper] * weight
        } else {
            paths
                .iter()
                .flat_map(|chain| chain.iter())
                .map(|draw| draw[flat_index])
                .sum::<f64>()
                / (chains * draws) as f64
        }
    })
}

pub(crate) fn hierarchical_state_array<'py>(
    py: Python<'py>,
    states: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let chains = states.len();
    let draws = states.first().map_or(0, Vec::len);
    Array4::from_shape_fn(
        (chains, draws, programs, steps),
        |(chain, draw, program, _step)| states[chain][draw][program],
    )
    .into_pyarray(py)
}

pub(crate) fn hierarchical_state_summary(
    states: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
    probability: Option<f64>,
) -> Array2<f64> {
    let chains = states.len();
    let draws = states.first().map_or(0, Vec::len);
    let by_program = (0..programs)
        .map(|program| {
            if let Some(probability) = probability {
                let mut values = states
                    .iter()
                    .flat_map(|chain| chain.iter())
                    .map(|draw| draw[program])
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let index = probability * (values.len() - 1) as f64;
                let lower = index.floor() as usize;
                let upper = index.ceil() as usize;
                let weight = index - lower as f64;
                values[lower] * (1.0 - weight) + values[upper] * weight
            } else {
                states
                    .iter()
                    .flat_map(|chain| chain.iter())
                    .map(|draw| draw[program])
                    .sum::<f64>()
                    / (chains * draws) as f64
            }
        })
        .collect::<Vec<_>>();
    Array2::from_shape_fn((programs, steps), |(program, _step)| by_program[program])
}

pub(crate) fn hierarchical_group_rollup_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    group_index: &[usize],
    group_count: usize,
    steps: usize,
) -> Bound<'py, PyArray4<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    let mut rollups = Array4::zeros((chains, draws, group_count, steps));
    for chain in 0..chains {
        for draw in 0..draws {
            for (program, &group) in group_index.iter().enumerate() {
                for step in 0..steps {
                    rollups[(chain, draw, group, step)] +=
                        paths[chain][draw][program * steps + step];
                }
            }
        }
    }
    rollups.into_pyarray(py)
}

pub(crate) fn hierarchical_total_rollup_array<'py>(
    py: Python<'py>,
    paths: &[Vec<Vec<f64>>],
    programs: usize,
    steps: usize,
) -> Bound<'py, PyArray3<f64>> {
    let chains = paths.len();
    let draws = paths.first().map_or(0, Vec::len);
    Array3::from_shape_fn((chains, draws, steps), |(chain, draw, step)| {
        (0..programs)
            .map(|program| paths[chain][draw][program * steps + step])
            .sum()
    })
    .into_pyarray(py)
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

#[pymethods]
impl PyBayesianHierarchicalMeanFit {
    #[getter]
    fn chains(&self) -> usize {
        self.posterior.chains.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.posterior.chains.first().map_or(0, Vec::len)
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
        let samples = PyDict::new(py);
        samples.set_item(
            "population_mean",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.population_mean),
        )?;
        samples.set_item(
            "group_variance",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.group_variance),
        )?;
        samples.set_item(
            "program_variance",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.program_variance),
        )?;
        samples.set_item(
            "observation_variance",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.observation_variance),
        )?;
        samples.set_item(
            "group_sd",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.group_variance.sqrt()),
        )?;
        samples.set_item(
            "program_sd",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.program_variance.sqrt()),
        )?;
        samples.set_item(
            "observation_sd",
            hierarchical_scalar_array(py, &self.posterior, |draw| draw.observation_variance.sqrt()),
        )?;
        samples.set_item(
            "group_mean",
            hierarchical_vector_array(py, &self.posterior, self.group_count(), |draw, index| {
                draw.group_means[index]
            }),
        )?;
        samples.set_item(
            "program_mean",
            hierarchical_vector_array(py, &self.posterior, self.program_count(), |draw, index| {
                draw.program_means[index]
            }),
        )?;
        Ok(samples)
    }

    /// Formatted rank-normalized R-hat, ESS, MCSE, and HDI diagnostics.
    fn summary(&self) -> String {
        self.posterior.diagnostics().to_table_with_sampler(Some(
            "Sampler: conjugate Gibbs (acceptance and divergences unavailable)",
        ))
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        forecast_diagnostics::diagnostics_list(py, &self.posterior.diagnostics())
    }

    #[getter]
    fn sampler_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        forecast_diagnostics::sampler_stats(
            py,
            "conjugate Gibbs",
            self.chains(),
            self.draws(),
            "all retained hierarchical parameters",
        )
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

    #[getter]
    fn state_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        hierarchical_state_array(
            py,
            &self.inner.state_means,
            self.program_count(),
            self.steps(),
        )
    }

    #[getter]
    fn observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        hierarchical_path_array(
            py,
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
        )
    }

    /// Draw-wise group totals indexed `(chain, draw, group, step)`.
    #[getter]
    fn group_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        hierarchical_group_rollup_array(
            py,
            &self.inner.observation_paths,
            &self.inner.group_index,
            self.inner.group_count,
            self.steps(),
        )
    }

    /// Draw-wise total across all programs, shaped `(chain, draw, step)`.
    #[getter]
    fn total_observation_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        hierarchical_total_rollup_array(
            py,
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
        )
    }

    #[getter]
    fn state_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        hierarchical_state_summary(
            &self.inner.state_means,
            self.program_count(),
            self.steps(),
            None,
        )
        .into_pyarray(py)
    }

    #[getter]
    fn observation_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        hierarchical_path_summary(
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
            None,
        )
        .into_pyarray(py)
    }

    fn state_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_probability(probability)?;
        Ok(hierarchical_state_summary(
            &self.inner.state_means,
            self.program_count(),
            self.steps(),
            Some(probability),
        )
        .into_pyarray(py))
    }

    fn observation_quantile<'py>(
        &self,
        py: Python<'py>,
        probability: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_probability(probability)?;
        Ok(hierarchical_path_summary(
            &self.inner.observation_paths,
            self.program_count(),
            self.steps(),
            Some(probability),
        )
        .into_pyarray(py))
    }

    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalMatrices<'py>> {
        validate_interval_level(level)?;
        let tail = (1.0 - level) / 2.0;
        Ok((
            hierarchical_path_summary(
                &self.inner.observation_paths,
                self.program_count(),
                self.steps(),
                Some(tail),
            )
            .into_pyarray(py),
            hierarchical_path_summary(
                &self.inner.observation_paths,
                self.program_count(),
                self.steps(),
                Some(1.0 - tail),
            )
            .into_pyarray(py),
        ))
    }

    fn state_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<PyIntervalMatrices<'py>> {
        validate_interval_level(level)?;
        let tail = (1.0 - level) / 2.0;
        Ok((
            hierarchical_state_summary(
                &self.inner.state_means,
                self.program_count(),
                self.steps(),
                Some(tail),
            )
            .into_pyarray(py),
            hierarchical_state_summary(
                &self.inner.state_means,
                self.program_count(),
                self.steps(),
                Some(1.0 - tail),
            )
            .into_pyarray(py),
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
