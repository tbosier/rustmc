//! Batch results: `BatchFit`, its `BatchResult` cells, and `batch_sample`.
use crate::builder::{
    compile_python_model, reject_discrete_priors_for_gradient_sampling, ModelSpec,
};
use crate::data_input::{merge_data_overrides, parse_data_dict, validate_matrix_storage};
use crate::fit_result::{display_sample_result, FitResult};
use crate::generic_results::{self, StoredBatchFit};
use crate::sampling::{parse_metric, validate_sample_config};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{self, SampleResult, SamplerType};
use std::collections::HashMap;
use std::sync::Arc;

/// One cell of a batch run.
///
/// Everything this exposes is read off the retained fit, which every cell
/// has: both construction sites supply one, so the accessors are infallible.
/// The fit used to be optional, and the `None` arm manufactured an error for
/// a "legacy" cell that no code path could produce. It used to also hold a
/// flattened `BatchModelResult` copy of the display draws, which made a third
/// posterior per cell alongside the raw and display trees.
#[pyclass(module = "rustmc")]
#[derive(Clone)]
pub(crate) struct BatchResult {
    pub(crate) full_fit: StoredBatchFit,
}

impl BatchResult {
    /// Display draws for this cell.
    fn display(&self) -> &SampleResult {
        self.full_fit.display()
    }
}
#[pymethods]
impl BatchResult {
    /// Internal regression-test hook: compare immutable payload ownership without exposing addresses.
    fn _shares_data(&self, other: &BatchResult, key: &str) -> bool {
        match (&self.full_fit, &other.full_fit) {
            (StoredBatchFit::Bound(a), StoredBatchFit::Bound(b)) => {
                a.binding.shares_payload_with(&b.binding, key)
            }
            _ => false,
        }
    }
    /// Internal regression-test hook: how many distinct posterior sample trees
    /// this cell retains. One when the display layer passes the raw draws
    /// through unchanged, two when a parameter is genuinely derived.
    fn _posterior_allocations(&self) -> usize {
        if std::ptr::eq(self.full_fit.raw(), self.full_fit.display()) {
            1
        } else {
            2
        }
    }

    /// Internal regression-test hook: whether `fit` reuses this cell's retained
    /// posterior rather than holding a copy of it. Compares ownership without
    /// exposing addresses.
    fn _shares_posterior_with(&self, fit: &FitResult) -> bool {
        std::ptr::eq(self.full_fit.raw(), &*fit.raw_result)
            && std::ptr::eq(self.full_fit.display(), &*fit.display_result)
    }

    #[getter]
    fn fit(&self) -> FitResult {
        self.full_fit.materialize()
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        generic_results::diagnostics(self.full_fit.display(), py)
    }

    fn summary(&self) -> String {
        self.full_fit.display().diagnostics().to_table()
    }

    fn transition_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::transition_diagnostics(self.full_fit.raw(), py)
    }

    #[pyo3(signature = (data=None, seed=42, expected=false, sizes=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        seed: u64,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.fit().predict(py, data, seed, expected, sizes)
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let dict = PyDict::new(py);
        let n_chains = display.samples.len();
        let n_draws = display.samples.first().map_or(0, Vec::len);
        for (pidx, name) in display.param_names.iter().enumerate() {
            let mut arr = Array2::<f64>::zeros((n_chains, n_draws));
            for (chain_idx, chain) in display.samples.iter().enumerate() {
                for (draw_idx, draw) in chain.iter().enumerate() {
                    arr[[chain_idx, draw_idx]] = draw[pidx];
                }
            }
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let means = display.mean();
        let dict = PyDict::new(py);
        for (name, val) in display.param_names.iter().zip(means.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let stds = display.std();
        let dict = PyDict::new(py);
        for (name, val) in display.param_names.iter().zip(stds.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let dict = PyDict::new(py);
        for (pidx, name) in display.param_names.iter().enumerate() {
            let vals: Vec<f64> = display
                .samples
                .iter()
                .flatten()
                .map(|draw| draw[pidx])
                .collect();
            let arr = PyArray1::from_vec(py, vals);
            dict.set_item(name, arr)?;
        }
        Ok(dict)
    }

    #[getter]
    fn chains(&self) -> usize {
        self.display().samples.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.display().samples.first().map_or(0, Vec::len)
    }

    #[getter]
    fn accept_rate(&self) -> f64 {
        let rates = &self.display().accept_rates;
        if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        }
    }

    #[getter]
    fn accept_rates(&self) -> Vec<f64> {
        self.display().accept_rates.clone()
    }

    #[getter]
    fn divergences(&self) -> usize {
        self.display().total_divergences()
    }

    #[getter]
    fn divergences_per_chain(&self) -> Vec<usize> {
        self.display().divergences.clone()
    }

    fn __repr__(&self) -> String {
        let display = self.display();
        let means = display.mean();
        let parts: Vec<String> = display
            .param_names
            .iter()
            .zip(means.iter())
            .map(|(n, m)| format!("{}={:.4}", n, m))
            .collect();
        format!(
            "BatchResult({} chains × {} draws, {})",
            display.samples.len(),
            display.samples.first().map_or(0, Vec::len),
            parts.join(", ")
        )
    }
}

#[pyclass(name = "BatchFit", module = "rustmc")]
pub(crate) struct PyBatchFit {
    pub(crate) ids: Vec<String>,
    pub(crate) results: Vec<Result<BatchResult, String>>,
}

#[pymethods]
impl PyBatchFit {
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }

    #[getter]
    fn errors(&self) -> HashMap<String, String> {
        self.ids
            .iter()
            .zip(&self.results)
            .filter_map(|(id, value)| {
                value
                    .as_ref()
                    .err()
                    .map(|error| (id.clone(), error.clone()))
            })
            .collect()
    }

    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Py<BatchResult>> {
        let index = self
            .ids
            .iter()
            .position(|value| value == id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown dataset ID '{id}'")))?;
        self.__getitem__(py, index as isize)
    }

    fn __len__(&self) -> usize {
        self.results.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<BatchResult>> {
        let len = self.results.len() as isize;
        let normalized = if index < 0 { len + index } else { index };
        if normalized < 0 || normalized >= len {
            return Err(PyIndexError::new_err("batch index out of range"));
        }
        let result = self
            .results
            .get(normalized as usize)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("batch index out of range"))?;
        let result = result.map_err(|error| {
            PyValueError::new_err(format!(
                "dataset '{}': {error}",
                self.ids[normalized as usize]
            ))
        })?;
        Py::new(py, result)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchFit({} datasets, {} failed)",
            self.results.len(),
            self.errors().len()
        )
    }
}

/// Run thousands of independent models in parallel through Rayon.
///
/// Each entry in `models` is a (ModelSpec, data_dict) pair. By default each gets
/// 1 NUTS chain for throughput, but the batch runner can be configured to use
/// multiple chains or fixed-step HMC when reliability matters more.
#[pyfunction]
#[pyo3(signature = (models, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, target_accept=0.8, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true, metric="auto"))]
// The Python API intentionally exposes each sampler option as a named argument.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_sample(
    py: Python<'_>,
    models: Vec<(Bound<'_, ModelSpec>, Bound<'_, PyDict>)>,
    chains: usize,
    draws: usize,
    warmup: usize,
    seed: u64,
    sampler: &str,
    step_size: f64,
    target_accept: f64,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
    show_progress: bool,
    metric: &str,
) -> PyResult<Vec<BatchResult>> {
    let metric = parse_metric(metric)?;
    validate_sample_config(
        chains,
        draws,
        warmup,
        step_size,
        target_accept,
        max_tree_depth,
        num_leapfrog_steps,
    )?;

    let mut compiled_models = Vec::with_capacity(models.len());

    for (spec_bound, data_bound) in &models {
        let spec = spec_bound.borrow();
        reject_discrete_priors_for_gradient_sampling(&spec.priors)?;

        // Bound data from ModelSpec is the base; call-site dict overrides/extends.
        let mut data_map: HashMap<String, Vec<f64>> = spec.bound_data_1d.clone();
        let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> = spec.bound_data_2d.clone();
        let (extra_1d, extra_2d) = parse_data_dict(data_bound)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, extra_1d, extra_2d);

        validate_matrix_storage(&matrix_map)?;

        compiled_models.push(compile_python_model(&spec, &data_map, &matrix_map)?);
    }

    let sampler = match sampler {
        "nuts" | "NUTS" => SamplerType::Nuts,
        "hmc" | "HMC" => SamplerType::Hmc,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
                sampler
            )))
        }
    };

    let config = sampler::BatchSampleConfig {
        sampler,
        num_chains: chains,
        num_draws: draws,
        num_warmup: warmup,
        step_size,
        target_accept,
        num_leapfrog_steps,
        max_tree_depth,
        seed,
        show_progress,
        metric,
    };

    let graphs: Vec<Graph> = compiled_models
        .iter()
        .map(|compiled| compiled.graph.clone())
        .collect();

    let results = py
        .allow_threads(|| sampler::batch_sample_graphs(graphs, config))
        .map_err(PyValueError::new_err)?;

    results
        .into_iter()
        .zip(compiled_models.iter())
        .zip(models.iter())
        .map(|((raw_result, compiled), (spec, _))| {
            let num_draws = raw_result.num_draws;
            let raw = Arc::new(SampleResult {
                samples: regroup_draws_by_chain(raw_result.samples, num_draws),
                unconstrained_samples: raw_result.unconstrained_samples,
                param_names: raw_result.param_names,
                accept_rates: raw_result.accept_rates,
                step_sizes: raw_result.step_sizes,
                divergences: raw_result.divergences,
                transitions: raw_result.transitions,
            });
            let display_result = display_sample_result(&raw, &compiled.display_params)?;
            Ok(BatchResult {
                full_fit: StoredBatchFit::Ready(Arc::new(FitResult {
                    raw_result: raw,
                    display_result,
                    graph: compiled.graph.clone(),
                    likelihood_names: compiled.likelihood_names.clone(),
                    definition: spec.borrow().structure_definition(),
                })),
            })
        })
        .collect()
}

/// Regroup a flat, chain-major draw list into per-chain blocks.
///
/// The draw buffers are moved rather than copied, so regrouping a batch result
/// does not duplicate the posterior. Draining the source is what keeps that
/// true: `split_off` would leave each chain holding the capacity of the whole
/// remaining suffix, which costs O(chains² × draws) descriptor slots.
pub(crate) fn regroup_draws_by_chain(flat: Vec<Vec<f64>>, num_draws: usize) -> Vec<Vec<Vec<f64>>> {
    if num_draws == 0 {
        return Vec::new();
    }
    let mut remaining = flat.into_iter();
    let mut chains = Vec::with_capacity(remaining.len().div_ceil(num_draws));
    loop {
        let chain: Vec<Vec<f64>> = remaining.by_ref().take(num_draws).collect();
        if chain.is_empty() {
            return chains;
        }
        chains.push(chain);
    }
}
