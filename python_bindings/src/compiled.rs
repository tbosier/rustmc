//! `CompiledModel` and `BoundModel`: structure compiled once, data bound per use.
use crate::batch::{BatchResult, PyBatchFit};
use crate::builder::ModelSpec;
use crate::data_input::{
    core_binding_from_maps, data_inputs_from_maps, merge_data_overrides, parse_data_dict,
    validate_core_binding, Data1d, Data2d,
};
use crate::fit_result::{display_sample_result, FitResult};
use crate::generic_results::{self, StoredBatchFit};
use crate::model_artifact;
use crate::sampling::{parse_metric, parse_sampler_type, validate_sample_config};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::{DataBinding as CoreDataBinding, MatrixBinding};
use rustmc_core::graph::Graph;
use rustmc_core::model::DisplayParamSpec;
use rustmc_core::sampler::{self, SamplerConfig};
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "BoundModel", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyBoundModel {
    pub(crate) structure: Arc<Graph>,
    pub(crate) binding: CoreDataBinding,
}

#[pyclass(name = "CompiledModel", module = "rustmc")]
#[derive(Clone)]
pub(crate) struct PyCompiledModel {
    pub(crate) definition: ModelSpec,
    pub(crate) structure: Arc<Graph>,
    pub(crate) likelihood_names: Vec<String>,
    pub(crate) display_params: Vec<DisplayParamSpec>,
    pub(crate) default_data_1d: Data1d,
    pub(crate) default_data_2d: Data2d,
}

impl PyCompiledModel {
    fn bind_any(&self, value: &Bound<'_, PyAny>, id: String) -> PyResult<CoreDataBinding> {
        if let Ok(bound) = value.downcast::<PyBoundModel>() {
            let bound = bound.borrow();
            if !Arc::ptr_eq(&bound.structure, &self.structure) {
                return Err(PyValueError::new_err(
                    "BoundModel belongs to a different CompiledModel",
                ));
            }
            let mut binding = bound.binding.clone();
            binding.set_id(id);
            return validate_core_binding(&self.structure, binding);
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("data must be a dict or BoundModel from this compiled model")
        })?;
        let mut one_d = self.default_data_1d.clone();
        let mut two_d = self.default_data_2d.clone();
        let (extra_1d, extra_2d) = parse_data_dict(dict)?;
        merge_data_overrides(&mut one_d, &mut two_d, extra_1d, extra_2d);
        validate_core_binding(
            &self.structure,
            core_binding_from_maps(&self.structure.schema, &one_d, &two_d, id, true, true)?,
        )
    }
}

#[pymethods]
impl PyBoundModel {
    #[getter]
    fn id(&self) -> &str {
        self.binding.id()
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.binding.n_obs()
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
}

/// Result for a single model in a batch run.
#[pymethods]
impl PyCompiledModel {
    /// Versioned declarative artifact; excludes all bound training data and defaults.
    fn to_json(&self) -> PyResult<String> {
        model_artifact::encode(self)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        model_artifact::decode(text)
    }

    /// Evaluate the native graph target and gradient in unconstrained coordinates.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        position: Vec<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        if position.len() != self.structure.param_count || position.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "position must be a finite vector matching the parameter dimension",
            ));
        }
        let binding = self.bind_any(data, "0".into())?;
        let mut evaluator = Evaluator::try_with_binding(&self.structure, binding)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        evaluator.compute(&self.structure, &position);
        Ok((evaluator.total_logp, evaluator.grad.into_pyarray(py)))
    }
    #[getter]
    fn dimensions(&self) -> HashMap<String, String> {
        self.structure
            .schema
            .observations
            .iter()
            .chain(&self.structure.schema.vectors)
            .chain(&self.structure.schema.matrices)
            .map(|s| (s.key.clone(), s.dim.clone()))
            .collect()
    }
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.structure.param_names.clone()
    }

    #[getter]
    fn required_keys(&self) -> Vec<String> {
        self.structure
            .schema
            .required_keys()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Stable for this process and useful for verifying Arc structure sharing.
    #[getter]
    fn structure_id(&self) -> usize {
        Arc::as_ptr(&self.structure) as usize
    }

    #[pyo3(signature = (data, id="0", strict=true, check_finite=true))]
    fn bind(
        &self,
        data: &Bound<'_, PyDict>,
        id: &str,
        strict: bool,
        check_finite: bool,
    ) -> PyResult<PyBoundModel> {
        let mut one_d = self.default_data_1d.clone();
        let mut two_d = self.default_data_2d.clone();
        let (extra_1d, extra_2d) = parse_data_dict(data)?;
        merge_data_overrides(&mut one_d, &mut two_d, extra_1d, extra_2d);
        let binding = core_binding_from_maps(
            &self.structure.schema,
            &one_d,
            &two_d,
            id.to_string(),
            strict,
            check_finite,
        )?;
        Ok(PyBoundModel {
            structure: Arc::clone(&self.structure),
            binding: validate_core_binding(&self.structure, binding)?,
        })
    }

    #[pyo3(signature = (data, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, target_accept=0.8, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true, init=None, metric="auto"))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        chains: usize,
        draws: usize,
        warmup: usize,
        seed: u64,
        threads: usize,
        step_size: f64,
        target_accept: f64,
        sampler: &str,
        max_tree_depth: usize,
        num_leapfrog_steps: usize,
        show_progress: bool,
        init: Option<Vec<Vec<f64>>>,
        metric: &str,
    ) -> PyResult<FitResult> {
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
        let binding = self.bind_any(data, "0".to_string())?;
        let sampler_type = parse_sampler_type(sampler)?;
        let config = SamplerConfig {
            sampler: sampler_type,
            num_chains: chains,
            num_draws: draws,
            num_warmup: warmup,
            step_size,
            target_accept,
            num_leapfrog_steps,
            max_tree_depth,
            seed,
            num_threads: threads,
            show_progress,
            metric,
        };
        let hydrated_graph = self.structure.with_binding(&binding);
        let result = py
            .allow_threads(|| {
                sampler::sample_bound_with_init(Arc::clone(&self.structure), binding, config, init)
            })
            .map_err(PyValueError::new_err)?;
        let raw_result = Arc::new(result);
        let display_result = display_sample_result(&raw_result, &self.display_params)?;
        Ok(FitResult {
            definition: self.definition.clone(),
            raw_result,
            display_result,
            graph: hydrated_graph,
            likelihood_names: self.likelihood_names.clone(),
        })
    }

    #[pyo3(signature = (datasets, ids=None, shared=None, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, target_accept=0.8, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true, threads=1, chunk_size=64, errors="raise", seed_policy="cell_id_v1", init=None, metric="auto"))]
    #[allow(clippy::too_many_arguments)]
    fn sample_batch(
        &self,
        py: Python<'_>,
        datasets: Vec<Bound<'_, PyAny>>,
        ids: Option<Vec<String>>,
        shared: Option<&Bound<'_, PyDict>>,
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
        threads: usize,
        chunk_size: usize,
        errors: &str,
        seed_policy: &str,
        init: Option<&Bound<'_, PyDict>>,
        metric: &str,
    ) -> PyResult<PyBatchFit> {
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
        let collect_errors = match errors {
            "raise" => false,
            "collect" => true,
            _ => return Err(PyValueError::new_err("errors must be 'raise' or 'collect'")),
        };
        let seed_policy = match seed_policy {
            "cell_id_v1" => sampler::BatchSeedPolicy::CellIdV1,
            "position_v0" => sampler::BatchSeedPolicy::PositionV0,
            _ => {
                return Err(PyValueError::new_err(
                    "seed_policy must be 'cell_id_v1' or 'position_v0'",
                ))
            }
        };
        let ids = ids.unwrap_or_else(|| (0..datasets.len()).map(|i| i.to_string()).collect());
        if ids.len() != datasets.len() {
            return Err(PyValueError::new_err(
                "ids length must equal datasets length",
            ));
        }
        let mut unique = std::collections::HashSet::new();
        if ids.iter().any(|id| !unique.insert(id)) {
            return Err(PyValueError::new_err("dataset ids must be unique"));
        }
        let mut initial_positions = HashMap::new();
        let mut initial_errors = HashMap::new();
        if let Some(initial) = init {
            for (key, value) in initial.iter() {
                let id: String = key.extract()?;
                if !ids.contains(&id) {
                    return Err(PyValueError::new_err(format!(
                        "initialization supplied for unknown dataset ID '{id}'"
                    )));
                }
                match value.extract::<Vec<Vec<f64>>>() {
                    Ok(positions) => {
                        initial_positions.insert(id, positions);
                    }
                    Err(error) => {
                        initial_errors.insert(id, format!("invalid init: {error}"));
                    }
                }
            }
        }
        // Convert defaults/shared payloads once. Cloning this map only clones
        // Arc handles, so a shared design matrix remains one allocation.
        let mut base_1d = self.default_data_1d.clone();
        let mut base_2d = self.default_data_2d.clone();
        let mut shared_keys = std::collections::HashSet::new();
        if let Some(shared) = shared {
            let (shared_1d, shared_2d) = parse_data_dict(shared)?;
            shared_keys.extend(shared_1d.keys().cloned());
            shared_keys.extend(shared_2d.keys().cloned());
            merge_data_overrides(&mut base_1d, &mut base_2d, shared_1d, shared_2d);
        }
        let base_inputs = data_inputs_from_maps(&base_1d, &base_2d);
        let bindings = datasets
            .iter()
            .zip(&ids)
            .map(|(data, id)| {
                if let Some(error) = initial_errors.get(id) {
                    return Err(PyValueError::new_err(error.clone()));
                }
                if let Ok(bound) = data.downcast::<PyBoundModel>() {
                    let bound = bound.borrow();
                    if !Arc::ptr_eq(&bound.structure, &self.structure) {
                        return Err(PyValueError::new_err(
                            "BoundModel belongs to a different CompiledModel",
                        ));
                    }
                    let mut binding = bound.binding.clone();
                    binding.set_id(id.clone());
                    return validate_core_binding(&self.structure, binding);
                }
                let dict = data.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("datasets must contain dicts or BoundModel objects")
                })?;
                let (extra_1d, extra_2d) = parse_data_dict(dict)?;
                if let Some(key) = extra_1d
                    .keys()
                    .chain(extra_2d.keys())
                    .find(|key| shared_keys.contains(*key))
                {
                    return Err(PyValueError::new_err(format!(
                        "data key '{}' appears in both shared and per-dataset inputs",
                        key
                    )));
                }
                let mut inputs = base_inputs.clone();
                for (key, values) in extra_1d {
                    inputs.matrices.remove(&key);
                    inputs.vectors.insert(key, Arc::from(values));
                }
                for (key, (values, n_rows, n_cols)) in extra_2d {
                    inputs.vectors.remove(&key);
                    inputs.matrices.insert(
                        key,
                        MatrixBinding {
                            data: Arc::from(values),
                            n_rows,
                            n_cols,
                        },
                    );
                }
                let binding =
                    CoreDataBinding::bind(&self.structure.schema, inputs, id.clone(), true, true)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                validate_core_binding(&self.structure, binding)
            })
            .map(|value| value.map_err(|error| error.to_string()))
            .collect::<Vec<_>>();
        let config = sampler::BatchSampleConfig {
            sampler: parse_sampler_type(sampler)?,
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
        let raw = py
            .allow_threads(|| {
                sampler::sample_batch_bound_with_initial(
                    Arc::clone(&self.structure),
                    ids.iter().cloned().zip(bindings.clone()).collect(),
                    config,
                    sampler::BoundBatchOptions {
                        threads,
                        chunk_size,
                        collect_errors,
                        seed_policy,
                    },
                    initial_positions,
                )
            })
            .map_err(PyValueError::new_err)?;
        let mut results = Vec::with_capacity(raw.len());
        for (item, binding) in raw.into_iter().zip(bindings) {
            results.push(match item {
                Err(error) => Err(error),
                Ok(raw_result) => {
                    let raw_result = Arc::new(raw_result);
                    let display_result = display_sample_result(&raw_result, &self.display_params)?;
                    let binding = binding.map_err(PyValueError::new_err)?;
                    Ok(BatchResult {
                        full_fit: StoredBatchFit::Bound(Arc::new(generic_results::BoundBatchFit {
                            structure: Arc::clone(&self.structure),
                            binding,
                            raw_result,
                            display_result,
                            likelihood_names: self.likelihood_names.clone(),
                            definition: self.definition.clone(),
                        })),
                    })
                }
            });
        }
        Ok(PyBatchFit { ids, results })
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
            "CompiledModel(params={}, required_keys={:?})",
            self.structure.param_count,
            self.required_keys()
        )
    }
}
