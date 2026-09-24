//! Module-level `sample` and `sample_prior_predictive`, and sampler options.
use crate::builder::{
    compile_python_model, reject_discrete_priors_for_gradient_sampling, ModelSpec,
};
use crate::data_input::{merge_data_overrides, parse_data_dict, validate_matrix_storage};
use crate::fit_result::{display_sample_result, FitResult};
use crate::model_error;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::data::DataBinding as CoreDataBinding;
use rustmc_core::model::DisplayParamSpec;
use rustmc_core::sampler::{self, SamplerConfig, SamplerType};
use rustmc_core::seeding::{stream_seed, PRIOR_PREDICT_SEED_DOMAIN};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) fn validate_sample_config(
    chains: usize,
    draws: usize,
    warmup: usize,
    step_size: f64,
    target_accept: f64,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
) -> PyResult<()> {
    if chains == 0 {
        return Err(PyValueError::new_err("chains must be >= 1"));
    }
    if draws == 0 {
        return Err(PyValueError::new_err("draws must be >= 1"));
    }
    if warmup == 0 {
        return Err(PyValueError::new_err("warmup must be >= 1"));
    }
    if !step_size.is_finite() || step_size < 0.0 {
        return Err(PyValueError::new_err(
            "step_size must be finite and >= 0 (0 enables adaptation)",
        ));
    }
    if !target_accept.is_finite() || target_accept <= 0.0 || target_accept >= 1.0 {
        return Err(PyValueError::new_err(
            "target_accept must be finite and strictly between 0 and 1",
        ));
    }
    if !(1..=63).contains(&max_tree_depth) {
        return Err(PyValueError::new_err(
            "max_tree_depth must be between 1 and 63",
        ));
    }
    if num_leapfrog_steps == 0 {
        return Err(PyValueError::new_err("num_leapfrog_steps must be >= 1"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (model_spec, data=None, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, target_accept=0.8, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true, init=None, metric="auto"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample(
    py: Python<'_>,
    model_spec: &ModelSpec,
    data: Option<&Bound<'_, PyDict>>,
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
    reject_discrete_priors_for_gradient_sampling(&model_spec.priors)?;
    // Start from data bound at build time, then let call-site data override/extend.
    let mut data_map: HashMap<String, Vec<f64>> = model_spec.bound_data_1d.clone();
    let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> =
        model_spec.bound_data_2d.clone();

    if let Some(data_dict) = data {
        let (extra_1d, extra_2d) = parse_data_dict(data_dict)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, extra_1d, extra_2d);
    }

    validate_matrix_storage(&matrix_map)?;

    if data_map.is_empty() && matrix_map.is_empty() && !model_spec.likelihoods.is_empty() {
        return Err(PyValueError::new_err(
            "No data provided. Pass data= to sample() or bind it via ModelBuilder(data=...).",
        ));
    }

    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;

    let sampler_type = match sampler {
        "nuts" | "NUTS" => SamplerType::Nuts,
        "hmc" | "HMC" => SamplerType::Hmc,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
                sampler
            )))
        }
    };

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

    let graph_for_predict = compiled.graph.clone();

    let result = py
        .allow_threads(|| {
            sampler::sample_bound_with_init(
                Arc::new(compiled.graph.structure_only()),
                CoreDataBinding::from_graph(&compiled.graph).map_err(|e| e.to_string())?,
                config,
                init,
            )
        })
        .map_err(PyValueError::new_err)?;
    let raw_result = Arc::new(result);
    let display_result = display_sample_result(&raw_result, &compiled.display_params)?;

    Ok(FitResult {
        definition: model_spec.structure_definition(),
        raw_result,
        display_result,
        graph: graph_for_predict,
        likelihood_names: compiled.likelihood_names,
    })
}

pub(crate) fn parse_metric(metric: &str) -> PyResult<sampler::MetricKind> {
    sampler::MetricKind::parse(metric).map_err(PyValueError::new_err)
}

pub(crate) fn parse_sampler_type(sampler: &str) -> PyResult<SamplerType> {
    match sampler {
        "nuts" | "NUTS" => Ok(SamplerType::Nuts),
        "hmc" | "HMC" => Ok(SamplerType::Hmc),
        _ => Err(PyValueError::new_err(format!(
            "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
            sampler
        ))),
    }
}

/// Draw samples from the **prior predictive** distribution.
///
/// Samples parameters from the model priors using their analytic distributions,
/// then runs a forward pass to generate predicted observations.
/// Use this to check whether your priors make sense before fitting.
///
/// A likelihood is not required.  With none declared, the result carries the
/// prior draws of the parameters and of any deterministic, and no predicted
/// observations -- which is exactly what "check whether your priors make sense
/// before fitting" means for a model whose likelihood is not written yet.
/// Potentials *are* refused: a custom density term supplies no random
/// generator, so a model carrying one has no prior to simulate from.
///
/// Parameters
/// ----------
/// model_spec : ModelSpec
///     A model definition, from `builder.build()`.
/// data : dict or None
///     Data dict (same as `sample()`).  Needed for the predictor covariates (x values).
/// n_samples : int
///     Number of prior predictive draws.
/// seed : int
///     RNG seed.
///
/// Returns
/// -------
/// dict
///     ``"<param_name>"`` → 1-D array of n_samples prior samples.
///     ``"<likelihood_name>"`` → 2-D array (n_samples, n_obs) of predicted y.
#[pyfunction]
#[pyo3(signature = (model_spec, data=None, n_samples=500, seed=42))]
pub(crate) fn sample_prior_predictive<'py>(
    py: Python<'py>,
    model_spec: &ModelSpec,
    data: Option<&Bound<'py, PyDict>>,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be >= 1"));
    }
    rustmc_core::model::reject_potentials_for_prior_predictive(&model_spec.potentials)
        .map_err(model_error)?;
    // ── Build data maps ───────────────────────────────────────────────────────
    let mut data_map: HashMap<String, Vec<f64>> = model_spec.bound_data_1d.clone();
    let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> =
        model_spec.bound_data_2d.clone();
    if let Some(d) = data {
        let (e1, e2) = parse_data_dict(d)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, e1, e2);
    }

    validate_matrix_storage(&matrix_map)?;

    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;
    let graph = compiled.graph.clone();
    let likelihood_names = compiled.likelihood_names.clone();
    let heads = graph.observation_heads();

    // ── Sample from priors and run forward passes ─────────────────────────────
    // The generator itself lives in the core so a `GraphModel` loaded outside
    // Python simulates from exactly the same code.
    let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, PRIOR_PREDICT_SEED_DOMAIN));
    let draws = rustmc_core::prior_sampling::prior_predictive(
        &graph,
        &model_spec.priors,
        &compiled.display_params,
        &compiled.auto_vector_params,
        n_samples,
        &mut rng,
    )
    .map_err(model_error)?;

    // ── Package results ───────────────────────────────────────────────────────
    let dict = PyDict::new(py);
    for (pi, spec) in compiled.display_params.iter().enumerate() {
        let name = match spec {
            DisplayParamSpec::Raw { name, .. } => name,
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name,
        };
        let arr = PyArray1::from_vec(py, draws.params[pi].clone());
        dict.set_item(name, arr)?;
    }
    for (li, name) in likelihood_names.iter().enumerate() {
        let n_obs = heads[li].n_obs;
        let arr = Array2::from_shape_vec((n_samples, n_obs), draws.predictions[li].clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray(py))?;
    }
    for (j, (name, _)) in graph.deterministics.iter().enumerate() {
        let n = draws.deterministic_lens[j];
        if n == 0 {
            dict.set_item(
                name,
                PyArray1::from_vec(py, draws.deterministics[j].clone()),
            )?;
        } else {
            dict.set_item(
                name,
                Array2::from_shape_vec((n_samples, n), draws.deterministics[j].clone())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(dict)
}
