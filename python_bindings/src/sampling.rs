//! Module-level `sample` and `sample_prior_predictive`, and the sampler
//! options every sampling entry point shares.
use crate::builder::{
    compile_python_model, reject_discrete_priors_for_gradient_sampling, ModelSpec,
};
use crate::fit_result::FitResult;
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
use rustmc_core::sampler::{BatchSampleConfig, MetricKind, SamplerConfig, SamplerType};
use rustmc_core::seeding::{stream_seed, PRIOR_PREDICT_SEED_DOMAIN};

fn validate_sample_config(
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

fn parse_sampler_type(sampler: &str) -> PyResult<SamplerType> {
    match sampler {
        "nuts" | "NUTS" => Ok(SamplerType::Nuts),
        "hmc" | "HMC" => Ok(SamplerType::Hmc),
        _ => Err(PyValueError::new_err(format!(
            "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
            sampler
        ))),
    }
}

/// The sampler options shared by `sample`, `CompiledModel.sample` and the
/// batch entry points, as Python passes them, checked once.
pub(crate) struct SamplerOptions<'a> {
    pub(crate) chains: usize,
    pub(crate) draws: usize,
    pub(crate) warmup: usize,
    pub(crate) seed: u64,
    pub(crate) step_size: f64,
    pub(crate) target_accept: f64,
    pub(crate) sampler: &'a str,
    pub(crate) max_tree_depth: usize,
    pub(crate) num_leapfrog_steps: usize,
    pub(crate) show_progress: bool,
    pub(crate) metric: &'a str,
}

impl SamplerOptions<'_> {
    /// Configuration for one fit on a pool of `threads` workers.
    pub(crate) fn single(&self, threads: usize) -> PyResult<SamplerConfig> {
        let batch = self.batch()?;
        Ok(SamplerConfig {
            sampler: batch.sampler,
            num_chains: batch.num_chains,
            num_draws: batch.num_draws,
            num_warmup: batch.num_warmup,
            step_size: batch.step_size,
            target_accept: batch.target_accept,
            num_leapfrog_steps: batch.num_leapfrog_steps,
            max_tree_depth: batch.max_tree_depth,
            seed: batch.seed,
            num_threads: threads,
            show_progress: batch.show_progress,
            metric: batch.metric,
        })
    }

    /// Configuration for a batch of fits.
    pub(crate) fn batch(&self) -> PyResult<BatchSampleConfig> {
        let metric = MetricKind::parse(self.metric).map_err(PyValueError::new_err)?;
        validate_sample_config(
            self.chains,
            self.draws,
            self.warmup,
            self.step_size,
            self.target_accept,
            self.max_tree_depth,
            self.num_leapfrog_steps,
        )?;
        Ok(BatchSampleConfig {
            sampler: parse_sampler_type(self.sampler)?,
            num_chains: self.chains,
            num_draws: self.draws,
            num_warmup: self.warmup,
            step_size: self.step_size,
            target_accept: self.target_accept,
            num_leapfrog_steps: self.num_leapfrog_steps,
            max_tree_depth: self.max_tree_depth,
            seed: self.seed,
            show_progress: self.show_progress,
            metric,
        })
    }
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
    let config = SamplerOptions {
        chains,
        draws,
        warmup,
        seed,
        step_size,
        target_accept,
        sampler,
        max_tree_depth,
        num_leapfrog_steps,
        show_progress,
        metric,
    }
    .single(threads)?;
    reject_discrete_priors_for_gradient_sampling(&model_spec.priors)?;
    // Start from data bound at build time, then let call-site data override/extend.
    let (data_map, matrix_map) = model_spec.data_with(data)?;
    if data_map.is_empty() && matrix_map.is_empty() && !model_spec.likelihoods.is_empty() {
        return Err(PyValueError::new_err(
            "No data provided. Pass data= to sample() or bind it via ModelBuilder(data=...).",
        ));
    }
    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;
    let binding = CoreDataBinding::from_graph(&compiled.graph)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let model = compiled.into_model(model_spec);
    let fit = py
        .allow_threads(|| model.sample(binding, config, init))
        .map_err(model_error)?;
    Ok(FitResult::new(fit))
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
    let (data_map, matrix_map) = model_spec.data_with(data)?;
    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;
    let graph = &compiled.graph;
    let heads = graph.observation_heads();

    // The generator itself lives in the core so a `GraphModel` loaded outside
    // Python simulates from exactly the same code.
    let mut draws = py
        .allow_threads(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, PRIOR_PREDICT_SEED_DOMAIN));
            rustmc_core::prior_sampling::prior_predictive(
                graph,
                &model_spec.priors,
                &compiled.display_params,
                &compiled.auto_vector_params,
                n_samples,
                &mut rng,
            )
        })
        .map_err(model_error)?;

    let dict = PyDict::new(py);
    for (pi, spec) in compiled.display_params.iter().enumerate() {
        let name = match spec {
            DisplayParamSpec::Raw { name, .. } => name,
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name,
        };
        let values = std::mem::take(&mut draws.params[pi]);
        dict.set_item(name, PyArray1::from_vec(py, values))?;
    }
    for (li, name) in compiled.likelihood_names.iter().enumerate() {
        let n_obs = heads[li].n_obs;
        let values = std::mem::take(&mut draws.predictions[li]);
        let arr = Array2::from_shape_vec((n_samples, n_obs), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray(py))?;
    }
    for (j, (name, _)) in graph.deterministics.iter().enumerate() {
        let n = draws.deterministic_lens[j];
        let values = std::mem::take(&mut draws.deterministics[j]);
        if n == 0 {
            dict.set_item(name, PyArray1::from_vec(py, values))?;
        } else {
            dict.set_item(
                name,
                Array2::from_shape_vec((n_samples, n), values)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(dict)
}
