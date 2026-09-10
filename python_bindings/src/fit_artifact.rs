//! Validated, versioned graph-fit persistence without executable payloads.
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustmc_core::hmc::TransitionStats;
use rustmc_core::sampler::SampleResult;

use super::{
    compile_python_model, constrained_draw_to_raw, core_binding_from_maps,
    derive_display_sample_result, model_artifact, Data1d, Data2d, FitResult, PyCompiledModel,
};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    format: String,
    version: u32,
    model: model_artifact::Artifact,
    training: TrainingData,
    posterior: Posterior,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingData {
    vectors: Data1d,
    matrices: Data2d,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Posterior {
    /// Samples retain the graph's constrained latent variables, before display derivations.
    coordinate_space: String,
    param_names: Vec<String>,
    samples: Vec<Vec<Vec<f64>>>,
    accept_rates: Vec<f64>,
    step_sizes: Vec<f64>,
    divergences: Vec<usize>,
    transitions: Vec<Vec<Transition>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Transition {
    is_warmup: bool,
    accepted: bool,
    accept_prob: f64,
    energy_error: EnergyError,
    divergent: bool,
    step_size: f64,
    num_leapfrog_steps: usize,
    tree_depth: Option<usize>,
}

/// Divergent transitions can legitimately have nonfinite energy errors. JSON numbers
/// remain finite; these three explicit tokens preserve the sampler's telemetry.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum EnergyError {
    Finite(f64),
    Special(SpecialEnergyError),
}
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum SpecialEnergyError {
    PositiveInfinity,
    NegativeInfinity,
    Nan,
}

impl From<&TransitionStats> for Transition {
    fn from(value: &TransitionStats) -> Self {
        let energy_error = if value.energy_error.is_finite() {
            EnergyError::Finite(value.energy_error)
        } else if value.energy_error.is_nan() {
            EnergyError::Special(SpecialEnergyError::Nan)
        } else if value.energy_error.is_sign_positive() {
            EnergyError::Special(SpecialEnergyError::PositiveInfinity)
        } else {
            EnergyError::Special(SpecialEnergyError::NegativeInfinity)
        };
        Self {
            is_warmup: value.is_warmup,
            accepted: value.accepted,
            accept_prob: value.accept_prob,
            energy_error,
            divergent: value.divergent,
            step_size: value.step_size,
            num_leapfrog_steps: value.num_leapfrog_steps,
            tree_depth: value.tree_depth,
        }
    }
}
impl Transition {
    fn into_core(self) -> PyResult<TransitionStats> {
        if !self.accept_prob.is_finite()
            || !(0.0..=1.0).contains(&self.accept_prob)
            || !self.step_size.is_finite()
            || self.step_size <= 0.0
            || self.num_leapfrog_steps == 0
        {
            return Err(invalid(
                "transition probabilities, step sizes, or trajectory lengths",
            ));
        }
        let energy_error = match self.energy_error {
            EnergyError::Finite(value) if value.is_finite() => value,
            EnergyError::Finite(_) => return Err(invalid("nonfinite numeric energy error")),
            EnergyError::Special(_) if !self.divergent => {
                return Err(invalid(
                    "nonfinite energy error on a nondivergent transition",
                ));
            }
            EnergyError::Special(SpecialEnergyError::PositiveInfinity) => f64::INFINITY,
            EnergyError::Special(SpecialEnergyError::NegativeInfinity) => f64::NEG_INFINITY,
            EnergyError::Special(SpecialEnergyError::Nan) => f64::NAN,
        };
        Ok(TransitionStats {
            is_warmup: self.is_warmup,
            accepted: self.accepted,
            accept_prob: self.accept_prob,
            energy_error,
            divergent: self.divergent,
            step_size: self.step_size,
            num_leapfrog_steps: self.num_leapfrog_steps,
            tree_depth: self.tree_depth,
        })
    }
}

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(format!("invalid graph fit artifact: {message}"))
}

fn training_data(fit: &FitResult) -> PyResult<TrainingData> {
    let mut vectors: Data1d = HashMap::new();
    for (slot, values) in fit
        .graph
        .schema
        .vectors
        .iter()
        .zip(&fit.graph.data_vectors)
        .chain(
            fit.graph
                .schema
                .observations
                .iter()
                .zip(&fit.graph.obs_vectors),
        )
    {
        if let Some(existing) = vectors.insert(slot.key.clone(), values.clone()) {
            if existing != *values {
                return Err(invalid(
                    "one data key refers to conflicting training payloads",
                ));
            }
        }
    }
    let matrices = fit
        .graph
        .schema
        .matrices
        .iter()
        .zip(&fit.graph.data_matrices)
        .map(|(slot, matrix)| {
            (
                slot.key.clone(),
                (matrix.data.clone(), matrix.n_rows, matrix.n_cols),
            )
        })
        .collect();
    Ok(TrainingData { vectors, matrices })
}

pub(super) fn model(fit: &FitResult) -> PyResult<PyCompiledModel> {
    let training = training_data(fit)?;
    let compiled = compile_python_model(&fit.definition, &training.vectors, &training.matrices)?;
    Ok(PyCompiledModel {
        definition: fit.definition.clone(),
        structure: Arc::new(compiled.graph.structure_only()),
        likelihood_names: compiled.likelihood_names,
        display_params: compiled.display_params,
        default_data_1d: training.vectors,
        default_data_2d: training.matrices,
    })
}

pub(super) fn encode(fit: &FitResult) -> PyResult<String> {
    let posterior = Posterior {
        coordinate_space: "constrained_graph_parameters".into(),
        param_names: fit.raw_result.param_names.clone(),
        samples: fit.raw_result.samples.clone(),
        accept_rates: fit.raw_result.accept_rates.clone(),
        step_sizes: fit.raw_result.step_sizes.clone(),
        divergences: fit.raw_result.divergences.clone(),
        transitions: fit
            .raw_result
            .transitions
            .iter()
            .map(|chain| chain.iter().map(Transition::from).collect())
            .collect(),
    };
    let compiled = model(fit)?;
    let artifact = Artifact {
        format: "rustmc.graph-fit".into(),
        version: 1,
        model: model_artifact::describe(&compiled),
        training: training_data(fit)?,
        posterior,
    };
    serde_json::to_string(&artifact).map_err(|error| invalid(&error.to_string()))
}

pub(super) fn decode(text: &str) -> PyResult<FitResult> {
    let artifact: Artifact =
        serde_json::from_str(text).map_err(|error| invalid(&error.to_string()))?;
    if artifact.format != "rustmc.graph-fit" || artifact.version != 1 {
        return Err(invalid("unsupported format/version"));
    }
    artifact
        .model
        .validate_parameter_limit(artifact.posterior.param_names.len())?;
    let compiled = model_artifact::reconstruct(artifact.model)?;
    let binding = core_binding_from_maps(
        &compiled.structure.schema,
        &artifact.training.vectors,
        &artifact.training.matrices,
        "restored".into(),
        true,
        true,
    )?;
    binding
        .validate_for(&compiled.structure)
        .map_err(|error| invalid(&error.to_string()))?;
    let graph = compiled.structure.with_binding(&binding);
    graph
        .validate_shapes()
        .map_err(|error| invalid(&error.to_string()))?;
    let raw_result = validate_posterior(artifact.posterior, &graph)?;
    let display_result = derive_display_sample_result(&raw_result, &compiled.display_params)?;
    if display_result
        .samples
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(invalid("nonfinite derived parameter draws"));
    }
    Ok(FitResult {
        definition: compiled.definition,
        raw_result,
        display_result,
        graph,
        likelihood_names: compiled.likelihood_names,
    })
}

fn validate_posterior(
    posterior: Posterior,
    graph: &rustmc_core::graph::Graph,
) -> PyResult<SampleResult> {
    if posterior.coordinate_space != "constrained_graph_parameters" {
        return Err(invalid("unsupported posterior coordinate space"));
    }
    if posterior.param_names != graph.param_names {
        return Err(invalid(
            "posterior parameter names/order differ from the model",
        ));
    }
    let chains = posterior.samples.len();
    let draws = posterior.samples.first().map_or(0, Vec::len);
    if chains == 0 || draws == 0 || graph.param_count == 0 {
        return Err(invalid(
            "posterior must contain chains, draws, and parameters",
        ));
    }
    let mut evaluator = rustmc_core::autodiff::Evaluator::try_new(graph)
        .map_err(|error| invalid(&error.to_string()))?;
    for chain in &posterior.samples {
        if chain.len() != draws {
            return Err(invalid("posterior draw counts differ between chains"));
        }
        for draw in chain {
            if draw.len() != graph.param_count || draw.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "posterior positions have invalid dimensions or nonfinite values",
                ));
            }
            let position = constrained_draw_to_raw(draw, &graph.param_transforms);
            if position.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "posterior position is outside its parameter transform support",
                ));
            }
            evaluator.compute(graph, &position);
            if !evaluator.total_logp.is_finite()
                || evaluator.grad.iter().any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "posterior position has a nonfinite target density or gradient",
                ));
            }
        }
    }
    if posterior.accept_rates.len() != chains
        || posterior.step_sizes.len() != chains
        || posterior.divergences.len() != chains
        || posterior.transitions.len() != chains
    {
        return Err(invalid(
            "diagnostic chain dimensions do not match posterior",
        ));
    }
    if posterior
        .accept_rates
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || posterior
            .step_sizes
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(invalid("invalid diagnostic acceptance rates or step sizes"));
    }
    let transition_count = posterior.transitions[0].len();
    let mut transitions = Vec::with_capacity(chains);
    for (chain_index, chain) in posterior.transitions.into_iter().enumerate() {
        if chain.len() != transition_count || chain.len() < draws {
            return Err(invalid(
                "diagnostic transition dimensions do not match posterior",
            ));
        }
        let warmup = chain.len() - draws;
        let chain: Vec<TransitionStats> = chain
            .into_iter()
            .map(Transition::into_core)
            .collect::<PyResult<_>>()?;
        if chain
            .iter()
            .enumerate()
            .any(|(index, transition)| transition.is_warmup != (index < warmup))
        {
            return Err(invalid(
                "warmup flags do not match the posterior draw count",
            ));
        }
        let divergences = chain.iter().filter(|t| !t.is_warmup && t.divergent).count();
        if divergences != posterior.divergences[chain_index] {
            return Err(invalid(
                "divergence counts disagree with transition diagnostics",
            ));
        }
        transitions.push(chain);
    }
    Ok(SampleResult {
        samples: posterior.samples,
        param_names: posterior.param_names,
        accept_rates: posterior.accept_rates,
        step_sizes: posterior.step_sizes,
        divergences: posterior.divergences,
        transitions,
    })
}
