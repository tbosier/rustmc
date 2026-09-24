//! The versioned `rustmc.graph-fit` artifact: a model artifact, its training
//! data and its posterior draws, validated on load. No executable payloads.
//!
//! Version 1 stores constrained draws only. Version 2 also stores the exact
//! sampler positions, which a rounded transform cannot recover (for example
//! `sigmoid(40)` rounds to one); both load, and saving writes version 2.
use crate::autodiff::Evaluator;
use crate::data::{DataBinding, DataInputs, MatrixBinding};
use crate::graph::{Graph, ParamTransform};
use crate::hmc::TransitionStats;
use crate::sampler::SampleResult;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

const FORMAT: &str = "rustmc.graph-fit";
const VERSION: u32 = 2;
const COORDINATE_SPACE: &str = "constrained_graph_parameters";

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    format: String,
    version: u32,
    model: ModelArtifact,
    training: TrainingData,
    posterior: Posterior,
}

/// Keys are sorted so that saving the same fit twice writes the same bytes.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingData {
    vectors: BTreeMap<String, Vec<f64>>,
    matrices: BTreeMap<String, (Vec<f64>, usize, usize)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Posterior {
    /// Samples retain the graph's constrained latent variables, before display derivations.
    coordinate_space: String,
    param_names: Vec<String>,
    samples: Vec<Vec<Vec<f64>>>,
    /// Version 2 retains exact sampler positions for non-invertible rounded transforms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unconstrained_samples: Option<Vec<Vec<Vec<f64>>>>,
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
    fn into_core(self) -> ModelResult<TransitionStats> {
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

fn invalid(message: &str) -> ModelError {
    ModelError::invalid(format!("invalid graph fit artifact: {message}"))
}

/// The magnitude the position-agreement check must budget for, on top of the
/// draw's own, when deciding whether stored raw and constrained coordinates
/// describe the same draw.
///
/// `Identity` and `Exp` derive the constrained value from the raw one alone, so
/// two correct evaluations agree to within a few ULP *of the value*, and the
/// value's own magnitude is the only scale in play. `Sigmoid` is the same: its
/// output lies in (0, 1) and approaches its endpoints multiplicatively, so
/// nothing cancels. These keep a purely value-relative budget.
///
/// A bounded transform is different. Both `lower + span * s(raw)` and the
/// equivalent `upper - span * s(-raw)` round an intermediate term whose size is
/// set by the *interval*, and the final add or subtract can then cancel that
/// term down to a result arbitrarily close to zero. So on an interval that
/// straddles zero the absolute disagreement between two correct formulations
/// stays of order `EPSILON * span` while `EPSILON * |value|` collapses, and a
/// value-relative budget rejects artifacts this library itself wrote:
/// `Uniform(-2, 3)` is enough, and `tests/fixtures/graph_fit_v2_uniform.json`
/// is a real one, written before the bounded transform was re-associated to
/// evaluate its density at the point it reports.
///
/// Budgeting the span fixes that. Across 3.2M `(lower, upper, raw)` points with
/// interval widths from 1e-30 to 1e308, the largest disagreement between the
/// two formulations is 0.125 of `8 * EPSILON * (span + |value|)`. No
/// value-relative budget can cover the same set, because the required ratio is
/// unbounded as a draw approaches zero.
///
/// This does not blunt corruption detection. A raw position that decodes to a
/// genuinely different draw misses by a fraction of the interval, not by an ULP
/// of it, which is many orders of magnitude outside this budget.
fn agreement_scale(transform: &ParamTransform) -> f64 {
    match transform {
        ParamTransform::BoundedSigmoid { lower, upper } => (upper - lower).abs(),
        ParamTransform::Identity | ParamTransform::Exp | ParamTransform::Sigmoid => 0.0,
    }
}

impl ModelFit {
    /// The training data by key, as the fit artifact stores it and as the
    /// Python `FitResult.model` offers it for rebinding.
    pub fn training_data(&self) -> ModelResult<(Data1d, Data2d)> {
        let training = self.training()?;
        Ok((
            training.vectors.into_iter().collect(),
            training.matrices.into_iter().collect(),
        ))
    }

    fn training(&self) -> ModelResult<TrainingData> {
        let schema = &self.model().structure.schema;
        let binding = self.binding();
        let mut vectors = BTreeMap::new();
        for (slot, values) in schema
            .vectors
            .iter()
            .zip(&binding.vectors)
            .chain(schema.observations.iter().zip(&binding.observations))
        {
            if let Some(existing) = vectors.insert(slot.key.clone(), values.to_vec()) {
                if existing[..] != values[..] {
                    return Err(invalid(
                        "one data key refers to conflicting training payloads",
                    ));
                }
            }
        }
        let matrices = schema
            .matrices
            .iter()
            .zip(&binding.matrices)
            .map(|(slot, matrix)| {
                (
                    slot.key.clone(),
                    (matrix.data.to_vec(), matrix.n_rows, matrix.n_cols),
                )
            })
            .collect();
        Ok(TrainingData { vectors, matrices })
    }

    /// Serialise the fit as a version-2 `rustmc.graph-fit` artifact: the
    /// model, the training data, the raw posterior draws and exact positions,
    /// and the sampler telemetry.
    pub fn to_json(&self) -> ModelResult<String> {
        let raw = self.raw_samples();
        if raw
            .samples
            .iter()
            .flatten()
            .flatten()
            .any(|x| !x.is_finite())
            || raw.unconstrained_samples.as_ref().is_some_and(|positions| {
                positions.iter().flatten().flatten().any(|x| !x.is_finite())
            })
        {
            return Err(invalid("posterior positions must be finite"));
        }
        let posterior = Posterior {
            coordinate_space: COORDINATE_SPACE.into(),
            param_names: raw.param_names.clone(),
            samples: raw.samples.clone(),
            unconstrained_samples: raw.unconstrained_samples.as_deref().cloned(),
            accept_rates: raw.accept_rates.clone(),
            step_sizes: raw.step_sizes.clone(),
            divergences: raw.divergences.clone(),
            transitions: raw
                .transitions
                .iter()
                .map(|chain| chain.iter().map(Transition::from).collect())
                .collect(),
        };
        let artifact = Artifact {
            format: FORMAT.into(),
            version: VERSION,
            model: self.model().artifact(),
            training: self.training()?,
            posterior,
        };
        serde_json::to_string(&artifact).map_err(|error| invalid(&error.to_string()))
    }

    /// Load and fully validate a version 1 or 2 `rustmc.graph-fit` artifact.
    ///
    /// Every stored draw is checked to be a finite point of the model's
    /// target with a finite gradient, the stored positions to agree with the
    /// stored draws, and the telemetry to agree with both. A discrete prior
    /// is refused: no gradient sampler can have produced its posterior.
    pub fn from_json(text: &str) -> ModelResult<Self> {
        let artifact: Artifact =
            serde_json::from_str(text).map_err(|error| invalid(&error.to_string()))?;
        if artifact.format != FORMAT || !matches!(artifact.version, 1 | 2) {
            return Err(invalid("unsupported format/version"));
        }
        if artifact.version == 1 && artifact.posterior.unconstrained_samples.is_some() {
            return Err(invalid(
                "version 1 does not support unconstrained posterior positions",
            ));
        }
        artifact
            .model
            .validate_parameter_limit(artifact.posterior.param_names.len())?;
        let model = GraphModel::from_artifact(artifact.model)?;
        reject_discrete_priors_for_gradient_sampling(&model.definition.priors)?;
        let inputs = DataInputs {
            vectors: artifact
                .training
                .vectors
                .into_iter()
                .map(|(key, values)| (key, Arc::from(values)))
                .collect(),
            matrices: artifact
                .training
                .matrices
                .into_iter()
                .map(|(key, (values, n_rows, n_cols))| {
                    let matrix = MatrixBinding {
                        data: Arc::from(values),
                        n_rows,
                        n_cols,
                    };
                    (key, matrix)
                })
                .collect(),
        };
        let binding = DataBinding::bind(&model.structure.schema, inputs, "restored", true, true)
            .map_err(|error| ModelError::invalid(error.to_string()))?;
        binding
            .validate_for(&model.structure)
            .map_err(|error| invalid(&error.to_string()))?;
        let graph = model.structure.with_binding(&binding);
        graph
            .validate_shapes()
            .map_err(|error| invalid(&error.to_string()))?;
        let raw = validate_posterior(artifact.posterior, &graph, artifact.version)?;
        let fit = Self::from_shared(model, binding, Arc::new(raw))?;
        if fit
            .samples
            .samples
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("nonfinite derived parameter draws"));
        }
        Ok(fit)
    }
}

fn validate_posterior(
    posterior: Posterior,
    graph: &Graph,
    version: u32,
) -> ModelResult<SampleResult> {
    if posterior.coordinate_space != COORDINATE_SPACE {
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
    let transformed = graph
        .param_transforms
        .iter()
        .any(|transform| !matches!(transform, ParamTransform::Identity));
    if version == 2 && transformed && posterior.unconstrained_samples.is_none() {
        return Err(invalid(
            "version 2 requires unconstrained posterior positions for transformed parameters",
        ));
    }
    if let Some(positions) = &posterior.unconstrained_samples {
        if positions.len() != chains || positions.iter().any(|chain| chain.len() != draws) {
            return Err(invalid(
                "unconstrained posterior chain/draw dimensions differ from samples",
            ));
        }
    }
    let retain_positions = transformed || posterior.unconstrained_samples.is_some();
    // Each draw costs a full target and gradient evaluation, so chains are
    // checked in parallel; the first failure in chain order is reported, as
    // a sequential scan would.
    let validated: Vec<ModelResult<Vec<Vec<f64>>>> = (0..chains)
        .into_par_iter()
        .map(|chain_index| {
            validate_chain(
                &posterior.samples[chain_index],
                posterior
                    .unconstrained_samples
                    .as_ref()
                    .map(|positions| &positions[chain_index]),
                draws,
                graph,
                retain_positions,
            )
        })
        .collect();
    let validated_positions = validated.into_iter().collect::<ModelResult<Vec<_>>>()?;
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
            .collect::<ModelResult<_>>()?;
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
        unconstrained_samples: retain_positions.then(|| Arc::new(validated_positions)),
        param_names: posterior.param_names,
        accept_rates: posterior.accept_rates,
        step_sizes: posterior.step_sizes,
        divergences: posterior.divergences,
        transitions,
    })
}

/// Check one chain's draws, returning their unconstrained positions when
/// `retain_positions` asks for them (and an empty list otherwise).
fn validate_chain(
    chain: &[Vec<f64>],
    stored_positions: Option<&Vec<Vec<f64>>>,
    draws: usize,
    graph: &Graph,
    retain_positions: bool,
) -> ModelResult<Vec<Vec<f64>>> {
    if chain.len() != draws {
        return Err(invalid("posterior draw counts differ between chains"));
    }
    let mut evaluator = Evaluator::try_new(graph).map_err(|error| invalid(&error.to_string()))?;
    let mut positions = Vec::with_capacity(if retain_positions { draws } else { 0 });
    for (draw_index, draw) in chain.iter().enumerate() {
        if draw.len() != graph.param_count || draw.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "posterior positions have invalid dimensions or nonfinite values",
            ));
        }
        let position = if let Some(stored) = stored_positions {
            let position = &stored[draw_index];
            if position.len() != graph.param_count || position.iter().any(|x| !x.is_finite()) {
                return Err(invalid(
                    "unconstrained posterior positions have invalid dimensions or nonfinite values",
                ));
            }
            for ((raw, displayed), transform) in
                position.iter().zip(draw).zip(&graph.param_transforms)
            {
                let expected = transform.apply(*raw);
                // Permit ordinary floating-point JSON reconstruction error,
                // but verify the supplied coordinates describe the same draw.
                let tolerance = 8.0
                    * f64::EPSILON
                    * (agreement_scale(transform) + expected.abs().max(displayed.abs()));
                if !expected.is_finite() || (expected - displayed).abs() > tolerance {
                    return Err(invalid(
                        "unconstrained posterior positions disagree with constrained samples",
                    ));
                }
            }
            position.clone()
        } else {
            constrained_draw_to_raw(draw, &graph.param_transforms)
        };
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
        if retain_positions {
            positions.push(position);
        }
    }
    Ok(positions)
}
