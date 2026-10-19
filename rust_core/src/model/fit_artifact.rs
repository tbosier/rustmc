//! The versioned `rustmc.graph-fit` artifact: a model artifact, its training
//! data and its posterior draws, validated on load. No executable payloads.
//!
//! Version 1 stores constrained draws only. Version 2 also stores the exact
//! sampler positions, which a rounded transform cannot recover (for example
//! `sigmoid(40)` rounds to one). Version 3 also stores each transition's
//! energy and, for NUTS, the maximum tree depth the chains ran with, from
//! which E-BFMI and tree-depth saturation are computed. Version 4 also stores
//! why each NUTS transition's tree stopped. All four load. Saving writes
//! version 4, except for a fit with less to store, which is written as the
//! newest version that holds all of it so that older readers can still load
//! it: version 3 for a NUTS fit without termination reasons (one loaded from
//! a version 3 artifact), and version 2 for a fit with no energies (one
//! loaded from a version 1 or 2 artifact).
use crate::autodiff::Evaluator;
use crate::data::{DataBinding, DataInputs, MatrixBinding};
use crate::graph::{Graph, ParamTransform};
use crate::hmc::{TransitionStats, TreeTermination};
use crate::sampler::SampleResult;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

const FORMAT: &str = "rustmc.graph-fit";
const VERSION: u32 = 4;
/// What a NUTS fit without termination reasons is saved as.
const VERSION_WITHOUT_TERMINATION: u32 = 3;
/// What a fit without energy telemetry is saved as.
const VERSION_WITHOUT_ENERGY: u32 = 2;
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
    /// Version 3, NUTS only: the maximum tree depth the chains ran with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_tree_depth: Option<usize>,
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
    /// Version 3: the Hamiltonian of the state the transition ended in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    energy: Option<f64>,
    /// Version 4, NUTS only: why the tree stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    termination: Option<Termination>,
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Termination {
    UTurn,
    Divergence,
    MaxTreeDepth,
}

impl From<TreeTermination> for Termination {
    fn from(value: TreeTermination) -> Self {
        match value {
            TreeTermination::UTurn => Self::UTurn,
            TreeTermination::Divergence => Self::Divergence,
            TreeTermination::MaxTreeDepth => Self::MaxTreeDepth,
        }
    }
}

impl From<Termination> for TreeTermination {
    fn from(value: Termination) -> Self {
        match value {
            Termination::UTurn => Self::UTurn,
            Termination::Divergence => Self::Divergence,
            Termination::MaxTreeDepth => Self::MaxTreeDepth,
        }
    }
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
            // A non-finite energy cannot be written as a JSON number; the
            // writer saves such a fit without energies (version 2).
            energy: value.energy.is_finite().then_some(value.energy),
            termination: value.termination.map(Termination::from),
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
            // Checked to be present and finite in version 3 and absent before.
            energy: self.energy.unwrap_or(f64::NAN),
            divergent: self.divergent,
            step_size: self.step_size,
            num_leapfrog_steps: self.num_leapfrog_steps,
            tree_depth: self.tree_depth,
            // Checked against the version and the tree in
            // `validate_terminations`.
            termination: self.termination.map(TreeTermination::from),
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
/// `Uniform(-2, 3)` is enough, and `rust_core/tests/fixtures/graph_fit_v2_uniform.json`
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

    /// Serialise the fit as a version-4 `rustmc.graph-fit` artifact: the
    /// model, the training data, the raw posterior draws and exact positions,
    /// and the sampler telemetry, energies and NUTS termination reasons
    /// included. A NUTS fit without termination reasons, which is only a fit
    /// loaded from a version 3 artifact, is written as version 3, and a fit
    /// whose energies are not all finite, which is only a fit loaded from a
    /// version 1 or 2 artifact, has none to store and is written as version 2.
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
        let has_energy = raw
            .transitions
            .iter()
            .flatten()
            .all(|transition| transition.energy.is_finite());
        let mut transitions: Vec<Vec<Transition>> = raw
            .transitions
            .iter()
            .map(|chain| chain.iter().map(Transition::from).collect())
            .collect();
        // HMC transitions have no tree and so no termination to store.
        let has_termination =
            raw.transitions.iter().flatten().all(|transition| {
                transition.tree_depth.is_none() || transition.termination.is_some()
            });
        let version = match (has_energy, has_termination) {
            (true, true) => VERSION,
            (true, false) => VERSION_WITHOUT_TERMINATION,
            (false, _) => VERSION_WITHOUT_ENERGY,
        };
        for transition in transitions.iter_mut().flatten() {
            if !has_energy {
                transition.energy = None;
            }
            if version < VERSION {
                transition.termination = None;
            }
        }
        let posterior = Posterior {
            coordinate_space: COORDINATE_SPACE.into(),
            param_names: raw.param_names.clone(),
            samples: raw.samples.clone(),
            unconstrained_samples: raw.unconstrained_samples.as_deref().cloned(),
            accept_rates: raw.accept_rates.clone(),
            step_sizes: raw.step_sizes.clone(),
            divergences: raw.divergences.clone(),
            transitions,
            max_tree_depth: raw.max_tree_depth.filter(|_| has_energy),
        };
        let artifact = Artifact {
            format: FORMAT.into(),
            version,
            model: self.model().artifact(),
            training: self.training()?,
            posterior,
        };
        serde_json::to_string(&artifact).map_err(|error| invalid(&error.to_string()))
    }

    /// Load and fully validate a version 1, 2, 3 or 4 `rustmc.graph-fit` artifact.
    ///
    /// Every stored draw is checked to be a finite point of the model's
    /// target with a finite gradient, the stored positions to agree with the
    /// stored draws, and the telemetry to agree with both. A discrete prior
    /// is refused: no gradient sampler can have produced its posterior.
    pub fn from_json(text: &str) -> ModelResult<Self> {
        let artifact: Artifact =
            serde_json::from_str(text).map_err(|error| invalid(&error.to_string()))?;
        if artifact.format != FORMAT || !matches!(artifact.version, 1..=4) {
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
    if version >= 2 && transformed && posterior.unconstrained_samples.is_none() {
        return Err(invalid(
            "versions 2 and 3 require unconstrained posterior positions for transformed parameters",
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
    let validated: Vec<ModelResult<ValidatedChain>> = (0..chains)
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
    let (validated_positions, potentials): (Vec<_>, Vec<_>) = validated
        .into_iter()
        .collect::<ModelResult<Vec<_>>>()?
        .into_iter()
        .unzip();
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
    let max_tree_depth = validate_energy_telemetry(&posterior, version)?;
    validate_terminations(&posterior, version, max_tree_depth)?;
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
        if version >= 3 {
            validate_draw_energies(&chain[warmup..], &potentials[chain_index])?;
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
        max_tree_depth,
    })
}

/// Check the version-3 telemetry, returning the maximum tree depth.
///
/// Version 3 stores a finite energy for every transition, and the maximum
/// tree depth exactly when the transitions are NUTS ones (every one has a tree
/// depth), each between 1 (every NUTS transition attempts one doubling) and
/// that maximum, with a leapfrog step count a tree of that depth can take.
/// Earlier versions store neither. Draw energies are checked against the
/// draws' potential energies in [`validate_draw_energies`].
fn validate_energy_telemetry(posterior: &Posterior, version: u32) -> ModelResult<Option<usize>> {
    let all = || posterior.transitions.iter().flatten();
    if version < 3 {
        if all().any(|transition| transition.energy.is_some()) || posterior.max_tree_depth.is_some()
        {
            return Err(invalid(
                "energies and the maximum tree depth need artifact version 3",
            ));
        }
        return Ok(None);
    }
    if !all().all(|transition| transition.energy.is_some_and(f64::is_finite)) {
        return Err(invalid(
            "version 3 requires a finite energy for every transition",
        ));
    }
    let nuts = all().all(|transition| transition.tree_depth.is_some());
    let hmc = all().all(|transition| transition.tree_depth.is_none());
    // The k-th doubling takes at most 2^(k-1) leapfrog steps, and it stops
    // early only on a divergence or U-turn, which also ends the tree. So every
    // doubling before the last ran in full, and a tree of depth d took
    // `2^(d-1)..=2^d - 1` steps. A depth of 0 is refused with the limit below.
    if all().any(|transition| {
        transition.tree_depth.is_some_and(|depth| {
            let steps = transition.num_leapfrog_steps;
            (1..64).contains(&depth)
                && (steps < 1usize << (depth - 1) || steps > (1usize << depth) - 1)
        })
    }) {
        return Err(invalid(
            "a tree depth disagrees with its transition's leapfrog step count",
        ));
    }
    match posterior.max_tree_depth {
        Some(limit)
            if nuts
                && (1..=63).contains(&limit)
                && all().all(|transition| {
                    transition
                        .tree_depth
                        .is_some_and(|depth| (1..=limit).contains(&depth))
                }) =>
        {
            Ok(Some(limit))
        }
        None if hmc => Ok(None),
        _ => Err(invalid(
            "the maximum tree depth disagrees with the transitions' tree depths",
        )),
    }
}

/// Check the version-4 termination reasons against the trees they ended.
///
/// Version 4 stores a reason for every NUTS transition and none for HMC ones;
/// earlier versions store none. A transition ends in a divergence exactly
/// when it is flagged divergent, and a tree that ran out of depth used every
/// doubling the limit allows, each in full: depth equal to the limit and
/// `2^depth - 1` leapfrog steps.
fn validate_terminations(
    posterior: &Posterior,
    version: u32,
    max_tree_depth: Option<usize>,
) -> ModelResult<()> {
    let all = || posterior.transitions.iter().flatten();
    if version < 4 {
        if all().any(|transition| transition.termination.is_some()) {
            return Err(invalid("termination reasons need artifact version 4"));
        }
        return Ok(());
    }
    if all().any(|transition| transition.termination.is_some() != transition.tree_depth.is_some()) {
        return Err(invalid(
            "version 4 stores a termination reason for every NUTS transition and none for HMC",
        ));
    }
    let impossible = all().any(|transition| match transition.termination {
        None => false,
        Some(reason) if (reason == Termination::Divergence) != transition.divergent => true,
        Some(Termination::MaxTreeDepth) => match (transition.tree_depth, max_tree_depth) {
            (Some(depth), Some(limit)) => {
                depth != limit || transition.num_leapfrog_steps != (1usize << depth) - 1
            }
            _ => true,
        },
        Some(_) => false,
    });
    if impossible {
        return Err(invalid(
            "a termination reason disagrees with its transition's tree",
        ));
    }
    Ok(())
}

/// Check one chain's draws, returning their unconstrained positions when
/// `retain_positions` asks for them (and an empty list otherwise).
/// A chain's retained unconstrained positions (empty unless they are kept)
/// and the potential energy, the negative log target, at every draw.
type ValidatedChain = (Vec<Vec<f64>>, Vec<f64>);

/// Check that each draw transition's energy is at least the potential energy
/// at the draw it ended in. The Hamiltonian is that potential plus a kinetic
/// energy that is never negative, so a lower energy cannot have come from the
/// sampler. The tolerance allows for rounding in a dense metric's kinetic
/// energy.
fn validate_draw_energies(transitions: &[TransitionStats], potentials: &[f64]) -> ModelResult<()> {
    if transitions
        .iter()
        .zip(potentials)
        .any(|(transition, &potential)| {
            transition.energy < potential - 1e-9 * (1.0 + potential.abs())
        })
    {
        return Err(invalid(
            "a stored energy is below the potential energy at its draw",
        ));
    }
    Ok(())
}

fn validate_chain(
    chain: &[Vec<f64>],
    stored_positions: Option<&Vec<Vec<f64>>>,
    draws: usize,
    graph: &Graph,
    retain_positions: bool,
) -> ModelResult<ValidatedChain> {
    if chain.len() != draws {
        return Err(invalid("posterior draw counts differ between chains"));
    }
    let mut evaluator = Evaluator::try_new(graph).map_err(|error| invalid(&error.to_string()))?;
    let mut positions = Vec::with_capacity(if retain_positions { draws } else { 0 });
    let mut potentials = Vec::with_capacity(draws);
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
        potentials.push(-evaluator.total_logp);
        if retain_positions {
            positions.push(position);
        }
    }
    Ok((positions, potentials))
}
