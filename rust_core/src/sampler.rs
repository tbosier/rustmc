use crate::autodiff::Evaluator;
use crate::data::DataBinding;
use crate::diagnostics::{self, DiagnosticsReport};
use crate::graph::{Graph, Op, ParamTransform};
use crate::hmc::{self, ChainResult, HmcConfig, TransitionStats};
use crate::nuts::{self, NutsConfig};
use crate::progress::{ProgressGuard, ProgressState};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerType {
    Nuts,
    Hmc,
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub sampler: SamplerType,
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub step_size: f64,
    /// Desired average acceptance probability for step-size adaptation.
    pub target_accept: f64,
    /// HMC only: fixed number of leapfrog steps.
    pub num_leapfrog_steps: usize,
    /// NUTS only: maximum tree depth (default 10).
    pub max_tree_depth: usize,
    pub seed: u64,
    pub num_threads: usize,
    pub show_progress: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerType::Nuts,
            num_chains: 4,
            num_draws: 1000,
            num_warmup: 500,
            step_size: 0.0,
            target_accept: 0.80,
            num_leapfrog_steps: 15,
            max_tree_depth: 10,
            seed: 42,
            num_threads: 0,
            show_progress: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchSampleConfig {
    pub sampler: SamplerType,
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub step_size: f64,
    pub target_accept: f64,
    pub num_leapfrog_steps: usize,
    pub max_tree_depth: usize,
    pub seed: u64,
    pub show_progress: bool,
}

impl Default for BatchSampleConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerType::Nuts,
            num_chains: 1,
            num_draws: 500,
            num_warmup: 300,
            step_size: 0.0,
            target_accept: 0.80,
            num_leapfrog_steps: 15,
            max_tree_depth: 8,
            seed: 42,
            show_progress: true,
        }
    }
}

impl SamplerConfig {
    /// Validate controls at the Rust boundary before any allocation or execution.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_chains == 0 || self.num_draws == 0 {
            return Err("chains and draws must be positive".into());
        }
        if self.num_warmup == 0 {
            return Err("warmup must be positive".into());
        }
        if !self.step_size.is_finite() || self.step_size < 0.0 {
            return Err("step_size must be finite and nonnegative".into());
        }
        validate_target_accept(self.target_accept)?;
        if !(1..=63).contains(&self.max_tree_depth) {
            return Err("max_tree_depth must be between 1 and 63".into());
        }
        if self.num_leapfrog_steps == 0 {
            return Err("num_leapfrog_steps must be positive".into());
        }
        self.num_warmup
            .checked_add(self.num_draws)
            .and_then(|n| n.checked_mul(self.num_chains))
            .filter(|n| *n <= isize::MAX as usize / std::mem::size_of::<TransitionStats>())
            .ok_or_else(|| "sampling allocation size overflow".to_string())?;
        Ok(())
    }
}

impl BatchSampleConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.sampler_config(self.seed).validate()
    }
    fn sampler_config(&self, seed: u64) -> SamplerConfig {
        SamplerConfig {
            sampler: self.sampler,
            num_chains: self.num_chains,
            num_draws: self.num_draws,
            num_warmup: self.num_warmup,
            step_size: self.step_size,
            target_accept: self.target_accept,
            num_leapfrog_steps: self.num_leapfrog_steps,
            max_tree_depth: self.max_tree_depth,
            seed,
            num_threads: 0,
            show_progress: false,
        }
    }
}

pub(crate) fn validate_initial_values(
    initial: Option<Vec<Vec<f64>>>,
    chains: usize,
    dimension: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let positions = initial.unwrap_or_else(|| vec![vec![0.0; dimension]; chains]);
    if positions.len() != chains
        || positions
            .iter()
            .any(|q| q.len() != dimension || q.iter().any(|x| !x.is_finite()))
    {
        return Err("init must contain one finite unconstrained parameter vector per chain".into());
    }
    Ok(positions)
}

#[derive(Debug, Clone)]
pub struct SampleResult {
    pub samples: Vec<Vec<Vec<f64>>>,
    /// Exact sampler positions, indexed by chain/draw/parameter, retained when
    /// any parameter is transformed. Constrained floating-point draws cannot
    /// always be inverted (for example sigmoid(40) rounds to one).
    pub unconstrained_samples: Option<Arc<Vec<Vec<Vec<f64>>>>>,
    pub accept_rates: Vec<f64>,
    pub step_sizes: Vec<f64>,
    pub divergences: Vec<usize>,
    pub transitions: Vec<Vec<TransitionStats>>,
    pub param_names: Vec<String>,
}

impl SampleResult {
    pub fn mean(&self) -> Vec<f64> {
        let n_params = self.param_names.len();
        let mut sums = vec![0.0; n_params];
        let mut count = 0usize;

        for chain in &self.samples {
            for draw in chain {
                for (i, v) in draw.iter().enumerate() {
                    sums[i] += v;
                }
                count += 1;
            }
        }

        sums.iter().map(|s| s / count as f64).collect()
    }

    pub fn std(&self) -> Vec<f64> {
        let means = self.mean();
        let n_params = self.param_names.len();
        let mut sum_sq = vec![0.0; n_params];
        let mut count = 0usize;

        for chain in &self.samples {
            for draw in chain {
                for (i, v) in draw.iter().enumerate() {
                    let diff = v - means[i];
                    sum_sq[i] += diff * diff;
                }
                count += 1;
            }
        }

        sum_sq.iter().map(|s| (s / count as f64).sqrt()).collect()
    }

    pub fn total_divergences(&self) -> usize {
        self.divergences.iter().sum()
    }

    pub fn diagnostics(&self) -> DiagnosticsReport {
        diagnostics::compute_diagnostics(
            &self.samples,
            &self.param_names,
            &self.accept_rates,
            self.total_divergences(),
        )
    }

    /// Structured per-transition telemetry aggregated across chains.
    pub fn transition_diagnostics(&self) -> diagnostics::TransitionDiagnosticsReport {
        diagnostics::compute_transition_diagnostics(&self.transitions)
    }
}

pub(crate) fn with_thread_pool<T, F>(num_threads: usize, f: F) -> Result<T, String>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    if num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|error| format!("could not build sampler thread pool: {error}"))?;
        Ok(pool.install(f))
    } else {
        Ok(f())
    }
}

fn validate_initial_target(
    graph: &Graph,
    binding: DataBinding,
    initial: &[f64],
) -> Result<(), String> {
    let mut evaluator =
        Evaluator::try_with_binding(graph, binding).map_err(|error| error.to_string())?;
    evaluator.compute(graph, initial);
    if !evaluator.total_logp.is_finite() {
        return Err(format!(
            "initial log density is not finite at the supplied initialization ({})",
            evaluator.total_logp
        ));
    }
    if let Some((index, value)) = evaluator
        .grad
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        let name = graph
            .param_names
            .get(index)
            .map_or("<unknown>", String::as_str);
        return Err(format!(
            "initial gradient for parameter '{}' is not finite at the supplied initialization ({})",
            name, value
        ));
    }
    Ok(())
}

/// Reject discrete latent parameters before any gradient-based sampling.
///
/// HMC and NUTS evolve a continuous Euclidean state, so a discrete latent needs
/// marginalisation or a discrete transition kernel. `Bernoulli::prior` and
/// `Poisson::prior` build exactly that — `Op::BernoulliLogP` / `Op::PoissonLogP`
/// over a free `Op::Param` — and the two fail differently, neither usefully:
///
/// - Bernoulli does not check its support at all (`bernoulli_logp_scalar` is
///   `x * ln p + (1 - x) * ln(1 - p)`, which at `p = 0.5` is constant over all
///   of R). The chain random-walks a flat direction and returns fractional
///   "draws" for a parameter whose support is {0, 1}; 20 draws reach -813.
/// - Poisson does check: `poisson_logp_scalar` delegates to
///   `count_sampling::log_mass`, which returns -inf when `count.fract() != 0`.
///   Every off-integer proposal is then rejected, so the chain is pinned to its
///   integer initialization, reporting divergence on every transition while the
///   step size collapses. Not a wrong number — no number at all.
///
/// Neither reports a support error, so without this check both look like an
/// ordinary fit that simply mixed badly.
///
/// The `ModelSpec` layer already refuses these priors, but that check is
/// upstream of the sampler: a Rust caller assembling a `Graph` by hand bypasses
/// it entirely. This scan sits on the graph itself, so it covers every
/// gradient-based entry point in this module, plus `model::GraphModel::sample`,
/// which funnels here.
///
/// The raw kernels `nuts::run_chain`, `nuts::run_chain_bound`, `hmc::run_chain`
/// and `hmc::run_chain_bound` are `pub` and do not pass through this module, so
/// they call this directly. That is why they return a `Result` rather than a
/// bare `ChainResult`. `sampler`'s own call sites use the `_unguarded` variants
/// instead, having already run this once at their boundary.
///
/// Observed data is never rejected, and not by a heuristic: the observation
/// likelihoods (`obs_logp_bernoulli_logit`, `obs_logp_poisson_log`) are a
/// different op entirely — `Op::ObsLogP`, whose response is `obs_data_idx`, an
/// index into the binding's observation vectors rather than a `NodeId` — so a
/// sampled value cannot occupy the response side of one, and they are not
/// visited here at all. A discrete *latent* is the other shape: `x: NodeId`
/// pointing at `Op::Param`.
///
/// A free parameter reaching `x` indirectly counts too —
/// `bernoulli_logp(graph.exp(param), p)`, or an artifact wiring an
/// `Add`/`Sigmoid` between them. Such a graph is equally invalid, and worse in
/// one respect: `Op::BernoulliLogP`'s backward pass propagates no adjoint to
/// `x` at all, so the term moves the density without moving the gradient.
/// `Graph::reachable_param` does the walk, over an exhaustive match on `Op`
/// that lives beside the enum so it cannot fall behind it. No constructor in
/// this crate builds that shape — `Bernoulli::prior` and `Poisson::prior` both
/// pass a bare parameter — so this is about what the published `Graph` API
/// lets a caller assemble.
pub(crate) fn reject_discrete_latent_parameters(graph: &Graph) -> Result<(), String> {
    let mut offenders: Vec<(usize, &str, &str)> = Vec::new();
    // Scan every node rather than just `graph.logp_terms`. For a graph built
    // through `Graph`'s own API the two are equivalent, because `bernoulli_logp`
    // and `poisson_logp` always register the term they create. Scanning all
    // nodes costs one extra pass and does not rely on that invariant holding for
    // every future constructor, so it fails closed if one ever forgets to
    // register its term.
    //
    // Cost is one pass over the nodes, plus one operand walk per discrete term
    // found. Models with no `Bernoulli`/`Poisson` prior — which is nearly all
    // of them — pay only the pass; a model with `d` discrete terms pays `d`
    // walks, each allocating and clearing one bitmap over the nodes. Both are
    // negligible beside the sampling that follows.
    for node in &graph.nodes {
        let (x, family) = match node.op {
            Op::BernoulliLogP { x, .. } => (x, "Bernoulli"),
            Op::PoissonLogP { x, .. } => (x, "Poisson"),
            _ => continue,
        };
        // A transform cannot rescue a discrete support, so every free parameter
        // under one of these densities is an offender regardless of its
        // `ParamTransform` — and regardless of how many nodes separate it from
        // the density. A discrete term over a constant reaches no parameter and
        // is not a latent, so it is left alone.
        let Some(index) = graph.reachable_param(x) else {
            continue;
        };
        let name = graph
            .param_names
            .get(index)
            .map_or("<unknown>", String::as_str);
        offenders.push((index, name, family));
    }
    if offenders.is_empty() {
        return Ok(());
    }
    offenders.sort_unstable();
    offenders.dedup();
    let listed: Vec<String> = offenders
        .iter()
        .map(|(_, name, family)| format!("'{name}' ({family})"))
        .collect();
    Err(format!(
        "discrete latent parameter(s) {} cannot be sampled with HMC/NUTS: \
         gradient-based samplers evolve a continuous state, so a discrete \
         parameter requires marginalisation or a discrete transition kernel. \
         Bernoulli and Poisson priors are available for prior-predictive \
         simulation only; for discrete observations use a Bernoulli-logit or \
         Poisson-log observation likelihood instead.",
        listed.join(", ")
    ))
}

fn validate_target_accept(target_accept: f64) -> Result<(), String> {
    if target_accept.is_finite() && target_accept > 0.0 && target_accept < 1.0 {
        Ok(())
    } else {
        Err("target_accept must be finite and strictly between 0 and 1".to_string())
    }
}

pub fn sample(graph: Graph, config: SamplerConfig) -> Result<SampleResult, String> {
    graph.validate_shapes().map_err(|e| e.to_string())?;
    let binding = DataBinding::from_graph(&graph).map_err(|e| e.to_string())?;
    sample_bound(Arc::new(graph.structure_only()), binding, config)
}

/// Sample one validated binding while sharing immutable model structure.
pub fn sample_bound(
    graph: Arc<Graph>,
    binding: DataBinding,
    config: SamplerConfig,
) -> Result<SampleResult, String> {
    sample_bound_with_init(graph, binding, config, None)
}

/// Sample with one explicit raw (unconstrained) initial vector per chain.
/// None preserves the legacy zero initialization.
pub fn sample_bound_with_init(
    graph: Arc<Graph>,
    binding: DataBinding,
    config: SamplerConfig,
    initial: Option<Vec<Vec<f64>>>,
) -> Result<SampleResult, String> {
    config.validate()?;
    // Single chokepoint: `sample`, `sample_bound`, both bound-batch entry points
    // and `model::GraphModel::sample` all funnel through here.
    reject_discrete_latent_parameters(&graph)?;
    let initial = validate_initial_values(initial, config.num_chains, graph.param_count)?;
    binding.validate_for(&graph).map_err(|e| e.to_string())?;
    for position in &initial {
        validate_initial_target(&graph, binding.clone(), position)?;
    }
    let param_names = graph.param_names.clone();

    // For progress bar, leapfrog count is approximate for NUTS
    let approx_leapfrog = match config.sampler {
        SamplerType::Hmc => config.num_leapfrog_steps,
        SamplerType::Nuts => 1 << (config.max_tree_depth / 2),
    };

    let progress_state = if config.show_progress {
        Some(Arc::new(ProgressState::new(
            config.num_chains,
            config.num_draws,
            config.num_warmup,
            approx_leapfrog,
        )))
    } else {
        None
    };

    let _progress_guard = progress_state
        .as_ref()
        .map(|ps| ProgressGuard::spawn(Arc::clone(ps)));

    let chain_indices: Vec<usize> = (0..config.num_chains).collect();

    let results: Vec<ChainResult> = with_thread_pool(config.num_threads, || {
        chain_indices
            .par_iter()
            .map(|&chain_idx| {
                let mut rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(chain_idx as u64));
                let prog_ref = progress_state.as_deref();

                match config.sampler {
                    SamplerType::Nuts => {
                        let nuts_config = NutsConfig {
                            step_size: config.step_size,
                            target_accept: config.target_accept,
                            max_tree_depth: config.max_tree_depth,
                            num_draws: config.num_draws,
                            num_warmup: config.num_warmup,
                        };
                        nuts::run_chain_bound_unguarded(
                            &graph,
                            binding.clone(),
                            &nuts_config,
                            &mut rng,
                            Some(initial[chain_idx].clone()),
                            prog_ref,
                        )
                    }
                    SamplerType::Hmc => {
                        let hmc_config = HmcConfig {
                            step_size: config.step_size,
                            target_accept: config.target_accept,
                            num_leapfrog_steps: config.num_leapfrog_steps,
                            num_draws: config.num_draws,
                            num_warmup: config.num_warmup,
                        };
                        hmc::run_chain_bound_unguarded(
                            &graph,
                            binding.clone(),
                            &hmc_config,
                            &mut rng,
                            Some(initial[chain_idx].clone()),
                            prog_ref,
                        )
                    }
                }
            })
            .collect()
    })?;

    let transforms = &graph.param_transforms;

    // Back-transform samples from unconstrained to constrained space
    let samples: Vec<Vec<Vec<f64>>> = results
        .iter()
        .map(|r| {
            r.samples
                .iter()
                .map(|draw| {
                    draw.iter()
                        .enumerate()
                        .map(|(i, &raw)| transforms[i].apply(raw))
                        .collect()
                })
                .collect()
        })
        .collect();

    for draw in samples.iter().flatten() {
        validate_constrained_draw(draw, &param_names)?;
    }

    let accept_rates: Vec<f64> = results.iter().map(|r| r.accept_rate).collect();
    let step_sizes: Vec<f64> = results.iter().map(|r| r.step_size).collect();
    let divergences: Vec<usize> = results.iter().map(|r| r.divergences).collect();
    let transitions: Vec<Vec<TransitionStats>> =
        results.iter().map(|r| r.transitions.clone()).collect();

    let unconstrained_samples = transforms
        .iter()
        .any(|t| !matches!(t, ParamTransform::Identity))
        .then(|| Arc::new(results.into_iter().map(|r| r.samples).collect()));

    Ok(SampleResult {
        samples,
        unconstrained_samples,
        accept_rates,
        step_sizes,
        divergences,
        transitions,
        param_names,
    })
}

/// Result for a single model in a batch run, with flattened constrained draws.
#[derive(Debug, Clone)]
pub struct BatchModelResult {
    pub samples: Vec<Vec<f64>>,
    /// Exact positions in chain/draw/parameter order for transformed graphs.
    pub unconstrained_samples: Option<Arc<Vec<Vec<Vec<f64>>>>>,
    pub param_names: Vec<String>,
    pub num_chains: usize,
    pub num_draws: usize,
    pub accept_rates: Vec<f64>,
    pub step_sizes: Vec<f64>,
    pub divergences: Vec<usize>,
    pub transitions: Vec<Vec<TransitionStats>>,
}

#[derive(Debug, Clone)]
pub struct BoundBatchResult {
    pub id: String,
    pub index: usize,
    pub result: BatchModelResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSeedPolicy {
    /// Stable version-one cell IDs, invariant to ordering and chunk boundaries.
    CellIdV1,
    /// Compatibility with the original generic batch API.
    PositionV0,
}

#[derive(Debug, Clone)]
pub struct BoundBatchOptions {
    pub threads: usize,
    pub chunk_size: usize,
    pub collect_errors: bool,
    pub seed_policy: BatchSeedPolicy,
}

impl Default for BoundBatchOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            chunk_size: 64,
            collect_errors: false,
            seed_policy: BatchSeedPolicy::CellIdV1,
        }
    }
}

/// Execute generic models on the same bounded pool and cell-ID scheme as forecasts.
/// Bind errors are cell results so callers can retain successful datasets.
pub fn sample_batch_bound_with_options(
    graph: Arc<Graph>,
    bindings: Vec<(String, Result<DataBinding, String>)>,
    config: BatchSampleConfig,
    options: BoundBatchOptions,
) -> Result<Vec<Result<SampleResult, String>>, String> {
    sample_batch_bound_with_initial(
        graph,
        bindings,
        config,
        options,
        std::collections::HashMap::new(),
    )
}

/// Batch initialization is keyed by stable dataset ID. Missing IDs use zero starts;
/// invalid per-cell initial positions follow the selected error collection policy.
pub fn sample_batch_bound_with_initial(
    graph: Arc<Graph>,
    bindings: Vec<(String, Result<DataBinding, String>)>,
    config: BatchSampleConfig,
    options: BoundBatchOptions,
    initial: std::collections::HashMap<String, Vec<Vec<f64>>>,
) -> Result<Vec<Result<SampleResult, String>>, String> {
    use crate::forecast_batch::{execute_batch, execute_batch_fail_fast, BatchError};
    config.validate()?;
    let ids: std::collections::HashSet<_> = bindings.iter().map(|(id, _)| id.as_str()).collect();
    if let Some(id) = initial.keys().find(|id| !ids.contains(id.as_str())) {
        return Err(format!(
            "initialization supplied for unknown dataset ID '{id}'"
        ));
    }
    let cells: Vec<_> = bindings
        .into_iter()
        .enumerate()
        .map(|(index, (id, binding))| {
            let position = initial.get(&id).cloned();
            (id, (index, binding, position))
        })
        .collect();
    type InitializedCell = (usize, Result<DataBinding, String>, Option<Vec<Vec<f64>>>);
    let fit = |(index, binding, initial): &InitializedCell, stable_seed| {
        let seed = match options.seed_policy {
            BatchSeedPolicy::CellIdV1 => stable_seed,
            BatchSeedPolicy::PositionV0 => config.seed.wrapping_add((*index as u64) << 32),
        };
        // num_threads=0 reuses the surrounding private pool for chain work.
        sample_bound_with_init(
            Arc::clone(&graph),
            binding.clone()?,
            config.sampler_config(seed),
            initial.clone(),
        )
    };
    if options.collect_errors {
        execute_batch(
            &cells,
            config.seed,
            options.threads,
            options.chunk_size,
            fit,
        )
    } else {
        execute_batch_fail_fast(
            &cells,
            config.seed,
            options.threads,
            options.chunk_size,
            fit,
        )
        .map(|results| results.into_iter().map(Ok).collect())
        .map_err(|error| match error {
            BatchError::Configuration(error) => error,
            BatchError::Cell { id, error } => format!("dataset '{id}': {error}"),
        })
    }
}

/// Fit many validated datasets against one Arc-shared structure. Results are
/// collected in input order; IDs travel with their originating datasets.
pub fn sample_batch_bound(
    graph: Arc<Graph>,
    bindings: Vec<DataBinding>,
    config: BatchSampleConfig,
) -> Result<Vec<BoundBatchResult>, String> {
    config.validate()?;
    let sampler_config = |seed| SamplerConfig {
        sampler: config.sampler,
        num_chains: config.num_chains,
        num_draws: config.num_draws,
        num_warmup: config.num_warmup,
        step_size: config.step_size,
        target_accept: config.target_accept,
        num_leapfrog_steps: config.num_leapfrog_steps,
        max_tree_depth: config.max_tree_depth,
        seed,
        num_threads: 1,
        show_progress: false,
    };
    let outcomes = with_thread_pool(0, || {
        bindings
            .into_par_iter()
            .enumerate()
            .map(|(index, binding)| {
                let id = binding.id().to_string();
                let seed = config.seed.wrapping_add((index as u64) << 32);
                sample_bound(Arc::clone(&graph), binding, sampler_config(seed)).map(|sample| {
                    let samples = sample.samples.into_iter().flatten().collect();
                    BoundBatchResult {
                        id,
                        index,
                        result: BatchModelResult {
                            samples,
                            unconstrained_samples: sample.unconstrained_samples,
                            param_names: sample.param_names,
                            num_chains: config.num_chains,
                            num_draws: config.num_draws,
                            accept_rates: sample.accept_rates,
                            step_sizes: sample.step_sizes,
                            divergences: sample.divergences,
                            transitions: sample.transitions,
                        },
                    }
                })
            })
            .collect::<Vec<_>>()
    })?;
    outcomes.into_iter().collect()
}

impl BatchModelResult {
    pub fn mean(&self) -> Vec<f64> {
        let n_params = self.param_names.len();
        let n_draws = self.samples.len();
        let mut sums = vec![0.0; n_params];
        for draw in &self.samples {
            for (i, v) in draw.iter().enumerate() {
                sums[i] += v;
            }
        }
        sums.iter().map(|s| s / n_draws as f64).collect()
    }

    pub fn std(&self) -> Vec<f64> {
        let means = self.mean();
        let n_params = self.param_names.len();
        let n_draws = self.samples.len();
        let mut sum_sq = vec![0.0; n_params];
        for draw in &self.samples {
            for (i, v) in draw.iter().enumerate() {
                let d = v - means[i];
                sum_sq[i] += d * d;
            }
        }
        sum_sq.iter().map(|s| (s / n_draws as f64).sqrt()).collect()
    }

    pub fn quantile(&self, param_idx: usize, q: f64) -> f64 {
        let mut vals: Vec<f64> = self.samples.iter().map(|d| d[param_idx]).collect();
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (q * (vals.len() - 1) as f64) as usize;
        vals[idx.min(vals.len() - 1)]
    }

    pub fn mean_accept_rate(&self) -> f64 {
        if self.accept_rates.is_empty() {
            0.0
        } else {
            self.accept_rates.iter().sum::<f64>() / self.accept_rates.len() as f64
        }
    }

    pub fn total_divergences(&self) -> usize {
        self.divergences.iter().sum()
    }
}

/// Run many independent models in parallel through one Rayon thread pool.
///
/// Each model gets `num_chains` chains. The default remains throughput-first
/// with a single chain, but callers can trade more work for more stable
/// inference diagnostics when needed.
pub fn batch_sample(
    models: Vec<(Graph, Vec<f64>)>,
    config: BatchSampleConfig,
) -> Result<Vec<BatchModelResult>, String> {
    batch_sample_graphs(models.into_iter().map(|(graph, _)| graph).collect(), config)
}

/// Fit independent data-owning graphs. Prefer bound batches for shared structures.
pub fn batch_sample_graphs(
    models: Vec<Graph>,
    config: BatchSampleConfig,
) -> Result<Vec<BatchModelResult>, String> {
    config.validate()?;
    let n_models = models.len();
    let chains_per_model = config.num_chains;
    let total_chain_runs = n_models * chains_per_model;

    for graph in &models {
        graph.validate_shapes().map_err(|e| e.to_string())?;
        // The only gradient-based entry point that does not reach
        // `sample_bound_with_init`; it drives `run_chain` directly.
        reject_discrete_latent_parameters(graph)?;
        let binding = DataBinding::from_graph(graph).map_err(|e| e.to_string())?;
        validate_initial_target(graph, binding, &vec![0.0; graph.param_count])?;
    }

    let progress_state = if config.show_progress {
        Some(Arc::new(ProgressState::new(
            total_chain_runs,
            config.num_draws,
            config.num_warmup,
            8, // approximate leapfrog for progress display
        )))
    } else {
        None
    };

    let _progress_guard = progress_state
        .as_ref()
        .map(|ps| ProgressGuard::spawn(Arc::clone(ps)));

    let results: Vec<BatchModelResult> = with_thread_pool(0, || {
        models
            .into_par_iter()
            .enumerate()
            .map(|(model_idx, graph)| {
                let prog_ref = progress_state.as_deref();
                // Built once per model rather than once per chain. The loop
                // above already proved every graph binds, and the kernels take
                // a binding by value, so this is the same `from_graph` call
                // `run_chain` used to make internally on each pass.
                let binding = DataBinding::from_graph(&graph)
                    .expect("graph data must have consistent shapes");
                let mut samples: Vec<Vec<f64>> = Vec::new();
                let mut unconstrained_samples = graph
                    .param_transforms
                    .iter()
                    .any(|t| !matches!(t, ParamTransform::Identity))
                    .then(|| Vec::with_capacity(chains_per_model));
                let mut accept_rates = Vec::with_capacity(chains_per_model);
                let mut step_sizes = Vec::with_capacity(chains_per_model);
                let mut divergences = Vec::with_capacity(chains_per_model);
                let mut transitions = Vec::with_capacity(chains_per_model);

                for chain_idx in 0..chains_per_model {
                    let seed = config
                        .seed
                        .wrapping_add((model_idx as u64) << 32)
                        .wrapping_add(chain_idx as u64);
                    let mut rng = ChaCha8Rng::seed_from_u64(seed);

                    let chain = match config.sampler {
                        SamplerType::Nuts => {
                            let nuts_config = NutsConfig {
                                step_size: config.step_size,
                                target_accept: config.target_accept,
                                max_tree_depth: config.max_tree_depth,
                                num_draws: config.num_draws,
                                num_warmup: config.num_warmup,
                            };
                            nuts::run_chain_bound_unguarded(
                                &graph,
                                binding.clone(),
                                &nuts_config,
                                &mut rng,
                                None,
                                prog_ref,
                            )
                        }
                        SamplerType::Hmc => {
                            let hmc_config = HmcConfig {
                                step_size: config.step_size,
                                target_accept: config.target_accept,
                                num_leapfrog_steps: config.num_leapfrog_steps,
                                num_draws: config.num_draws,
                                num_warmup: config.num_warmup,
                            };
                            hmc::run_chain_bound_unguarded(
                                &graph,
                                binding.clone(),
                                &hmc_config,
                                &mut rng,
                                None,
                                prog_ref,
                            )
                        }
                    };

                    let transforms = &graph.param_transforms;
                    samples.extend(chain.samples.iter().map(|draw| {
                        draw.iter()
                            .enumerate()
                            .map(|(i, &raw)| transforms[i].apply(raw))
                            .collect()
                    }));
                    accept_rates.push(chain.accept_rate);
                    step_sizes.push(chain.step_size);
                    divergences.push(chain.divergences);
                    transitions.push(chain.transitions);
                    if let Some(raw) = &mut unconstrained_samples {
                        raw.push(chain.samples);
                    }
                }

                BatchModelResult {
                    samples,
                    unconstrained_samples: unconstrained_samples.map(Arc::new),
                    param_names: graph.param_names.clone(),
                    num_chains: chains_per_model,
                    num_draws: config.num_draws,
                    accept_rates,
                    step_sizes,
                    divergences,
                    transitions,
                }
            })
            .collect()
    })?;

    for result in &results {
        for draw in &result.samples {
            validate_constrained_draw(draw, &result.param_names)?;
        }
    }
    Ok(results)
}

fn validate_constrained_draw(draw: &[f64], names: &[String]) -> Result<(), String> {
    if let Some(index) = draw.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "sampled parameter '{}' is nonfinite after transformation",
            names[index]
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::current_num_threads;

    #[test]
    fn transformed_results_retain_exact_raw_tail_positions() {
        let mut graph = Graph::new();
        crate::distributions::BetaDist::prior(&mut graph, "p", 0.01, 0.01);
        let config = SamplerConfig {
            num_chains: 2,
            num_draws: 4,
            num_warmup: 1,
            step_size: 1e-9,
            max_tree_depth: 2,
            show_progress: false,
            ..Default::default()
        };
        let result = sample_bound_with_init(
            Arc::new(graph.structure_only()),
            DataBinding::from_graph(&graph).unwrap(),
            config.clone(),
            Some(vec![vec![40.0], vec![50.0]]),
        )
        .unwrap();
        let raw = result.unconstrained_samples.as_ref().unwrap();
        for (chain_index, chain) in raw.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(config.seed + chain_index as u64);
            let expected = nuts::run_chain(
                &graph,
                &NutsConfig {
                    step_size: config.step_size,
                    target_accept: config.target_accept,
                    max_tree_depth: config.max_tree_depth,
                    num_draws: config.num_draws,
                    num_warmup: config.num_warmup,
                },
                &mut rng,
                Some(vec![40.0 + 10.0 * chain_index as f64]),
                None,
            )
            .expect("continuous test model must run");
            assert_eq!(chain, &expected.samples);
            assert!(chain.iter().all(|q| q[0].is_finite() && q[0] > 39.0));
            assert!(result.samples[chain_index].iter().all(|q| q[0] == 1.0));
        }
        let cloned = result.clone();
        assert!(Arc::ptr_eq(
            raw,
            cloned.unconstrained_samples.as_ref().unwrap()
        ));
        let mut identity = Graph::new();
        crate::distributions::Normal::prior(&mut identity, "x", 0.0, 1.0);
        assert!(sample(identity, config)
            .unwrap()
            .unconstrained_samples
            .is_none());
    }

    #[test]
    fn batch_paths_retain_raw_chain_axes() {
        let mut graph = Graph::new();
        crate::distributions::BetaDist::prior(&mut graph, "p", 0.01, 0.01);
        let config = BatchSampleConfig {
            num_chains: 2,
            num_draws: 20,
            num_warmup: 50,
            show_progress: false,
            ..Default::default()
        };
        let legacy = batch_sample_graphs(vec![graph.clone()], config.clone()).unwrap();
        let bound = sample_batch_bound(
            Arc::new(graph.structure_only()),
            vec![DataBinding::from_graph(&graph).unwrap()],
            config,
        )
        .unwrap();
        let raw = legacy[0].unconstrained_samples.as_ref().unwrap();
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0].len(), 20);
        assert_eq!(raw, bound[0].result.unconstrained_samples.as_ref().unwrap());
        for (position, constrained) in raw.iter().flatten().zip(&legacy[0].samples) {
            assert_eq!(graph.param_transforms[0].apply(position[0]), constrained[0]);
        }
    }

    #[test]
    fn thread_pool_helper_uses_requested_parallelism() {
        let one = with_thread_pool(1, current_num_threads).unwrap();
        let two = with_thread_pool(2, current_num_threads).unwrap();

        assert_eq!(one, 1);
        assert_eq!(two, 2);
    }

    #[test]
    fn invalid_sampling_controls_are_rejected_in_rust() {
        let mut graph = Graph::new();
        crate::distributions::Normal::prior(&mut graph, "x", 0.0, 1.0);
        let base = SamplerConfig {
            show_progress: false,
            ..Default::default()
        };
        let invalid = [
            SamplerConfig {
                num_chains: 0,
                ..base.clone()
            },
            SamplerConfig {
                num_draws: 0,
                ..base.clone()
            },
            SamplerConfig {
                num_leapfrog_steps: 0,
                ..base.clone()
            },
            SamplerConfig {
                max_tree_depth: 64,
                ..base.clone()
            },
            SamplerConfig {
                num_draws: usize::MAX,
                ..base.clone()
            },
            SamplerConfig {
                step_size: f64::NAN,
                ..base.clone()
            },
        ];
        for config in invalid {
            assert!(sample(graph.clone(), config).is_err());
        }
        let binding = DataBinding::from_graph(&graph).unwrap();
        let structure = Arc::new(graph.structure_only());
        assert!(sample_bound_with_init(
            structure.clone(),
            binding.clone(),
            base.clone(),
            Some(vec![vec![0.0]])
        )
        .is_err());
        assert!(sample_bound_with_init(
            structure,
            binding,
            base,
            Some(vec![vec![f64::INFINITY]; 4])
        )
        .is_err());
    }

    #[test]
    fn sample_result_exposes_transition_diagnostics() {
        let result = SampleResult {
            samples: vec![],
            unconstrained_samples: None,
            accept_rates: vec![0.8],
            step_sizes: vec![0.1],
            divergences: vec![1],
            transitions: vec![vec![
                TransitionStats {
                    is_warmup: true,
                    accepted: true,
                    accept_prob: 0.9,
                    energy_error: 0.2,
                    divergent: false,
                    step_size: 0.1,
                    num_leapfrog_steps: 4,
                    tree_depth: Some(2),
                },
                TransitionStats {
                    is_warmup: false,
                    accepted: false,
                    accept_prob: 0.7,
                    energy_error: -0.3,
                    divergent: true,
                    step_size: 0.1,
                    num_leapfrog_steps: 4,
                    tree_depth: Some(2),
                },
            ]],
            param_names: vec!["x".to_string()],
        };

        let report = result.transition_diagnostics();
        assert_eq!(report.total_transitions, 2);
        assert_eq!(report.total_warmup_transitions, 1);
        assert_eq!(report.total_divergences, 1);
        assert_eq!(report.total_leapfrog_steps, 8);
        assert_eq!(report.chains.len(), 1);
        assert!(report.mean_accept_prob > 0.0);
    }

    #[test]
    fn sampling_rejects_a_non_finite_default_initial_target() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let sigma = graph.add_param("sigma");
        let zero = graph.add_constant(0.0);
        graph.normal_logp(x, zero, sigma);

        let error = sample(
            graph,
            SamplerConfig {
                num_chains: 1,
                num_draws: 1,
                num_warmup: 1,
                show_progress: false,
                ..SamplerConfig::default()
            },
        )
        .unwrap_err();
        assert!(error.contains("initial log density is not finite"));
    }

    #[test]
    fn sampling_rejects_invalid_target_accept() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, one);

        let error = sample(
            graph,
            SamplerConfig {
                target_accept: 1.0,
                num_chains: 1,
                num_draws: 1,
                num_warmup: 1,
                show_progress: false,
                ..SamplerConfig::default()
            },
        )
        .unwrap_err();
        assert!(error.contains("target_accept"));
    }
}
