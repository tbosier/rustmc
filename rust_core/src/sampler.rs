use crate::autodiff::Evaluator;
use crate::data::DataBinding;
use crate::diagnostics::{self, DiagnosticsReport};
use crate::graph::{Graph, Op, ParamTransform};
use crate::hmc::{self, ChainResult, HmcConfig, TransitionStats};
pub use crate::mass_matrix::MetricKind;
use crate::nuts::{self, NutsConfig};
use crate::progress::{ProgressGuard, ProgressState};
use crate::seeding::{chain_seed, SAMPLER_FIT_SEED_DOMAIN, SAMPLER_INIT_SEED_DOMAIN};
use rand::{Rng, SeedableRng};
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
    /// How warmup estimates the metric of vector parameters; see
    /// [`MetricKind`].
    pub metric: MetricKind,
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
            metric: MetricKind::Auto,
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
    /// How warmup estimates the metric of vector parameters.
    pub metric: MetricKind,
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
            metric: MetricKind::Auto,
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
            metric: self.metric,
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

/// Check caller-supplied starting points: one finite vector per chain.
pub(crate) fn validate_initial_values(
    initial: Option<Vec<Vec<f64>>>,
    chains: usize,
    dimension: usize,
) -> Result<Option<Vec<Vec<f64>>>, String> {
    let Some(positions) = initial else {
        return Ok(None);
    };
    if positions.len() != chains
        || positions
            .iter()
            .any(|q| q.len() != dimension || q.iter().any(|x| !x.is_finite()))
    {
        return Err("init must contain one finite unconstrained parameter vector per chain".into());
    }
    Ok(Some(positions))
}

/// Half-width of the box an unsupplied start is drawn from, in unconstrained
/// coordinates: Stan's `init_radius`.
const RANDOM_INIT_RADIUS: f64 = 2.0;
/// Draws tried before an unsupplied start falls back to the origin, as many as
/// Stan tries.
const RANDOM_INIT_ATTEMPTS: usize = 100;

/// Find a start for `chain` when the caller supplied none.
///
/// Each coordinate is drawn uniformly from (-2, 2) on the unconstrained scale,
/// Stan's convention, from a stream of its own so the draws do not depend on
/// or disturb the chain's sampling stream. Starting every chain at the origin
/// made them agree before they had explored anything, which is exactly what
/// R-hat needs them not to do. A draw is kept when `usable` accepts it (the
/// density and gradient are finite there); after
/// [`RANDOM_INIT_ATTEMPTS`] refusals the origin is tried, and if that is
/// refused too the fit fails with an error asking for `init`.
///
/// `usable` returns `Err` for an evaluation failure, which ends the search:
/// that is not something another random point can fix.
pub(crate) fn random_initial_position(
    seed: u64,
    chain: usize,
    dimension: usize,
    mut usable: impl FnMut(&[f64]) -> Result<bool, String>,
) -> Result<Vec<f64>, String> {
    let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(seed, chain, SAMPLER_INIT_SEED_DOMAIN));
    let mut position = vec![0.0; dimension];
    for _ in 0..RANDOM_INIT_ATTEMPTS {
        for value in position.iter_mut() {
            *value = rng.gen_range(-RANDOM_INIT_RADIUS..RANDOM_INIT_RADIUS);
        }
        if usable(&position)? {
            return Ok(position);
        }
    }
    position.fill(0.0);
    if usable(&position)? {
        return Ok(position);
    }
    Err(format!(
        "chain {chain}: no initial point with a finite log density and gradient was found \
         ({RANDOM_INIT_ATTEMPTS} uniform draws on (-{RANDOM_INIT_RADIUS}, {RANDOM_INIT_RADIUS}) \
         in unconstrained space, then the origin); pass init= with a valid starting point"
    ))
}

/// The RNG for fitting chain `chain` of a fit seeded with `seed`.
///
/// Chains are keyed through [`chain_seed`] rather than `seed + chain`, so a fit
/// seeded 42 does not share chain 1's stream with chain 0 of a fit seeded 43.
pub(crate) fn chain_rng(seed: u64, chain: usize) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(chain_seed(seed, chain, SAMPLER_FIT_SEED_DOMAIN))
}

/// Starting point for a raw kernel: the caller's vector, checked, or the
/// origin.
pub(crate) fn kernel_initial_position(
    init: Option<Vec<f64>>,
    dimension: usize,
) -> Result<Vec<f64>, String> {
    match init {
        None => Ok(vec![0.0; dimension]),
        Some(position) if position.len() == dimension && position.iter().all(|x| x.is_finite()) => {
            Ok(position)
        }
        Some(position) if position.len() != dimension => Err(format!(
            "init must be a finite unconstrained vector of length {dimension}, got length {}",
            position.len()
        )),
        Some(_) => Err("init must contain only finite unconstrained values".to_string()),
    }
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
    /// Posterior mean per parameter, from the same implementation the summary
    /// table uses.
    ///
    /// See [`diagnostics::scaled_moments`]: the draws are centred and scaled
    /// before they are summed, so a posterior whose draws sit near the top of
    /// the representable range reports a finite mean rather than an infinity,
    /// and reports the same one `diagnostics()` does.
    pub fn mean(&self) -> Vec<f64> {
        let n_params = self.param_names.len();
        if !self.is_rectangular() {
            return vec![f64::NAN; n_params];
        }
        (0..n_params)
            .map(|i| diagnostics::scaled_moments(|| self.draws_of(i)).0)
            .collect()
    }

    /// Posterior standard deviation per parameter, from the same implementation
    /// the summary table uses.
    ///
    /// This is the sample standard deviation (`n - 1` in the denominator), as
    /// the summary table has always reported; before the two paths were shared
    /// this one divided by `n` and the two disagreed by `sqrt(n / (n - 1))`.
    pub fn std(&self) -> Vec<f64> {
        let n_params = self.param_names.len();
        if !self.is_rectangular() {
            return vec![f64::NAN; n_params];
        }
        (0..n_params)
            .map(|i| diagnostics::scaled_moments(|| self.draws_of(i)).1)
            .collect()
    }

    /// Whether `samples` is the rectangular chain × draw × parameter array the
    /// sampler produces.
    ///
    /// These fields are public, so a caller can assemble one that is not.
    /// `compute_diagnostics` refuses such an array outright — a ragged one has
    /// no chain axis to compute R-hat along — and reports NaN for every
    /// parameter; the moments agree with it rather than indexing past the end
    /// of a short draw or averaging different parameters over different numbers
    /// of draws.
    fn is_rectangular(&self) -> bool {
        let Some(first) = self.samples.first() else {
            return false;
        };
        let n_draws = first.len();
        let n_params = self.param_names.len();
        n_draws > 0
            && self.samples.iter().all(|chain| {
                chain.len() == n_draws && chain.iter().all(|draw| draw.len() == n_params)
            })
    }

    /// Every draw of parameter `index`, chain-major. Requires
    /// [`Self::is_rectangular`].
    ///
    /// The order is the order the naive loop accumulated in, and it is the
    /// order `BatchModelResult` accumulates in, so the two keep reporting the
    /// same value for the same draws.
    fn draws_of(&self, index: usize) -> impl Iterator<Item = f64> + '_ {
        self.samples
            .iter()
            .flat_map(move |chain| chain.iter().map(move |draw| draw[index]))
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
/// Both densities now check their support — `bernoulli_logp_scalar` returns
/// -inf off `{0, 1}` and `count_sampling::log_mass` returns -inf for a
/// fractional count — so every off-integer proposal is rejected and the chain
/// is pinned to its integer initialization, reporting divergence on every
/// transition while the step size collapses. Not a wrong number, but no number
/// at all, and no support error either: without this check the run looks like
/// an ordinary fit that simply mixed badly.
///
/// Before the support check landed, Bernoulli failed worse than that rather
/// than better: `x * ln p + (1 - x) * ln(1 - p)` is constant over all of R at
/// `p = 0.5`, so the chain random-walked a flat direction and returned
/// fractional "draws" for a parameter whose support is `{0, 1}`, reaching -813
/// within 20 draws.
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
/// `None` draws each chain's start uniformly from (-2, 2) in unconstrained
/// space, retrying until the density and gradient are finite there.
pub fn sample_bound_with_init(
    graph: Arc<Graph>,
    binding: DataBinding,
    config: SamplerConfig,
    initial: Option<Vec<Vec<f64>>>,
) -> Result<SampleResult, String> {
    config.validate()?;
    // Single chokepoint: `sample`, `sample_bound`, the bound-batch entry point
    // and `model::GraphModel::sample` all funnel through here.
    reject_discrete_latent_parameters(&graph)?;
    let initial = validate_initial_values(initial, config.num_chains, graph.param_count)?;
    binding.validate_for(&graph).map_err(|e| e.to_string())?;
    if let Some(positions) = &initial {
        for position in positions {
            validate_initial_target(&graph, binding.clone(), position)?;
        }
    }

    let progress_state = config.show_progress.then(|| {
        // For the progress bar, leapfrog count is approximate for NUTS.
        let approx_leapfrog = match config.sampler {
            SamplerType::Hmc => config.num_leapfrog_steps,
            SamplerType::Nuts => 1 << (config.max_tree_depth / 2),
        };
        Arc::new(ProgressState::new(
            config.num_chains,
            config.num_draws,
            config.num_warmup,
            approx_leapfrog,
        ))
    });
    let _progress_guard = progress_state
        .as_ref()
        .map(|ps| ProgressGuard::spawn(Arc::clone(ps)));

    let results = with_thread_pool(config.num_threads, || {
        run_chains(
            &graph,
            &binding,
            &config,
            initial.as_deref(),
            progress_state.as_deref(),
        )
    })??;
    let (samples, unconstrained_samples) = constrain_chains(&graph, &results)?;
    Ok(SampleResult {
        samples,
        unconstrained_samples,
        accept_rates: results.iter().map(|r| r.accept_rate).collect(),
        step_sizes: results.iter().map(|r| r.step_size).collect(),
        divergences: results.iter().map(|r| r.divergences).collect(),
        transitions: results.into_iter().map(|r| r.transitions).collect(),
        param_names: graph.param_names.clone(),
    })
}

/// Run every chain of one fit on the current Rayon pool.
///
/// Chain `c` samples from [`chain_rng`]`(config.seed, c)` and starts at
/// `initial[c]`, or at a [`random_initial_position`] when none was supplied.
fn run_chains(
    graph: &Graph,
    binding: &DataBinding,
    config: &SamplerConfig,
    initial: Option<&[Vec<f64>]>,
    progress: Option<&ProgressState>,
) -> Result<Vec<ChainResult>, String> {
    (0..config.num_chains)
        .into_par_iter()
        .map(|chain| {
            let position = match initial {
                Some(positions) => positions[chain].clone(),
                None => random_graph_initial_position(graph, binding, config.seed, chain)?,
            };
            let mut rng = chain_rng(config.seed, chain);
            match config.sampler {
                SamplerType::Nuts => nuts::run_chain_bound_unguarded(
                    graph,
                    binding.clone(),
                    &NutsConfig {
                        step_size: config.step_size,
                        target_accept: config.target_accept,
                        max_tree_depth: config.max_tree_depth,
                        num_draws: config.num_draws,
                        num_warmup: config.num_warmup,
                        metric: config.metric,
                    },
                    &mut rng,
                    Some(position),
                    progress,
                ),
                SamplerType::Hmc => hmc::run_chain_bound_unguarded(
                    graph,
                    binding.clone(),
                    &HmcConfig {
                        step_size: config.step_size,
                        target_accept: config.target_accept,
                        num_leapfrog_steps: config.num_leapfrog_steps,
                        num_draws: config.num_draws,
                        num_warmup: config.num_warmup,
                        metric: config.metric,
                    },
                    &mut rng,
                    Some(position),
                    progress,
                ),
            }
        })
        .collect()
}

fn random_graph_initial_position(
    graph: &Graph,
    binding: &DataBinding,
    seed: u64,
    chain: usize,
) -> Result<Vec<f64>, String> {
    let mut evaluator =
        Evaluator::try_with_binding(graph, binding.clone()).map_err(|error| error.to_string())?;
    random_initial_position(seed, chain, graph.param_count, |position| {
        evaluator.compute(graph, position);
        Ok(evaluator.total_logp.is_finite() && evaluator.grad.iter().all(|g| g.is_finite()))
    })
}

type ChainDraws = Vec<Vec<Vec<f64>>>;

/// Back-transform every chain's draws to the constrained scale, keeping the
/// raw positions too when any parameter is transformed.
fn constrain_chains(
    graph: &Graph,
    results: &[ChainResult],
) -> Result<(ChainDraws, Option<Arc<ChainDraws>>), String> {
    let transforms = &graph.param_transforms;
    let samples: ChainDraws = results
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
        validate_constrained_draw(draw, &graph.param_names)?;
    }
    let unconstrained = transforms
        .iter()
        .any(|t| !matches!(t, ParamTransform::Identity))
        .then(|| Arc::new(results.iter().map(|r| r.samples.clone()).collect()));
    Ok((samples, unconstrained))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSeedPolicy {
    /// Stable version-one cell IDs, invariant to ordering and chunk boundaries.
    CellIdV1,
    /// Dataset `i` is fitted with seed `seed + (i << 32)`, the positional
    /// scheme of the original generic batch API. Only the cell seed is
    /// preserved: the chains below it are keyed and started as in any other
    /// fit, so draws from releases that seeded chain `c` as `seed + c` and
    /// started every chain at the origin are not reproduced.
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
///
/// Initialization is keyed by stable dataset ID; a dataset without one draws
/// random starts like [`sample_bound_with_init`]. Invalid per-cell initial
/// positions follow the selected error collection policy.
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
        // The policy chooses the cell's seed; its chains are then keyed from
        // that seed exactly as a single fit's are.
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

impl BatchModelResult {
    /// Posterior mean per parameter; see [`SampleResult::mean`].
    pub fn mean(&self) -> Vec<f64> {
        let n_params = self.param_names.len();
        if !self.is_rectangular() {
            return vec![f64::NAN; n_params];
        }
        (0..n_params)
            .map(|i| diagnostics::scaled_moments(|| self.draws_of(i)).0)
            .collect()
    }

    /// Posterior standard deviation per parameter; see [`SampleResult::std`].
    pub fn std(&self) -> Vec<f64> {
        let n_params = self.param_names.len();
        if !self.is_rectangular() {
            return vec![f64::NAN; n_params];
        }
        (0..n_params)
            .map(|i| diagnostics::scaled_moments(|| self.draws_of(i)).1)
            .collect()
    }

    /// Whether every draw carries every parameter; see
    /// [`SampleResult::is_rectangular`].
    fn is_rectangular(&self) -> bool {
        !self.samples.is_empty()
            && self
                .samples
                .iter()
                .all(|draw| draw.len() == self.param_names.len())
    }

    /// Every draw of parameter `index`. `samples` is already chain-major, so
    /// this is the same sequence `SampleResult::draws_of` yields.
    fn draws_of(&self, index: usize) -> impl Iterator<Item = f64> + '_ {
        self.samples.iter().map(move |draw| draw[index])
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

/// Fit independent data-owning graphs, one per entry, through one Rayon pool.
///
/// Each model gets `num_chains` chains, seeded as a single fit seeded
/// `config.seed + (model_index << 32)` would be, and starting from random
/// initial points. Prefer [`sample_batch_bound_with_initial`] when the models
/// share one structure.
pub fn batch_sample_graphs(
    models: Vec<Graph>,
    config: BatchSampleConfig,
) -> Result<Vec<BatchModelResult>, String> {
    config.validate()?;
    let mut bindings = Vec::with_capacity(models.len());
    for graph in &models {
        graph.validate_shapes().map_err(|e| e.to_string())?;
        reject_discrete_latent_parameters(graph)?;
        bindings.push(DataBinding::from_graph(graph).map_err(|e| e.to_string())?);
    }

    let progress_state = config.show_progress.then(|| {
        Arc::new(ProgressState::new(
            models.len() * config.num_chains,
            config.num_draws,
            config.num_warmup,
            8, // approximate leapfrog for progress display
        ))
    });
    let _progress_guard = progress_state
        .as_ref()
        .map(|ps| ProgressGuard::spawn(Arc::clone(ps)));

    with_thread_pool(0, || {
        models
            .into_par_iter()
            .zip(bindings)
            .enumerate()
            .map(|(model_index, (graph, binding))| {
                let seed = config.seed.wrapping_add((model_index as u64) << 32);
                let chains = run_chains(
                    &graph,
                    &binding,
                    &config.sampler_config(seed),
                    None,
                    progress_state.as_deref(),
                )?;
                let (samples, unconstrained_samples) = constrain_chains(&graph, &chains)?;
                Ok(BatchModelResult {
                    samples: samples.into_iter().flatten().collect(),
                    unconstrained_samples,
                    param_names: graph.param_names.clone(),
                    num_chains: config.num_chains,
                    num_draws: config.num_draws,
                    accept_rates: chains.iter().map(|c| c.accept_rate).collect(),
                    step_sizes: chains.iter().map(|c| c.step_size).collect(),
                    divergences: chains.iter().map(|c| c.divergences).collect(),
                    transitions: chains.into_iter().map(|c| c.transitions).collect(),
                })
            })
            .collect()
    })?
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
            let mut rng = chain_rng(config.seed, chain_index);
            let expected = nuts::run_chain(
                &graph,
                &NutsConfig {
                    step_size: config.step_size,
                    target_accept: config.target_accept,
                    max_tree_depth: config.max_tree_depth,
                    num_draws: config.num_draws,
                    num_warmup: config.num_warmup,
                    metric: config.metric,
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
        // The position policy seeds dataset `i` as `seed + (i << 32)`, the
        // same seed the per-graph batch gives model `i`.
        let bound = sample_batch_bound_with_initial(
            Arc::new(graph.structure_only()),
            vec![("0".into(), Ok(DataBinding::from_graph(&graph).unwrap()))],
            config,
            BoundBatchOptions {
                seed_policy: BatchSeedPolicy::PositionV0,
                ..Default::default()
            },
            Default::default(),
        )
        .unwrap();
        let bound = bound[0].as_ref().unwrap();
        let raw = legacy[0].unconstrained_samples.as_ref().unwrap();
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0].len(), 20);
        assert_eq!(raw, bound.unconstrained_samples.as_ref().unwrap());
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
    fn sampling_reports_when_no_random_start_is_usable() {
        // A negative scale makes the density -inf at every point, so every
        // random draw and then the origin are refused.
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let negative = graph.add_constant(-1.0);
        graph.normal_logp(x, zero, negative);

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
        assert!(error.contains("no initial point"), "{error}");
        assert!(error.contains("init="), "{error}");
    }

    #[test]
    fn supplied_non_finite_start_is_still_rejected() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let sigma = graph.add_param("sigma");
        let zero = graph.add_constant(0.0);
        graph.normal_logp(x, zero, sigma);
        let binding = DataBinding::from_graph(&graph).unwrap();
        let error = sample_bound_with_init(
            Arc::new(graph.structure_only()),
            binding,
            SamplerConfig {
                num_chains: 1,
                num_draws: 1,
                num_warmup: 1,
                show_progress: false,
                ..SamplerConfig::default()
            },
            Some(vec![vec![0.0, 0.0]]),
        )
        .unwrap_err();
        assert!(error.contains("initial log density is not finite"));
    }

    #[test]
    fn unsupplied_starts_differ_across_chains_and_avoid_bad_regions() {
        // `sigma` is an unconstrained raw parameter here, so half of the box
        // (-2, 2) is outside the support; the search must skip it.
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let sigma = graph.add_param("sigma");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, sigma);
        graph.normal_logp(sigma, one, one);
        let binding = DataBinding::from_graph(&graph).unwrap();
        let starts: Vec<Vec<f64>> = (0..4)
            .map(|chain| random_graph_initial_position(&graph, &binding, 42, chain).unwrap())
            .collect();
        for (chain, start) in starts.iter().enumerate() {
            assert!(start.iter().all(|v| v.abs() < 2.0), "{start:?}");
            assert!(start[1] > 0.0, "chain {chain} started outside the support");
            for other in &starts[..chain] {
                assert_ne!(start, other);
            }
        }
        // Reproducible, and independent of the sampling stream.
        assert_eq!(
            starts[1],
            random_graph_initial_position(&graph, &binding, 42, 1).unwrap()
        );
    }

    #[test]
    fn adjacent_seeds_do_not_reproduce_each_others_chains() {
        let mut graph = Graph::new();
        crate::distributions::Normal::prior(&mut graph, "x", 0.0, 1.0);
        let config = |seed| SamplerConfig {
            num_chains: 2,
            num_draws: 50,
            num_warmup: 50,
            seed,
            show_progress: false,
            ..SamplerConfig::default()
        };
        let first = sample(graph.clone(), config(42)).unwrap();
        let second = sample(graph, config(43)).unwrap();
        for chain in &first.samples {
            for other in &second.samples {
                assert_ne!(chain, other);
            }
        }
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
