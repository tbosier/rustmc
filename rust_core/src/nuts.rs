//! No-U-Turn Sampler (NUTS) — Hoffman & Gelman (2014) with multinomial
//! sampling (Betancourt 2017).
//!
//! This follows the core NUTS design used by PyMC and Stan:
//!   - Iterative tree doubling (extend trajectory forward or backward)
//!   - Endpoint-momentum U-turn checks on subtrees
//!   - Multinomial candidate selection weighted by exp(-H)
//!   - Divergence detection via energy error threshold
//!   - Max tree depth cap (default 10)

use crate::adaptation::WarmupAdapter;
use crate::autodiff::Evaluator;
use crate::data::DataBinding;
use crate::graph::Graph;
use crate::hmc::{acceptance_probability, ChainResult, TransitionStats, MAX_DELTA_H};
use crate::mass_matrix::{MassMatrix, MetricKind};
use crate::progress::ProgressState;
use crate::sampler::{kernel_initial_position, reject_discrete_latent_parameters};
use crate::target::GradientEvaluator;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct NutsConfig {
    pub step_size: f64,
    /// Desired average Metropolis acceptance probability during adaptation.
    pub target_accept: f64,
    pub max_tree_depth: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    /// How warmup estimates the metric of vector parameters.
    pub metric: MetricKind,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            step_size: 0.0,
            target_accept: 0.80,
            max_tree_depth: 10,
            num_draws: 1000,
            num_warmup: 500,
            metric: MetricKind::Auto,
        }
    }
}

/// A point on the Hamiltonian trajectory: (position, momentum, gradient, log-probability).
#[derive(Clone)]
struct PhasePoint {
    q: Vec<f64>,
    p: Vec<f64>,
    grad: Vec<f64>,
    logp: f64,
}

impl PhasePoint {
    fn energy(&self, mass: &MassMatrix, scratch: &mut [f64]) -> f64 {
        let ke = mass.kinetic_energy(&self.p, scratch);
        -self.logp + ke
    }
}

/// Result of building one subtree during the doubling process.
struct TreeResult {
    /// Leftmost point of the subtree.
    left: PhasePoint,
    /// Rightmost point of the subtree.
    right: PhasePoint,
    /// The candidate sample (multinomial-selected from valid leaves).
    proposal: PhasePoint,
    /// Log of the sum of weights (for multinomial combining).
    log_sum_weight: f64,
    /// Number of leapfrog steps taken.
    n_leapfrog: usize,
    /// Whether a U-turn was detected inside this subtree.
    turning: bool,
    /// Whether a divergence was detected.
    diverging: bool,
    /// Sum of the leafwise Metropolis acceptance probabilities.
    sum_accept_prob: f64,
    /// Number of leapfrog leaves contributing to `sum_accept_prob`.
    n_accept_prob: usize,
}

/// Run a single NUTS chain with windowed block-structured mass matrix adaptation.
///
/// Warmup uses a windowed schedule after Stan's, identical to it from 500
/// warmup iterations (see `adaptation::WarmupSchedule` for where and why the
/// two differ below that):
///   Init buffer (75 draws; 15% of warmup when that is fewer):
///       step-size dual-averaging only, identity mass matrix.
///   Mass-matrix windows (doubling: 25 → 50 → 100 → …; a window whose
///   successor would not fit is extended to the terminal buffer):
///       At the end of each window the block-structured mass matrix is updated
///       from Welford estimates collected in that window, the step size is
///       searched again under the new metric and dual averaging restarts.
///   Terminal buffer (50 draws; 10% of warmup when that is fewer, but never
///   fewer than 25):
///       step-size dual-averaging only, final fixed mass matrix.
/// Warmups too short for a 10-draw window adapt only the step size.
///
/// # Errors
///
/// Returns the rejection message from
/// [`crate::sampler::reject_discrete_latent_parameters`] if `graph` carries a
/// discrete latent. This entry point does not go through `sampler`, so the
/// check has to happen here or not at all — see [`run_chain_bound`]. Also
/// returns an error when the graph's data do not bind, or when `init` is not a
/// finite vector with one entry per parameter. `None` starts at the origin.
pub fn run_chain(
    graph: &Graph,
    config: &NutsConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
    progress: Option<&ProgressState>,
) -> Result<ChainResult, String> {
    reject_discrete_latent_parameters(graph)?;
    let binding = DataBinding::from_graph(graph).map_err(|error| error.to_string())?;
    run_chain_bound_unguarded(graph, binding, config, rng, init, progress)
}

/// Run a chain against a validated dataset without embedding it in `Graph`.
///
/// # Errors
///
/// As [`run_chain`]. These kernels are `pub`, so a caller can reach them
/// without passing through any `sampler` entry point; refusing a model the
/// gradient-based samplers cannot evaluate needs an error channel here, which
/// is why both return a `Result` rather than a bare [`ChainResult`].
pub fn run_chain_bound(
    graph: &Graph,
    binding: DataBinding,
    config: &NutsConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
    progress: Option<&ProgressState>,
) -> Result<ChainResult, String> {
    reject_discrete_latent_parameters(graph)?;
    run_chain_bound_unguarded(graph, binding, config, rng, init, progress)
}

/// [`run_chain_bound`] without the discrete-latent check.
///
/// For callers inside `sampler`, which run the check once at their own
/// boundary and would otherwise repeat a whole-graph scan per chain.
pub(crate) fn run_chain_bound_unguarded(
    graph: &Graph,
    binding: DataBinding,
    config: &NutsConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
    progress: Option<&ProgressState>,
) -> Result<ChainResult, String> {
    let mut evaluator =
        Evaluator::try_with_binding(graph, binding).map_err(|error| error.to_string())?;
    let position = kernel_initial_position(init, graph.param_count)?;
    Ok(run_chain_with_evaluator(
        graph,
        config,
        rng,
        position,
        progress,
        &mut evaluator,
    ))
}

pub(crate) fn run_chain_with_evaluator(
    graph: &Graph,
    config: &NutsConfig,
    rng: &mut ChaCha8Rng,
    init: Vec<f64>,
    progress: Option<&ProgressState>,
    evaluator: &mut impl GradientEvaluator,
) -> ChainResult {
    let dim = graph.param_count;
    let total_iters = config.num_warmup + config.num_draws;

    let q = init;
    let mut samples = Vec::with_capacity(config.num_draws);
    let mut transitions = Vec::with_capacity(total_iters);
    let mut n_divergences = 0usize;
    let mut sum_accept_prob = 0.0f64;
    let mut total_iters_done = 0u64;

    let mut mass = MassMatrix::from_graph(graph);
    let mut scratch = vec![0.0f64; dim];
    let mut pool = PointPool::new(dim);

    // Windowed warmup: step-size-only buffers around doubling
    // metric-estimation windows; see `adaptation::WarmupSchedule`.
    let mut adapter = WarmupAdapter::new(
        graph,
        config.num_warmup,
        config.target_accept,
        config.step_size,
        config.metric,
    );
    let mut step_size = if config.step_size > 0.0 {
        config.step_size
    } else {
        adapter.initial_step_size(graph, evaluator, &q, &mass, rng, &mut scratch)
    };

    // Compute initial state
    evaluator.compute(graph, &q);
    let mut current = PhasePoint {
        q: q.clone(),
        p: vec![0.0; dim],
        grad: evaluator.gradient().to_vec(),
        logp: evaluator.log_density(),
    };

    for iter in 0..total_iters {
        if evaluator.has_failed() {
            break;
        }
        let is_warmup = iter < config.num_warmup;
        let step_size_used = step_size;

        mass.sample_momentum_into(rng, &mut current.p, &mut scratch);

        let h0 = current.energy(&mass, &mut scratch);

        // Build the NUTS tree
        let (proposal, tree_stats) = build_tree_iterative(
            graph,
            evaluator,
            &current,
            step_size,
            &mass,
            h0,
            config.max_tree_depth,
            rng,
            &mut scratch,
            &mut pool,
        );

        if evaluator.has_failed() {
            break;
        }

        // Multinomial weighting handles candidate selection internally.  A
        // divergence terminates trajectory construction, but does not
        // invalidate a candidate selected from the valid trajectory prefix.
        update_current(&mut current, &proposal);
        pool.give(proposal);

        let accept_stat = tree_stats.mean_accept_prob;
        // Retain warmup telemetry, but report posterior-draw diagnostics only.
        if !is_warmup {
            if tree_stats.diverging {
                n_divergences += 1;
            }
            sum_accept_prob += accept_stat;
            total_iters_done += 1;
        }

        if let Some(p) = progress {
            p.increment();
            if !is_warmup && tree_stats.diverging {
                p.add_divergence();
            }
        }

        if is_warmup {
            step_size = adapter.after_transition(
                iter,
                accept_stat,
                &current.q,
                graph,
                evaluator,
                &mut mass,
                rng,
                &mut scratch,
            );
        }

        if !is_warmup {
            samples.push(current.q.clone());
        }

        transitions.push(TransitionStats {
            is_warmup,
            accepted: !tree_stats.diverging,
            accept_prob: accept_stat,
            energy_error: tree_stats.energy_error,
            divergent: tree_stats.diverging,
            step_size: step_size_used,
            num_leapfrog_steps: tree_stats.n_leapfrog,
            tree_depth: Some(tree_stats.tree_depth),
        });
    }

    let accept_rate = if total_iters_done > 0 {
        sum_accept_prob / total_iters_done as f64
    } else {
        0.0
    };

    ChainResult {
        samples,
        accept_rate,
        step_size,
        divergences: n_divergences,
        transitions,
    }
}

struct TreeStats {
    diverging: bool,
    mean_accept_prob: f64,
    energy_error: f64,
    tree_depth: usize,
    n_leapfrog: usize,
}

fn update_current(current: &mut PhasePoint, proposal: &PhasePoint) {
    current.q.copy_from_slice(&proposal.q);
    current.grad.copy_from_slice(&proposal.grad);
    current.logp = proposal.logp;
}

/// Recycled phase points for tree construction.
///
/// A trajectory of depth `j` holds a bounded number of live points (the
/// endpoints and proposal of each subtree on the recursion stack), so once
/// the pool has grown to that size tree building stops allocating: every
/// point a merge discards goes back here and is overwritten by the next
/// leapfrog step.
struct PointPool {
    dim: usize,
    free: Vec<PhasePoint>,
}

impl PointPool {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            free: Vec::new(),
        }
    }

    fn take(&mut self) -> PhasePoint {
        self.free.pop().unwrap_or_else(|| PhasePoint {
            q: vec![0.0; self.dim],
            p: vec![0.0; self.dim],
            grad: vec![0.0; self.dim],
            logp: 0.0,
        })
    }

    fn copy_of(&mut self, source: &PhasePoint) -> PhasePoint {
        let mut point = self.take();
        point.q.copy_from_slice(&source.q);
        point.p.copy_from_slice(&source.p);
        point.grad.copy_from_slice(&source.grad);
        point.logp = source.logp;
        point
    }

    fn give(&mut self, point: PhasePoint) {
        self.free.push(point);
    }

    fn give_tree(&mut self, tree: TreeResult) {
        self.give(tree.left);
        self.give(tree.right);
        self.give(tree.proposal);
    }
}

/// Build the NUTS tree iteratively by doubling depth.
///
/// At each depth j, the tree has 2^j leaves. We randomly choose to extend
/// the trajectory forward (+ε) or backward (-ε). After extending, we check
/// the endpoint-momentum U-turn criterion across the full tree. If a U-turn is
/// detected or a divergence occurs, we stop and return the current candidate.
///
/// The returned proposal is a pooled point; the caller hands it back to
/// `pool` once it has copied what it needs.
// NUTS tree construction passes explicit state and reusable buffers on its hot path.
#[allow(clippy::too_many_arguments)]
fn build_tree_iterative(
    graph: &Graph,
    evaluator: &mut impl GradientEvaluator,
    initial: &PhasePoint,
    eps: f64,
    mass: &MassMatrix,
    h0: f64,
    max_depth: usize,
    rng: &mut ChaCha8Rng,
    scratch: &mut [f64],
    pool: &mut PointPool,
) -> (PhasePoint, TreeStats) {
    let mut left = pool.copy_of(initial);
    let mut right = pool.copy_of(initial);
    let mut proposal = pool.copy_of(initial);
    let mut log_sum_weight = 0.0f64; // log(exp(-H(initial))) normalized
    let mut depth = 0;
    let mut n_leapfrog_total = 0;
    let mut sum_accept_stat = 0.0f64;
    let mut n_accept_stat = 0usize;
    let mut diverging = false;

    while depth < max_depth {
        // Choose direction: extend forward or backward
        let direction: f64 = if rng.gen::<bool>() { 1.0 } else { -1.0 };

        let subtree = if direction > 0.0 {
            build_subtree(
                graph, evaluator, &right, eps, mass, h0, depth, rng, scratch, pool,
            )
        } else {
            build_subtree(
                graph, evaluator, &left, -eps, mass, h0, depth, rng, scratch, pool,
            )
        };

        depth += 1;
        n_leapfrog_total += subtree.n_leapfrog;

        sum_accept_stat += subtree.sum_accept_prob;
        n_accept_stat += subtree.n_accept_prob;

        if subtree.diverging {
            diverging = true;
            pool.give_tree(subtree);
            break;
        }

        if subtree.turning {
            pool.give_tree(subtree);
            break;
        }

        // Progressive multinomial sampling for a newly doubled subtree uses
        // min(1, W_subtree / W_existing).  This differs deliberately from the
        // normalized selection used while recursively merging equal-depth
        // halves below.
        let accept_prob = progressive_selection_prob(subtree.log_sum_weight, log_sum_weight);
        let TreeResult {
            left: sub_left,
            right: sub_right,
            proposal: sub_proposal,
            log_sum_weight: sub_log_sum_weight,
            ..
        } = subtree;
        if rng.gen::<f64>() < accept_prob {
            pool.give(std::mem::replace(&mut proposal, sub_proposal));
        } else {
            pool.give(sub_proposal);
        }

        log_sum_weight = log_sum_exp(log_sum_weight, sub_log_sum_weight);

        // Update tree boundaries
        if direction > 0.0 {
            pool.give(std::mem::replace(&mut right, sub_right));
            pool.give(sub_left);
        } else {
            pool.give(std::mem::replace(&mut left, sub_left));
            pool.give(sub_right);
        }

        // Check U-turn across the full tree
        if check_uturn(&left, &right, mass, scratch) {
            break;
        }
    }
    pool.give(left);
    pool.give(right);

    let mean_accept = if n_accept_stat > 0 {
        (sum_accept_stat / n_accept_stat as f64).min(1.0)
    } else {
        0.0
    };
    let energy_error = proposal.energy(mass, scratch) - h0;

    (
        proposal,
        TreeStats {
            diverging,
            mean_accept_prob: mean_accept,
            energy_error,
            tree_depth: depth,
            n_leapfrog: n_leapfrog_total,
        },
    )
}

/// Recursively build a balanced binary subtree of given depth.
///
/// depth=0: take a single leapfrog step.
/// depth=j: build two subtrees of depth j-1 and combine.
// Recursive tree construction shares the same explicit sampler state and buffers.
#[allow(clippy::too_many_arguments)]
fn build_subtree(
    graph: &Graph,
    evaluator: &mut impl GradientEvaluator,
    point: &PhasePoint,
    eps: f64,
    mass: &MassMatrix,
    h0: f64,
    depth: usize,
    rng: &mut ChaCha8Rng,
    scratch: &mut [f64],
    pool: &mut PointPool,
) -> TreeResult {
    if depth == 0 {
        // Base case: single leapfrog step
        let mut next = pool.take();
        leapfrog(graph, evaluator, point, eps, mass, &mut next, scratch);
        let h_new = next.energy(mass, scratch);
        let delta_h = h_new - h0;
        let diverging = delta_h > MAX_DELTA_H || !delta_h.is_finite();
        let accept_prob = acceptance_probability(delta_h);
        let log_weight = if diverging {
            f64::NEG_INFINITY
        } else {
            -delta_h
        };

        return TreeResult {
            left: pool.copy_of(&next),
            right: pool.copy_of(&next),
            proposal: next,
            log_sum_weight: log_weight,
            n_leapfrog: 1,
            turning: false,
            diverging,
            sum_accept_prob: accept_prob,
            n_accept_prob: 1,
        };
    }

    // Build first half
    let inner = build_subtree(
        graph,
        evaluator,
        point,
        eps,
        mass,
        h0,
        depth - 1,
        rng,
        scratch,
        pool,
    );
    if inner.diverging || inner.turning {
        return inner;
    }

    // Build second half from the appropriate endpoint
    let start_point = if eps > 0.0 { &inner.right } else { &inner.left };
    let outer = build_subtree(
        graph,
        evaluator,
        start_point,
        eps,
        mass,
        h0,
        depth - 1,
        rng,
        scratch,
        pool,
    );

    let n_leapfrog = inner.n_leapfrog + outer.n_leapfrog;
    let sum_accept_prob = inner.sum_accept_prob + outer.sum_accept_prob;
    let n_accept_prob = inner.n_accept_prob + outer.n_accept_prob;

    if outer.diverging {
        pool.give_tree(outer);
        return TreeResult {
            n_leapfrog,
            turning: false,
            diverging: true,
            sum_accept_prob,
            n_accept_prob,
            ..inner
        };
    }

    // Combine proposals via multinomial weighting
    let log_sum = log_sum_exp(inner.log_sum_weight, outer.log_sum_weight);
    let accept_outer = normalized_selection_prob(outer.log_sum_weight, log_sum);
    let proposal = if rng.gen::<f64>() < accept_outer {
        pool.give(inner.proposal);
        outer.proposal
    } else {
        pool.give(outer.proposal);
        inner.proposal
    };

    // Merge boundaries: inner is "closer" to start, outer is "farther"
    let (left, right) = if eps > 0.0 {
        pool.give(inner.right);
        pool.give(outer.left);
        (inner.left, outer.right)
    } else {
        pool.give(inner.left);
        pool.give(outer.right);
        (outer.left, inner.right)
    };

    // Check U-turn on the merged subtree
    let turning = outer.turning || check_uturn(&left, &right, mass, scratch);

    TreeResult {
        left,
        right,
        proposal,
        log_sum_weight: log_sum,
        n_leapfrog,
        turning,
        diverging: false,
        sum_accept_prob,
        n_accept_prob,
    }
}

/// Single leapfrog step (half-step momentum, full-step position, half-step
/// momentum) from `point` into `out`, without allocating.
fn leapfrog(
    graph: &Graph,
    evaluator: &mut impl GradientEvaluator,
    point: &PhasePoint,
    eps: f64,
    mass: &MassMatrix,
    out: &mut PhasePoint,
    scratch: &mut [f64],
) {
    // Half step momentum
    for ((momentum, &old_momentum), &gradient) in
        out.p.iter_mut().zip(point.p.iter()).zip(point.grad.iter())
    {
        *momentum = old_momentum + 0.5 * eps * gradient;
    }
    // Full step position
    mass.velocity_into(&out.p, &mut out.q, scratch);
    for (position, &old_position) in out.q.iter_mut().zip(point.q.iter()) {
        *position = old_position + eps * *position;
    }
    // Evaluate gradient at new position
    evaluator.compute(graph, &out.q);
    out.logp = evaluator.log_density();
    out.grad.copy_from_slice(evaluator.gradient());
    // Half step momentum
    for (momentum, &gradient) in out.p.iter_mut().zip(out.grad.iter()) {
        *momentum += 0.5 * eps * gradient;
    }
}

/// Endpoint-momentum U-turn check: the trajectory is turning if the momentum
/// at either end would decrease the distance between the endpoints.
///
///   (q_right - q_left) · p_left < 0  OR
///   (q_right - q_left) · p_right < 0
///
/// For a constant Euclidean metric, the transform to canonical whitened
/// coordinates cancels from these dot products.
fn check_uturn(
    left: &PhasePoint,
    right: &PhasePoint,
    mass: &MassMatrix,
    scratch: &mut [f64],
) -> bool {
    mass.uturn(&left.q, &left.p, &right.q, &right.p, scratch)
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY && b == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn progressive_selection_prob(candidate_log_weight: f64, existing_log_weight: f64) -> f64 {
    if candidate_log_weight == f64::NEG_INFINITY {
        0.0
    } else {
        (candidate_log_weight - existing_log_weight).min(0.0).exp()
    }
}

fn normalized_selection_prob(candidate_log_weight: f64, total_log_weight: f64) -> f64 {
    if candidate_log_weight == f64::NEG_INFINITY {
        0.0
    } else {
        (candidate_log_weight - total_log_weight).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use rand::SeedableRng;

    /// `y = a + s x + noise` with an unknown noise scale when `sigma` is
    /// `None`; `x_scale` sets how far apart the posterior scales of `a` and
    /// `s` are.
    fn regression(n: usize, x_scale: f64, sigma: Option<f64>, prior_a: f64) -> Graph {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut normal = || -> f64 { StandardNormal.sample(&mut rng) };
        let x: Vec<f64> = (0..n).map(|_| x_scale * normal()).collect();
        let noise = sigma.unwrap_or(1.0);
        let y: Vec<f64> = x.iter().map(|x| 1.0 + 0.5 * x + noise * normal()).collect();
        let mut graph = Graph::new();
        let a = crate::distributions::Normal::prior(&mut graph, "a", 0.0, prior_a);
        let s = crate::distributions::Normal::prior(&mut graph, "s", 0.0, 10.0);
        let scale = match sigma {
            Some(sigma) => graph.add_constant(sigma),
            None => crate::distributions::HalfNormal::prior(&mut graph, "sigma", 5.0),
        };
        let x = graph.add_data("x", x);
        let slope = graph.scalar_mul_data(s, x);
        let mu = graph.scalar_broadcast_add(a, slope);
        let obs = graph.add_obs_data(y);
        graph.normal_obs_logp(mu, scale, obs);
        graph
    }

    #[test]
    fn very_short_warmups_do_not_end_on_an_unsettled_step_size() {
        // A 10% terminal buffer is two iterations at a 20-iteration warmup.
        // Installing a metric before it left dual averaging two updates after
        // its restart, centred ten times above the searched step: step sizes
        // near 1.4 where this model settles near 0.7, and about one draw in
        // ten divergent. Up to 40 iterations no metric is installed now; from
        // 41 one is, with 25 iterations to settle after it.
        let graph = regression(30, 1.0, None, 10.0);
        for num_warmup in [20, 25, 30, 41, 50, 60] {
            for seed in 0..4 {
                let config = NutsConfig {
                    num_warmup,
                    num_draws: 500,
                    ..NutsConfig::default()
                };
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let chain = run_chain(&graph, &config, &mut rng, None, None).unwrap();
                assert_eq!(chain.divergences, 0, "warmup {num_warmup} seed {seed}");
                assert!(
                    chain.step_size < 1.1,
                    "warmup {num_warmup} seed {seed}: {}",
                    chain.step_size
                );
            }
        }
    }

    #[test]
    fn warmups_just_past_the_default_buffers_keep_their_metric() {
        // Intercept and slope posterior sds of about 7 and 0.007. At warmup
        // 151 to 153 the last window used to hold one to three draws, whose
        // estimate (for one draw, the regularized unit variance) replaced the
        // 25-draw one before it, costing hundreds of leapfrog steps per draw
        // instead of about 16.
        let graph = regression(200, 1000.0, Some(100.0), 1000.0);
        for num_warmup in [150, 151, 152, 153, 155] {
            let config = NutsConfig {
                num_warmup,
                num_draws: 300,
                ..NutsConfig::default()
            };
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            let chain = run_chain(&graph, &config, &mut rng, None, None).unwrap();
            let draws = &chain.transitions[num_warmup..];
            let steps = draws.iter().map(|t| t.num_leapfrog_steps).sum::<usize>() as f64
                / draws.len() as f64;
            assert!(steps < 60.0, "warmup {num_warmup}: {steps} steps per draw");
        }
    }

    #[test]
    fn output_only_deterministic_preserves_sampling_from_zero() {
        use crate::graph::ElementwiseOp;
        let mut graph = Graph::new();
        let x = crate::distributions::Normal::prior(&mut graph, "x", 0.0, 1.0);
        let config = NutsConfig {
            num_draws: 20,
            num_warmup: 20,
            ..NutsConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let baseline = run_chain(&graph, &config, &mut rng, Some(vec![0.0]), None)
            .expect("continuous test model must run");
        let square = graph.elementwise(ElementwiseOp::Mul, x, Some(x));
        let abs = graph.elementwise(ElementwiseOp::Sqrt, square, None);
        graph.deterministics.push(("abs_x".into(), abs));
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let actual = run_chain(&graph, &config, &mut rng, Some(vec![0.0]), None)
            .expect("continuous test model must run");
        assert_eq!(actual.samples, baseline.samples);
        assert_eq!(actual.divergences, baseline.divergences);
    }

    #[test]
    fn terminating_expansion_is_counted() {
        let mut graph = Graph::new();
        crate::distributions::Normal::prior(&mut graph, "x", 0.0, 1.0);
        for step_size in [10.0, 100.0] {
            let config = NutsConfig {
                step_size,
                max_tree_depth: 5,
                num_draws: 1,
                num_warmup: 0,
                ..NutsConfig::default()
            };
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let chain = run_chain(&graph, &config, &mut rng, Some(vec![0.0]), None)
                .expect("continuous test model must run");
            let stats = &chain.transitions[0];
            assert_eq!(stats.num_leapfrog_steps, 1);
            assert_eq!(stats.tree_depth, Some(1));
            assert_eq!(stats.divergent, step_size == 100.0);
        }
    }

    #[test]
    fn small_shape_gamma_recovers_underflow_tail_mass() {
        let mut graph = Graph::new();
        crate::distributions::Gamma::prior(&mut graph, "x", 0.001, 1.0);
        let config = NutsConfig {
            num_draws: 6000,
            num_warmup: 1000,
            ..NutsConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let chain = run_chain(&graph, &config, &mut rng, Some(vec![-1000.0]), None)
            .expect("continuous test model must run");
        let fraction = chain.samples.iter().filter(|q| q[0] < -1000.0).count() as f64
            / chain.samples.len() as f64;
        // For x=exp(-1000), Gamma(.001,1)'s lower CDF is
        // x^alpha/Gamma(alpha+1); the omitted correction is O(exp(-1000)).
        let expected = (-1.0 - crate::autodiff::ln_gamma(1.001)).exp();
        assert!(
            (fraction - expected).abs() < 0.045,
            "tail fraction {fraction} != {expected}"
        );
    }

    #[test]
    fn boundary_concentrated_beta_recovers_both_raw_tails() {
        let mut graph = Graph::new();
        crate::distributions::BetaDist::prior(&mut graph, "x", 0.01, 0.01);
        let config = NutsConfig {
            num_draws: 6000,
            num_warmup: 1000,
            ..NutsConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let chain = run_chain(&graph, &config, &mut rng, Some(vec![0.0]), None)
            .expect("continuous test model must run");
        for sign in [-1.0, 1.0] {
            let fraction = chain.samples.iter().filter(|q| sign * q[0] > 40.0).count() as f64
                / chain.samples.len() as f64;
            // The Beta(.01,.01) integral above logit 40 is 0.3352.
            assert!(
                (fraction - 0.3352).abs() < 0.045,
                "tail fraction {fraction}"
            );
        }
    }

    #[test]
    fn nuts_chain_reports_posterior_only_diagnostics() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, one);
        let config = NutsConfig {
            step_size: 0.1,
            target_accept: 0.80,
            max_tree_depth: 3,
            num_draws: 3,
            num_warmup: 2,
            metric: MetricKind::Auto,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(9);

        let chain = run_chain(&graph, &config, &mut rng, None, None)
            .expect("continuous test model must run");
        let posterior: Vec<_> = chain
            .transitions
            .iter()
            .filter(|transition| !transition.is_warmup)
            .collect();

        assert_eq!(chain.samples.len(), 3);
        assert_eq!(posterior.len(), 3);
        assert_eq!(
            chain.divergences,
            posterior
                .iter()
                .filter(|transition| transition.divergent)
                .count()
        );
        let expected_accept = posterior
            .iter()
            .map(|transition| transition.accept_prob)
            .sum::<f64>()
            / posterior.len() as f64;
        assert_eq!(chain.accept_rate, expected_accept);
    }

    #[test]
    fn subtree_acceptance_stat_is_leafwise_not_a_weight_sum() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, one);
        let mass = MassMatrix::from_graph(&graph);
        let mut found_clipped_leaf = false;
        for q in [-2.0, -1.0, 0.5, 1.0, 2.0] {
            for p in [-2.0, -0.5, 0.5, 2.0] {
                for eps in [0.25, 0.5, 1.0] {
                    let mut evaluator = Evaluator::new(&graph);
                    evaluator.compute(&graph, &[q]);
                    let initial = PhasePoint {
                        q: vec![q],
                        p: vec![p],
                        grad: evaluator.grad.clone(),
                        logp: evaluator.total_logp,
                    };
                    let mut scratch = vec![0.0];
                    let h0 = initial.energy(&mass, &mut scratch);
                    let mut rng = ChaCha8Rng::seed_from_u64(11);
                    let tree = build_subtree(
                        &graph,
                        &mut evaluator,
                        &initial,
                        eps,
                        &mass,
                        h0,
                        1,
                        &mut rng,
                        &mut scratch,
                        &mut PointPool::new(1),
                    );
                    assert_eq!(tree.n_accept_prob, 2);
                    assert!(tree.sum_accept_prob >= 0.0 && tree.sum_accept_prob <= 2.0);
                    if (tree.sum_accept_prob - tree.log_sum_weight.exp()).abs() > 1e-6 {
                        found_clipped_leaf = true;
                    }
                }
            }
        }
        assert!(
            found_clipped_leaf,
            "negative control: some weight sums must differ from clipped leafwise acceptance"
        );
    }

    #[test]
    fn nuts_leaf_non_finite_energy_errors_have_zero_acceptance_probability() {
        assert_eq!(acceptance_probability(f64::INFINITY), 0.0);
        assert_eq!(acceptance_probability(f64::NEG_INFINITY), 0.0);
        assert_eq!(acceptance_probability(f64::NAN), 0.0);

        assert_eq!(acceptance_probability(0.0), 1.0);
        assert_eq!(acceptance_probability(-1.0), 1.0);
        assert!((acceptance_probability(1.0) - (-1.0_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn progressive_and_recursive_selection_use_distinct_denominators() {
        let existing = 0.0_f64;
        let candidate = 0.0_f64;
        let combined = log_sum_exp(existing, candidate);
        assert_eq!(progressive_selection_prob(candidate, existing), 1.0);
        assert_eq!(normalized_selection_prob(candidate, combined), 0.5);

        assert_eq!(progressive_selection_prob(f64::NEG_INFINITY, existing), 0.0);
        assert_eq!(normalized_selection_prob(f64::NEG_INFINITY, combined), 0.0);

        let lighter = -(2.0_f64).ln();
        assert_eq!(progressive_selection_prob(lighter, existing), 0.5);
        let normalized = normalized_selection_prob(lighter, log_sum_exp(existing, lighter));
        assert!((normalized - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn late_divergent_suffix_retains_seeded_valid_prefix_proposal() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let y = graph.add_param("y");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        let narrow = graph.add_constant(0.1);
        graph.normal_logp(x, zero, narrow);
        graph.normal_logp(y, zero, one);
        let mass = MassMatrix::from_graph(&graph);
        let mut evaluator = Evaluator::new(&graph);
        evaluator.compute(&graph, &[0.0, 0.0]);
        let mut current = PhasePoint {
            q: vec![0.0, 0.0],
            p: vec![0.01, 1.0],
            grad: evaluator.grad.clone(),
            logp: evaluator.total_logp,
        };
        let initial_q = current.q.clone();
        let mut scratch = vec![0.0; 2];
        let h0 = current.energy(&mass, &mut scratch);
        let mut rng = ChaCha8Rng::seed_from_u64(3);

        let (proposal, stats) = build_tree_iterative(
            &graph,
            &mut evaluator,
            &current,
            0.5,
            &mass,
            h0,
            8,
            &mut rng,
            &mut scratch,
            &mut PointPool::new(2),
        );

        assert!(stats.diverging);
        assert_eq!(
            stats.n_leapfrog, 3,
            "divergence must occur after a valid leaf"
        );
        assert_ne!(proposal.q, initial_q, "valid prefix proposal was discarded");
        assert!(proposal.q.iter().all(|value| value.is_finite()));

        update_current(&mut current, &proposal);
        assert_eq!(current.q, proposal.q);
        assert_ne!(current.q, initial_q);
    }

    /// `y ~ Normal(X b, 1)` with `b ~ Normal(0, 5)` over a vector parameter.
    fn vector_regression(x: Vec<f64>, n_rows: usize, dim: usize, y: Vec<f64>) -> Graph {
        let mut graph = Graph::new();
        let start = graph.add_vector_params("b", dim);
        graph.vector_normal_logp(start, dim, 0.0, 5.0);
        let matrix = graph.store_matrix(x, n_rows, dim);
        let mu = graph.mat_vec_mul(matrix, start, dim, None);
        let one = graph.add_constant(1.0);
        let obs = graph.add_obs_data(y);
        graph.normal_obs_logp(mu, one, obs);
        graph
    }

    fn mean_draw_leapfrog_steps(graph: &Graph, metric: MetricKind, seed: u64) -> f64 {
        let config = NutsConfig {
            num_warmup: 500,
            num_draws: 200,
            metric,
            ..NutsConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dim = graph.param_count;
        let chain = run_chain(graph, &config, &mut rng, Some(vec![0.0; dim]), None)
            .expect("continuous test model must run");
        let draws: Vec<_> = chain.transitions.iter().filter(|t| !t.is_warmup).collect();
        draws.iter().map(|t| t.num_leapfrog_steps).sum::<usize>() as f64 / draws.len() as f64
    }

    #[test]
    fn default_metric_matches_diagonal_on_an_isotropic_vector() {
        // A dense block estimated from a 200-draw window in 100 dimensions
        // is mostly noise; when every vector parameter got one, this target
        // took ~140 leapfrog steps per draw against ~10 for diagonal. The
        // window has two draws per dimension, so the auto rule does evaluate a
        // dense estimate here and has to reject it.
        let dim = 100;
        let mut x = vec![0.0; dim * dim];
        for i in 0..dim {
            x[i * dim + i] = 1.0;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let y: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let graph = vector_regression(x, dim, dim, y);
        for seed in [3, 4] {
            let auto = mean_draw_leapfrog_steps(&graph, MetricKind::Auto, seed);
            let diagonal = mean_draw_leapfrog_steps(&graph, MetricKind::Diagonal, seed);
            assert!(auto < 40.0, "auto metric took {auto} steps per draw");
            assert!(
                auto <= 1.25 * diagonal,
                "auto {auto} vs diagonal {diagonal}"
            );
        }
    }

    #[test]
    fn default_metric_keeps_the_dense_benefit_for_correlated_coefficients() {
        // Columns of X correlated at 0.9 make the posterior of b nearly
        // singular along their sum: a diagonal metric needs several times
        // the integration a dense one does. 50 dimensions at the default
        // warmup leaves a last window of four draws per dimension; an earlier
        // auto rule that demanded five stayed diagonal here at ~100 steps per
        // draw against ~12 for dense.
        for dim in [10, 50] {
            correlated_coefficients_case(dim);
        }
    }

    fn correlated_coefficients_case(dim: usize) {
        let n_rows = 400;
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut normal =
            || -> f64 { rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng) };
        let mut x = Vec::with_capacity(n_rows * dim);
        let mut y = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let shared = normal();
            let mut mean = 0.0;
            for k in 0..dim {
                let value = 0.9_f64.sqrt() * shared + 0.1_f64.sqrt() * normal();
                mean += value * (k as f64 / dim as f64 - 0.5);
                x.push(value);
            }
            y.push(mean + normal());
        }
        let graph = vector_regression(x, n_rows, dim, y);
        let auto = mean_draw_leapfrog_steps(&graph, MetricKind::Auto, 4);
        let diagonal = mean_draw_leapfrog_steps(&graph, MetricKind::Diagonal, 4);
        let dense = mean_draw_leapfrog_steps(&graph, MetricKind::Dense, 4);
        assert!(auto < 0.5 * diagonal, "auto {auto} vs diagonal {diagonal}");
        assert!(auto <= 1.5 * dense, "auto {auto} vs dense {dense}");
    }
}
