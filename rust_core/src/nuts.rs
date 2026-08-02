//! No-U-Turn Sampler (NUTS) — Hoffman & Gelman (2014) with multinomial
//! sampling (Betancourt 2017).
//!
//! This follows the core NUTS design used by PyMC and Stan:
//!   - Iterative tree doubling (extend trajectory forward or backward)
//!   - Endpoint-momentum U-turn checks on subtrees
//!   - Multinomial candidate selection weighted by exp(-H)
//!   - Divergence detection via energy error threshold
//!   - Max tree depth cap (default 10)

use crate::autodiff::Evaluator;
use crate::data::DataBinding;
use crate::graph::Graph;
use crate::hmc::{acceptance_probability, ChainResult, TransitionStats, MAX_DELTA_H};
use crate::mass_matrix::{MassMatrix, MassMatrixAccumulator};
use crate::progress::ProgressState;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct NutsConfig {
    pub step_size: f64,
    pub max_tree_depth: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            step_size: 0.0,
            max_tree_depth: 10,
            num_draws: 1000,
            num_warmup: 500,
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
/// Warmup schedule (mirrors Stan's default):
///   Init buffer  (~75 draws or 15% of warmup, whichever is smaller):
///       step-size dual-averaging only, identity mass matrix.
///   Mass-matrix windows (doubling: 25 → 50 → 100 → 200 → …):
///       At the end of each window the block-structured mass matrix is updated
///       from Welford online variance estimates collected in that window.
///       Dual averaging and step size are reset after each update so the
///       sampler can re-converge with the new metric.
///   Terminal buffer (~50 draws or 10% of warmup):
///       step-size dual-averaging only, final fixed mass matrix.
pub fn run_chain(
    graph: &Graph,
    config: &NutsConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
    progress: Option<&ProgressState>,
) -> ChainResult {
    let binding = DataBinding::from_graph(graph).expect("graph data must have consistent shapes");
    run_chain_bound(graph, binding, config, rng, init, progress)
}

/// Run a chain against a validated dataset without embedding it in `Graph`.
pub fn run_chain_bound(
    graph: &Graph,
    binding: DataBinding,
    config: &NutsConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
    progress: Option<&ProgressState>,
) -> ChainResult {
    let dim = graph.param_count;
    let total_iters = config.num_warmup + config.num_draws;

    let mut evaluator = Evaluator::with_binding(graph, binding);
    let q = init.unwrap_or_else(|| vec![0.0; dim]);
    let mut samples = Vec::with_capacity(config.num_draws);
    let mut transitions = Vec::with_capacity(total_iters);
    let mut n_divergences = 0usize;
    let mut sum_accept_prob = 0.0f64;
    let mut total_iters_done = 0u64;

    let mut mass = MassMatrix::from_graph(graph);
    let mut mass_acc = MassMatrixAccumulator::from_graph(graph);
    let mut scratch = vec![0.0f64; dim];
    let mut w_count = 0usize;

    // --- Windowed warmup schedule (Stan defaults) ---
    // init_buffer: step size only, identity mass
    // windows:     growing mass-estimation windows
    // term_buffer: step size only, fixed final mass
    let init_buffer = 75_usize.min(config.num_warmup * 15 / 100).max(1);
    let term_buffer = 50_usize.min(config.num_warmup * 10 / 100);
    let terminal_start = config.num_warmup.saturating_sub(term_buffer);
    // First mass-matrix window ends after 25 draws past init_buffer (clamped).
    let first_window = 25_usize;
    let mut next_window_end = (init_buffer + first_window).min(terminal_start);
    let mut window_size = first_window;

    // Step-size initialization
    let mut step_size = if config.step_size > 0.0 {
        config.step_size
    } else {
        find_initial_step_size(graph, &mut evaluator, &q, &mass, &mut scratch, rng)
    };

    // Dual averaging (target = 0.80 for NUTS, following Stan)
    let target_accept = 0.80;
    let mut da_mu = (10.0 * step_size).ln();
    let da_gamma = 0.05;
    let da_t0 = 10.0;
    let da_kappa = 0.75;
    let mut log_eps_bar = step_size.ln();
    let mut h_bar = 0.0f64;
    let mut adapt_count = 0u64;

    // Compute initial state
    evaluator.compute(graph, &q);
    let mut current = PhasePoint {
        q: q.clone(),
        p: vec![0.0; dim],
        grad: evaluator.grad.clone(),
        logp: evaluator.total_logp,
    };

    for iter in 0..total_iters {
        let is_warmup = iter < config.num_warmup;
        let step_size_used = step_size;

        mass.sample_momentum_into(rng, &mut current.p, &mut scratch);

        let h0 = current.energy(&mass, &mut scratch);

        // Build the NUTS tree
        let (proposal, tree_stats) = build_tree_iterative(
            graph,
            &mut evaluator,
            &current,
            step_size,
            &mass,
            h0,
            config.max_tree_depth,
            dim,
            rng,
            &mut scratch,
        );

        // Multinomial weighting handles candidate selection internally.  A
        // divergence terminates trajectory construction, but does not
        // invalidate a candidate selected from the valid trajectory prefix.
        update_current(&mut current, &proposal);

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

        // --- Warmup adaptation ---
        if is_warmup {
            // Dual averaging step-size update (always, throughout warmup)
            adapt_count += 1;
            let m = adapt_count as f64;
            let w = 1.0 / (m + da_t0);
            h_bar = (1.0 - w) * h_bar + w * (target_accept - accept_stat);
            let log_eps = da_mu - (m.sqrt() / da_gamma) * h_bar;
            step_size = log_eps.exp();
            let m_pow = m.powf(-da_kappa);
            log_eps_bar = m_pow * log_eps + (1.0 - m_pow) * log_eps_bar;

            // Mass-matrix estimation: collect samples in the current window
            let in_window = iter >= init_buffer && iter < terminal_start;
            if in_window {
                mass_acc.update(&current.q);
                w_count += 1;
            }

            // End of window: update mass matrix, reset adaptation
            let final_window = iter + 1 >= terminal_start;
            let window_done = iter + 1 >= next_window_end;
            // Recalibrating the metric and step size needs a meaningful
            // terminal buffer. With very short warmup schedules, retain the
            // preceding metric instead of installing a last-minute estimate
            // that dual averaging has too few iterations to stabilize.
            let enough_terminal_adaptation = !final_window || term_buffer >= first_window;
            if window_done
                && enough_terminal_adaptation
                && iter >= init_buffer
                && iter < terminal_start
                && w_count > 3
            {
                mass = mass_acc.finalize();

                // A new metric changes both momentum scale and velocity, so
                // the old metric's step size is no longer calibrated. Find a
                // reasonable value under the new geometry before restarting
                // dual averaging for this window.
                step_size = find_initial_step_size(
                    graph,
                    &mut evaluator,
                    &current.q,
                    &mass,
                    &mut scratch,
                    rng,
                );
                da_mu = (10.0 * step_size).ln();
                log_eps_bar = step_size.ln();
                adapt_count = 0;
                h_bar = 0.0;

                // Advance window (doubling schedule)
                window_size *= 2;
                next_window_end = (iter + 1 + window_size).min(terminal_start);

                // Reset accumulator for next window
                mass_acc = MassMatrixAccumulator::from_graph(graph);
                w_count = 0;
            }
        }

        // At end of warmup, lock in the dual-averaged step size
        if iter == config.num_warmup.saturating_sub(1) && config.num_warmup > 0 {
            step_size = log_eps_bar.exp();
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

/// Build the NUTS tree iteratively by doubling depth.
///
/// At each depth j, the tree has 2^j leaves. We randomly choose to extend
/// the trajectory forward (+ε) or backward (-ε). After extending, we check
/// the endpoint-momentum U-turn criterion across the full tree. If a U-turn is
/// detected or a divergence occurs, we stop and return the current candidate.
// NUTS tree construction passes explicit state and reusable buffers on its hot path.
#[allow(clippy::too_many_arguments)]
fn build_tree_iterative(
    graph: &Graph,
    evaluator: &mut Evaluator,
    initial: &PhasePoint,
    eps: f64,
    mass: &MassMatrix,
    h0: f64,
    max_depth: usize,
    dim: usize,
    rng: &mut ChaCha8Rng,
    scratch: &mut [f64],
) -> (PhasePoint, TreeStats) {
    let mut left = initial.clone();
    let mut right = initial.clone();
    let mut proposal = initial.clone();
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
                graph, evaluator, &right, eps, mass, h0, depth, dim, rng, scratch,
            )
        } else {
            build_subtree(
                graph, evaluator, &left, -eps, mass, h0, depth, dim, rng, scratch,
            )
        };

        n_leapfrog_total += subtree.n_leapfrog;

        sum_accept_stat += subtree.sum_accept_prob;
        n_accept_stat += subtree.n_accept_prob;

        if subtree.diverging {
            diverging = true;
            break;
        }

        if subtree.turning {
            break;
        }

        // Progressive multinomial sampling for a newly doubled subtree uses
        // min(1, W_subtree / W_existing).  This differs deliberately from the
        // normalized selection used while recursively merging equal-depth
        // halves below.
        let accept_prob = progressive_selection_prob(subtree.log_sum_weight, log_sum_weight);
        if rng.gen::<f64>() < accept_prob {
            proposal = subtree.proposal;
        }

        log_sum_weight = log_sum_exp(log_sum_weight, subtree.log_sum_weight);

        // Update tree boundaries
        if direction > 0.0 {
            right = subtree.right;
        } else {
            left = subtree.left;
        }

        // Check U-turn across the full tree
        if check_uturn(&left, &right, mass, scratch) {
            break;
        }

        depth += 1;
    }

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
    evaluator: &mut Evaluator,
    point: &PhasePoint,
    eps: f64,
    mass: &MassMatrix,
    h0: f64,
    depth: usize,
    dim: usize,
    rng: &mut ChaCha8Rng,
    scratch: &mut [f64],
) -> TreeResult {
    if depth == 0 {
        // Base case: single leapfrog step
        let next = leapfrog(graph, evaluator, point, eps, mass, dim, scratch);
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
            left: next.clone(),
            right: next.clone(),
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
        dim,
        rng,
        scratch,
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
        dim,
        rng,
        scratch,
    );

    if outer.diverging {
        return TreeResult {
            left: inner.left,
            right: inner.right,
            proposal: inner.proposal,
            log_sum_weight: inner.log_sum_weight,
            n_leapfrog: inner.n_leapfrog + outer.n_leapfrog,
            turning: false,
            diverging: true,
            sum_accept_prob: inner.sum_accept_prob + outer.sum_accept_prob,
            n_accept_prob: inner.n_accept_prob + outer.n_accept_prob,
        };
    }

    // Combine proposals via multinomial weighting
    let log_sum = log_sum_exp(inner.log_sum_weight, outer.log_sum_weight);
    let accept_outer = normalized_selection_prob(outer.log_sum_weight, log_sum);
    let proposal = if rng.gen::<f64>() < accept_outer {
        outer.proposal
    } else {
        inner.proposal
    };

    // Merge boundaries: inner is "closer" to start, outer is "farther"
    let (left, right) = if eps > 0.0 {
        (inner.left, outer.right)
    } else {
        (outer.left, inner.right)
    };

    // Check U-turn on the merged subtree
    let turning = outer.turning || check_uturn(&left, &right, mass, scratch);

    TreeResult {
        left,
        right,
        proposal,
        log_sum_weight: log_sum,
        n_leapfrog: inner.n_leapfrog + outer.n_leapfrog,
        turning,
        diverging: false,
        sum_accept_prob: inner.sum_accept_prob + outer.sum_accept_prob,
        n_accept_prob: inner.n_accept_prob + outer.n_accept_prob,
    }
}

/// Single leapfrog step (half-step momentum, full-step position, half-step momentum).
fn leapfrog(
    graph: &Graph,
    evaluator: &mut Evaluator,
    point: &PhasePoint,
    eps: f64,
    mass: &MassMatrix,
    dim: usize,
    scratch: &mut [f64],
) -> PhasePoint {
    let mut p_new = vec![0.0; dim];
    let mut q_new = vec![0.0; dim];

    // Half step momentum
    for ((momentum, &old_momentum), &gradient) in p_new
        .iter_mut()
        .zip(point.p.iter())
        .zip(point.grad.iter())
        .take(dim)
    {
        *momentum = old_momentum + 0.5 * eps * gradient;
    }
    // Full step position
    mass.velocity_into(&p_new, &mut q_new, scratch);
    for (position, &old_position) in q_new.iter_mut().zip(point.q.iter()).take(dim) {
        *position = old_position + eps * *position;
    }
    // Evaluate gradient at new position
    evaluator.compute(graph, &q_new);
    let logp_new = evaluator.total_logp;
    let grad_new = evaluator.grad.clone();
    // Half step momentum
    for i in 0..dim {
        p_new[i] += 0.5 * eps * grad_new[i];
    }

    PhasePoint {
        q: q_new,
        p: p_new,
        grad: grad_new,
        logp: logp_new,
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

/// Find initial step size — same algorithm as hmc.rs.
fn find_initial_step_size(
    graph: &Graph,
    evaluator: &mut Evaluator,
    q: &[f64],
    mass: &MassMatrix,
    scratch: &mut [f64],
    rng: &mut ChaCha8Rng,
) -> f64 {
    evaluator.compute(graph, q);
    let logp0 = evaluator.total_logp;
    let grad0: Vec<f64> = evaluator.grad.clone();
    let dim = q.len();
    let mut p0 = vec![0.0; dim];
    mass.sample_momentum_into(rng, &mut p0, scratch);

    let mut eps = 1.0;

    let initial_point = PhasePoint {
        q: q.to_vec(),
        p: p0,
        grad: grad0,
        logp: logp0,
    };

    let test = leapfrog(graph, evaluator, &initial_point, eps, mass, dim, scratch);
    let h0 = initial_point.energy(mass, scratch);
    let h1 = test.energy(mass, scratch);
    let log_ratio = h0 - h1;

    let log_half = 0.5_f64.ln();
    let direction = if log_ratio > log_half { 1.0 } else { -1.0 };

    for _ in 0..50 {
        let t = leapfrog(graph, evaluator, &initial_point, eps, mass, dim, scratch);
        let lr = h0 - t.energy(mass, scratch);
        if !lr.is_finite() {
            eps *= 0.5;
            break;
        }
        if direction > 0.0 && lr < log_half {
            break;
        }
        if direction < 0.0 && lr > log_half {
            break;
        }
        eps *= 2.0_f64.powf(direction);
    }

    eps.clamp(1e-10, 1e3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use rand::SeedableRng;

    #[test]
    fn initial_step_size_search_is_not_pinned_to_lower_bound() {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, one);
        let mut evaluator = Evaluator::new(&graph);
        let mass = MassMatrix::from_graph(&graph);
        let mut scratch = vec![0.0; 1];
        let mut rng = ChaCha8Rng::seed_from_u64(17);

        let step_size = find_initial_step_size(
            &graph,
            &mut evaluator,
            &[0.0],
            &mass,
            &mut scratch,
            &mut rng,
        );

        assert!(step_size.is_finite());
        assert!(
            step_size > 1e-6,
            "initial step-size search collapsed to {step_size}"
        );
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
            max_tree_depth: 3,
            num_draws: 3,
            num_warmup: 2,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(9);

        let chain = run_chain(&graph, &config, &mut rng, None, None);
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
                        1,
                        &mut rng,
                        &mut scratch,
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
            2,
            &mut rng,
            &mut scratch,
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
}
