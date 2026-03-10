//! No-U-Turn Sampler (NUTS) — Hoffman & Gelman (2014) with multinomial
//! sampling (Betancourt 2017).
//!
//! This follows the same algorithm used by PyMC and Stan:
//!   - Iterative tree doubling (extend trajectory forward or backward)
//!   - Generalized U-turn criterion on subtrees
//!   - Multinomial candidate selection weighted by exp(-H)
//!   - Divergence detection via energy error threshold
//!   - Max tree depth cap (default 10)

use crate::autodiff::Evaluator;
use crate::graph::Graph;
use crate::hmc::ChainResult;
use crate::progress::ProgressState;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const MAX_DELTA_H: f64 = 1000.0;

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
    fn energy(&self, inv_mass: &[f64]) -> f64 {
        let ke: f64 = self
            .p
            .iter()
            .zip(inv_mass.iter())
            .map(|(&pi, &im)| 0.5 * pi * pi * im)
            .sum();
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
    /// Depth of this subtree.
    depth: usize,
    /// Number of leapfrog steps taken.
    n_leapfrog: usize,
    /// Whether a U-turn was detected inside this subtree.
    turning: bool,
    /// Whether a divergence was detected.
    diverging: bool,
}

/// Run a single NUTS chain with windowed diagonal mass matrix adaptation.
///
/// Warmup schedule (mirrors Stan's default):
///   Init buffer  (~75 draws or 15% of warmup, whichever is smaller):
///       step-size dual-averaging only, identity mass matrix.
///   Mass-matrix windows (doubling: 25 → 50 → 100 → 200 → …):
///       At the end of each window the diagonal mass matrix is updated
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
    let dim = graph.param_count;
    let total_iters = config.num_warmup + config.num_draws;

    let mut evaluator = Evaluator::new(graph);
    let q = init.unwrap_or_else(|| vec![0.0; dim]);
    let mut samples = Vec::with_capacity(config.num_draws);
    let mut n_divergences = 0usize;
    let mut sum_accept_prob = 0.0f64;
    let mut total_iters_done = 0u64;

    // Diagonal mass matrix (M⁻¹ diagonal and √M diagonal for momentum sampling).
    let mut inv_mass_diag = vec![1.0f64; dim];
    let mut mass_sqrt = vec![1.0f64; dim];

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

    // Welford online variance accumulator for the current mass-matrix window.
    let mut w_count = 0usize;
    let mut w_mean = vec![0.0f64; dim];
    let mut w_m2 = vec![0.0f64; dim]; // sum of squared deviations

    // Step-size initialization
    let mut step_size = if config.step_size > 0.0 {
        config.step_size
    } else {
        find_initial_step_size(graph, &mut evaluator, &q, &inv_mass_diag, &mass_sqrt, dim, rng)
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

        // Sample momentum from N(0, M) where M = diag(mass_sqrt²)
        for i in 0..dim {
            let z: f64 = StandardNormal.sample(rng);
            current.p[i] = z * mass_sqrt[i];
        }

        let h0 = current.energy(&inv_mass_diag);

        // Build the NUTS tree
        let (proposal, tree_stats) = build_tree_iterative(
            graph,
            &mut evaluator,
            &current,
            step_size,
            &inv_mass_diag,
            h0,
            config.max_tree_depth,
            dim,
            rng,
        );

        // Accept the NUTS proposal (multinomial weighting handles acceptance internally)
        if !tree_stats.diverging {
            current.q.copy_from_slice(&proposal.q);
            current.grad.copy_from_slice(&proposal.grad);
            current.logp = proposal.logp;
        }

        if tree_stats.diverging {
            n_divergences += 1;
        }

        let accept_stat = tree_stats.mean_accept_prob;
        sum_accept_prob += accept_stat;
        total_iters_done += 1;

        if let Some(p) = progress {
            p.increment();
            if tree_stats.diverging {
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
                welford_update(&current.q, &mut w_count, &mut w_mean, &mut w_m2);

                // End of window: update mass matrix, reset adaptation
                let window_done = (iter + 1 >= next_window_end) || (iter + 1 >= terminal_start);
                if window_done && w_count > 3 {
                    // Apply Welford variance estimate to mass matrix.
                    // Stan-style regularisation: shrink toward 1 with weight 5/(n+5)
                    // to prevent catastrophically large inv_mass from near-zero variance.
                    let n = w_count as f64;
                    for i in 0..dim {
                        let var_hat = if w_count > 1 {
                            w_m2[i] / (w_count - 1) as f64
                        } else {
                            1.0
                        };
                        let var = (n / (n + 5.0)) * var_hat + 1e-3 * (5.0 / (n + 5.0));
                        inv_mass_diag[i] = 1.0 / var;
                        mass_sqrt[i] = var.sqrt();
                    }

                    // Keep the current step size — the dual averaging state
                    // already has a reasonable estimate and re-running
                    // find_initial_step_size causes a transient instability
                    // (~20 extra divergences) while dual averaging re-converges.
                    // Just update the dual averaging anchor and reset accumulators.
                    da_mu = (10.0 * step_size).ln();
                    log_eps_bar = step_size.ln();
                    adapt_count = 0;
                    h_bar = 0.0;

                    // Advance window (doubling schedule)
                    window_size *= 2;
                    next_window_end = (iter + 1 + window_size).min(terminal_start);

                    // Reset Welford accumulator for next window
                    w_count = 0;
                    w_mean.iter_mut().for_each(|x| *x = 0.0);
                    w_m2.iter_mut().for_each(|x| *x = 0.0);
                }
            }
        }

        // At end of warmup, lock in the dual-averaged step size
        if iter == config.num_warmup.saturating_sub(1) && config.num_warmup > 0 {
            step_size = log_eps_bar.exp();
        }

        if !is_warmup {
            samples.push(current.q.clone());
        }
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
    }
}

/// Welford online algorithm for computing variance incrementally.
/// Numerically stable alternative to the two-pass variance formula.
fn welford_update(x: &[f64], count: &mut usize, mean: &mut [f64], m2: &mut [f64]) {
    *count += 1;
    let n = *count as f64;
    for i in 0..x.len() {
        let delta = x[i] - mean[i];
        mean[i] += delta / n;
        let delta2 = x[i] - mean[i];
        m2[i] += delta * delta2;
    }
}

struct TreeStats {
    diverging: bool,
    mean_accept_prob: f64,
}

/// Build the NUTS tree iteratively by doubling depth.
///
/// At each depth j, the tree has 2^j leaves. We randomly choose to extend
/// the trajectory forward (+ε) or backward (-ε). After extending, we check
/// the generalized U-turn criterion across the full tree. If a U-turn is
/// detected or a divergence occurs, we stop and return the current candidate.
fn build_tree_iterative(
    graph: &Graph,
    evaluator: &mut Evaluator,
    initial: &PhasePoint,
    eps: f64,
    inv_mass: &[f64],
    h0: f64,
    max_depth: usize,
    dim: usize,
    rng: &mut ChaCha8Rng,
) -> (PhasePoint, TreeStats) {
    let mut left = initial.clone();
    let mut right = initial.clone();
    let mut proposal = initial.clone();
    let mut log_sum_weight = 0.0f64; // log(exp(-H(initial))) normalized
    let mut depth = 0;
    let mut _n_leapfrog_total = 0;
    let mut sum_accept_stat = 0.0f64;
    let mut n_accept_stat = 0usize;
    let mut diverging = false;

    while depth < max_depth {
        // Choose direction: extend forward or backward
        let direction: f64 = if rng.gen::<bool>() { 1.0 } else { -1.0 };

        let subtree = if direction > 0.0 {
            build_subtree(
                graph, evaluator, &right, eps, inv_mass, h0, depth, dim, rng,
            )
        } else {
            build_subtree(
                graph, evaluator, &left, -eps, inv_mass, h0, depth, dim, rng,
            )
        };

        _n_leapfrog_total += subtree.n_leapfrog;

        if subtree.diverging {
            diverging = true;
            break;
        }

        if subtree.turning {
            break;
        }

        // Multinomial combination: accept subtree's proposal with probability
        // exp(subtree.log_sum_weight - log_sum_weight)
        let accept_prob =
            (subtree.log_sum_weight - log_sum_weight).min(0.0).exp();
        if rng.gen::<f64>() < accept_prob {
            proposal = subtree.proposal;
        }

        log_sum_weight = log_sum_exp(log_sum_weight, subtree.log_sum_weight);

        // Compute per-leaf acceptance statistics for the subtree
        let n_leaves = 1usize << subtree.depth;
        sum_accept_stat += subtree.log_sum_weight.exp().min(n_leaves as f64);
        n_accept_stat += n_leaves;

        // Update tree boundaries
        if direction > 0.0 {
            right = subtree.right;
        } else {
            left = subtree.left;
        }

        // Check U-turn across the full tree
        if check_uturn(&left, &right, inv_mass) {
            break;
        }

        depth += 1;
    }

    let mean_accept = if n_accept_stat > 0 {
        (sum_accept_stat / n_accept_stat as f64).min(1.0)
    } else {
        0.0
    };

    (
        proposal,
        TreeStats {
            diverging,
            mean_accept_prob: mean_accept,
        },
    )
}

/// Recursively build a balanced binary subtree of given depth.
///
/// depth=0: take a single leapfrog step.
/// depth=j: build two subtrees of depth j-1 and combine.
fn build_subtree(
    graph: &Graph,
    evaluator: &mut Evaluator,
    point: &PhasePoint,
    eps: f64,
    inv_mass: &[f64],
    h0: f64,
    depth: usize,
    dim: usize,
    rng: &mut ChaCha8Rng,
) -> TreeResult {
    if depth == 0 {
        // Base case: single leapfrog step
        let next = leapfrog(graph, evaluator, point, eps, inv_mass, dim);
        let h_new = next.energy(inv_mass);
        let delta_h = h_new - h0;
        let diverging = delta_h > MAX_DELTA_H || !delta_h.is_finite();
        let log_weight = if diverging { f64::NEG_INFINITY } else { -delta_h };

        return TreeResult {
            left: next.clone(),
            right: next.clone(),
            proposal: next,
            log_sum_weight: log_weight,
            depth: 0,
            n_leapfrog: 1,
            turning: false,
            diverging,
        };
    }

    // Build first half
    let inner = build_subtree(graph, evaluator, point, eps, inv_mass, h0, depth - 1, dim, rng);
    if inner.diverging || inner.turning {
        return inner;
    }

    // Build second half from the appropriate endpoint
    let start_point = if eps > 0.0 { &inner.right } else { &inner.left };
    let outer = build_subtree(
        graph, evaluator, start_point, eps, inv_mass, h0, depth - 1, dim, rng,
    );

    if outer.diverging {
        return TreeResult {
            left: inner.left,
            right: inner.right,
            proposal: inner.proposal,
            log_sum_weight: inner.log_sum_weight,
            depth,
            n_leapfrog: inner.n_leapfrog + outer.n_leapfrog,
            turning: false,
            diverging: true,
        };
    }

    // Combine proposals via multinomial weighting
    let log_sum = log_sum_exp(inner.log_sum_weight, outer.log_sum_weight);
    let accept_outer = (outer.log_sum_weight - log_sum).exp();
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
    let turning = outer.turning || check_uturn(&left, &right, inv_mass);

    TreeResult {
        left,
        right,
        proposal,
        log_sum_weight: log_sum,
        depth,
        n_leapfrog: inner.n_leapfrog + outer.n_leapfrog,
        turning,
        diverging: false,
    }
}

/// Single leapfrog step (half-step momentum, full-step position, half-step momentum).
fn leapfrog(
    graph: &Graph,
    evaluator: &mut Evaluator,
    point: &PhasePoint,
    eps: f64,
    inv_mass: &[f64],
    dim: usize,
) -> PhasePoint {
    let mut p_new = vec![0.0; dim];
    let mut q_new = vec![0.0; dim];

    // Half step momentum
    for i in 0..dim {
        p_new[i] = point.p[i] + 0.5 * eps * point.grad[i];
    }
    // Full step position
    for i in 0..dim {
        q_new[i] = point.q[i] + eps * inv_mass[i] * p_new[i];
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

/// Generalized U-turn check: the trajectory is turning if the momentum
/// at either end would decrease the distance between the endpoints.
///
///   (q_right - q_left) · (M⁻¹ p_left) < 0  OR
///   (q_right - q_left) · (M⁻¹ p_right) < 0
fn check_uturn(left: &PhasePoint, right: &PhasePoint, inv_mass: &[f64]) -> bool {
    let mut dot_left = 0.0f64;
    let mut dot_right = 0.0f64;
    for i in 0..left.q.len() {
        let dq = right.q[i] - left.q[i];
        dot_left += dq * (inv_mass[i] * left.p[i]);
        dot_right += dq * (inv_mass[i] * right.p[i]);
    }
    dot_left < 0.0 || dot_right < 0.0
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY && b == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Find initial step size — same algorithm as hmc.rs.
fn find_initial_step_size(
    graph: &Graph,
    evaluator: &mut Evaluator,
    q: &[f64],
    inv_mass_diag: &[f64],
    mass_sqrt: &[f64],
    dim: usize,
    rng: &mut ChaCha8Rng,
) -> f64 {
    evaluator.compute(graph, q);
    let logp0 = evaluator.total_logp;
    let grad0: Vec<f64> = evaluator.grad.clone();

    let p0: Vec<f64> = (0..dim)
        .map(|i| {
            let z: f64 = StandardNormal.sample(rng);
            z * mass_sqrt[i]
        })
        .collect();
    let _ke0: f64 = (0..dim)
        .map(|i| 0.5 * p0[i] * p0[i] * inv_mass_diag[i])
        .sum();

    let mut eps = 1.0;

    let initial_point = PhasePoint {
        q: q.to_vec(),
        p: p0,
        grad: grad0,
        logp: logp0,
    };

    let test = leapfrog(graph, evaluator, &initial_point, eps, inv_mass_diag, dim);
    let h0 = initial_point.energy(inv_mass_diag);
    let h1 = test.energy(inv_mass_diag);
    let log_ratio = h0 - h1;

    let direction = if log_ratio > (-0.5_f64).ln() {
        1.0
    } else {
        -1.0
    };

    for _ in 0..50 {
        let t = leapfrog(graph, evaluator, &initial_point, eps, inv_mass_diag, dim);
        let lr = h0 - t.energy(inv_mass_diag);
        if !lr.is_finite() {
            eps *= 0.5;
            break;
        }
        if direction > 0.0 && lr < (-0.5_f64).ln() {
            break;
        }
        if direction < 0.0 && lr > (-0.5_f64).ln() {
            break;
        }
        eps *= 2.0_f64.powf(direction);
    }

    eps.clamp(1e-10, 1e3)
}
