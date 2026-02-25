use crate::autodiff::Evaluator;
use crate::graph::Graph;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone)]
pub struct HmcConfig {
    pub step_size: f64,
    pub num_leapfrog_steps: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            step_size: 0.0, // 0 = auto-detect
            num_leapfrog_steps: 15,
            num_draws: 1000,
            num_warmup: 500,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChainResult {
    pub samples: Vec<Vec<f64>>,
    pub accept_rate: f64,
}

/// Run a single HMC chain with diagonal mass matrix adaptation.
///
/// Warmup is split into three phases (following Stan's approach):
///   Phase 1 (first 15%):  step-size adaptation only, identity mass matrix
///   Phase 2 (15%–90%):    collect samples → estimate diagonal mass matrix
///   Phase 3 (last 10%):   final step-size adaptation with the adapted mass matrix
///
/// All workspace buffers are pre-allocated. The `Evaluator` performs
/// zero-allocation gradient computation.
pub fn run_chain(
    graph: &Graph,
    config: &HmcConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
) -> ChainResult {
    let dim = graph.param_count;
    let total_iters = config.num_warmup + config.num_draws;

    let mut evaluator = Evaluator::new(graph);
    let mut q = init.unwrap_or_else(|| vec![0.0; dim]);
    let mut q_prop = vec![0.0; dim];
    let mut p = vec![0.0; dim];
    let mut p_prop = vec![0.0; dim];
    let mut grad = vec![0.0; dim];
    let mut samples = Vec::with_capacity(config.num_draws);
    let mut accepted = 0u64;
    let mut total = 0u64;

    // Diagonal mass matrix: M = diag(mass_diag)
    //   p ~ N(0, M)  →  p_i = sqrt(mass_diag[i]) * z_i
    //   q_i += ε * p_i / mass_diag[i]
    //   KE = Σ p_i² / (2 * mass_diag[i])
    let mut mass_diag = vec![1.0f64; dim];
    let mut inv_mass_diag = vec![1.0f64; dim];
    let mut mass_sqrt = vec![1.0f64; dim];

    // Warmup phase boundaries
    let phase1_end = config.num_warmup * 15 / 100;
    let phase2_end = config.num_warmup * 90 / 100;
    let mut warmup_q_sum = vec![0.0f64; dim];
    let mut warmup_q_sq_sum = vec![0.0f64; dim];
    let mut warmup_count = 0usize;

    // Auto step-size initialization
    let mut step_size = if config.step_size > 0.0 {
        config.step_size
    } else {
        find_initial_step_size(graph, &mut evaluator, &q, &inv_mass_diag, &mass_sqrt, dim, rng)
    };

    // Dual-averaging state
    let target_accept = 0.80;
    let da_mu = (10.0 * step_size).ln();
    let da_gamma = 0.05;
    let da_t0 = 10.0;
    let da_kappa = 0.75;
    let mut log_eps_bar = step_size.ln();
    let mut h_bar = 0.0f64;
    let mut adapt_count = 0u64;

    for iter in 0..total_iters {
        let is_warmup = iter < config.num_warmup;

        evaluator.compute(graph, &q);
        let logp_current = evaluator.total_logp;
        grad.copy_from_slice(&evaluator.grad);

        // Sample momentum: p ~ N(0, M)
        for i in 0..dim {
            let z: f64 = StandardNormal.sample(rng);
            p[i] = z * mass_sqrt[i];
        }

        q_prop.copy_from_slice(&q);
        p_prop.copy_from_slice(&p);

        // Half step for momentum
        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad[i];
        }

        // Leapfrog integration with mass matrix
        for step in 0..config.num_leapfrog_steps {
            // Full step for position: q += ε * M⁻¹ * p
            for i in 0..dim {
                q_prop[i] += step_size * inv_mass_diag[i] * p_prop[i];
            }

            evaluator.compute(graph, &q_prop);
            grad.copy_from_slice(&evaluator.grad);

            if step < config.num_leapfrog_steps - 1 {
                for i in 0..dim {
                    p_prop[i] += step_size * grad[i];
                }
            }
        }

        // Half step at end
        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad[i];
        }

        // Negate momentum for reversibility
        for v in p_prop.iter_mut() {
            *v = -*v;
        }

        // Acceptance — reuse logp from the last leapfrog gradient computation
        let logp_prop = evaluator.total_logp;

        let ke_current: f64 = (0..dim).map(|i| 0.5 * p[i] * p[i] * inv_mass_diag[i]).sum();
        let ke_prop: f64 = (0..dim).map(|i| 0.5 * p_prop[i] * p_prop[i] * inv_mass_diag[i]).sum();
        let h_current = -logp_current + ke_current;
        let h_prop = -logp_prop + ke_prop;
        let log_accept_ratio = h_current - h_prop;
        let accept_prob = log_accept_ratio.min(0.0).exp();

        total += 1;
        if log_accept_ratio.is_finite() && rng.gen::<f64>().ln() < log_accept_ratio {
            q.copy_from_slice(&q_prop);
            accepted += 1;
        }

        // --- Warmup adaptation ---
        if is_warmup {
            // Dual averaging for step size
            adapt_count += 1;
            let m = adapt_count as f64;
            let w = 1.0 / (m + da_t0);
            h_bar = (1.0 - w) * h_bar + w * (target_accept - accept_prob);
            let log_eps = da_mu - (m.sqrt() / da_gamma) * h_bar;
            step_size = log_eps.exp();
            let m_pow = m.powf(-da_kappa);
            log_eps_bar = m_pow * log_eps + (1.0 - m_pow) * log_eps_bar;

            // Phase 2: accumulate running statistics for mass matrix
            if iter >= phase1_end && iter < phase2_end {
                for i in 0..dim {
                    warmup_q_sum[i] += q[i];
                    warmup_q_sq_sum[i] += q[i] * q[i];
                }
                warmup_count += 1;
            }

            // Phase 2→3 transition: set mass matrix from estimated variances
            if iter == phase2_end && warmup_count > 10 {
                let n = warmup_count as f64;
                for i in 0..dim {
                    let mean = warmup_q_sum[i] / n;
                    let var = warmup_q_sq_sum[i] / n - mean * mean;
                    if var > 1e-8 {
                        mass_diag[i] = var;
                        inv_mass_diag[i] = 1.0 / var;
                        mass_sqrt[i] = var.sqrt();
                    }
                }
                // Reset dual averaging for phase 3 with the new mass matrix
                adapt_count = 0;
                h_bar = 0.0;
                let new_eps = find_initial_step_size(
                    graph,
                    &mut evaluator,
                    &q,
                    &inv_mass_diag,
                    &mass_sqrt,
                    dim,
                    rng,
                );
                step_size = new_eps;
                log_eps_bar = new_eps.ln();
            }
        }

        // Fix step size at end of warmup
        if iter == config.num_warmup.saturating_sub(1) && config.num_warmup > 0 {
            step_size = log_eps_bar.exp();
        }

        if !is_warmup {
            samples.push(q.clone());
        }
    }

    ChainResult {
        samples,
        accept_rate: accepted as f64 / total as f64,
    }
}

/// Find a reasonable initial step size using a doubling/halving search.
///
/// Starting from ε=1, take one leapfrog step and check the acceptance
/// probability. Double or halve ε until the acceptance is near 0.5.
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
    let ke0: f64 = (0..dim)
        .map(|i| 0.5 * p0[i] * p0[i] * inv_mass_diag[i])
        .sum();

    let mut eps = 1.0;

    // One leapfrog step to gauge acceptance at eps=1
    let log_ratio = one_step_log_ratio(
        graph, evaluator, q, &p0, &grad0, inv_mass_diag, eps, dim, logp0, ke0,
    );

    let direction = if log_ratio > (-0.5_f64).ln() {
        1.0
    } else {
        -1.0
    };

    for _ in 0..50 {
        let lr = one_step_log_ratio(
            graph, evaluator, q, &p0, &grad0, inv_mass_diag, eps, dim, logp0, ke0,
        );
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

/// Compute the log acceptance ratio for a single leapfrog step at step size `eps`.
fn one_step_log_ratio(
    graph: &Graph,
    evaluator: &mut Evaluator,
    q: &[f64],
    p0: &[f64],
    grad0: &[f64],
    inv_mass: &[f64],
    eps: f64,
    dim: usize,
    logp0: f64,
    ke0: f64,
) -> f64 {
    let mut p1 = vec![0.0; dim];
    let mut q1 = vec![0.0; dim];
    for i in 0..dim {
        p1[i] = p0[i] + 0.5 * eps * grad0[i];
        q1[i] = q[i] + eps * inv_mass[i] * p1[i];
    }
    evaluator.compute(graph, &q1);
    for i in 0..dim {
        p1[i] += 0.5 * eps * evaluator.grad[i];
    }
    let logp1 = evaluator.total_logp;
    let ke1: f64 = (0..dim).map(|i| 0.5 * p1[i] * p1[i] * inv_mass[i]).sum();
    (logp1 - ke1) - (logp0 - ke0)
}
