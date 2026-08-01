use crate::autodiff::Evaluator;
use crate::data::DataBinding;
use crate::graph::Graph;
use crate::mass_matrix::{MassMatrix, MassMatrixAccumulator};
use crate::progress::ProgressState;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

pub(crate) const MAX_DELTA_H: f64 = 1000.0;

pub(crate) fn acceptance_probability(energy_error: f64) -> f64 {
    if !energy_error.is_finite() {
        0.0
    } else {
        (-energy_error).min(0.0).exp()
    }
}

/// Per-transition sampler telemetry shared by HMC and NUTS.
///
/// `tree_depth` is only populated for NUTS. `num_leapfrog_steps` always
/// records the actual integrator work done by the transition.
#[derive(Debug, Clone)]
pub struct TransitionStats {
    pub is_warmup: bool,
    pub accepted: bool,
    pub accept_prob: f64,
    pub energy_error: f64,
    pub divergent: bool,
    pub step_size: f64,
    pub num_leapfrog_steps: usize,
    pub tree_depth: Option<usize>,
}

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
    pub step_size: f64,
    pub divergences: usize,
    pub transitions: Vec<TransitionStats>,
}

/// Run a single HMC chain with block-structured mass matrix adaptation.
///
/// Warmup is split into three phases (following Stan's approach):
///   Phase 1 (first 15%):  step-size adaptation only, identity mass matrix
///   Phase 2 (15%–90%):    collect samples → estimate the block-structured metric
///   Phase 3 (last 10%):   final step-size adaptation with the adapted mass matrix
///
/// All workspace buffers are pre-allocated. The `Evaluator` performs
/// zero-allocation gradient computation.
pub fn run_chain(
    graph: &Graph,
    config: &HmcConfig,
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
    config: &HmcConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
    progress: Option<&ProgressState>,
) -> ChainResult {
    let dim = graph.param_count;
    let total_iters = config.num_warmup + config.num_draws;

    let mut evaluator = Evaluator::with_binding(graph, binding);
    let mut q = init.unwrap_or_else(|| vec![0.0; dim]);
    let mut q_prop = vec![0.0; dim];
    let mut p = vec![0.0; dim];
    let mut p_prop = vec![0.0; dim];
    let mut grad = vec![0.0; dim];
    let mut velocity = vec![0.0; dim];
    let mut scratch = vec![0.0; dim];
    let mut samples = Vec::with_capacity(config.num_draws);
    let mut transitions = Vec::with_capacity(total_iters);
    let mut accepted = 0u64;
    let mut total = 0u64;
    let mut n_divergences = 0usize;

    let mut mass = MassMatrix::from_graph(graph);
    let mut mass_acc = MassMatrixAccumulator::from_graph(graph);

    // Warmup phase boundaries
    let phase1_end = config.num_warmup * 15 / 100;
    let phase2_end = config.num_warmup * 90 / 100;
    let mut warmup_count = 0usize;

    // Auto step-size initialization
    let mut step_size = if config.step_size > 0.0 {
        config.step_size
    } else {
        find_initial_step_size(graph, &mut evaluator, &q, &mass, &mut scratch, rng)
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
        let step_size_used = step_size;

        evaluator.compute(graph, &q);
        let logp_current = evaluator.total_logp;
        grad.copy_from_slice(&evaluator.grad);

        mass.sample_momentum_into(rng, &mut p, &mut scratch);

        q_prop.copy_from_slice(&q);
        p_prop.copy_from_slice(&p);

        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad[i];
        }

        for step in 0..config.num_leapfrog_steps {
            mass.velocity_into(&p_prop, &mut velocity, &mut scratch);
            for i in 0..dim {
                q_prop[i] += step_size * velocity[i];
            }

            evaluator.compute(graph, &q_prop);
            grad.copy_from_slice(&evaluator.grad);

            if step < config.num_leapfrog_steps - 1 {
                for i in 0..dim {
                    p_prop[i] += step_size * grad[i];
                }
            }
        }

        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad[i];
        }

        for v in p_prop.iter_mut() {
            *v = -*v;
        }

        let logp_prop = evaluator.total_logp;
        let ke_current = mass.kinetic_energy(&p, &mut scratch);
        let ke_prop = mass.kinetic_energy(&p_prop, &mut scratch);
        let h_current = -logp_current + ke_current;
        let h_prop = -logp_prop + ke_prop;
        let log_accept_ratio = h_current - h_prop;
        let energy_error = h_prop - h_current;
        let accept_prob = acceptance_probability(energy_error);

        let divergent = energy_error > MAX_DELTA_H || !energy_error.is_finite();
        let mut accepted_transition = false;
        if !divergent && rng.gen::<f64>().ln() < log_accept_ratio {
            q.copy_from_slice(&q_prop);
            accepted_transition = true;
        }

        // Warmup transitions are retained in `transitions` for auditability,
        // but must not affect posterior-sampling diagnostics.
        if !is_warmup {
            total += 1;
            if divergent {
                n_divergences += 1;
            }
            if accepted_transition {
                accepted += 1;
            }
        }

        if let Some(pbar) = progress {
            pbar.increment();
            if !is_warmup && divergent {
                pbar.add_divergence();
            }
        }

        if is_warmup {
            adapt_count += 1;
            let m = adapt_count as f64;
            let w = 1.0 / (m + da_t0);
            h_bar = (1.0 - w) * h_bar + w * (target_accept - accept_prob);
            let log_eps = da_mu - (m.sqrt() / da_gamma) * h_bar;
            step_size = log_eps.exp();
            let m_pow = m.powf(-da_kappa);
            log_eps_bar = m_pow * log_eps + (1.0 - m_pow) * log_eps_bar;

            if iter >= phase1_end && iter < phase2_end {
                mass_acc.update(&q);
                warmup_count += 1;
            }

            if iter == phase2_end && warmup_count > 10 {
                mass = mass_acc.finalize();
                mass_acc.reset();
                adapt_count = 0;
                h_bar = 0.0;
                let new_eps =
                    find_initial_step_size(graph, &mut evaluator, &q, &mass, &mut scratch, rng);
                step_size = new_eps;
                log_eps_bar = new_eps.ln();
            }
        }

        if iter == config.num_warmup.saturating_sub(1) && config.num_warmup > 0 {
            step_size = log_eps_bar.exp();
        }

        if !is_warmup {
            samples.push(q.clone());
        }

        transitions.push(TransitionStats {
            is_warmup,
            accepted: accepted_transition,
            accept_prob,
            energy_error,
            divergent,
            step_size: step_size_used,
            num_leapfrog_steps: config.num_leapfrog_steps,
            tree_depth: None,
        });
    }

    ChainResult {
        samples,
        accept_rate: if total > 0 {
            accepted as f64 / total as f64
        } else {
            0.0
        },
        step_size,
        divergences: n_divergences,
        transitions,
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
    mass: &MassMatrix,
    scratch: &mut [f64],
    rng: &mut ChaCha8Rng,
) -> f64 {
    evaluator.compute(graph, q);
    let logp0 = evaluator.total_logp;
    let grad0: Vec<f64> = evaluator.grad.clone();
    let dim = q.len();
    let mut p0 = vec![0.0; dim];
    let mut p1 = vec![0.0; dim];
    let mut q1 = vec![0.0; dim];
    let mut velocity = vec![0.0; dim];
    mass.sample_momentum_into(rng, &mut p0, scratch);
    let ke0 = mass.kinetic_energy(&p0, scratch);

    let mut eps = 1.0;

    // One leapfrog step to gauge acceptance at eps=1
    let log_ratio = one_step_log_ratio(
        graph,
        evaluator,
        q,
        &p0,
        &grad0,
        mass,
        eps,
        logp0,
        ke0,
        &mut p1,
        &mut q1,
        &mut velocity,
        scratch,
    );

    let direction = if log_ratio > (-0.5_f64).ln() {
        1.0
    } else {
        -1.0
    };

    for _ in 0..50 {
        let lr = one_step_log_ratio(
            graph,
            evaluator,
            q,
            &p0,
            &grad0,
            mass,
            eps,
            logp0,
            ke0,
            &mut p1,
            &mut q1,
            &mut velocity,
            scratch,
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
// This hot-path helper keeps its work buffers explicit to avoid per-step allocation.
#[allow(clippy::too_many_arguments)]
fn one_step_log_ratio(
    graph: &Graph,
    evaluator: &mut Evaluator,
    q: &[f64],
    p0: &[f64],
    grad0: &[f64],
    mass: &MassMatrix,
    eps: f64,
    logp0: f64,
    ke0: f64,
    p1: &mut [f64],
    q1: &mut [f64],
    velocity: &mut [f64],
    scratch: &mut [f64],
) -> f64 {
    let dim = q.len();
    for i in 0..dim {
        p1[i] = p0[i] + 0.5 * eps * grad0[i];
    }
    mass.velocity_into(p1, velocity, scratch);
    for i in 0..dim {
        q1[i] = q[i] + eps * velocity[i];
    }
    evaluator.compute(graph, q1);
    for (momentum, &gradient) in p1.iter_mut().zip(evaluator.grad.iter()).take(dim) {
        *momentum += 0.5 * eps * gradient;
    }
    let logp1 = evaluator.total_logp;
    let ke1 = mass.kinetic_energy(p1, scratch);
    (logp1 - ke1) - (logp0 - ke0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use rand::SeedableRng;

    fn simple_gaussian_graph() -> Graph {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, one);
        graph
    }

    #[test]
    fn hmc_chain_emits_transition_stats() {
        let graph = simple_gaussian_graph();
        let config = HmcConfig {
            step_size: 0.1,
            num_leapfrog_steps: 2,
            num_draws: 3,
            num_warmup: 2,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let chain = run_chain(&graph, &config, &mut rng, None, None);

        assert_eq!(chain.samples.len(), 3);
        assert_eq!(chain.transitions.len(), 5);
        assert_eq!(chain.transitions.iter().filter(|t| t.is_warmup).count(), 2);
        assert_eq!(chain.transitions.iter().filter(|t| !t.is_warmup).count(), 3);
        assert!(chain.transitions.iter().all(|t| t.step_size > 0.0));
        assert!(chain
            .transitions
            .iter()
            .all(|t| t.accept_prob.is_finite() && t.energy_error.is_finite()));
        let posterior_transitions: Vec<_> = chain
            .transitions
            .iter()
            .filter(|transition| !transition.is_warmup)
            .collect();
        assert_eq!(
            chain.divergences,
            posterior_transitions
                .iter()
                .filter(|transition| transition.divergent)
                .count()
        );
        let posterior_accept_rate = posterior_transitions
            .iter()
            .filter(|transition| transition.accepted)
            .count() as f64
            / posterior_transitions.len() as f64;
        assert_eq!(chain.accept_rate, posterior_accept_rate);
    }

    #[test]
    fn hmc_flags_large_finite_energy_errors_as_divergent() {
        let graph = simple_gaussian_graph();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let unstable = run_chain(
            &graph,
            &HmcConfig {
                step_size: 10.0,
                num_leapfrog_steps: 2,
                num_draws: 1,
                num_warmup: 0,
            },
            &mut rng,
            None,
            None,
        );
        assert!(unstable.transitions[0].energy_error.is_finite());
        assert!(unstable.transitions[0].energy_error > MAX_DELTA_H);
        assert!(unstable.transitions[0].divergent);
        assert_eq!(unstable.transitions[0].accept_prob, 0.0);

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let stable = run_chain(
            &graph,
            &HmcConfig {
                step_size: 0.1,
                num_leapfrog_steps: 2,
                num_draws: 1,
                num_warmup: 0,
            },
            &mut rng,
            None,
            None,
        );
        assert!(!stable.transitions[0].divergent);
    }

    #[test]
    fn hmc_non_finite_energy_errors_have_zero_acceptance_probability() {
        assert_eq!(acceptance_probability(f64::INFINITY), 0.0);
        assert_eq!(acceptance_probability(f64::NEG_INFINITY), 0.0);
        assert_eq!(acceptance_probability(f64::NAN), 0.0);

        assert_eq!(acceptance_probability(0.0), 1.0);
        assert_eq!(acceptance_probability(-1.0), 1.0);
        assert!((acceptance_probability(1.0) - (-1.0_f64).exp()).abs() < 1e-15);
    }
}
