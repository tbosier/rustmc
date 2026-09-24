use crate::adaptation::WarmupAdapter;
use crate::autodiff::Evaluator;
use crate::data::DataBinding;
use crate::graph::Graph;
use crate::mass_matrix::{MassMatrix, MetricKind};
use crate::progress::ProgressState;
use crate::sampler::{kernel_initial_position, reject_discrete_latent_parameters};
use crate::target::GradientEvaluator;
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
    /// HMC: the Metropolis proposal was accepted. NUTS: the tree did not
    /// diverge; multinomial sampling may still retain the initial state.
    pub accepted: bool,
    pub accept_prob: f64,
    /// HMC proposal energy error, or NUTS selected-state energy error (which
    /// can be zero when the initial state is selected).
    pub energy_error: f64,
    pub divergent: bool,
    pub step_size: f64,
    pub num_leapfrog_steps: usize,
    /// Number of attempted NUTS doubling expansions, including the expansion
    /// that terminates with a U-turn or divergence.
    pub tree_depth: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct HmcConfig {
    pub step_size: f64,
    /// Desired average Metropolis acceptance probability during adaptation.
    pub target_accept: f64,
    pub num_leapfrog_steps: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    /// How warmup estimates the metric of vector parameters.
    pub metric: MetricKind,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            step_size: 0.0, // 0 = auto-detect
            target_accept: 0.80,
            num_leapfrog_steps: 15,
            num_draws: 1000,
            num_warmup: 500,
            metric: MetricKind::Auto,
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
/// Warmup follows Stan's windowed schedule, shared with NUTS: step-size
/// adaptation only in the initial and terminal buffers, and doubling
/// metric-estimation windows between them (see `adaptation::WarmupSchedule`).
///
/// Workspace buffers are allocated once per chain; a transition allocates
/// only the retained draw.
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
    config: &HmcConfig,
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
    config: &HmcConfig,
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
    config: &HmcConfig,
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
    config: &HmcConfig,
    rng: &mut ChaCha8Rng,
    init: Vec<f64>,
    progress: Option<&ProgressState>,
    evaluator: &mut impl GradientEvaluator,
) -> ChainResult {
    let dim = graph.param_count;
    let total_iters = config.num_warmup + config.num_draws;

    let mut q = init;
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
    // The same windowed schedule, dual averaging and step-size search as NUTS.
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

    // The density and gradient at the current position are cached across
    // iterations: a rejected proposal leaves them unchanged and an accepted
    // one already evaluated them at its endpoint.
    evaluator.compute(graph, &q);
    let mut logp_current = evaluator.log_density();
    let mut grad_current = evaluator.gradient().to_vec();

    'iterations: for iter in 0..total_iters {
        if evaluator.has_failed() {
            break;
        }
        let is_warmup = iter < config.num_warmup;
        let step_size_used = step_size;

        mass.sample_momentum_into(rng, &mut p, &mut scratch);

        q_prop.copy_from_slice(&q);
        p_prop.copy_from_slice(&p);

        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad_current[i];
        }

        for step in 0..config.num_leapfrog_steps {
            mass.velocity_into(&p_prop, &mut velocity, &mut scratch);
            for i in 0..dim {
                q_prop[i] += step_size * velocity[i];
            }

            evaluator.compute(graph, &q_prop);
            if evaluator.has_failed() {
                break 'iterations;
            }
            grad.copy_from_slice(evaluator.gradient());

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

        let logp_prop = evaluator.log_density();
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
            grad_current.copy_from_slice(&grad);
            logp_current = logp_prop;
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
            step_size = adapter.after_transition(
                iter,
                accept_prob,
                &q,
                graph,
                evaluator,
                &mut mass,
                rng,
                &mut scratch,
            );
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
    fn metric_reset_recenters_step_size_adaptation() {
        let graph = simple_gaussian_graph();
        let config = HmcConfig {
            step_size: 0.001,
            num_leapfrog_steps: 3,
            num_draws: 2,
            num_warmup: 100,
            ..HmcConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let chain = run_chain(&graph, &config, &mut rng, Some(vec![0.0]), None)
            .expect("continuous test model must run");
        // Warmup 100 is too short for Stan's default buffers, so the schedule
        // falls back to 15% / 75% / 10%: one window [15, 90), with the metric
        // replaced after transition 89.
        let first_after_reset = &chain.transitions[90];
        let second_after_reset = &chain.transitions[91];
        assert!((first_after_reset.step_size - config.step_size).abs() > 1e-3);
        // The first update of the new adaptation phase is centered on the
        // initial step found with the new metric, not the pre-warmup step.
        let expected = 10.0
            * first_after_reset.step_size
            * (-(config.target_accept - first_after_reset.accept_prob) / (11.0 * 0.05)).exp();
        assert!((second_after_reset.step_size - expected).abs() < 1e-10);
    }

    #[test]
    fn hmc_chain_emits_transition_stats() {
        let graph = simple_gaussian_graph();
        let config = HmcConfig {
            step_size: 0.1,
            target_accept: 0.80,
            num_leapfrog_steps: 2,
            num_draws: 3,
            num_warmup: 2,
            metric: MetricKind::Auto,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let chain = run_chain(&graph, &config, &mut rng, None, None)
            .expect("continuous test model must run");

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
                target_accept: 0.80,
                num_leapfrog_steps: 2,
                num_draws: 1,
                num_warmup: 0,
                metric: MetricKind::Auto,
            },
            &mut rng,
            None,
            None,
        )
        .expect("continuous test model must run");
        assert!(unstable.transitions[0].energy_error.is_finite());
        assert!(unstable.transitions[0].energy_error > MAX_DELTA_H);
        assert!(unstable.transitions[0].divergent);
        assert_eq!(unstable.transitions[0].accept_prob, 0.0);

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let stable = run_chain(
            &graph,
            &HmcConfig {
                step_size: 0.1,
                target_accept: 0.80,
                num_leapfrog_steps: 2,
                num_draws: 1,
                num_warmup: 0,
                metric: MetricKind::Auto,
            },
            &mut rng,
            None,
            None,
        )
        .expect("continuous test model must run");
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
