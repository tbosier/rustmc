use crate::autodiff::grad_logp;
use crate::graph::Graph;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Configuration for the HMC sampler.
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
            step_size: 0.01,
            num_leapfrog_steps: 20,
            num_draws: 1000,
            num_warmup: 500,
        }
    }
}

/// Result of a single HMC chain run.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub samples: Vec<Vec<f64>>,
    pub accept_rate: f64,
}

/// Run a single HMC chain.
///
/// The graph is taken by shared reference — it is read-only during sampling.
/// Each chain gets its own RNG for reproducibility.
pub fn run_chain(
    graph: &Graph,
    config: &HmcConfig,
    rng: &mut ChaCha8Rng,
    init: Option<Vec<f64>>,
) -> ChainResult {
    let dim = graph.param_count;
    let mut q = init.unwrap_or_else(|| vec![0.0; dim]);
    let total_iters = config.num_warmup + config.num_draws;

    let mut samples = Vec::with_capacity(config.num_draws);
    let mut accepted = 0u64;
    let mut total = 0u64;

    // Dual-averaging step-size adaptation during warmup
    let mut step_size = config.step_size;
    let target_accept = 0.65;
    let mu = (10.0 * step_size).ln();
    let gamma = 0.05;
    let t0 = 10.0;
    let kappa = 0.75;
    let mut log_eps_bar = 0.0f64;
    let mut h_bar = 0.0f64;

    for iter in 0..total_iters {
        let is_warmup = iter < config.num_warmup;

        let (logp_current, grad_current) = grad_logp(graph, &q);

        // Sample momentum
        let p: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();

        // Leapfrog integration
        let mut q_prop = q.clone();
        let mut p_prop = p.clone();
        let mut grad = grad_current;

        // Half step for momentum
        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad[i];
        }

        for step in 0..config.num_leapfrog_steps {
            // Full step for position
            for i in 0..dim {
                q_prop[i] += step_size * p_prop[i];
            }

            // Recompute gradient
            let (_, new_grad) = grad_logp(graph, &q_prop);
            grad = new_grad;

            // Full step for momentum (except at end)
            if step < config.num_leapfrog_steps - 1 {
                for i in 0..dim {
                    p_prop[i] += step_size * grad[i];
                }
            }
        }

        // Half step for momentum at end
        for i in 0..dim {
            p_prop[i] += 0.5 * step_size * grad[i];
        }

        // Negate momentum for reversibility (not needed for acceptance, but correct)
        for pi in &mut p_prop {
            *pi = -*pi;
        }

        // Compute Hamiltonian
        let (logp_prop, _) = grad_logp(graph, &q_prop);
        let ke_current: f64 = p.iter().map(|pi| 0.5 * pi * pi).sum();
        let ke_prop: f64 = p_prop.iter().map(|pi| 0.5 * pi * pi).sum();
        let h_current = -logp_current + ke_current;
        let h_prop = -logp_prop + ke_prop;
        let log_accept_ratio = h_current - h_prop;
        let accept_prob = log_accept_ratio.min(0.0).exp();

        total += 1;
        if log_accept_ratio.is_finite() && rng.gen::<f64>().ln() < log_accept_ratio {
            q = q_prop;
            accepted += 1;
        }

        // Dual averaging adaptation during warmup
        if is_warmup {
            let m = (iter + 1) as f64;
            let w = 1.0 / (m + t0);
            h_bar = (1.0 - w) * h_bar + w * (target_accept - accept_prob);
            let log_eps = mu - (m.sqrt() / gamma) * h_bar;
            step_size = log_eps.exp();
            let m_pow = m.powf(-kappa);
            log_eps_bar = m_pow * log_eps + (1.0 - m_pow) * log_eps_bar;
        }

        // After warmup, fix step size
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

// Future: NUTS (No-U-Turn Sampler) will extend this module with adaptive
// tree-building logic and dual-averaging step-size tuning.
//
// Future: Online/streaming posterior updates can be implemented by allowing
// the sampler to ingest new data batches and continue from the current state.
