use crate::diagnostics::{self, DiagnosticsReport};
use crate::graph::Graph;
use crate::hmc::{self, ChainResult, HmcConfig, TransitionStats};
use crate::nuts::{self, NutsConfig};
use crate::progress::{self, ProgressState};
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
            num_leapfrog_steps: 15,
            max_tree_depth: 8,
            seed: 42,
            show_progress: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SampleResult {
    pub samples: Vec<Vec<Vec<f64>>>,
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

fn with_thread_pool<T, F>(num_threads: usize, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    if num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build Rayon thread pool");
        pool.install(f)
    } else {
        f()
    }
}

pub fn sample(graph: Graph, config: SamplerConfig) -> Result<SampleResult, String> {
    graph.validate_shapes().map_err(|e| e.to_string())?;
    let graph = Arc::new(graph);
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

    let progress_handle = progress_state
        .as_ref()
        .map(|ps| progress::spawn_progress_thread(Arc::clone(ps)));

    let chain_indices: Vec<usize> = (0..config.num_chains).collect();

    let results: Vec<ChainResult> = with_thread_pool(config.num_threads, || {
        chain_indices
            .par_iter()
            .map(|&chain_idx| {
                let mut rng = ChaCha8Rng::seed_from_u64(config.seed + chain_idx as u64);
                let prog_ref = progress_state.as_deref();

                match config.sampler {
                    SamplerType::Nuts => {
                        let nuts_config = NutsConfig {
                            step_size: config.step_size,
                            max_tree_depth: config.max_tree_depth,
                            num_draws: config.num_draws,
                            num_warmup: config.num_warmup,
                        };
                        nuts::run_chain(&graph, &nuts_config, &mut rng, None, prog_ref)
                    }
                    SamplerType::Hmc => {
                        let hmc_config = HmcConfig {
                            step_size: config.step_size,
                            num_leapfrog_steps: config.num_leapfrog_steps,
                            num_draws: config.num_draws,
                            num_warmup: config.num_warmup,
                        };
                        hmc::run_chain(&graph, &hmc_config, &mut rng, None, prog_ref)
                    }
                }
            })
            .collect()
    });

    if let Some(ps) = &progress_state {
        ps.finish();
    }
    if let Some(h) = progress_handle {
        h.join().ok();
    }

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

    let accept_rates: Vec<f64> = results.iter().map(|r| r.accept_rate).collect();
    let step_sizes: Vec<f64> = results.iter().map(|r| r.step_size).collect();
    let divergences: Vec<usize> = results.iter().map(|r| r.divergences).collect();
    let transitions: Vec<Vec<TransitionStats>> =
        results.iter().map(|r| r.transitions.clone()).collect();

    Ok(SampleResult {
        samples,
        accept_rates,
        step_sizes,
        divergences,
        transitions,
        param_names,
    })
}

/// Lightweight result for a single model in a batch run (1 chain).
#[derive(Debug, Clone)]
pub struct BatchModelResult {
    pub samples: Vec<Vec<f64>>,
    pub param_names: Vec<String>,
    pub num_chains: usize,
    pub num_draws: usize,
    pub accept_rates: Vec<f64>,
    pub step_sizes: Vec<f64>,
    pub divergences: Vec<usize>,
    pub transitions: Vec<Vec<TransitionStats>>,
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
    let n_models = models.len();
    let chains_per_model = config.num_chains.max(1);
    let total_chain_runs = n_models * chains_per_model;

    for (graph, _) in &models {
        graph.validate_shapes().map_err(|e| e.to_string())?;
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

    let progress_handle = progress_state
        .as_ref()
        .map(|ps| progress::spawn_progress_thread(Arc::clone(ps)));

    let results: Vec<BatchModelResult> = with_thread_pool(0, || {
        models
            .into_par_iter()
            .enumerate()
            .map(|(model_idx, (graph, obs_y))| {
                let _ = obs_y; // data is already baked into the graph
                let prog_ref = progress_state.as_deref();
                let mut samples: Vec<Vec<f64>> = Vec::new();
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
                                max_tree_depth: config.max_tree_depth,
                                num_draws: config.num_draws,
                                num_warmup: config.num_warmup,
                            };
                            nuts::run_chain(&graph, &nuts_config, &mut rng, None, prog_ref)
                        }
                        SamplerType::Hmc => {
                            let hmc_config = HmcConfig {
                                step_size: config.step_size,
                                num_leapfrog_steps: config.num_leapfrog_steps,
                                num_draws: config.num_draws,
                                num_warmup: config.num_warmup,
                            };
                            hmc::run_chain(&graph, &hmc_config, &mut rng, None, prog_ref)
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
                }

                BatchModelResult {
                    samples,
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
    });

    if let Some(ps) = &progress_state {
        ps.finish();
    }
    if let Some(h) = progress_handle {
        h.join().ok();
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::current_num_threads;

    #[test]
    fn thread_pool_helper_uses_requested_parallelism() {
        let one = with_thread_pool(1, current_num_threads);
        let two = with_thread_pool(2, current_num_threads);

        assert_eq!(one, 1);
        assert_eq!(two, 2);
    }

    #[test]
    fn sample_result_exposes_transition_diagnostics() {
        let result = SampleResult {
            samples: vec![],
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
}
