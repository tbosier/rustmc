use crate::diagnostics::{self, DiagnosticsReport};
use crate::graph::Graph;
use crate::hmc::{self, ChainResult, HmcConfig};
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
pub struct SampleResult {
    pub samples: Vec<Vec<Vec<f64>>>,
    pub accept_rates: Vec<f64>,
    pub step_sizes: Vec<f64>,
    pub divergences: Vec<usize>,
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
}

pub fn sample(graph: Graph, config: SamplerConfig) -> SampleResult {
    if config.num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .ok();
    }

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

    let results: Vec<ChainResult> = chain_indices
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
        .collect();

    if let Some(ps) = &progress_state {
        ps.finish();
    }
    if let Some(h) = progress_handle {
        h.join().ok();
    }

    let samples: Vec<Vec<Vec<f64>>> = results.iter().map(|r| r.samples.clone()).collect();
    let accept_rates: Vec<f64> = results.iter().map(|r| r.accept_rate).collect();
    let step_sizes: Vec<f64> = results.iter().map(|r| r.step_size).collect();
    let divergences: Vec<usize> = results.iter().map(|r| r.divergences).collect();

    SampleResult {
        samples,
        accept_rates,
        step_sizes,
        divergences,
        param_names,
    }
}
