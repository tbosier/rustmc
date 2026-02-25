use crate::graph::Graph;
use crate::hmc::{self, ChainResult, HmcConfig};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Arc;

/// Configuration for the multi-chain sampler.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub step_size: f64,
    pub num_leapfrog_steps: usize,
    pub seed: u64,
    /// Number of threads. 0 means use Rayon's default (all cores).
    pub num_threads: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            num_chains: 4,
            num_draws: 1000,
            num_warmup: 500,
            step_size: 0.0,
            num_leapfrog_steps: 15,
            seed: 42,
            num_threads: 0,
        }
    }
}

/// Result of sampling across all chains.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// samples[chain][draw][param]
    pub samples: Vec<Vec<Vec<f64>>>,
    pub accept_rates: Vec<f64>,
    pub param_names: Vec<String>,
}

impl SampleResult {
    /// Get posterior mean for each parameter.
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

    /// Get posterior standard deviation for each parameter.
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
}

/// Run parallel HMC chains on the given graph.
///
/// The graph is wrapped in an `Arc` and shared read-only across all chains.
/// Each chain gets a deterministic RNG seeded from `config.seed + chain_index`,
/// guaranteeing reproducible results regardless of thread scheduling.
pub fn sample(graph: Graph, config: SamplerConfig) -> SampleResult {
    if config.num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .ok();
    }

    let graph = Arc::new(graph);
    let param_names = graph.param_names.clone();

    let hmc_config = HmcConfig {
        step_size: config.step_size,
        num_leapfrog_steps: config.num_leapfrog_steps,
        num_draws: config.num_draws,
        num_warmup: config.num_warmup,
    };

    let chain_indices: Vec<usize> = (0..config.num_chains).collect();

    let results: Vec<ChainResult> = chain_indices
        .par_iter()
        .map(|&chain_idx| {
            let mut rng = ChaCha8Rng::seed_from_u64(config.seed + chain_idx as u64);
            hmc::run_chain(&graph, &hmc_config, &mut rng, None)
        })
        .collect();

    let samples: Vec<Vec<Vec<f64>>> = results.iter().map(|r| r.samples.clone()).collect();
    let accept_rates: Vec<f64> = results.iter().map(|r| r.accept_rate).collect();

    SampleResult {
        samples,
        accept_rates,
        param_names,
    }
}

// Future: Particle filtering can be integrated as an alternative sampler
// that reuses the same Graph and autodiff infrastructure.
//
// Future: Distributed posterior aggregation for multi-machine setups can
// be added by serializing SampleResult and merging across nodes.
