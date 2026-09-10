//! Native custom log densities sampled by the same HMC/NUTS kernels as graphs.
//!
//! Positions and gradients are in unconstrained coordinates. Implementations own
//! any transforms and Jacobians. Return `Ok(NEG_INFINITY)` for points outside the
//! target support; return `Err` for an evaluation failure. Prediction is a separate
//! model operation and is deliberately not inferred from a log density.

use crate::autodiff::Evaluator;
use crate::graph::Graph;
use crate::hmc::{self, HmcConfig};
use crate::nuts::{self, NutsConfig};
use crate::sampler::{
    validate_initial_values, with_thread_pool, SampleResult, SamplerConfig, SamplerType,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashSet;

pub trait LogDensity: Sync {
    fn dimension(&self) -> usize;
    /// Write every gradient element, even when an element is zero.
    fn log_density_gradient(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, String>;
    fn parameter_names(&self) -> Vec<String> {
        (0..self.dimension())
            .map(|i| format!("parameter[{i}]"))
            .collect()
    }
}

pub(crate) trait GradientEvaluator {
    fn compute(&mut self, graph: &Graph, position: &[f64]);
    fn log_density(&self) -> f64;
    fn gradient(&self) -> &[f64];
    /// Statically false for graph evaluators, so graph hot loops incur no branch.
    fn has_failed(&self) -> bool {
        false
    }
}

impl GradientEvaluator for Evaluator {
    fn compute(&mut self, graph: &Graph, position: &[f64]) {
        Evaluator::compute(self, graph, position);
    }
    fn log_density(&self) -> f64 {
        self.total_logp
    }
    fn gradient(&self) -> &[f64] {
        &self.grad
    }
}

struct TargetEvaluator<'a, T: LogDensity + ?Sized> {
    target: &'a T,
    gradient: Vec<f64>,
    log_density: f64,
    failure: Option<String>,
}

impl<T: LogDensity + ?Sized> GradientEvaluator for TargetEvaluator<'_, T> {
    fn compute(&mut self, _graph: &Graph, position: &[f64]) {
        if self.failure.is_some() {
            return;
        }
        self.gradient.fill(f64::NAN);
        if position.iter().any(|x| !x.is_finite()) {
            self.log_density = f64::NEG_INFINITY;
            self.gradient.fill(0.0);
            return;
        }
        match self
            .target
            .log_density_gradient(position, &mut self.gradient)
        {
            Ok(value) if value == f64::NEG_INFINITY => {
                self.log_density = value;
                self.gradient.fill(0.0);
            }
            Ok(value) if value.is_finite() && self.gradient.iter().all(|g| g.is_finite()) => {
                self.log_density = value;
            }
            result => {
                self.failure.get_or_insert_with(|| match result {
                    Err(message) => message,
                    _ => "custom target returned a nonfinite density or incomplete/nonfinite gradient".into(),
                });
                self.log_density = f64::NEG_INFINITY;
                self.gradient.fill(0.0);
            }
        }
    }
    fn log_density(&self) -> f64 {
        self.log_density
    }
    fn gradient(&self) -> &[f64] {
        &self.gradient
    }
    fn has_failed(&self) -> bool {
        self.failure.is_some()
    }
}

/// Sample a custom target. Initial positions are raw, with one row per chain.
/// Result draws remain raw; apply the model's transforms for presentation.
pub fn sample_target<T: LogDensity + ?Sized>(
    target: &T,
    config: SamplerConfig,
    initial: Option<Vec<Vec<f64>>>,
) -> Result<SampleResult, String> {
    config.validate()?;
    let dimension = target.dimension();
    let names = target.parameter_names();
    if dimension == 0
        || names.len() != dimension
        || names.iter().any(String::is_empty)
        || names.iter().collect::<HashSet<_>>().len() != dimension
    {
        return Err(
            "custom target needs a positive dimension and unique nonempty parameter names".into(),
        );
    }
    let positions = validate_initial_values(initial, config.num_chains, dimension)?;
    let mut graph = Graph::new();
    for name in &names {
        graph.add_param(name);
    }
    let chains = with_thread_pool(config.num_threads, || {
        positions
            .par_iter()
            .enumerate()
            .map(|(chain, position)| {
                let mut evaluator = TargetEvaluator {
                    target,
                    gradient: vec![0.0; dimension],
                    log_density: f64::NAN,
                    failure: None,
                };
                evaluator.compute(&graph, position);
                if let Some(failure) = evaluator.failure.take() {
                    return Err(format!("chain {chain}: {failure}"));
                }
                if !evaluator.log_density.is_finite() {
                    return Err(format!("chain {chain}: initial log density is not finite"));
                }
                let mut rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(chain as u64));
                let result = match config.sampler {
                    SamplerType::Hmc => hmc::run_chain_with_evaluator(
                        &graph,
                        &HmcConfig {
                            step_size: config.step_size,
                            target_accept: config.target_accept,
                            num_leapfrog_steps: config.num_leapfrog_steps,
                            num_draws: config.num_draws,
                            num_warmup: config.num_warmup,
                        },
                        &mut rng,
                        Some(position.clone()),
                        None,
                        &mut evaluator,
                    ),
                    SamplerType::Nuts => nuts::run_chain_with_evaluator(
                        &graph,
                        &NutsConfig {
                            step_size: config.step_size,
                            target_accept: config.target_accept,
                            max_tree_depth: config.max_tree_depth,
                            num_draws: config.num_draws,
                            num_warmup: config.num_warmup,
                        },
                        &mut rng,
                        Some(position.clone()),
                        None,
                        &mut evaluator,
                    ),
                };
                match evaluator.failure {
                    Some(error) => Err(format!("chain {chain}: {error}")),
                    None => Ok(result),
                }
            })
            .collect::<Result<Vec<_>, String>>()
    })??;
    Ok(SampleResult {
        samples: chains.iter().map(|c| c.samples.clone()).collect(),
        accept_rates: chains.iter().map(|c| c.accept_rate).collect(),
        step_sizes: chains.iter().map(|c| c.step_size).collect(),
        divergences: chains.iter().map(|c| c.divergences).collect(),
        transitions: chains.into_iter().map(|c| c.transitions).collect(),
        param_names: names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Normal;
    impl LogDensity for Normal {
        fn dimension(&self) -> usize {
            1
        }
        fn log_density_gradient(&self, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
            g[0] = -(q[0] - 2.0) / 4.0;
            Ok(-(q[0] - 2.0).powi(2) / 8.0)
        }
    }
    #[test]
    fn custom_targets_reuse_both_samplers_and_recover_gaussian_moments() {
        for sampler in [SamplerType::Nuts, SamplerType::Hmc] {
            let config = SamplerConfig {
                sampler,
                num_chains: 4,
                num_draws: 1200,
                num_warmup: 400,
                show_progress: false,
                seed: 789,
                num_threads: 2,
                num_leapfrog_steps: 1,
                ..Default::default()
            };
            let fit = sample_target(
                &Normal,
                config,
                Some(vec![vec![-2.], vec![0.], vec![3.], vec![5.]]),
            )
            .unwrap();
            assert!(
                (fit.mean()[0] - 2.0).abs() < 0.25,
                "{:?}: mean {:?}, std {:?}, diag {}",
                sampler,
                fit.mean(),
                fit.std(),
                fit.diagnostics().to_table()
            );
            assert!((fit.std()[0] - 2.0).abs() < 0.25);
        }
    }
    #[test]
    fn invalid_gradient_and_initial_shape_are_errors() {
        struct Bad;
        impl LogDensity for Bad {
            fn dimension(&self) -> usize {
                1
            }
            fn log_density_gradient(&self, _: &[f64], _: &mut [f64]) -> Result<f64, String> {
                Ok(0.)
            }
        }
        let config = SamplerConfig {
            show_progress: false,
            ..Default::default()
        };
        assert!(sample_target(&Bad, config.clone(), None)
            .unwrap_err()
            .contains("gradient"));
        assert!(sample_target(&Normal, config, Some(vec![vec![]]))
            .unwrap_err()
            .contains("init"));
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailsAfterInitialization {
        calls: AtomicUsize,
    }
    impl LogDensity for FailsAfterInitialization {
        fn dimension(&self) -> usize {
            1
        }
        fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
            if self.calls.fetch_add(1, Ordering::Relaxed) >= 2 {
                return Err("backend unavailable".into());
            }
            gradient[0] = -q[0];
            Ok(-0.5 * q[0] * q[0])
        }
    }
    #[test]
    fn evaluation_failures_abort_chain_and_do_not_repeat_callbacks() {
        for sampler in [SamplerType::Hmc, SamplerType::Nuts] {
            let target = FailsAfterInitialization {
                calls: AtomicUsize::new(0),
            };
            let config = SamplerConfig {
                sampler,
                num_chains: 1,
                num_draws: 500,
                num_warmup: 20,
                step_size: 0.1,
                num_threads: 1,
                show_progress: false,
                ..Default::default()
            };
            let error = sample_target(&target, config, None).unwrap_err();
            assert!(error.contains("backend unavailable"));
            assert_eq!(target.calls.load(Ordering::Relaxed), 3);
        }
    }

    struct TruncatedNormal {
        outside: AtomicUsize,
    }
    impl LogDensity for TruncatedNormal {
        fn dimension(&self) -> usize {
            1
        }
        fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
            if q[0].abs() > 1.0 {
                self.outside.fetch_add(1, Ordering::Relaxed);
                return Ok(f64::NEG_INFINITY);
            }
            gradient[0] = -q[0];
            Ok(-0.5 * q[0] * q[0])
        }
    }
    #[test]
    fn support_rejections_are_not_errors_and_thread_counts_preserve_draws() {
        for sampler in [SamplerType::Hmc, SamplerType::Nuts] {
            let target = TruncatedNormal {
                outside: AtomicUsize::new(0),
            };
            let config = SamplerConfig {
                sampler,
                num_chains: 2,
                num_draws: 100,
                num_warmup: 30,
                step_size: 0.2,
                num_threads: 1,
                show_progress: false,
                ..Default::default()
            };
            let single = sample_target(&target, config.clone(), None).unwrap();
            let parallel = sample_target(
                &target,
                SamplerConfig {
                    num_threads: 2,
                    ..config
                },
                None,
            )
            .unwrap();
            assert_eq!(single.samples, parallel.samples);
            assert!(single.samples.iter().flatten().all(|q| q[0].abs() <= 1.0));
            assert!(target.outside.load(Ordering::Relaxed) > 0);
        }
    }
}
