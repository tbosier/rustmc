//! Native custom log densities sampled by the same HMC/NUTS kernels as graphs.
//!
//! Positions and gradients are in unconstrained coordinates. Implementations own
//! any transforms and Jacobians. Return `Ok(NEG_INFINITY)` for points outside the
//! target support; return `Err` for an evaluation failure. Prediction is a separate
//! model operation and is deliberately not inferred from a log density.
//!
//! Only `Err` ends a fit. A density that is NaN or `+inf`, or a gradient element
//! that is non-finite or was never written, makes that point unusable, as `-inf`
//! does: inside a trajectory the kernels record it as a divergence or a rejected
//! proposal, as they do for graph models, and a random start retries elsewhere.
//! An explicit `init` must be usable, so there it is an error.
//!
//! A custom target's parameters are all scalars, so `MetricKind::Dense` and
//! `MetricKind::Auto`, which only estimate dense blocks within vector
//! parameters, give it a diagonal metric.

use crate::autodiff::Evaluator;
use crate::graph::Graph;
use crate::sampler::{
    chain_rng, check_retained_draws, random_initial_position, run_kernel, validate_initial_values,
    with_thread_pool, SampleResult, SamplerConfig,
};
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

impl<T: LogDensity + ?Sized> TargetEvaluator<'_, T> {
    /// Call the target at `position`, leaving its gradient in `self.gradient`.
    /// Elements the target did not write are NaN.
    fn evaluate(&mut self, position: &[f64]) -> Result<f64, String> {
        self.gradient.fill(f64::NAN);
        self.target
            .log_density_gradient(position, &mut self.gradient)
    }

    /// Whether the last [`Self::evaluate`] returned `value` with a finite
    /// density and a finite, fully written gradient.
    fn usable(&self, value: f64) -> bool {
        value.is_finite() && self.gradient.iter().all(|g| g.is_finite())
    }

    fn mark_outside_support(&mut self) {
        self.log_density = f64::NEG_INFINITY;
        self.gradient.fill(0.0);
    }
}

impl<T: LogDensity + ?Sized> GradientEvaluator for TargetEvaluator<'_, T> {
    fn compute(&mut self, _graph: &Graph, position: &[f64]) {
        if self.failure.is_some() {
            return;
        }
        if position.iter().any(|x| !x.is_finite()) {
            self.mark_outside_support();
            return;
        }
        match self.evaluate(position) {
            Ok(value) if self.usable(value) => self.log_density = value,
            // NaN, +inf or a bad gradient is treated like -inf: the kernels
            // see a point they cannot move to, not a failed fit.
            Ok(_) => self.mark_outside_support(),
            Err(message) => {
                self.failure = Some(message);
                self.mark_outside_support();
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

/// Sample a custom target. Initial positions are raw, with one row per chain;
/// `None` draws each chain's start uniformly from (-2, 2) as graph fits do.
/// Result draws remain raw; apply the model's transforms for presentation.
///
/// Every parameter is a scalar here, so `MetricKind::Dense` and
/// `MetricKind::Auto` adapt a diagonal metric; see the module docs for how
/// non-finite values are treated.
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
    check_retained_draws(config.num_chains, config.num_draws, dimension)?;
    let positions = validate_initial_values(initial, config.num_chains, dimension)?;
    let mut graph = Graph::new();
    for name in &names {
        graph.add_param(name);
    }
    let chains = with_thread_pool(config.num_threads, || {
        (0..config.num_chains)
            .into_par_iter()
            .map(|chain| {
                let mut evaluator = TargetEvaluator {
                    target,
                    gradient: vec![0.0; dimension],
                    log_density: f64::NAN,
                    failure: None,
                };
                let position = match &positions {
                    // A supplied start must be usable; unlike a random one,
                    // there is no other point to try.
                    Some(positions) => {
                        let position = positions[chain].clone();
                        let value = evaluator
                            .evaluate(&position)
                            .map_err(|message| format!("chain {chain}: {message}"))?;
                        if !value.is_finite() {
                            return Err(format!(
                                "chain {chain}: initial log density is not finite"
                            ));
                        }
                        if !evaluator.usable(value) {
                            return Err(format!(
                                "chain {chain}: custom target returned an incomplete or \
                                 nonfinite gradient at the initial point"
                            ));
                        }
                        position
                    }
                    // A non-finite density or gradient only rules out this
                    // point, as it does for graph models. An `Err` is the
                    // target reporting that it cannot evaluate at all, which
                    // another point will not fix.
                    None => random_initial_position(config.seed, chain, dimension, |position| {
                        let value = evaluator
                            .evaluate(position)
                            .map_err(|message| format!("chain {chain}: {message}"))?;
                        Ok(evaluator.usable(value))
                    })?,
                };
                let mut rng = chain_rng(config.seed, chain);
                let result = run_kernel(&config, &graph, &mut rng, position, None, &mut evaluator);
                match evaluator.failure {
                    Some(error) => Err(format!("chain {chain}: {error}")),
                    None => Ok(result),
                }
            })
            .collect::<Result<Vec<_>, String>>()
    })??;
    Ok(SampleResult::from_chains(chains, names))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SamplerType;
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
    use crate::sampler::SamplerType;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A standard normal whose gradient is NaN for x < -1: usable from the
    /// origin, so a random start that lands there must be retried rather than
    /// failing the fit.
    struct NanGradientBelowMinusOne;
    impl LogDensity for NanGradientBelowMinusOne {
        fn dimension(&self) -> usize {
            1
        }
        fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
            gradient[0] = if q[0] < -1.0 { f64::NAN } else { -q[0] };
            Ok(-0.5 * q[0] * q[0])
        }
    }

    #[test]
    fn random_starts_retry_points_with_a_non_finite_gradient() {
        // Chains start uniformly on (-2, 2), so a quarter of first draws land
        // in the NaN region; with eight chains at least one does.
        let fit = sample_target(
            &NanGradientBelowMinusOne,
            SamplerConfig {
                num_chains: 8,
                num_draws: 5,
                num_warmup: 5,
                // Chains barely move, so any failure is the start search's.
                step_size: 1e-6,
                max_tree_depth: 1,
                num_threads: 1,
                show_progress: false,
                ..Default::default()
            },
            None,
        )
        .expect("an unusable random start must be retried, not fail the fit");
        assert!(fit.samples.iter().flatten().all(|q| q[0] >= -1.0));
    }

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

    /// A standard normal whose density overflows to `+inf`, with an infinite
    /// gradient, beyond |x| = 3: a tail a large early step reaches.
    struct OverflowingTail {
        tail: AtomicUsize,
    }
    impl LogDensity for OverflowingTail {
        fn dimension(&self) -> usize {
            1
        }
        fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
            if q[0].abs() > 3.0 {
                self.tail.fetch_add(1, Ordering::Relaxed);
                gradient[0] = f64::INFINITY;
                return Ok(f64::INFINITY);
            }
            gradient[0] = -q[0];
            Ok(-0.5 * q[0] * q[0])
        }
    }

    #[test]
    fn non_finite_values_inside_a_trajectory_are_rejections_not_errors() {
        for sampler in [SamplerType::Hmc, SamplerType::Nuts] {
            let target = OverflowingTail {
                tail: AtomicUsize::new(0),
            };
            let config = SamplerConfig {
                sampler,
                num_chains: 2,
                num_draws: 200,
                num_warmup: 100,
                num_threads: 1,
                show_progress: false,
                ..Default::default()
            };
            let fit = sample_target(&target, config.clone(), None)
                .expect("a non-finite density inside a trajectory must not fail the fit");
            assert!(target.tail.load(Ordering::Relaxed) > 0, "{sampler:?}");
            assert!(fit.samples.iter().flatten().all(|q| q[0].abs() <= 3.0));
            // A supplied start in that tail is still refused.
            let error =
                sample_target(&target, config, Some(vec![vec![4.0], vec![0.0]])).unwrap_err();
            assert!(error.contains("initial log density"), "{error}");
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
