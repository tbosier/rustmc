//! Native custom log densities sampled by the same HMC/NUTS kernels as graphs.
//!
//! Positions and gradients are in unconstrained coordinates. Implementations own
//! any transforms and Jacobians. Return `Ok(NEG_INFINITY)` for points outside the
//! target support; return `Err` for an evaluation failure. Prediction is a separate
//! model operation and is deliberately not inferred from a log density.
//!
//! Only `Err` ends a fit. A density that is NaN or `+inf`, or a gradient element
//! that is non-finite or was never written, is handled exactly as `-inf` is, and
//! as a graph model's `-inf` is: the point counts as outside the support, with a
//! zero gradient. A trajectory that reaches such a point is a divergence and
//! never moves to it: a NUTS leaf there ends the tree, whose draw comes from
//! the points before it, and an HMC trajectory stops there and its proposal is
//! rejected, wherever its remaining steps would have ended. A random start
//! retries elsewhere; an explicit `init` must be usable, so there it is an
//! error.
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

    /// Flat with a zero gradient on (-10, 10), except for a NaN band on
    /// (0.9, 1.1), and `-inf` outside. Every point asked about is logged in
    /// order, so a test can replay each trajectory.
    struct NanBand {
        calls: std::sync::Mutex<Vec<f64>>,
    }
    impl NanBand {
        fn outside(x: f64) -> bool {
            (x > 0.9 && x < 1.1) || x.abs() >= 10.0
        }
    }
    impl LogDensity for NanBand {
        fn dimension(&self) -> usize {
            1
        }
        fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
            self.calls.lock().unwrap().push(q[0]);
            gradient[0] = 0.0;
            Ok(if q[0] > 0.9 && q[0] < 1.1 {
                f64::NAN
            } else if q[0].abs() < 10.0 {
                0.0
            } else {
                f64::NEG_INFINITY
            })
        }
    }

    /// Run one chain of `sampler` on [`NanBand`] from the origin with unit
    /// metric, step size 1, `num_leapfrog_steps` HMC steps and no warmup,
    /// and return it with the log of the points each transition evaluated.
    fn nan_band_chain(
        sampler: SamplerType,
        num_leapfrog_steps: usize,
    ) -> (crate::hmc::ChainResult, Vec<Vec<f64>>) {
        let target = NanBand {
            calls: std::sync::Mutex::new(Vec::new()),
        };
        let mut graph = Graph::new();
        graph.add_param("x");
        let mut evaluator = TargetEvaluator {
            target: &target,
            gradient: vec![0.0],
            log_density: f64::NAN,
            failure: None,
        };
        let config = SamplerConfig {
            sampler,
            num_draws: 2000,
            num_warmup: 0,
            step_size: 1.0,
            num_leapfrog_steps,
            max_tree_depth: 4,
            ..Default::default()
        };
        let mut rng = crate::sampler::chain_rng(5, 0);
        let chain = run_kernel(&config, &graph, &mut rng, vec![0.0], None, &mut evaluator);
        assert!(!evaluator.has_failed());
        let calls = target.calls.into_inner().unwrap();
        // One evaluation at the start, then every transition's leapfrog
        // steps in order: no step-size search or adaptation runs.
        assert_eq!(calls[0], 0.0);
        let mut log = &calls[1..];
        let mut trajectories = Vec::new();
        for transition in &chain.transitions {
            let (trajectory, rest) = log.split_at(transition.num_leapfrog_steps);
            trajectories.push(trajectory.to_vec());
            log = rest;
        }
        assert!(log.is_empty(), "unaccounted evaluations: {log:?}");
        (chain, trajectories)
    }

    #[test]
    fn an_hmc_trajectory_through_a_point_outside_the_support_is_a_rejected_divergence() {
        for steps in [2, 3] {
            let (chain, trajectories) = nan_band_chain(SamplerType::Hmc, steps);
            let mut start = 0.0;
            // Trajectories that left the support at step k + 1 and would have
            // ended back inside it: with two steps, the trace from review
            // (from q with momentum p, q + p is in the band and q + 2p is not).
            let mut through_and_back = vec![0; steps - 1];
            for ((transition, draw), trajectory) in chain
                .transitions
                .iter()
                .zip(&chain.samples)
                .zip(&trajectories)
            {
                let left = trajectory.iter().any(|&x| NanBand::outside(x));
                assert_eq!(transition.divergent, left, "{trajectory:?} from {start}");
                if left {
                    assert!(!transition.accepted, "{trajectory:?} from {start}");
                    assert_eq!(transition.energy_error, f64::INFINITY);
                    assert_eq!(transition.accept_prob, 0.0);
                    assert_eq!(draw[0], start);
                    // The trajectory stops at the point outside the support.
                    assert!(NanBand::outside(*trajectory.last().unwrap()));
                    let p = trajectory[0] - start;
                    let stopped_at = trajectory.len();
                    if stopped_at < steps && !NanBand::outside(start + steps as f64 * p) {
                        through_and_back[stopped_at - 1] += 1;
                    }
                } else {
                    // A flat density has no energy error inside the support.
                    assert!(transition.accepted);
                    assert_eq!(trajectory.len(), steps);
                }
                assert!(!NanBand::outside(draw[0]));
                start = draw[0];
            }
            assert!(
                through_and_back.iter().all(|&n| n > 0),
                "{steps} steps: {through_and_back:?}"
            );
        }
    }

    #[test]
    fn a_nuts_leaf_outside_the_support_ends_the_tree_as_a_divergence() {
        let (chain, trajectories) = nan_band_chain(SamplerType::Nuts, 2);
        let mut divergences = 0;
        for ((transition, draw), trajectory) in chain
            .transitions
            .iter()
            .zip(&chain.samples)
            .zip(&trajectories)
        {
            let outside: Vec<bool> = trajectory.iter().map(|&x| NanBand::outside(x)).collect();
            assert_eq!(transition.divergent, outside.contains(&true));
            // Nothing is evaluated after the first leaf outside the support.
            assert!(!outside[..outside.len() - 1].contains(&true));
            assert!(!NanBand::outside(draw[0]));
            divergences += usize::from(transition.divergent);
        }
        assert!(divergences > 0);
        assert!(trajectories
            .iter()
            .any(|t| t.last().is_some_and(|&x| x > 0.9 && x < 1.1)));
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
