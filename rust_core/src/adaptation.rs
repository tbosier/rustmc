//! Warmup adaptation shared by the HMC and NUTS kernels.
//!
//! Both kernels adapt the same way: Stan's windowed schedule decides when
//! the metric is re-estimated, Nesterov dual averaging tunes the step size
//! throughout warmup, and every metric update is followed by a fresh
//! step-size search under the new geometry. Keeping one implementation means
//! the two kernels cannot drift apart.

use crate::graph::Graph;
use crate::mass_matrix::{MassMatrix, MassMatrixAccumulator, MetricKind};
use crate::target::GradientEvaluator;
use rand_chacha::ChaCha8Rng;

/// Stan's default initial fast-adaptation buffer.
const INIT_BUFFER: usize = 75;
/// Stan's default terminal fast-adaptation buffer.
const TERM_BUFFER: usize = 50;
/// Stan's default first slow-adaptation window.
const BASE_WINDOW: usize = 25;
/// Below this many warmup iterations Stan adapts only the step size.
const MIN_ADAPTATION_WARMUP: usize = 20;

/// Stan's windowed warmup schedule: an initial buffer, a run of doubling
/// metric-estimation windows, and a terminal buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WarmupSchedule {
    /// Half-open `[start, end)` iteration ranges whose positions estimate the
    /// metric. The metric is replaced at the end of each window.
    pub windows: Vec<(usize, usize)>,
}

impl WarmupSchedule {
    /// The schedule Stan's `windowed_adaptation` produces for `num_warmup`.
    ///
    /// When the default buffers and base window do not fit, Stan falls back to
    /// 15% / 75% / 10%. A window whose successor could not reach twice its own
    /// size before the terminal buffer is extended to the terminal buffer
    /// instead, so the last estimate uses the longest window available rather
    /// than a truncated one.
    pub(crate) fn stan(num_warmup: usize) -> Self {
        if num_warmup < MIN_ADAPTATION_WARMUP {
            return Self {
                windows: Vec::new(),
            };
        }
        let (init_buffer, term_buffer, base_window) =
            if INIT_BUFFER + BASE_WINDOW + TERM_BUFFER > num_warmup {
                let init = num_warmup * 15 / 100;
                let term = num_warmup / 10;
                (init, term, num_warmup - init - term)
            } else {
                (INIT_BUFFER, TERM_BUFFER, BASE_WINDOW)
            };
        let slow_end = num_warmup - term_buffer;
        let mut windows = Vec::new();
        let mut start = init_buffer;
        let mut size = base_window;
        let mut end = start + size;
        loop {
            windows.push((start, end));
            if end >= slow_end {
                break;
            }
            start = end;
            size *= 2;
            end = start + size;
            // Stan compares inclusive window ends: `next_end_incl + 2 * size
            // >= slow_end` is `end + 2 * size > slow_end` here.
            if end + 2 * size > slow_end {
                end = slow_end;
            }
        }
        Self { windows }
    }
}

/// Nesterov dual averaging of `log(step_size)` (Hoffman & Gelman 2014, §3.2),
/// with Stan's constants.
#[derive(Debug, Clone)]
pub(crate) struct DualAveraging {
    target_accept: f64,
    mu: f64,
    log_step_bar: f64,
    h_bar: f64,
    count: u64,
}

impl DualAveraging {
    const GAMMA: f64 = 0.05;
    const T0: f64 = 10.0;
    const KAPPA: f64 = 0.75;

    pub(crate) fn new(step_size: f64, target_accept: f64) -> Self {
        let mut state = Self {
            target_accept,
            mu: 0.0,
            log_step_bar: 0.0,
            h_bar: 0.0,
            count: 0,
        };
        state.restart(step_size);
        state
    }

    /// Re-centre on `step_size` and forget the accumulated statistics, as
    /// Stan does after each metric update.
    pub(crate) fn restart(&mut self, step_size: f64) {
        self.mu = (10.0 * step_size).ln();
        self.log_step_bar = step_size.ln();
        self.h_bar = 0.0;
        self.count = 0;
    }

    /// Record one transition's acceptance statistic and return the step size
    /// to use next.
    pub(crate) fn update(&mut self, accept_stat: f64) -> f64 {
        self.count += 1;
        let m = self.count as f64;
        let w = 1.0 / (m + Self::T0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - accept_stat.min(1.0));
        let log_step = self.mu - (m.sqrt() / Self::GAMMA) * self.h_bar;
        let m_pow = m.powf(-Self::KAPPA);
        self.log_step_bar = m_pow * log_step + (1.0 - m_pow) * self.log_step_bar;
        log_step.exp()
    }

    /// The averaged step size, fixed for sampling at the end of warmup.
    pub(crate) fn adapted_step_size(&self) -> f64 {
        self.log_step_bar.exp()
    }
}

/// Everything warmup adapts, driven one transition at a time.
pub(crate) struct WarmupAdapter {
    num_warmup: usize,
    windows: Vec<(usize, usize)>,
    next_window: usize,
    metric: MetricKind,
    dual: DualAveraging,
    accumulator: MassMatrixAccumulator,
    workspace: StepSizeWorkspace,
}

impl WarmupAdapter {
    pub(crate) fn new(
        graph: &Graph,
        num_warmup: usize,
        target_accept: f64,
        initial_step_size: f64,
        metric: MetricKind,
    ) -> Self {
        let windows = WarmupSchedule::stan(num_warmup).windows;
        let first_len = windows.first().map_or(0, |(start, end)| end - start);
        Self {
            num_warmup,
            next_window: 0,
            metric,
            dual: DualAveraging::new(initial_step_size, target_accept),
            accumulator: MassMatrixAccumulator::for_window(graph, metric, first_len),
            workspace: StepSizeWorkspace::new(graph.param_count),
            windows,
        }
    }

    /// Search for an initial step size from the chain's starting point.
    pub(crate) fn initial_step_size(
        &mut self,
        graph: &Graph,
        evaluator: &mut impl GradientEvaluator,
        q: &[f64],
        mass: &MassMatrix,
        rng: &mut ChaCha8Rng,
        scratch: &mut [f64],
    ) -> f64 {
        let step_size = find_reasonable_step_size(
            graph,
            evaluator,
            q,
            1.0,
            mass,
            rng,
            &mut self.workspace,
            scratch,
        );
        self.dual.restart(step_size);
        step_size
    }

    /// Adapt after warmup transition `iter`, which left the chain at `q` with
    /// acceptance statistic `accept_stat`. Returns the step size for the next
    /// transition and may replace `mass`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn after_transition(
        &mut self,
        iter: usize,
        accept_stat: f64,
        q: &[f64],
        graph: &Graph,
        evaluator: &mut impl GradientEvaluator,
        mass: &mut MassMatrix,
        rng: &mut ChaCha8Rng,
        scratch: &mut [f64],
    ) -> f64 {
        let mut step_size = self.dual.update(accept_stat);
        if let Some(&(start, end)) = self.windows.get(self.next_window) {
            if (start..end).contains(&iter) {
                self.accumulator.update(q);
            }
            if iter + 1 == end {
                *mass = self.accumulator.finalize();
                self.next_window += 1;
                if let Some(&(start, end)) = self.windows.get(self.next_window) {
                    self.accumulator =
                        MassMatrixAccumulator::for_window(graph, self.metric, end - start);
                }
                // A new metric changes both the momentum scale and the
                // velocity, so the old step size is no longer calibrated.
                // Search again under the new geometry, from the current step
                // size as Stan does, and restart dual averaging there.
                step_size = find_reasonable_step_size(
                    graph,
                    evaluator,
                    q,
                    step_size,
                    mass,
                    rng,
                    &mut self.workspace,
                    scratch,
                );
                self.dual.restart(step_size);
            }
        }
        if iter + 1 == self.num_warmup {
            step_size = self.dual.adapted_step_size();
        }
        step_size
    }
}

const MIN_STEP_SIZE: f64 = 1e-10;
const MAX_STEP_SIZE: f64 = 1e3;
/// Enough doublings or halvings to cross the whole clamp range from 1.
const MAX_STEP_SIZE_PROBES: usize = 64;

/// Reusable buffers for [`find_reasonable_step_size`].
pub(crate) struct StepSizeWorkspace {
    p0: Vec<f64>,
    p1: Vec<f64>,
    q1: Vec<f64>,
    grad0: Vec<f64>,
}

impl StepSizeWorkspace {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            p0: vec![0.0; dim],
            p1: vec![0.0; dim],
            q1: vec![0.0; dim],
            grad0: vec![0.0; dim],
        }
    }
}

/// Hoffman & Gelman's heuristic for a step size whose one-step acceptance
/// ratio is near 1/2, starting from `initial`.
///
/// Each probe draws fresh momentum, as Stan does, so one unlucky draw does not
/// decide the whole search. A probe whose energy is not finite counts as a
/// rejection: while halving, the search keeps halving until the step is
/// usable; while doubling, it stops at the last step whose probe was finite.
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_reasonable_step_size(
    graph: &Graph,
    evaluator: &mut impl GradientEvaluator,
    q: &[f64],
    initial: f64,
    mass: &MassMatrix,
    rng: &mut ChaCha8Rng,
    workspace: &mut StepSizeWorkspace,
    scratch: &mut [f64],
) -> f64 {
    evaluator.compute(graph, q);
    let logp0 = evaluator.log_density();
    workspace.grad0.copy_from_slice(evaluator.gradient());
    let log_half = 0.5_f64.ln();

    let mut eps = if initial.is_finite() && initial > 0.0 {
        initial.clamp(MIN_STEP_SIZE, MAX_STEP_SIZE)
    } else {
        1.0
    };
    let first = probe_log_ratio(
        graph, evaluator, q, logp0, eps, mass, rng, workspace, scratch,
    );
    let doubling = first > log_half;
    for _ in 0..MAX_STEP_SIZE_PROBES {
        if evaluator.has_failed() {
            break;
        }
        let next = if doubling { eps * 2.0 } else { eps * 0.5 };
        if !(MIN_STEP_SIZE..=MAX_STEP_SIZE).contains(&next) {
            break;
        }
        let log_ratio = probe_log_ratio(
            graph, evaluator, q, logp0, next, mass, rng, workspace, scratch,
        );
        if doubling {
            if log_ratio == f64::NEG_INFINITY {
                // The doubled step left the region where the integrator is
                // defined at all; keep the last step that was.
                break;
            }
            eps = next;
            if log_ratio <= log_half {
                break;
            }
        } else {
            eps = next;
            if log_ratio > log_half {
                break;
            }
        }
    }
    eps
}

/// Log Metropolis ratio of one leapfrog step of size `eps` from `q` with
/// freshly drawn momentum; non-finite energies map to `-inf`.
#[allow(clippy::too_many_arguments)]
fn probe_log_ratio(
    graph: &Graph,
    evaluator: &mut impl GradientEvaluator,
    q: &[f64],
    logp0: f64,
    eps: f64,
    mass: &MassMatrix,
    rng: &mut ChaCha8Rng,
    workspace: &mut StepSizeWorkspace,
    scratch: &mut [f64],
) -> f64 {
    let StepSizeWorkspace { p0, p1, q1, grad0 } = workspace;
    mass.sample_momentum_into(rng, p0, scratch);
    let ke0 = mass.kinetic_energy(p0, scratch);
    for ((p1, &p0), &g0) in p1.iter_mut().zip(p0.iter()).zip(grad0.iter()) {
        *p1 = p0 + 0.5 * eps * g0;
    }
    mass.velocity_into(p1, q1, scratch);
    for (q1, &q0) in q1.iter_mut().zip(q) {
        *q1 = q0 + eps * *q1;
    }
    evaluator.compute(graph, q1);
    for (p1, &g1) in p1.iter_mut().zip(evaluator.gradient()) {
        *p1 += 0.5 * eps * g1;
    }
    let logp1 = evaluator.log_density();
    let ke1 = mass.kinetic_energy(p1, scratch);
    let log_ratio = (logp1 - ke1) - (logp0 - ke0);
    if log_ratio.is_finite() {
        log_ratio
    } else {
        f64::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Evaluator;
    use rand::SeedableRng;

    /// Stan's `windowed_adaptation`, transcribed from the C++ with its
    /// inclusive window ends and per-iteration counter, returning the
    /// half-open ranges whose draws each metric estimate uses.
    fn stan_reference(num_warmup: usize) -> Vec<(usize, usize)> {
        if num_warmup < 20 {
            return Vec::new();
        }
        let (mut init, mut term, mut base) = (75usize, 50usize, 25usize);
        if init + base + term > num_warmup {
            init = (0.15 * num_warmup as f64) as usize;
            term = (0.1 * num_warmup as f64) as usize;
            base = num_warmup - (init + term);
        }
        let mut window_size = base;
        let mut next_window = init + window_size - 1;
        let mut windows = Vec::new();
        let mut window_start = None;
        for counter in 0..num_warmup {
            let in_window = counter >= init && counter < num_warmup - term;
            if in_window && window_start.is_none() {
                window_start = Some(counter);
            }
            if counter == next_window {
                windows.push((window_start.take().unwrap(), counter + 1));
                // compute_next_window()
                if next_window == num_warmup - term - 1 {
                    continue;
                }
                window_size *= 2;
                next_window = counter + window_size;
                if next_window != num_warmup - term - 1 {
                    let boundary = next_window + 2 * window_size;
                    if boundary >= num_warmup - term {
                        next_window = num_warmup - term - 1;
                    }
                }
            }
        }
        windows
    }

    #[test]
    fn warmup_windows_match_stan() {
        let expected: [(usize, &[(usize, usize)]); 5] = [
            (150, &[(75, 100)]),
            (500, &[(75, 100), (100, 150), (150, 250), (250, 450)]),
            (750, &[(75, 100), (100, 150), (150, 250), (250, 700)]),
            (
                1000,
                &[(75, 100), (100, 150), (150, 250), (250, 450), (450, 950)],
            ),
            (
                2000,
                &[
                    (75, 100),
                    (100, 150),
                    (150, 250),
                    (250, 450),
                    (450, 850),
                    (850, 1950),
                ],
            ),
        ];
        for (num_warmup, windows) in expected {
            assert_eq!(
                WarmupSchedule::stan(num_warmup).windows,
                windows,
                "warmup {num_warmup}"
            );
            assert_eq!(stan_reference(num_warmup), windows, "warmup {num_warmup}");
        }
        for num_warmup in 0..3000 {
            assert_eq!(
                WarmupSchedule::stan(num_warmup).windows,
                stan_reference(num_warmup),
                "warmup {num_warmup}"
            );
        }
    }

    #[test]
    fn dual_averaging_restart_recentres_on_the_new_step() {
        let mut dual = DualAveraging::new(0.1, 0.8);
        for _ in 0..20 {
            dual.update(0.3);
        }
        dual.restart(2.0);
        // First update after a restart: h_bar = (0.8 - a) / 11 and
        // log eps = ln(10 * 2) - h_bar / 0.05.
        let step = dual.update(0.5);
        let expected = (20.0f64.ln() - (0.3 / 11.0) / 0.05).exp();
        assert!((step - expected).abs() < 1e-12);
        assert!((dual.adapted_step_size() - expected).abs() < 1e-12);
    }

    fn standard_normal() -> (Graph, crate::graph::NodeId) {
        let mut graph = Graph::new();
        let x = graph.add_param("x");
        let zero = graph.add_constant(0.0);
        let one = graph.add_constant(1.0);
        graph.normal_logp(x, zero, one);
        (graph, x)
    }

    fn search(graph: &Graph, seed: u64) -> f64 {
        let mut evaluator = Evaluator::new(graph);
        let mass = MassMatrix::from_graph(graph);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        find_reasonable_step_size(
            graph,
            &mut evaluator,
            &[0.0],
            1.0,
            &mass,
            &mut rng,
            &mut StepSizeWorkspace::new(1),
            &mut [0.0],
        )
    }

    #[test]
    fn step_size_search_is_not_pinned_to_lower_bound() {
        let (graph, _) = standard_normal();
        let step_size = search(&graph, 17);
        assert!(step_size.is_finite());
        assert!(
            step_size > 1e-2,
            "initial step-size search collapsed to {step_size}"
        );
    }

    #[test]
    fn step_size_search_keeps_halving_through_non_finite_probes() {
        // Finite only on (-1e-3, 1e-3): from x = 0 any step of order one
        // leaves the support, so the search must keep halving until a probe
        // lands inside rather than stopping after one halving.
        let (mut graph, x) = standard_normal();
        let width = graph.add_constant(1e-3);
        let minus_one = graph.add_constant(-1.0);
        let above = graph.add(x, width);
        graph.positive_support(above);
        let negated = graph.mul(x, minus_one);
        let below = graph.add(negated, width);
        graph.positive_support(below);
        // Each probe draws its own momentum, so an occasional tiny draw
        // stops the search early; the typical result is still of the order
        // of the support's width.
        let mut steps: Vec<f64> = (0..21).map(|seed| search(&graph, seed)).collect();
        assert!(
            steps
                .iter()
                .all(|&step| (MIN_STEP_SIZE..=0.25).contains(&step)),
            "{steps:?}"
        );
        steps.sort_by(f64::total_cmp);
        assert!(steps[10] < 0.01, "median step {}", steps[10]);
    }
}
