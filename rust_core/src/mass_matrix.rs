use crate::graph::Graph;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Largest parameter block that may receive a dense metric.
const DENSE_BLOCK_MAX_DIM: usize = 512;
/// Under [`MetricKind::Auto`], whether a window of `window_len` draws can
/// give a block of `dim` elements a dense metric: each half of the window
/// must hold more draws than the block has elements, or its sample
/// covariance is singular and [`dense_beats_diagonal`] has nothing to score
/// on. This is deliberately a low bar: a strongly correlated block gains an
/// order of magnitude in step size from even a noisy dense estimate, and the
/// decision rule keeps other blocks diagonal.
fn auto_window_can_be_dense(window_len: usize, dim: usize) -> bool {
    window_len / 2 > dim
}
const REGULARIZATION_WEIGHT: f64 = 5.0;
const BASE_JITTER: f64 = 1e-3;

/// How warmup estimates the metric of a parameter block with more than one
/// element. Scalar parameters always get a scalar metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetricKind {
    /// Diagonal, except that a vector parameter of at most 512 elements gets a
    /// dense block from a window whose halves each hold more draws than it has
    /// elements, when a dense estimate from each half leaves the other half
    /// better conditioned than a diagonal one, on average over the two: by
    /// any amount in early windows and by a factor of 2 in the last one (see
    /// `dense_beats_diagonal`). Weakly correlated blocks are meant to stay
    /// diagonal, as under Stan's default metric; this is a statistical
    /// decision, not a guarantee. Dense blocks never span more than one
    /// vector parameter, and scalar parameters are always diagonal.
    #[default]
    Auto,
    /// Diagonal for every block, as Stan's default `diag_e` metric.
    Diagonal,
    /// Dense within every vector parameter of at most 512 elements,
    /// estimated from each window whatever its length; a window with fewer
    /// than two draws per element has its correlations shrunk toward zero so
    /// that a short window cannot give a near-singular estimate (see
    /// `shrink_correlations`).
    /// Unlike Stan's `dense_e`, which estimates one covariance over all
    /// parameters, there is no correlation across parameters or between
    /// scalar parameters.
    Dense,
}

impl MetricKind {
    /// Parse the user-facing spelling: `"auto"`, `"diag"` or `"dense"`.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "auto" => Ok(Self::Auto),
            "diag" | "diagonal" => Ok(Self::Diagonal),
            "dense" => Ok(Self::Dense),
            _ => Err(format!(
                "metric must be 'auto', 'diag' or 'dense', not '{name}'"
            )),
        }
    }
}

impl std::str::FromStr for MetricKind {
    type Err = String;

    /// The same spellings as [`MetricKind::parse`], so `"dense".parse()` works.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::parse(name)
    }
}

#[derive(Debug, Clone)]
pub struct MassMatrix {
    dim: usize,
    blocks: Vec<MassBlock>,
}

#[derive(Debug, Clone)]
struct MassBlock {
    start: usize,
    len: usize,
    kind: BlockKind,
}

#[derive(Debug, Clone)]
enum BlockKind {
    Scalar {
        /// Adapted position variance, equal to the inverse mass metric.
        variance: f64,
    },
    Diagonal {
        /// Adapted position variances, equal to the diagonal inverse mass metric.
        variances: Vec<f64>,
    },
    Dense {
        /// Cholesky factor of the adapted position covariance (inverse mass metric).
        chol: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
pub struct MassMatrixAccumulator {
    blocks: Vec<AccumulatorBlock>,
}

#[derive(Debug, Clone)]
enum AccumulatorBlock {
    Scalar {
        start: usize,
        count: usize,
        mean: f64,
        m2: f64,
    },
    Diagonal {
        start: usize,
        count: usize,
        mean: Vec<f64>,
        m2: Vec<f64>,
    },
    Dense {
        start: usize,
        dim: usize,
        /// Every draw of the window; the metric is estimated from these.
        all: Moments,
        /// Deviations from the previous mean, reused across updates.
        delta: Vec<f64>,
        /// Under [`MetricKind::Auto`], each half's moments: whether the
        /// metric is dense is decided from how well each half's estimate
        /// whitens the other (see [`dense_beats_diagonal`]).
        halves: Option<Halves>,
    },
}

/// What [`MetricKind::Auto`] keeps of a dense-eligible window: its two
/// halves apart, and the margin a dense estimate must win by.
#[derive(Debug, Clone)]
struct Halves {
    /// The draw count at which the second half begins.
    split: usize,
    first: Moments,
    second: Moments,
    margin: f64,
}

/// Running mean and centred cross-product sum of a block's draws, by
/// Welford's update; the sum is the lower triangle of a full `dim * dim`
/// array.
#[derive(Debug, Clone)]
struct Moments {
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Moments {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim * dim],
        }
    }

    fn update(&mut self, x: &[f64], delta: &mut [f64]) {
        let dim = x.len();
        self.count += 1;
        let n = self.count as f64;
        for ((delta_i, mean_i), &x_i) in delta.iter_mut().zip(&mut self.mean).zip(x) {
            *delta_i = x_i - *mean_i;
            *mean_i += *delta_i / n;
        }
        // M2 += (x - mean_old)(x - mean_new)^T, lower triangle.
        for (i, &delta_i) in delta.iter().enumerate() {
            let row = &mut self.m2[i * dim..i * dim + i + 1];
            for (j, value) in row.iter_mut().enumerate() {
                *value += delta_i * (x[j] - self.mean[j]);
            }
        }
    }

    /// The sample covariance with both triangles filled, or `None` from
    /// fewer than two draws.
    fn covariance(&self) -> Option<Vec<f64>> {
        if self.count < 2 {
            return None;
        }
        let dim = self.mean.len();
        let scale = 1.0 / (self.count as f64 - 1.0);
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..=i {
                let value = self.m2[i * dim + j] * scale;
                cov[i * dim + j] = value;
                cov[j * dim + i] = value;
            }
        }
        Some(cov)
    }
}

impl MassMatrix {
    /// The unit metric. Vector blocks start diagonal whatever the adapted
    /// metric will be: a dense identity costs O(d²) per leapfrog step for
    /// exactly the same dynamics.
    pub fn identity(graph: &Graph) -> Self {
        let mut blocks = Vec::with_capacity(graph.param_spans.len());
        for span in &graph.param_spans {
            blocks.push(MassBlock::identity(span.start, span.len));
        }
        Self {
            dim: graph.param_count,
            blocks,
        }
    }

    pub fn sample_momentum_into<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        momentum: &mut [f64],
        scratch: &mut [f64],
    ) {
        debug_assert_eq!(momentum.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        for block in &self.blocks {
            block.sample_momentum_into(rng, momentum, scratch);
        }
    }

    pub fn velocity_into(&self, momentum: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        debug_assert_eq!(momentum.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        for block in &self.blocks {
            block.velocity_into(momentum, out, scratch);
        }
    }

    pub fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> f64 {
        debug_assert_eq!(momentum.len(), self.dim);
        debug_assert!(scratch.len() >= self.dim);

        let mut ke = 0.0f64;
        for block in &self.blocks {
            ke += block.kinetic_energy(momentum, scratch);
        }
        ke
    }

    pub fn uturn(&self, left_q: &[f64], left_p: &[f64], right_q: &[f64], right_p: &[f64]) -> bool {
        debug_assert_eq!(left_q.len(), self.dim);
        debug_assert_eq!(right_q.len(), self.dim);
        debug_assert_eq!(left_p.len(), self.dim);
        debug_assert_eq!(right_p.len(), self.dim);

        let mut dot_left = 0.0f64;
        let mut dot_right = 0.0f64;
        for block in &self.blocks {
            let (dl, dr) = block.uturn_terms(left_q, left_p, right_q, right_p);
            dot_left += dl;
            dot_right += dr;
        }
        dot_left < 0.0 || dot_right < 0.0
    }

    /// Whether any block carries a dense factor (for tests and diagnostics).
    #[cfg(test)]
    pub(crate) fn has_dense_block(&self) -> bool {
        self.blocks
            .iter()
            .any(|block| matches!(block.kind, BlockKind::Dense { .. }))
    }
}

impl MassMatrixAccumulator {
    /// An accumulator for the last adaptation window, of `window_len` draws,
    /// whose metric the chain samples with; see [`Self::for_warmup_window`].
    pub fn for_window(graph: &Graph, kind: MetricKind, window_len: usize) -> Self {
        Self::for_warmup_window(graph, kind, window_len, true)
    }

    /// An accumulator for one adaptation window of `window_len` draws.
    ///
    /// The window length is known when the window opens, so under
    /// [`MetricKind::Auto`] which blocks may become dense is settled here, and
    /// a block that will be diagonal never pays for the O(d²) cross products.
    /// Under `Auto` a dense estimate must beat the diagonal one by
    /// [`LAST_WINDOW_MARGIN`] in the `last` window, whose metric the draws are
    /// sampled with, and by [`EARLY_WINDOW_MARGIN`] in the others, whose
    /// metric only samples the next window.
    pub fn for_warmup_window(
        graph: &Graph,
        kind: MetricKind,
        window_len: usize,
        last: bool,
    ) -> Self {
        let margin = if last {
            LAST_WINDOW_MARGIN
        } else {
            EARLY_WINDOW_MARGIN
        };
        let mut blocks = Vec::with_capacity(graph.param_spans.len());
        for span in &graph.param_spans {
            let dense = span.len > 1
                && span.len <= DENSE_BLOCK_MAX_DIM
                && match kind {
                    MetricKind::Diagonal => false,
                    MetricKind::Dense => true,
                    MetricKind::Auto => auto_window_can_be_dense(window_len, span.len),
                };
            blocks.push(AccumulatorBlock::new(
                span.start,
                span.len,
                dense,
                (kind == MetricKind::Auto).then_some((window_len / 2, margin)),
            ));
        }
        Self { blocks }
    }

    /// Draws accumulated so far.
    pub fn draws(&self) -> usize {
        self.blocks.first().map_or(0, AccumulatorBlock::count)
    }

    pub fn update(&mut self, q: &[f64]) {
        for block in &mut self.blocks {
            block.update(q);
        }
    }

    pub fn finalize(&self) -> MassMatrix {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            blocks.push(block.finalize());
        }
        let dim = blocks
            .iter()
            .map(|block| block.start + block.len)
            .max()
            .unwrap_or(0);
        MassMatrix { dim, blocks }
    }
}

impl MassBlock {
    fn identity(start: usize, len: usize) -> Self {
        let kind = if len == 1 {
            BlockKind::Scalar { variance: 1.0 }
        } else {
            BlockKind::Diagonal {
                variances: vec![1.0; len],
            }
        };
        Self { start, len, kind }
    }

    fn sample_momentum_into<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        momentum: &mut [f64],
        scratch: &mut [f64],
    ) {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { variance, .. } => {
                let z: f64 = StandardNormal.sample(rng);
                // The adapted position covariance is the inverse mass metric,
                // so momentum has variance M = variance^-1.
                momentum[self.start] = z / variance.sqrt();
            }
            BlockKind::Diagonal { variances, .. } => {
                for (offset, &variance) in variances.iter().enumerate() {
                    let z: f64 = StandardNormal.sample(rng);
                    momentum[self.start + offset] = z / variance.sqrt();
                }
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                for value in scratch_block.iter_mut() {
                    *value = StandardNormal.sample(rng);
                }
                // If Sigma = L L^T is the adapted position covariance, then
                // p = L^-T z has covariance Sigma^-1, the momentum mass M.
                solve_lower_transpose_in_place(chol, scratch_block);
                momentum[range].copy_from_slice(scratch_block);
            }
        }
    }

    fn velocity_into(&self, momentum: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { variance } => {
                out[self.start] = momentum[self.start] * variance;
            }
            BlockKind::Diagonal { variances } => {
                for (offset, &variance) in variances.iter().enumerate() {
                    out[self.start + offset] = momentum[self.start + offset] * variance;
                }
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                // M^-1 p = Sigma p = L (L^T p).
                dense_matvec_lower_transpose(chol, &momentum[range.clone()], scratch_block);
                dense_matvec_lower(chol, scratch_block, &mut out[range]);
            }
        }
    }

    fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> f64 {
        let range = self.start..self.start + self.len;
        match &self.kind {
            BlockKind::Scalar { variance } => {
                0.5 * momentum[self.start] * momentum[self.start] * variance
            }
            BlockKind::Diagonal { variances } => {
                let mut ke = 0.0f64;
                for (offset, &variance) in variances.iter().enumerate() {
                    let p = momentum[self.start + offset];
                    ke += 0.5 * p * p * variance;
                }
                ke
            }
            BlockKind::Dense { chol } => {
                let scratch_block = &mut scratch[range.clone()];
                // p^T Sigma p = ||L^T p||^2.
                dense_matvec_lower_transpose(chol, &momentum[range], scratch_block);
                0.5 * scratch_block.iter().map(|value| value * value).sum::<f64>()
            }
        }
    }

    fn uturn_terms(
        &self,
        left_q: &[f64],
        left_p: &[f64],
        right_q: &[f64],
        right_p: &[f64],
    ) -> (f64, f64) {
        // In whitened canonical coordinates z=L^-1 q and p_z=L^T p,
        // delta_z^T p_z = delta_q^T p: the constant metric cancels.
        // Using velocity here distorts the turning direction and can cause
        // dense-metric trajectories to run to maximum tree depth.
        let mut dot_left = 0.0f64;
        let mut dot_right = 0.0f64;
        for offset in 0..self.len {
            let index = self.start + offset;
            let delta = right_q[index] - left_q[index];
            dot_left += delta * left_p[index];
            dot_right += delta * right_p[index];
        }
        (dot_left, dot_right)
    }
}

impl AccumulatorBlock {
    /// A block that, when `halves` gives a split count and a margin, also
    /// keeps the moments of the draws before and after that count apart.
    fn new(start: usize, len: usize, dense: bool, halves: Option<(usize, f64)>) -> Self {
        if len == 1 {
            Self::Scalar {
                start,
                count: 0,
                mean: 0.0,
                m2: 0.0,
            }
        } else if dense {
            Self::Dense {
                start,
                dim: len,
                all: Moments::new(len),
                delta: vec![0.0; len],
                halves: halves.map(|(split, margin)| Halves {
                    split,
                    first: Moments::new(len),
                    second: Moments::new(len),
                    margin,
                }),
            }
        } else {
            Self::Diagonal {
                start,
                count: 0,
                mean: vec![0.0; len],
                m2: vec![0.0; len],
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::Scalar { count, .. } | Self::Diagonal { count, .. } => *count,
            Self::Dense { all, .. } => all.count,
        }
    }

    fn update(&mut self, q: &[f64]) {
        match self {
            Self::Scalar {
                start,
                count,
                mean,
                m2,
            } => {
                let x = q[*start];
                *count += 1;
                let n = *count as f64;
                let delta = x - *mean;
                *mean += delta / n;
                let delta2 = x - *mean;
                *m2 += delta * delta2;
            }
            Self::Diagonal {
                start,
                count,
                mean,
                m2,
            } => {
                *count += 1;
                let n = *count as f64;
                for (offset, mean_i) in mean.iter_mut().enumerate() {
                    let x = q[*start + offset];
                    let delta = x - *mean_i;
                    *mean_i += delta / n;
                    let delta2 = x - *mean_i;
                    m2[offset] += delta * delta2;
                }
            }
            Self::Dense {
                start,
                dim,
                all,
                delta,
                halves,
            } => {
                let x = &q[*start..*start + *dim];
                if let Some(halves) = halves {
                    let half = if all.count < halves.split {
                        &mut halves.first
                    } else {
                        &mut halves.second
                    };
                    half.update(x, delta);
                }
                all.update(x, delta);
            }
        }
    }

    fn finalize(&self) -> MassBlock {
        match self {
            Self::Scalar {
                start, count, m2, ..
            } => {
                let variance = regularize_variance(
                    if *count > 1 {
                        *m2 / (*count as f64 - 1.0)
                    } else {
                        1.0
                    },
                    *count,
                );
                MassBlock {
                    start: *start,
                    len: 1,
                    kind: BlockKind::Scalar { variance },
                }
            }
            Self::Diagonal {
                start, count, m2, ..
            } => {
                let variances = m2
                    .iter()
                    .map(|&m2| {
                        regularize_variance(
                            if *count > 1 {
                                m2 / (*count as f64 - 1.0)
                            } else {
                                1.0
                            },
                            *count,
                        )
                    })
                    .collect();
                MassBlock {
                    start: *start,
                    len: m2.len(),
                    kind: BlockKind::Diagonal { variances },
                }
            }
            Self::Dense {
                start,
                dim,
                all,
                halves,
                ..
            } => {
                let dim = *dim;
                let Some(mut cov) = all.covariance() else {
                    return MassBlock::identity(*start, dim);
                };
                let count = all.count;
                if let Some(halves) = halves {
                    if !dense_beats_diagonal(&halves.first, &halves.second, halves.margin) {
                        let variances = (0..dim)
                            .map(|i| regularize_variance(cov[i * dim + i], count))
                            .collect();
                        return MassBlock {
                            start: *start,
                            len: dim,
                            kind: BlockKind::Diagonal { variances },
                        };
                    }
                }

                shrink_correlations(&mut cov, dim, count);
                // Stan's dense_e regularization: shrink toward a small
                // multiple of the identity, weighted by the draw count.
                let n = count as f64;
                let shrink = n / (n + REGULARIZATION_WEIGHT);
                let jitter = BASE_JITTER * (REGULARIZATION_WEIGHT / (n + REGULARIZATION_WEIGHT));
                for i in 0..dim {
                    for j in 0..dim {
                        cov[i * dim + j] *= shrink;
                    }
                    cov[i * dim + i] += jitter;
                }

                MassBlock {
                    start: *start,
                    len: dim,
                    kind: BlockKind::Dense {
                        chol: cholesky_with_jitter(cov, dim),
                    },
                }
            }
        }
    }
}

/// Draws per dimension from which a dense estimate is used unshrunk.
const UNSHRUNK_DRAWS_PER_DIM: f64 = 2.0;

/// Shrink the off-diagonal of the sample covariance `cov` of `draws` draws
/// toward zero when the window holds fewer than [`UNSHRUNK_DRAWS_PER_DIM`]
/// draws per dimension, and return the weight used.
///
/// Such a window gives a sample covariance that is singular, or nearly so, in
/// directions the target is not, and whitening by it forces a tiny step size
/// along them: without this, a 300-dimensional isotropic block under
/// `metric="dense"` took about 30 times the diagonal metric's leapfrog steps
/// per iteration (`benchmarks/metric_adaptation.py`). Shrinking the
/// correlation matrix toward the identity by `w` keeps every variance and
/// lifts every eigenvalue of the correlation estimate to at least `w`. The
/// weight `1 - n / (2d)` rises from zero at two draws per dimension to one
/// half at one draw per dimension, where the estimate becomes singular.
///
/// This bounds the damage, not the noise: just above two draws per dimension
/// an isotropic block's estimate is still as noisy as sampling makes it
/// (condition number near 30 at `n = 2d`), and a dense request there costs
/// about twice the diagonal metric's steps. A weight that stays positive
/// further up removes that, but shrinking any correlation also lifts the
/// smallest eigenvalues of a strongly correlated block, the very directions
/// a dense metric is for: ramping to zero at three draws per dimension cost
/// the 20- and 50-dimensional correlated regressions of that benchmark 10 to
/// 25% more steps. The weight depends on the draw count alone for the same
/// reason; data-driven targets such as Ledoit–Wolf or Schäfer–Strimmer read a
/// regression posterior's many small correlations as noise.
fn shrink_correlations(cov: &mut [f64], dim: usize, draws: usize) -> f64 {
    let weight = (1.0 - draws as f64 / (UNSHRUNK_DRAWS_PER_DIM * dim as f64)).max(0.0);
    if weight > 0.0 {
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    cov[i * dim + j] *= 1.0 - weight;
                }
            }
        }
    }
    weight
}

/// Whether a dense metric should precondition a block better than the
/// diagonal one, judged by cross-validation on the window's two halves.
///
/// A metric `M` leaves the integrator the target covariance `Σ` whitened by
/// `M`, and the step size and trajectory length NUTS needs follow that
/// matrix's condition number. Each half's sample covariance stands in for
/// `Σ` while the other half's estimates the metric, so both metrics are
/// scored on draws they were not estimated from: a dense estimate that only
/// fits its own half's noise whitens the other half no better than the
/// diagonal one, and usually worse. Both candidates are scored on the same
/// held-out covariance, so its own sampling noise and autocorrelation affect
/// them alike and need no estimate; the rule this replaced had to estimate
/// the window's effective sample size, and could not do so reliably when
/// coordinates mixed at different speeds. The dense metric must leave a
/// geometric mean of the two folds' condition numbers smaller by `margin`
/// ([`LAST_WINDOW_MARGIN`] or [`EARLY_WINDOW_MARGIN`]). A half too short for
/// a nonsingular covariance chooses diagonal.
/// The condition numbers come from a fixed number of power and inverse-power
/// iterations, which can only underestimate them.
fn dense_beats_diagonal(first: &Moments, second: &Moments, margin: f64) -> bool {
    let (Some(a), Some(b)) = (first.covariance(), second.covariance()) else {
        return false;
    };
    let dim = first.mean.len();
    let mut diagonal = 0.0;
    let mut dense = 0.0;
    for (train, train_count, test) in [(&a, first.count, &b), (&b, second.count, &a)] {
        let Some(whitened) = whiten_by_variances(train, test, dim) else {
            return false;
        };
        diagonal += log_condition(&whitened, dim);
        dense += whiten_by_covariance(train, train_count, test, dim)
            .map_or(f64::INFINITY, |whitened| log_condition(&whitened, dim));
    }
    dense.is_finite() && dense / 2.0 < diagonal / 2.0 - margin.ln()
}

/// How much better conditioned, on held-out draws, a dense metric must leave
/// a block than the diagonal one in the last warmup window, whose metric the
/// draws are sampled with: a factor of 2 in the condition number. In
/// simulated warmup windows (`benchmarks/auto_metric_noise.py`) this kept
/// every uncorrelated window diagonal, including windows whose halves held
/// barely more draws than dimensions and windows whose coordinates mixed at
/// very different speeds, where a factor of 1 went dense in up to 91 of 100.
/// A dense leapfrog step also does O(d²) more arithmetic than a diagonal one.
const LAST_WINDOW_MARGIN: f64 = 2.0;

/// The same margin for the earlier windows, whose metric only samples the
/// next window: 1, no margin. A wrong dense choice there costs one window's
/// sampling and the next window decides again, while a missed one leaves the
/// next window's draws to mix slowly along the correlation and so degrades
/// every later estimate. With 2 in every window the 10- and 20-dimensional
/// correlated regressions of `benchmarks/auto_metric_targets.py` took 8% and
/// 19% more leapfrog steps per unit of ESS than with this split (medians
/// over ten seeds at 1000 warmup iterations), and the other targets there
/// did not improve.
const EARLY_WINDOW_MARGIN: f64 = 1.0;

/// `test` whitened by the variances on `train`'s diagonal: the metric a
/// diagonal estimate from `train` leaves for draws distributed as `test`.
fn whiten_by_variances(train: &[f64], test: &[f64], dim: usize) -> Option<Vec<f64>> {
    let inv_sd: Vec<f64> = (0..dim)
        .map(|i| train[i * dim + i].sqrt().recip())
        .collect();
    if inv_sd.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut whitened = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            whitened[i * dim + j] = test[i * dim + j] * inv_sd[i] * inv_sd[j];
        }
    }
    Some(whitened)
}

/// `L^-1 test L^-T` for the Cholesky factor `L` of `train`, with
/// `train`'s correlations shrunk as a dense estimate from `train_count`
/// draws is (see [`shrink_correlations`]). `None` when that factor does not
/// exist.
fn whiten_by_covariance(
    train: &[f64],
    train_count: usize,
    test: &[f64],
    dim: usize,
) -> Option<Vec<f64>> {
    let mut lower = train.to_vec();
    shrink_correlations(&mut lower, dim, train_count);
    if !cholesky_lower_in_place(&mut lower, dim) {
        return None;
    }
    // `half` holds L^-1 test by rows: row j is L^-1 applied to column j of
    // `test`, which is row j because `test` is symmetric.
    let mut half = test.to_vec();
    for row in half.chunks_exact_mut(dim) {
        solve_lower_in_place(&lower, row);
    }
    // Column k of L^-1 (L^-1 test)^T is L^-1 applied to row k of L^-1 test,
    // which is column k of `half`.
    let mut column = vec![0.0; dim];
    let mut whitened = vec![0.0; dim * dim];
    for k in 0..dim {
        for j in 0..dim {
            column[j] = half[j * dim + k];
        }
        solve_lower_in_place(&lower, &mut column);
        for i in 0..dim {
            whitened[i * dim + k] = column[i];
        }
    }
    for i in 0..dim {
        for j in 0..i {
            let mean = 0.5 * (whitened[i * dim + j] + whitened[j * dim + i]);
            whitened[i * dim + j] = mean;
            whitened[j * dim + i] = mean;
        }
    }
    Some(whitened)
}

/// The natural log of a symmetric matrix's condition number, from its
/// extreme eigenvalues by power and inverse-power iteration; infinite when
/// the matrix is not numerically positive definite.
fn log_condition(matrix: &[f64], dim: usize) -> f64 {
    const ITERATIONS: usize = 100;
    let mut chol = matrix.to_vec();
    if !cholesky_lower_in_place(&mut chol, dim) {
        return f64::INFINITY;
    }
    let start: Vec<f64> = (0..dim).map(|i| 1.0 + i as f64 / dim as f64).collect();
    let mut v = start.clone();
    let mut w = vec![0.0; dim];
    let mut largest = 0.0;
    for _ in 0..ITERATIONS {
        normalize(&mut v);
        symmetric_matvec(matrix, &v, &mut w, dim);
        largest = dot(&v, &w);
        std::mem::swap(&mut v, &mut w);
    }
    let mut v = start;
    let mut smallest_inverse = 0.0;
    for _ in 0..ITERATIONS {
        normalize(&mut v);
        w.copy_from_slice(&v);
        solve_lower_in_place(&chol, &mut w);
        solve_lower_transpose_in_place(&chol, &mut w);
        smallest_inverse = dot(&v, &w);
        std::mem::swap(&mut v, &mut w);
    }
    let condition = largest * smallest_inverse;
    if condition.is_finite() && condition > 0.0 {
        condition.ln()
    } else {
        f64::INFINITY
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f64]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 && norm.is_finite() {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

fn symmetric_matvec(a: &[f64], x: &[f64], out: &mut [f64], dim: usize) {
    for i in 0..dim {
        out[i] = dot(&a[i * dim..(i + 1) * dim], x);
    }
}

fn solve_lower_in_place(lower: &[f64], rhs: &mut [f64]) {
    let dim = rhs.len();
    for i in 0..dim {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower[i * dim + j] * rhs[j];
        }
        rhs[i] = sum / lower[i * dim + i];
    }
}

fn regularize_variance(variance: f64, count: usize) -> f64 {
    let n = count as f64;
    let shrunk = if count > 0 {
        (n / (n + REGULARIZATION_WEIGHT)) * variance
            + BASE_JITTER * (REGULARIZATION_WEIGHT / (n + REGULARIZATION_WEIGHT))
    } else {
        1.0
    };
    shrunk.max(1e-12)
}

fn identity_lower(dim: usize) -> Vec<f64> {
    let mut chol = vec![0.0; dim * dim];
    for i in 0..dim {
        chol[i * dim + i] = 1.0;
    }
    chol
}

fn dense_matvec_lower(lower: &[f64], x: &[f64], out: &mut [f64]) {
    let dim = x.len();
    for i in 0..dim {
        let mut sum = 0.0f64;
        for j in 0..=i {
            sum += lower[i * dim + j] * x[j];
        }
        out[i] = sum;
    }
}

fn dense_matvec_lower_transpose(lower: &[f64], x: &[f64], out: &mut [f64]) {
    let dim = x.len();
    for i in 0..dim {
        let mut sum = 0.0f64;
        for j in i..dim {
            sum += lower[j * dim + i] * x[j];
        }
        out[i] = sum;
    }
}

fn solve_lower_transpose_in_place(lower: &[f64], rhs: &mut [f64]) {
    let dim = rhs.len();
    for i in (0..dim).rev() {
        let mut sum = rhs[i];
        for j in i + 1..dim {
            sum -= lower[j * dim + i] * rhs[j];
        }
        rhs[i] = sum / lower[i * dim + i];
    }
}

/// Cholesky factor of `cov`, adding diagonal jitter only if the plain
/// factorization fails. The regularized estimate is already positive
/// definite in exact arithmetic, so the jitter is a rounding safeguard and
/// must not distort a factorization that succeeds without it.
fn cholesky_with_jitter(cov: Vec<f64>, dim: usize) -> Vec<f64> {
    let mut candidate = cov.clone();
    if cholesky_lower_in_place(&mut candidate, dim) {
        return candidate;
    }
    let mut jitter = BASE_JITTER;
    for _ in 0..8 {
        candidate.copy_from_slice(&cov);
        for i in 0..dim {
            candidate[i * dim + i] += jitter;
        }
        if cholesky_lower_in_place(&mut candidate, dim) {
            return candidate;
        }
        jitter *= 10.0;
    }

    identity_lower(dim)
}

fn cholesky_lower_in_place(a: &mut [f64], dim: usize) -> bool {
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum -= a[i * dim + k] * a[j * dim + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return false;
                }
                a[i * dim + j] = sum.sqrt();
            } else {
                let diag = a[j * dim + j];
                if !diag.is_finite() || diag <= 0.0 {
                    return false;
                }
                a[i * dim + j] = sum / diag;
            }
        }
        for j in i + 1..dim {
            a[i * dim + j] = 0.0;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected} +/- {tolerance}, got {actual}"
        );
    }

    #[test]
    fn scalar_block_uses_position_variance_as_inverse_mass_metric() {
        let block = MassBlock {
            start: 0,
            len: 1,
            kind: BlockKind::Scalar { variance: 4.0 },
        };
        let mut scratch = [0.0];
        let momentum = [1.5];
        let mut velocity = [0.0];

        block.velocity_into(&momentum, &mut velocity, &mut scratch);
        assert_eq!(velocity, [6.0]);
        assert_eq!(block.kinetic_energy(&momentum, &mut scratch), 4.5);

        let (left, right) = block.uturn_terms(&[0.0], &momentum, &[2.0], &[-0.5]);
        assert_eq!((left, right), (3.0, -1.0));

        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut sampled = [0.0];
        let mut sum_sq = 0.0;
        let draws = 20_000;
        for _ in 0..draws {
            block.sample_momentum_into(&mut rng, &mut sampled, &mut scratch);
            sum_sq += sampled[0] * sampled[0];
        }
        assert_close(sum_sq / draws as f64, 0.25, 0.015);
    }

    #[test]
    fn diagonal_block_operations_share_the_same_inverse_mass_metric() {
        let block = MassBlock {
            start: 0,
            len: 2,
            kind: BlockKind::Diagonal {
                variances: vec![4.0, 0.25],
            },
        };
        let mut scratch = [0.0; 2];
        let momentum = [1.0, -2.0];
        let mut velocity = [0.0; 2];

        block.velocity_into(&momentum, &mut velocity, &mut scratch);
        assert_eq!(velocity, [4.0, -0.5]);
        assert_eq!(block.kinetic_energy(&momentum, &mut scratch), 2.5);

        let left_q = [1.0, -1.0];
        let right_q = [3.0, 2.0];
        let right_p = [-0.5, 4.0];
        let (left, right) = block.uturn_terms(&left_q, &momentum, &right_q, &right_p);
        let delta = [right_q[0] - left_q[0], right_q[1] - left_q[1]];
        assert_eq!(left, delta[0] * momentum[0] + delta[1] * momentum[1]);
        assert_eq!(right, delta[0] * right_p[0] + delta[1] * right_p[1]);

        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let mut sampled = [0.0; 2];
        let mut sum_sq = [0.0; 2];
        let draws = 20_000;
        for _ in 0..draws {
            block.sample_momentum_into(&mut rng, &mut sampled, &mut scratch);
            for i in 0..2 {
                sum_sq[i] += sampled[i] * sampled[i];
            }
        }
        assert_close(sum_sq[0] / draws as f64, 0.25, 0.015);
        assert_close(sum_sq[1] / draws as f64, 4.0, 0.15);
    }

    #[test]
    fn dense_block_operations_share_the_same_inverse_mass_metric() {
        // L L^T = [[4, 1], [1, 1]], whose inverse is
        // [[1/3, -1/3], [-1/3, 4/3]].
        let block = MassBlock {
            start: 0,
            len: 2,
            kind: BlockKind::Dense {
                chol: vec![2.0, 0.0, 0.5, 0.75_f64.sqrt()],
            },
        };
        let mut scratch = [0.0; 2];
        let momentum = [1.0, -2.0];
        let mut velocity = [0.0; 2];

        block.velocity_into(&momentum, &mut velocity, &mut scratch);
        assert_close(velocity[0], 2.0, 1e-12);
        assert_close(velocity[1], -1.0, 1e-12);
        assert_close(block.kinetic_energy(&momentum, &mut scratch), 2.0, 1e-12);

        let left_q = [1.0, -1.0];
        let right_q = [3.0, 2.0];
        let right_p = [-0.5, 4.0];
        let (left, right) = block.uturn_terms(&left_q, &momentum, &right_q, &right_p);
        let delta = [right_q[0] - left_q[0], right_q[1] - left_q[1]];
        assert_close(left, delta[0] * momentum[0] + delta[1] * momentum[1], 1e-12);
        assert_close(right, delta[0] * right_p[0] + delta[1] * right_p[1], 1e-12);

        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let mut sampled = [0.0; 2];
        let mut second_moment = [0.0; 3];
        let draws = 40_000;
        for _ in 0..draws {
            block.sample_momentum_into(&mut rng, &mut sampled, &mut scratch);
            second_moment[0] += sampled[0] * sampled[0];
            second_moment[1] += sampled[0] * sampled[1];
            second_moment[2] += sampled[1] * sampled[1];
        }
        assert_close(second_moment[0] / draws as f64, 1.0 / 3.0, 0.02);
        assert_close(second_moment[1] / draws as f64, -1.0 / 3.0, 0.02);
        assert_close(second_moment[2] / draws as f64, 4.0 / 3.0, 0.04);
    }

    fn vector_graph(dim: usize) -> Graph {
        let mut graph = Graph::new();
        let start = graph.add_vector_params("b", dim);
        graph.vector_normal_logp(start, dim, 0.0, 1.0);
        graph
    }

    /// `count` draws from N(0, Σ) with unit variances and equal correlation `rho`.
    fn equicorrelated_draws(dim: usize, rho: f64, count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let shared: f64 = StandardNormal.sample(&mut rng);
                (0..dim)
                    .map(|_| {
                        let own: f64 = StandardNormal.sample(&mut rng);
                        rho.sqrt() * shared + (1.0 - rho).sqrt() * own
                    })
                    .collect()
            })
            .collect()
    }

    fn estimate(graph: &Graph, kind: MetricKind, draws: &[Vec<f64>]) -> MassMatrix {
        let mut accumulator = MassMatrixAccumulator::for_window(graph, kind, draws.len());
        for draw in draws {
            accumulator.update(draw);
        }
        accumulator.finalize()
    }

    #[test]
    fn auto_metric_keeps_isotropic_blocks_diagonal() {
        // Enough draws to be eligible for a dense estimate; the sample
        // covariance's own noise is the only structure there is to find.
        for (dim, count) in [(20, 100), (20, 400), (50, 1100)] {
            let graph = vector_graph(dim);
            let draws = equicorrelated_draws(dim, 0.0, count, dim as u64);
            assert!(!estimate(&graph, MetricKind::Auto, &draws).has_dense_block());
            assert!(estimate(&graph, MetricKind::Dense, &draws).has_dense_block());
            assert!(!estimate(&graph, MetricKind::Diagonal, &draws).has_dense_block());
        }
    }

    #[test]
    fn auto_metric_uses_a_dense_block_for_strong_correlation() {
        let graph = vector_graph(10);
        let draws = equicorrelated_draws(10, 0.9, 200, 5);
        assert!(estimate(&graph, MetricKind::Auto, &draws).has_dense_block());
        // Too few draws per dimension to estimate it, however correlated.
        let short = &draws[..15];
        assert!(!estimate(&graph, MetricKind::Auto, short).has_dense_block());
        assert!(estimate(&graph, MetricKind::Dense, short).has_dense_block());
    }

    #[test]
    fn autocorrelated_isotropic_draws_stay_diagonal() {
        // MCMC draws are not independent: an AR(1) chain with coefficient 0.6
        // carries a quarter of the information of independent draws, so the
        // sample covariance's noise is that of a quarter of the draws. Judged
        // by the independent-draw noise bound, many of these windows looked
        // correlated enough to go dense.
        let dim = 40;
        let graph = vector_graph(dim);
        for seed in 0..20 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut state = vec![0.0; dim];
            let draws: Vec<Vec<f64>> = (0..200)
                .map(|_| {
                    for value in state.iter_mut() {
                        let z: f64 = StandardNormal.sample(&mut rng);
                        *value = 0.6 * *value + 0.8 * z;
                    }
                    state.clone()
                })
                .collect();
            assert!(
                !estimate(&graph, MetricKind::Auto, &draws).has_dense_block(),
                "seed {seed}"
            );
        }
    }

    /// Condition number of the covariance a dense block's factor encodes.
    fn dense_condition(mass: &MassMatrix) -> f64 {
        let BlockKind::Dense { chol } = &mass.blocks[0].kind else {
            panic!("expected a dense block");
        };
        let dim = mass.blocks[0].len;
        let cov = faer::Mat::from_fn(dim, dim, |i, j| {
            (0..dim)
                .map(|k| chol[i * dim + k] * chol[j * dim + k])
                .sum::<f64>()
        });
        let eigenvalues = cov.selfadjoint_eigenvalues(faer::Side::Lower);
        let (min, max) = eigenvalues
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        max / min
    }

    #[test]
    fn dense_estimates_from_few_draws_per_dimension_stay_well_conditioned() {
        // Twenty draws in thirty dimensions: the sample covariance is singular
        // and, regularized only toward a small multiple of the identity, had
        // eigenvalues of 2e-4 beside ones near 4 (condition near 2e4).
        let graph = vector_graph(30);
        for seed in 0..5 {
            let draws = equicorrelated_draws(30, 0.0, 20, seed);
            let condition = dense_condition(&estimate(&graph, MetricKind::Dense, &draws));
            assert!(condition < 30.0, "seed {seed}: condition {condition}");
        }
    }

    #[test]
    fn correlations_are_shrunk_only_below_two_draws_per_dimension() {
        let dim = 4;
        let original: Vec<f64> = (0..dim * dim)
            .map(|k| if k % (dim + 1) == 0 { 2.0 } else { 0.9 })
            .collect();
        for (draws, weight) in [(8, 0.0), (100, 0.0), (6, 0.25), (4, 0.5), (1, 0.875)] {
            let mut cov = original.clone();
            assert_eq!(shrink_correlations(&mut cov, dim, draws), weight);
            for (k, (&shrunk, &value)) in cov.iter().zip(&original).enumerate() {
                let expected = if k % (dim + 1) == 0 {
                    value
                } else {
                    value * (1.0 - weight)
                };
                assert_eq!(shrunk, expected, "draws {draws}");
            }
        }
    }

    #[test]
    fn dense_factorization_adds_no_jitter_when_it_succeeds() {
        let cov = vec![4.0, 1.0, 1.0, 1.0];
        let chol = cholesky_with_jitter(cov, 2);
        assert_eq!(chol, vec![2.0, 0.0, 0.5, 0.75_f64.sqrt()]);
        // A singular matrix still factors, through the jittered retry.
        let singular = cholesky_with_jitter(vec![1.0, 1.0, 1.0, 1.0], 2);
        assert!(singular.iter().all(|value| value.is_finite()));
        assert!(singular[3] > 0.0);
    }

    #[test]
    fn metric_names_parse() {
        assert_eq!(MetricKind::parse("auto"), Ok(MetricKind::Auto));
        assert_eq!(MetricKind::parse("diag"), Ok(MetricKind::Diagonal));
        assert_eq!(MetricKind::parse("dense"), Ok(MetricKind::Dense));
        assert!(MetricKind::parse("unit").is_err());
        assert_eq!("diagonal".parse(), Ok(MetricKind::Diagonal));
        assert_eq!(
            "unit".parse::<MetricKind>(),
            Err(MetricKind::parse("unit").unwrap_err())
        );
    }
}
