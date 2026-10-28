//! A Normal observation term on a grouped linear predictor,
//! `y ~ Normal(mu + tau * z[group], sigma)`, evaluated from per-group
//! statistics of its observations instead of row by row.
//!
//! Within a group every row has the same mean `m_g = mu + tau * z_g`, so the
//! term's sum of squares splits exactly into a part that depends only on the
//! data and a part that depends on the parameters only through `m_g`:
//!
//! ```text
//! sum over rows i of g of (y_i - m_g)^2 = S_g^2 + n_g * (ybar_g - m_g)^2
//! ```
//!
//! with `n_g` the group's rows, `ybar_g` their mean and `S_g` the Euclidean
//! norm of their deviations from it. The evaluator computes those three per
//! group once, when it is built or its data are rebound: it sorts the rows by
//! group (a stable counting sort) and reduces each group's contiguous segment.
//! No pass of a gradient then touches the rows. The forward pass transforms
//! each effect of a group that has rows and makes one loop over those groups
//! for the term's log density and the sums its adjoints need; the reverse
//! pass makes one more over them, for the effects' adjoints, and takes the
//! adjoints of `mu`, `tau` and `sigma` from the sums, all in closed form. A group with no rows takes
//! no part, and the group effects are transformed once per group that has
//! rows, so a gather with more groups than rows costs no more than its rows.
//!
//! Accuracy. The mean is kept as an unevaluated sum of two doubles, so
//! `ybar_g - m_g` carries a few units of roundoff relative to itself, as one
//! row's `y_i - m_g` did, even when the data sit far from zero; `S_g` is a
//! scaled, compensated norm. Observations within `2^-960` of zero, where
//! the low half of the mean would be subnormal, send the chain to the
//! separate nodes. The loop over the groups sums in four interleaved lanes.
//! In exact arithmetic each group's summand equals the sum of its rows'
//! summands, so its magnitude is at most theirs, and it carries a few units
//! of roundoff of its own (the square of the rounded norm, the two-double
//! mean). A sum over `G` groups is so within about `(G + c) u` of exact,
//! relative to the sum of the row summands' magnitudes, where the row loops'
//! bound was `(N + c) u` over `N` rows; `u` is the unit roundoff and `c` a
//! small constant. That is a bound, not a promise for every input; the tests
//! measure both paths against a double-double reference. Each per-row
//! quantity is formed as the separate term formed it for one row and only
//! then multiplied by the group's row count. A call whose scale, mean,
//! product scale or effects lie outside `2^±128`, or a chain whose
//! observations exceed `2^128`, runs the separate nodes, and inside that
//! range nothing the pass forms can overflow (see `SAFE_MAGNITUDE`). Where
//! a row's adjoint falls in the subnormal range, both paths lose precision,
//! and they lose it differently. The results are not the separate nodes'
//! bits: sums are taken in another order.
//!
//! Groups whose effects mix transforms run fused too, each transformed once
//! with its own transform.
//!
//! The gather, product and sum vectors of a fused chain are not written. The
//! evaluator keeps each group's transformed effect from the last forward
//! pass, and [`Evaluator::vec_elem`], which is how predictions, posterior and
//! prior predictive draws, pointwise log-likelihoods and deterministics read
//! node values, computes a row's value of any of those three nodes on demand
//! from it, with the same expressions the separate nodes use, so those reads
//! give the separate nodes' bits.
//!
//! A chain is fused only when nothing else can observe the difference: the
//! only nodes between its own are constants and data, which take no part in
//! the reverse pass, each vector in it feeds only the next node, the group
//! indices are data, and the scale of the product is not also the chain's
//! mean or scale. The chain is decided at its gather and runs at its term,
//! once any constant between has its value. Every call checks that the graph
//! still holds the chain the evaluator found, and runs the separate nodes
//! otherwise.

use super::{scale_is_valid, Evaluator, NodeKind};
use crate::data::{is_group_index, DataBinding};
use crate::graph::{ElementwiseOp, Graph, NodeId, ObsFamily, Op, ParamTransform};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Fused forward and reverse passes run by every evaluator dropped so far,
/// in this process. Each evaluator counts its own and adds them here when it
/// is dropped, so sampling threads don't share a counter per gradient.
static FUSED_FORWARD_PASSES: AtomicU64 = AtomicU64::new(0);
static FUSED_REVERSE_PASSES: AtomicU64 = AtomicU64::new(0);

/// Set to run every grouped Normal chain as its separate nodes, in every
/// evaluator of the process; see [`set_grouped_normal_fusion`].
static FUSION_DISABLED: AtomicBool = AtomicBool::new(false);

/// How many fused grouped-Normal forward and reverse passes the evaluators
/// dropped so far in this process have run. Tests and benchmarks read it to
/// show that the fused loops ran.
#[doc(hidden)]
pub fn fused_grouped_normal_passes() -> (u64, u64) {
    (
        FUSED_FORWARD_PASSES.load(Ordering::Relaxed),
        FUSED_REVERSE_PASSES.load(Ordering::Relaxed),
    )
}

/// Allow or forbid fused grouped-Normal chains in every evaluator of this
/// process, and return whether they were allowed. Each forward pass reads
/// the setting once, when it starts, so an evaluation already running
/// finishes as it began. For tests that compare an output computed through
/// the fused chains with the same output computed through the separate
/// nodes. It is process-wide: a test that turns fusion off must not run
/// beside one that counts fused passes, and changing it while a fit runs
/// changes that fit's draws.
#[doc(hidden)]
pub fn set_grouped_normal_fusion(enabled: bool) -> bool {
    !FUSION_DISABLED.swap(!enabled, Ordering::Relaxed)
}

/// Whether [`set_grouped_normal_fusion`] allows fused chains.
fn fusion_allowed() -> bool {
    !FUSION_DISABLED.load(Ordering::Relaxed)
}

/// No slot: a group of the gather that no row selects.
const NO_SLOT: u32 = u32::MAX;

/// `2^-960`. A chain whose observations come closer to zero than this,
/// other than zero itself, runs as separate nodes; see [`GroupStats::build`].
const TINY_OBSERVATION: f64 = f64::MIN_POSITIVE * (1u64 << 62) as f64;

/// `2^128`. The fused pass runs only while the observations, the mean, the
/// product scale and every effect it reads are at most this in magnitude
/// and the term's scale lies between its reciprocal and it; any other call
/// runs the separate nodes. Inside that range no quantity the pass forms
/// can overflow: with at most `2^32` rows, a mean is at most `2^257`, a
/// standardized residual `2^386`, a row's adjoint of its mean `2^514`, that
/// times an effect `2^642`, and any sum over the groups below `2^710`.
const SAFE_MAGNITUDE: f64 = f64::from_bits((1023 + 128) << 52);

/// What the fused pass needs of a chain's observations, per group that has
/// rows (a slot), in increasing group order.
#[derive(Debug, Default)]
struct GroupStats {
    /// Whether the statistics describe the current binding and are all
    /// finite. A chain without usable statistics runs as separate nodes.
    usable: bool,
    /// The group of each slot.
    groups: Vec<usize>,
    /// The slot of each group of the gather, or [`NO_SLOT`].
    slot_of_group: Vec<u32>,
    /// Rows per slot.
    counts: Vec<f64>,
    /// Each slot's mean as an unevaluated sum `mean_hi + mean_lo`.
    mean_hi: Vec<f64>,
    mean_lo: Vec<f64>,
    /// The Euclidean norm of each slot's deviations from its mean.
    spread: Vec<f64>,
}

/// `a + b` and its rounding error, exactly (Knuth's two-sum).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virtual = s - a;
    let a_virtual = s - b_virtual;
    (s, (a - a_virtual) + (b - b_virtual))
}

/// The mean of `values`, `n` of them, as an unevaluated sum of two doubles
/// whose error is of order `u^2` times the mean of their absolute values.
fn double_mean(values: impl Iterator<Item = f64>, n: f64) -> (f64, f64) {
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    for value in values {
        let (sum, error) = two_sum(hi, value);
        hi = sum;
        lo += error;
    }
    let (hi, lo) = two_sum(hi, lo);
    let mean_hi = hi / n;
    // The remainder of a correctly rounded division is representable, and a
    // fused multiply-add computes it with one rounding, so exactly.
    let remainder = (-mean_hi).mul_add(n, hi);
    (mean_hi, (remainder + lo) / n)
}

/// `sqrt(sum of squares)` of `values`, scaled by their largest magnitude so
/// that no square overflows or underflows first, with the squares summed by
/// Neumaier's compensated sum.
fn norm(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let scale = values.clone().fold(0.0f64, |max, v| max.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for value in values {
        let x = value / scale;
        let square = x * x;
        let t = sum + square;
        compensation += if sum >= square {
            (sum - t) + square
        } else {
            (square - t) + sum
        };
        sum = t;
    }
    scale * (sum + compensation).sqrt()
}

impl GroupStats {
    /// Compute the statistics of `obs` grouped by `groups`, for a gather of
    /// `n_groups` groups, reusing the allocations. Not usable when the two
    /// lengths differ from `len`, an index is not a group, or a statistic is
    /// not finite.
    fn build(&mut self, groups: &[f64], obs: &[f64], n_groups: usize, len: usize) {
        self.usable = false;
        self.groups.clear();
        self.slot_of_group.clear();
        self.counts.clear();
        self.mean_hi.clear();
        self.mean_lo.clear();
        self.spread.clear();
        if groups.len() != len
            || obs.len() != len
            || u32::try_from(len).is_err()
            || !groups.iter().all(|&g| is_group_index(g, n_groups))
        {
            return;
        }
        // A stable counting sort of the rows by group: `order` lists the rows
        // of group 0, then of group 1, and so on, each in row order.
        let mut starts = vec![0usize; n_groups + 1];
        for &group in groups {
            starts[group as usize + 1] += 1;
        }
        for g in 0..n_groups {
            starts[g + 1] += starts[g];
        }
        let mut next = starts[..n_groups].to_vec();
        let mut order = vec![0u32; len];
        for (row, &group) in groups.iter().enumerate() {
            let group = group as usize;
            order[next[group]] = row as u32;
            next[group] += 1;
        }
        self.slot_of_group.resize(n_groups, NO_SLOT);
        for g in 0..n_groups {
            let segment = &order[starts[g]..starts[g + 1]];
            if segment.is_empty() {
                continue;
            }
            self.slot_of_group[g] = self.groups.len() as u32;
            self.groups.push(g);
            let n = segment.len() as f64;
            let rows = segment.iter().map(|&row| obs[row as usize]);
            let (mean_hi, mean_lo) = double_mean(rows.clone(), n);
            self.counts.push(n);
            self.mean_hi.push(mean_hi);
            self.mean_lo.push(mean_lo);
            self.spread
                .push(norm(rows.map(|y| (y - mean_hi) - mean_lo)));
        }
        // The mean's low half must not be subnormal, or it loses the digits
        // that make `ybar_g - m_g` accurate: refuse observations within
        // `2^-960` of zero, other than zero itself.
        let tiny = obs.iter().any(|&y| y != 0.0 && y.abs() < TINY_OBSERVATION);
        let huge = obs.iter().any(|&y| y.abs() > SAFE_MAGNITUDE);
        self.usable = !tiny
            && !huge
            && self
                .mean_hi
                .iter()
                .chain(&self.mean_lo)
                .chain(&self.spread)
                .all(|v| v.is_finite());
    }
}

/// A fused chain, by node index: `gather`, then optionally `product =
/// tau * gather` or `gather * tau`, then optionally `sum = mu + (product or
/// gather)`, then the Normal term `obs` on the last of them. Any other node
/// between them is a constant or data.
#[derive(Debug)]
pub(super) struct GroupedNormal {
    gather: usize,
    product: Option<(usize, usize)>,
    /// Whether the product is written `tau * gather`, rather than
    /// `gather * tau`.
    scale_first: bool,
    sum: Option<(usize, usize)>,
    obs: usize,
    sigma: usize,
    param_start: usize,
    n_params: usize,
    /// The gather's index node, and the data slot it read when the
    /// evaluator was built, which the fused pass reads.
    indices_node: usize,
    indices: usize,
    obs_data_idx: usize,
    len: usize,
    stats: GroupStats,
    /// Per slot, the group's transformed effect from the last forward pass,
    /// which [`Evaluator::grouped_value`] reads.
    values: Vec<f64>,
    /// Per slot, each of the group's rows' adjoint of its mean for a term
    /// adjoint of one, `(ybar_g - m_g) / sigma^2`, from the last forward
    /// pass: the value each row of the separate term would have added.
    row_adjoints: Vec<f64>,
    /// The term's sum of squared standardized residuals, the sum over rows
    /// of `row_adjoints`, and that sum weighted by each row's effect.
    sum_sq: f64,
    mean_adjoint: f64,
    scale_adjoint: f64,
}

/// The fused chains of an evaluator and where they start and end.
#[derive(Debug)]
pub(super) struct GroupedPlan {
    chains: Vec<GroupedNormal>,
    /// Per node, the chain whose gather it is.
    starts: Vec<Option<usize>>,
    /// Per node, the chain whose Normal term it is.
    ends: Vec<Option<usize>>,
    /// Per node, the chain whose gather, product or sum it is.
    members: Vec<Option<usize>>,
    /// Per chain, whether the last forward pass found the chain still in
    /// the graph and ran it fused, so the reverse pass may too, and reads of
    /// its vectors compute them.
    fused_this_call: Vec<bool>,
    /// Every node's readers in the graph of the current call, counted in one
    /// pass when the first chain of the call needs them, so a call costs one
    /// scan of the graph however many chains it checks, and none when every
    /// chain fails a cheaper check first. Empty for a plan with no chains.
    readers: Vec<u32>,
    readers_counted: bool,
    /// [`fusion_allowed`] as the current call found it when it began.
    allowed: bool,
    forward_passes: u64,
    /// Calls that found a chain fusable but its values outside
    /// [`SAFE_MAGNITUDE`], and ran its separate nodes.
    fallbacks: u64,
    reverse_passes: u64,
}

impl Drop for GroupedPlan {
    fn drop(&mut self) {
        FUSED_FORWARD_PASSES.fetch_add(self.forward_passes, Ordering::Relaxed);
        FUSED_REVERSE_PASSES.fetch_add(self.reverse_passes, Ordering::Relaxed);
    }
}

/// Call `f` with every node `op` reads. Exhaustive, so a new operation
/// cannot be missed by the single-consumer check.
fn for_each_input(op: &Op, mut f: impl FnMut(NodeId)) {
    match op {
        Op::Elementwise { a, b, .. } => {
            f(*a);
            if let Some(b) = b {
                f(*b);
            }
        }
        Op::Gather { indices, .. } => f(*indices),
        Op::Sum(a) | Op::Exp(a) | Op::Sigmoid(a) | Op::ScalarBroadcast(a) => f(*a),
        Op::BroadcastObservation { scalar, .. } => f(*scalar),
        Op::Param(_) | Op::Constant(_) | Op::Data(_) => {}
        Op::Add(a, b)
        | Op::Mul(a, b)
        | Op::ScalarMulData(a, b)
        | Op::VectorAdd(a, b)
        | Op::ScalarBroadcastAdd(a, b) => {
            f(*a);
            f(*b);
        }
        Op::BoundedSigmoid { raw, .. } => f(*raw),
        Op::NormalLogP { x, mu, sigma } => {
            f(*x);
            f(*mu);
            f(*sigma);
        }
        Op::ObsLogP {
            linpred_vec, aux, ..
        } => {
            f(*linpred_vec);
            if let Some(aux) = aux {
                f(*aux);
            }
        }
        Op::LogHalfNormalLogP { x, sigma } => {
            f(*x);
            f(*sigma);
        }
        Op::StudentTLogP { x, nu, mu, sigma } => {
            f(*x);
            f(*nu);
            f(*mu);
            f(*sigma);
        }
        Op::PositiveSupport { x } => f(*x),
        Op::BernoulliLogP { x, p } => {
            f(*x);
            f(*p);
        }
        Op::PoissonLogP { x, lam } => {
            f(*x);
            f(*lam);
        }
        Op::LogGammaLogP { x, alpha, beta } => {
            f(*x);
            f(*alpha);
            f(*beta);
        }
        Op::FusedLinearMu {
            param_nodes,
            intercept,
            ..
        } => {
            param_nodes.iter().for_each(|&node| f(node));
            if let Some(intercept) = intercept {
                f(*intercept);
            }
        }
        Op::MatVecMul { intercept, .. } => {
            if let Some(intercept) = intercept {
                f(*intercept);
            }
        }
        Op::VectorNormalLogP { .. }
        | Op::VectorHalfNormalLogP { .. }
        | Op::VectorStudentTLogP { .. }
        | Op::VectorGammaLogP { .. }
        | Op::VectorBetaLogP { .. }
        | Op::VectorUniformLogP { .. } => {}
    }
}

impl GroupedNormal {
    /// The chain ending at the Normal term `obs`, if `graph` holds one the
    /// fused pass can evaluate. `consumers` counts each node's readers.
    fn find(
        graph: &Graph,
        obs: usize,
        consumers: &[usize],
        node_kind: &[NodeKind],
        node_lengths: &[usize],
        binding: &DataBinding,
    ) -> Option<Self> {
        let Op::ObsLogP {
            family: ObsFamily::Normal,
            linpred_vec,
            aux: Some(sigma),
            obs_data_idx,
        } = graph.nodes[obs].op
        else {
            return None;
        };
        let scalar = |node: NodeId| matches!(node_kind[node.0], NodeKind::Scalar);
        let mut next = linpred_vec.0;
        let mut sum = None;
        if let Op::ScalarBroadcastAdd(mu, vec) = graph.nodes[next].op {
            if !scalar(mu) {
                return None;
            }
            sum = Some((next, mu.0));
            next = vec.0;
        }
        let mut product = None;
        let mut scale_first = true;
        if let Op::Elementwise {
            operator: ElementwiseOp::Mul,
            a,
            b: Some(b),
        } = graph.nodes[next].op
        {
            let (tau, vec) = match (scalar(a), scalar(b)) {
                (true, false) => (a, b),
                (false, true) => {
                    scale_first = false;
                    (b, a)
                }
                _ => return None,
            };
            product = Some((next, tau.0));
            next = vec.0;
        }
        let Op::Gather {
            param_start,
            n_params,
            indices,
        } = graph.nodes[next].op
        else {
            return None;
        };
        let NodeKind::DataRef(indices_data) = node_kind[indices.0] else {
            return None;
        };
        let len = binding.observations[obs_data_idx].len();
        let chain = [Some(next), product.map(|p| p.0), sum.map(|s| s.0)];
        let chain_ok = scalar(sigma)
            && chain.iter().flatten().all(|&node| {
                consumers[node] == 1
                    && node_lengths[node] == len
                    && matches!(node_kind[node], NodeKind::ComputedVec(_))
            })
            && binding.vectors[indices_data].len() == len;
        // The fused pass adds each scalar's adjoint once, so the product's
        // scale must not also be the term's scale or the chain's mean.
        let tau_ok =
            product.is_none_or(|(_, tau)| tau != sigma.0 && sum.is_none_or(|(_, mu)| tau != mu));
        let mut found = Self {
            gather: next,
            product,
            scale_first,
            sum,
            obs,
            sigma: sigma.0,
            param_start,
            n_params,
            indices_node: indices.0,
            indices: indices_data,
            obs_data_idx,
            len,
            stats: GroupStats::default(),
            values: Vec::new(),
            row_adjoints: Vec::new(),
            sum_sq: 0.0,
            mean_adjoint: 0.0,
            scale_adjoint: 0.0,
        };
        if !(chain_ok && tau_ok && found.in_order_with_only_constants_between(graph)) {
            return None;
        }
        found.rebuild(binding);
        Some(found)
    }

    /// Recompute the statistics from `binding`, and size the per-call
    /// buffers for them.
    fn rebuild(&mut self, binding: &DataBinding) {
        let groups = binding.vectors.get(self.indices).map_or(&[][..], |v| v);
        let obs = binding
            .observations
            .get(self.obs_data_idx)
            .map_or(&[][..], |v| v);
        self.stats.build(groups, obs, self.n_params, self.len);
        let slots = self.stats.groups.len();
        self.values.clear();
        self.values.reserve(slots);
        self.row_adjoints.clear();
        self.row_adjoints.resize(slots, 0.0);
    }

    /// The chain's gather, product and sum, those it has.
    fn vectors(&self) -> [Option<usize>; 3] {
        [
            Some(self.gather),
            self.product.map(|p| p.0),
            self.sum.map(|s| s.0),
        ]
    }

    /// Whether the chain's nodes come in order and every other node between
    /// its gather and its term is a constant or data, whose reverse passes
    /// add nothing, so running the chain at its term changes no other node.
    fn in_order_with_only_constants_between(&self, graph: &Graph) -> bool {
        let order = self.vectors().into_iter().flatten().chain([self.obs]);
        let increasing = order
            .clone()
            .zip(order.skip(1))
            .all(|(earlier, later)| earlier < later);
        increasing
            && (self.gather + 1..self.obs).all(|node| {
                self.vectors().contains(&Some(node))
                    || matches!(graph.nodes[node].op, Op::Constant(_) | Op::Data(_))
            })
    }

    /// Whether each vector in the chain still has exactly one reader, given
    /// every node's readers in the graph as it is now: a graph changed after
    /// construction could have given one another.
    fn read_once(&self, readers: &[u32]) -> bool {
        self.vectors()
            .into_iter()
            .flatten()
            .all(|node| readers[node] == 1)
    }

    /// Whether `graph` still holds this chain, node for node. The node count
    /// cannot change without a new evaluator, but the operations can.
    fn still_in(&self, graph: &Graph) -> bool {
        let gather_ok = matches!(
            graph.nodes[self.gather].op,
            Op::Gather { param_start, n_params, indices }
                if param_start == self.param_start && n_params == self.n_params
                    && indices.0 == self.indices_node
        );
        let input_of_sum = self.product.map_or(self.gather, |(p, _)| p);
        let linpred = self.sum.map_or(input_of_sum, |(s, _)| s);
        let product_ok = self.product.is_none_or(|(p, tau)| {
            let (scale, vector) = if self.scale_first {
                (tau, self.gather)
            } else {
                (self.gather, tau)
            };
            matches!(
                graph.nodes[p].op,
                Op::Elementwise { operator: ElementwiseOp::Mul, a, b: Some(b) }
                    if a.0 == scale && b.0 == vector
            )
        });
        let sum_ok = self.sum.is_none_or(|(s, mu)| {
            matches!(
                graph.nodes[s].op,
                Op::ScalarBroadcastAdd(a, b) if a.0 == mu && b.0 == input_of_sum
            )
        });
        let obs_ok = matches!(
            graph.nodes[self.obs].op,
            Op::ObsLogP { family: ObsFamily::Normal, linpred_vec, aux: Some(aux), obs_data_idx }
                if linpred_vec.0 == linpred && aux.0 == self.sigma
                    && obs_data_idx == self.obs_data_idx
        );
        gather_ok
            && product_ok
            && sum_ok
            && obs_ok
            && self.in_order_with_only_constants_between(graph)
    }
}

impl GroupedPlan {
    pub(super) fn new(
        graph: &Graph,
        node_kind: &[NodeKind],
        node_lengths: &[usize],
        binding: &DataBinding,
    ) -> Self {
        let n = graph.nodes.len();
        let mut plan = Self {
            chains: Vec::new(),
            starts: vec![None; n],
            ends: vec![None; n],
            members: vec![None; n],
            fused_this_call: Vec::new(),
            readers: Vec::new(),
            readers_counted: false,
            allowed: true,
            forward_passes: 0,
            fallbacks: 0,
            reverse_passes: 0,
        };
        let Some(consumers) = Self::consumers(graph) else {
            return plan;
        };
        for obs in 0..n {
            if let Some(chain) =
                GroupedNormal::find(graph, obs, &consumers, node_kind, node_lengths, binding)
            {
                let c = Some(plan.chains.len());
                plan.starts[chain.gather] = c;
                plan.ends[chain.obs] = c;
                for node in chain.vectors().into_iter().flatten() {
                    plan.members[node] = c;
                }
                plan.chains.push(chain);
            }
        }
        plan.fused_this_call = vec![false; plan.chains.len()];
        if !plan.chains.is_empty() {
            plan.readers = vec![0; n];
        }
        plan
    }

    /// Recompute every chain's statistics for a binding of the same shape;
    /// until the next forward pass, reads of the chains' vectors read the
    /// buffers, as for a chain that ran as separate nodes.
    pub(super) fn rebind(&mut self, binding: &DataBinding) {
        for chain in &mut self.chains {
            chain.rebuild(binding);
        }
        self.fused_this_call.fill(false);
    }

    /// Readers per node, or `None` when a node reads one past the end.
    fn consumers(graph: &Graph) -> Option<Vec<usize>> {
        let mut consumers = vec![0usize; graph.nodes.len()];
        let mut in_range = true;
        for node in &graph.nodes {
            for_each_input(&node.op, |input| match consumers.get_mut(input.0) {
                Some(count) => *count += 1,
                None => in_range = false,
            });
        }
        in_range.then_some(consumers)
    }

    pub(super) fn chain_starting_at(&self, node: usize) -> Option<usize> {
        self.starts.get(node).copied().flatten()
    }

    pub(super) fn chain_ending_at(&self, node: usize) -> Option<usize> {
        self.ends.get(node).copied().flatten()
    }

    /// Whether the evaluator found any chain to fuse.
    pub(super) fn has_chains(&self) -> bool {
        !self.chains.is_empty()
    }

    /// Forget the reader counts of the previous call, since the graph may
    /// have changed, and read whether the process allows fusion, once for
    /// the whole call.
    pub(super) fn begin_call(&mut self) {
        self.readers_counted = false;
        self.allowed = fusion_allowed();
    }

    /// Whether the process allowed fusion when this call began.
    pub(super) fn allowed(&self) -> bool {
        self.allowed
    }

    /// Whether chain `c`'s vectors each have one reader in `graph`, counting
    /// every node's readers first if this call has not yet. The count reads
    /// each node's inputs once, without allocating; a node past the end
    /// counts for no node, as `consumers` would refuse such a graph.
    fn read_once(&mut self, graph: &Graph, c: usize) -> bool {
        if !self.readers_counted {
            self.readers.iter_mut().for_each(|count| *count = 0);
            let readers = &mut self.readers;
            for node in &graph.nodes {
                for_each_input(&node.op, |input| {
                    if let Some(count) = readers.get_mut(input.0) {
                        *count += 1;
                    }
                });
            }
            self.readers_counted = true;
        }
        self.chains[c].read_once(&self.readers)
    }

    /// Whether this call runs chain `c` fused.
    pub(super) fn fused(&self, c: usize) -> bool {
        self.fused_this_call[c]
    }

    /// Whether `node` is a gather, product or sum of a chain this call runs
    /// fused, which the node-by-node passes then skip.
    pub(super) fn skips(&self, node: usize) -> bool {
        self.members
            .get(node)
            .copied()
            .flatten()
            .is_some_and(|c| self.fused_this_call[c])
    }
}

/// The fused pass over a chain's groups: each group's mean from its effect,
/// in the separate nodes' expressions, its standardized mean residual, and
/// the term's sums. Four interleaved lanes, so the sums vectorize.
struct GroupPass<'a, const PRODUCT: bool, const SCALE_FIRST: bool, const SUM: bool> {
    values: &'a [f64],
    counts: &'a [f64],
    mean_hi: &'a [f64],
    mean_lo: &'a [f64],
    spread: &'a [f64],
    tau: f64,
    mu: f64,
    sv: f64,
    row_adjoints: &'a mut [f64],
}

/// One group's row adjoint `g = (ybar_g - m_g) / sigma^2`, its share of
/// the sum of squares, and its rows' sums of `g` and of `g` times the
/// effect. Each per-row quantity is formed as the separate term formed it
/// for one row, `(y - m) / sigma` and then `/ sigma`, and multiplied by the
/// group's rows last, so no intermediate overflows where the rows' did not.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn group_terms<const PRODUCT: bool, const SCALE_FIRST: bool, const SUM: bool>(
    z: f64,
    n: f64,
    mean_hi: f64,
    mean_lo: f64,
    spread: f64,
    tau: f64,
    mu: f64,
    sv: f64,
) -> (f64, f64, f64, f64) {
    let mut m = z;
    if PRODUCT {
        m = if SCALE_FIRST {
            ElementwiseOp::Mul.value(tau, m)
        } else {
            ElementwiseOp::Mul.value(m, tau)
        };
    }
    if SUM {
        // `ScalarBroadcastAdd`'s `mu + v`; addition commutes exactly.
        m += mu;
    }
    let e = ((mean_hi - m) + mean_lo) / sv;
    let s = spread / sv;
    let g = e / sv;
    (g, s * s + n * (e * e), n * g, n * (g * z))
}

impl<const PRODUCT: bool, const SCALE_FIRST: bool, const SUM: bool> pulp::WithSimd
    for GroupPass<'_, PRODUCT, SCALE_FIRST, SUM>
{
    type Output = [f64; 3];

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> [f64; 3] {
        let n = self.values.len();
        let (values, counts) = (&self.values[..n], &self.counts[..n]);
        let (mean_hi, mean_lo) = (&self.mean_hi[..n], &self.mean_lo[..n]);
        let spread = &self.spread[..n];
        let out = &mut self.row_adjoints[..n];
        let mut lanes = [[0.0f64; 4]; 3];
        let whole = n - n % 4;
        let term = |j: usize| {
            group_terms::<PRODUCT, SCALE_FIRST, SUM>(
                values[j], counts[j], mean_hi[j], mean_lo[j], spread[j], self.tau, self.mu, self.sv,
            )
        };
        for start in (0..whole).step_by(4) {
            for lane in 0..4 {
                let (g, q, mean, scale) = term(start + lane);
                out[start + lane] = g;
                lanes[0][lane] += q;
                lanes[1][lane] += mean;
                lanes[2][lane] += scale;
            }
        }
        let mut tails = [0.0f64; 3];
        for (j, slot) in out.iter_mut().enumerate().skip(whole) {
            let (g, q, mean, scale) = term(j);
            *slot = g;
            tails[0] += q;
            tails[1] += mean;
            tails[2] += scale;
        }
        std::array::from_fn(|k| {
            let l = lanes[k];
            (l[0] + l[1]) + (l[2] + l[3]) + tails[k]
        })
    }
}

impl Evaluator {
    /// Decide, at chain `c`'s gather, whether this call runs it fused: the
    /// chain has usable statistics, its span lies inside the parameters and
    /// their transforms, the graph still holds it, and its vectors' only
    /// readers are its own nodes.
    // Out of line, as are the fused passes. Inlined into the evaluator's node
    // loops, the earlier fused loops made a graph that declined fusion about
    // 7% slower.
    #[inline(never)]
    pub(super) fn grouped_decide(&mut self, graph: &Graph, params: &[f64], c: usize) {
        let chain = &self.grouped.chains[c];
        let span_fits = chain
            .param_start
            .checked_add(chain.n_params)
            .is_some_and(|end| end <= params.len() && end <= graph.param_transforms.len());
        // The cheaper checks first; the reader count scans the graph, once
        // per call.
        let fused = self.fusion_enabled()
            && chain.stats.usable
            && span_fits
            && chain.still_in(graph)
            && self.grouped.read_once(graph, c);
        self.grouped.fused_this_call[c] = fused;
    }

    /// Run chain `c` of the grouped plan forward, at its term, after any
    /// constant between its nodes has its value: transform each group's
    /// effect, and take the term's log density and the sums its reverse pass
    /// reads in one loop over the groups. Only for a chain
    /// [`Self::grouped_decide`] chose to fuse in this call.
    ///
    /// When a value the pass reads lies outside [`SAFE_MAGNITUDE`], it
    /// writes the chain's vectors instead, marks the chain unfused for this
    /// call and returns false, and the caller runs the separate term; the
    /// reverse pass then runs the separate nodes too.
    #[inline(never)]
    pub(super) fn grouped_forward(&mut self, graph: &Graph, params: &[f64], c: usize) -> bool {
        let tau = self.grouped.chains[c]
            .product
            .map(|(_, tau)| self.scalars[tau]);
        let mu = self.grouped.chains[c].sum.map(|(_, mu)| self.scalars[mu]);
        let sv = self.scalars[self.grouped.chains[c].sigma];
        let chain = &mut self.grouped.chains[c];
        chain.values.clear();
        chain.values.extend(chain.stats.groups.iter().map(|&g| {
            let k = chain.param_start + g;
            graph.param_transforms[k].apply(params[k])
        }));
        let small = |v: f64| v.abs() <= SAFE_MAGNITUDE;
        let in_range = chain.values.iter().all(|&v| small(v))
            && tau.is_none_or(small)
            && mu.is_none_or(small)
            && (!scale_is_valid(sv) || (small(sv) && small(1.0 / sv)));
        if !in_range {
            self.grouped.fused_this_call[c] = false;
            self.grouped.fallbacks += 1;
            self.write_grouped_vectors(c);
            return false;
        }
        self.grouped.forward_passes += 1;
        let GroupedNormal {
            stats,
            values,
            row_adjoints,
            scale_first,
            obs,
            len,
            ..
        } = &mut self.grouped.chains[c];
        if !scale_is_valid(sv) {
            // As the separate term: outside the scale's support the density
            // is zero, not NaN.
            self.scalars[*obs] = f64::NEG_INFINITY;
            return true;
        }
        macro_rules! pass {
            ($product:literal, $scale_first:literal, $sum:literal) => {
                pulp::Arch::new().dispatch(GroupPass::<$product, $scale_first, $sum> {
                    values,
                    counts: &stats.counts,
                    mean_hi: &stats.mean_hi,
                    mean_lo: &stats.mean_lo,
                    spread: &stats.spread,
                    tau: tau.unwrap_or(0.0),
                    mu: mu.unwrap_or(0.0),
                    sv,
                    row_adjoints,
                })
            };
        }
        // One loop per shape of chain, so no group branches on the shape.
        let [sum_sq, mean_adjoint, scale_adjoint] =
            match (tau.is_some(), *scale_first, mu.is_some()) {
                (false, _, false) => pass!(false, false, false),
                (false, _, true) => pass!(false, false, true),
                (true, false, false) => pass!(true, false, false),
                (true, false, true) => pass!(true, false, true),
                (true, true, false) => pass!(true, true, false),
                (true, true, true) => pass!(true, true, true),
            };
        let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
        self.scalars[*obs] = *len as f64 * log_norm - 0.5 * sum_sq;
        let chain = &mut self.grouped.chains[c];
        chain.sum_sq = sum_sq;
        chain.mean_adjoint = mean_adjoint;
        chain.scale_adjoint = scale_adjoint;
        true
    }

    /// Write chain `c`'s gather, product and sum vectors from the effects
    /// its forward pass just took, as the separate nodes would have, for a
    /// call that runs its separate term after all.
    #[cold]
    fn write_grouped_vectors(&mut self, c: usize) {
        let chain = &self.grouped.chains[c];
        let offset = |node: usize| match self.node_kind[node] {
            NodeKind::ComputedVec(off) => off,
            _ => unreachable!("checked when the chain was found"),
        };
        let groups = &self.binding.vectors[chain.indices][..chain.len];
        let tau = chain.product.map(|(_, tau)| self.scalars[tau]);
        let mu = chain.sum.map(|(_, mu)| self.scalars[mu]);
        let gather = offset(chain.gather);
        let product = chain.product.map(|(p, _)| offset(p));
        let sum = chain.sum.map(|(s, _)| offset(s));
        for (i, &group) in groups.iter().enumerate() {
            let mut value = chain.values[chain.stats.slot_of_group[group as usize] as usize];
            self.vec_buf[gather + i] = value;
            if let (Some(off), Some(tau)) = (product, tau) {
                value = if chain.scale_first {
                    ElementwiseOp::Mul.value(tau, value)
                } else {
                    ElementwiseOp::Mul.value(value, tau)
                };
                self.vec_buf[off + i] = value;
            }
            if let (Some(off), Some(mu)) = (sum, mu) {
                self.vec_buf[off + i] = mu + value;
            }
        }
    }

    /// Run chain `c` of the grouped plan in reverse, if this call's forward
    /// pass ran it fused. `a_s`, the term's adjoint, is not zero. Returns
    /// false, having changed nothing, when the separate nodes must run
    /// instead.
    ///
    /// The term's adjoints in closed form, from the sums of the forward
    /// pass: `mu` takes `a_s` times the sum over rows of each row's adjoint
    /// of its mean, `tau` the same weighted by each row's effect, each effect
    /// its group's rows times `a_s * tau` times its row adjoint and its
    /// transform's derivative, and `sigma` `a_s * (sum_sq - n) / sigma`. An adjoint that
    /// is zero is not added, as the separate passes skip a vector node whose
    /// adjoints are all zero, so a gradient later nodes left at -0.0 stays so.
    #[inline(never)]
    pub(super) fn grouped_reverse(
        &mut self,
        graph: &Graph,
        params: &[f64],
        c: usize,
        a_s: f64,
    ) -> bool {
        if !self.grouped.fused_this_call[c] {
            return false;
        }
        self.grouped.reverse_passes += 1;
        let chain = &self.grouped.chains[c];
        let sv = self.scalars[chain.sigma];
        if !scale_is_valid(sv) {
            // The term adds nothing, so every node before it is inactive.
            return true;
        }
        let sigma_adjoint = !(self.constant_at_construction[chain.sigma]
            && matches!(graph.nodes[chain.sigma].op, Op::Constant(_)));
        if sigma_adjoint {
            self.adj_scalars[chain.sigma] += a_s * ((chain.sum_sq - chain.len as f64) / sv);
        }
        if let Some((_, mu)) = chain.sum {
            if chain.mean_adjoint != 0.0 {
                self.adj_scalars[mu] += a_s * chain.mean_adjoint;
            }
        }
        let tau = match chain.product {
            Some((_, tau)) => {
                if chain.scale_adjoint != 0.0 {
                    self.adj_scalars[tau] += a_s * chain.scale_adjoint;
                }
                Some(self.scalars[tau])
            }
            None => None,
        };
        let grad = &mut self.grad[..];
        for (((&group, &row_adjoint), &value), &n) in chain
            .stats
            .groups
            .iter()
            .zip(&chain.row_adjoints)
            .zip(&chain.values)
            .zip(&chain.stats.counts)
        {
            // One row's adjoint of the gathered effect, as the separate
            // product passed it to the gather, then the group's rows.
            let mut adjoint = a_s * row_adjoint;
            if let Some(tau) = tau {
                adjoint *= tau;
            }
            if adjoint == 0.0 {
                continue;
            }
            let k = chain.param_start + group;
            let derivative = match graph.param_transforms[k] {
                // The value the forward pass took is this derivative.
                ParamTransform::Exp => value,
                ref transform => transform.derivative(params[k]),
            };
            grad[k] += n * (adjoint * derivative);
        }
        true
    }

    /// Row `i` of `node`'s vector, if `node` is the gather, product or sum of
    /// a chain the last forward pass ran fused, which did not write it:
    /// computed from the row's group effect with the separate nodes'
    /// expressions, so it has their bits.
    #[inline]
    pub(super) fn grouped_value(&self, node: usize, i: usize) -> Option<f64> {
        let c = self.grouped.members.get(node).copied().flatten()?;
        if !self.grouped.fused_this_call[c] {
            return None;
        }
        let chain = &self.grouped.chains[c];
        let group = self.binding.vectors[chain.indices][i] as usize;
        let mut value = chain.values[chain.stats.slot_of_group[group] as usize];
        if node == chain.gather {
            return Some(value);
        }
        if let Some((product, tau)) = chain.product {
            let tau = self.scalars[tau];
            value = if chain.scale_first {
                ElementwiseOp::Mul.value(tau, value)
            } else {
                ElementwiseOp::Mul.value(value, tau)
            };
            if node == product {
                return Some(value);
            }
        }
        let (_, mu) = chain.sum.expect("a member past the product is the sum");
        Some(self.scalars[mu] + value)
    }
}

#[cfg(test)]
mod tests {
    use super::super::reference::{eval_logp, grad_logp};
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Unit roundoff.
    const U: f64 = f64::EPSILON / 2.0;

    /// One grouped chain or two sharing their scalars, built the way the
    /// model compiler builds `mu + tau * z[group]` under a Normal likelihood.
    #[derive(Clone, Copy, Debug)]
    struct Shape {
        groups: usize,
        mean: bool,
        scale: bool,
        constant_sigma: bool,
        positive_z: bool,
        weighted: bool,
        second_term: bool,
        /// The product written `z[group] * tau`.
        scale_last: bool,
        /// The product's scale a constant created between the gather and the
        /// product, as the compiler builds `0.8 * z[group]`.
        constant_scale: bool,
        /// The product's scale able to be negative or zero: the raw
        /// parameter itself rather than its exponential, or -0.8 rather than
        /// 0.8 when it is a constant.
        signed_scale: bool,
    }

    impl Shape {
        fn from_flags(groups: usize, flags: u32) -> Self {
            let bit = |k: u32| flags & (1 << k) != 0;
            Shape {
                groups,
                mean: bit(0),
                scale: bit(1),
                constant_sigma: bit(2),
                positive_z: bit(3),
                weighted: bit(4),
                second_term: bit(5),
                scale_last: bit(6),
                constant_scale: bit(7),
                signed_scale: bit(8),
            }
        }

        fn plain(rows: usize, groups: usize) -> (Self, Vec<usize>) {
            (
                Shape::from_flags(groups, 0b11),
                default_groups(rows, groups),
            )
        }
    }

    /// The row pattern the earlier tests used: every group in turn, not
    /// sorted.
    fn default_groups(rows: usize, groups: usize) -> Vec<usize> {
        (0..rows).map(|i| (i * 7 + i / 3) % groups).collect()
    }

    struct Term {
        obs: usize,
        gather: usize,
        linpred: usize,
        group: Vec<usize>,
        y: Vec<f64>,
    }

    struct Built {
        graph: Graph,
        params: usize,
        terms: Vec<Term>,
        sigma: NodeId,
        /// The product's scale node, when there is a product.
        tau: Option<NodeId>,
        z_start: usize,
        sigma_param: Option<usize>,
        weight_param: Option<usize>,
    }

    fn build(shape: Shape, group: &[usize], y: impl Fn(usize) -> f64) -> Built {
        let rows = group.len();
        let mut g = Graph::new();
        let mu = g.add_param("mu");
        let zero = g.add_constant(0.0);
        let five = g.add_constant(5.0);
        let one = g.add_constant(1.0);
        g.normal_logp(mu, zero, five);
        let raw_tau = g.add_param("log_tau");
        let tau = if shape.signed_scale {
            raw_tau
        } else {
            g.exp(raw_tau)
        };
        g.log_half_normal_logp(raw_tau, one);
        let transform = if shape.positive_z {
            ParamTransform::Exp
        } else {
            ParamTransform::Identity
        };
        let z = g.add_vector_params_with_transform("z", shape.groups, transform);
        g.vector_normal_logp(z, shape.groups, 0.0, 1.0);
        let (sigma, sigma_param) = if shape.constant_sigma {
            (g.add_constant(0.7), None)
        } else {
            let raw = g.add_param("log_sigma");
            g.log_half_normal_logp(raw, one);
            (g.exp(raw), Some(g.param_count - 1))
        };
        let weight = shape.weighted.then(|| g.add_param("w"));
        let weight_param = weight.map(|_| g.param_count - 1);
        let group_node = g.add_data("group", group.iter().map(|&k| k as f64).collect());
        let mut terms = Vec::new();
        let mut scale_node = None;
        for term in 0..1 + shape.second_term as usize {
            let gathered = g.gather(z, shape.groups, group_node);
            let mut linpred = gathered;
            if shape.scale {
                let scale = match (shape.constant_scale, shape.signed_scale) {
                    (true, false) => g.add_constant(0.8),
                    (true, true) => g.add_constant(-0.8),
                    (false, _) => tau,
                };
                scale_node = Some(scale);
                linpred = if shape.scale_last {
                    g.elementwise(ElementwiseOp::Mul, linpred, Some(scale))
                } else {
                    g.elementwise(ElementwiseOp::Mul, scale, Some(linpred))
                };
            }
            if shape.mean {
                linpred = g.scalar_broadcast_add(mu, linpred);
            }
            let values: Vec<f64> = (0..rows).map(|i| y(i + 1000 * term)).collect();
            let obs = g.add_obs_data(values.clone());
            let node = g.obs_logp_normal(linpred, sigma, obs);
            if let Some(weight) = weight {
                let weighted = g.mul(node, weight);
                g.add_logp_term(weighted);
            }
            terms.push(Term {
                obs: node.0,
                gather: gathered.0,
                linpred: linpred.0,
                group: group.to_vec(),
                y: values,
            });
        }
        let params = g.param_count;
        Built {
            graph: g,
            params,
            terms,
            sigma,
            tau: scale_node,
            z_start: z,
            sigma_param,
            weight_param,
        }
    }

    /// Every node's value, and the log density and gradient, from `evaluator`
    /// after `compute` at `point`: the log density first, then the gradient,
    /// then each node in order.
    fn everything(evaluator: &mut Evaluator, g: &Graph, point: &[f64]) -> Vec<f64> {
        evaluator.compute(g, point);
        read_all(evaluator, g)
    }

    fn read_all(evaluator: &Evaluator, g: &Graph) -> Vec<f64> {
        let mut out = vec![evaluator.total_logp];
        out.extend_from_slice(&evaluator.grad);
        for node in &g.nodes {
            let len = evaluator.node_len(node.id);
            if len == 0 {
                out.push(evaluator.scalar_at(node.id));
            } else {
                out.extend((0..len).map(|i| evaluator.vec_elem(node.id, i)));
            }
        }
        out
    }

    fn same(a: f64, b: f64) -> bool {
        // Exact bits, NaN payloads aside.
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    fn assert_same(label: &str, fused: &[f64], other: &[f64]) {
        assert_eq!(fused.len(), other.len(), "{label}");
        for (k, (&a, &b)) in fused.iter().zip(other).enumerate() {
            assert!(
                same(a, b),
                "{label}: value {k}: fused {a:e} ({:#x}), separate {b:e} ({:#x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    /// Which entries of `everything` may differ between the fused and the
    /// separate passes, and by how much: the log density, the gradient, each
    /// grouped term and each node that reads one. Every other entry, the
    /// chain's vectors included, must have the same bits.
    ///
    /// The fused pass and the row loops compute the same sums in different
    /// orders. A recursive sum of `k` terms is within `(k - 1) u` times the
    /// sum of the terms' magnitudes of the exact one, and each term carries a
    /// few more roundings of its own; the row loops' sums have `N` terms and
    /// the fused pass's at most `G`, so the two are within
    /// `(N + G + 16) u M` of each other, where `M` is the sum of the
    /// magnitudes of the row summands. The tolerance is four times that.
    /// `M` adds to each row's standardized residual `4 u (|y| + |m|) /
    /// sigma`, for the mean kept as two doubles, whose error is of order
    /// `u^2` times the data.
    fn tolerances(built: &Built, shape: Shape, separate: &Evaluator, point: &[f64]) -> Vec<f64> {
        let g = &built.graph;
        let n_values = 1 + built.params + g.nodes.len() - 1
            + g.nodes
                .iter()
                .map(|node| separate.node_len(node.id).max(1))
                .sum::<usize>()
            - (g.nodes.len() - 1);
        let mut tol = vec![0.0f64; n_values];
        let mut logp_magnitude = 0.0f64;
        let mut grad_magnitude = vec![0.0f64; built.params];
        let rows: usize = built.terms.iter().map(|t| t.y.len()).sum();
        let factor = 4.0 * (rows + shape.groups + 16) as f64 * U;
        let sv = separate.scalar_at(built.sigma);
        let a_t = built.weight_param.map_or(1.0, |w| 1.0 + point[w]);
        let tau = built.tau.map_or(1.0, |t| separate.scalar_at(t));
        let mut term_tol = Vec::new();
        for term in &built.terms {
            if !scale_is_valid(sv) {
                term_tol.push(0.0);
                continue;
            }
            let residuals = &separate.observation_cache[term.obs];
            let mut q = 0.0f64;
            for (i, (&d, &y)) in residuals.iter().zip(&term.y).enumerate() {
                let m = separate.vec_elem(NodeId(term.linpred), i);
                let z = separate.vec_elem(NodeId(term.gather), i);
                let a = d.abs() + 4.0 * U * (y.abs() + m.abs()) / sv;
                q += a * a;
                let row = |grad: &mut [f64], k: usize, v: f64| grad[k] += a_t.abs() * v;
                if shape.mean {
                    row(&mut grad_magnitude, 0, a / sv);
                }
                if shape.scale && !shape.constant_scale {
                    let dtau = if shape.signed_scale { 1.0 } else { tau.abs() };
                    row(&mut grad_magnitude, 1, dtau * a * z.abs() / sv);
                }
                let k = built.z_start + term.group[i];
                let dz = if shape.positive_z {
                    point[k].exp()
                } else {
                    1.0
                };
                let scale = if shape.scale { tau.abs() } else { 1.0 };
                row(&mut grad_magnitude, k, scale * dz * a / sv);
                if let Some(s) = built.sigma_param {
                    // d sigma / d raw is sigma.
                    row(&mut grad_magnitude, s, a * a + 1.0);
                }
            }
            if let Some(w) = built.weight_param {
                grad_magnitude[w] += 0.5 * q;
            }
            logp_magnitude += (1.0 + a_t.abs()) * 0.5 * q;
            term_tol.push(factor * 0.5 * q);
        }
        // The log density and gradient also sum their nodes' contributions,
        // which round again around a contribution that changed.
        let terms = g.logp_terms.len() as f64;
        let logp_sum: f64 = g
            .logp_terms
            .iter()
            .map(|t| separate.scalar_at(*t).abs())
            .filter(|v| v.is_finite())
            .sum();
        tol[0] = factor * logp_magnitude + 4.0 * terms * U * logp_sum;
        for (k, m) in grad_magnitude.iter().enumerate() {
            tol[1 + k] = factor * m + 16.0 * U * separate.grad[k].abs();
        }
        // The terms' own values, and the weighted copies that read them.
        let mut offset = 1 + built.params;
        for node in &g.nodes {
            let len = separate.node_len(node.id).max(1);
            if let Some(t) = built.terms.iter().position(|t| t.obs == node.id.0) {
                tol[offset] = term_tol[t] + 4.0 * U * separate.scalar_at(node.id).abs();
            }
            if let Op::Mul(a, _) = node.op {
                if let Some(t) = built.terms.iter().position(|t| t.obs == a.0) {
                    tol[offset] = (term_tol[t] + 4.0 * U * separate.scalar_at(a).abs())
                        * a_t.abs().max(1.0)
                        + 4.0 * U * separate.scalar_at(node.id).abs();
                }
            }
            offset += len;
        }
        tol
    }

    /// Compare the fused values with the separate ones within `tol`; a pair
    /// that is not both finite must be both non-finite. Returns how many
    /// pairs were non-finite on one side as NaN and on the other as an
    /// infinity, which is a different overflow, not an error, and allowed
    /// only where the log density itself is not finite.
    fn assert_close(label: &str, fused: &[f64], separate: &[f64], tol: &[f64]) -> usize {
        assert_eq!(fused.len(), separate.len(), "{label}");
        let degenerate = !separate[0].is_finite();
        let mut kinds = 0;
        for (k, ((&a, &b), &t)) in fused.iter().zip(separate).zip(tol).enumerate() {
            if a.is_finite() && b.is_finite() {
                assert!(
                    (a - b).abs() <= t,
                    "{label}: value {k}: fused {a:e}, separate {b:e}, difference {:e} > {t:e}",
                    (a - b).abs()
                );
            } else {
                assert!(
                    !a.is_finite() && !b.is_finite(),
                    "{label}: value {k}: fused {a:e}, separate {b:e}"
                );
                if t == 0.0 {
                    assert!(
                        same(a, b),
                        "{label}: value {k}: fused {a:e}, separate {b:e}"
                    );
                } else if !same(a, b) {
                    assert!(
                        degenerate,
                        "{label}: value {k}: fused {a:e}, separate {b:e}"
                    );
                    kinds += 1;
                }
            }
        }
        kinds
    }

    /// The parameter points each shape is checked at: ordinary values, a
    /// zero product scale (the product's and gather's adjoints vanish), a
    /// scale near overflow, a tiny and a huge noise scale, and a positive
    /// group effect whose transform's derivative overflows.
    fn points(params: usize, shape: Shape) -> Vec<Vec<f64>> {
        let base = |k: usize| 0.37 * ((k * 5) % 7) as f64 - 0.9;
        let mut points = vec![(0..params).map(base).collect::<Vec<_>>()];
        let mut set = |edit: &dyn Fn(&mut Vec<f64>)| {
            let mut p = (0..params).map(base).collect::<Vec<_>>();
            edit(&mut p);
            points.push(p);
        };
        set(&|p| p[1] = -800.0);
        set(&|p| p[1] = 709.0);
        if !shape.constant_sigma {
            let sigma = 2 + shape.groups;
            for raw in [-700.0, -30.0, 30.0, 700.0] {
                set(&|p| p[sigma] = raw);
            }
        }
        if shape.positive_z {
            set(&|p| {
                p[1] = -800.0;
                p[2] = 710.0;
            });
        }
        if shape.signed_scale && !shape.constant_scale {
            // A negative, a zero and a negative-zero scale.
            for tau in [-1.3, 0.0, -0.0] {
                set(&|p| p[1] = tau);
            }
        }
        if shape.weighted {
            // The term's adjoint is 1 + w, since `obs_logp_normal` adds the
            // term to the log density as well: zero, negative, and 1 from a
            // negative zero.
            for w in [-1.0, -2.5, -0.0] {
                set(&|p| *p.last_mut().unwrap() = w);
            }
        }
        points
    }

    /// Whether the term's adjoint at `point` is zero, which only a weight of
    /// -1 gives; the reverse pass then skips the term, fused or not.
    fn inactive(shape: Shape, point: &[f64]) -> bool {
        shape.weighted && *point.last().unwrap() == -1.0
    }

    /// Check `built` at each point against the same chains run as their
    /// separate nodes, and check that every call ran fused. Returns the
    /// overflow pairs `assert_close` counted.
    fn check_shape(label: &str, built: &Built, shape: Shape, points: &[Vec<f64>]) -> usize {
        let g = &built.graph;
        let mut fused = Evaluator::new(g);
        let mut separate = Evaluator::new(g);
        separate.unfused_groups = true;
        let chains = built.terms.len() as u64;
        assert_eq!(fused.grouped.chains.len() as u64, chains, "{label}");
        let mut kinds = 0;
        for (k, point) in points.iter().enumerate() {
            let plan = &fused.grouped;
            let before = (plan.forward_passes, plan.reverse_passes, plan.fallbacks);
            let f = everything(&mut fused, g, point);
            let s = everything(&mut separate, g, point);
            let plan = &fused.grouped;
            let ran = plan.forward_passes - before.0;
            let fell_back = plan.fallbacks - before.2;
            // Every chain either ran fused or, outside the fused pass's
            // range, fell back; the first, ordinary point always runs fused.
            assert_eq!(ran + fell_back, chains, "{label} at {point:?}");
            assert!(k > 0 || ran == chains, "{label} at {point:?}");
            let reversed = if inactive(shape, point) { 0 } else { ran };
            assert_eq!(
                plan.reverse_passes - before.1,
                reversed,
                "{label} at {point:?}"
            );
            if fell_back > 0 {
                // The separate nodes ran: their bits exactly.
                assert_same(&format!("{label} at {point:?}, fallen back"), &f, &s);
            } else {
                let tol = tolerances(built, shape, &separate, point);
                kinds += assert_close(&format!("{label} at {point:?}"), &f, &s, &tol);
            }
        }
        assert_eq!(separate.grouped.forward_passes, 0, "{label}");
        kinds
    }

    /// The fused pass against the same chains run as their separate nodes:
    /// every shape of chain the compiler builds, at ordinary and extreme
    /// points, with more groups than rows among them. The log density, the
    /// gradient and the terms within the summation bound of `tolerances`,
    /// every other node value, the chains' vectors included, to the bit.
    #[test]
    fn grouped_normal_chains_match_their_separate_nodes() {
        let mut kinds = 0usize;
        for rows in [1usize, 2, 5, 37] {
            for groups in [1usize, 3, 10] {
                for flags in 0..512u32 {
                    let shape = Shape::from_flags(groups, flags);
                    let group = default_groups(rows, groups);
                    let built = build(shape, &group, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
                    let label = format!("{rows} rows, {groups} groups, flags {flags:09b}");
                    kinds += check_shape(&label, &built, shape, &points(built.params, shape));
                }
            }
        }
        // Overflowing points give NaN on one side and an infinity on the
        // other in some gradients, where the sums meet infinities in another
        // order; the log density there is -inf on both.
        eprintln!("{kinds} pairs non-finite in different ways");
    }

    /// The same comparison over random shapes: 1 to 3,000 rows, 1 to 300
    /// groups, indices sorted, unsorted, leaving groups empty or all in one
    /// group, data near zero or far from it at several scales, random
    /// points and scales near zero and huge.
    #[test]
    fn random_shapes_match_their_separate_nodes_within_the_summation_bound() {
        let mut rng = ChaCha8Rng::seed_from_u64(20261028);
        let log_uniform = |rng: &mut ChaCha8Rng, high: f64| -> usize {
            rng.gen_range(0.0..high.ln()).exp().floor() as usize
        };
        for case in 0..400 {
            let rows = log_uniform(&mut rng, 3001.0).max(1);
            let groups = log_uniform(&mut rng, 301.0).max(1);
            let pattern = rng.gen_range(0..5);
            let mut group: Vec<usize> = match pattern {
                0 | 1 => (0..rows).map(|_| rng.gen_range(0..groups)).collect(),
                // Only every third group has rows.
                2 => (0..rows)
                    .map(|_| (rng.gen_range(0..groups.div_ceil(3)) * 3).min(groups - 1))
                    .collect(),
                3 => vec![rng.gen_range(0..groups); rows],
                _ => (0..rows).map(|i| i % groups).collect(),
            };
            if pattern == 1 {
                group.sort_unstable();
            }
            let offset = [0.0, 1e3, -1e6, 1e8][rng.gen_range(0..4)];
            let spread = [1e-3, 1.0, 1e3][rng.gen_range(0..3)];
            let noise: Vec<f64> = (0..2 * rows + 2000)
                .map(|_| rng.gen_range(-1.0..1.0f64) * spread)
                .collect();
            let shape = Shape::from_flags(groups, rng.gen_range(0..512));
            let built = build(shape, &group, |i| offset + noise[i % noise.len()]);
            let mut points: Vec<Vec<f64>> = (0..3)
                .map(|_| {
                    (0..built.params)
                        .map(|_| rng.gen_range(-1.5..1.5))
                        .collect()
                })
                .collect();
            // A mean near the data, where the residuals cancel most.
            let mut near = points[0].clone();
            near[0] = offset;
            points.push(near);
            if let Some(s) = built.sigma_param {
                for raw in [-40.0, 40.0] {
                    let mut p = points[0].clone();
                    p[s] = raw;
                    points.push(p);
                }
            }
            if let Some(w) = built.weight_param {
                for weight in [-1.0, -2.5, 0.0] {
                    let mut p = points[0].clone();
                    p[w] = weight;
                    points.push(p);
                }
            }
            let label = format!("case {case}: {rows} rows, {groups} groups, pattern {pattern}, offset {offset}, {shape:?}");
            check_shape(&label, &built, shape, &points);
        }
    }

    /// Every way a recorded draw reads node values: a forward pass alone,
    /// then `vec_elem` and `scalar_at` on every node, as predictions,
    /// posterior and prior predictive draws, pointwise log-likelihoods and
    /// deterministics do. The chains' vectors, which the fused pass does not
    /// write, must read as the separate nodes wrote them, to the bit, after a
    /// forward pass alone, after a gradient, and after a forward pass that
    /// follows a gradient at another point.
    #[test]
    fn reads_after_a_forward_pass_match_the_separate_nodes() {
        for flags in 0..512u32 {
            for (rows, groups) in [(23usize, 4usize), (3, 9)] {
                let shape = Shape::from_flags(groups, flags);
                let group = default_groups(rows, groups);
                let built = build(shape, &group, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
                let g = &built.graph;
                let mut fused = Evaluator::new(g);
                let mut separate = Evaluator::new(g);
                separate.unfused_groups = true;
                let [a, b] = [0.3, -0.45].map(|shift| {
                    (0..built.params)
                        .map(|k| 0.29 * ((k * 3) % 5) as f64 - 0.6 + shift)
                        .collect::<Vec<_>>()
                });
                let vectors = |e: &Evaluator| {
                    // Every vector node's values: the reads that skip the
                    // log density's own sums.
                    let mut out = Vec::new();
                    for node in &g.nodes {
                        let len = e.node_len(node.id);
                        out.extend((0..len).map(|i| e.vec_elem(node.id, i)));
                    }
                    out
                };
                let label = format!("flags {flags:09b}, {rows} rows");
                fused.forward(g, &a);
                separate.forward(g, &a);
                assert_same(
                    &format!("{label}, forward"),
                    &vectors(&fused),
                    &vectors(&separate),
                );
                fused.compute(g, &b);
                separate.compute(g, &b);
                assert_same(
                    &format!("{label}, compute"),
                    &vectors(&fused),
                    &vectors(&separate),
                );
                let at_b = tolerances(&built, shape, &separate, &b);
                fused.forward(g, &a);
                separate.forward(g, &a);
                assert_same(
                    &format!("{label}, forward after compute"),
                    &vectors(&fused),
                    &vectors(&separate),
                );
                // Scalars too, against the separate nodes within the bound;
                // the gradients are still those of `b`.
                let mut tol = tolerances(&built, shape, &separate, &a);
                tol[1..1 + built.params].copy_from_slice(&at_b[1..1 + built.params]);
                assert_close(&label, &read_all(&fused, g), &read_all(&separate, g), &tol);
                assert_eq!(fused.grouped.forward_passes, 3 * built.terms.len() as u64);
            }
        }
    }

    /// The statistics against each group's rows found directly: the rows
    /// per group, the slots in group order with none for an empty group, the
    /// mean against a compensated sum, and the spread against the norm of
    /// the deviations from that mean.
    #[test]
    fn group_statistics_match_each_groups_rows() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for case in 0..300 {
            let rows = rng.gen_range(1..400);
            let groups = rng.gen_range(1..60);
            let used = rng.gen_range(1..=groups);
            let indices: Vec<f64> = (0..rows).map(|_| rng.gen_range(0..used) as f64).collect();
            let offset = [0.0, 1e4, -1e8][case % 3];
            let y: Vec<f64> = (0..rows)
                .map(|_| offset + rng.gen_range(-1.0..1.0))
                .collect();
            let mut stats = GroupStats::default();
            stats.build(&indices, &y, groups, rows);
            assert!(stats.usable, "case {case}");
            let mut slot = 0;
            for g in 0..groups {
                let members: Vec<f64> = (0..rows)
                    .filter(|&i| indices[i] as usize == g)
                    .map(|i| y[i])
                    .collect();
                if members.is_empty() {
                    assert_eq!(stats.slot_of_group[g], NO_SLOT, "case {case} group {g}");
                    continue;
                }
                assert_eq!(stats.slot_of_group[g], slot as u32, "case {case} group {g}");
                assert_eq!(stats.groups[slot], g);
                let n = members.len();
                assert_eq!(stats.counts[slot], n as f64, "case {case} group {g}");
                let (mut s, mut c) = (0.0f64, 0.0f64);
                for &v in &members {
                    let t = s + v;
                    c += if s.abs() >= v.abs() {
                        (s - t) + v
                    } else {
                        (v - t) + s
                    };
                    s = t;
                }
                let mean = (s + c) / n as f64;
                let scale = members.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
                let error = (stats.mean_hi[slot] - mean) + stats.mean_lo[slot];
                assert!(
                    error.abs() <= 4.0 * U * scale,
                    "case {case} group {g}: {error:e}"
                );
                let spread = members
                    .iter()
                    .map(|v| (v - mean) * (v - mean))
                    .sum::<f64>()
                    .sqrt();
                let largest = members.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let tol = 8.0 * (n + 4) as f64 * U * spread + 8.0 * (n as f64).sqrt() * U * largest;
                assert!(
                    (stats.spread[slot] - spread).abs() <= tol,
                    "case {case} group {g}: {} against {spread}",
                    stats.spread[slot]
                );
                slot += 1;
            }
            assert_eq!(stats.groups.len(), slot);
        }
        // An index outside the groups, and lengths that disagree, leave the
        // statistics unusable, so the chain runs as separate nodes.
        let mut stats = GroupStats::default();
        stats.build(&[0.0, 3.0], &[1.0, 2.0], 3, 2);
        assert!(!stats.usable);
        stats.build(&[0.0, 1.0], &[1.0], 3, 2);
        assert!(!stats.usable);
    }

    /// The group sums against the row sums, both against a reference taken
    /// in double-double arithmetic from the same rows and means: the sum of
    /// squared residuals and the sum of residuals, over random shapes with
    /// data near zero and far from it. With the scale 1, the separate
    /// nodes' sums are exactly `sum d_i^2` and `sum d_i` of the residuals
    /// they kept. The fused sums must be within `(G + 16) u` of the reference
    /// relative to the sum of magnitudes, and no worse than the row sums at
    /// their worst.
    #[test]
    fn group_sums_are_at_least_as_accurate_as_the_row_sums() {
        // Double-double helpers: `(hi, lo)` with `hi + lo` the value.
        fn add(a: (f64, f64), b: f64) -> (f64, f64) {
            let (s, e) = two_sum(a.0, b);
            two_sum(s, e + a.1)
        }
        fn square(h: f64, l: f64) -> (f64, f64) {
            let p = h * h;
            let e = h.mul_add(h, -p);
            two_sum(p, e + 2.0 * h * l)
        }
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let (mut worst_fused, mut worst_rows) = ([0.0f64; 2], [0.0f64; 2]);
        let (mut sum_fused, mut sum_rows) = ([0.0f64; 2], [0.0f64; 2]);
        let cases = 300;
        for case in 0..cases {
            let rows = rng.gen_range(1..3000);
            let groups = rng.gen_range(1..200);
            let group: Vec<usize> = (0..rows).map(|_| rng.gen_range(0..groups)).collect();
            let offset = [0.0, 1e4, -3e6, 1e8][case % 4];
            let spread = [1e-4, 1.0, 1e2][(case / 4) % 3];
            let y: Vec<f64> = (0..rows + 1000)
                .map(|_| offset + spread * rng.gen_range(-1.0..1.0))
                .collect();
            let mut shape = Shape::from_flags(groups, 0b11);
            shape.constant_sigma = false;
            let built = build(shape, &group, |i| y[i]);
            let g = &built.graph;
            let mut point: Vec<f64> = (0..built.params)
                .map(|_| rng.gen_range(-1.0..1.0) * spread.min(1.0))
                .collect();
            point[0] = offset;
            point[built.sigma_param.unwrap()] = 0.0;
            let mut fused = Evaluator::new(g);
            let mut separate = Evaluator::new(g);
            separate.unfused_groups = true;
            fused.compute(g, &point);
            separate.compute(g, &point);
            assert_eq!(separate.scalar_at(built.sigma), 1.0);
            let term = &built.terms[0];
            let (mut q_ref, mut r_ref, mut magnitude) = ((0.0, 0.0), (0.0, 0.0), 0.0f64);
            for (i, &yi) in term.y.iter().enumerate() {
                let m = separate.vec_elem(NodeId(term.linpred), i);
                assert_eq!(
                    m.to_bits(),
                    fused.vec_elem(NodeId(term.linpred), i).to_bits()
                );
                let (h, l) = two_sum(yi, -m);
                let sq = square(h, l);
                q_ref = add(add(q_ref, sq.0), sq.1);
                r_ref = add(add(r_ref, h), l);
                magnitude += h.abs();
            }
            let residuals = &separate.observation_cache[term.obs];
            let q_rows = residuals.iter().fold(0.0, |s, d| s + d * d);
            let r_rows = residuals.iter().fold(0.0, |s, d| s + d);
            let chain = &fused.grouped.chains[0];
            let q = q_ref.0 + q_ref.1;
            let r = r_ref.0 + r_ref.1;
            let errors_fused = [
                (chain.sum_sq - q).abs() / q,
                (chain.mean_adjoint - r).abs() / magnitude,
            ];
            let errors_rows = [(q_rows - q).abs() / q, (r_rows - r).abs() / magnitude];
            let bound = (chain.stats.groups.len() + 16) as f64 * U;
            for k in 0..2 {
                assert!(
                    errors_fused[k] <= bound,
                    "case {case}, sum {k}: {:e} > {bound:e}",
                    errors_fused[k]
                );
                worst_fused[k] = worst_fused[k].max(errors_fused[k]);
                worst_rows[k] = worst_rows[k].max(errors_rows[k]);
                sum_fused[k] += errors_fused[k];
                sum_rows[k] += errors_rows[k];
            }
        }
        eprintln!(
            "relative errors, sum of squares and sum: worst fused {worst_fused:?}, rows \
             {worst_rows:?}; mean fused {:?}, rows {:?}",
            sum_fused.map(|s| s / cases as f64),
            sum_rows.map(|s| s / cases as f64)
        );
        for k in 0..2 {
            assert!(worst_fused[k] <= worst_rows[k], "{k}");
            assert!(sum_fused[k] <= sum_rows[k], "{k}");
        }
    }

    /// A grouped model whose responses equal its mean exactly at the checked
    /// point, so every residual is zero: the fused pass must find each
    /// group's row adjoint exactly zero, add no adjoint to the effects, and
    /// agree with the separate nodes.
    #[test]
    fn zero_residuals_give_exactly_zero_row_adjoints() {
        let mut shape = Shape::from_flags(3, 0b1011);
        shape.constant_sigma = false;
        let group = default_groups(9, 3);
        let point = [0.25, -0.5, 0.1, 0.9, -0.3, 0.2];
        let tau = (-0.5f64).exp();
        let z = [0.1f64.exp(), 0.9f64.exp(), (-0.3f64).exp()];
        let built = build(shape, &group, |i| 0.25 + tau * z[group[i % 9]]);
        let g = &built.graph;
        let mut fused = Evaluator::new(g);
        let mut separate = Evaluator::new(g);
        separate.unfused_groups = true;
        let f = everything(&mut fused, g, &point);
        let s = everything(&mut separate, g, &point);
        assert!(separate.observation_cache[built.terms[0].obs]
            .iter()
            .all(|&d| d == 0.0));
        let chain = &fused.grouped.chains[0];
        assert!(chain.row_adjoints.iter().all(|&g| g == 0.0));
        assert_eq!(chain.mean_adjoint, 0.0);
        assert_eq!(chain.scale_adjoint, 0.0);
        assert_eq!(chain.sum_sq, 0.0);
        assert_close(
            "zero residuals",
            &f,
            &s,
            &tolerances(&built, shape, &separate, &point),
        );
        // Only the prior's gradient reaches the effects.
        for k in 2..5 {
            assert_eq!(fused.grad[k], separate.grad[k], "effect {k}");
        }
        assert_eq!(fused.grouped.reverse_passes, 1);
    }

    /// The fused gradient against the reference evaluator, which knows
    /// nothing of fusion, and against central differences of its log density.
    #[test]
    fn grouped_normal_chains_match_the_reference_gradients() {
        for flags in 0..512u32 {
            let shape = Shape::from_flags(4, flags);
            let group = default_groups(23, 4);
            let built = build(shape, &group, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
            let g = &built.graph;
            let base = (0..built.params)
                .map(|k| 0.31 * ((k * 3) % 5) as f64 - 0.6)
                .collect::<Vec<_>>();
            let mut points = vec![base.clone()];
            if shape.weighted {
                // A negative weight on the term, 1 + w = -1.5.
                let mut negative = base;
                *negative.last_mut().unwrap() = -2.5;
                points.push(negative);
            }
            if shape.scale && shape.signed_scale && !shape.constant_scale {
                // A negative product scale.
                let mut negative = points[0].clone();
                negative[1] = -1.3;
                points.push(negative);
            }
            let mut fused = Evaluator::new(g);
            for point in &points {
                let before = fused.grouped.reverse_passes;
                fused.compute(g, point);
                assert_eq!(
                    fused.grouped.reverse_passes,
                    before + 1 + shape.second_term as u64
                );
                let (logp, grad) = grad_logp(g, point);
                let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * b.abs().max(1.0);
                assert!(close(fused.total_logp, logp, 1e-12), "{flags:09b}");
                for (k, (&a, &b)) in fused.grad.iter().zip(&grad).enumerate() {
                    assert!(close(a, b, 1e-10), "{flags:09b} param {k}: {a} against {b}");
                    let eps = 1e-6;
                    let mut up = point.clone();
                    let mut down = point.clone();
                    up[k] += eps;
                    down[k] -= eps;
                    let numeric = (eval_logp(g, &up) - eval_logp(g, &down)) / (2.0 * eps);
                    assert!(
                        close(a, numeric, 1e-6),
                        "{flags:09b} param {k}: {a} against {numeric}"
                    );
                }
            }
        }
    }

    /// Chains the fused pass must not take, each checked against the
    /// separate nodes: a gather that also feeds a second term, a node between
    /// the gather and its product, the mean also the product's scale, and a
    /// Bernoulli likelihood.
    #[test]
    fn partial_and_aliased_chains_run_as_separate_nodes() {
        let rows = 11;
        let indices = (0..rows).map(|i| (i % 3) as f64).collect::<Vec<_>>();
        let y = (0..rows).map(|i| i as f64 / 5.0 - 1.0).collect::<Vec<_>>();
        let mut graphs = Vec::new();
        let base = |g: &mut Graph| {
            let mu = g.add_param("mu");
            let raw = g.add_param("log_tau");
            let tau = g.exp(raw);
            let z = g.add_vector_params("z", 3);
            let sigma = g.add_constant(0.8);
            let group = g.add_data("group", indices.clone());
            (mu, tau, z, sigma, group)
        };
        {
            let mut g = Graph::new();
            let (mu, tau, z, sigma, group) = base(&mut g);
            let gathered = g.gather(z, 3, group);
            let product = g.elementwise(ElementwiseOp::Mul, tau, Some(gathered));
            let mean = g.scalar_broadcast_add(mu, product);
            let obs = g.add_obs_data(y.clone());
            g.obs_logp_normal(mean, sigma, obs);
            let total = g.sum(gathered);
            g.add_logp_term(total);
            graphs.push(("gather read twice", g));
        }
        {
            let mut g = Graph::new();
            let (mu, tau, z, sigma, group) = base(&mut g);
            let gathered = g.gather(z, 3, group);
            let between = g.exp(mu);
            let product = g.elementwise(ElementwiseOp::Mul, between, Some(gathered));
            let mean = g.scalar_broadcast_add(mu, product);
            let obs = g.add_obs_data(y.clone());
            g.obs_logp_normal(mean, sigma, obs);
            let _ = tau;
            graphs.push(("node between", g));
        }
        {
            let mut g = Graph::new();
            let (mu, _, z, sigma, group) = base(&mut g);
            let gathered = g.gather(z, 3, group);
            let product = g.elementwise(ElementwiseOp::Mul, mu, Some(gathered));
            let mean = g.scalar_broadcast_add(mu, product);
            let obs = g.add_obs_data(y.clone());
            g.obs_logp_normal(mean, sigma, obs);
            graphs.push(("mean is the scale", g));
        }
        {
            let mut g = Graph::new();
            let (mu, tau, z, _, group) = base(&mut g);
            let gathered = g.gather(z, 3, group);
            let product = g.elementwise(ElementwiseOp::Mul, tau, Some(gathered));
            let mean = g.scalar_broadcast_add(mu, product);
            let obs = g.add_obs_data((0..rows).map(|i| (i % 2) as f64).collect());
            g.obs_logp_bernoulli_logit(mean, obs);
            graphs.push(("Bernoulli", g));
        }
        {
            // The term's scale computed from the mean between the sum and the
            // term, and the mean read again after the chain: three
            // contributions to the mean's adjoint whose order a fused chain
            // would change.
            let mut g = Graph::new();
            let (mu, tau, z, _, group) = base(&mut g);
            let gathered = g.gather(z, 3, group);
            let product = g.elementwise(ElementwiseOp::Mul, tau, Some(gathered));
            let mean = g.scalar_broadcast_add(mu, product);
            let sigma = g.exp(mu);
            let obs = g.add_obs_data(y.clone());
            g.obs_logp_normal(mean, sigma, obs);
            let zero = g.add_constant(0.0);
            let two = g.add_constant(2.0);
            g.normal_logp(mu, zero, two);
            graphs.push(("scale computed between", g));
        }
        for (label, g) in &graphs {
            assert_eq!(g.param_count, 5, "{label}");
            let mut evaluator = Evaluator::new(g);
            let mut separate = Evaluator::new(g);
            separate.unfused_groups = true;
            // A grid wide enough that sums taken in another order round
            // differently at some of its points.
            for a in 0..40 {
                for b in 0..5 {
                    let point = [
                        -1.5 + 0.0731 * a as f64,
                        0.3 * b as f64 - 0.6,
                        0.4,
                        -0.2,
                        0.9,
                    ];
                    assert_same(
                        &format!("{label} at {point:?}"),
                        &everything(&mut evaluator, g, &point),
                        &everything(&mut separate, g, &point),
                    );
                }
            }
            assert_eq!(evaluator.grouped.forward_passes, 0, "{label}");
        }
    }

    /// A graph changed after the evaluator found its chain: the term turned
    /// LogNormal, and a later term made to read the gather. The evaluator
    /// must notice and agree with one built for the changed graph.
    #[test]
    fn a_chain_changed_after_construction_runs_as_separate_nodes() {
        let (shape, group) = Shape::plain(13, 3);
        let built = build(shape, &group, |i| 0.3 + (i % 5) as f64 / 4.0);
        let point = (0..built.params)
            .map(|k| 0.2 * k as f64 - 0.3)
            .collect::<Vec<_>>();
        let obs = built.terms[0].obs;

        let mut g = built.graph.clone();
        let mut evaluator = Evaluator::new(&g);
        evaluator.compute(&g, &point);
        assert_eq!(evaluator.grouped.forward_passes, 1);
        if let Op::ObsLogP { family, .. } = &mut g.nodes[obs].op {
            *family = ObsFamily::LogNormal;
        }
        let changed = everything(&mut evaluator, &g, &point);
        assert_same(
            "LogNormal",
            &changed,
            &everything(&mut Evaluator::new(&g), &g, &point),
        );
        assert_eq!(evaluator.grouped.forward_passes, 1);

        let mut g = built.graph.clone();
        let spare = g.add_data("spare", vec![0.0; group.len()]);
        let total = g.sum(spare);
        g.add_logp_term(total);
        let mut evaluator = Evaluator::new(&g);
        evaluator.compute(&g, &point);
        assert_eq!(evaluator.grouped.forward_passes, 1);
        let gather = evaluator.grouped.chains[0].gather;
        g.nodes[total.0].op = Op::Sum(NodeId(gather));
        let changed = everything(&mut evaluator, &g, &point);
        let mut fresh = Evaluator::new(&g);
        assert_same(
            "second reader",
            &changed,
            &everything(&mut fresh, &g, &point),
        );
        assert_eq!(evaluator.grouped.forward_passes, 1);
        assert_eq!(fresh.grouped.forward_passes, 0);
    }

    /// Values from two evaluators of the same graph, one or both of which may
    /// have run a chain fused: the chains' vectors and every node outside
    /// them to the bit when both ran the same way, and the log density,
    /// gradient and terms within `1e-12` of the other's magnitude, far below
    /// any change to the graph, for a graph of a dozen rows of order-one
    /// values.
    fn assert_nearly(label: &str, a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "{label}");
        for (k, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                same(x, y) || (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                "{label}: value {k}: {x:e} against {y:e}"
            );
        }
    }

    /// Every field of the graph the fused pass reads, changed one at a time
    /// after the evaluator found its chain: the gather's indices and span,
    /// the product's scale, operand order and operator, the mean, the term's
    /// scale, observations and family, and the effects' transform. Each time
    /// the evaluator must agree with one built for the changed graph. A
    /// changed transform is read on every call, so that chain still runs
    /// fused; every other change must send it to the separate nodes.
    #[test]
    fn every_field_the_fused_pass_reads_changed_after_construction() {
        let rows = 12;
        let mut g = Graph::new();
        let mu = g.add_param("mu");
        let other_mu = g.add_param("other_mu");
        let raw_tau = g.add_param("log_tau");
        let tau = g.exp(raw_tau);
        let other_tau = g.add_constant(1.7);
        let z = g.add_vector_params("z", 3);
        let other_z = g.add_vector_params("w", 3);
        let raw_sigma = g.add_param("log_sigma");
        let sigma = g.exp(raw_sigma);
        let other_sigma = g.add_constant(0.4);
        let group = g.add_data("group", (0..rows).map(|i| (i % 3) as f64).collect());
        let other_group = g.add_data(
            "other",
            (0..rows).map(|i| ((i * 2 + 1) % 3) as f64).collect(),
        );
        let gathered = g.gather(z, 3, group);
        let product = g.elementwise(ElementwiseOp::Mul, tau, Some(gathered));
        let mean = g.scalar_broadcast_add(mu, product);
        let obs = g.add_obs_data((0..rows).map(|i| 0.5 + i as f64 / 7.0).collect());
        let other_obs = g.add_obs_data((0..rows).map(|i| 2.0 - i as f64 / 9.0).collect());
        let term = g.obs_logp_normal(mean, sigma, obs);
        let point = (0..g.param_count)
            .map(|k| 0.27 * k as f64 - 0.8)
            .collect::<Vec<_>>();

        type Change = Box<dyn Fn(&mut Graph)>;
        let changes: Vec<(&str, bool, Change)> = vec![
            (
                "indices",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[gathered.0].op = Op::Gather {
                        param_start: z,
                        n_params: 3,
                        indices: other_group,
                    };
                }),
            ),
            (
                "span",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[gathered.0].op = Op::Gather {
                        param_start: other_z,
                        n_params: 3,
                        indices: group,
                    };
                }),
            ),
            (
                "product scale",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[product.0].op = Op::Elementwise {
                        operator: ElementwiseOp::Mul,
                        a: other_tau,
                        b: Some(gathered),
                    };
                }),
            ),
            (
                "operand order",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[product.0].op = Op::Elementwise {
                        operator: ElementwiseOp::Mul,
                        a: gathered,
                        b: Some(tau),
                    };
                }),
            ),
            (
                "operator",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[product.0].op = Op::Elementwise {
                        operator: ElementwiseOp::Add,
                        a: tau,
                        b: Some(gathered),
                    };
                }),
            ),
            (
                "mean",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[mean.0].op = Op::ScalarBroadcastAdd(other_mu, product);
                }),
            ),
            (
                "term scale",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[term.0].op = Op::ObsLogP {
                        family: ObsFamily::Normal,
                        linpred_vec: mean,
                        aux: Some(other_sigma),
                        obs_data_idx: obs,
                    };
                }),
            ),
            (
                "observations",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[term.0].op = Op::ObsLogP {
                        family: ObsFamily::Normal,
                        linpred_vec: mean,
                        aux: Some(sigma),
                        obs_data_idx: other_obs,
                    };
                }),
            ),
            (
                "family",
                false,
                Box::new(move |g: &mut Graph| {
                    g.nodes[term.0].op = Op::ObsLogP {
                        family: ObsFamily::LogNormal,
                        linpred_vec: mean,
                        aux: Some(sigma),
                        obs_data_idx: obs,
                    };
                }),
            ),
            (
                "transform",
                true,
                Box::new(move |g: &mut Graph| {
                    for k in z..z + 3 {
                        g.param_transforms[k] = ParamTransform::Exp;
                    }
                }),
            ),
        ];
        for (label, still_fused, change) in &changes {
            let mut changed = g.clone();
            let mut evaluator = Evaluator::new(&changed);
            evaluator.compute(&changed, &point);
            assert_eq!(evaluator.grouped.forward_passes, 1, "{label}");
            change(&mut changed);
            let stale = everything(&mut evaluator, &changed, &point);
            let mut fresh = Evaluator::new(&changed);
            assert_nearly(label, &stale, &everything(&mut fresh, &changed, &point));
            let mut separate = Evaluator::new(&changed);
            separate.unfused_groups = true;
            assert_nearly(label, &stale, &everything(&mut separate, &changed, &point));
            let expected = if *still_fused { 2 } else { 1 };
            assert_eq!(evaluator.grouped.forward_passes, expected, "{label}");
        }
    }

    /// Inputs at the edges of the exponent range. Each of these a fused pass
    /// would have got wrong where the rows did not: per-row adjoints near
    /// overflow that the rows cancel in order, an effect gradient whose row
    /// adjoint is subnormal before a huge product scale multiplies it, and
    /// observations too close to zero or too large for the group statistics.
    /// Each must run the separate nodes and give their bits. A point just
    /// inside the fused pass's range must run fused and agree with them.
    #[test]
    fn exponent_range_edges_run_the_separate_nodes() {
        assert_eq!(SAFE_MAGNITUDE, 2f64.powi(128));
        assert_eq!(TINY_OBSERVATION, 2f64.powi(-960));
        let run = |g: &Graph, point: &[f64]| {
            let mut fused = Evaluator::new(g);
            let mut separate = Evaluator::new(g);
            separate.unfused_groups = true;
            let f = everything(&mut fused, g, point);
            let s = everything(&mut separate, g, point);
            assert!(s.iter().all(|v| v.is_finite()), "{s:?}");
            (f, s, fused.grouped.forward_passes, fused.grouped.fallbacks)
        };
        let separate_only = |label: &str, g: &Graph, point: &[f64], fallbacks: u64| {
            let (f, s, passes, fell_back) = run(g, point);
            assert_same(label, &f, &s);
            assert_eq!((passes, fell_back), (0, fallbacks), "{label}");
        };
        // `mu + scale * z[group]` with a constant scale and noise scale, one
        // or two groups, rows `y`.
        let model = |groups: usize, group: Vec<f64>, y: Vec<f64>, scale: f64, sigma: f64| {
            let mut g = Graph::new();
            let mu = g.add_param("mu");
            let z = g.add_vector_params("z", groups);
            let group = g.add_data("group", group);
            let gathered = g.gather(z, groups, group);
            let scale = g.add_constant(scale);
            let product = g.elementwise(ElementwiseOp::Mul, scale, Some(gathered));
            let mean = g.scalar_broadcast_add(mu, product);
            let sigma = g.add_constant(sigma);
            let obs = g.add_obs_data(y);
            g.obs_logp_normal(mean, sigma, obs);
            g
        };
        // Rows' adjoints of the mean of about +-1e308, which the separate
        // nodes add in row order and cancel, and which a group's row count
        // would multiply past the largest double.
        let alternating = model(
            2,
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1e-280, -1e-280, 1e-280, -1e-280],
            1e-294,
            1e-294,
        );
        separate_only("adjoints near overflow", &alternating, &[0.0, 0.0, 0.0], 1);
        // A row adjoint of the effect in the subnormal range, multiplied by
        // a product scale of 1e308.
        for y in [[0.03, 0.0], [0.05, 0.0]] {
            let subnormal = model(1, vec![0.0; 2], y.to_vec(), 1e308, 1e161);
            separate_only("subnormal adjoint", &subnormal, &[0.0, 0.0], 1);
        }
        // An effect, a mean or a noise scale outside 2^128.
        let ordinary = model(1, vec![0.0; 3], vec![0.5, -0.2, 0.1], 0.8, 0.7);
        separate_only("huge effect", &ordinary, &[0.0, 1e39], 1);
        separate_only("huge mean", &ordinary, &[-1e39, 0.0], 1);
        let tiny_sigma = model(1, vec![0.0; 3], vec![0.5, -0.2, 0.1], 0.8, 1e-39);
        separate_only("tiny noise scale", &tiny_sigma, &[0.0, 0.0], 1);
        // Observations next to zero, or beyond 2^128: the statistics are not
        // usable, so the chain never runs fused.
        let subnormal_data = model(1, vec![0.0; 2], vec![0.0, f64::from_bits(1)], 1.0, 1e-162);
        separate_only("subnormal observations", &subnormal_data, &[0.0, 0.0], 0);
        let huge_data = model(1, vec![0.0; 2], vec![1e39, 0.0], 1.0, 1e39);
        separate_only("huge observations", &huge_data, &[0.0, 0.0], 0);
        // Just inside the range: a noise scale of 1e-30 and effects of 1e30.
        // Group 1's residuals cancel, so its effect's gradient is zero up to
        // the rounding of residuals of 2.5e30 divided twice by the scale:
        // within the summation bound, `64 u` times the rows' magnitudes,
        // `sum (|y| + |m|) / sigma^2`, about 1e91.
        let inside = model(2, vec![0.0, 1.0, 1.0], vec![1e30, -2e30, 3e30], 0.5, 1e-30);
        let (f, s, passes, fell_back) = run(&inside, &[1e30, 2e30, -1e30]);
        let bound = 64.0 * U * (6e30 + 3e30) / (1e-30 * 1e-30);
        for k in 1..4 {
            assert!(
                (f[k] - s[k]).abs() <= bound,
                "gradient {k}: {} against {}",
                f[k],
                s[k]
            );
        }
        let mut rest = (f.clone(), s.clone());
        rest.0.drain(1..4);
        rest.1.drain(1..4);
        assert_nearly("inside the range", &rest.0, &rest.1);
        assert_eq!((passes, fell_back), (1, 0));
    }

    /// A gather whose adjoints are all zero must leave its gradients exactly
    /// as later nodes left them. A matrix product's reverse pass, with an
    /// entry so small that `-0.5 * entry` underflows in a fused multiply-add,
    /// leaves the effect's gradient at -0.0; adding the inactive gather's
    /// zeros would make it +0.0. The separate nodes skip the gather, and so
    /// must the fused pass.
    #[test]
    fn an_inactive_gather_keeps_a_negative_zero_gradient() {
        let mut g = Graph::new();
        let one = g.add_constant(1.0);
        let z = g.add_vector_params("z", 1);
        let group = g.add_data("group", vec![0.0]);
        let gathered = g.gather(z, 1, group);
        let obs = g.add_obs_data(vec![0.0]);
        g.obs_logp_normal(gathered, one, obs);
        let matrix = g.store_matrix(vec![f64::from_bits(1)], 1, 1);
        let eta = g.mat_vec_mul(matrix, z, 1, None);
        let other_obs = g.add_obs_data(vec![-0.5]);
        g.obs_logp_normal(eta, one, other_obs);
        let mut fused = Evaluator::new(&g);
        let mut separate = Evaluator::new(&g);
        separate.unfused_groups = true;
        let point = [0.0];
        assert_same(
            "inactive gather",
            &everything(&mut fused, &g, &point),
            &everything(&mut separate, &g, &point),
        );
        assert_eq!(fused.grouped.reverse_passes, 1);
        if super::super::has_fused_multiply_add() {
            // The case the test is for: the gradient is a negative zero.
            assert_eq!(fused.grad[0].to_bits(), (-0.0f64).to_bits());
        }
    }

    /// Groups whose effects mix transforms, which the separate gather runs
    /// row by row: each group that has rows is transformed once with its own
    /// transform, so the chain runs fused and agrees with the separate nodes.
    #[test]
    fn mixed_transforms_run_fused() {
        let (shape, group) = Shape::plain(20, 4);
        let built = build(shape, &group, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
        let mut g = built.graph;
        let point = (0..built.params)
            .map(|k| 0.31 * ((k * 3) % 5) as f64 - 0.6)
            .collect::<Vec<_>>();
        // The first effect, parameter 2, positive; the other three not.
        g.param_transforms[2] = ParamTransform::Exp;
        let mut mixed = Evaluator::new(&g);
        let mut separate = Evaluator::new(&g);
        separate.unfused_groups = true;
        assert_nearly(
            "mixed transforms",
            &everything(&mut mixed, &g, &point),
            &everything(&mut separate, &g, &point),
        );
        assert_eq!(mixed.grouped.forward_passes, 1);
    }

    /// Rebinding data recomputes every chain's statistics: an evaluator
    /// built for one dataset and rebound to another, of the same shape or
    /// not, gives the same bits as one built for the second, and agrees with
    /// the separate nodes on it.
    #[test]
    fn rebinding_recomputes_the_statistics() {
        let (shape, first_groups) = Shape::plain(30, 4);
        let first = build(shape, &first_groups, |i| (i % 7) as f64 / 3.0);
        let point = (0..first.params)
            .map(|k| 0.21 * k as f64 - 0.4)
            .collect::<Vec<_>>();
        let second_groups: Vec<usize> = (0..30).map(|i| (i * i + 1) % 4).collect();
        let second = build(shape, &second_groups, |i| 4.0 - (i % 5) as f64 / 2.0);
        let third_groups: Vec<usize> = (0..17).map(|i| (i * 3) % 4).collect();
        let third = build(shape, &third_groups, |i| (i % 3) as f64 - 1.0);
        for target in [&second, &third] {
            let mut rebound = Evaluator::new(&first.graph);
            rebound.compute(&first.graph, &point);
            let binding = DataBinding::from_graph(&target.graph).unwrap();
            rebound.rebind(&first.graph, binding).unwrap();
            let g = &target.graph;
            let rows = target.terms[0].y.len();
            let label = format!("rebound to {rows} rows");
            let mut fresh = Evaluator::new(g);
            assert_same(
                &label,
                &everything(&mut rebound, g, &point),
                &everything(&mut fresh, g, &point),
            );
            let mut separate = Evaluator::new(g);
            separate.unfused_groups = true;
            let s = everything(&mut separate, g, &point);
            let tol = tolerances(target, shape, &separate, &point);
            assert_close(&label, &everything(&mut rebound, g, &point), &s, &tol);
            assert!(rebound.grouped.forward_passes >= 2, "{label}");
        }
    }

    /// A seeded NUTS fit of a grouped model runs the fused pass, gives the
    /// same draws when run again, and the same draws on one thread as on
    /// four: the process-wide count grows by at least one pass per leapfrog
    /// step.
    #[test]
    fn a_seeded_fit_runs_the_fused_pass_and_repeats_on_any_thread_count() {
        let (shape, group) = Shape::plain(40, 5);
        let mut shape = shape;
        shape.mean = true;
        shape.constant_sigma = false;
        let built = build(shape, &group, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
        let fit = |threads: usize| {
            let config = crate::sampler::SamplerConfig {
                num_chains: 4,
                num_draws: 50,
                num_warmup: 50,
                seed: 7,
                num_threads: threads,
                show_progress: false,
                ..Default::default()
            };
            crate::sampler::sample(built.graph.clone(), config).expect("the fit runs")
        };
        let before = fused_grouped_normal_passes();
        let result = fit(1);
        let steps: usize = result
            .transitions
            .iter()
            .flatten()
            .map(|t| t.num_leapfrog_steps)
            .sum();
        let after = fused_grouped_normal_passes();
        assert!(
            after.1 - before.1 >= steps as u64,
            "{before:?} {after:?} {steps}"
        );
        let bits = |r: &crate::sampler::SampleResult| {
            r.samples
                .iter()
                .flatten()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&result), bits(&fit(1)), "a second run");
        assert_eq!(bits(&result), bits(&fit(4)), "four threads");
    }
}
