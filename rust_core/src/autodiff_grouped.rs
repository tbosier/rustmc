//! One forward and one reverse loop for a Normal observation term on a
//! grouped linear predictor, `y ~ Normal(mu + tau * z[group], sigma)`, in
//! place of the four row-length passes of its gather, product, broadcast sum
//! and likelihood nodes.
//!
//! The fused loops compute every value the separate nodes computed, with the
//! same expressions and the same order of every sum, and they still write the
//! nodes' vectors, which predictions and deterministics read. So log
//! densities, gradients, those vectors and seeded draws keep their bits. What
//! the fusion saves is the round trip of each row through three vectors in
//! each pass, and the evaluator's dispatch between them.
//!
//! A chain is fused only when nothing else can observe the difference: the
//! only nodes between its own are constants and data, which take no part in
//! the reverse pass, each vector in it feeds only the next node, the group
//! indices are data, and the scale of the product is not also the chain's
//! mean or scale, whose adjoints the separate nodes would have added to it
//! first. The chain is decided at its gather and runs at its term, once any
//! constant between has its value. Every call checks that the graph still
//! holds the chain the evaluator found, and runs the separate nodes otherwise.

use super::{scale_is_valid, Evaluator, NodeKind};
use crate::data::DataBinding;
use crate::graph::{ElementwiseOp, Graph, NodeId, ObsFamily, Op};
use std::sync::atomic::{AtomicU64, Ordering};

/// Fused forward and reverse passes run by every evaluator dropped so far,
/// in this process. Each evaluator counts its own and adds them here when it
/// is dropped, so sampling threads don't share a counter per gradient.
static FUSED_FORWARD_PASSES: AtomicU64 = AtomicU64::new(0);
static FUSED_REVERSE_PASSES: AtomicU64 = AtomicU64::new(0);

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

/// A fused chain, by node index: `gather`, then optionally `product =
/// tau * gather` or `gather * tau`, then optionally `sum = mu + (product or
/// gather)`, then the Normal term `obs` on the last of them. Any other node
/// between them is a constant or data.
#[derive(Debug, Clone)]
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
    /// evaluator was built, which the fused loops read.
    indices_node: usize,
    indices: usize,
    obs_data_idx: usize,
    len: usize,
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
    /// Per chain, whether this call's forward pass found the chain still in
    /// the graph and ran it fused, so the reverse pass may too.
    fused_this_call: Vec<bool>,
    /// Every node's readers in the graph of the current call, counted in one
    /// pass when the first chain of the call needs them, so a call costs one
    /// scan of the graph however many chains it checks, and none when every
    /// chain fails a cheaper check first. Empty for a plan with no chains.
    readers: Vec<u32>,
    readers_counted: bool,
    forward_passes: u64,
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
    /// fused loops reproduce exactly. `consumers` counts each node's readers.
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
        // The product's scale takes its adjoint after the term's scale and
        // the chain's mean took theirs, so it must not be either.
        let tau_ok =
            product.is_none_or(|(_, tau)| tau != sigma.0 && sum.is_none_or(|(_, mu)| tau != mu));
        let found = Self {
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
        };
        (chain_ok && tau_ok && found.in_order_with_only_constants_between(graph)).then_some(found)
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
    /// add nothing, so running the chain at its term reorders no sum.
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
            forward_passes: 0,
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

    /// Forget the reader counts of the previous call; the graph may have
    /// changed since.
    pub(super) fn begin_call(&mut self) {
        self.readers_counted = false;
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

/// Split `buf` into the `len` rows at each offset, in increasing order, as
/// disjoint slices; an absent offset gives an empty slice.
fn disjoint_rows(buf: &mut [f64], offsets: [Option<usize>; 3], len: usize) -> [&mut [f64]; 3] {
    let mut out: [&mut [f64]; 3] = [&mut [], &mut [], &mut []];
    let mut rest = buf;
    let mut base = 0;
    for (slot, offset) in out.iter_mut().zip(offsets) {
        if let Some(offset) = offset {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(offset - base);
            let (rows, tail) = tail.split_at_mut(len);
            *slot = rows;
            rest = tail;
            base = offset + len;
        }
    }
    out
}

/// The forward rows of a fused chain, as slices the loop cannot alias.
struct ForwardRows<'a> {
    groups: &'a [f64],
    obs: &'a [f64],
    values: &'a [f64],
    fallback: &'a dyn Fn(usize) -> f64,
    tau: f64,
    mu: f64,
    sv: f64,
    gather_out: &'a mut [f64],
    product_out: &'a mut [f64],
    sum_out: &'a mut [f64],
    residuals: &'a mut [f64],
}

impl ForwardRows<'_> {
    /// The forward pass of each node, row by row: the gather, the product in
    /// its node's operand order, `ScalarBroadcastAdd`'s `mu + v`, and the
    /// Normal term's residual and sum of squares, which it returns.
    fn run<const PRODUCT: bool, const SCALE_FIRST: bool, const SUM: bool, const VALID: bool>(
        self,
    ) -> f64 {
        let len = self.gather_out.len();
        let mut sum_sq = 0.0f64;
        for i in 0..len {
            let group = self.groups[i] as usize;
            let mut m = match self.values.get(group) {
                Some(&value) => value,
                None => (self.fallback)(group),
            };
            self.gather_out[i] = m;
            if PRODUCT {
                m = if SCALE_FIRST {
                    ElementwiseOp::Mul.value(self.tau, m)
                } else {
                    ElementwiseOp::Mul.value(m, self.tau)
                };
                self.product_out[i] = m;
            }
            if SUM {
                let total = self.mu + m;
                m = total;
                self.sum_out[i] = m;
            }
            if VALID {
                let d = (self.obs[i] - m) / self.sv;
                self.residuals[i] = d;
                sum_sq += d * d;
            }
        }
        sum_sq
    }
}

/// The reverse rows of a fused chain.
struct ReverseRows<'a> {
    residuals: &'a [f64],
    groups: &'a [f64],
    gathered: &'a [f64],
    derivatives: &'a [f64],
    grad: &'a mut [f64],
    a_s: f64,
    sv: f64,
    tau: f64,
    /// The product scale's adjoint before the product's reverse pass.
    dtau: f64,
}

/// What the fused reverse rows add, and which nodes would have run.
struct ReverseSums {
    dsigma: f64,
    dmu: f64,
    dtau: f64,
    any_sum: bool,
    any_product: bool,
    any_gather: bool,
}

impl ReverseRows<'_> {
    /// Each node's reverse pass, row by row, in the separate passes' order;
    /// `None` when a group index falls past the gather's parameters.
    fn run<const PRODUCT: bool, const SUM: bool, const SIGMA: bool>(self) -> Option<ReverseSums> {
        let (mut dsigma, mut dmu, mut dtau) = (0.0f64, 0.0f64, self.dtau);
        let (mut any_sum, mut any_product, mut any_gather) = (false, false, false);
        for (i, &d) in self.residuals.iter().enumerate() {
            // The term's adjoint of its mean, into a zeroed buffer.
            let mut adjoint = 0.0 + self.a_s * (d / self.sv);
            if SIGMA {
                dsigma += (d * d - 1.0) / self.sv;
            }
            if SUM {
                any_sum |= adjoint != 0.0;
                dmu += adjoint;
                // Into the product's zeroed adjoint buffer.
                let product_adjoint = 0.0 + adjoint;
                adjoint = product_adjoint;
            }
            if PRODUCT {
                any_product |= adjoint != 0.0;
                adjoint = if adjoint == 0.0 {
                    0.0
                } else {
                    // `Mul`'s adjoints, `(upstream * b, upstream * a)`: in
                    // either operand order the scale takes `upstream * z`
                    // and the gather `upstream * tau`.
                    dtau += adjoint * self.gathered[i];
                    0.0 + adjoint * self.tau
                };
            }
            any_gather |= adjoint != 0.0;
            let group = self.groups[i] as usize;
            match (self.grad.get_mut(group), self.derivatives.get(group)) {
                (Some(slot), Some(&derivative)) => *slot += adjoint * derivative,
                _ => return None,
            }
        }
        Some(ReverseSums {
            dsigma,
            dmu,
            dtau,
            any_sum,
            any_product,
            any_gather,
        })
    }
}

impl Evaluator {
    /// Decide, at chain `c`'s gather, whether this call runs it fused: the
    /// graph still holds it, its vectors' only readers are its own nodes, and
    /// the separate gather would transform each group once rather than each
    /// row (see [`Evaluator::gather_by_group`]), which the fused loops always
    /// do. A gather with more groups than rows keeps the per-row loop.
    // Out of line, as are the fused passes. Inlined into the evaluator's node
    // loops, they made a graph that declines fusion (200 rows and 5,000
    // groups) about 7% slower; out of line it measured within 0.3% of main.
    // The loops still carry the plan's dispatch, so their code is not main's.
    #[inline(never)]
    pub(super) fn grouped_decide(&mut self, graph: &Graph, params: &[f64], c: usize) {
        let chain = &self.grouped.chains[c];
        let span = (
            chain.param_start,
            chain.n_params,
            NodeId(chain.indices_node),
        );
        // The cheaper checks first; the reader count scans the graph, once
        // per call.
        let fused = self.fusion_enabled()
            && self.gather_by_group(graph, params, span, chain.len)
            && chain.still_in(graph)
            && self.grouped.read_once(graph, c);
        self.grouped.fused_this_call[c] = fused;
    }

    /// Run chain `c` of the grouped plan forward, at its term, after any
    /// constant between its nodes has its value. Only for a chain
    /// [`Self::grouped_decide`] chose to fuse in this call.
    #[inline(never)]
    pub(super) fn grouped_forward(&mut self, graph: &Graph, params: &[f64], c: usize) {
        let chain = self.grouped.chains[c].clone();
        // As `Op::Gather`: each group transformed once, and a group past the
        // parameters it was given transformed from the row.
        let mut values = std::mem::take(&mut self.group_scratch);
        values.clear();
        values.extend((0..chain.n_params).map(|g| {
            let k = chain.param_start + g;
            graph.param_transforms[k].apply(params[k])
        }));
        let tau = chain.product.map(|(_, tau)| self.scalars[tau]);
        let mu = chain.sum.map(|(_, mu)| self.scalars[mu]);
        let sv = self.scalars[chain.sigma];
        let valid = scale_is_valid(sv);
        let offset = |node: usize| match self.node_kind[node] {
            NodeKind::ComputedVec(off) => off,
            _ => unreachable!("checked when the chain was found"),
        };
        let gather_off = offset(chain.gather);
        let product_off = chain.product.map(|(p, _)| offset(p));
        let sum_off = chain.sum.map(|(s, _)| offset(s));
        let mut residuals = std::mem::take(&mut self.observation_cache[chain.obs]);
        if valid {
            residuals.resize(chain.len, 0.0);
        }
        let groups = &self.binding.vectors[chain.indices];
        let obs = &self.binding.observations[chain.obs_data_idx];
        let fallback = |group: usize| {
            let k = chain.param_start + group;
            graph.param_transforms[k].apply(params[k])
        };
        let [gather_out, product_out, sum_out] = disjoint_rows(
            &mut self.vec_buf,
            [Some(gather_off), product_off, sum_off],
            chain.len,
        );
        let rows = ForwardRows {
            groups: &groups[..chain.len],
            obs: &obs[..chain.len],
            values: &values,
            fallback: &fallback,
            tau: tau.unwrap_or(0.0),
            mu: mu.unwrap_or(0.0),
            sv,
            gather_out,
            product_out,
            sum_out,
            residuals: &mut residuals,
        };
        // One loop per shape of chain, so no row branches on the shape.
        let sum_sq = match (tau.is_some(), chain.scale_first, mu.is_some(), valid) {
            (false, _, false, false) => rows.run::<false, false, false, false>(),
            (false, _, false, true) => rows.run::<false, false, false, true>(),
            (false, _, true, false) => rows.run::<false, false, true, false>(),
            (false, _, true, true) => rows.run::<false, false, true, true>(),
            (true, false, false, false) => rows.run::<true, false, false, false>(),
            (true, false, false, true) => rows.run::<true, false, false, true>(),
            (true, false, true, false) => rows.run::<true, false, true, false>(),
            (true, false, true, true) => rows.run::<true, false, true, true>(),
            (true, true, false, false) => rows.run::<true, true, false, false>(),
            (true, true, false, true) => rows.run::<true, true, false, true>(),
            (true, true, true, false) => rows.run::<true, true, true, false>(),
            (true, true, true, true) => rows.run::<true, true, true, true>(),
        };
        self.observation_cache[chain.obs] = residuals;
        self.group_scratch = values;
        self.scalars[chain.obs] = if valid {
            let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
            obs.len() as f64 * log_norm - 0.5 * sum_sq
        } else {
            f64::NEG_INFINITY
        };
        self.grouped.forward_passes += 1;
    }

    /// Run chain `c` of the grouped plan in reverse, if this call's forward
    /// pass ran it fused. `a_s`, the term's adjoint, is not zero. Returns
    /// false, having changed nothing, when the separate nodes must run
    /// instead; the reverse pass then stops skipping them.
    ///
    /// Each node's adjoints are what its own reverse pass would have added,
    /// in its order, and each is applied only if that pass would have run:
    /// the separate passes skip a vector node whose adjoints are all zero.
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
        let chain = self.grouped.chains[c].clone();
        let sv = self.scalars[chain.sigma];
        self.grouped.reverse_passes += 1;
        if !scale_is_valid(sv) {
            // The term adds nothing, so every node before it is inactive.
            return true;
        }
        let sigma_adjoint = !(self.constant_at_construction[chain.sigma]
            && matches!(graph.nodes[chain.sigma].op, Op::Constant(_)));
        let tau = chain.product.map(|(_, tau)| self.scalars[tau]);
        let mut derivatives = std::mem::take(&mut self.group_scratch);
        derivatives.clear();
        derivatives.extend((0..chain.n_params).map(|g| {
            let k = chain.param_start + g;
            graph.param_transforms[k].derivative(params[k])
        }));
        // The gather's gradients start from what later nodes added.
        let mut grad = std::mem::take(&mut self.grouped_grad);
        grad.clear();
        grad.extend_from_slice(&self.grad[chain.param_start..chain.param_start + chain.n_params]);
        let gather_off = match self.node_kind[chain.gather] {
            NodeKind::ComputedVec(off) => off,
            _ => unreachable!("checked when the chain was found"),
        };
        let residuals = &self.observation_cache[chain.obs];
        let rows = ReverseRows {
            residuals,
            groups: &self.binding.vectors[chain.indices][..residuals.len()],
            gathered: &self.vec_buf[gather_off..gather_off + residuals.len()],
            derivatives: &derivatives,
            grad: &mut grad,
            a_s,
            sv,
            tau: tau.unwrap_or(0.0),
            dtau: chain.product.map_or(0.0, |(_, tau)| self.adj_scalars[tau]),
        };
        // One loop per shape of chain, so no row branches on the shape.
        let sums = match (tau.is_some(), chain.sum.is_some(), sigma_adjoint) {
            (false, false, false) => rows.run::<false, false, false>(),
            (false, false, true) => rows.run::<false, false, true>(),
            (false, true, false) => rows.run::<false, true, false>(),
            (false, true, true) => rows.run::<false, true, true>(),
            (true, false, false) => rows.run::<true, false, false>(),
            (true, false, true) => rows.run::<true, false, true>(),
            (true, true, false) => rows.run::<true, true, false>(),
            (true, true, true) => rows.run::<true, true, true>(),
        };
        // A group index past the gather's parameters reads a parameter the
        // separate pass would have written in place; leave those to it.
        let Some(ReverseSums {
            dsigma,
            dmu,
            dtau,
            any_sum,
            any_product,
            any_gather,
        }) = sums
        else {
            self.group_scratch = derivatives;
            self.grouped_grad = grad;
            self.grouped.reverse_passes -= 1;
            self.grouped.fused_this_call[c] = false;
            return false;
        };
        // In the separate passes' order: the term, its mean, the product.
        if sigma_adjoint {
            self.adj_scalars[chain.sigma] += a_s * dsigma;
        }
        let sum_ran = chain.sum.is_none() || any_sum;
        if let (Some((_, mu)), true) = (chain.sum, any_sum) {
            self.adj_scalars[mu] += dmu;
        }
        let product_ran = sum_ran && (chain.product.is_none() || any_product);
        if let (Some((_, tau)), true) = (chain.product, sum_ran && any_product) {
            self.adj_scalars[tau] = dtau;
        }
        if product_ran && any_gather {
            self.grad[chain.param_start..chain.param_start + chain.n_params].copy_from_slice(&grad);
        }
        self.group_scratch = derivatives;
        self.grouped_grad = grad;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::reference::{eval_logp, grad_logp};
    use super::*;
    use crate::graph::ParamTransform;

    /// One grouped chain or two sharing their scalars, built the way the
    /// model compiler builds `mu + tau * z[group]` under a Normal likelihood.
    #[derive(Clone, Copy)]
    struct Shape {
        rows: usize,
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

    struct Built {
        graph: Graph,
        params: usize,
        terms: Vec<usize>,
    }

    fn build(shape: Shape, y: impl Fn(usize) -> f64) -> Built {
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
        let sigma = if shape.constant_sigma {
            g.add_constant(0.7)
        } else {
            let raw = g.add_param("log_sigma");
            g.log_half_normal_logp(raw, one);
            g.exp(raw)
        };
        let weight = shape.weighted.then(|| g.add_param("w"));
        let group = g.add_data(
            "group",
            (0..shape.rows)
                .map(|i| ((i * 7 + i / 3) % shape.groups) as f64)
                .collect(),
        );
        let mut terms = Vec::new();
        for term in 0..1 + shape.second_term as usize {
            let gathered = g.gather(z, shape.groups, group);
            let mut linpred = gathered;
            if shape.scale {
                let scale = match (shape.constant_scale, shape.signed_scale) {
                    (true, false) => g.add_constant(0.8),
                    (true, true) => g.add_constant(-0.8),
                    (false, _) => tau,
                };
                linpred = if shape.scale_last {
                    g.elementwise(ElementwiseOp::Mul, linpred, Some(scale))
                } else {
                    g.elementwise(ElementwiseOp::Mul, scale, Some(linpred))
                };
            }
            if shape.mean {
                linpred = g.scalar_broadcast_add(mu, linpred);
            }
            let obs = g.add_obs_data((0..shape.rows).map(|i| y(i + 1000 * term)).collect());
            let node = g.obs_logp_normal(linpred, sigma, obs);
            if let Some(weight) = weight {
                let weighted = g.mul(node, weight);
                g.add_logp_term(weighted);
            }
            terms.push(node.0);
        }
        let params = g.param_count;
        Built {
            graph: g,
            params,
            terms,
        }
    }

    fn same(a: f64, b: f64) -> bool {
        // Exact bits, NaN payloads included, as this build computes them.
        a.to_bits() == b.to_bits()
    }

    /// Every node's value, and the log density and gradient, from `evaluator`
    /// after `compute` at `point`.
    fn everything(evaluator: &mut Evaluator, g: &Graph, point: &[f64]) -> Vec<f64> {
        evaluator.compute(g, point);
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
            set(&|p| p[sigma] = -700.0);
            set(&|p| p[sigma] = 700.0);
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

    /// The fused loops against the same chain run as its separate nodes, and
    /// against the per-element loops those replaced: the log density, the
    /// gradient and every node's value, to the bit, at every point. Each
    /// variant must also have run fused.
    #[test]
    fn grouped_normal_chains_match_their_separate_nodes_bit_for_bit() {
        let mut nan_values = 0usize;
        let mut cases = 0usize;
        for rows in [1usize, 2, 5, 37] {
            for groups in [1usize, 3, 10] {
                for flags in 0..512u32 {
                    let bit = |k: u32| flags & (1 << k) != 0;
                    let shape = Shape {
                        rows,
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
                    };
                    let built = build(shape, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
                    let g = &built.graph;
                    let mut fused = Evaluator::new(g);
                    let mut separate = Evaluator::new(g);
                    separate.unfused_groups = true;
                    let mut per_element = Evaluator::new(g);
                    per_element.per_element = true;
                    let chains = 1 + shape.second_term as usize;
                    assert_eq!(fused.grouped.chains.len(), chains);
                    for point in points(built.params, shape) {
                        let f = everything(&mut fused, g, &point);
                        let label =
                            format!("{rows} rows, {groups} groups, flags {flags:09b}, {point:?}");
                        assert_same(&label, &f, &everything(&mut separate, g, &point));
                        assert_same(&label, &f, &everything(&mut per_element, g, &point));
                        nan_values += f.iter().filter(|v| v.is_nan()).count();
                        cases += 1;
                    }
                    // A gather with more groups than rows keeps the per-row
                    // loop, so its chain runs as separate nodes.
                    let all = points(built.params, shape);
                    let active = all.iter().filter(|p| !inactive(shape, p)).count();
                    let (runs, reverse_runs) = if groups <= rows {
                        (
                            all.len() as u64 * chains as u64,
                            active as u64 * chains as u64,
                        )
                    } else {
                        (0, 0)
                    };
                    assert_eq!(fused.grouped.forward_passes, runs);
                    assert_eq!(fused.grouped.reverse_passes, reverse_runs);
                    assert_eq!(separate.grouped.forward_passes, 0);
                }
            }
        }
        // The overflowing points must reach the non-finite branches.
        assert!(nan_values > 0, "{cases} cases");
    }

    /// A grouped model whose responses equal its mean exactly at the checked
    /// point, so every residual and every adjoint of the chain is zero and
    /// the separate passes skip their nodes.
    #[test]
    fn zero_residuals_match_the_separate_nodes_bit_for_bit() {
        let shape = Shape {
            rows: 9,
            groups: 3,
            mean: true,
            scale: true,
            constant_sigma: false,
            positive_z: true,
            weighted: false,
            second_term: false,
            scale_last: false,
            constant_scale: false,
            signed_scale: false,
        };
        let point = [0.25, -0.5, 0.1, 0.9, -0.3, 0.2];
        let tau = (-0.5f64).exp();
        let z = [0.1f64.exp(), 0.9f64.exp(), (-0.3f64).exp()];
        let built = build(shape, |i| {
            let group = (i * 7 + i / 3) % 3;
            0.25 + tau * z[group]
        });
        let g = &built.graph;
        let mut fused = Evaluator::new(g);
        let mut separate = Evaluator::new(g);
        separate.unfused_groups = true;
        let f = everything(&mut fused, g, &point);
        let residuals = &fused.observation_cache[built.terms[0]];
        assert!(residuals.len() == 9 && residuals.iter().all(|&d| d == 0.0));
        assert_same("zero residuals", &f, &everything(&mut separate, g, &point));
        assert_eq!(fused.grouped.reverse_passes, 1);
    }

    /// The fused gradient against the reference evaluator, which knows
    /// nothing of fusion, and against central differences of its log density.
    #[test]
    fn grouped_normal_chains_match_the_reference_gradients() {
        for flags in 0..512u32 {
            let bit = |k: u32| flags & (1 << k) != 0;
            let shape = Shape {
                rows: 23,
                groups: 4,
                mean: bit(0),
                scale: bit(1),
                constant_sigma: bit(2),
                positive_z: bit(3),
                weighted: bit(4),
                second_term: bit(5),
                scale_last: bit(6),
                constant_scale: bit(7),
                signed_scale: bit(8),
            };
            let built = build(shape, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
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

    /// Chains the fused loops must not take, each checked against the
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
        let shape = Shape {
            rows: 13,
            groups: 3,
            mean: true,
            scale: true,
            constant_sigma: false,
            positive_z: false,
            weighted: false,
            second_term: false,
            scale_last: false,
            constant_scale: false,
            signed_scale: false,
        };
        let built = build(shape, |i| 0.3 + (i % 5) as f64 / 4.0);
        let point = (0..built.params)
            .map(|k| 0.2 * k as f64 - 0.3)
            .collect::<Vec<_>>();
        let obs = built.terms[0];

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
        let spare = g.add_data("spare", vec![0.0; shape.rows]);
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

    /// Every field of the graph the fused loops read, changed one at a time
    /// after the evaluator found its chain: the gather's indices and span,
    /// the product's scale, operand order and operator, the mean, the term's
    /// scale, observations and family, and the effects' transform. Each time
    /// the evaluator must agree, to the bit, with one built for the changed
    /// graph. A changed transform is read on every call, so that chain still
    /// runs fused; every other change must send it to the separate nodes.
    #[test]
    fn every_field_the_fused_loops_read_changed_after_construction() {
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
            assert_same(label, &stale, &everything(&mut fresh, &changed, &point));
            let expected = if *still_fused { 2 } else { 1 };
            assert_eq!(evaluator.grouped.forward_passes, expected, "{label}");
        }
    }

    /// A gather whose adjoints are all zero must leave its gradients exactly
    /// as later nodes left them. A matrix product's reverse pass, with an
    /// entry so small that `-0.5 * entry` underflows in a fused multiply-add,
    /// leaves the effect's gradient at -0.0; adding the inactive gather's
    /// zeros would make it +0.0. The separate nodes skip the gather, and so
    /// must the fused loops.
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

    /// Groups whose effects mix transforms keep the separate gather's per-row
    /// loop, as `gather_by_group` decides, and so the separate nodes; the
    /// results must still match them to the bit.
    #[test]
    fn mixed_transforms_keep_the_separate_nodes() {
        let shape = Shape {
            rows: 20,
            groups: 4,
            mean: true,
            scale: true,
            constant_sigma: false,
            positive_z: false,
            weighted: false,
            second_term: false,
            scale_last: false,
            constant_scale: false,
            signed_scale: false,
        };
        let built = build(shape, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
        let mut g = built.graph;
        let mut uniform = Evaluator::new(&g);
        let point = (0..built.params)
            .map(|k| 0.31 * ((k * 3) % 5) as f64 - 0.6)
            .collect::<Vec<_>>();
        uniform.compute(&g, &point);
        assert_eq!(uniform.grouped.forward_passes, 1);
        // The first effect, parameter 2, positive; the other three not.
        g.param_transforms[2] = ParamTransform::Exp;
        let mut mixed = Evaluator::new(&g);
        let mut separate = Evaluator::new(&g);
        separate.unfused_groups = true;
        assert_same(
            "mixed transforms",
            &everything(&mut mixed, &g, &point),
            &everything(&mut separate, &g, &point),
        );
        assert_eq!(mixed.grouped.forward_passes, 0);
    }

    /// A seeded NUTS fit of a grouped model runs the fused loops: the
    /// process-wide count grows by at least one pass per leapfrog step.
    #[test]
    fn a_seeded_fit_runs_the_fused_loops() {
        let shape = Shape {
            rows: 40,
            groups: 5,
            mean: true,
            scale: true,
            constant_sigma: false,
            positive_z: false,
            weighted: false,
            second_term: false,
            scale_last: false,
            constant_scale: false,
            signed_scale: false,
        };
        let built = build(shape, |i| ((i * 13) % 17) as f64 / 4.0 - 2.0);
        let before = fused_grouped_normal_passes();
        let config = crate::sampler::SamplerConfig {
            num_chains: 2,
            num_draws: 50,
            num_warmup: 50,
            seed: 7,
            num_threads: 1,
            show_progress: false,
            ..Default::default()
        };
        let result = crate::sampler::sample(built.graph, config).expect("the fit runs");
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
    }
}
