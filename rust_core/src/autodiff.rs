use crate::data::DataBinding;
use crate::graph::{Graph, GraphShapeError, NodeId, Op, ParamTransform};

/// The crate's single logistic sigmoid; see [`crate::graph::stable_sigmoid`].
pub(crate) use crate::graph::stable_sigmoid as sigmoid_stable;
use crate::graph::{bounded_sigmoid, bounded_sigmoid_adjoint, stable_sigmoid_derivative};

// ---------------------------------------------------------------------------
// Evaluator — zero-allocation gradient computation
// ---------------------------------------------------------------------------

/// Classifies each graph node for the Evaluator's storage layout.
#[derive(Clone, Copy)]
enum NodeKind {
    Scalar,
    /// References graph.data_vectors[idx] directly — no copy needed.
    DataRef(usize),
    /// Computed vector stored at byte offset `off` in the flat vec_buf.
    ComputedVec(usize),
}

/// Pre-allocated evaluator that computes log-probability and its gradient
/// with zero heap allocations in the hot loop.
///
/// Created once per chain (or once globally) from a `&Graph`. All vector
/// intermediates live in a single contiguous `vec_buf` / `adj_vec_buf`
/// and are overwritten in place on each call.
pub struct Evaluator {
    binding: DataBinding,
    node_lengths: Vec<usize>,
    node_kind: Vec<NodeKind>,
    /// Scalar value per node (unused slots for vector/data nodes).
    scalars: Vec<f64>,
    /// Flat buffer for all computed-vector forward values.
    /// Layout: [slot_0: vec_len floats][slot_1: vec_len floats]...
    vec_buf: Vec<f64>,
    /// Adjoint per scalar node.
    adj_scalars: Vec<f64>,
    /// Flat buffer for all computed-vector adjoint values.
    adj_vec_buf: Vec<f64>,
    /// Output gradient vector w.r.t. parameters.
    pub grad: Vec<f64>,
    /// Cached total log-probability from the last `compute` call.
    pub total_logp: f64,
    /// Map from param index to node id, for extracting gradients.
    /// `None` for vector params that have no `Op::Param` node (e.g. MatVecMul params).
    param_node_ids: Vec<Option<usize>>,
    /// Reusable scratch space for constrained vector parameters and gradients.
    param_scratch: Vec<f64>,
}

fn validate_binding_slots(graph: &Graph, binding: &DataBinding) -> Result<(), GraphShapeError> {
    binding
        .validate_for(graph)
        .map_err(|error| GraphShapeError::new(error.to_string()))?;
    validate_slot_coverage(graph, binding)
}

/// Check that `binding` provides every slot the graph indexes by raw index.
///
/// Separate from [`validate_binding_slots`] because it has to run before
/// [`validate_node_lengths`], which reads those slots directly, and
/// `validate_node_lengths` has a second caller in [`Graph::validate_shapes`]
/// that has no evaluator and no reason to re-run the payload validation.
pub(crate) fn validate_slot_coverage(
    graph: &Graph,
    binding: &DataBinding,
) -> Result<(), GraphShapeError> {
    let mut required_vectors = 0usize;
    let mut required_observations = 0usize;
    let mut required_matrices = 0usize;
    for node in &graph.nodes {
        // Exhaustive and deliberately without a catch-all, for the reason
        // [`Op::visit_dependencies`] gives. An op that reads a binding slot by
        // raw index -- not through a node id and not through the parameter
        // vector -- has to be counted here or the slot it reads is never
        // checked against the binding, and `validate_node_lengths` then indexes
        // past the end of a binding this function has just called complete.
        // `_ => {}` silently gave every new variant the wrong answer;
        // `Op::BroadcastObservation` and `Op::FusedLinearMu` were already
        // sitting in it.
        match &node.op {
            Op::Data(index) => required_vectors = required_vectors.max(*index + 1),
            // Predictor columns stored with `store_data_vec`, which creates no
            // `Op::Data` node to be counted above.
            Op::FusedLinearMu { data_indices, .. } => {
                for index in data_indices {
                    required_vectors = required_vectors.max(*index + 1);
                }
            }
            Op::ObsLogP { obs_data_idx, .. } | Op::BroadcastObservation { obs_data_idx, .. } => {
                required_observations = required_observations.max(*obs_data_idx + 1)
            }
            Op::MatVecMul { matrix_idx, .. } => {
                required_matrices = required_matrices.max(*matrix_idx + 1)
            }
            // Everything else reaches its inputs through a node id or the
            // parameter vector, so the slots it depends on are counted by
            // whichever op above owns them.
            Op::Elementwise { .. }
            | Op::Gather { .. }
            | Op::Sum(_)
            | Op::Param(_)
            | Op::Constant(_)
            | Op::Add(_, _)
            | Op::Mul(_, _)
            | Op::Exp(_)
            | Op::Sigmoid(_)
            | Op::BoundedSigmoid { .. }
            | Op::ScalarMulData(_, _)
            | Op::VectorAdd(_, _)
            | Op::ScalarBroadcastAdd(_, _)
            | Op::ScalarBroadcast(_)
            | Op::NormalLogP { .. }
            | Op::LogHalfNormalLogP { .. }
            | Op::StudentTLogP { .. }
            | Op::PositiveSupport { .. }
            | Op::BernoulliLogP { .. }
            | Op::PoissonLogP { .. }
            | Op::LogGammaLogP { .. }
            | Op::VectorNormalLogP { .. }
            | Op::VectorHalfNormalLogP { .. }
            | Op::VectorStudentTLogP { .. }
            | Op::VectorGammaLogP { .. }
            | Op::VectorBetaLogP { .. }
            | Op::VectorUniformLogP { .. } => {}
        }
    }
    if binding.vectors.len() < required_vectors
        || binding.observations.len() < required_observations
        || binding.matrices.len() < required_matrices
    {
        return Err(GraphShapeError::new(
            "binding does not provide every data slot referenced by graph",
        ));
    }
    Ok(())
}

/// Check the references a graph's nodes make before anything indexes by them.
///
/// `Graph::nodes` is public and a `NodeId` is an unchecked index, so a
/// hand-built graph can name a node that comes later (or does not exist), a
/// parameter past `param_count`, or a node whose id disagrees with its
/// position. Every pass below indexes per-node buffers by those ids in
/// declaration order, so each of these is an out-of-bounds read or a value read
/// before it is computed.
fn validate_topology(graph: &Graph) -> Result<(), GraphShapeError> {
    for (position, node) in graph.nodes.iter().enumerate() {
        if node.id.0 != position {
            return Err(GraphShapeError::new(format!(
                "node at position {position} carries id {}",
                node.id.0
            )));
        }
        let mut later_node = None;
        let mut bad_span = None;
        node.op.visit_dependencies(
            &mut |operand| {
                if operand.0 >= position {
                    later_node.get_or_insert(operand.0);
                }
            },
            &mut |start, len| {
                if start
                    .checked_add(len)
                    .is_none_or(|end| end > graph.param_count)
                {
                    bad_span.get_or_insert((start, len));
                }
            },
        );
        if let Some(operand) = later_node {
            return Err(GraphShapeError::new(format!(
                "node {position} reads node {operand}, which is not an earlier node"
            )));
        }
        if let Some((start, len)) = bad_span {
            return Err(GraphShapeError::new(format!(
                "node {position} reads parameters {start}..{} of {}",
                start.saturating_add(len),
                graph.param_count
            )));
        }
    }
    for id in graph
        .logp_terms
        .iter()
        .chain(graph.deterministics.iter().map(|(_, id)| id))
    {
        if id.0 >= graph.nodes.len() {
            return Err(GraphShapeError::new(format!(
                "graph output refers to node {}, past the last node",
                id.0
            )));
        }
    }
    Ok(())
}

/// The operands `op` reads as scalars (through `scalars[id]`), which must
/// therefore not be vector-valued nodes.
fn scalar_operands(op: &Op) -> Vec<NodeId> {
    match op {
        Op::Add(a, b) | Op::Mul(a, b) => vec![*a, *b],
        Op::Exp(a) | Op::Sigmoid(a) => vec![*a],
        Op::BoundedSigmoid { raw, .. } => vec![*raw],
        Op::ScalarMulData(scalar, _)
        | Op::ScalarBroadcastAdd(scalar, _)
        | Op::ScalarBroadcast(scalar)
        | Op::BroadcastObservation { scalar, .. } => vec![*scalar],
        Op::NormalLogP { x, mu, sigma } => vec![*x, *mu, *sigma],
        Op::LogHalfNormalLogP { x, sigma } => vec![*x, *sigma],
        Op::StudentTLogP { x, nu, mu, sigma } => vec![*x, *nu, *mu, *sigma],
        Op::PositiveSupport { x } => vec![*x],
        Op::BernoulliLogP { x, p } => vec![*x, *p],
        Op::PoissonLogP { x, lam } => vec![*x, *lam],
        Op::LogGammaLogP { x, alpha, beta } => vec![*x, *alpha, *beta],
        Op::ObsLogP { aux, .. } => aux.iter().copied().collect(),
        Op::FusedLinearMu {
            param_nodes,
            intercept,
            ..
        } => param_nodes.iter().chain(intercept).copied().collect(),
        Op::MatVecMul { intercept, .. } => intercept.iter().copied().collect(),
        Op::Elementwise { .. }
        | Op::Gather { .. }
        | Op::Sum(_)
        | Op::Param(_)
        | Op::Constant(_)
        | Op::Data(_)
        | Op::VectorAdd(_, _)
        | Op::VectorNormalLogP { .. }
        | Op::VectorHalfNormalLogP { .. }
        | Op::VectorStudentTLogP { .. }
        | Op::VectorGammaLogP { .. }
        | Op::VectorBetaLogP { .. }
        | Op::VectorUniformLogP { .. } => Vec::new(),
    }
}

/// Derive every vector length from its inputs; scalars have length zero.
pub(crate) fn validate_node_lengths(
    graph: &Graph,
    binding: &DataBinding,
) -> Result<Vec<usize>, GraphShapeError> {
    validate_topology(graph)?;
    let mut output_names: std::collections::HashSet<&str> =
        graph.param_names.iter().map(String::as_str).collect();
    for name in graph
        .schema
        .observations
        .iter()
        .filter_map(|slot| match &slot.kind {
            crate::data::SlotKind::Observation { likelihood } => Some(likelihood.as_str()),
            _ => None,
        })
        .chain(graph.deterministics.iter().map(|(name, _)| name.as_str()))
    {
        if name.is_empty() || !output_names.insert(name) {
            return Err(GraphShapeError::new(format!(
                "output name '{name}' must be unique"
            )));
        }
    }
    let mut lengths: Vec<usize> = Vec::with_capacity(graph.nodes.len());
    let mut dimensions: Vec<Option<&str>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let merge = |a: NodeId, b: NodeId| -> Result<usize, GraphShapeError> {
            let (x, y) = (lengths[a.0], lengths[b.0]);
            if x != 0 && y != 0 && x != y {
                Err(GraphShapeError::new(format!(
                    "expression shape mismatch: lengths {x} and {y}"
                )))
            } else {
                Ok(x.max(y))
            }
        };
        let len = match &node.op {
            Op::Data(i) => binding.vectors[*i].len(),
            Op::Elementwise { a, b, .. } => {
                if let Some(b) = b {
                    merge(*a, *b)?
                } else {
                    lengths[a.0]
                }
            }
            Op::ScalarMulData(_, v) | Op::ScalarBroadcastAdd(_, v) => lengths[v.0],
            Op::VectorAdd(a, b) => merge(*a, *b)?,
            Op::ScalarBroadcast(_) => binding.n_obs(),
            Op::BroadcastObservation { obs_data_idx, .. } => {
                binding.observations[*obs_data_idx].len()
            }
            Op::Gather {
                param_start,
                n_params,
                indices,
            } => {
                if param_start
                    .checked_add(*n_params)
                    .is_none_or(|end| end > graph.param_count)
                {
                    return Err(GraphShapeError::new("invalid parameter span"));
                }
                let Op::Data(di) = graph.nodes[indices.0].op else {
                    return Err(GraphShapeError::new("group indices must be data"));
                };
                if binding.vectors[di].iter().any(|x| {
                    !x.is_finite() || *x < 0.0 || x.fract() != 0.0 || *x >= *n_params as f64
                }) {
                    return Err(GraphShapeError::new(format!(
                        "group indices must be integers in [0, {n_params})"
                    )));
                }
                lengths[indices.0]
            }
            Op::FusedLinearMu { data_indices, .. } => {
                let n = data_indices
                    .first()
                    .map_or(0, |i| binding.vectors[*i].len());
                if data_indices.iter().any(|i| binding.vectors[*i].len() != n) {
                    return Err(GraphShapeError::new("linear predictor data lengths differ"));
                }
                n
            }
            Op::MatVecMul {
                matrix_idx,
                n_params,
                ..
            } => {
                let m = &binding.matrices[*matrix_idx];
                if m.n_cols != *n_params || m.n_rows.checked_mul(m.n_cols) != Some(m.data.len()) {
                    return Err(GraphShapeError::new(
                        "matrix shape does not match parameter span",
                    ));
                }
                m.n_rows
            }
            Op::ObsLogP {
                linpred_vec,
                obs_data_idx,
                ..
            } => {
                if lengths[linpred_vec.0] != binding.observations[*obs_data_idx].len() {
                    return Err(GraphShapeError::new(
                        "observation and predictor lengths differ",
                    ));
                }
                0
            }
            // Scalar-valued ops: length zero. Exhaustive and deliberately
            // without a catch-all, for the reason [`Op::visit_dependencies`]
            // gives -- under `_ => 0` a new vector-producing variant would be
            // given length zero, its elements would never be allocated in
            // `vec_buf`, and the shape would be silently wrong rather than a
            // build failure.
            Op::Sum(_)
            | Op::Param(_)
            | Op::Constant(_)
            | Op::Add(_, _)
            | Op::Mul(_, _)
            | Op::Exp(_)
            | Op::Sigmoid(_)
            | Op::BoundedSigmoid { .. }
            | Op::NormalLogP { .. }
            | Op::LogHalfNormalLogP { .. }
            | Op::StudentTLogP { .. }
            | Op::PositiveSupport { .. }
            | Op::BernoulliLogP { .. }
            | Op::PoissonLogP { .. }
            | Op::LogGammaLogP { .. }
            | Op::VectorNormalLogP { .. }
            | Op::VectorHalfNormalLogP { .. }
            | Op::VectorStudentTLogP { .. }
            | Op::VectorGammaLogP { .. }
            | Op::VectorBetaLogP { .. }
            | Op::VectorUniformLogP { .. } => 0,
        };
        let vector_dim = |i: usize| {
            graph
                .schema
                .vectors
                .get(i)
                .map_or("obs", |s| s.dim.as_str())
        };
        let obs_dim = |i: usize| {
            graph
                .schema
                .observations
                .get(i)
                .map_or("obs", |s| s.dim.as_str())
        };
        let merge_dim = |a: NodeId, b: NodeId| -> Result<Option<&str>, GraphShapeError> {
            match (dimensions[a.0], dimensions[b.0]) {
                (Some(x), Some(y)) if x != y => Err(GraphShapeError::new(format!(
                    "expression dimensions '{x}' and '{y}' differ"
                ))),
                (a, b) => Ok(a.or(b)),
            }
        };
        let dimension = match &node.op {
            Op::Data(i) => Some(vector_dim(*i)),
            Op::Elementwise { a, b, .. } => {
                if let Some(b) = b {
                    merge_dim(*a, *b)?
                } else {
                    dimensions[a.0]
                }
            }
            Op::VectorAdd(a, b) => merge_dim(*a, *b)?,
            Op::ScalarMulData(_, v) | Op::ScalarBroadcastAdd(_, v) => dimensions[v.0],
            Op::ScalarBroadcast(_) => Some(obs_dim(0)),
            Op::BroadcastObservation { obs_data_idx, .. } => Some(obs_dim(*obs_data_idx)),
            Op::Gather { indices, .. } => dimensions[indices.0],
            Op::FusedLinearMu { data_indices, .. } => {
                let dim = data_indices.first().map(|i| vector_dim(*i));
                if data_indices.iter().any(|i| Some(vector_dim(*i)) != dim) {
                    return Err(GraphShapeError::new("linear predictor dimensions differ"));
                }
                dim
            }
            Op::MatVecMul { matrix_idx, .. } => Some(
                graph
                    .schema
                    .matrices
                    .get(*matrix_idx)
                    .map_or("obs", |s| s.dim.as_str()),
            ),
            Op::ObsLogP {
                linpred_vec,
                obs_data_idx,
                ..
            } => {
                if dimensions[linpred_vec.0] != Some(obs_dim(*obs_data_idx)) {
                    return Err(GraphShapeError::new(
                        "observation and predictor dimensions differ",
                    ));
                }
                None
            }
            // Scalar-valued ops carry no named dimension. Same exhaustiveness
            // rule as the length match above: `_ => None` would let a new
            // vector-producing variant opt out of the dimension agreement
            // check without anyone noticing it had.
            Op::Sum(_)
            | Op::Param(_)
            | Op::Constant(_)
            | Op::Add(_, _)
            | Op::Mul(_, _)
            | Op::Exp(_)
            | Op::Sigmoid(_)
            | Op::BoundedSigmoid { .. }
            | Op::NormalLogP { .. }
            | Op::LogHalfNormalLogP { .. }
            | Op::StudentTLogP { .. }
            | Op::PositiveSupport { .. }
            | Op::BernoulliLogP { .. }
            | Op::PoissonLogP { .. }
            | Op::LogGammaLogP { .. }
            | Op::VectorNormalLogP { .. }
            | Op::VectorHalfNormalLogP { .. }
            | Op::VectorStudentTLogP { .. }
            | Op::VectorGammaLogP { .. }
            | Op::VectorBetaLogP { .. }
            | Op::VectorUniformLogP { .. } => None,
        };
        // A vector-producing op with no elements would be stored as a scalar
        // and reach the evaluator's unreachable vector arms.
        let produces_vector = matches!(
            node.op,
            Op::Gather { .. }
                | Op::BroadcastObservation { .. }
                | Op::ScalarMulData(_, _)
                | Op::VectorAdd(_, _)
                | Op::ScalarBroadcastAdd(_, _)
                | Op::ScalarBroadcast(_)
                | Op::FusedLinearMu { .. }
                | Op::MatVecMul { .. }
        );
        if produces_vector && len == 0 {
            return Err(GraphShapeError::new(format!(
                "vector operation at node {} has no elements",
                node.id.0
            )));
        }
        if let Some(operand) = scalar_operands(&node.op)
            .into_iter()
            .find(|operand| lengths[operand.0] != 0)
        {
            return Err(GraphShapeError::new(format!(
                "node {} reads node {} as a scalar, but it is a vector of length {}",
                node.id.0, operand.0, lengths[operand.0]
            )));
        }
        dimensions.push(dimension);
        lengths.push(len);
    }
    // The total log density sums node scalars, so a vector-valued term would
    // silently contribute nothing.
    if let Some(term) = graph.logp_terms.iter().find(|term| lengths[term.0] != 0) {
        return Err(GraphShapeError::new(format!(
            "log-density term node {} is a vector of length {}; sum it first",
            term.0, lengths[term.0]
        )));
    }
    Ok(lengths)
}

impl Evaluator {
    pub fn try_new(graph: &Graph) -> Result<Self, GraphShapeError> {
        let binding =
            DataBinding::from_graph(graph).map_err(|e| GraphShapeError::new(e.to_string()))?;
        Self::try_with_binding(graph, binding)
    }

    /// Construct an evaluator for immutable structure plus a validated dataset.
    pub fn try_with_binding(graph: &Graph, binding: DataBinding) -> Result<Self, GraphShapeError> {
        let n = graph.nodes.len();
        // Coverage first: the length pass reads `binding.vectors[i]`,
        // `binding.observations[i]` and `binding.matrices[i]` at the raw indices
        // the graph carries, so a missing slot is an out-of-bounds index there
        // rather than an error. Nothing else moves -- a binding that covers its
        // slots reaches the length pass and the payload validation in the order
        // it always did.
        validate_slot_coverage(graph, &binding)?;
        let node_lengths = validate_node_lengths(graph, &binding)?;
        validate_binding_slots(graph, &binding)?;

        let mut node_kind = Vec::with_capacity(n);
        let mut vec_buffer_len = 0usize;
        for node in &graph.nodes {
            let len = node_lengths[node.id.0];
            let kind = if let Op::Data(idx) = node.op {
                NodeKind::DataRef(idx)
            } else if len > 0 {
                let offset = vec_buffer_len;
                vec_buffer_len += len;
                NodeKind::ComputedVec(offset)
            } else {
                NodeKind::Scalar
            };
            node_kind.push(kind);
        }

        let mut param_node_ids: Vec<Option<usize>> = vec![None; graph.param_count];
        for node in &graph.nodes {
            if let Op::Param(pidx) = node.op {
                param_node_ids[pidx] = Some(node.id.0);
            }
        }

        Ok(Self {
            binding,
            node_lengths,
            node_kind,
            scalars: vec![0.0; n],
            vec_buf: vec![0.0; vec_buffer_len],
            adj_scalars: vec![0.0; n],
            adj_vec_buf: vec![0.0; vec_buffer_len],
            grad: vec![0.0; graph.param_count],
            total_logp: 0.0,
            param_node_ids,
            param_scratch: vec![0.0; graph.param_count],
        })
    }

    /// [`Self::try_new`] for graphs already known to be valid.
    ///
    /// # Panics
    ///
    /// If the graph fails shape validation; library code uses `try_new`.
    pub fn new(graph: &Graph) -> Self {
        Self::try_new(graph).expect("graph shape validation failed")
    }

    /// [`Self::try_with_binding`] for bindings already known to match.
    ///
    /// # Panics
    ///
    /// If the binding does not match the structure.
    pub fn with_binding(graph: &Graph, binding: DataBinding) -> Self {
        Self::try_with_binding(graph, binding).expect("validated binding does not match structure")
    }

    /// Reuse allocations while changing only the dataset payload and row count.
    pub fn rebind(&mut self, graph: &Graph, binding: DataBinding) -> Result<(), GraphShapeError> {
        validate_slot_coverage(graph, &binding)?;
        validate_binding_slots(graph, &binding)?;
        if validate_node_lengths(graph, &binding)? == self.node_lengths {
            self.binding = binding;
        } else {
            *self = Self::try_with_binding(graph, binding)?;
        }
        Ok(())
    }

    /// Read a vector element from either a Data node (graph reference) or
    /// a computed-vector node (vec_buf).
    #[inline(always)]
    fn read_vec(&self, node_id: usize, i: usize) -> f64 {
        match self.node_kind[node_id] {
            NodeKind::DataRef(di) => self.binding.vectors[di][i],
            NodeKind::ComputedVec(off) => self.vec_buf[off + i],
            NodeKind::Scalar => self.scalars[node_id],
        }
    }

    fn accumulate(&mut self, node: NodeId, i: usize, value: f64) {
        match self.node_kind[node.0] {
            NodeKind::Scalar => self.adj_scalars[node.0] += value,
            NodeKind::ComputedVec(off) => self.adj_vec_buf[off + i] += value,
            NodeKind::DataRef(_) => {}
        }
    }
    pub fn node_len(&self, node: NodeId) -> usize {
        self.node_lengths[node.0]
    }

    /// Read the scalar value of a node after `compute()`.
    pub fn scalar_at(&self, node: NodeId) -> f64 {
        self.scalars[node.0]
    }

    /// Read the i-th element of a vector node after `compute()`.
    ///
    /// The evaluator owns every value it reads, so `_graph` is unused; it is
    /// kept so existing callers need not change.
    pub fn vec_elem(&self, node: NodeId, i: usize, _graph: &Graph) -> f64 {
        self.read_vec(node.0, i)
    }

    /// Compute log-probability and its gradient. Results are stored in
    /// `self.total_logp` and `self.grad`. No heap allocations occur.
    pub fn compute(&mut self, graph: &Graph, params: &[f64]) {
        self.forward(graph, params);
        self.backward(graph, params);
    }

    /// Evaluate node values and `self.total_logp` without the reverse pass.
    ///
    /// For callers that only read forward values (prediction, deterministic
    /// outputs); `self.grad` is left as it was.
    pub fn forward(&mut self, graph: &Graph, params: &[f64]) {
        for node in &graph.nodes {
            let idx = node.id.0;
            // The one match over `Op` here that keeps its catch-all. Unlike the
            // shape passes above, the default is not a guess that a new variant
            // could silently fall into: a node's own length *is* its length, so
            // `node_lengths[idx]` is right for every variant that exists and
            // every variant that could be added. `Op::ObsLogP` is the single
            // exception because it is a scalar node -- length zero -- that still
            // has to walk its observation vector.
            let vl = match &node.op {
                Op::ObsLogP { obs_data_idx, .. } => self.binding.observations[*obs_data_idx].len(),
                _ => self.node_lengths[idx],
            };
            match &node.op {
                Op::Elementwise { operator, a, b } => {
                    for i in 0..vl.max(1) {
                        let av = self.read_vec(a.0, i);
                        let bv = b.map_or(0.0, |b| self.read_vec(b.0, i));
                        let value = operator.value(av, bv);
                        match self.node_kind[idx] {
                            NodeKind::ComputedVec(off) => self.vec_buf[off + i] = value,
                            _ => self.scalars[idx] = value,
                        }
                    }
                }
                Op::Gather {
                    param_start,
                    indices,
                    ..
                } => {
                    let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
                        unreachable!()
                    };
                    for i in 0..vl {
                        let k = *param_start + self.read_vec(indices.0, i) as usize;
                        self.vec_buf[off + i] = graph.param_transforms[k].apply(params[k]);
                    }
                }
                Op::Sum(a) => {
                    self.scalars[idx] = (0..self.node_lengths[a.0].max(1))
                        .map(|i| self.read_vec(a.0, i))
                        .sum()
                }
                Op::BroadcastObservation { scalar, .. } => {
                    let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
                        unreachable!()
                    };
                    self.vec_buf[off..off + vl].fill(self.scalars[scalar.0]);
                }
                Op::Param(pidx) => self.scalars[idx] = params[*pidx],
                Op::Constant(c) => self.scalars[idx] = *c,
                Op::Data(_) => {}
                Op::Add(a, b) => self.scalars[idx] = self.scalars[a.0] + self.scalars[b.0],
                Op::Mul(a, b) => self.scalars[idx] = self.scalars[a.0] * self.scalars[b.0],
                Op::Exp(a) => self.scalars[idx] = self.scalars[a.0].exp(),
                Op::Sigmoid(a) => self.scalars[idx] = sigmoid_stable(self.scalars[a.0]),
                Op::BoundedSigmoid { raw, lower, upper } => {
                    self.scalars[idx] = bounded_sigmoid(self.scalars[raw.0], *lower, *upper);
                }
                Op::ScalarMulData(scalar, data) => {
                    let s = self.scalars[scalar.0];
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        let d = self.read_vec(data.0, i);
                        self.vec_buf[out_off + i] = s * d;
                    }
                }
                Op::VectorAdd(a, b) => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        let va = self.read_vec(a.0, i);
                        let vb = self.read_vec(b.0, i);
                        self.vec_buf[out_off + i] = va + vb;
                    }
                }
                Op::ScalarBroadcastAdd(scalar, vec) => {
                    let s = self.scalars[scalar.0];
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        let v = self.read_vec(vec.0, i);
                        self.vec_buf[out_off + i] = s + v;
                    }
                }
                Op::ScalarBroadcast(scalar) => {
                    let s = self.scalars[scalar.0];
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        self.vec_buf[out_off + i] = s;
                    }
                }
                Op::NormalLogP { x, mu, sigma } => {
                    let xv = self.scalars[x.0];
                    let mv = self.scalars[mu.0];
                    let sv = self.scalars[sigma.0];
                    self.scalars[idx] = normal_logp_scalar(xv, mv, sv);
                }
                Op::LogHalfNormalLogP { x, sigma } => {
                    self.scalars[idx] =
                        log_half_normal_logp(self.scalars[x.0], self.scalars[sigma.0]);
                }
                Op::StudentTLogP { x, nu, mu, sigma } => {
                    self.scalars[idx] = student_t_logp_scalar(
                        self.scalars[x.0],
                        self.scalars[nu.0],
                        self.scalars[mu.0],
                        self.scalars[sigma.0],
                    );
                }
                Op::PositiveSupport { x } => {
                    let x = self.scalars[x.0];
                    self.scalars[idx] = if x.is_finite() && x > 0.0 {
                        0.0
                    } else {
                        f64::NEG_INFINITY
                    };
                }
                Op::BernoulliLogP { x, p } => {
                    self.scalars[idx] = bernoulli_logp_scalar(self.scalars[x.0], self.scalars[p.0]);
                }
                Op::PoissonLogP { x, lam } => {
                    self.scalars[idx] = poisson_logp_scalar(self.scalars[x.0], self.scalars[lam.0]);
                }
                Op::LogGammaLogP { x, alpha, beta } => {
                    self.scalars[idx] = log_gamma_logp(
                        self.scalars[x.0],
                        self.scalars[alpha.0],
                        self.scalars[beta.0],
                    );
                }
                Op::ObsLogP {
                    family,
                    linpred_vec,
                    aux,
                    obs_data_idx,
                } => {
                    let obs = &self.binding.observations[*obs_data_idx];
                    match family {
                        crate::graph::ObsFamily::Normal => {
                            let sigma_node = aux.expect("Normal obs logp requires sigma");
                            let sv = self.scalars[sigma_node.0];
                            // As `normal_logp_scalar`: outside the scale's
                            // support the density is zero, not NaN.
                            if !scale_is_valid(sv) {
                                self.scalars[idx] = f64::NEG_INFINITY;
                                continue;
                            }

                            let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
                            let n = obs.len() as f64;
                            let mut sum_sq = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let m = self.read_vec(linpred_vec.0, i);
                                let d = (y - m) / sv;
                                sum_sq += d * d;
                            }
                            self.scalars[idx] = n * log_norm - 0.5 * sum_sq;
                        }
                        crate::graph::ObsFamily::BernoulliLogit => {
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                sum += bernoulli_logit_logp(y, eta);
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::PoissonLog => {
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                sum += crate::count_sampling::log_mass_from_log_rate(y, eta);
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::ExponentialLog => {
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                sum += eta - y * eta.exp();
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::LogNormal => {
                            let sigma_node = aux.expect("LogNormal obs logp requires sigma");
                            let sv = self.scalars[sigma_node.0];
                            if !scale_is_valid(sv) {
                                self.scalars[idx] = f64::NEG_INFINITY;
                                continue;
                            }

                            let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
                            let mut sum = 0.0f64;
                            for (i, &observation) in obs.iter().take(vl).enumerate() {
                                let y = observation;
                                let m = self.read_vec(linpred_vec.0, i);
                                let ly = y.ln();
                                let d = (ly - m) / sv;
                                sum += log_norm - ly - 0.5 * d * d;
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::NegativeBinomialLog => {
                            let alpha_node = aux.expect("NegativeBinomial obs logp requires alpha");
                            let av = self.scalars[alpha_node.0];
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                sum += crate::negative_binomial::log_mass(y, eta, av);
                            }
                            self.scalars[idx] = sum;
                        }
                    }
                }
                Op::FusedLinearMu {
                    param_nodes,
                    data_indices,
                    intercept,
                } => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let base = intercept.map_or(0.0, |n| self.scalars[n.0]);
                    let out = &mut self.vec_buf[out_off..out_off + vl];
                    for v in out.iter_mut() {
                        *v = base;
                    }
                    for (k, &pn) in param_nodes.iter().enumerate() {
                        let beta = self.scalars[pn.0];
                        let data = &self.binding.vectors[data_indices[k]];
                        for i in 0..vl {
                            out[i] += beta * data[i];
                        }
                    }
                }
                Op::MatVecMul {
                    matrix_idx,
                    param_start,
                    n_params,
                    intercept,
                } => {
                    use faer::{col, linalg::matmul::matmul, mat, Parallelism};
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let matrix = &self.binding.matrices[*matrix_idx];
                    let base = intercept.map_or(0.0, |n| self.scalars[n.0]);
                    let out = &mut self.vec_buf[out_off..out_off + vl];
                    out.fill(base);
                    let x = mat::from_row_major_slice::<f64>(
                        &matrix.data,
                        matrix.n_rows,
                        matrix.n_cols,
                    );
                    let transforms =
                        &graph.param_transforms[*param_start..*param_start + *n_params];
                    let all_identity = transforms
                        .iter()
                        .all(|t| matches!(t, ParamTransform::Identity));
                    let beta_slice = if all_identity {
                        &params[*param_start..*param_start + *n_params]
                    } else {
                        let scratch =
                            &mut self.param_scratch[*param_start..*param_start + *n_params];
                        for k in 0..*n_params {
                            scratch[k] = transforms[k].apply(params[param_start + k]);
                        }
                        scratch
                    };
                    let beta_col = col::from_slice::<f64>(beta_slice);
                    let out_col = col::from_slice_mut::<f64>(out);
                    // Use Rayon threads for matrices large enough to amortise spawn cost.
                    let par = if matrix.n_rows * matrix.n_cols >= 100_000 {
                        Parallelism::Rayon(0)
                    } else {
                        Parallelism::None
                    };
                    matmul(
                        out_col.as_2d_mut(),
                        x,
                        beta_col.as_2d(),
                        Some(1.0),
                        1.0,
                        par,
                    );
                }
                Op::VectorNormalLogP {
                    param_start,
                    n_params,
                    mu,
                    sigma,
                } => {
                    let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();

                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        let z = (v - mu) / sigma;
                        sum += log_norm - 0.5 * z * z;
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorHalfNormalLogP {
                    param_start,
                    n_params,
                    sigma,
                } => {
                    // Combine density and Jacobian using the log scale ratio.
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum += log_half_normal_logp(raw, *sigma);
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorStudentTLogP {
                    param_start,
                    n_params,
                    nu,
                    mu,
                    sigma,
                } => {
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        sum += student_t_logp_scalar(v, *nu, *mu, *sigma);
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorGammaLogP {
                    param_start,
                    n_params,
                    alpha,
                    beta,
                } => {
                    // Combine density and Jacobian before exponentiation.
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum += log_gamma_logp(raw, *alpha, *beta);
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorBetaLogP {
                    param_start,
                    n_params,
                    alpha,
                    beta,
                } => {
                    // s = sigmoid(raw), logp = lnΓ(α+β)-lnΓ(α)-lnΓ(β) + α·log(s) + β·log(1-s)
                    // Jacobian of sigmoid = s·(1-s), so log|J| = log(s) + log(1-s)
                    // Combined: lnΓ(α+β)-lnΓ(α)-lnΓ(β) + (α-1)·log(s) + (β-1)·log(1-s) + log(s) + log(1-s)
                    //         = lnΓ(α+β)-lnΓ(α)-lnΓ(β) + α·log(s) + β·log(1-s)
                    let log_norm = ln_gamma(alpha + beta) - ln_gamma(*alpha) - ln_gamma(*beta);
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum += log_norm - alpha * softplus(-raw) - beta * softplus(raw);
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorUniformLogP {
                    param_start,
                    n_params,
                    lower,
                    upper,
                } => {
                    if !uniform_bounds_valid(*lower, *upper) {
                        self.scalars[idx] = f64::NEG_INFINITY;
                        continue;
                    }
                    // s = sigmoid(raw), logp_uniform = -log(hi-lo) (const), Jacobian = s·(1-s)·(hi-lo)
                    // Combined: -log(hi-lo) + log(s·(1-s)·(hi-lo)) = log(s·(1-s)) = log(s) + log(1-s)
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum -= softplus(-raw) + softplus(raw);
                    }
                    self.scalars[idx] = sum;
                }
            }
        }

        // Total log-probability
        self.total_logp = graph.logp_terms.iter().map(|id| self.scalars[id.0]).sum();
    }

    /// Reverse pass over the values the last [`Self::forward`] left behind.
    fn backward(&mut self, graph: &Graph, params: &[f64]) {
        // Zero adjoint buffers and gradient
        self.adj_scalars.iter_mut().for_each(|x| *x = 0.0);
        self.adj_vec_buf.iter_mut().for_each(|x| *x = 0.0);
        self.grad.iter_mut().for_each(|x| *x = 0.0);

        // Seed
        for &id in &graph.logp_terms {
            self.adj_scalars[id.0] += 1.0;
        }

        for node in graph.nodes.iter().rev() {
            let idx = node.id.0;
            // The one match over `Op` here that keeps its catch-all. Unlike the
            // shape passes above, the default is not a guess that a new variant
            // could silently fall into: a node's own length *is* its length, so
            // `node_lengths[idx]` is right for every variant that exists and
            // every variant that could be added. `Op::ObsLogP` is the single
            // exception because it is a scalar node -- length zero -- that still
            // has to walk its observation vector.
            let vl = match &node.op {
                Op::ObsLogP { obs_data_idx, .. } => self.binding.observations[*obs_data_idx].len(),
                _ => self.node_lengths[idx],
            };
            let a_s = self.adj_scalars[idx];
            // Output-only nodes and inactive reverse edges cannot affect the
            // target, even when their local derivative is infinite or NaN.
            let active = match self.node_kind[idx] {
                NodeKind::ComputedVec(off) => self.adj_vec_buf[off..off + vl]
                    .iter()
                    .any(|&adj| adj != 0.0),
                _ => a_s != 0.0,
            };
            if !active {
                continue;
            }

            match &node.op {
                Op::Elementwise { operator, a, b } => {
                    for i in 0..vl.max(1) {
                        let av = self.read_vec(a.0, i);
                        let bv = b.map_or(0.0, |b| self.read_vec(b.0, i));
                        let upstream = match self.node_kind[idx] {
                            NodeKind::ComputedVec(off) => self.adj_vec_buf[off + i],
                            _ => a_s,
                        };
                        if upstream == 0.0 {
                            continue;
                        }
                        // Composed with the upstream adjoint rather than
                        // multiplied by it afterwards: several of these local
                        // derivatives leave the exponent range on their own
                        // while the product does not. See
                        // `ElementwiseOp::adjoints`.
                        let (da, db) = operator.adjoints(upstream, av, bv);
                        self.accumulate(*a, i, da);
                        if let Some(b) = b {
                            self.accumulate(*b, i, db);
                        }
                    }
                }
                Op::Gather {
                    param_start,
                    indices,
                    ..
                } => {
                    let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
                        unreachable!()
                    };
                    for i in 0..vl {
                        let k = *param_start + self.read_vec(indices.0, i) as usize;
                        self.grad[k] += self.adj_vec_buf[off + i]
                            * graph.param_transforms[k].derivative(params[k]);
                    }
                }
                Op::Sum(a) => {
                    for i in 0..self.node_lengths[a.0].max(1) {
                        self.accumulate(*a, i, a_s);
                    }
                }
                Op::BroadcastObservation { scalar, .. } => {
                    let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
                        unreachable!()
                    };
                    self.adj_scalars[scalar.0] +=
                        self.adj_vec_buf[off..off + vl].iter().sum::<f64>();
                }
                Op::Param(_) | Op::Constant(_) | Op::Data(_) => {}

                Op::Add(a, b) => {
                    self.adj_scalars[a.0] += a_s;
                    self.adj_scalars[b.0] += a_s;
                }
                Op::Mul(a, b) => {
                    let va = self.scalars[a.0];
                    let vb = self.scalars[b.0];
                    self.adj_scalars[a.0] += a_s * vb;
                    self.adj_scalars[b.0] += a_s * va;
                }
                Op::Exp(a) => {
                    let va = self.scalars[a.0].exp();
                    self.adj_scalars[a.0] += a_s * va;
                }
                Op::Sigmoid(a) => {
                    self.adj_scalars[a.0] += a_s * stable_sigmoid_derivative(self.scalars[a.0]);
                }
                Op::BoundedSigmoid { raw, lower, upper } => {
                    // Ordered so that neither a huge span nor a tiny one leaves
                    // the exponent range; see `bounded_sigmoid_adjoint`.
                    self.adj_scalars[raw.0] +=
                        bounded_sigmoid_adjoint(a_s, self.scalars[raw.0], *lower, *upper);
                }

                Op::ScalarMulData(scalar, data) => {
                    let s = self.scalars[scalar.0];
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let mut ds = 0.0f64;
                    for i in 0..vl {
                        let upstream = self.adj_vec_buf[out_off + i];
                        let d_val = self.read_vec(data.0, i);
                        ds += upstream * d_val;
                        // Propagate to data's adjoint (only if it's a computed vec)
                        if let NodeKind::ComputedVec(d_off) = self.node_kind[data.0] {
                            self.adj_vec_buf[d_off + i] += upstream * s;
                        }
                    }
                    self.adj_scalars[scalar.0] += ds;
                }
                Op::VectorAdd(a, b) => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        let upstream = self.adj_vec_buf[out_off + i];
                        if let NodeKind::ComputedVec(a_off) = self.node_kind[a.0] {
                            self.adj_vec_buf[a_off + i] += upstream;
                        }
                        if let NodeKind::ComputedVec(b_off) = self.node_kind[b.0] {
                            self.adj_vec_buf[b_off + i] += upstream;
                        }
                    }
                }
                Op::ScalarBroadcastAdd(scalar, vec) => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let mut ds = 0.0f64;
                    for i in 0..vl {
                        let upstream = self.adj_vec_buf[out_off + i];
                        ds += upstream;
                        if let NodeKind::ComputedVec(v_off) = self.node_kind[vec.0] {
                            self.adj_vec_buf[v_off + i] += upstream;
                        }
                    }
                    self.adj_scalars[scalar.0] += ds;
                }
                Op::ScalarBroadcast(scalar) => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    // d(loss)/d(scalar) = sum of d(loss)/d(out[i]) over all i
                    let ds: f64 = self.adj_vec_buf[out_off..out_off + vl].iter().sum();
                    self.adj_scalars[scalar.0] += ds;
                }
                Op::NormalLogP { x, mu, sigma } => {
                    let xv = self.scalars[x.0];
                    let mv = self.scalars[mu.0];
                    let sv = self.scalars[sigma.0];
                    if !scale_is_valid(sv) {
                        continue;
                    }
                    let z = (xv - mv) / sv;
                    self.adj_scalars[x.0] += a_s * (-z / sv);
                    self.adj_scalars[mu.0] += a_s * (z / sv);
                    self.adj_scalars[sigma.0] += a_s * ((z * z - 1.0) / sv);
                }
                Op::LogHalfNormalLogP { x, sigma } => {
                    let raw = self.scalars[x.0];
                    let scale = self.scalars[sigma.0];
                    if scale.is_finite() && scale > 0.0 {
                        let z2 = (2.0 * (raw - scale.ln())).exp();
                        self.adj_scalars[x.0] += a_s * (1.0 - z2);
                        self.adj_scalars[sigma.0] += a_s * ((z2 - 1.0) / scale);
                    }
                }
                Op::StudentTLogP { x, nu, mu, sigma } => {
                    let (dx, dsigma, dnu) = student_t_derivatives(
                        self.scalars[x.0],
                        self.scalars[nu.0],
                        self.scalars[mu.0],
                        self.scalars[sigma.0],
                    );
                    self.adj_scalars[x.0] += a_s * dx;
                    self.adj_scalars[mu.0] -= a_s * dx;
                    self.adj_scalars[sigma.0] += a_s * dsigma;
                    self.adj_scalars[nu.0] += a_s * dnu;
                }
                Op::PositiveSupport { .. } => {}
                Op::BernoulliLogP { x, p } => {
                    self.adj_scalars[p.0] +=
                        a_s * bernoulli_logp_dp(self.scalars[x.0], self.scalars[p.0]);
                }
                Op::PoissonLogP { x, lam } => {
                    self.adj_scalars[lam.0] +=
                        a_s * poisson_logp_dlam(self.scalars[x.0], self.scalars[lam.0]);
                }
                Op::LogGammaLogP { x, alpha, beta } => {
                    let raw = self.scalars[x.0];
                    let a = self.scalars[alpha.0];
                    let rate = self.scalars[beta.0];
                    if a.is_finite() && a > 0.0 && rate.is_finite() && rate > 0.0 {
                        let log_scaled = raw + rate.ln();
                        let scaled = log_scaled.exp();
                        self.adj_scalars[x.0] += a_s * (a - scaled);
                        self.adj_scalars[alpha.0] += a_s * (log_scaled - digamma(a));
                        self.adj_scalars[beta.0] += a_s * ((a - scaled) / rate);
                    }
                }
                Op::ObsLogP {
                    family,
                    linpred_vec,
                    aux,
                    obs_data_idx,
                } => {
                    let obs = &self.binding.observations[*obs_data_idx];
                    match family {
                        crate::graph::ObsFamily::Normal => {
                            let sigma_node = aux.expect("Normal obs logp requires sigma");
                            let sv = self.scalars[sigma_node.0];
                            if !scale_is_valid(sv) {
                                continue;
                            }

                            let mut dsigma = 0.0f64;

                            let mu_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };

                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let m = self.read_vec(linpred_vec.0, i);
                                let diff = (y - m) / sv;
                                if let Some(off) = mu_off {
                                    self.adj_vec_buf[off + i] += a_s * (diff / sv);
                                }
                                dsigma += (diff * diff - 1.0) / sv;
                            }
                            self.adj_scalars[sigma_node.0] += a_s * dsigma;
                        }
                        crate::graph::ObsFamily::BernoulliLogit => {
                            let eta_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                let grad = bernoulli_logit_grad(y, eta);
                                if let Some(off) = eta_off {
                                    self.adj_vec_buf[off + i] += a_s * grad;
                                }
                            }
                        }
                        crate::graph::ObsFamily::PoissonLog => {
                            let eta_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                let grad = y - eta.exp();
                                if let Some(off) = eta_off {
                                    self.adj_vec_buf[off + i] += a_s * grad;
                                }
                            }
                        }
                        crate::graph::ObsFamily::ExponentialLog => {
                            let eta_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                let grad = 1.0 - y * eta.exp();
                                if let Some(off) = eta_off {
                                    self.adj_vec_buf[off + i] += a_s * grad;
                                }
                            }
                        }
                        crate::graph::ObsFamily::LogNormal => {
                            let sigma_node = aux.expect("LogNormal obs logp requires sigma");
                            let sv = self.scalars[sigma_node.0];
                            if !scale_is_valid(sv) {
                                continue;
                            }

                            let mu_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            let mut dsigma = 0.0f64;
                            for (i, &observation) in obs.iter().take(vl).enumerate() {
                                let y = observation;
                                let ly = y.ln();
                                let m = self.read_vec(linpred_vec.0, i);
                                let d = (ly - m) / sv;
                                if let Some(off) = mu_off {
                                    self.adj_vec_buf[off + i] += a_s * (d / sv);
                                }
                                dsigma += (d * d - 1.0) / sv;
                            }
                            self.adj_scalars[sigma_node.0] += a_s * dsigma;
                        }
                        crate::graph::ObsFamily::NegativeBinomialLog => {
                            let alpha_node = aux.expect("NegativeBinomial obs logp requires alpha");
                            let av = self.scalars[alpha_node.0];
                            let eta_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            let mut dalpha = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                let (deta, da) = crate::negative_binomial::gradients(y, eta, av);
                                if let Some(off) = eta_off {
                                    self.adj_vec_buf[off + i] += a_s * deta;
                                }
                                dalpha += da;
                            }
                            self.adj_scalars[alpha_node.0] += a_s * dalpha;
                        }
                    }
                }
                Op::FusedLinearMu {
                    param_nodes,
                    data_indices,
                    intercept,
                } => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let adj = &self.adj_vec_buf[out_off..out_off + vl];
                    for (k, &pn) in param_nodes.iter().enumerate() {
                        let data = &self.binding.vectors[data_indices[k]];
                        let mut ds = 0.0f64;
                        ds += adj
                            .iter()
                            .zip(data.iter())
                            .take(vl)
                            .map(|(a, d)| a * d)
                            .sum::<f64>();
                        self.adj_scalars[pn.0] += ds;
                    }
                    if let Some(n) = intercept {
                        let mut ds = 0.0f64;
                        ds += adj.iter().take(vl).sum::<f64>();
                        self.adj_scalars[n.0] += ds;
                    }
                }
                Op::MatVecMul {
                    matrix_idx,
                    param_start,
                    n_params,
                    intercept,
                } => {
                    use faer::{col, linalg::matmul::matmul, mat, Parallelism};
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let matrix = &self.binding.matrices[*matrix_idx];
                    let x = mat::from_row_major_slice::<f64>(
                        &matrix.data,
                        matrix.n_rows,
                        matrix.n_cols,
                    );
                    let adj_slice = &self.adj_vec_buf[out_off..out_off + vl];
                    let adj_col = col::from_slice::<f64>(adj_slice);
                    let transforms =
                        &graph.param_transforms[*param_start..*param_start + *n_params];
                    let all_identity = transforms
                        .iter()
                        .all(|t| matches!(t, ParamTransform::Identity));
                    let par = if matrix.n_rows * matrix.n_cols >= 100_000 {
                        Parallelism::Rayon(0)
                    } else {
                        Parallelism::None
                    };
                    // grad += X^T @ adj
                    if all_identity {
                        let target = &mut self.grad[*param_start..*param_start + *n_params];
                        let grad_col = col::from_slice_mut::<f64>(target);
                        matmul(
                            grad_col.as_2d_mut(),
                            x.transpose(),
                            adj_col.as_2d(),
                            Some(1.0),
                            1.0,
                            par,
                        );
                    } else {
                        let target =
                            &mut self.param_scratch[*param_start..*param_start + *n_params];
                        target.fill(0.0);
                        let grad_col = col::from_slice_mut::<f64>(target);
                        matmul(
                            grad_col.as_2d_mut(),
                            x.transpose(),
                            adj_col.as_2d(),
                            Some(1.0),
                            1.0,
                            par,
                        );
                        for k in 0..*n_params {
                            self.grad[param_start + k] +=
                                target[k] * transforms[k].derivative(params[param_start + k]);
                        }
                    }
                    if let Some(n) = intercept {
                        let ds: f64 = adj_slice.iter().sum();
                        self.adj_scalars[n.0] += ds;
                    }
                }
                Op::VectorNormalLogP {
                    param_start,
                    n_params,
                    mu,
                    sigma,
                } => {
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        self.grad[param_start + k] += a_s * (-((v - mu) / sigma) / sigma);
                    }
                }
                Op::VectorHalfNormalLogP {
                    param_start,
                    n_params,
                    sigma,
                } => {
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        self.grad[param_start + k] +=
                            a_s * (1.0 - (2.0 * (raw - sigma.ln())).exp());
                    }
                }
                Op::VectorStudentTLogP {
                    param_start,
                    n_params,
                    nu,
                    mu,
                    sigma,
                } => {
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        self.grad[param_start + k] +=
                            a_s * student_t_derivatives(v, *nu, *mu, *sigma).0;
                    }
                }
                Op::VectorGammaLogP {
                    param_start,
                    n_params,
                    alpha,
                    beta,
                } => {
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        self.grad[param_start + k] += a_s * (alpha - (raw + beta.ln()).exp());
                    }
                }
                Op::VectorBetaLogP {
                    param_start,
                    n_params,
                    alpha,
                    beta,
                } => {
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        let s = sigmoid_stable(raw);
                        // d/draw = α·(1-s) - β·s
                        self.grad[param_start + k] +=
                            a_s * (alpha * sigmoid_stable(-raw) - beta * s);
                    }
                }
                Op::VectorUniformLogP {
                    param_start,
                    n_params,
                    lower,
                    upper,
                } => {
                    if !uniform_bounds_valid(*lower, *upper) {
                        continue;
                    }
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        let s = sigmoid_stable(raw);
                        // d/draw = 1 - 2s
                        self.grad[param_start + k] += a_s * (1.0 - 2.0 * s);
                    }
                }
            }
        }

        // Extract parameter gradients from adj_scalars for regular Param nodes;
        // vector-param gradients are already accumulated directly into self.grad.
        for (pidx, nid_opt) in self.param_node_ids.iter().enumerate() {
            if let Some(nid) = nid_opt {
                self.grad[pidx] += self.adj_scalars[*nid];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference evaluator — differential-testing oracle, not public API
// ---------------------------------------------------------------------------

/// Allocating re-implementation of the whole IR, used only to cross-check the
/// zero-allocation `Evaluator`. It is deliberately not exported: it has no
/// non-test callers and it panics on its entry points when a node's shape is
/// not what it expected.
#[cfg(test)]
#[path = "autodiff_reference.rs"]
mod reference;
#[cfg(test)]
pub(crate) use reference::{eval_logp, grad_logp};

/// Whether `sigma` is a usable scale: finite and strictly positive.
#[inline]
fn scale_is_valid(sigma: f64) -> bool {
    sigma.is_finite() && sigma > 0.0
}

fn normal_logp_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if !scale_is_valid(sigma) {
        return f64::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;
    -0.5 * std::f64::consts::TAU.ln() - sigma.ln() - 0.5 * z * z
}

// Whole-vector observation kernels. Only the reference evaluator uses these:
// the Evaluator fuses the same arithmetic into its own single pass.
#[cfg(test)]
mod obs_logp_sums {

    pub(super) fn normal_obs_logp_sum(mu: &[f64], sigma: f64, obs: &[f64]) -> f64 {
        if !super::scale_is_valid(sigma) {
            return f64::NEG_INFINITY;
        }
        let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
        let n = obs.len() as f64;
        let sum_sq: f64 = mu
            .iter()
            .zip(obs.iter())
            .map(|(m, o)| {
                let d = (o - m) / sigma;
                d * d
            })
            .sum();
        n * log_norm - 0.5 * sum_sq
    }

    pub(super) fn bernoulli_logit_obs_logp_sum(eta: &[f64], obs: &[f64]) -> f64 {
        eta.iter()
            .zip(obs.iter())
            .map(|(e, y)| super::bernoulli_logit_logp(*y, *e))
            .sum()
    }

    pub(super) fn poisson_log_obs_logp_sum(eta: &[f64], obs: &[f64]) -> f64 {
        eta.iter()
            .zip(obs.iter())
            .map(|(e, y)| crate::count_sampling::log_mass_from_log_rate(*y, *e))
            .sum()
    }

    pub(super) fn exponential_log_obs_logp_sum(eta: &[f64], obs: &[f64]) -> f64 {
        eta.iter()
            .zip(obs.iter())
            .map(|(e, y)| e - y * e.exp())
            .sum()
    }

    pub(super) fn log_normal_obs_logp_sum(mu: &[f64], sigma: f64, obs: &[f64]) -> f64 {
        if !super::scale_is_valid(sigma) {
            return f64::NEG_INFINITY;
        }
        let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
        mu.iter()
            .zip(obs.iter())
            .map(|(m, y)| {
                let ly = y.ln();
                let d = (ly - m) / sigma;
                log_norm - ly - 0.5 * d * d
            })
            .sum()
    }

    pub(super) fn negative_binomial_log_obs_logp_sum(eta: &[f64], alpha: f64, obs: &[f64]) -> f64 {
        eta.iter()
            .zip(obs)
            .map(|(&e, &y)| crate::negative_binomial::log_mass(y, e, alpha))
            .sum()
    }
}
#[cfg(test)]
use obs_logp_sums::*;

// Combined transformed densities avoid materializing exp(raw), and form
// scale ratios in log space before squaring or multiplying extreme values.
fn log_half_normal_logp(raw: f64, sigma: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = raw - sigma.ln();
    0.5 * (2.0 / std::f64::consts::PI).ln() + z - 0.5 * (2.0 * z).exp()
}

fn log_gamma_logp(raw: f64, alpha: f64, beta: f64) -> f64 {
    if !alpha.is_finite() || alpha <= 0.0 || !beta.is_finite() || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = raw + beta.ln();
    alpha * z - ln_gamma(alpha) - z.exp()
}

fn student_t_logp_scalar(x: f64, nu: f64, mu: f64, sigma: f64) -> f64 {
    if !nu.is_finite() || nu <= 0.0 || !sigma.is_finite() || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let log_ratio = 2.0 * ((x - mu).abs().ln() - sigma.ln()) - nu.ln();
    ln_gamma(0.5 * (nu + 1.0))
        - ln_gamma(0.5 * nu)
        - 0.5 * (nu.ln() + std::f64::consts::PI.ln())
        - sigma.ln()
        - 0.5 * (nu + 1.0) * softplus(log_ratio)
}

// Return derivatives with respect to x, sigma, and nu. Log-ratio arithmetic
// retains Student-t's polynomial tails even when the squared residual overflows.
fn student_t_derivatives(x: f64, nu: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
    let diff = x - mu;
    let log_abs_diff = diff.abs().ln();
    let log_ratio = 2.0 * (log_abs_diff - sigma.ln()) - nu.ln();
    let log_tail = softplus(log_ratio);
    let weight = sigmoid_stable(log_ratio);
    let dx = if diff == 0.0 {
        0.0
    } else {
        -diff.signum()
            * ((nu + 1.0).ln() + log_abs_diff - 2.0 * sigma.ln() - nu.ln() - log_tail).exp()
    };
    let scale_term = (nu + 1.0) * weight - 1.0;
    let dsigma = scale_term / sigma;
    let dnu = 0.5 * digamma(0.5 * (nu + 1.0)) - 0.5 * digamma(0.5 * nu) - 0.5 * log_tail
        + 0.5 * scale_term / nu;
    (dx, dsigma, dnu)
}

fn uniform_bounds_valid(lower: f64, upper: f64) -> bool {
    lower.is_finite() && upper.is_finite() && lower < upper && (upper - lower).is_finite()
}

/// Bernoulli log mass. `-inf` off the support, which is `{0, 1}`.
///
/// This is reachable: `model::GraphModel::log_density` evaluates it for a
/// loaded artifact, so it has to be the density it claims to be even though
/// gradient-based *sampling* of a discrete latent is refused elsewhere.
/// Without the support check `x = 0.5` had a finite density, and at `p = 0.5`
/// the "density" was constant over all of R.
///
/// There is no clamp on `p`, and that is the other half of the fix. Clamping to
/// `[1e-12, 1 - 1e-12]` gave the impossible outcome `x = 1, p = 0` a log mass of
/// about -27.6 — merely unlikely — and it moved every `p` outside that band. The
/// clamp existed to avoid `0 * ln(0)`, which is NaN where the limit is 0; that
/// is handled here by branching on `x` instead of multiplying by it, so the term
/// that would be multiplied by zero is never formed at all.
///
/// `ln_1p(-p)` rather than `(1 - p).ln()`: for `p` below about 1e-16 the
/// subtraction rounds to exactly 1 and the log to exactly 0, discarding the
/// whole of `-p`.
fn bernoulli_logp_scalar(x: f64, p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NEG_INFINITY;
    }
    if x == 1.0 {
        p.ln()
    } else if x == 0.0 {
        (-p).ln_1p()
    } else {
        f64::NEG_INFINITY
    }
}

/// d/dp of [`bernoulli_logp_scalar`], zero wherever that is a constant `-inf`.
///
/// A density that is `-inf` everywhere in a neighbourhood has no slope, and a
/// score that moves while the density does not is worse than no score: it sends
/// a sampler off in a direction the density does not support. At `p = 0` with
/// `x = 1` the slope is genuinely infinite, which is the limit from inside the
/// support and not a lost value.
///
/// `1 / p` overflows for every `p` below 5.6e-309, where `ln(p)` is still an
/// ordinary -710. The composition through whatever produced `p` would often be
/// representable — a `sigmoid` link makes it exactly 1 — but the adjoint at the
/// `p` node is `1 / p` whatever order the factors are taken in, so there is
/// nothing to reassociate: the intermediate itself is the unrepresentable
/// quantity. The clamp this replaced returned 1e12 there, finite and wrong by
/// 296 orders of magnitude; an infinity is refused at the sampler boundary
/// instead of being believed.
fn bernoulli_logp_dp(x: f64, p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return 0.0;
    }
    // `p == -0.0` is accepted by that range check, and `1.0 / -0.0` is negative
    // infinity — the wrong sign for a limit taken from inside `[0, 1]`, where
    // the density only approaches its endpoint from above. The density itself
    // does not distinguish the two zeros, so the score must not either.
    let p = p + 0.0;
    if x == 1.0 {
        1.0 / p
    } else if x == 0.0 {
        -1.0 / (1.0 - p)
    } else {
        0.0
    }
}

/// Poisson log mass, over the exactly representable count range.
///
/// [`crate::count_sampling::log_mass`] already refuses a negative, fractional or
/// nonfinite count and a negative or nonfinite rate, and treats `rate == 0` as
/// the point mass at zero. It carries no clamp of any kind, so unlike the
/// Bernoulli case there was nothing here to correct.
fn poisson_logp_scalar(x: f64, lam: f64) -> f64 {
    crate::count_sampling::log_mass(x, lam)
}

/// d/dlam of [`poisson_logp_scalar`], zero wherever that is a constant `-inf`.
///
/// `(x - lam) / lam` is the score on the support. Not `x / lam - 1`: the
/// quotient rounds to something near 1 and the subtraction then cancels away
/// most of what is left, which is precisely the region a count model lives in.
/// At `x = 1` and `lam` one ulp below it the old form returned
/// 2.220446049250313e-16 for a score of 1.1102230246251568e-16 — twice the
/// right answer — and at `x = 1e14, lam = x + 1` it was 0.08% off. `x - lam` is
/// exact whenever the two are within a factor of two of each other, by
/// Sterbenz, so the new form has one rounding and no cancellation.
///
/// Off the support the mass is `-inf` and the score is zero; at `x == 0` the
/// mass is `-lam` and the score is `-1`, which the general form would compute
/// as `-0 / 0` when `lam` is also zero.
fn poisson_logp_dlam(x: f64, lam: f64) -> f64 {
    if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || !lam.is_finite() || lam < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return -1.0;
    }
    // As in `bernoulli_logp_dp`: `lam == -0.0` passes `lam < 0.0` and would give
    // a negative infinity for a limit that is positive. `count_sampling::log_mass`
    // treats both zeros identically through its `rate == 0.0` branch.
    (x - lam) / (lam + 0.0)
}

pub(crate) fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// log P(y | eta) for a Bernoulli observation with logit `eta`.
///
/// `y * eta - softplus(eta)` is the same quantity and cancels catastrophically
/// once `eta` saturates: at `y = 1, eta = 40` softplus returns `40` to the last
/// bit, the subtraction gives exactly `0.0`, and the true value is
/// `-4.2483542552915888e-18`. Writing each branch as a single softplus of the
/// sign that does not cancel keeps the tail. `observation.rs` already did this;
/// the graph evaluator and its reference did not, so one family had two
/// numerically different implementations and nothing compared them.
///
/// `y` outside `{0, 1}` keeps the general expression rather than silently
/// snapping to a branch; the family rejects such an observation on validation.
pub(crate) fn bernoulli_logit_logp(y: f64, eta: f64) -> f64 {
    if y == 1.0 {
        -softplus(-eta)
    } else if y == 0.0 {
        -softplus(eta)
    } else {
        y * eta - softplus(eta)
    }
}

/// d/d(eta) of [`bernoulli_logit_logp`], which is `y - sigmoid(eta)`.
///
/// That difference cancels for the same reason: `sigmoid(40)` rounds to exactly
/// one, so `1 - sigmoid(40)` is `0.0` where the true derivative is
/// `4.2483542552915888e-18`. A saturated observation then contributes no
/// gradient at all, and a predictor with a large scale multiplies that zero
/// instead of a small number -- at `eta = 40 + 1e20 * b` the gradient in `b` is
/// `0` rather than about `424.8`. `sigmoid(-eta)` is the same value without the
/// subtraction.
pub(crate) fn bernoulli_logit_grad(y: f64, eta: f64) -> f64 {
    if y == 1.0 {
        sigmoid_stable(-eta)
    } else if y == 0.0 {
        -sigmoid_stable(eta)
    } else {
        y - sigmoid_stable(eta)
    }
}

/// Lanczos approximation to ln(Γ(x)) for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    let y = x;
    let tmp = y + 5.5;
    let tmp = tmp - (y + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015f64;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + i as f64);
    }
    (std::f64::consts::TAU.sqrt() * ser / y).ln() - tmp
}

/// Digamma function ψ(x) = d/dx ln(Γ(x)), via asymptotic series + recurrence.
fn digamma(mut x: f64) -> f64 {
    // The recurrence below steps x up by one until it reaches 8, which never
    // terminates for -inf or for x <= -2^53 (where x + 1 == x) and takes |x|
    // steps for any large negative x. Poles and non-finite arguments have no
    // value; other negative arguments go through the reflection formula
    // psi(x) = psi(1 - x) - pi / tan(pi x).
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        return digamma(1.0 - x) - std::f64::consts::PI / (std::f64::consts::PI * x).tan();
    }
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion for large x
    result += x.ln() - 0.5 / x;
    let x2 = 1.0 / (x * x);
    result -= x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, ObsFamily};

    #[test]
    fn positive_support_retains_finite_positive_scale_domain() {
        let mut graph = Graph::new();
        let scale = graph.add_param("scale");
        graph.positive_support(scale);
        for x in [
            -1.0,
            -0.0,
            0.0,
            1e-300,
            0.5,
            1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let expected = if x.is_finite() && x > 0.0 {
                0.0
            } else {
                f64::NEG_INFINITY
            };
            let mut evaluator = Evaluator::new(&graph);
            evaluator.compute(&graph, &[x]);
            assert_eq!(evaluator.total_logp, expected);
            assert_eq!(evaluator.grad, vec![0.0]);
            assert_eq!(grad_logp(&graph, &[x]), (expected, vec![0.0]));
        }
    }

    #[test]
    fn digamma_terminates_and_reflects_for_negative_arguments() {
        let euler = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + euler).abs() < 1e-9);
        // psi(-2.5) = psi(3.5) - pi / tan(-2.5 pi) = psi(3.5), since
        // tan(-2.5 pi) is infinite: 1.1031566406452432.
        assert!((digamma(-2.5) - 1.103_156_640_645_243).abs() < 1e-9);
        // psi(-0.5) = 0.03648997397857652.
        assert!((digamma(-0.5) - 0.036_489_973_978_576_5).abs() < 1e-9);
        // These used to loop forever: x + 1 == x below -2^53, and -inf + 1
        // is -inf.
        for pole in [0.0, -1.0, -1e300, f64::MIN, f64::NEG_INFINITY, f64::NAN] {
            assert!(digamma(pole).is_nan(), "{pole}");
        }
        assert_eq!(digamma(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn observation_scales_outside_their_support_give_zero_density() {
        for (lognormal, sigma) in [(false, 0.0), (false, -1.0), (true, 0.0), (true, -2.0)] {
            let mut g = Graph::new();
            let s = g.add_param("s");
            let mu = g.add_constant(1.0);
            let x = g.add_data("x", vec![1.0, 1.0]);
            let linpred = g.scalar_mul_data(mu, x);
            let obs = g.add_obs_data(vec![0.5, 2.0]);
            if lognormal {
                g.obs_logp_lognormal(linpred, s, obs);
            } else {
                g.obs_logp_normal(linpred, s, obs);
            }
            let mut evaluator = Evaluator::new(&g);
            evaluator.compute(&g, &[sigma]);
            // Previously ln(sigma) made this NaN, where the scalar Normal
            // density already returned -inf for the same scale.
            assert_eq!(evaluator.total_logp, f64::NEG_INFINITY);
            assert!(evaluator.grad.iter().all(|g| g.is_finite()));
            assert_eq!(eval_logp(&g, &[sigma]), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        let cases = [
            (0.5, 0.5 * std::f64::consts::PI.ln()),
            (1.0, 0.0),
            (5.0, 24.0_f64.ln()),
            (10.0, 362_880.0_f64.ln()),
        ];

        for (x, expected) in cases {
            let actual = ln_gamma(x);
            assert!(
                (actual - expected).abs() < 1e-10,
                "ln_gamma({x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_normal_logp_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let mu = g.add_constant(0.0);
        let sigma = g.add_constant(1.0);
        g.normal_logp(x, mu, sigma);

        let params = vec![1.5];
        let (logp, grad) = grad_logp(&g, &params);
        assert!((logp - (-0.5 * 1.5_f64.powi(2) - 0.5 * std::f64::consts::TAU.ln())).abs() < 1e-10);
        assert!((grad[0] - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn forward_only_evaluation_matches_the_values_of_a_full_pass() {
        let mut g = Graph::new();
        let beta = g.add_param("beta");
        let zero = g.add_constant(0.0);
        let one = g.add_constant(1.0);
        g.normal_logp(beta, zero, one);
        let x_data = g.add_data("x", vec![1.0, 2.0, 3.0]);
        let mu = g.scalar_mul_data(beta, x_data);
        let obs = g.add_obs_data(vec![2.5, 5.0, 7.5]);
        g.normal_obs_logp(mu, one, obs);

        let mut full = Evaluator::new(&g);
        full.compute(&g, &[0.7]);
        let mut forward = Evaluator::new(&g);
        forward.forward(&g, &[0.7]);
        assert_eq!(forward.total_logp, full.total_logp);
        for i in 0..3 {
            assert_eq!(forward.vec_elem(mu, i, &g), full.vec_elem(mu, i, &g));
        }
        // No reverse pass ran, so the gradient buffer is untouched.
        assert_eq!(forward.grad, vec![0.0]);
        assert_ne!(full.grad, vec![0.0]);
    }

    #[test]
    fn test_gradient_finite_diff() {
        let mut g = Graph::new();
        let beta = g.add_param("beta");
        let mu_const = g.add_constant(0.0);
        let sigma = g.add_constant(1.0);
        g.normal_logp(beta, mu_const, sigma);

        let x_data = g.add_data("x", vec![1.0, 2.0, 3.0]);
        let mu_vec = g.scalar_mul_data(beta, x_data);
        let obs_idx = g.add_obs_data(vec![2.5, 5.0, 7.5]);
        g.normal_obs_logp(mu_vec, sigma, obs_idx);

        let params = vec![2.4];
        let (_, grad) = grad_logp(&g, &params);

        let eps = 1e-6;
        let num =
            (eval_logp(&g, &[params[0] + eps]) - eval_logp(&g, &[params[0] - eps])) / (2.0 * eps);
        assert!(
            (grad[0] - num).abs() < 1e-4,
            "analytic={}, numerical={}",
            grad[0],
            num
        );
    }

    #[test]
    fn test_multivariate_gradient() {
        let mut g = Graph::new();
        let b0 = g.add_param("b0");
        let b1 = g.add_param("b1");
        let b2 = g.add_param("b2");
        let sigma = g.add_constant(1.0);

        let mu0 = g.add_constant(0.0);
        g.normal_logp(b0, mu0, sigma);
        g.normal_logp(b1, mu0, sigma);
        g.normal_logp(b2, mu0, sigma);

        let x1 = g.add_data("x1", vec![1.0, 2.0, 3.0, 4.0]);
        let x2 = g.add_data("x2", vec![0.5, 1.5, 2.5, 3.5]);

        let v1 = g.scalar_mul_data(b1, x1);
        let v2 = g.scalar_mul_data(b2, x2);
        let v12 = g.vector_add(v1, v2);
        let mu_vec = g.scalar_broadcast_add(b0, v12);

        let obs_idx = g.add_obs_data(vec![3.0, 7.0, 11.0, 15.0]);
        g.normal_obs_logp(mu_vec, sigma, obs_idx);

        let params = vec![0.5, 1.8, 1.2];
        let (_, grad) = grad_logp(&g, &params);

        let eps = 1e-6;
        for i in 0..3 {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let num = (eval_logp(&g, &p_plus) - eval_logp(&g, &p_minus)) / (2.0 * eps);
            assert!(
                (grad[i] - num).abs() < 1e-4,
                "param {}: analytic={}, numerical={}",
                i,
                grad[i],
                num
            );
        }
    }

    #[test]
    fn test_evaluator_matches_grad_logp() {
        let mut g = Graph::new();
        let b0 = g.add_param("b0");
        let b1 = g.add_param("b1");
        let b2 = g.add_param("b2");
        let sigma = g.add_constant(1.0);
        let mu0 = g.add_constant(0.0);
        g.normal_logp(b0, mu0, sigma);
        g.normal_logp(b1, mu0, sigma);
        g.normal_logp(b2, mu0, sigma);

        let x1 = g.add_data("x1", vec![1.0, 2.0, 3.0, 4.0]);
        let x2 = g.add_data("x2", vec![0.5, 1.5, 2.5, 3.5]);
        let v1 = g.scalar_mul_data(b1, x1);
        let v2 = g.scalar_mul_data(b2, x2);
        let v12 = g.vector_add(v1, v2);
        let mu_vec = g.scalar_broadcast_add(b0, v12);
        let obs_idx = g.add_obs_data(vec![3.0, 7.0, 11.0, 15.0]);
        g.normal_obs_logp(mu_vec, sigma, obs_idx);

        let params = vec![0.5, 1.8, 1.2];
        let (logp_old, grad_old) = grad_logp(&g, &params);

        let mut eval = Evaluator::new(&g);
        eval.compute(&g, &params);

        assert!(
            (eval.total_logp - logp_old).abs() < 1e-10,
            "logp mismatch: {} vs {}",
            eval.total_logp,
            logp_old
        );
        for (i, &reference) in grad_old.iter().enumerate().take(3) {
            assert!(
                (eval.grad[i] - reference).abs() < 1e-10,
                "grad[{}] mismatch: {} vs {}",
                i,
                eval.grad[i],
                reference
            );
        }
    }

    fn finite_diff_check(g: &Graph, params: &[f64], tol: f64) {
        let (_, grad) = grad_logp(g, params);
        let eps = 1e-6;
        for i in 0..params.len() {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let num = (eval_logp(g, &p_plus) - eval_logp(g, &p_minus)) / (2.0 * eps);
            assert!(
                (grad[i] - num).abs() < tol,
                "param {}: analytic={}, numerical={}, diff={}",
                i,
                grad[i],
                num,
                (grad[i] - num).abs()
            );
        }
    }

    #[test]
    fn test_student_t_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let nu = g.add_constant(4.0);
        let mu = g.add_constant(1.0);
        let sigma = g.add_constant(2.0);
        g.student_t_logp(x, nu, mu, sigma);
        finite_diff_check(&g, &[1.8], 1e-4);
    }

    #[test]
    fn test_poisson_gradient() {
        let mut g = Graph::new();
        let lam = g.add_param("lam");
        let x = g.add_constant(5.0);
        g.poisson_logp(x, lam);
        finite_diff_check(&g, &[3.0], 1e-4);
    }

    #[test]
    fn test_bernoulli_gradient() {
        let mut g = Graph::new();
        let p = g.add_param("p");
        let x = g.add_constant(1.0);
        g.bernoulli_logp(x, p);
        finite_diff_check(&g, &[0.7], 1e-4);
    }

    #[test]
    fn test_bernoulli_logit_obs_gradient() {
        let mut g = Graph::new();
        let eta = g.add_param("eta");
        let eta_vec = g.scalar_broadcast(eta);
        let obs_idx = g.add_obs_data(vec![1.0, 0.0, 1.0, 1.0]);
        g.obs_logp_bernoulli_logit(eta_vec, obs_idx);

        let heads = g.observation_heads();
        assert_eq!(heads.len(), 1);
        assert_eq!(heads[0].family, ObsFamily::BernoulliLogit);
        assert_eq!(heads[0].n_obs, 4);

        let structure_heads = g.structure_only().observation_heads();
        assert_eq!(structure_heads.len(), 1);
        assert_eq!(structure_heads[0].n_obs, 0);

        finite_diff_check(&g, &[0.3], 1e-4);
    }

    #[test]
    fn negative_binomial_logp_and_gradient_are_consistent() {
        let mut g = Graph::new();
        let eta = g.add_param("eta");
        let alpha = g.add_param("alpha");
        let eta_vec = g.scalar_broadcast(eta);
        let obs_idx = g.add_obs_data(vec![0.0, 1.0, 4.0, 9.0]);
        g.obs_logp_negative_binomial_log(eta_vec, alpha, obs_idx);

        // The former ln_gamma sign error changed the density without changing
        // digamma, so finite differences specifically failed for alpha.
        full_finite_diff_check(&g, &[1.0, 3.0], 2e-5);
    }

    /// Test MatVecMul + VectorNormalLogP forward and gradient via finite differences.
    /// Model: 10 observations, 5 beta parameters, 1 scalar intercept.
    /// Prior: beta ~ Normal(0,1) via VectorNormalLogP
    ///        intercept ~ Normal(0,10) via NormalLogP
    /// Likelihood: Normal(intercept + X @ beta, sigma=1)
    #[test]
    fn test_mat_vec_mul_gradient() {
        // 10 obs × 5 params
        #[rustfmt::skip]
        let x_data: Vec<f64> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.7, 1.8, 1.9, 2.0,
            2.1, 2.2, 2.3, 2.4, 2.5,
            2.6, 2.7, 2.8, 2.9, 3.0,
            3.1, 3.2, 3.3, 3.4, 3.5,
            3.6, 3.7, 3.8, 3.9, 4.0,
            4.1, 4.2, 4.3, 4.4, 4.5,
            4.6, 4.7, 4.8, 4.9, 5.0,
        ];
        let obs: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];

        let mut g = Graph::new();

        // intercept: scalar param, Normal(0,10) prior
        let intercept = g.add_param("intercept");
        let mu0 = g.add_constant(0.0);
        let sigma10 = g.add_constant(10.0);
        g.normal_logp(intercept, mu0, sigma10);

        // beta: vector params, VectorNormal(0,1) prior
        let param_start = g.add_vector_params("beta", 5);
        g.vector_normal_logp(param_start, 5, 0.0, 1.0);

        // mu = intercept + X @ beta
        let matrix_idx = g.store_matrix(x_data, 10, 5);
        let mu_node = g.mat_vec_mul(matrix_idx, param_start, 5, Some(intercept));

        // likelihood
        let sigma1 = g.add_constant(1.0);
        let obs_idx = g.add_obs_data(obs);
        g.normal_obs_logp(mu_node, sigma1, obs_idx);

        // params = [intercept, beta[0..5]] = 6 total
        let params = vec![0.5, 0.1, 0.2, -0.1, 0.3, -0.2];

        // Check that Evaluator and grad_logp agree
        let (logp_ref, grad_ref) = grad_logp(&g, &params);

        let mut eval = Evaluator::new(&g);
        eval.compute(&g, &params);

        assert!(
            (eval.total_logp - logp_ref).abs() < 1e-8,
            "logp mismatch: Evaluator={} grad_logp={}",
            eval.total_logp,
            logp_ref
        );
        for (i, &reference) in grad_ref.iter().enumerate().take(params.len()) {
            assert!(
                (eval.grad[i] - reference).abs() < 1e-8,
                "grad[{}] mismatch: Evaluator={} grad_logp={}",
                i,
                eval.grad[i],
                reference
            );
        }

        // Finite-difference check on grad_logp
        finite_diff_check(&g, &params, 1e-4);

        // Also check Evaluator gradient with finite differences on eval_logp
        let eps = 1e-6;
        for i in 0..params.len() {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let num = (eval_logp(&g, &p_plus) - eval_logp(&g, &p_minus)) / (2.0 * eps);
            assert!(
                (eval.grad[i] - num).abs() < 1e-4,
                "Evaluator grad[{}]: analytic={}, numerical={}",
                i,
                eval.grad[i],
                num
            );
        }
    }

    #[test]
    fn test_mat_vec_mul_applies_constraint_and_chain_rule() {
        use crate::graph::ParamTransform;

        let transforms = [
            ParamTransform::Identity,
            ParamTransform::Exp,
            ParamTransform::Sigmoid,
            ParamTransform::BoundedSigmoid {
                lower: -2.0,
                upper: 3.0,
            },
        ];

        for transform in transforms {
            let mut g = Graph::new();
            let start = g.add_vector_params_with_transform("beta", 2, transform.clone());
            let matrix_idx = g.store_matrix(vec![1.0, 2.0, -0.5, 3.0, 4.0, -1.0], 3, 2);
            let mu = g.mat_vec_mul(matrix_idx, start, 2, None);
            let sigma = g.add_constant(1.3);
            let obs_idx = g.add_obs_data(vec![0.5, -1.0, 2.0]);
            g.normal_obs_logp(mu, sigma, obs_idx);

            let params = vec![-0.4, 0.7];
            full_finite_diff_check(&g, &params, 2e-5);

            let mut evaluator = Evaluator::new(&g);
            evaluator.compute(&g, &params);
            let beta0 = transform.apply(params[0]);
            let beta1 = transform.apply(params[1]);
            let expected = [
                beta0 + 2.0 * beta1,
                -0.5 * beta0 + 3.0 * beta1,
                4.0 * beta0 - beta1,
            ];
            for (i, expected_value) in expected.into_iter().enumerate() {
                assert!(
                    (evaluator.vec_elem(mu, i, &g) - expected_value).abs() < 1e-12,
                    "transform {:?}, row {}",
                    transform,
                    i
                );
            }
        }
    }

    /// Helper: check both grad_logp and Evaluator against finite differences.
    fn full_finite_diff_check(g: &Graph, params: &[f64], tol: f64) {
        // Check free-standing grad_logp
        finite_diff_check(g, params, tol);

        // Check Evaluator
        let mut eval = Evaluator::new(g);
        eval.compute(g, params);
        let (logp_ref, grad_ref) = grad_logp(g, params);
        assert!(
            (eval.total_logp - logp_ref).abs() < 1e-8,
            "logp mismatch: Evaluator={} grad_logp={}",
            eval.total_logp,
            logp_ref
        );
        let eps = 1e-6;
        for i in 0..params.len() {
            assert!(
                (eval.grad[i] - grad_ref[i]).abs() < 1e-8,
                "grad[{}] mismatch: Evaluator={} grad_logp={}",
                i,
                eval.grad[i],
                grad_ref[i]
            );
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let num = (eval_logp(g, &p_plus) - eval_logp(g, &p_minus)) / (2.0 * eps);
            assert!(
                (eval.grad[i] - num).abs() < tol,
                "Evaluator grad[{}]: analytic={}, numerical={}",
                i,
                eval.grad[i],
                num
            );
        }
    }

    #[test]
    fn test_vector_half_normal_logp() {
        use crate::graph::ParamTransform;
        let mut g = Graph::new();
        let param_start = g.add_vector_params_with_transform("x", 3, ParamTransform::Exp);
        g.vector_half_normal_logp(param_start, 3, 2.0);
        // raw values (unconstrained); exp(raw) > 0 always
        let params = vec![0.5, -0.3, 1.2];
        full_finite_diff_check(&g, &params, 1e-4);
    }

    #[test]
    fn test_vector_student_t_logp() {
        let mut g = Graph::new();
        let param_start = g.add_vector_params("x", 3);
        g.vector_student_t_logp(param_start, 3, 4.0, 1.0, 2.0);
        let params = vec![0.5, -0.3, 1.8];
        full_finite_diff_check(&g, &params, 1e-4);
    }

    #[test]
    fn test_vector_gamma_logp() {
        use crate::graph::ParamTransform;
        let mut g = Graph::new();
        let param_start = g.add_vector_params_with_transform("x", 3, ParamTransform::Exp);
        g.vector_gamma_logp(param_start, 3, 2.0, 1.5);
        let params = vec![0.5, -0.3, 1.2];
        full_finite_diff_check(&g, &params, 1e-4);
    }

    #[test]
    fn test_vector_beta_logp() {
        use crate::graph::ParamTransform;
        let mut g = Graph::new();
        let param_start = g.add_vector_params_with_transform("x", 3, ParamTransform::Sigmoid);
        g.vector_beta_logp(param_start, 3, 2.0, 5.0);
        let params = vec![0.5, -0.3, 1.2];
        full_finite_diff_check(&g, &params, 1e-4);
    }

    #[test]
    fn test_vector_uniform_logp() {
        use crate::graph::ParamTransform;
        let mut g = Graph::new();
        let param_start = g.add_vector_params_with_transform(
            "x",
            3,
            ParamTransform::BoundedSigmoid {
                lower: 0.0,
                upper: 1.0,
            },
        );
        g.vector_uniform_logp(param_start, 3, 0.0, 1.0);
        let params = vec![0.5, -0.3, 1.2];
        full_finite_diff_check(&g, &params, 1e-4);
    }
}

#[cfg(test)]
mod expression_tests {
    use super::*;
    use crate::graph::ElementwiseOp as E;
    #[test]
    fn grouped_ragged_gradients_match_reference_and_finite_differences() {
        let mut g = Graph::new();
        let start = g.add_vector_params("beta", 2);
        g.vector_normal_logp(start, 2, 0.0, 1.0);
        let indices = g.add_data("group", vec![0.0, 1.0, 0.0]);
        let grouped = g.gather(start, 2, indices);
        let mu = g.elementwise(E::Tanh, grouped, None);
        let y = g.add_named_obs_data("y", "amount", vec![1.0, 2.0, 3.0]);
        let sigma = g.add_constant(1.0);
        g.normal_obs_logp(mu, sigma, y);
        let z = g.add_named_obs_data("z", "other", vec![0.0; 5]);
        g.schema.observations[1].dim = "other".into();
        let c = g.add_constant(0.3);
        let eta = g.broadcast_observation(c, z);
        g.obs_logp_bernoulli_logit(eta, z);
        let squared = g.elementwise(E::Mul, grouped, Some(grouped));
        let sum = g.sum(squared);
        let weight = g.add_constant(-0.1);
        let potential = g.mul(sum, weight);
        g.add_logp_term(potential);
        let q = vec![0.2, -0.4];
        let mut eval = Evaluator::new(&g);
        eval.compute(&g, &q);
        let (lp, grad) = grad_logp(&g, &q);
        assert!((lp - eval.total_logp).abs() < 1e-10);
        for i in 0..2 {
            assert!((grad[i] - eval.grad[i]).abs() < 1e-10);
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            let fd = (eval_logp(&g, &plus) - eval_logp(&g, &minus)) / 2e-6;
            assert!((fd - eval.grad[i]).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
mod tail_and_inactive_regressions {
    use super::*;
    use crate::distributions::{BetaDist, Normal, Uniform};
    use crate::graph::ElementwiseOp;

    #[test]
    fn scalar_and_vector_bounded_priors_preserve_raw_tails() {
        for (alpha, beta) in [(0.01, 0.01), (0.3, 2.0), (1.0, 1.0)] {
            for uniform in [false, true] {
                let mut scalar = Graph::new();
                // Include a preceding parameter so raw parameter/node indices differ.
                Normal::prior(&mut scalar, "offset", 0.0, 1.0);
                if uniform {
                    Uniform::prior(&mut scalar, "x", -3.0, 7.0);
                } else {
                    BetaDist::prior(&mut scalar, "x", alpha, beta);
                }
                let mut vector = Graph::new();
                Normal::prior(&mut vector, "offset", 0.0, 1.0);
                let start = vector.add_vector_params_with_transform(
                    "x",
                    1,
                    if uniform {
                        ParamTransform::BoundedSigmoid {
                            lower: -3.0,
                            upper: 7.0,
                        }
                    } else {
                        ParamTransform::Sigmoid
                    },
                );
                if uniform {
                    vector.vector_uniform_logp(start, 1, -3.0, 7.0);
                } else {
                    vector.vector_beta_logp(start, 1, alpha, beta);
                }
                for raw in [-1000.0_f64, -100.0, -40.0, 0.0, 40.0, 100.0, 1000.0] {
                    let (a, b) = if uniform { (1.0, 1.0) } else { (alpha, beta) };
                    let norm = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
                    // Analytic expression based on |raw|, independent of sigmoid rounding.
                    let expected = -0.5 * std::f64::consts::TAU.ln() + norm
                        - a * (-raw).max(0.0)
                        - b * raw.max(0.0)
                        - (a + b) * (-raw.abs()).exp().ln_1p();
                    let expected_grad = if raw >= 0.0 {
                        let e = (-raw).exp();
                        (a * e - b) / (1.0 + e)
                    } else {
                        let e = raw.exp();
                        (a - b * e) / (1.0 + e)
                    };
                    for graph in [&scalar, &vector] {
                        let mut evaluator = Evaluator::new(graph);
                        evaluator.compute(graph, &[0.0, raw]);
                        let reference = grad_logp(graph, &[0.0, raw]);
                        assert!((evaluator.total_logp - expected).abs() < 1e-10);
                        assert!((evaluator.grad[1] - expected_grad).abs() < 1e-12);
                        assert!((reference.0 - expected).abs() < 1e-10);
                        assert!((reference.1[1] - expected_grad).abs() < 1e-12);
                        let h = 1e-3;
                        let numeric = (eval_logp(graph, &[0.0, raw + h])
                            - eval_logp(graph, &[0.0, raw - h]))
                            / (2.0 * h);
                        assert!((numeric - expected_grad).abs() < 1e-7);
                    }
                }
            }
        }
    }

    #[test]
    fn inactive_vector_elements_do_not_propagate_singular_derivatives() {
        let mut graph = Graph::new();
        let start = graph.add_vector_params("x", 2);
        graph.vector_normal_logp(start, 2, 0.0, 1.0);
        let indices = graph.add_data("indices", vec![0.0, 1.0]);
        let x = graph.gather(start, 2, indices);
        let squared = graph.elementwise(ElementwiseOp::Mul, x, Some(x));
        let abs = graph.elementwise(ElementwiseOp::Sqrt, squared, None);
        let mask = graph.add_data("mask", vec![0.0, 1.0]);
        let active_abs = graph.elementwise(ElementwiseOp::Mul, abs, Some(mask));
        let potential = graph.sum(active_abs);
        graph.add_logp_term(potential);
        let params = [0.0, 0.4];
        let mut evaluator = Evaluator::new(&graph);
        evaluator.compute(&graph, &params);
        let reference = grad_logp(&graph, &params);
        assert!(evaluator.total_logp.is_finite());
        assert_eq!(evaluator.grad, vec![0.0, 0.6]);
        assert_eq!(reference, (evaluator.total_logp, evaluator.grad));
    }

    #[test]
    fn output_only_singular_derivatives_do_not_change_target() {
        for vector in [false, true] {
            let mut graph = Graph::new();
            let x = if vector {
                let start = graph.add_vector_params("x", 2);
                graph.vector_normal_logp(start, 2, 0.0, 1.0);
                let indices = graph.add_data("indices", vec![0.0, 1.0]);
                graph.gather(start, 2, indices)
            } else {
                Normal::prior(&mut graph, "x", 0.0, 1.0)
            };
            let params = vec![0.0; graph.param_count];
            let baseline = grad_logp(&graph, &params);
            let squared = graph.elementwise(ElementwiseOp::Mul, x, Some(x));
            let abs = graph.elementwise(ElementwiseOp::Sqrt, squared, None);
            graph.deterministics.push(("abs_x".into(), abs));
            let mut evaluator = Evaluator::new(&graph);
            evaluator.compute(&graph, &params);
            assert_eq!(evaluator.total_logp, baseline.0);
            assert_eq!(evaluator.grad, baseline.1);
            assert_eq!(grad_logp(&graph, &params), baseline);
            // Reused buffers must also remain unaffected away from the singularity.
            evaluator.compute(&graph, &vec![0.4; graph.param_count]);
            evaluator.compute(&graph, &params);
            assert_eq!(evaluator.grad, baseline.1);
        }
    }
}

#[cfg(test)]
mod extreme_scale_regressions {
    use super::*;
    use crate::distributions::{Exponential, Gamma, HalfNormal, Normal};

    /// The reference evaluator must agree with the `Evaluator` where only the
    /// *composed* adjoint is representable.
    ///
    /// Every other differential test runs at ordinary scales, where the two agree
    /// whether the oracle composes or multiplies afterwards. That is why the
    /// oracle was able to drift: it kept `derivatives(..)` then `* upstream`
    /// after the `Evaluator` moved to `adjoints(..)`, and nothing noticed.
    ///
    /// The target is `-(1/b) * scale`, so the division's upstream adjoint is
    /// `-scale` rather than 1. `derivatives` alone cannot rescue this: `-1/b^2`
    /// overflows for a small `b` however it is associated, and only folding
    /// `scale` in keeps the product in range.
    #[test]
    fn the_reference_evaluator_composes_adjoints_like_the_evaluator() {
        for (scale, b, expected) in [
            (1e-200, 1e-200, 1e200),
            (1e-160, 1e-180, 1e200),
            (1e200, 1e200, 1e-200),
        ] {
            let mut graph = Graph::new();
            let param = graph.add_param("b");
            let one = graph.add_constant(1.0);
            let ratio = graph.elementwise(crate::graph::ElementwiseOp::Div, one, Some(param));
            let scale_node = graph.add_constant(scale);
            let scaled =
                graph.elementwise(crate::graph::ElementwiseOp::Mul, ratio, Some(scale_node));
            let term = graph.elementwise(crate::graph::ElementwiseOp::Neg, scaled, None);
            graph.add_node_as_logp(term);

            let mut evaluator = Evaluator::new(&graph);
            evaluator.compute(&graph, &[b]);
            let (reference_logp, reference_grad) = grad_logp(&graph, &[b]);

            assert_eq!(
                evaluator.total_logp, reference_logp,
                "log density disagrees at scale {scale}, b {b}"
            );
            assert_eq!(
                evaluator.grad[0], reference_grad[0],
                "gradient disagrees at scale {scale}, b {b}: evaluator {} vs reference {}",
                evaluator.grad[0], reference_grad[0]
            );
            let error = (evaluator.grad[0] - expected).abs() / expected;
            assert!(
                error < 1e-9,
                "gradient at scale {scale}, b {b} is {} not {expected}",
                evaluator.grad[0]
            );
        }
    }

    fn check(graph: &Graph, params: &[f64], logp: f64, gradients: &[f64]) {
        let mut evaluator = Evaluator::new(graph);
        evaluator.compute(graph, params);
        let reference = grad_logp(graph, params);
        for (lp, grads) in [
            (evaluator.total_logp, &evaluator.grad),
            (reference.0, &reference.1),
        ] {
            assert!(
                (lp - logp).abs() < 1e-9 * (1.0 + logp.abs()),
                "logp {lp} != {logp}"
            );
            for (actual, expected) in grads.iter().zip(gradients) {
                assert!(actual.is_finite());
                let tolerance = 1e-9 * expected.abs().max(1e-300);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "gradient {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn gamma_and_exponential_raw_tails_match_analytic_density() {
        for alpha in [0.001, 1.0, 3.0] {
            for (raw, beta) in [(-1000.0, 1.0), (-720.0, 1.0), (0.3, 2.0), (710.0, 1e-300)] {
                let mut scalar = Graph::new();
                Gamma::prior(&mut scalar, "x", alpha, beta);
                let mut vector = Graph::new();
                let start = vector.add_vector_params_with_transform("x", 1, ParamTransform::Exp);
                vector.vector_gamma_logp(start, 1, alpha, beta);
                let z = raw + beta.ln();
                let expected = alpha * z - ln_gamma(alpha) - z.exp();
                for graph in [&scalar, &vector] {
                    check(graph, &[raw], expected, &[alpha - z.exp()]);
                }
                if alpha == 1.0 {
                    let mut exponential = Graph::new();
                    Exponential::prior(&mut exponential, "x", beta);
                    check(&exponential, &[raw], expected, &[1.0 - z.exp()]);
                }
            }
        }
    }

    #[test]
    fn half_normal_scaled_tails_preserve_hierarchical_gradients() {
        for sigma in [1e-200_f64, 1.0, 1e200] {
            for ratio in [0.5_f64, 2.0] {
                let raw = sigma.ln() + ratio.ln();
                // Use the actual representable raw log-ratio in the oracle.
                let z = raw - sigma.ln();
                let squared_ratio = z.exp().powi(2);
                let lp = 0.5 * (2.0 / std::f64::consts::PI).ln() + z - 0.5 * squared_ratio;
                let mut scalar = Graph::new();
                HalfNormal::prior(&mut scalar, "x", sigma);
                check(&scalar, &[raw], lp, &[1.0 - squared_ratio]);
                let mut vector = Graph::new();
                let start = vector.add_vector_params_with_transform("x", 1, ParamTransform::Exp);
                vector.vector_half_normal_logp(start, 1, sigma);
                check(&vector, &[raw], lp, &[1.0 - squared_ratio]);
                let mut hierarchical = Graph::new();
                let scale = hierarchical.add_param("sigma");
                HalfNormal::prior_with_node_sigma(&mut hierarchical, "x", scale);
                check(
                    &hierarchical,
                    &[sigma, raw],
                    lp,
                    &[(squared_ratio - 1.0) / sigma, 1.0 - squared_ratio],
                );
            }
        }
    }

    #[test]
    fn transformed_hyperparameter_gradients_match_finite_differences() {
        let mut graph = Graph::new();
        let alpha = graph.add_param("alpha");
        let rate = graph.add_param("rate");
        let raw = graph.add_param_with_transform("x", ParamTransform::Exp);
        graph.log_gamma_logp(raw, alpha, rate);
        for beta in [1e-200_f64, 0.7, 1e200] {
            let params = [1.3, beta, -beta.ln() + 0.4];
            let (_, analytic) = grad_logp(&graph, &params);
            for index in 0..3 {
                let h = if index == 1 { beta * 1e-5 } else { 1e-5 };
                let mut plus = params;
                let mut minus = params;
                plus[index] += h;
                minus[index] -= h;
                let numeric = (eval_logp(&graph, &plus) - eval_logp(&graph, &minus)) / (2.0 * h);
                assert!((analytic[index] / numeric - 1.0).abs() < 1e-7);
            }
            let mut exponential = Graph::new();
            let rate = exponential.add_param("rate");
            Exponential::prior_with_node_rate(&mut exponential, "x", rate);
            check(&exponential, &[beta, -beta.ln()], -1.0, &[0.0, 0.0]);
        }
    }

    #[test]
    fn normal_priors_and_observations_are_invariant_to_units() {
        for sigma in [1e-200_f64, 1.0, 1e200] {
            let lp = -0.5 * std::f64::consts::TAU.ln() - sigma.ln() - 0.125;
            let mut scalar = Graph::new();
            Normal::prior(&mut scalar, "x", 0.0, sigma);
            check(&scalar, &[0.5 * sigma], lp, &[-0.5 / sigma]);
            let mut vector = Graph::new();
            let start = vector.add_vector_params("x", 1);
            vector.vector_normal_logp(start, 1, 0.0, sigma);
            check(&vector, &[0.5 * sigma], lp, &[-0.5 / sigma]);
            for log_normal in [false, true] {
                let mut graph = Graph::new();
                let mu = graph.add_param("mu");
                let scale = graph.add_param("sigma");
                let obs = graph.add_obs_data(vec![if log_normal { 1.0 } else { 0.0 }]);
                let predictor = graph.broadcast_observation(mu, obs);
                if log_normal {
                    graph.obs_logp_lognormal(predictor, scale, obs);
                } else {
                    graph.normal_obs_logp(predictor, scale, obs);
                }
                check(
                    &graph,
                    &[-0.5 * sigma, sigma],
                    lp,
                    &[0.5 / sigma, -0.75 / sigma],
                );
            }
        }
    }

    #[test]
    fn student_t_preserves_scaled_density_and_polynomial_tails() {
        for sigma in [1e-200_f64, 1.0, 1e200] {
            let nu = 4.0;
            let z = 0.5_f64;
            let lp = ln_gamma(2.5)
                - ln_gamma(2.0)
                - 0.5 * (4.0 * std::f64::consts::PI).ln()
                - sigma.ln()
                - 2.5 * (1.0 + z * z / nu).ln();
            let dx = -5.0 * z / (nu + z * z) / sigma;
            let mut scalar = Graph::new();
            crate::distributions::StudentT::prior(&mut scalar, "x", nu, 0.0, sigma);
            let mut vector = Graph::new();
            let start = vector.add_vector_params("x", 1);
            vector.vector_student_t_logp(start, 1, nu, 0.0, sigma);
            check(&scalar, &[z * sigma], lp, &[dx]);
            check(&vector, &[z * sigma], lp, &[dx]);
        }
        let mut scalar = Graph::new();
        crate::distributions::StudentT::prior(&mut scalar, "x", 0.001, 0.0, 1.0);
        let x = 1e200_f64;
        let expected = ln_gamma(0.5005)
            - ln_gamma(0.0005)
            - 0.5 * (0.001 * std::f64::consts::PI).ln()
            - 0.5005 * (2.0 * x.ln() - 0.001_f64.ln());
        check(&scalar, &[x], expected, &[-1.001 / x]);
    }

    #[test]
    fn log_normal_observations_preserve_subnormal_values() {
        let y = (-740.0_f64).exp();
        let ly = y.ln();
        let mut graph = Graph::new();
        let mu = graph.add_param("mu");
        let sigma = graph.add_param("sigma");
        let obs = graph.add_obs_data(vec![y]);
        let predictor = graph.broadcast_observation(mu, obs);
        graph.obs_logp_lognormal(predictor, sigma, obs);
        check(
            &graph,
            &[ly - 1.0, 2.0],
            -0.5 * std::f64::consts::TAU.ln() - 2.0_f64.ln() - ly - 0.125,
            &[0.25, -0.375],
        );
    }
}

#[cfg(test)]
mod power_boundary_regressions {
    use super::*;
    use crate::{
        distributions::{HalfNormal, Normal},
        graph::ElementwiseOp,
    };

    #[test]
    fn zero_power_preserves_scalar_target_and_gradient_at_zero() {
        let mut graph = Graph::new();
        let x = Normal::prior(&mut graph, "x", 0.0, 1.0);
        let zero = graph.add_constant(0.0);
        let constant = graph.elementwise(ElementwiseOp::Pow, x, Some(zero));
        graph.add_logp_term(constant);
        for raw in [-0.3, 0.0, 0.3] {
            let expected = (
                1.0 - 0.5 * std::f64::consts::TAU.ln() - 0.5 * raw * raw,
                vec![-raw],
            );
            let mut evaluator = Evaluator::new(&graph);
            evaluator.compute(&graph, &[raw]);
            assert!((evaluator.total_logp - expected.0).abs() < 1e-14);
            assert_eq!(evaluator.grad, expected.1);
            let reference = grad_logp(&graph, &[raw]);
            assert!((reference.0 - expected.0).abs() < 1e-14);
            assert_eq!(reference.1, expected.1);
        }
    }

    #[test]
    fn zero_data_with_learned_positive_exponent_has_finite_gradient() {
        for vector in [false, true] {
            let mut graph = Graph::new();
            let x = graph.add_data("x", vec![0.0, 1.0, 2.0]);
            let exponent = if vector {
                let start = graph.add_vector_params_with_transform("b", 3, ParamTransform::Exp);
                graph.vector_half_normal_logp(start, 3, 1.0);
                let indices = graph.add_data("indices", vec![0.0, 1.0, 2.0]);
                graph.gather(start, 3, indices)
            } else {
                HalfNormal::prior(&mut graph, "b", 1.0)
            };
            let powered = graph.elementwise(ElementwiseOp::Pow, x, Some(exponent));
            let total = graph.sum(powered);
            let penalty = graph.elementwise(ElementwiseOp::Neg, total, None);
            graph.add_logp_term(penalty);
            for position in [-0.7, 0.0, 0.4] {
                let params = vec![position; graph.param_count];
                let mut evaluator = Evaluator::new(&graph);
                evaluator.compute(&graph, &params);
                let reference = grad_logp(&graph, &params);
                assert!(evaluator.total_logp.is_finite());
                assert_eq!(reference, (evaluator.total_logp, evaluator.grad.clone()));
                for i in 0..params.len() {
                    let mut plus = params.clone();
                    let mut minus = params.clone();
                    plus[i] += 1e-6;
                    minus[i] -= 1e-6;
                    let numerical = (eval_logp(&graph, &plus) - eval_logp(&graph, &minus)) / 2e-6;
                    assert!((numerical - evaluator.grad[i]).abs() < 1e-8);
                }
            }
        }
    }
}

/// Checks that used to live in `rust_core/tests/output_boundaries.rs` and
/// `rust_core/tests/poisson_density.rs`, moved here when the reference
/// evaluator stopped being public API. They pin the *reference* evaluator to
/// independent expectations, which the surviving `Evaluator` assertions in
/// those files do not do.
#[cfg(test)]
mod reference_boundary_coverage {
    use super::*;
    use crate::distributions::Uniform;
    use crate::graph::{ObsFamily, ParamTransform};

    /// Was `output_boundaries.rs:34`: an invalid interval must be rejected in
    /// unconstrained coordinates by the reference evaluator too, with a zero
    /// gradient rather than a NaN.
    #[test]
    fn reference_rejects_invalid_uniform_ranges() {
        for (lower, upper) in [
            (1.0, 1.0),
            (2.0, 1.0),
            (-1e308, 1e308),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
        ] {
            for scalar in [false, true] {
                let mut graph = Graph::new();
                if scalar {
                    Uniform::prior(&mut graph, "x", lower, upper);
                } else {
                    let start = graph.add_vector_params_with_transform(
                        "x",
                        1,
                        ParamTransform::BoundedSigmoid { lower, upper },
                    );
                    graph.vector_uniform_logp(start, 1, lower, upper);
                }
                for raw in [-40.0, 0.0, 40.0] {
                    assert_eq!(
                        grad_logp(&graph, &[raw]),
                        (f64::NEG_INFINITY, vec![0.0]),
                        "lower={lower} upper={upper} scalar={scalar} raw={raw}"
                    );
                }
            }
        }
    }

    fn poisson_graph(count: f64) -> Graph {
        let mut graph = Graph::new();
        let eta = graph.add_param("eta");
        let observed = graph.add_obs_data(vec![count]);
        let means = graph.broadcast_observation(eta, observed);
        graph.obs_logp_poisson_log(means, observed);
        graph
    }

    /// Was `poisson_density.rs:37-40`: at rates where Stirling's series is the
    /// only usable form, the reference `ObsFamily::PoissonLog` branch must
    /// reproduce the same mode, curvature and score as the Evaluator.
    #[test]
    fn reference_matches_high_rate_poisson_density_and_score() {
        for count in [1e14_f64, 1e15, 8e15] {
            let graph = poisson_graph(count);
            let expected_mode =
                -0.5 * (std::f64::consts::TAU.ln() + count.ln()) - 1.0 / (12.0 * count);
            let center = count.ln();
            let sd = 1.0 / count.sqrt();
            for z in [-1.0, 0.0, 1.0] {
                let eta = center + z * sd;
                let (logp, grad) = grad_logp(&graph, &[eta]);
                assert!((logp - (expected_mode - 0.5 * z * z)).abs() < 3e-6);
                assert_eq!(grad[0], count - eta.exp());
                assert_eq!(
                    logp,
                    crate::observation::log_density(ObsFamily::PoissonLog, count, eta, None)
                        .unwrap()
                );
            }
        }
    }

    /// Was `poisson_density.rs:66`: where the rate itself underflows, the log
    /// density is still finite and equals `y*eta - ln(y!)`.
    #[test]
    fn reference_keeps_finite_densities_when_poisson_rates_underflow() {
        for count in [0.0, 1.0, 20.0] {
            for eta in [-740.0, -1000.0] {
                let graph = poisson_graph(count);
                let expected = count * eta - ln_gamma(count + 1.0);
                let (logp, _) = grad_logp(&graph, &[eta]);
                assert!(logp.is_finite(), "count={count} eta={eta} logp={logp}");
                assert!((logp - expected).abs() < 1e-10);
            }
        }
    }
}

#[cfg(test)]
mod bernoulli_logit_tail {
    use super::*;

    /// The Bernoulli-logit density and its derivative keep their saturated tail.
    ///
    /// Both constants come from an out-of-crate evaluation (Python `decimal` at 120
    /// significant digits) of `-ln(1 + exp(-40))` and `exp(-40) / (1 + exp(-40))`.
    /// They agree to every digit shown, which is itself the point: the two
    /// quantities are equal at this scale, and both used to be returned as zero.
    #[test]
    fn bernoulli_logit_keeps_the_tail_a_saturated_logit_still_has() {
        const TRUE_VALUE: f64 = 4.248_354_255_291_589e-18;

        // The forms that were in the evaluator and its reference, for contrast.
        let cancelled_logp = 1.0 * 40.0 - softplus(40.0);
        let cancelled_grad = 1.0 - crate::graph::stable_sigmoid(40.0);
        assert_eq!(
            cancelled_logp, 0.0,
            "the subtraction this test exists for must still cancel"
        );
        assert_eq!(cancelled_grad, 0.0, "likewise for the derivative");

        for (y, eta, sign) in [(1.0, 40.0, 1.0), (0.0, -40.0, -1.0)] {
            let logp = bernoulli_logit_logp(y, eta);
            let grad = bernoulli_logit_grad(y, eta);
            assert!(
                (logp / -TRUE_VALUE - 1.0).abs() < 1e-12,
                "logp({y}, {eta}) = {logp}, expected {}",
                -TRUE_VALUE
            );
            assert!(
                (grad / (sign * TRUE_VALUE) - 1.0).abs() < 1e-12,
                "grad({y}, {eta}) = {grad}, expected {}",
                sign * TRUE_VALUE
            );
        }

        // An unsaturated logit is unaffected: log P(1 | 2) = -ln(1 + exp(-2)).
        let ordinary = bernoulli_logit_logp(1.0, 2.0);
        let reference = -(-2.0f64).exp().ln_1p(); // -ln(1 + exp(-2))
        assert!(
            (ordinary - reference).abs() < 1e-15,
            "logp(1, 2) = {ordinary} vs {reference}"
        );
    }
}
