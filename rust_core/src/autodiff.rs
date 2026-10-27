use crate::data::DataBinding;
use crate::graph::{ElementwiseOp, Graph, GraphShapeError, NodeId, ObsFamily, Op, ParamTransform};

use crate::graph::{
    bounded_sigmoid, bounded_sigmoid_adjoint, stable_sigmoid, stable_sigmoid_derivative,
};
use crate::numerics::digamma;
/// The special functions moved to the crate-private `numerics` module.
/// `ln_gamma` is re-exported here so `rustmc_core::autodiff::ln_gamma` keeps
/// working; `softplus` stays reachable at this path for in-crate callers.
pub use crate::numerics::ln_gamma;
pub(crate) use crate::numerics::softplus;

// ---------------------------------------------------------------------------
// Evaluator — zero-allocation gradient computation
// ---------------------------------------------------------------------------

/// Classifies each graph node for the Evaluator's storage layout.
#[derive(Clone, Copy)]
enum NodeKind {
    Scalar,
    /// Reads `binding.vectors[idx]` in place; no copy is made.
    DataRef(usize),
    /// Computed vector stored at byte offset `off` in the flat vec_buf.
    ComputedVec(usize),
}

/// One operand of an elementwise node, resolved once per node: a vector
/// read by element, or a scalar every element shares.
#[derive(Clone, Copy)]
enum Operand<'a> {
    Vector(&'a [f64]),
    Scalar(f64),
}

impl Operand<'_> {
    #[inline(always)]
    fn at(self, i: usize) -> f64 {
        match self {
            Operand::Vector(values) => values[i],
            Operand::Scalar(value) => value,
        }
    }
}

/// `node` read by element, resolved once: `binding`'s data for a data node,
/// its slot of `vec_buf` for a computed vector, or its scalar value for every
/// element. The same values `Evaluator::read_vec` reads, without matching on
/// the node per element. Locals like this one are what keep a loop that
/// stores into another buffer from reloading them on every element.
#[inline(always)]
fn operand<'a>(
    node_kind: &[NodeKind],
    scalars: &[f64],
    binding: &'a DataBinding,
    vec_buf: &'a [f64],
    node: NodeId,
) -> Operand<'a> {
    match node_kind[node.0] {
        NodeKind::Scalar => Operand::Scalar(scalars[node.0]),
        NodeKind::DataRef(di) => Operand::Vector(&binding.vectors[di]),
        NodeKind::ComputedVec(off) => Operand::Vector(&vec_buf[off..]),
    }
}

/// Where an elementwise node's reverse pass sends one operand's adjoint: a
/// scalar's running sum, which the caller writes back, a computed vector's
/// slots from an offset, or nowhere for data.
enum AdjointTarget {
    Scalar(f64),
    Vector(usize),
    None,
}

impl AdjointTarget {
    #[inline(always)]
    fn add(&mut self, adjoints: &mut [f64], i: usize, value: f64) {
        match self {
            AdjointTarget::Scalar(sum) => *sum += value,
            AdjointTarget::Vector(off) => adjoints[*off + i] += value,
            AdjointTarget::None => {}
        }
    }
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
    /// Per node, whether it was an `Op::Constant` when the evaluator was
    /// built, as `param_node_ids` records the parameters of that graph.
    constant_at_construction: Vec<bool>,
    /// Reusable scratch space for constrained vector parameters and gradients.
    param_scratch: Vec<f64>,
    /// `ln(y)` for every observation slot a LogNormal term reads, indexed like
    /// `binding.observations` and empty for every other slot. The data never
    /// change for a binding, so the forward pass reads these instead of taking
    /// a logarithm per observation per gradient. Rebuilt whenever the binding
    /// changes, including by a same-shape [`Self::rebind`]. A slot is therefore
    /// either empty or holds the logarithms of the binding's current values,
    /// and the forward pass fills an empty one it needs: see
    /// [`Self::ensure_log_observations`].
    log_observations: Vec<Vec<f64>>,
    /// Per node, a value the forward pass of an observation term computed for
    /// each observation: the standardized residual `(y - mu) / sigma` of a
    /// Normal term (of `ln(y)` for LogNormal), and the exponential a
    /// Bernoulli-logit or Poisson-log term took, `exp(-|eta|)` and
    /// `exp(eta)`. The reverse pass, which runs only after the forward pass
    /// on the same parameters, reads them instead of computing them again.
    /// Empty for every other node.
    observation_cache: Vec<Vec<f64>>,
    /// Per-group values or derivatives for the `Op::Gather` being evaluated,
    /// reused across nodes and calls. Construction and a same-shape rebind
    /// size it for the widest gather of the graph they were given, so
    /// evaluating that graph never grows it.
    group_scratch: Vec<f64>,
    /// Normal terms on a grouped linear predictor that run as one forward
    /// and one reverse loop; see `autodiff_grouped.rs`.
    grouped: grouped::GroupedPlan,
    /// The fused reverse loop's copy of a gather's gradients, sized like
    /// `group_scratch`.
    grouped_grad: Vec<f64>,
    /// Test-only: send every node through its per-element loop, so tests can
    /// compare the resolved loops with the code they replaced.
    #[cfg(test)]
    per_element: bool,
    /// Test-only: run every grouped Normal chain as its separate nodes, so
    /// tests can compare the fused loops with them.
    #[cfg(test)]
    unfused_groups: bool,
}

/// The group count of the widest `Op::Gather` in `graph`, which sizes
/// `Evaluator::group_scratch` so that evaluating `graph` never grows it.
fn widest_gather(graph: &Graph) -> usize {
    graph
        .nodes
        .iter()
        .map(|node| match node.op {
            Op::Gather { n_params, .. } => n_params,
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

/// Fill `out` with the `ln(y)` cache described on `Evaluator::log_observations`,
/// reusing its allocations.
fn fill_log_observations(graph: &Graph, binding: &DataBinding, out: &mut Vec<Vec<f64>>) {
    out.resize_with(binding.observations.len(), Vec::new);
    out.iter_mut().for_each(Vec::clear);
    for node in &graph.nodes {
        if let Op::ObsLogP {
            family: ObsFamily::LogNormal,
            obs_data_idx,
            ..
        } = node.op
        {
            let cache = &mut out[obs_data_idx];
            if cache.is_empty() {
                cache.extend(binding.observations[obs_data_idx].iter().map(|y| y.ln()));
            }
        }
    }
}

/// Size `out`, the `Evaluator::observation_cache` cache, for `graph` and
/// `binding`, reusing its allocations: one value per observation for each
/// Normal, LogNormal, Bernoulli-logit and Poisson-log term, and nothing for
/// every other node.
fn fill_observation_cache(graph: &Graph, binding: &DataBinding, out: &mut Vec<Vec<f64>>) {
    out.resize_with(graph.nodes.len(), Vec::new);
    for (node, cache) in graph.nodes.iter().zip(out.iter_mut()) {
        match node.op {
            Op::ObsLogP {
                family:
                    ObsFamily::Normal
                    | ObsFamily::LogNormal
                    | ObsFamily::BernoulliLogit
                    | ObsFamily::PoissonLog,
                obs_data_idx,
                ..
            } => cache.resize(binding.observations[obs_data_idx].len(), 0.0),
            _ => cache.clear(),
        }
    }
}

/// The payload checks of [`DataBinding::validate_for`], as a shape error.
fn validate_binding_payload(graph: &Graph, binding: &DataBinding) -> Result<(), GraphShapeError> {
    binding
        .validate_for(graph)
        .map_err(|error| GraphShapeError::new(error.to_string()))
}

/// Check that `binding` provides every slot the graph indexes by raw index.
///
/// Separate from [`validate_binding_payload`] because it has to run before
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
///
/// The parameter metadata is public too, so it is checked against
/// `param_count` here: the evaluator reads `param_transforms[k]` for every
/// parameter a node reaches, and the samplers read names and transforms per
/// parameter and the metric per span. An observation op's auxiliary node is
/// checked against its family for the same reason: the evaluator reads the
/// scale or dispersion of the families that have one without re-checking it.
fn validate_topology(graph: &Graph) -> Result<(), GraphShapeError> {
    let count = graph.param_count;
    if graph.param_names.len() != count || graph.param_transforms.len() != count {
        return Err(GraphShapeError::new(format!(
            "graph declares {count} parameters but has {} names and {} transforms",
            graph.param_names.len(),
            graph.param_transforms.len()
        )));
    }
    let mut next_start = 0usize;
    for span in &graph.param_spans {
        let end = span.start.checked_add(span.len);
        if span.start != next_start || end.is_none_or(|end| end > count) {
            return Err(GraphShapeError::new(format!(
                "parameter spans must tile 0..{count} in order; found {}..{} where {next_start} was next",
                span.start,
                span.start.saturating_add(span.len)
            )));
        }
        next_start = span.start + span.len;
    }
    if next_start != count {
        return Err(GraphShapeError::new(format!(
            "parameter spans cover 0..{next_start} of {count} parameters"
        )));
    }
    for (position, node) in graph.nodes.iter().enumerate() {
        if node.id.0 != position {
            return Err(GraphShapeError::new(format!(
                "node at position {position} carries id {}",
                node.id.0
            )));
        }
        if let Op::ObsLogP { family, aux, .. } = &node.op {
            let needs_aux = matches!(
                family,
                ObsFamily::Normal | ObsFamily::LogNormal | ObsFamily::NegativeBinomialLog
            );
            if needs_aux != aux.is_some() {
                let requirement = if needs_aux {
                    "requires an auxiliary scale node"
                } else {
                    "takes no auxiliary node"
                };
                return Err(GraphShapeError::new(format!(
                    "{family:?} observation at node {position} {requirement}"
                )));
            }
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
                // `DataBinding::validate_for` makes the same check with a
                // better message, but only for a binding it is given; this one
                // guards the `as usize` index in the forward pass whatever the
                // caller validated.
                if binding.vectors[di]
                    .iter()
                    .any(|x| !crate::data::is_group_index(*x, *n_params))
                {
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
        // `from_graph` has already run the payload validation.
        let binding =
            DataBinding::from_graph(graph).map_err(|e| GraphShapeError::new(e.to_string()))?;
        validate_slot_coverage(graph, &binding)?;
        let node_lengths = validate_node_lengths(graph, &binding)?;
        Ok(Self::from_checked(graph, binding, node_lengths))
    }

    /// Construct an evaluator for immutable structure plus a validated dataset.
    pub fn try_with_binding(graph: &Graph, binding: DataBinding) -> Result<Self, GraphShapeError> {
        // Coverage first: the length pass reads `binding.vectors[i]`,
        // `binding.observations[i]` and `binding.matrices[i]` at the raw indices
        // the graph carries, so a missing slot is an out-of-bounds index there
        // rather than an error. Nothing else moves -- a binding that covers its
        // slots reaches the length pass and the payload validation in the order
        // it always did.
        validate_slot_coverage(graph, &binding)?;
        let node_lengths = validate_node_lengths(graph, &binding)?;
        validate_binding_payload(graph, &binding)?;
        Ok(Self::from_checked(graph, binding, node_lengths))
    }

    /// Lay out the buffers for a graph and binding that have passed slot
    /// coverage, the length pass (which produced `node_lengths`) and the
    /// payload validation. Each constructor runs those checks exactly once.
    fn from_checked(graph: &Graph, binding: DataBinding, node_lengths: Vec<usize>) -> Self {
        let n = graph.nodes.len();
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

        let constant_at_construction = graph
            .nodes
            .iter()
            .map(|node| matches!(node.op, Op::Constant(_)))
            .collect();

        let mut log_observations = Vec::new();
        fill_log_observations(graph, &binding, &mut log_observations);
        let mut observation_cache = Vec::new();
        fill_observation_cache(graph, &binding, &mut observation_cache);
        // Build the Poisson log-factorial table now rather than during a fit.
        crate::count_sampling::small_log_factorials();
        let grouped = grouped::GroupedPlan::new(graph, &node_kind, &node_lengths, &binding);
        Self {
            log_observations,
            observation_cache,
            group_scratch: Vec::with_capacity(widest_gather(graph)),
            grouped,
            grouped_grad: Vec::with_capacity(widest_gather(graph)),
            #[cfg(test)]
            per_element: false,
            #[cfg(test)]
            unfused_groups: false,
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
            constant_at_construction,
            param_scratch: vec![0.0; graph.param_count],
        }
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
        validate_binding_payload(graph, &binding)?;
        let node_lengths = validate_node_lengths(graph, &binding)?;
        if node_lengths == self.node_lengths {
            fill_log_observations(graph, &binding, &mut self.log_observations);
            fill_observation_cache(graph, &binding, &mut self.observation_cache);
            self.group_scratch.clear();
            self.group_scratch.reserve(widest_gather(graph));
            self.grouped_grad.clear();
            self.grouped_grad.reserve(widest_gather(graph));
            self.binding = binding;
        } else {
            *self = Self::from_checked(graph, binding, node_lengths);
        }
        Ok(())
    }

    /// The reverse pass of the Normal or LogNormal observation term `idx`,
    /// with mean `mu` and scale `sigma`, from the standardized residuals its
    /// forward pass stored.
    ///
    /// A scale that is a constant now and was one when the evaluator was
    /// built gets no adjoint: its reverse arm does nothing, and the gradient
    /// map built with the evaluator cannot name it. Checking only the current
    /// node would drop the gradient of a parameter that a graph changed after
    /// construction turned into a constant, which the map still reads.
    fn add_gaussian_residual_adjoints(
        &mut self,
        graph: &Graph,
        idx: usize,
        mu: usize,
        sigma: usize,
        a_s: f64,
    ) {
        let residuals = &self.observation_cache[idx];
        let mu_adjoints = match self.node_kind[mu] {
            NodeKind::ComputedVec(off) => Some(&mut self.adj_vec_buf[off..off + residuals.len()]),
            _ => None,
        };
        let constant = self.constant_at_construction.get(sigma) == Some(&true)
            && matches!(graph.nodes[sigma].op, Op::Constant(_));
        let sigma_adjoint = (!constant).then_some(&mut self.adj_scalars[sigma]);
        gaussian_residual_adjoints(
            residuals,
            self.scalars[sigma],
            a_s,
            mu_adjoints,
            sigma_adjoint,
        );
    }

    /// Whether loops may read operands resolved once per node; false only in
    /// tests that ask for the per-element loops.
    #[inline(always)]
    fn resolved_loops(&self) -> bool {
        #[cfg(test)]
        if self.per_element {
            return false;
        }
        true
    }

    /// Whether grouped Normal chains may run fused. The per-element test
    /// mode turns them off too, since they read operands as resolved slices.
    fn fusion_enabled(&self) -> bool {
        #[cfg(test)]
        if self.unfused_groups {
            return false;
        }
        self.resolved_loops()
    }

    /// Whether a gather of `n_params` groups from `param_start` over `vl` rows
    /// transforms each group once instead of each row. Only when its indices
    /// are data, the span is not empty and lies inside the parameters and
    /// their transforms, there are no more groups than rows, and every group
    /// has the same kind of transform. Then the groups take at most as many
    /// transforms, of the one kind the rows would have taken, as the per-row
    /// loop would; with mixed kinds, rows that all select a cheap group would
    /// otherwise pay for the dear ones they never read. Every other gather runs
    /// the per-row loop, which reads only the parameters its rows select,
    /// exactly as before.
    fn gather_by_group(
        &self,
        graph: &Graph,
        params: &[f64],
        (param_start, n_params, indices): (usize, usize, NodeId),
        vl: usize,
    ) -> bool {
        let span_fits = param_start
            .checked_add(n_params)
            .is_some_and(|end| end <= params.len() && end <= graph.param_transforms.len());
        if !(matches!(self.node_kind[indices.0], NodeKind::DataRef(_))
            && span_fits
            && n_params > 0
            && n_params <= vl
            && self.resolved_loops())
        {
            return false;
        }
        let transforms = &graph.param_transforms[param_start..param_start + n_params];
        let kind = std::mem::discriminant(&transforms[0]);
        transforms
            .iter()
            .all(|transform| std::mem::discriminant(transform) == kind)
    }

    /// The forward pass of an elementwise node with a vector output, with
    /// each operand resolved once instead of once per element and the operator
    /// chosen outside the loop. Every element takes the same operations, on
    /// the same operands, as in the per-element loop. Returns false, having
    /// written nothing, when the node has
    /// no vector output or an operand sits at or after the output in
    /// `vec_buf`, which only a graph changed after construction can arrange;
    /// the caller then runs the per-element loop.
    fn elementwise_vector_forward(
        &mut self,
        idx: usize,
        vl: usize,
        operator: ElementwiseOp,
        a: NodeId,
        b: Option<NodeId>,
    ) -> bool {
        let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
            return false;
        };
        if vl == 0 || !self.resolved_loops() {
            return false;
        }
        let (inputs, outputs) = self.vec_buf.split_at_mut(off);
        let operand = |node: NodeId| match self.node_kind[node.0] {
            NodeKind::Scalar => Some(Operand::Scalar(self.scalars[node.0])),
            NodeKind::DataRef(di) => Some(Operand::Vector(&self.binding.vectors[di])),
            NodeKind::ComputedVec(o) if o + vl <= off => Some(Operand::Vector(&inputs[o..])),
            NodeKind::ComputedVec(_) => None,
        };
        let Some(av) = operand(a) else {
            return false;
        };
        let bv = match b {
            Some(b) => match operand(b) {
                Some(bv) => bv,
                None => return false,
            },
            None => Operand::Scalar(0.0),
        };
        let out = &mut outputs[..vl];
        macro_rules! fill {
            ($($op:ident),*) => {
                match operator {
                    $(ElementwiseOp::$op => {
                        for (i, value) in out.iter_mut().enumerate() {
                            *value = ElementwiseOp::$op.value(av.at(i), bv.at(i));
                        }
                    })*
                }
            };
        }
        fill!(Add, Sub, Mul, Div, Pow, Neg, Exp, Log, Sigmoid, Sqrt, Tanh, Softplus, Sin, Cos);
        true
    }

    /// The reverse pass of an elementwise node with a vector output, resolved
    /// and dispatched as in [`Self::elementwise_vector_forward`]. Each element
    /// skips a zero upstream adjoint and otherwise adds the same adjoints in
    /// the same order as the per-element loop: a scalar operand's adjoint
    /// takes them one element at a time, in element order. Returns false,
    /// having changed nothing, when the per-element loop must run instead:
    /// no vector output, both operands the same node, or an operand's adjoint
    /// at or after the output's in `adj_vec_buf`.
    fn elementwise_vector_backward(
        &mut self,
        idx: usize,
        vl: usize,
        operator: ElementwiseOp,
        a: NodeId,
        b: Option<NodeId>,
    ) -> bool {
        let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
            return false;
        };
        if vl == 0 || b == Some(a) || !self.resolved_loops() {
            return false;
        }
        let below_output = |node: NodeId| match self.node_kind[node.0] {
            NodeKind::ComputedVec(o) => o + vl <= off,
            _ => true,
        };
        if !below_output(a) || !b.is_none_or(below_output) {
            return false;
        }
        let operand = |node: NodeId| match self.node_kind[node.0] {
            NodeKind::Scalar => Operand::Scalar(self.scalars[node.0]),
            NodeKind::DataRef(di) => Operand::Vector(&self.binding.vectors[di]),
            NodeKind::ComputedVec(o) => Operand::Vector(&self.vec_buf[o..]),
        };
        let av = operand(a);
        let bv = b.map_or(Operand::Scalar(0.0), operand);
        let outs = &self.vec_buf[off..off + vl];
        let (lower, upper) = self.adj_vec_buf.split_at_mut(off);
        let upstreams = &upper[..vl];
        let target = |node: NodeId| match self.node_kind[node.0] {
            NodeKind::Scalar => AdjointTarget::Scalar(self.adj_scalars[node.0]),
            NodeKind::ComputedVec(o) => AdjointTarget::Vector(o),
            NodeKind::DataRef(_) => AdjointTarget::None,
        };
        let mut a_target = target(a);
        let mut b_target = b.map_or(AdjointTarget::None, target);
        macro_rules! propagate {
            ($($op:ident),*) => {
                match operator {
                    $(ElementwiseOp::$op => {
                        for (i, (&upstream, &out)) in upstreams.iter().zip(outs).enumerate() {
                            if upstream == 0.0 {
                                continue;
                            }
                            let (da, db) = ElementwiseOp::$op
                                .adjoints_with_value(upstream, av.at(i), bv.at(i), out);
                            a_target.add(lower, i, da);
                            if b.is_some() {
                                b_target.add(lower, i, db);
                            }
                        }
                    })*
                }
            };
        }
        propagate!(Add, Sub, Mul, Div, Pow, Neg, Exp, Log, Sigmoid, Sqrt, Tanh, Softplus, Sin, Cos);
        if let AdjointTarget::Scalar(sum) = a_target {
            self.adj_scalars[a.0] = sum;
        }
        if let (Some(b), AdjointTarget::Scalar(sum)) = (b, b_target) {
            self.adj_scalars[b.0] = sum;
        }
        true
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

    /// Make `log_observations[slot]` hold `ln(y)` for observation slot `slot`.
    ///
    /// Construction and every rebind fill it for each LogNormal term of the
    /// graph they were given, so this does nothing on that graph. A graph
    /// changed afterwards (the `Graph` fields are public) can carry a LogNormal
    /// term on a slot nobody filled; taking the logarithms here evaluates it
    /// exactly as before the cache existed, instead of against an empty slot,
    /// which would silently drop the likelihood. Only that case allocates.
    fn ensure_log_observations(&mut self, slot: usize) {
        let observations = &self.binding.observations[slot];
        if self.log_observations[slot].len() != observations.len() {
            self.log_observations[slot] = observations.iter().map(|y| y.ln()).collect();
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
    pub fn vec_elem(&self, node: NodeId, i: usize) -> f64 {
        self.read_vec(node.0, i)
    }

    /// Compute log-probability and its gradient. Results are stored in
    /// `self.total_logp` and `self.grad`. No heap allocations occur on the
    /// graph the evaluator was built for or last rebound to; a graph changed
    /// since then so that an existing observation node becomes a Normal,
    /// LogNormal, Bernoulli-logit or Poisson-log term allocates once for that
    /// term's cache (see `ensure_log_observations` and `observation_cache`).
    /// Adding nodes needs a new evaluator.
    pub fn compute(&mut self, graph: &Graph, params: &[f64]) {
        self.forward(graph, params);
        self.backward(graph, params);
    }

    /// Evaluate node values and `self.total_logp` without the reverse pass.
    ///
    /// For callers that only read forward values (prediction, deterministic
    /// outputs); `self.grad` is left as it was.
    pub fn forward(&mut self, graph: &Graph, params: &[f64]) {
        self.grouped.begin_call();
        // Graphs without grouped chains skip the lookups below.
        let grouped = self.grouped.has_chains();
        for node in &graph.nodes {
            let idx = node.id.0;
            // A grouped chain decided at its gather runs fused at its term.
            if grouped {
                if let Some(chain) = self.grouped.chain_starting_at(idx) {
                    self.grouped_decide(graph, params, chain);
                }
                if self.grouped.skips(idx) {
                    continue;
                }
                if let Some(chain) = self.grouped.chain_ending_at(idx) {
                    if self.grouped.fused(chain) {
                        self.grouped_forward(graph, params, chain);
                        continue;
                    }
                }
            }
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
                    if self.elementwise_vector_forward(idx, vl, *operator, *a, *b) {
                        continue;
                    }
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
                    n_params,
                    indices,
                } => {
                    let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
                        unreachable!()
                    };
                    if self.gather_by_group(graph, params, (*param_start, *n_params, *indices), vl)
                    {
                        let NodeKind::DataRef(di) = self.node_kind[indices.0] else {
                            unreachable!()
                        };
                        // Each group's value, transformed once rather than once
                        // per row that selects it.
                        let mut values = std::mem::take(&mut self.group_scratch);
                        values.clear();
                        values.extend((0..*n_params).map(|g| {
                            let k = *param_start + g;
                            graph.param_transforms[k].apply(params[k])
                        }));
                        let groups = &self.binding.vectors[di][..vl];
                        let out = &mut self.vec_buf[off..off + vl];
                        for (value, &group) in out.iter_mut().zip(groups) {
                            let group = group as usize;
                            *value = match values.get(group) {
                                Some(&value) => value,
                                None => {
                                    let k = *param_start + group;
                                    graph.param_transforms[k].apply(params[k])
                                }
                            };
                        }
                        self.group_scratch = values;
                    } else {
                        for i in 0..vl {
                            let k = *param_start + self.read_vec(indices.0, i) as usize;
                            self.vec_buf[off + i] = graph.param_transforms[k].apply(params[k]);
                        }
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
                Op::Sigmoid(a) => self.scalars[idx] = stable_sigmoid(self.scalars[a.0]),
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
                    let resolved = self.resolved_loops();
                    let (inputs, outputs) = self.vec_buf.split_at_mut(out_off);
                    let values = match self.node_kind[vec.0] {
                        _ if !resolved => None,
                        NodeKind::DataRef(di) => Some(&self.binding.vectors[di][..vl]),
                        NodeKind::ComputedVec(o) if o + vl <= out_off => Some(&inputs[o..o + vl]),
                        _ => None,
                    };
                    if let Some(values) = values {
                        for (out, &v) in outputs[..vl].iter_mut().zip(values) {
                            *out = s + v;
                        }
                    } else {
                        for i in 0..vl {
                            let v = self.read_vec(vec.0, i);
                            self.vec_buf[out_off + i] = s + v;
                        }
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
                            let mut residuals = std::mem::take(&mut self.observation_cache[idx]);
                            residuals.resize(vl.min(obs.len()), 0.0);
                            let mut sum_sq = 0.0f64;
                            let resolved = self.resolved_loops();
                            let means = operand(
                                &self.node_kind,
                                &self.scalars,
                                &self.binding,
                                &self.vec_buf,
                                *linpred_vec,
                            );
                            for (i, (&y, d)) in obs.iter().zip(residuals.iter_mut()).enumerate() {
                                let m = if resolved {
                                    means.at(i)
                                } else {
                                    self.read_vec(linpred_vec.0, i)
                                };
                                *d = (y - m) / sv;
                                sum_sq += *d * *d;
                            }
                            self.observation_cache[idx] = residuals;
                            self.scalars[idx] = n * log_norm - 0.5 * sum_sq;
                        }
                        crate::graph::ObsFamily::BernoulliLogit => {
                            // Moved out and back so the loop can read `self`.
                            // Construction and every rebind size it for the
                            // graph they were given, so the resize does
                            // nothing there; a node changed afterward into
                            // this term allocates once here.
                            let mut exps = std::mem::take(&mut self.observation_cache[idx]);
                            exps.resize(vl.min(obs.len()), 0.0);
                            let mut sum = 0.0f64;
                            for (i, (&y, e)) in obs.iter().zip(exps.iter_mut()).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                *e = bernoulli_logit_exp(y, eta);
                                sum += bernoulli_logit_logp_given(y, eta, *e);
                            }
                            self.observation_cache[idx] = exps;
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::PoissonLog => {
                            let mut exps = std::mem::take(&mut self.observation_cache[idx]);
                            exps.resize(vl.min(obs.len()), 0.0);
                            let mut sum = 0.0f64;
                            for (i, (&y, rate)) in obs.iter().zip(exps.iter_mut()).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                *rate = eta.exp();
                                sum += crate::count_sampling::log_mass_from_log_rate_given_rate(
                                    y, eta, *rate,
                                );
                            }
                            self.observation_cache[idx] = exps;
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
                            self.ensure_log_observations(*obs_data_idx);
                            let log_obs = &self.log_observations[*obs_data_idx];
                            let mut residuals = std::mem::take(&mut self.observation_cache[idx]);
                            residuals.resize(vl.min(log_obs.len()), 0.0);
                            let mut sum = 0.0f64;
                            let resolved = self.resolved_loops();
                            let means = operand(
                                &self.node_kind,
                                &self.scalars,
                                &self.binding,
                                &self.vec_buf,
                                *linpred_vec,
                            );
                            for (i, (&ly, d)) in
                                log_obs.iter().zip(residuals.iter_mut()).enumerate()
                            {
                                let m = if resolved {
                                    means.at(i)
                                } else {
                                    self.read_vec(linpred_vec.0, i)
                                };
                                *d = (ly - m) / sv;
                                sum += log_norm - ly - 0.5 * *d * *d;
                            }
                            self.observation_cache[idx] = residuals;
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
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let matrix = &self.binding.matrices[*matrix_idx];
                    let base = intercept.map_or(0.0, |n| self.scalars[n.0]);
                    let out = &mut self.vec_buf[out_off..out_off + vl];
                    out.fill(base);
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
                    add_matrix_times_vector(&matrix.data, matrix.n_rows, beta_slice, out);
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
                    if !scale_is_valid(*sigma) {
                        self.scalars[idx] = invalid_vector_logp(*n_params);
                        continue;
                    }
                    let ln_sigma = sigma.ln();
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum += log_half_normal_kernel(raw, ln_sigma);
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
                    if !student_t_args_valid(*nu, *sigma) {
                        self.scalars[idx] = invalid_vector_logp(*n_params);
                        continue;
                    }
                    let (ln_nu, ln_sigma) = (nu.ln(), sigma.ln());
                    let log_normaliser = student_t_log_normaliser(*nu, ln_nu, ln_sigma);
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        sum += student_t_kernel(v - mu, *nu, ln_nu, ln_sigma, log_normaliser);
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
                    if !gamma_args_valid(*alpha, *beta) {
                        self.scalars[idx] = invalid_vector_logp(*n_params);
                        continue;
                    }
                    let (ln_beta, ln_gamma_alpha) = (beta.ln(), ln_gamma(*alpha));
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum += log_gamma_kernel(raw, *alpha, ln_beta, ln_gamma_alpha);
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

        let grouped = self.grouped.has_chains();
        for node in graph.nodes.iter().rev() {
            let idx = node.id.0;
            // A grouped chain's term ran its nodes' reverse passes fused.
            if grouped && self.grouped.skips(idx) {
                continue;
            }
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
            if grouped {
                if let Some(chain) = self.grouped.chain_ending_at(idx) {
                    if self.grouped_reverse(graph, params, chain, a_s) {
                        continue;
                    }
                }
            }

            match &node.op {
                Op::Elementwise { operator, a, b } => {
                    if self.elementwise_vector_backward(idx, vl, *operator, *a, *b) {
                        continue;
                    }
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
                        let out = self.read_vec(idx, i);
                        let (da, db) = operator.adjoints_with_value(upstream, av, bv, out);
                        self.accumulate(*a, i, da);
                        if let Some(b) = b {
                            self.accumulate(*b, i, db);
                        }
                    }
                }
                Op::Gather {
                    param_start,
                    n_params,
                    indices,
                } => {
                    let NodeKind::ComputedVec(off) = self.node_kind[idx] else {
                        unreachable!()
                    };
                    if self.gather_by_group(graph, params, (*param_start, *n_params, *indices), vl)
                    {
                        let NodeKind::DataRef(di) = self.node_kind[indices.0] else {
                            unreachable!()
                        };
                        // Each group's derivative, taken once rather than once
                        // per row; every row still adds its own product in row
                        // order.
                        let mut derivatives = std::mem::take(&mut self.group_scratch);
                        derivatives.clear();
                        derivatives.extend((0..*n_params).map(|g| {
                            let k = *param_start + g;
                            graph.param_transforms[k].derivative(params[k])
                        }));
                        let groups = &self.binding.vectors[di][..vl];
                        let adjoints = &self.adj_vec_buf[off..off + vl];
                        let grad = &mut self.grad[..];
                        for (&group, &adjoint) in groups.iter().zip(adjoints) {
                            let group = group as usize;
                            let k = *param_start + group;
                            let derivative = match derivatives.get(group) {
                                Some(&derivative) => derivative,
                                None => graph.param_transforms[k].derivative(params[k]),
                            };
                            grad[k] += adjoint * derivative;
                        }
                        self.group_scratch = derivatives;
                    } else {
                        for i in 0..vl {
                            let k = *param_start + self.read_vec(indices.0, i) as usize;
                            self.grad[k] += self.adj_vec_buf[off + i]
                                * graph.param_transforms[k].derivative(params[k]);
                        }
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
                    // d/da exp(a) is the node's own value.
                    self.adj_scalars[a.0] += a_s * self.scalars[idx];
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
                    let resolved = self.resolved_loops();
                    let (lower, upper) = self.adj_vec_buf.split_at_mut(out_off);
                    let upstreams = &upper[..vl];
                    match self.node_kind[vec.0] {
                        NodeKind::ComputedVec(v_off) if resolved && v_off + vl <= out_off => {
                            for (adjoint, &upstream) in
                                lower[v_off..v_off + vl].iter_mut().zip(upstreams)
                            {
                                ds += upstream;
                                *adjoint += upstream;
                            }
                        }
                        NodeKind::ComputedVec(_) => {
                            for i in 0..vl {
                                let upstream = self.adj_vec_buf[out_off + i];
                                ds += upstream;
                                if let NodeKind::ComputedVec(v_off) = self.node_kind[vec.0] {
                                    self.adj_vec_buf[v_off + i] += upstream;
                                }
                            }
                        }
                        _ => {
                            for &upstream in upstreams {
                                ds += upstream;
                            }
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
                    // As for every other density: where `nu` or `sigma` is
                    // outside its support the density is a constant -inf, so
                    // nothing it reads gets a slope. Unguarded, `ln` of the
                    // invalid argument made every adjoint NaN.
                    let (nu_value, sigma_value) = (self.scalars[nu.0], self.scalars[sigma.0]);
                    if !student_t_args_valid(nu_value, sigma_value) {
                        continue;
                    }
                    let (dx, dsigma, dnu) = student_t_derivatives(
                        self.scalars[x.0],
                        nu_value,
                        self.scalars[mu.0],
                        sigma_value,
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

                            self.add_gaussian_residual_adjoints(
                                graph,
                                idx,
                                linpred_vec.0,
                                sigma_node.0,
                                a_s,
                            );
                        }
                        crate::graph::ObsFamily::BernoulliLogit => {
                            let eta_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            let exps = &self.observation_cache[idx];
                            for (i, (&y, &e)) in obs.iter().zip(exps).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i);
                                let grad = bernoulli_logit_grad_given(y, eta, e);
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
                            let rates = &self.observation_cache[idx];
                            for (i, (&y, &rate)) in obs.iter().zip(rates).enumerate() {
                                let grad = y - rate;
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

                            self.add_gaussian_residual_adjoints(
                                graph,
                                idx,
                                linpred_vec.0,
                                sigma_node.0,
                                a_s,
                            );
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
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let matrix = &self.binding.matrices[*matrix_idx];
                    let adj_slice = &self.adj_vec_buf[out_off..out_off + vl];
                    let transforms =
                        &graph.param_transforms[*param_start..*param_start + *n_params];
                    let all_identity = transforms
                        .iter()
                        .all(|t| matches!(t, ParamTransform::Identity));
                    // grad += X^T @ adj
                    if all_identity {
                        let target = &mut self.grad[*param_start..*param_start + *n_params];
                        add_transpose_times_vector(&matrix.data, matrix.n_rows, adj_slice, target);
                    } else {
                        let target =
                            &mut self.param_scratch[*param_start..*param_start + *n_params];
                        target.fill(0.0);
                        add_transpose_times_vector(&matrix.data, matrix.n_rows, adj_slice, target);
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
                    let ln_sigma = sigma.ln();
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        self.grad[param_start + k] += a_s * (1.0 - (2.0 * (raw - ln_sigma)).exp());
                    }
                }
                Op::VectorStudentTLogP {
                    param_start,
                    n_params,
                    nu,
                    mu,
                    sigma,
                } => {
                    // Only the location score is needed: `nu` and `sigma` are
                    // constants of the op, so their digamma terms are never formed.
                    let (ln_sigma, ln_nu, ln_nu1) = (sigma.ln(), nu.ln(), (nu + 1.0).ln());
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        self.grad[param_start + k] +=
                            a_s * student_t_dx(v - mu, ln_sigma, ln_nu, ln_nu1);
                    }
                }
                Op::VectorGammaLogP {
                    param_start,
                    n_params,
                    alpha,
                    beta,
                } => {
                    let ln_beta = beta.ln();
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        self.grad[param_start + k] += a_s * (alpha - (raw + ln_beta).exp());
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
                        let s = stable_sigmoid(raw);
                        // d/draw = α·(1-s) - β·s
                        self.grad[param_start + k] +=
                            a_s * (alpha * stable_sigmoid(-raw) - beta * s);
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
                        let s = stable_sigmoid(raw);
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

/// Allocating second evaluator of the whole IR, used only to cross-check the
/// zero-allocation `Evaluator`'s storage and adjoint plumbing; it shares the
/// local derivative kernels, so it does not check those. It is deliberately
/// not exported: it has no non-test callers and it panics on its entry points
/// when a node's shape is not what it expected.
#[cfg(test)]
#[path = "autodiff_reference.rs"]
mod reference;

#[path = "autodiff_grouped.rs"]
mod grouped;
pub use grouped::fused_grouped_normal_passes;
#[cfg(test)]
pub(crate) use reference::{eval_logp, grad_logp};

/// Designs up to this many columns use the loops below; wider ones use faer.
///
/// Measured with `examples/gradient_cost.rs` (records in
/// `benchmarks/results/2026-10-24-gradient-cost.txt` and
/// `2026-10-25-gradient-cost-wide-designs.txt`): the loops are several times
/// faster on narrow designs, faer's general `matmul` being slow on a narrow
/// row-major product, and still 6% to 22% faster at 96 and 128 columns than
/// faer's fastest runs. At 256 the two are even for a Normal likelihood and
/// faer is faster for a Bernoulli-logit one at 500 rows.
const LOOP_MAX_COLUMNS: usize = 128;

/// `out[i] += row_i · beta` for the row-major `rows x beta.len()` matrix.
fn add_matrix_times_vector(matrix: &[f64], rows: usize, beta: &[f64], out: &mut [f64]) {
    let columns = beta.len();
    if columns == 0 {
        return;
    }
    // The loops pair rows with outputs in blocks, so the shapes must agree.
    debug_assert_eq!((matrix.len(), out.len()), (rows * columns, rows));
    if columns <= LOOP_MAX_COLUMNS
        && with_fused_multiply_add(MatrixTimesVector {
            matrix,
            beta,
            out: &mut *out,
        })
    {
        return;
    }
    use faer::{col, linalg::matmul::matmul, mat};
    matmul(
        col::from_slice_mut::<f64>(out).as_2d_mut(),
        mat::from_row_major_slice::<f64>(matrix, rows, columns),
        col::from_slice::<f64>(beta).as_2d(),
        Some(1.0),
        1.0,
        FAER_PARALLELISM,
    );
}

/// `target[k] += sum_i matrix[i][k] * weights[i]` for the row-major
/// `rows x target.len()` matrix.
fn add_transpose_times_vector(matrix: &[f64], rows: usize, weights: &[f64], target: &mut [f64]) {
    let columns = target.len();
    if columns == 0 {
        return;
    }
    // The loops pair rows with weights in blocks, so the shapes must agree.
    debug_assert_eq!((matrix.len(), weights.len()), (rows * columns, rows));
    if columns <= LOOP_MAX_COLUMNS
        && with_fused_multiply_add(TransposeTimesVector {
            matrix,
            weights,
            target: &mut *target,
        })
    {
        return;
    }
    use faer::{col, linalg::matmul::matmul, mat};
    matmul(
        col::from_slice_mut::<f64>(target).as_2d_mut(),
        mat::from_row_major_slice::<f64>(matrix, rows, columns).transpose(),
        col::from_slice::<f64>(weights).as_2d(),
        Some(1.0),
        1.0,
        FAER_PARALLELISM,
    );
}

/// Runs `kernel` where `f64::mul_add` is a hardware fused multiply-add, and
/// returns whether it ran.
///
/// The loops accumulate with `mul_add`, as faer's kernels do on a CPU with
/// FMA: one rounding per term, so a product added straight into the
/// accumulator that brings it back into range does not overflow
/// (`2 * 1e308 - 1e308` is `1e308`, not `inf`). A product landing in an
/// accumulator that cannot, or a running sum that overflows on its own,
/// still gives `inf`. Without the
/// hardware instruction `mul_add` is a slow library call, so on x86-64 CPUs
/// without AVX2 and FMA this returns `false` and the caller uses faer.
fn with_fused_multiply_add<K: pulp::WithSimd<Output = ()>>(kernel: K) -> bool {
    if !has_fused_multiply_add() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let simd = pulp::x86::V3::try_new().expect("checked by has_fused_multiply_add");
        <pulp::x86::V3 as pulp::Simd>::vectorize(simd, kernel);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Fused multiply-add is part of the base aarch64 instruction set.
        kernel.with_simd(pulp::Scalar::new());
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = kernel;
    true
}

/// Whether the narrow products run as fused loops on this CPU, rather than
/// falling back to faer.
fn has_fused_multiply_add() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        pulp::x86::V3::try_new().is_some()
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

struct MatrixTimesVector<'a> {
    matrix: &'a [f64],
    beta: &'a [f64],
    out: &'a mut [f64],
}

impl pulp::WithSimd for MatrixTimesVector<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) {
        // Each output already holds its intercept, which the dot product
        // starts from, so the first column's product is fused with it.
        let columns = self.beta.len();
        let mut blocks = self.matrix.chunks_exact(ROWS_PER_BLOCK * columns);
        let mut outs = self.out.chunks_exact_mut(ROWS_PER_BLOCK);
        for (block, out) in (&mut blocks).zip(&mut outs) {
            dot_rows_onto(block, self.beta, out);
        }
        let rest = blocks.remainder();
        for (row, out) in rest.chunks_exact(columns).zip(outs.into_remainder()) {
            *out = dot_onto(row, self.beta, *out);
        }
    }
}

struct TransposeTimesVector<'a> {
    matrix: &'a [f64],
    weights: &'a [f64],
    target: &'a mut [f64],
}

impl pulp::WithSimd for TransposeTimesVector<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) {
        let columns = self.target.len();
        if columns > STREAM_MIN_COLUMNS {
            add_rows_streaming(self.matrix, self.weights, self.target);
            return;
        }
        let mut start = 0;
        while columns - start >= 32 {
            add_column_block::<32>(self.matrix, columns, start, self.weights, self.target);
            start += 32;
        }
        for width in [16, 8, 4] {
            if columns - start >= width {
                match width {
                    16 => add_column_block::<16>(
                        self.matrix,
                        columns,
                        start,
                        self.weights,
                        self.target,
                    ),
                    8 => add_column_block::<8>(
                        self.matrix,
                        columns,
                        start,
                        self.weights,
                        self.target,
                    ),
                    _ => add_column_block::<4>(
                        self.matrix,
                        columns,
                        start,
                        self.weights,
                        self.target,
                    ),
                }
                start += width;
            }
        }
        for column in start..columns {
            add_column_block::<1>(self.matrix, columns, column, self.weights, self.target);
        }
    }
}

/// The parallelism the wide products hand faer.
///
/// Both have a single column, and faer 0.19 routes every single-column product
/// to a serial kernel without reading this argument:
/// `linalg::matmul::matmul` calls `matmul_with_conj_gemm_dispatch`, whose
/// `m == 1 && n == 1`, `k == 1` and `n == 1` branches return before the
/// parallel gemm path (`src/linalg/matmul/mod.rs` in faer 0.19.4). Chains
/// already run in parallel on the Rayon pool. Revisit this on a faer upgrade,
/// where `mat_vec_mul_is_bitwise_identical_across_thread_counts` is the check
/// that a parallel product still gives seeded sampling the same bits.
const FAER_PARALLELISM: faer::Parallelism<'static> = faer::Parallelism::None;

/// Rows the forward product interleaves. `dot_onto` keeps four accumulator
/// lanes per row; running four rows side by side puts sixteen independent
/// chains in flight and loads each slice of `b` once for all four rows.
const ROWS_PER_BLOCK: usize = 4;

/// `out[r] = dot_onto(row r of block, b, out[r])` for `ROWS_PER_BLOCK` rows
/// at once. Each row's accumulators take the same products in the same order
/// as in `dot_onto`, so every result has the same bits, except that which payload survives when two NaNs meet is the compiler's choice; the rows only
/// share the loads of `b`.
#[inline(always)]
fn dot_rows_onto(block: &[f64], b: &[f64], out: &mut [f64]) {
    let columns = b.len();
    let mut lanes = [[0.0f64; 4]; ROWS_PER_BLOCK];
    let mut tails = [0.0f64; ROWS_PER_BLOCK];
    for r in 0..ROWS_PER_BLOCK {
        if columns >= 4 {
            lanes[r][0] = out[r];
        } else {
            tails[r] = out[r];
        }
    }
    let rows: [&[f64]; ROWS_PER_BLOCK] =
        std::array::from_fn(|r| &block[r * columns..(r + 1) * columns]);
    let whole = columns - columns % 4;
    for r in 0..ROWS_PER_BLOCK {
        for (x, y) in rows[r][whole..].iter().zip(&b[whole..]) {
            tails[r] = x.mul_add(*y, tails[r]);
        }
    }
    for c in (0..whole).step_by(4) {
        let y = &b[c..c + 4];
        for r in 0..ROWS_PER_BLOCK {
            let x = &rows[r][c..c + 4];
            for lane in 0..4 {
                lanes[r][lane] = x[lane].mul_add(y[lane], lanes[r][lane]);
            }
        }
    }
    for r in 0..ROWS_PER_BLOCK {
        let l = lanes[r];
        out[r] = (l[0] + l[1]) + (l[2] + l[3]) + tails[r];
    }
}

/// Above this many columns the transpose product streams four rows at a time
/// across the columns; up to it, blocks of columns stay in registers across
/// all rows, which is faster while a block is a few registers wide.
const STREAM_MIN_COLUMNS: usize = 64;

/// The transpose update four rows at a time, streaming across the columns:
/// each sum still takes the rows' fused products in row order, so the bits are
/// those of the row-by-row update, except that which payload survives when two NaNs meet is the compiler's choice.
#[inline(always)]
fn add_rows_streaming(matrix: &[f64], weights: &[f64], target: &mut [f64]) {
    let columns = target.len();
    let mut blocks = matrix.chunks_exact(4 * columns);
    let mut block_weights = weights.chunks_exact(4);
    for (block, w) in (&mut blocks).zip(&mut block_weights) {
        let (r0, rest) = block.split_at(columns);
        let (r1, rest) = rest.split_at(columns);
        let (r2, r3) = rest.split_at(columns);
        for ((((sum, &a0), &a1), &a2), &a3) in target.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3) {
            let mut s = *sum;
            s = w[0].mul_add(a0, s);
            s = w[1].mul_add(a1, s);
            s = w[2].mul_add(a2, s);
            s = w[3].mul_add(a3, s);
            *sum = s;
        }
    }
    for (row, &weight) in blocks
        .remainder()
        .chunks_exact(columns)
        .zip(block_weights.remainder())
    {
        for (sum, &entry) in target.iter_mut().zip(row) {
            *sum = weight.mul_add(entry, *sum);
        }
    }
}

/// `target[start + j] += sum_i matrix[i][start + j] * weights[i]` for
/// `j < W`, accumulating in registers across all rows. Each sum takes the
/// rows' fused products in row order, as the row-by-row update did, so the
/// bits are the same, except that which payload survives when two NaNs meet is the compiler's choice.
#[inline(always)]
fn add_column_block<const W: usize>(
    matrix: &[f64],
    columns: usize,
    start: usize,
    weights: &[f64],
    target: &mut [f64],
) {
    let mut sums: [f64; W] = target[start..start + W].try_into().unwrap();
    for (row, &weight) in matrix.chunks_exact(columns).zip(weights) {
        let entries: &[f64; W] = row[start..start + W].try_into().unwrap();
        for j in 0..W {
            sums[j] = weight.mul_add(entries[j], sums[j]);
        }
    }
    target[start..start + W].copy_from_slice(&sums);
}

/// `start + a · b`, in four interleaved fused accumulators so the summation
/// order is fixed and does not depend on threads. `start` begins the
/// accumulator that takes the first column, so the first column's product is
/// fused with it: with one column the result is the single fused
/// `a[0] * b[0] + start`, as in faer.
#[inline(always)]
fn dot_onto(a: &[f64], b: &[f64], start: f64) -> f64 {
    let mut lanes = [0.0f64; 4];
    let (a_chunks, b_chunks) = (a.chunks_exact(4), b.chunks_exact(4));
    let mut tail = 0.0f64;
    if a.len() >= 4 {
        lanes[0] = start;
    } else {
        tail = start;
    }
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        tail = x.mul_add(*y, tail);
    }
    for (x, y) in a_chunks.zip(b_chunks) {
        for lane in 0..4 {
            lanes[lane] = x[lane].mul_add(y[lane], lanes[lane]);
        }
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail
}

/// The summed log density of `n` elements whose constant arguments are
/// outside their support: `-inf`, or the empty sum.
fn invalid_vector_logp(n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        f64::NEG_INFINITY
    }
}

/// Whether `sigma` is a usable scale: finite and strictly positive.
#[inline]
fn scale_is_valid(sigma: f64) -> bool {
    sigma.is_finite() && sigma > 0.0
}

/// Add `a_s` times the gradient of a Normal log density, summed over
/// observations, with respect to each mean and to the scale `sigma`, given the
/// standardized residuals `(y - mu) / sigma`. The arithmetic and its order are
/// those of the loop that recomputed each residual from `y` and `mu`, so every
/// value is the same to the bit, except that which payload survives when two
/// NaNs meet in an adjoint is the compiler's choice, as Rust leaves it.
fn gaussian_residual_adjoints(
    residuals: &[f64],
    sigma: f64,
    a_s: f64,
    mu_adjoints: Option<&mut [f64]>,
    sigma_adjoint: Option<&mut f64>,
) {
    if let Some(adjoints) = mu_adjoints {
        for (adjoint, &d) in adjoints.iter_mut().zip(residuals) {
            *adjoint += a_s * (d / sigma);
        }
    }
    if let Some(sigma_adjoint) = sigma_adjoint {
        let mut dsigma = 0.0f64;
        for &d in residuals {
            dsigma += (d * d - 1.0) / sigma;
        }
        *sigma_adjoint += a_s * dsigma;
    }
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
//
// Each density is split into a validity check and a kernel that takes the
// logarithms of its constant arguments, so the vector priors -- whose scale,
// shape and rate are constants of the op -- take those logarithms once per op
// instead of once per element. The kernels perform the same operations in the
// same order either way, so hoisting changes no bits; see
// `vector_priors_match_their_scalar_densities_bit_for_bit`.
fn log_half_normal_logp(raw: f64, sigma: f64) -> f64 {
    if !scale_is_valid(sigma) {
        return f64::NEG_INFINITY;
    }
    log_half_normal_kernel(raw, sigma.ln())
}

fn log_half_normal_kernel(raw: f64, ln_sigma: f64) -> f64 {
    let z = raw - ln_sigma;
    0.5 * (2.0 / std::f64::consts::PI).ln() + z - 0.5 * (2.0 * z).exp()
}

fn gamma_args_valid(alpha: f64, beta: f64) -> bool {
    alpha.is_finite() && alpha > 0.0 && beta.is_finite() && beta > 0.0
}

fn log_gamma_logp(raw: f64, alpha: f64, beta: f64) -> f64 {
    if !gamma_args_valid(alpha, beta) {
        return f64::NEG_INFINITY;
    }
    log_gamma_kernel(raw, alpha, beta.ln(), ln_gamma(alpha))
}

fn log_gamma_kernel(raw: f64, alpha: f64, ln_beta: f64, ln_gamma_alpha: f64) -> f64 {
    let z = raw + ln_beta;
    alpha * z - ln_gamma_alpha - z.exp()
}

fn student_t_args_valid(nu: f64, sigma: f64) -> bool {
    nu.is_finite() && nu > 0.0 && scale_is_valid(sigma)
}

fn student_t_logp_scalar(x: f64, nu: f64, mu: f64, sigma: f64) -> f64 {
    if !student_t_args_valid(nu, sigma) {
        return f64::NEG_INFINITY;
    }
    let (ln_nu, ln_sigma) = (nu.ln(), sigma.ln());
    student_t_kernel(
        x - mu,
        nu,
        ln_nu,
        ln_sigma,
        student_t_log_normaliser(nu, ln_nu, ln_sigma),
    )
}

/// The terms of the Student-t log density that do not depend on `x`, summed
/// left to right as [`student_t_kernel`] continues them.
fn student_t_log_normaliser(nu: f64, ln_nu: f64, ln_sigma: f64) -> f64 {
    ln_gamma(0.5 * (nu + 1.0))
        - ln_gamma(0.5 * nu)
        - 0.5 * (ln_nu + std::f64::consts::PI.ln())
        - ln_sigma
}

fn student_t_kernel(diff: f64, nu: f64, ln_nu: f64, ln_sigma: f64, log_normaliser: f64) -> f64 {
    let log_ratio = 2.0 * (diff.abs().ln() - ln_sigma) - ln_nu;
    log_normaliser - 0.5 * (nu + 1.0) * softplus(log_ratio)
}

/// d/dx of the Student-t log density at `x - mu == diff`, from the logs of
/// `sigma`, `nu` and `nu + 1`. Log-ratio arithmetic retains the polynomial
/// tails even where the squared residual overflows.
fn student_t_dx(diff: f64, ln_sigma: f64, ln_nu: f64, ln_nu1: f64) -> f64 {
    if diff == 0.0 {
        return 0.0;
    }
    let log_abs_diff = diff.abs().ln();
    let log_tail = softplus(2.0 * (log_abs_diff - ln_sigma) - ln_nu);
    student_t_dx_from_logs(diff, log_abs_diff, log_tail, ln_sigma, ln_nu, ln_nu1)
}

/// [`student_t_dx`] at a nonzero `diff`, for a caller that already has
/// `ln|diff|` and the log tail `softplus(2 (ln|diff| - ln sigma) - ln nu)`.
fn student_t_dx_from_logs(
    diff: f64,
    log_abs_diff: f64,
    log_tail: f64,
    ln_sigma: f64,
    ln_nu: f64,
    ln_nu1: f64,
) -> f64 {
    -diff.signum() * (ln_nu1 + log_abs_diff - 2.0 * ln_sigma - ln_nu - log_tail).exp()
}

// Return derivatives with respect to x, sigma, and nu. Log-ratio arithmetic
// retains Student-t's polynomial tails even when the squared residual overflows.
// Each logarithm and the softplus are taken once and shared by all three.
fn student_t_derivatives(x: f64, nu: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
    let diff = x - mu;
    let (ln_sigma, ln_nu) = (sigma.ln(), nu.ln());
    let log_abs_diff = diff.abs().ln();
    let log_ratio = 2.0 * (log_abs_diff - ln_sigma) - ln_nu;
    let log_tail = softplus(log_ratio);
    let weight = stable_sigmoid(log_ratio);
    let dx = if diff == 0.0 {
        0.0
    } else {
        let ln_nu1 = (nu + 1.0).ln();
        student_t_dx_from_logs(diff, log_abs_diff, log_tail, ln_sigma, ln_nu, ln_nu1)
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
///
/// The evaluator calls the `_given` form below with the exponential it keeps
/// for the reverse pass; this form is the reference the tests compare it with.
#[cfg(test)]
pub(crate) fn bernoulli_logit_logp(y: f64, eta: f64) -> f64 {
    bernoulli_logit_logp_given(y, eta, bernoulli_logit_exp(y, eta))
}

/// The one exponential [`bernoulli_logit_logp`] and [`bernoulli_logit_grad`]
/// both take, so the evaluator takes it once per observation for both passes.
///
/// Each evaluates `softplus` and `stable_sigmoid` at `x = -eta` when `y` is 1
/// and at `x = eta` otherwise, and every branch of those two exponentiates
/// `-x` when `x > 0` (or, for the sigmoid, `x >= 0`, which differs only at
/// zero, where both exponents give 1) and `x` itself otherwise. This takes
/// the same exponent, so the values are the same to the bit, NaN included.
fn bernoulli_logit_exp(y: f64, eta: f64) -> f64 {
    let x = if y == 1.0 { -eta } else { eta };
    if x > 0.0 {
        (-x).exp()
    } else {
        x.exp()
    }
}

/// [`bernoulli_logit_logp`] given `e` from [`bernoulli_logit_exp`].
fn bernoulli_logit_logp_given(y: f64, eta: f64, e: f64) -> f64 {
    // `softplus(x)` with `e = exp(-|x|)`.
    let softplus = |x: f64| if x > 0.0 { x + e.ln_1p() } else { e.ln_1p() };
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
///
/// The evaluator calls the `_given` form below with the exponential it keeps
/// for the reverse pass; this form is the reference the tests compare it with.
#[cfg(test)]
pub(crate) fn bernoulli_logit_grad(y: f64, eta: f64) -> f64 {
    bernoulli_logit_grad_given(y, eta, bernoulli_logit_exp(y, eta))
}

/// [`bernoulli_logit_grad`] given `e` from [`bernoulli_logit_exp`].
fn bernoulli_logit_grad_given(y: f64, eta: f64, e: f64) -> f64 {
    // `stable_sigmoid(x)` with `e = exp(-|x|)`.
    let sigmoid = |x: f64| {
        if x >= 0.0 {
            1.0 / (1.0 + e)
        } else {
            e / (1.0 + e)
        }
    };
    if y == 1.0 {
        sigmoid(-eta)
    } else if y == 0.0 {
        -sigmoid(eta)
    } else {
        y - sigmoid(eta)
    }
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
        g.obs_logp_normal(mu, one, obs);

        let mut full = Evaluator::new(&g);
        full.compute(&g, &[0.7]);
        let mut forward = Evaluator::new(&g);
        forward.forward(&g, &[0.7]);
        assert_eq!(forward.total_logp, full.total_logp);
        for i in 0..3 {
            assert_eq!(forward.vec_elem(mu, i), full.vec_elem(mu, i));
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
        g.obs_logp_normal(mu_vec, sigma, obs_idx);

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
        g.obs_logp_normal(mu_vec, sigma, obs_idx);

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
        g.obs_logp_normal(mu_vec, sigma, obs_idx);

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

    /// `nu` and `sigma` as learned parameters, so the `dnu` and `dsigma`
    /// branches of the score are checked too; the test above holds them
    /// constant.
    #[test]
    fn student_t_gradient_in_learned_nu_and_sigma_matches_finite_differences() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let raw_nu = g.add_param("raw_nu");
        let raw_sigma = g.add_param("raw_sigma");
        let mu = g.add_constant(0.4);
        let nu = g.exp(raw_nu);
        let sigma = g.exp(raw_sigma);
        g.student_t_logp(x, nu, mu, sigma);
        for params in [[1.8, 1.2, 0.3], [-0.7, -0.5, -1.1], [6.0, 2.5, 0.0]] {
            full_finite_diff_check(&g, &params, 1e-5);
        }
    }

    /// Outside the support of `nu` or `sigma` the density is a constant -inf,
    /// and its score is zero rather than NaN.
    #[test]
    fn student_t_outside_its_support_has_a_zero_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let nu = g.add_param("nu");
        let sigma = g.add_param("sigma");
        let mu = g.add_constant(0.0);
        g.student_t_logp(x, nu, mu, sigma);
        for params in [
            [0.5, 3.0, -1.0],
            [0.5, 3.0, 0.0],
            [0.5, -2.0, 1.0],
            [0.5, 0.0, 1.0],
        ] {
            let mut evaluator = Evaluator::new(&g);
            evaluator.compute(&g, &params);
            assert_eq!(evaluator.total_logp, f64::NEG_INFINITY, "{params:?}");
            assert_eq!(evaluator.grad, vec![0.0; 3], "{params:?}");
            assert_eq!(grad_logp(&g, &params), (f64::NEG_INFINITY, vec![0.0; 3]));
        }
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
        g.obs_logp_normal(mu_node, sigma1, obs_idx);

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
            g.obs_logp_normal(mu, sigma, obs_idx);

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
                    (evaluator.vec_elem(mu, i) - expected_value).abs() < 1e-12,
                    "transform {:?}, row {}",
                    transform,
                    i
                );
            }
        }
    }

    /// The evaluator takes the exponential once and passes it to both passes;
    /// that must give the same bits as the formulas that take it themselves,
    /// across signs, zeros, saturation, infinities and NaNs of either sign.
    #[test]
    fn bernoulli_logit_given_forms_match_the_reference_bit_for_bit() {
        let etas = [
            0.0,
            -0.0,
            1e-300,
            -1e-300,
            0.5,
            -0.5,
            3.0,
            -3.0,
            40.0,
            -40.0,
            745.0,
            -745.0,
            1e308,
            -1e308,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff8_0000_0000_1234),
        ];
        for &eta in &etas {
            for y in [0.0, 1.0, 0.25] {
                let e = bernoulli_logit_exp(y, eta);
                // Bits, NaN sign and payload included.
                let same = |a: f64, b: f64| a.to_bits() == b.to_bits();
                assert!(
                    same(
                        bernoulli_logit_logp_given(y, eta, e),
                        bernoulli_logit_logp(y, eta)
                    ),
                    "logp y={y} eta={eta}"
                );
                assert!(
                    same(
                        bernoulli_logit_grad_given(y, eta, e),
                        bernoulli_logit_grad(y, eta)
                    ),
                    "grad y={y} eta={eta}"
                );
                // And against the formulas before they shared `e`.
                let old_logp = if y == 1.0 {
                    -softplus(-eta)
                } else if y == 0.0 {
                    -softplus(eta)
                } else {
                    y * eta - softplus(eta)
                };
                let old_grad = if y == 1.0 {
                    stable_sigmoid(-eta)
                } else if y == 0.0 {
                    -stable_sigmoid(eta)
                } else {
                    y - stable_sigmoid(eta)
                };
                assert!(
                    same(bernoulli_logit_logp(y, eta), old_logp),
                    "old logp y={y} eta={eta}"
                );
                assert!(
                    same(bernoulli_logit_grad(y, eta), old_grad),
                    "old grad y={y} eta={eta}"
                );
            }
        }
    }

    /// The narrow products accumulate with fused multiply-adds, so a gradient
    /// whose sum is representable stays finite when a product along the way
    /// is not: `1e308 * 2 - 1e308` is `1e308` fused and `inf` unfused. faer's
    /// kernels behaved the same way on a CPU with FMA, and the sampler rejects
    /// a non-finite gradient.
    #[test]
    fn narrow_mat_vec_mul_gradient_survives_an_intermediate_overflow() {
        let mut g = Graph::new();
        let start = g.add_vector_params("beta", 2);
        g.vector_normal_logp(start, 2, 0.0, 1.0);
        let matrix_idx = g.store_matrix(vec![-1e308, 0.0, 1e308, 0.0], 2, 2);
        let mu = g.mat_vec_mul(matrix_idx, start, 2, None);
        let sigma = g.add_constant(1.0);
        let obs_idx = g.add_obs_data(vec![1.0, 2.0]);
        g.obs_logp_normal(mu, sigma, obs_idx);
        let mut evaluator = Evaluator::new(&g);
        evaluator.compute(&g, &[0.0, 0.0]);
        // faer's scalar fallback, on a CPU without FMA, multiplies and adds
        // separately and gives `inf` here.
        if has_fused_multiply_add() {
            assert_eq!(evaluator.grad, vec![1e308, 0.0]);
        }
    }

    /// The forward product starts the first column's accumulator from the
    /// intercept, so `2 * 1e308` in the first column, brought back by an
    /// intercept of `-1e308`, is fused with it instead of overflowing first,
    /// with one column (as in faer's one-column kernel) and with four or more.
    #[test]
    fn narrow_mat_vec_mul_fuses_the_intercept_with_the_first_column() {
        for columns in [1usize, 3, 4, 9] {
            let mut g = Graph::new();
            let start = g.add_vector_params("beta", columns);
            g.vector_normal_logp(start, columns, 0.0, 1.0);
            let intercept = g.add_constant(-1e308);
            let mut design = vec![0.0; 2 * columns];
            design[0] = 1e308;
            design[columns] = 1e308;
            let matrix_idx = g.store_matrix(design, 2, columns);
            let mu = g.mat_vec_mul(matrix_idx, start, columns, Some(intercept));
            let mut beta = vec![0.0; columns];
            beta[0] = 2.0;
            let mut evaluator = Evaluator::new(&g);
            evaluator.compute(&g, &beta);
            if has_fused_multiply_add() {
                assert_eq!(evaluator.vec_elem(mu, 0), 1e308, "{columns} columns");
                assert_eq!(evaluator.vec_elem(mu, 1), 1e308, "{columns} columns");
            }
        }
    }

    /// The product loops interleave rows and keep blocks of columns in
    /// registers, which must not change any result: every output of either
    /// product has the bits of the row-by-row loops it replaced, which are
    /// reproduced here. The shapes cover every row count modulo the four rows
    /// interleaved, every width modulo four, and the widths on either side of
    /// each column-block size and of the 64 columns where the reverse product
    /// starts streaming. One pass uses moderate entries, so that every output
    /// is finite and a change of order shows in its bits; a second adds
    /// signed zeros, values near overflow and an infinity. No case has two
    /// NaNs with different payloads, whose sum keeps a payload of the
    /// compiler's choosing. `mul_add` is correctly rounded with or without a
    /// hardware instruction, so the scalar path runs everywhere and the
    /// dispatched one wherever it exists.
    #[test]
    fn blocked_product_loops_match_the_row_by_row_loops_bit_for_bit() {
        fn row_by_row(matrix: &[f64], beta: &[f64], out: &mut [f64]) {
            for (row, out) in matrix.chunks_exact(beta.len()).zip(out) {
                *out = dot_onto(row, beta, *out);
            }
        }
        fn transpose_row_by_row(matrix: &[f64], weights: &[f64], target: &mut [f64]) {
            let columns = target.len();
            for (row, &weight) in matrix.chunks_exact(columns).zip(weights) {
                for (sum, &entry) in target.iter_mut().zip(row) {
                    *sum = weight.mul_add(entry, *sum);
                }
            }
        }
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut draw = move |extreme: bool| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match state % 37 {
                0 if extreme => -0.0,
                1 if extreme => 1e308,
                2 if extreme => -1e308,
                3 if extreme => f64::INFINITY,
                _ => (state >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0,
            }
        };
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let row_counts = [1usize, 2, 3, 4, 5, 6, 8, 10, 11];
        let widths = [
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 31, 32, 33, 34, 47, 63, 64, 65, 66,
            100,
        ];
        let mut finite_moderate_outputs = 0;
        for extreme in [false, true] {
            for rows in row_counts {
                for columns in widths {
                    let mut next = || draw(extreme);
                    let matrix = (0..rows * columns).map(|_| next()).collect::<Vec<_>>();
                    let beta = (0..columns).map(|_| next()).collect::<Vec<_>>();
                    let weights = (0..rows).map(|_| next()).collect::<Vec<_>>();
                    let intercepts = (0..rows).map(|_| next()).collect::<Vec<_>>();
                    let start = (0..columns).map(|_| next()).collect::<Vec<_>>();

                    let mut expected = intercepts.clone();
                    row_by_row(&matrix, &beta, &mut expected);
                    let mut expected_target = start.clone();
                    transpose_row_by_row(&matrix, &weights, &mut expected_target);
                    if !extreme {
                        finite_moderate_outputs += expected
                            .iter()
                            .chain(&expected_target)
                            .filter(|x| x.is_finite())
                            .count();
                    }

                    let mut scalar = intercepts.clone();
                    pulp::WithSimd::with_simd(
                        MatrixTimesVector {
                            matrix: &matrix,
                            beta: &beta,
                            out: &mut scalar,
                        },
                        pulp::Scalar::new(),
                    );
                    let mut scalar_target = start.clone();
                    pulp::WithSimd::with_simd(
                        TransposeTimesVector {
                            matrix: &matrix,
                            weights: &weights,
                            target: &mut scalar_target,
                        },
                        pulp::Scalar::new(),
                    );
                    assert_eq!(bits(&scalar), bits(&expected), "{rows} x {columns}");
                    assert_eq!(
                        bits(&scalar_target),
                        bits(&expected_target),
                        "{rows} x {columns}"
                    );

                    let mut dispatched = intercepts.clone();
                    let mut dispatched_target = start.clone();
                    let ran = with_fused_multiply_add(MatrixTimesVector {
                        matrix: &matrix,
                        beta: &beta,
                        out: &mut dispatched,
                    }) && with_fused_multiply_add(TransposeTimesVector {
                        matrix: &matrix,
                        weights: &weights,
                        target: &mut dispatched_target,
                    });
                    assert_eq!(ran, has_fused_multiply_add());
                    if ran {
                        assert_eq!(bits(&dispatched), bits(&expected), "{rows} x {columns}");
                        assert_eq!(
                            bits(&dispatched_target),
                            bits(&expected_target),
                            "{rows} x {columns}"
                        );
                    }
                }
            }
        }
        // Every output of the moderate pass is finite, so its bits compare
        // ordinary sums rather than infinities and NaNs.
        let moderate_outputs: usize = row_counts
            .iter()
            .map(|rows| widths.iter().map(|columns| rows + columns).sum::<usize>())
            .sum();
        assert_eq!(finite_moderate_outputs, moderate_outputs);
    }

    /// Designs of 65 to 128 columns moved from faer to the loops. Both products
    /// must stay within the rounding error any order of `n` fused or separate
    /// multiply-adds can make, `gamma_n * sum |a_i b_i|` with
    /// `gamma_n = n u / (1 - n u)`, of a compensated sum, at widths through and
    /// beyond the change, at row counts that leave a remainder of the four rows
    /// the loops interleave. The bound assumes that no product, partial sum or
    /// rounding error overflows or underflows. The entries are uniform on
    /// [-1, 1] in steps of 2^-52, which keeps every exponent far from both
    /// ends of the range. Outside it the bound need not hold: the loops can
    /// overflow on a sum that another order would cancel first, and the
    /// reference can lose a rounding error that is not representable.
    #[test]
    fn wide_products_stay_within_the_rounding_error_bound() {
        // Neumaier's compensated sum of each product and its rounding error,
        // which `mul_add` gives exactly when that error is representable, as
        // it is when the factors' exponents sit well above the subnormal range,
        // as here. Under that condition and without overflow the sum is
        // accurate to about one rounding of the result. Also returns the sum
        // of the products' magnitudes.
        fn reference(terms: impl Iterator<Item = (f64, f64)>, start: f64) -> (f64, f64) {
            let (mut sum, mut compensation, mut magnitude) = (start, 0.0f64, start.abs());
            for (a, b) in terms {
                let product = a * b;
                let error = a.mul_add(b, -product);
                for term in [product, error] {
                    let next = sum + term;
                    compensation += if sum.abs() >= term.abs() {
                        (sum - next) + term
                    } else {
                        (term - next) + sum
                    };
                    sum = next;
                }
                magnitude += product.abs();
            }
            (sum + compensation, magnitude)
        }
        let mut state = 0x8f1b_bcdc_bfa5_3e0bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let unit = f64::EPSILON / 2.0;
        for (rows, columns) in [
            (5usize, 65usize),
            (7, 96),
            (9, 100),
            (13, 128),
            (6, 129),
            (3, 200),
        ] {
            let matrix = (0..rows * columns).map(|_| next()).collect::<Vec<_>>();
            let beta = (0..columns).map(|_| next()).collect::<Vec<_>>();
            let weights = (0..rows).map(|_| next()).collect::<Vec<_>>();
            let intercepts = (0..rows).map(|_| next()).collect::<Vec<_>>();
            let start = (0..columns).map(|_| next()).collect::<Vec<_>>();

            let mut out = intercepts.clone();
            add_matrix_times_vector(&matrix, rows, &beta, &mut out);
            let gamma = |n: usize| n as f64 * unit / (1.0 - n as f64 * unit);
            for (i, &value) in out.iter().enumerate() {
                let row = &matrix[i * columns..(i + 1) * columns];
                let (exact, magnitude) =
                    reference(row.iter().copied().zip(beta.iter().copied()), intercepts[i]);
                let bound = gamma(columns + 1) * magnitude + unit * exact.abs();
                assert!(
                    (value - exact).abs() <= bound,
                    "{rows} x {columns}, row {i}"
                );
            }

            let mut target = start.clone();
            add_transpose_times_vector(&matrix, rows, &weights, &mut target);
            for (k, &value) in target.iter().enumerate() {
                let column = (0..rows).map(|i| (matrix[i * columns + k], weights[i]));
                let (exact, magnitude) = reference(column, start[k]);
                let bound = gamma(rows + 1) * magnitude + unit * exact.abs();
                assert!(
                    (value - exact).abs() <= bound,
                    "{rows} x {columns}, column {k}"
                );
            }
        }
    }

    /// The loops' summation order decides whether a sum that cancels overflows
    /// first, at 65 to 128 columns as at narrower widths. Here every row's
    /// exact product is 0, but its first accumulator lane takes columns 0, 4,
    /// 8 and 12, `1e308 + 1e308 - 1e308 - 1e308`, and overflows on the second
    /// term. faer 0.19, which these widths used before, returns a finite
    /// value for this design; this pins the loops' result so the trade is
    /// visible.
    #[test]
    fn wide_product_loops_can_overflow_on_a_sum_that_cancels() {
        let (rows, columns) = (4usize, 96usize);
        let mut matrix = vec![0.0; rows * columns];
        for row in matrix.chunks_exact_mut(columns) {
            row[0] = 1e308;
            row[4] = 1e308;
            row[8] = -1e308;
            row[12] = -1e308;
        }
        let beta = vec![1.0; columns];
        let mut out = vec![0.0; rows];
        add_matrix_times_vector(&matrix, rows, &beta, &mut out);
        if has_fused_multiply_add() {
            assert_eq!(out, vec![f64::INFINITY; rows]);
        }
    }

    /// A Normal regression on a 100-column design, which now runs on the
    /// loops, against finite differences.
    #[test]
    fn wide_design_gradients_match_finite_differences() {
        let (rows, columns) = (9usize, 100usize);
        let mut g = Graph::new();
        let start = g.add_vector_params("beta", columns);
        g.vector_normal_logp(start, columns, 0.0, 1.0);
        let design = (0..rows * columns)
            .map(|i| ((i * 37) % 23) as f64 / 11.0 - 1.0)
            .collect();
        let matrix = g.store_matrix(design, rows, columns);
        let intercept = g.add_param("intercept");
        let eta = g.mat_vec_mul(matrix, start, columns, Some(intercept));
        let sigma = g.add_constant(1.5);
        let obs = g.add_obs_data((0..rows).map(|i| i as f64 / 3.0 - 1.0).collect());
        g.obs_logp_normal(eta, sigma, obs);
        let params = (0..=columns)
            .map(|k| ((k * 13) % 7) as f64 / 10.0 - 0.3)
            .collect::<Vec<_>>();
        full_finite_diff_check(&g, &params, 2e-5);
    }

    /// Seeded sampling must not depend on how many threads the pool has. With
    /// faer 0.19 the wide products are serial whatever `FAER_PARALLELISM` says
    /// (see its documentation), and the narrow ones are loops, so today this
    /// holds by construction; it is here for the faer upgrade that makes a
    /// parallel product possible. The 400 x 300 design is large enough that
    /// faer would split it.
    #[test]
    fn mat_vec_mul_is_bitwise_identical_across_thread_counts() {
        let (rows, cols) = (400usize, 300usize);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut g = Graph::new();
        let start = g.add_vector_params("beta", cols);
        g.vector_normal_logp(start, cols, 0.0, 1.0);
        let one = g.add_constant(1.0);
        let matrix: Vec<f64> = (0..rows * cols).map(|_| next()).collect();
        let matrix_idx = g.store_matrix(matrix, rows, cols);
        let mu = g.mat_vec_mul(matrix_idx, start, cols, None);
        let obs = g.add_obs_data((0..rows).map(|_| next()).collect());
        g.obs_logp_normal(mu, one, obs);
        let params: Vec<f64> = (0..cols).map(|_| next()).collect();

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut evaluator = Evaluator::new(&g);
                    evaluator.compute(&g, &params);
                    let bits: Vec<u64> = evaluator.grad.iter().map(|x| x.to_bits()).collect();
                    (evaluator.total_logp.to_bits(), bits)
                })
        };
        let serial = run(1);
        for threads in [2, 4, 7] {
            assert_eq!(run(threads), serial, "{threads} threads");
        }
    }

    /// Every density op and every elementwise operator, with each of its
    /// inputs a parameter (through `exp` or `sigmoid` where the input has a
    /// restricted domain), against central finite differences. The reference
    /// evaluator shares the derivative kernels, so this is what checks them.
    #[test]
    fn every_density_and_operator_matches_finite_differences() {
        use crate::graph::{ElementwiseOp as E, ParamTransform};
        let mut cases: Vec<(String, Graph)> = Vec::new();
        let mut case = |name: &str, build: &dyn Fn(&mut Graph)| {
            let mut g = Graph::new();
            build(&mut g);
            cases.push((name.to_string(), g));
        };
        let positive = |g: &mut Graph, name: &str| {
            let raw = g.add_param(name);
            g.exp(raw)
        };

        case("NormalLogP", &|g| {
            let (x, mu) = (g.add_param("x"), g.add_param("mu"));
            let sigma = positive(g, "sigma");
            g.normal_logp(x, mu, sigma);
        });
        case("LogHalfNormalLogP", &|g| {
            let x = g.add_param("x");
            let sigma = positive(g, "sigma");
            g.log_half_normal_logp(x, sigma);
        });
        case("StudentTLogP", &|g| {
            let (x, mu) = (g.add_param("x"), g.add_param("mu"));
            let nu = positive(g, "nu");
            let sigma = positive(g, "sigma");
            g.student_t_logp(x, nu, mu, sigma);
        });
        for outcome in [0.0, 1.0] {
            case(&format!("BernoulliLogP({outcome})"), &|g| {
                let raw = g.add_param("p");
                let p = g.sigmoid(raw);
                let x = g.add_constant(outcome);
                g.bernoulli_logp(x, p);
            });
        }
        case("PoissonLogP", &|g| {
            let lam = positive(g, "lam");
            let x = g.add_constant(3.0);
            g.poisson_logp(x, lam);
        });
        case("LogGammaLogP", &|g| {
            let x = g.add_param("x");
            let alpha = positive(g, "alpha");
            let beta = positive(g, "beta");
            g.log_gamma_logp(x, alpha, beta);
        });
        case("PositiveSupport", &|g| {
            let x = positive(g, "x");
            g.positive_support(x);
            let zero = g.add_constant(0.0);
            let one = g.add_constant(1.0);
            g.normal_logp(x, zero, one);
        });

        let families: [(&str, Vec<f64>); 6] = [
            ("Normal", vec![0.3, -1.2, 2.0, 0.7]),
            ("BernoulliLogit", vec![1.0, 0.0, 0.0, 1.0]),
            ("PoissonLog", vec![0.0, 2.0, 5.0, 1.0]),
            ("ExponentialLog", vec![0.4, 1.5, 3.0, 0.2]),
            ("LogNormal", vec![0.4, 1.5, 3.0, 0.2]),
            ("NegativeBinomialLog", vec![0.0, 2.0, 5.0, 1.0]),
        ];
        for (family, y) in families {
            case(&format!("ObsLogP({family})"), &|g| {
                let (a, b) = (g.add_param("a"), g.add_param("b"));
                let x = g.add_data("x", vec![0.5, -1.0, 1.5, 0.25]);
                let bx = g.scalar_mul_data(b, x);
                let eta = g.scalar_broadcast_add(a, bx);
                let obs = g.add_obs_data(y.clone());
                let aux = positive(g, "aux");
                match family {
                    "Normal" => g.obs_logp_normal(eta, aux, obs),
                    "BernoulliLogit" => g.obs_logp_bernoulli_logit(eta, obs),
                    "PoissonLog" => g.obs_logp_poisson_log(eta, obs),
                    "ExponentialLog" => g.obs_logp_exponential_log(eta, obs),
                    "LogNormal" => g.obs_logp_lognormal(eta, aux, obs),
                    _ => g.obs_logp_negative_binomial_log(eta, aux, obs),
                };
                // The families without an auxiliary node still need a
                // density in `aux` for the gradient to be defined.
                let zero = g.add_constant(0.0);
                let one = g.add_constant(1.0);
                g.normal_logp(aux, zero, one);
            });
        }

        case("VectorNormalLogP", &|g| {
            let start = g.add_vector_params("v", 3);
            g.vector_normal_logp(start, 3, 0.4, 1.7);
        });
        case("VectorHalfNormalLogP", &|g| {
            let start = g.add_vector_params_with_transform("v", 3, ParamTransform::Exp);
            g.vector_half_normal_logp(start, 3, 1.3);
        });
        case("VectorStudentTLogP", &|g| {
            let start = g.add_vector_params("v", 3);
            g.vector_student_t_logp(start, 3, 2.5, -0.3, 0.8);
        });
        case("VectorGammaLogP", &|g| {
            let start = g.add_vector_params_with_transform("v", 3, ParamTransform::Exp);
            g.vector_gamma_logp(start, 3, 1.5, 2.5);
        });
        case("VectorBetaLogP", &|g| {
            let start = g.add_vector_params_with_transform("v", 3, ParamTransform::Sigmoid);
            g.vector_beta_logp(start, 3, 2.5, 1.5);
        });
        case("VectorUniformLogP", &|g| {
            let transform = ParamTransform::BoundedSigmoid {
                lower: -2.0,
                upper: 3.0,
            };
            let start = g.add_vector_params_with_transform("v", 3, transform);
            g.vector_uniform_logp(start, 3, -2.0, 3.0);
        });

        let unary = [
            E::Neg,
            E::Exp,
            E::Log,
            E::Sigmoid,
            E::Sqrt,
            E::Tanh,
            E::Softplus,
            E::Sin,
            E::Cos,
        ];
        let binary = [E::Add, E::Sub, E::Mul, E::Div, E::Pow];
        for operator in unary.into_iter().chain(binary) {
            let is_binary = binary
                .iter()
                .any(|op| std::mem::discriminant(op) == std::mem::discriminant(&operator));
            case(&format!("Elementwise({operator:?})"), &|g| {
                // A positive left operand keeps Log, Sqrt and Pow in their
                // domain; a positive right one keeps Div away from zero.
                let a = positive(g, "a");
                let b = is_binary.then(|| positive(g, "b"));
                let value = g.elementwise(operator, a, b);
                let mu = g.add_constant(0.2);
                let sigma = g.add_constant(1.5);
                g.normal_logp(value, mu, sigma);
            });
        }

        let points = [[0.3, -0.4, 0.8], [-0.9, 0.6, 0.1], [1.1, 0.2, -0.7]];
        for (name, g) in &cases {
            for point in points {
                let params: Vec<f64> = point.iter().cycle().take(g.param_count).copied().collect();
                let result = std::panic::catch_unwind(|| full_finite_diff_check(g, &params, 1e-4));
                assert!(result.is_ok(), "{name} at {params:?}");
            }
        }
    }

    /// The `ln(y)` cache is built from the graph an evaluator was constructed
    /// for. A graph changed afterwards so that a term becomes LogNormal must
    /// still be evaluated as LogNormal, as it was before the cache existed:
    /// read against the empty cache, the likelihood silently dropped out
    /// (log density and gradient both zero here).
    #[test]
    fn lognormal_term_added_after_construction_is_evaluated() {
        let mut g = Graph::new();
        let mu = g.add_param("mu");
        let one = g.add_constant(1.0);
        let obs = g.add_obs_data(vec![2.0, 0.5, 3.0]);
        let linpred = g.broadcast_observation(mu, obs);
        let term = g.obs_logp_normal(linpred, one, obs);
        let mut evaluator = Evaluator::new(&g);
        if let Op::ObsLogP { family, .. } = &mut g.nodes[term.0].op {
            *family = ObsFamily::LogNormal;
        }
        let mut fresh = Evaluator::new(&g);
        for params in [[0.0], [0.4]] {
            evaluator.compute(&g, &params);
            fresh.compute(&g, &params);
            assert_ne!(evaluator.total_logp, 0.0);
            assert_eq!(evaluator.total_logp.to_bits(), fresh.total_logp.to_bits());
            assert_eq!(evaluator.grad, fresh.grad);
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

    /// The Normal and LogNormal reverse passes read the standardized residuals
    /// the forward pass stored instead of recomputing them. Starting from
    /// nonzero adjoints, they must add the bits the recomputing loop added,
    /// which is reproduced here. A lone NaN keeps its payload. Rust leaves
    /// open which payload the sum of two NaNs keeps, so no case adds two.
    #[test]
    fn gaussian_residual_adjoints_match_the_recomputing_loop_bit_for_bit() {
        // Moderate values, so that most scale adjoints are finite sums, then
        // values near overflow, an infinity and a NaN with a payload.
        let moderate = (
            [-40.0, -3.0, -0.0, 0.0, 1e-310, 0.25, 7.5, 12.0],
            [-2.0, 0.0, -0.0, 1e-300, 0.25, 1e5, -7.0, 3.5],
        );
        let extreme = (
            [-1e300, -3.0, -0.0, 0.0, 1e-310, 0.25, 7.5, 1e300],
            [
                -2.0,
                0.0,
                -0.0,
                1e-300,
                0.25,
                1e308,
                f64::INFINITY,
                f64::from_bits(0x7ff8_0000_0000_1234),
            ],
        );
        let sigmas = [1e-300, 1e-12, 0.5, 1.0, 3.0, 1e300];
        let scales = [1.0, -0.5, 1e-320, 1e300];
        let mut finite_scale_adjoints = 0;
        for (ys, mus) in [moderate, extreme] {
            for &sigma in &sigmas {
                for &a_s in &scales {
                    for shift in 0..mus.len() {
                        let mu = (0..ys.len())
                            .map(|i| mus[(i + shift) % mus.len()])
                            .collect::<Vec<_>>();
                        let start = (0..ys.len())
                            .map(|i| 0.1 * i as f64 - 0.3)
                            .collect::<Vec<_>>();

                        let mut expected = start.clone();
                        let mut expected_sigma = 0.75;
                        let mut dsigma = 0.0f64;
                        for (i, (&y, &m)) in ys.iter().zip(&mu).enumerate() {
                            let diff = (y - m) / sigma;
                            expected[i] += a_s * (diff / sigma);
                            dsigma += (diff * diff - 1.0) / sigma;
                        }
                        expected_sigma += a_s * dsigma;

                        let residuals = ys
                            .iter()
                            .zip(&mu)
                            .map(|(&y, &m)| (y - m) / sigma)
                            .collect::<Vec<_>>();
                        let mut actual = start.clone();
                        let mut actual_sigma = 0.75;
                        gaussian_residual_adjoints(
                            &residuals,
                            sigma,
                            a_s,
                            Some(&mut actual),
                            Some(&mut actual_sigma),
                        );
                        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(&actual), bits(&expected), "{sigma} {a_s} {shift}");
                        assert_eq!(
                            actual_sigma.to_bits(),
                            expected_sigma.to_bits(),
                            "{sigma} {a_s} {shift}"
                        );
                        finite_scale_adjoints += expected_sigma.is_finite() as usize;
                    }
                }
            }
        }
        // Of the 384 cases, so the scale's sums are checked on finite values
        // and not only on a NaN that every sum carries.
        assert_eq!(finite_scale_adjoints, 128);
    }

    /// A non-centered varying-intercept graph, `mu + e(a, b)` with
    /// `e = (op(a, b)) * w`, observed through a Normal term with an inferred
    /// scale and a LogNormal term with a constant one. `a` and `b` are a
    /// gathered vector, a scalar parameter, data, or a second gathered vector
    /// over other groups, in every pairing, so two distinct computed vectors
    /// meet in one node; `w` has
    /// zeros, so some of the elementwise node's upstream adjoints are zero and
    /// take the skip. The gathered parameters have an identity or an `Exp`
    /// transform, and `tau` has a prior of its own.
    fn indexed_graph(
        op: ElementwiseOp,
        kinds: (usize, Option<usize>),
        transform: ParamTransform,
    ) -> Graph {
        let mut g = Graph::new();
        let mu = g.add_param("mu");
        let tau = g.add_param("tau");
        let z = g.add_vector_params_with_transform("z", 3, transform.clone());
        let log_sigma = g.add_param("log_sigma");
        let second = g.add_vector_params_with_transform("v", 3, transform);
        let group = g.add_data("group", vec![0.0, 2.0, 1.0, 1.0, 0.0, 2.0]);
        let gathered = g.gather(z, 3, group);
        let other_group = g.add_data("other_group", vec![2.0, 2.0, 0.0, 1.0, 1.0, 0.0]);
        let other = g.gather(second, 3, other_group);
        let x = g.add_data("x", vec![0.0, -1.0, 2.5, 1e-300, 3.0, 0.5]);
        let w = g.add_data("w", vec![1.0, 0.0, -2.0, 1.0, 0.0, 0.5]);
        let pick = |kind: usize| [gathered, tau, x, other][kind];
        let e = g.elementwise(op, pick(kinds.0), kinds.1.map(pick));
        let weighted = g.elementwise(ElementwiseOp::Mul, e, Some(w));
        let mean = g.scalar_broadcast_add(mu, weighted);
        let sigma = g.exp(log_sigma);
        let levels = g.add_obs_data(vec![0.3, -1.2, 2.0, 0.0, 1.1, -0.4]);
        g.obs_logp_normal(mean, sigma, levels);
        let half = g.add_constant(0.5);
        let sizes = g.add_obs_data(vec![1.5, 0.2, 3.0, 1.0, 0.7, 2.2]);
        g.obs_logp_lognormal(mean, half, sizes);
        // Added last, so the reverse pass reaches it first and `tau` already
        // holds an adjoint when the elementwise node adds its own.
        let zero = g.add_constant(0.0);
        let three = g.add_constant(3.0);
        g.normal_logp(tau, zero, three);
        g
    }

    /// The loops that read operands resolved once per node, and the gathers
    /// that transform each group once, must give the per-element loops' bits:
    /// the log density, every gradient component and every node's own values,
    /// for every elementwise operator, every pairing of two gathered vectors,
    /// a scalar and data, both transforms, and parameters that send some values
    /// to zero, infinity or NaN. NaNs are compared by their bits too. Rust
    /// guarantees neither the sign nor the payload of a NaN, nor the same
    /// results from functions like `exp` and `powf` on every platform, so a
    /// match here is a fact about the build and platform that ran it.
    #[test]
    fn resolved_loops_match_the_per_element_loops_bit_for_bit() {
        use ElementwiseOp::*;
        let binary = [Add, Sub, Mul, Div, Pow];
        let unary = [Neg, Exp, Log, Sigmoid, Sqrt, Tanh, Softplus, Sin, Cos];
        let mut cases = Vec::new();
        for op in binary {
            for a in 0..4 {
                for b in 0..4 {
                    // Two scalars give a scalar node, which has no vector loop.
                    if (a, b) != (1, 1) {
                        cases.push((op, (a, Some(b))));
                    }
                }
            }
        }
        for op in unary {
            for a in [0, 2, 3] {
                cases.push((op, (a, None)));
            }
        }
        // mu, tau, z[0..3], log_sigma, v[0..3].
        let points: [[f64; 9]; 5] = [
            [0.4, 1.3, -0.2, 0.7, 1.9, -0.1, 0.6, -1.1, 0.05],
            [-1.0, 0.0, 0.0, -0.5, 2.0, 0.3, -0.0, 1.5, -2.0],
            [0.0, -2.5, 3.0, -1e-3, 0.25, 1.0, 0.0, 2.5, -0.25],
            [2.0, 1e300, -1e300, 40.0, -40.0, -0.7, 1e300, -3.0, 40.0],
            [0.1, -0.0, 800.0, -800.0, 1e-310, 0.0, -800.0, 1e-310, 0.5],
        ];
        let mut finite_gradients = 0;
        let mut compared = 0;
        for transform in [ParamTransform::Identity, ParamTransform::Exp] {
            for &(op, kinds) in &cases {
                let g = indexed_graph(op, kinds, transform.clone());
                let mut resolved = Evaluator::new(&g);
                let mut per_element = Evaluator::new(&g);
                per_element.per_element = true;
                for point in &points {
                    resolved.compute(&g, point);
                    per_element.compute(&g, point);
                    let label = format!("{op:?} {kinds:?} {transform:?} {point:?}");
                    assert_eq!(
                        resolved.total_logp.to_bits(),
                        per_element.total_logp.to_bits(),
                        "{label}"
                    );
                    for (k, (&l, &r)) in resolved.grad.iter().zip(&per_element.grad).enumerate() {
                        assert_eq!(
                            l.to_bits(),
                            r.to_bits(),
                            "{label}, gradient {k}: {l:e} {r:e}"
                        );
                        finite_gradients += l.is_finite() as usize;
                        compared += 1;
                    }
                    // Every node's own values too, which a later node can hide:
                    // a sign of zero erased by an addition, say.
                    for node in &g.nodes {
                        let length = per_element.node_len(node.id);
                        let values = |e: &Evaluator| -> Vec<u64> {
                            if length == 0 {
                                vec![e.scalar_at(node.id).to_bits()]
                            } else {
                                (0..length)
                                    .map(|i| e.vec_elem(node.id, i).to_bits())
                                    .collect()
                            }
                        };
                        assert_eq!(
                            values(&resolved),
                            values(&per_element),
                            "{label}, node {:?}",
                            node.id
                        );
                    }
                }
            }
        }
        // Most components are ordinary finite numbers, so the comparison is
        // not carried by NaNs that every path would share.
        assert!(
            finite_gradients * 2 > compared,
            "{finite_gradients} of {compared}"
        );
    }

    /// The resolved loops against the reference evaluator and finite
    /// differences, on varying-intercept graphs with each transform.
    #[test]
    fn indexed_graphs_match_the_reference_gradients() {
        for transform in [ParamTransform::Identity, ParamTransform::Exp] {
            for (op, kinds) in [
                (ElementwiseOp::Mul, (1, Some(0))),
                (ElementwiseOp::Mul, (0, Some(1))),
                (ElementwiseOp::Add, (0, Some(2))),
                (ElementwiseOp::Div, (0, Some(1))),
                (ElementwiseOp::Tanh, (0, None)),
                (ElementwiseOp::Mul, (0, Some(3))),
                (ElementwiseOp::Sub, (3, Some(0))),
                (ElementwiseOp::Div, (0, Some(3))),
            ] {
                let g = indexed_graph(op, kinds, transform.clone());
                full_finite_diff_check(&g, &[0.4, 1.3, -0.2, 0.7, 0.3, -0.1, 0.6, -1.1, 0.8], 2e-5);
            }
        }
    }

    /// A gather whose span a graph change moved past the parameters, as no
    /// fresh graph can have, still runs the per-row loop, which reads only
    /// the parameters its rows select, and gives what it gave before.
    #[test]
    fn a_gather_moved_past_the_parameters_still_reads_only_its_rows() {
        let mut g = Graph::new();
        let z = g.add_vector_params("z", 2);
        let rows = g.add_data("group", vec![0.0, 0.0]);
        let gathered = g.gather(z, 2, rows);
        let one = g.add_constant(1.0);
        let obs = g.add_obs_data(vec![0.5, -0.5]);
        g.obs_logp_normal(gathered, one, obs);
        let mut resolved = Evaluator::new(&g);
        let mut per_element = Evaluator::new(&g);
        per_element.per_element = true;
        let Op::Gather { param_start, .. } = &mut g.nodes[gathered.0].op else {
            unreachable!()
        };
        // Rows select group 0, now parameter 1; the span runs to parameter 2.
        *param_start = 1;
        resolved.compute(&g, &[0.3, -0.7]);
        per_element.compute(&g, &[0.3, -0.7]);
        assert_eq!(
            resolved.total_logp.to_bits(),
            per_element.total_logp.to_bits()
        );
        assert_eq!(resolved.vec_elem(gathered, 0), -0.7);
        let bits = |e: &Evaluator| e.grad.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&resolved), bits(&per_element));
        assert_eq!(resolved.grad[0], 0.0);
    }

    /// Normal and LogNormal terms on a design matrix, with an inferred and a
    /// constant scale, against the reference evaluator and finite
    /// differences, which recompute every residual.
    #[test]
    fn gaussian_terms_match_the_reference_gradients() {
        for lognormal in [false, true] {
            for inferred_scale in [false, true] {
                let mut g = Graph::new();
                let start = g.add_vector_params("beta", 3);
                let design = (0..15).map(|i| ((i * 7) % 11) as f64 / 5.0 - 1.0).collect();
                let matrix = g.store_matrix(design, 5, 3);
                let eta = g.mat_vec_mul(matrix, start, 3, None);
                let sigma = if inferred_scale {
                    let raw = g.add_param("log_sigma");
                    g.exp(raw)
                } else {
                    g.add_constant(0.7)
                };
                let obs = g.add_obs_data(vec![0.3, 1.9, 0.8, 2.6, 1.1]);
                if lognormal {
                    g.obs_logp_lognormal(eta, sigma, obs);
                } else {
                    g.obs_logp_normal(eta, sigma, obs);
                }
                let mut params = vec![0.4, -0.3, 0.9];
                if inferred_scale {
                    params.push(-0.2);
                }
                full_finite_diff_check(&g, &params, 1e-5);
            }
        }
    }

    /// The gradient map is built with the evaluator, so a parameter scale that
    /// the graph later turns into a constant still reads its adjoint. Its
    /// value is whatever the recomputing loop gave before the reverse pass
    /// learned to skip constant scales: here `(2^2 - 1) / 1` for Normal and
    /// `(0^2 - 1) / 1` for LogNormal.
    #[test]
    fn a_scale_turned_constant_after_construction_keeps_its_old_gradient() {
        for (lognormal, observation, expected) in [(false, 2.0, 3.0), (true, 1.0, -1.0)] {
            let mut g = Graph::new();
            let sigma = g.add_param("sigma");
            let mu = g.add_data("mu", vec![0.0]);
            let obs = g.add_obs_data(vec![observation]);
            if lognormal {
                g.obs_logp_lognormal(mu, sigma, obs);
            } else {
                g.obs_logp_normal(mu, sigma, obs);
            }
            let mut evaluator = Evaluator::new(&g);
            g.nodes[sigma.0].op = Op::Constant(1.0);
            evaluator.compute(&g, &[1.0]);
            assert_eq!(evaluator.grad[0].to_bits(), f64::to_bits(expected));
        }
    }

    /// `student_t_derivatives` shares its logarithms and softplus with the
    /// location score instead of letting `student_t_dx` take them again. It
    /// must give the bits of the single expression it had before the vector
    /// priors split that score out, which is reproduced here.
    #[test]
    fn student_t_derivatives_match_their_unsplit_form_bit_for_bit() {
        fn unsplit(x: f64, nu: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
            let diff = x - mu;
            let log_abs_diff = diff.abs().ln();
            let log_ratio = 2.0 * (log_abs_diff - sigma.ln()) - nu.ln();
            let log_tail = softplus(log_ratio);
            let weight = stable_sigmoid(log_ratio);
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
        let same = |l: f64, r: f64| l.to_bits() == r.to_bits() || (l.is_nan() && r.is_nan());
        let xs = [-1e300, -40.0, -2.5, -1e-12, 0.0, 1e-300, 0.7, 3.0, 1e200];
        let nus = [1e-3, 0.5, 1.0, 4.0, 30.0, 1e8];
        let sigmas = [1e-200, 0.01, 1.0, 2.5, 1e150];
        for &x in &xs {
            for &nu in &nus {
                for &sigma in &sigmas {
                    for mu in [0.0, 0.7, -3.25] {
                        let (a, b, c) = student_t_derivatives(x, nu, mu, sigma);
                        let (p, q, r) = unsplit(x, nu, mu, sigma);
                        assert!(
                            same(a, p) && same(b, q) && same(c, r),
                            "x={x} nu={nu} mu={mu} sigma={sigma}"
                        );
                        let ln_nu1 = (nu + 1.0).ln();
                        let dx = student_t_dx(x - mu, sigma.ln(), nu.ln(), ln_nu1);
                        assert!(same(dx, p), "dx: x={x} nu={nu} mu={mu} sigma={sigma}");
                    }
                }
            }
        }
    }

    /// The vector priors take the logarithms of their constant arguments once
    /// per op rather than once per element. That is only a hoist if every
    /// element still gets exactly the bits the per-element density and score
    /// give it, including where an argument is outside its support.
    #[test]
    fn vector_priors_match_their_scalar_densities_bit_for_bit() {
        let raws = [-3.5, -0.4, 0.0, 0.7, 2.25, 9.0];
        let n = raws.len();
        let bits = |x: f64| (0.0 + x).to_bits();
        let check =
            |g: &Graph, node: NodeId, logp: &dyn Fn(f64) -> f64, dx: &dyn Fn(f64) -> f64| {
                let mut evaluator = Evaluator::new(g);
                evaluator.compute(g, &raws);
                let mut expected = 0.0f64;
                for raw in raws {
                    expected += logp(raw);
                }
                assert_eq!(bits(evaluator.scalar_at(node)), bits(expected));
                if expected.is_finite() {
                    for (k, raw) in raws.iter().enumerate() {
                        assert_eq!(bits(evaluator.grad[k]), bits(dx(*raw)), "element {k}");
                    }
                }
            };
        for sigma in [0.3, 1.0, 17.5, 0.0, -1.0, f64::NAN] {
            let mut g = Graph::new();
            let start = g.add_vector_params("x", n);
            let node = g.vector_half_normal_logp(start, n, sigma);
            check(&g, node, &|raw| log_half_normal_logp(raw, sigma), &|raw| {
                1.0 - (2.0 * (raw - sigma.ln())).exp()
            });
        }
        for (alpha, beta) in [
            (0.5, 2.0),
            (3.0, 0.25),
            (40.0, 7.0),
            (0.0, 1.0),
            (2.0, -1.0),
        ] {
            let mut g = Graph::new();
            let start = g.add_vector_params("x", n);
            let node = g.vector_gamma_logp(start, n, alpha, beta);
            check(&g, node, &|raw| log_gamma_logp(raw, alpha, beta), &|raw| {
                alpha - (raw + beta.ln()).exp()
            });
        }
        for (nu, mu, sigma) in [
            (1.0, 0.0, 1.0),
            (3.0, 0.7, 0.2),
            (30.0, -2.0, 5.0),
            (0.5, 2.25, 1e-3),
            (0.0, 0.0, 1.0),
            (3.0, 0.0, -2.0),
        ] {
            let mut g = Graph::new();
            let start = g.add_vector_params("x", n);
            let node = g.vector_student_t_logp(start, n, nu, mu, sigma);
            check(
                &g,
                node,
                &|v| student_t_logp_scalar(v, nu, mu, sigma),
                &|v| student_t_derivatives(v, nu, mu, sigma).0,
            );
        }
        // An empty span sums to zero whatever its arguments.
        let mut g = Graph::new();
        let start = g.add_vector_params("x", 0);
        let node = g.vector_gamma_logp(start, 0, -1.0, 1.0);
        let mut evaluator = Evaluator::new(&g);
        evaluator.compute(&g, &[]);
        assert_eq!(evaluator.scalar_at(node).to_bits(), 0.0f64.to_bits());
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
        g.obs_logp_normal(mu, sigma, y);
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
            graph.add_logp_term(term);

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
                    graph.obs_logp_normal(predictor, scale, obs);
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
