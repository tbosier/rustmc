use crate::data::DataBinding;
use crate::graph::{Graph, GraphShapeError, NodeId, Op, ParamTransform};

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
    let mut required_vectors = 0usize;
    let mut required_observations = 0usize;
    let mut required_matrices = 0usize;
    for node in &graph.nodes {
        match &node.op {
            Op::Data(index) => required_vectors = required_vectors.max(*index + 1),
            Op::ObsLogP { obs_data_idx, .. } => {
                required_observations = required_observations.max(*obs_data_idx + 1)
            }
            Op::MatVecMul { matrix_idx, .. } => {
                required_matrices = required_matrices.max(*matrix_idx + 1)
            }
            _ => {}
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

/// Derive every vector length from its inputs; scalars have length zero.
pub(crate) fn validate_node_lengths(
    graph: &Graph,
    binding: &DataBinding,
) -> Result<Vec<usize>, GraphShapeError> {
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
            _ => 0,
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
            _ => None,
        };
        dimensions.push(dimension);
        lengths.push(len);
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

    pub fn new(graph: &Graph) -> Self {
        Self::try_new(graph).expect("graph shape validation failed")
    }

    pub fn with_binding(graph: &Graph, binding: DataBinding) -> Self {
        Self::try_with_binding(graph, binding).expect("validated binding does not match structure")
    }

    /// Reuse allocations while changing only the dataset payload and row count.
    pub fn rebind(&mut self, graph: &Graph, binding: DataBinding) -> Result<(), GraphShapeError> {
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
    fn read_vec(&self, node_id: usize, i: usize, _graph: &Graph) -> f64 {
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
    pub fn vec_elem(&self, node: NodeId, i: usize, graph: &Graph) -> f64 {
        self.read_vec(node.0, i, graph)
    }

    /// Copy a full vector node into a Vec after `compute()`.
    #[deprecated(note = "prefer node_len and vec_elem to avoid allocation")]
    pub fn vec_to_owned(&self, node: NodeId, graph: &Graph) -> Vec<f64> {
        (0..self.node_lengths[node.0])
            .map(|i| self.read_vec(node.0, i, graph))
            .collect()
    }

    /// Compute log-probability and its gradient. Results are stored in
    /// `self.total_logp` and `self.grad`. No heap allocations occur.
    pub fn compute(&mut self, graph: &Graph, params: &[f64]) {
        // === Forward pass ===
        for node in &graph.nodes {
            let idx = node.id.0;
            let vl = match &node.op {
                Op::ObsLogP { obs_data_idx, .. } => self.binding.observations[*obs_data_idx].len(),
                _ => self.node_lengths[idx],
            };
            match &node.op {
                Op::Elementwise { operator, a, b } => {
                    for i in 0..vl.max(1) {
                        let av = self.read_vec(a.0, i, graph);
                        let bv = b.map_or(0.0, |b| self.read_vec(b.0, i, graph));
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
                        let k = *param_start + self.read_vec(indices.0, i, graph) as usize;
                        self.vec_buf[off + i] = graph.param_transforms[k].apply(params[k]);
                    }
                }
                Op::Sum(a) => {
                    self.scalars[idx] = (0..self.node_lengths[a.0].max(1))
                        .map(|i| self.read_vec(a.0, i, graph))
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
                Op::Sub(a, b) => self.scalars[idx] = self.scalars[a.0] - self.scalars[b.0],
                Op::Mul(a, b) => self.scalars[idx] = self.scalars[a.0] * self.scalars[b.0],
                Op::Div(a, b) => self.scalars[idx] = self.scalars[a.0] / self.scalars[b.0],
                Op::Neg(a) => self.scalars[idx] = -self.scalars[a.0],
                Op::Exp(a) => self.scalars[idx] = self.scalars[a.0].exp(),
                Op::Log(a) => self.scalars[idx] = self.scalars[a.0].ln(),
                Op::Sigmoid(a) => {
                    let v = self.scalars[a.0];
                    self.scalars[idx] = 1.0 / (1.0 + (-v).exp());
                }
                Op::Square(a) => {
                    let v = self.scalars[a.0];
                    self.scalars[idx] = v * v;
                }
                Op::ScalarMulData(scalar, data) => {
                    let s = self.scalars[scalar.0];
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        let d = self.read_vec(data.0, i, graph);
                        self.vec_buf[out_off + i] = s * d;
                    }
                }
                Op::VectorAdd(a, b) => {
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    for i in 0..vl {
                        let va = self.read_vec(a.0, i, graph);
                        let vb = self.read_vec(b.0, i, graph);
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
                        let v = self.read_vec(vec.0, i, graph);
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
                Op::HalfNormalLogP { x, sigma } => {
                    self.scalars[idx] =
                        half_normal_logp_scalar(self.scalars[x.0], self.scalars[sigma.0]);
                }
                Op::StudentTLogP { x, nu, mu, sigma } => {
                    self.scalars[idx] = student_t_logp_scalar(
                        self.scalars[x.0],
                        self.scalars[nu.0],
                        self.scalars[mu.0],
                        self.scalars[sigma.0],
                    );
                }
                Op::UniformLogP { x, lower, upper } => {
                    self.scalars[idx] = uniform_logp_scalar(
                        self.scalars[x.0],
                        self.scalars[lower.0],
                        self.scalars[upper.0],
                    );
                }
                Op::BernoulliLogP { x, p } => {
                    self.scalars[idx] = bernoulli_logp_scalar(self.scalars[x.0], self.scalars[p.0]);
                }
                Op::PoissonLogP { x, lam } => {
                    self.scalars[idx] = poisson_logp_scalar(self.scalars[x.0], self.scalars[lam.0]);
                }
                Op::GammaLogP { x, alpha, beta } => {
                    self.scalars[idx] = gamma_logp_scalar(
                        self.scalars[x.0],
                        self.scalars[alpha.0],
                        self.scalars[beta.0],
                    );
                }
                Op::BetaLogP { x, alpha, beta } => {
                    self.scalars[idx] = beta_logp_scalar(
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
                            let s2 = sv * sv;
                            let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
                            let n = obs.len() as f64;
                            let mut sum_sq = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let m = self.read_vec(linpred_vec.0, i, graph);
                                let d = y - m;
                                sum_sq += d * d;
                            }
                            self.scalars[idx] = n * log_norm - 0.5 * sum_sq / s2;
                        }
                        crate::graph::ObsFamily::BernoulliLogit => {
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                sum += y * eta - softplus(eta);
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::PoissonLog => {
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                sum += y * eta - eta.exp() - ln_gamma(y + 1.0);
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::ExponentialLog => {
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                sum += eta - y * eta.exp();
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::LogNormal => {
                            let sigma_node = aux.expect("LogNormal obs logp requires sigma");
                            let sv = self.scalars[sigma_node.0];
                            let s2 = sv * sv;
                            let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
                            let mut sum = 0.0f64;
                            for (i, &observation) in obs.iter().take(vl).enumerate() {
                                let y = observation.max(1e-300);
                                let m = self.read_vec(linpred_vec.0, i, graph);
                                let ly = y.ln();
                                let d = ly - m;
                                sum += log_norm - ly - 0.5 * d * d / s2;
                            }
                            self.scalars[idx] = sum;
                        }
                        crate::graph::ObsFamily::NegativeBinomialLog => {
                            let alpha_node = aux.expect("NegativeBinomial obs logp requires alpha");
                            let av = self.scalars[alpha_node.0];
                            let mut sum = 0.0f64;
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                let mu = eta.exp();
                                sum += ln_gamma(y + av) - ln_gamma(av) - ln_gamma(y + 1.0)
                                    + av * (av.ln() - (av + mu).ln())
                                    + y * (eta - (av + mu).ln());
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
                    let s2 = sigma * sigma;
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        let d = v - mu;
                        sum += log_norm - 0.5 * d * d / s2;
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorHalfNormalLogP {
                    param_start,
                    n_params,
                    sigma,
                } => {
                    // Combined logp(exp(raw), sigma) + raw (Jacobian)
                    let log_norm = (2.0 / (sigma * std::f64::consts::TAU.sqrt())).ln();
                    let s2 = sigma * sigma;
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        // log(sqrt(2/π)/σ) - exp(2·raw)/(2σ²) + raw
                        sum += log_norm - (2.0 * raw).exp() / (2.0 * s2) + raw;
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
                    let log_norm = ln_gamma(0.5 * (nu + 1.0))
                        - ln_gamma(0.5 * nu)
                        - 0.5 * (nu * std::f64::consts::PI * sigma * sigma).ln();
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        let z = (v - mu) / sigma;
                        sum += log_norm - 0.5 * (nu + 1.0) * (1.0 + z * z / nu).ln();
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorGammaLogP {
                    param_start,
                    n_params,
                    alpha,
                    beta,
                } => {
                    // Combined logp(exp(raw), alpha, beta) + raw (Jacobian = exp(raw), log = raw)
                    // = α·log(β) - lnΓ(α) + α·raw - β·exp(raw)
                    let log_norm = alpha * beta.ln() - ln_gamma(*alpha);
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        sum += log_norm + alpha * raw - beta * raw.exp();
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
                        let s = 1.0 / (1.0 + (-raw).exp());
                        sum += log_norm + alpha * s.ln() + beta * (1.0 - s).ln();
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorUniformLogP {
                    param_start,
                    n_params,
                    ..
                } => {
                    // s = sigmoid(raw), logp_uniform = -log(hi-lo) (const), Jacobian = s·(1-s)·(hi-lo)
                    // Combined: -log(hi-lo) + log(s·(1-s)·(hi-lo)) = log(s·(1-s)) = log(s) + log(1-s)
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        let s = 1.0 / (1.0 + (-raw).exp());
                        sum += s.ln() + (1.0 - s).ln();
                    }
                    self.scalars[idx] = sum;
                }
            }
        }

        // Total log-probability
        self.total_logp = graph.logp_terms.iter().map(|id| self.scalars[id.0]).sum();

        // === Backward pass ===
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
            let vl = match &node.op {
                Op::ObsLogP { obs_data_idx, .. } => self.binding.observations[*obs_data_idx].len(),
                _ => self.node_lengths[idx],
            };
            let a_s = self.adj_scalars[idx];

            match &node.op {
                Op::Elementwise { operator, a, b } => {
                    for i in 0..vl.max(1) {
                        let av = self.read_vec(a.0, i, graph);
                        let bv = b.map_or(0.0, |b| self.read_vec(b.0, i, graph));
                        let (da, db) = operator.derivatives(av, bv);
                        let upstream = match self.node_kind[idx] {
                            NodeKind::ComputedVec(off) => self.adj_vec_buf[off + i],
                            _ => a_s,
                        };
                        self.accumulate(*a, i, upstream * da);
                        if let Some(b) = b {
                            self.accumulate(*b, i, upstream * db);
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
                        let k = *param_start + self.read_vec(indices.0, i, graph) as usize;
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
                Op::Sub(a, b) => {
                    self.adj_scalars[a.0] += a_s;
                    self.adj_scalars[b.0] -= a_s;
                }
                Op::Mul(a, b) => {
                    let va = self.scalars[a.0];
                    let vb = self.scalars[b.0];
                    self.adj_scalars[a.0] += a_s * vb;
                    self.adj_scalars[b.0] += a_s * va;
                }
                Op::Div(a, b) => {
                    let va = self.scalars[a.0];
                    let vb = self.scalars[b.0];
                    self.adj_scalars[a.0] += a_s / vb;
                    self.adj_scalars[b.0] -= a_s * va / (vb * vb);
                }
                Op::Neg(a) => self.adj_scalars[a.0] -= a_s,
                Op::Exp(a) => {
                    let va = self.scalars[a.0].exp();
                    self.adj_scalars[a.0] += a_s * va;
                }
                Op::Log(a) => self.adj_scalars[a.0] += a_s / self.scalars[a.0],
                Op::Sigmoid(a) => {
                    let s = self.scalars[idx];
                    self.adj_scalars[a.0] += a_s * s * (1.0 - s);
                }
                Op::Square(a) => self.adj_scalars[a.0] += a_s * 2.0 * self.scalars[a.0],

                Op::ScalarMulData(scalar, data) => {
                    let s = self.scalars[scalar.0];
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let mut ds = 0.0f64;
                    for i in 0..vl {
                        let upstream = self.adj_vec_buf[out_off + i];
                        let d_val = self.read_vec(data.0, i, graph);
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
                    let diff = xv - mv;
                    let s2 = sv * sv;
                    self.adj_scalars[x.0] += a_s * (-diff / s2);
                    self.adj_scalars[mu.0] += a_s * (diff / s2);
                    self.adj_scalars[sigma.0] += a_s * (diff * diff / (s2 * sv) - 1.0 / sv);
                }
                Op::HalfNormalLogP { x, sigma } => {
                    let xv = self.scalars[x.0];
                    let sv = self.scalars[sigma.0];
                    if xv >= 0.0 {
                        self.adj_scalars[x.0] += a_s * (-xv / (sv * sv));
                        self.adj_scalars[sigma.0] += a_s * (xv * xv / (sv * sv * sv) - 1.0 / sv);
                    }
                }
                Op::StudentTLogP { x, nu, mu, sigma } => {
                    let xv = self.scalars[x.0];
                    let nv = self.scalars[nu.0];
                    let mv = self.scalars[mu.0];
                    let sv = self.scalars[sigma.0];
                    let z = (xv - mv) / sv;
                    let z2 = z * z;
                    let denom = 1.0 + z2 / nv;
                    // d/dx
                    self.adj_scalars[x.0] += a_s * (-(nv + 1.0) * z / (sv * nv * denom));
                    // d/dmu
                    self.adj_scalars[mu.0] += a_s * ((nv + 1.0) * z / (sv * nv * denom));
                    // d/dsigma
                    self.adj_scalars[sigma.0] +=
                        a_s * ((nv + 1.0) * z2 / (sv * nv * denom) - 1.0 / sv);
                    // d/dnu
                    self.adj_scalars[nu.0] += a_s
                        * (0.5 * digamma(0.5 * (nv + 1.0))
                            - 0.5 * digamma(0.5 * nv)
                            - 0.5 / nv
                            - 0.5 * (1.0 + z2 / nv).ln()
                            + 0.5 * (nv + 1.0) * z2 / (nv * nv * denom));
                }
                Op::UniformLogP { x: _, lower, upper } => {
                    let lv = self.scalars[lower.0];
                    let uv = self.scalars[upper.0];
                    let range = uv - lv;
                    if range > 0.0 {
                        self.adj_scalars[lower.0] += a_s / range;
                        self.adj_scalars[upper.0] -= a_s / range;
                    }
                }
                Op::BernoulliLogP { x, p } => {
                    let xv = self.scalars[x.0];
                    let pv = self.scalars[p.0].clamp(1e-12, 1.0 - 1e-12);
                    self.adj_scalars[p.0] += a_s * (xv / pv - (1.0 - xv) / (1.0 - pv));
                }
                Op::PoissonLogP { x, lam } => {
                    let xv = self.scalars[x.0];
                    let lv = self.scalars[lam.0];
                    self.adj_scalars[lam.0] += a_s * (xv / lv - 1.0);
                }
                Op::GammaLogP { x, alpha, beta } => {
                    let xv = self.scalars[x.0];
                    let av = self.scalars[alpha.0];
                    let bv = self.scalars[beta.0];
                    if xv > 0.0 {
                        self.adj_scalars[x.0] += a_s * ((av - 1.0) / xv - bv);
                        self.adj_scalars[alpha.0] += a_s * (bv.ln() - digamma(av) + xv.ln());
                        self.adj_scalars[beta.0] += a_s * (av / bv - xv);
                    }
                }
                Op::BetaLogP { x, alpha, beta } => {
                    let xv = self.scalars[x.0];
                    let av = self.scalars[alpha.0];
                    let bv = self.scalars[beta.0];
                    if xv > 0.0 && xv < 1.0 {
                        self.adj_scalars[x.0] += a_s * ((av - 1.0) / xv - (bv - 1.0) / (1.0 - xv));
                        self.adj_scalars[alpha.0] +=
                            a_s * (digamma(av + bv) - digamma(av) + xv.ln());
                        self.adj_scalars[beta.0] +=
                            a_s * (digamma(av + bv) - digamma(bv) + (1.0 - xv).ln());
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
                            let s2 = sv * sv;
                            let mut dsigma = 0.0f64;

                            let mu_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };

                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let m = self.read_vec(linpred_vec.0, i, graph);
                                let diff = y - m;
                                if let Some(off) = mu_off {
                                    self.adj_vec_buf[off + i] += a_s * diff / s2;
                                }
                                dsigma += diff * diff / (s2 * sv) - 1.0 / sv;
                            }
                            self.adj_scalars[sigma_node.0] += a_s * dsigma;
                        }
                        crate::graph::ObsFamily::BernoulliLogit => {
                            let eta_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            for (i, &y) in obs.iter().take(vl).enumerate() {
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                let grad = y - sigmoid_stable(eta);
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
                                let eta = self.read_vec(linpred_vec.0, i, graph);
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
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                let grad = 1.0 - y * eta.exp();
                                if let Some(off) = eta_off {
                                    self.adj_vec_buf[off + i] += a_s * grad;
                                }
                            }
                        }
                        crate::graph::ObsFamily::LogNormal => {
                            let sigma_node = aux.expect("LogNormal obs logp requires sigma");
                            let sv = self.scalars[sigma_node.0];
                            let s2 = sv * sv;
                            let mu_off = match self.node_kind[linpred_vec.0] {
                                NodeKind::ComputedVec(o) => Some(o),
                                _ => None,
                            };
                            let mut dsigma = 0.0f64;
                            for (i, &observation) in obs.iter().take(vl).enumerate() {
                                let y = observation.max(1e-300);
                                let ly = y.ln();
                                let m = self.read_vec(linpred_vec.0, i, graph);
                                let d = ly - m;
                                if let Some(off) = mu_off {
                                    self.adj_vec_buf[off + i] += a_s * d / s2;
                                }
                                dsigma += d * d / (s2 * sv) - 1.0 / sv;
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
                                let eta = self.read_vec(linpred_vec.0, i, graph);
                                let mu = eta.exp();
                                let denom = av + mu;
                                let deta = av * (y - mu) / denom;
                                if let Some(off) = eta_off {
                                    self.adj_vec_buf[off + i] += a_s * deta;
                                }
                                dalpha += digamma(y + av) - digamma(av) + av.ln() + 1.0
                                    - denom.ln()
                                    - (y + av) / denom;
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
                    let s2 = sigma * sigma;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        self.grad[param_start + k] += a_s * (-(v - mu) / s2);
                    }
                }
                Op::VectorHalfNormalLogP {
                    param_start,
                    n_params,
                    sigma,
                } => {
                    let s2 = sigma * sigma;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        // d/draw = -exp(2·raw)/σ² + 1
                        self.grad[param_start + k] += a_s * (-(2.0 * raw).exp() / s2 + 1.0);
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
                        let z = (v - mu) / sigma;
                        // d/dv = -(ν+1)·z / (σ·ν·(1 + z²/ν))
                        self.grad[param_start + k] +=
                            a_s * (-(nu + 1.0) * z / (sigma * nu * (1.0 + z * z / nu)));
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
                        // d/draw = α - β·exp(raw)
                        self.grad[param_start + k] += a_s * (alpha - beta * raw.exp());
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
                        let s = 1.0 / (1.0 + (-raw).exp());
                        // d/draw = α·(1-s) - β·s
                        self.grad[param_start + k] += a_s * (alpha * (1.0 - s) - beta * s);
                    }
                }
                Op::VectorUniformLogP {
                    param_start,
                    n_params,
                    ..
                } => {
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        let s = 1.0 / (1.0 + (-raw).exp());
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
// Original free functions (kept for tests and simple use)
// ---------------------------------------------------------------------------

pub use reference::{eval_logp, forward, grad_logp, Value};
#[path = "autodiff_reference.rs"]
pub mod reference;

fn normal_logp_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    let diff = x - mu;
    -0.5 * (diff * diff) / (sigma * sigma) - sigma.ln() - 0.5 * std::f64::consts::TAU.ln()
}

fn normal_obs_logp_sum(mu: &[f64], sigma: f64, obs: &[f64]) -> f64 {
    let s2 = sigma * sigma;
    let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
    let n = obs.len() as f64;
    let sum_sq: f64 = mu
        .iter()
        .zip(obs.iter())
        .map(|(m, o)| {
            let d = o - m;
            d * d
        })
        .sum();
    n * log_norm - 0.5 * sum_sq / s2
}

fn bernoulli_logit_obs_logp_sum(eta: &[f64], obs: &[f64]) -> f64 {
    eta.iter()
        .zip(obs.iter())
        .map(|(e, y)| y * e - softplus(*e))
        .sum()
}

fn poisson_log_obs_logp_sum(eta: &[f64], obs: &[f64]) -> f64 {
    eta.iter()
        .zip(obs.iter())
        .map(|(e, y)| y * e - e.exp() - ln_gamma(y + 1.0))
        .sum()
}

fn exponential_log_obs_logp_sum(eta: &[f64], obs: &[f64]) -> f64 {
    eta.iter()
        .zip(obs.iter())
        .map(|(e, y)| e - y * e.exp())
        .sum()
}

fn log_normal_obs_logp_sum(mu: &[f64], sigma: f64, obs: &[f64]) -> f64 {
    let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
    let s2 = sigma * sigma;
    mu.iter()
        .zip(obs.iter())
        .map(|(m, y)| {
            let ly = y.max(1e-300).ln();
            let d = ly - m;
            log_norm - ly - 0.5 * d * d / s2
        })
        .sum()
}

fn negative_binomial_log_obs_logp_sum(eta: &[f64], alpha: f64, obs: &[f64]) -> f64 {
    eta.iter()
        .zip(obs.iter())
        .map(|(e, y)| {
            let mu = e.exp();
            ln_gamma(y + alpha) - ln_gamma(alpha) - ln_gamma(y + 1.0)
                + alpha * (alpha.ln() - (alpha + mu).ln())
                + y * (e - (alpha + mu).ln())
        })
        .sum()
}

fn half_normal_logp_scalar(x: f64, sigma: f64) -> f64 {
    if x < 0.0 {
        return f64::NEG_INFINITY;
    }
    (2.0 / (sigma * std::f64::consts::TAU.sqrt())).ln() - x * x / (2.0 * sigma * sigma)
}

fn student_t_logp_scalar(x: f64, nu: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    ln_gamma(0.5 * (nu + 1.0))
        - ln_gamma(0.5 * nu)
        - 0.5 * (nu * std::f64::consts::PI * sigma * sigma).ln()
        - 0.5 * (nu + 1.0) * (1.0 + z * z / nu).ln()
}

fn uniform_logp_scalar(x: f64, lower: f64, upper: f64) -> f64 {
    if x < lower || x > upper {
        f64::NEG_INFINITY
    } else {
        -(upper - lower).ln()
    }
}

fn bernoulli_logp_scalar(x: f64, p: f64) -> f64 {
    let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
    x * p_clamped.ln() + (1.0 - x) * (1.0 - p_clamped).ln()
}

fn poisson_logp_scalar(x: f64, lam: f64) -> f64 {
    x * lam.ln() - lam - ln_gamma(x + 1.0)
}

fn gamma_logp_scalar(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    alpha * beta.ln() - ln_gamma(alpha) + (alpha - 1.0) * x.ln() - beta * x
}

fn beta_logp_scalar(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return f64::NEG_INFINITY;
    }
    ln_gamma(alpha + beta) - ln_gamma(alpha) - ln_gamma(beta)
        + (alpha - 1.0) * x.ln()
        + (beta - 1.0) * (1.0 - x).ln()
}

fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
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
    fn test_half_normal_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let sigma = g.add_constant(2.0);
        g.half_normal_logp(x, sigma);
        finite_diff_check(&g, &[1.5], 1e-4);
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
    fn test_gamma_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let alpha = g.add_constant(2.0);
        let beta = g.add_constant(1.5);
        g.gamma_logp(x, alpha, beta);
        finite_diff_check(&g, &[1.2], 1e-4);
    }

    #[test]
    fn test_beta_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let alpha = g.add_constant(2.0);
        let beta = g.add_constant(5.0);
        g.beta_logp(x, alpha, beta);
        finite_diff_check(&g, &[0.3], 1e-4);
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
