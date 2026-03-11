use crate::graph::{Graph, NodeId, Op};

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
    vec_len: usize,
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
}

impl Evaluator {
    pub fn new(graph: &Graph) -> Self {
        let n = graph.nodes.len();
        let vec_len = graph.data_vectors.first().map(|v| v.len())
            .or_else(|| graph.obs_vectors.first().map(|v| v.len()))
            .or_else(|| graph.data_matrices.first().map(|m| m.n_rows))
            .unwrap_or(0);

        let mut node_kind = Vec::with_capacity(n);
        let mut vec_slot_count = 0usize;

        for node in &graph.nodes {
            let kind = match &node.op {
                Op::Data(idx) => NodeKind::DataRef(*idx),
                Op::ScalarMulData(_, _)
                | Op::VectorAdd(_, _)
                | Op::ScalarBroadcastAdd(_, _)
                | Op::ScalarBroadcast(_)
                | Op::FusedLinearMu { .. }
                | Op::MatVecMul { .. } => {
                    let offset = vec_slot_count * vec_len;
                    vec_slot_count += 1;
                    NodeKind::ComputedVec(offset)
                }
                _ => NodeKind::Scalar,
            };
            node_kind.push(kind);
        }

        let mut param_node_ids: Vec<Option<usize>> = vec![None; graph.param_count];
        for node in &graph.nodes {
            if let Op::Param(pidx) = node.op {
                param_node_ids[pidx] = Some(node.id.0);
            }
        }

        Self {
            vec_len,
            node_kind,
            scalars: vec![0.0; n],
            vec_buf: vec![0.0; vec_slot_count * vec_len],
            adj_scalars: vec![0.0; n],
            adj_vec_buf: vec![0.0; vec_slot_count * vec_len],
            grad: vec![0.0; graph.param_count],
            total_logp: 0.0,
            param_node_ids,
        }
    }

    /// Read a vector element from either a Data node (graph reference) or
    /// a computed-vector node (vec_buf).
    #[inline(always)]
    fn read_vec(&self, node_id: usize, i: usize, graph: &Graph) -> f64 {
        match self.node_kind[node_id] {
            NodeKind::DataRef(di) => unsafe { *graph.data_vectors[di].get_unchecked(i) },
            NodeKind::ComputedVec(off) => unsafe { *self.vec_buf.get_unchecked(off + i) },
            NodeKind::Scalar => unreachable!(),
        }
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
    pub fn vec_to_owned(&self, node: NodeId, graph: &Graph) -> Vec<f64> {
        (0..self.vec_len).map(|i| self.read_vec(node.0, i, graph)).collect()
    }

    /// Compute log-probability and its gradient. Results are stored in
    /// `self.total_logp` and `self.grad`. No heap allocations occur.
    pub fn compute(&mut self, graph: &Graph, params: &[f64]) {
        let vl = self.vec_len;

        // === Forward pass ===
        for node in &graph.nodes {
            let idx = node.id.0;
            match &node.op {
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
                    self.scalars[idx] = half_normal_logp_scalar(self.scalars[x.0], self.scalars[sigma.0]);
                }
                Op::StudentTLogP { x, nu, mu, sigma } => {
                    self.scalars[idx] = student_t_logp_scalar(
                        self.scalars[x.0], self.scalars[nu.0],
                        self.scalars[mu.0], self.scalars[sigma.0],
                    );
                }
                Op::UniformLogP { x, lower, upper } => {
                    self.scalars[idx] = uniform_logp_scalar(
                        self.scalars[x.0], self.scalars[lower.0], self.scalars[upper.0],
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
                        self.scalars[x.0], self.scalars[alpha.0], self.scalars[beta.0],
                    );
                }
                Op::BetaLogP { x, alpha, beta } => {
                    self.scalars[idx] = beta_logp_scalar(
                        self.scalars[x.0], self.scalars[alpha.0], self.scalars[beta.0],
                    );
                }
                Op::NormalObsLogP {
                    mu_vec,
                    sigma,
                    obs_data_idx,
                } => {
                    let sv = self.scalars[sigma.0];
                    let obs = &graph.obs_vectors[*obs_data_idx];
                    let s2 = sv * sv;
                    let log_norm = -0.5 * std::f64::consts::TAU.ln() - sv.ln();
                    let n = obs.len() as f64;
                    let mut sum_sq = 0.0f64;
                    for i in 0..vl {
                        let m = self.read_vec(mu_vec.0, i, graph);
                        let d = obs[i] - m;
                        sum_sq += d * d;
                    }
                    self.scalars[idx] = n * log_norm - 0.5 * sum_sq / s2;
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
                        let data = &graph.data_vectors[data_indices[k]];
                        for i in 0..vl {
                            out[i] += beta * data[i];
                        }
                    }
                }
                Op::MatVecMul { matrix_idx, param_start, n_params, intercept } => {
                    use faer::{col, mat, linalg::matmul::matmul, Parallelism};
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let matrix = &graph.data_matrices[*matrix_idx];
                    let base = intercept.map_or(0.0, |n| self.scalars[n.0]);
                    let out = &mut self.vec_buf[out_off..out_off + vl];
                    out.fill(base);
                    let x = mat::from_row_major_slice::<f64>(
                        &matrix.data, matrix.n_rows, matrix.n_cols,
                    );
                    let beta_slice = &params[*param_start..*param_start + *n_params];
                    let beta_col = col::from_slice::<f64>(beta_slice);
                    let out_col = col::from_slice_mut::<f64>(out);
                    // Use Rayon threads for matrices large enough to amortise spawn cost.
                    let par = if matrix.n_rows * matrix.n_cols >= 100_000 {
                        Parallelism::Rayon(0)
                    } else {
                        Parallelism::None
                    };
                    matmul(out_col.as_2d_mut(), x, beta_col.as_2d(), Some(1.0), 1.0, par);
                }
                Op::VectorNormalLogP { param_start, n_params, mu, sigma } => {
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
                Op::VectorHalfNormalLogP { param_start, n_params, sigma } => {
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
                Op::VectorStudentTLogP { param_start, n_params, nu, mu, sigma } => {
                    let log_norm = ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu)
                        - 0.5 * (nu * std::f64::consts::PI * sigma * sigma).ln();
                    let mut sum = 0.0f64;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        let z = (v - mu) / sigma;
                        sum += log_norm - 0.5 * (nu + 1.0) * (1.0 + z * z / nu).ln();
                    }
                    self.scalars[idx] = sum;
                }
                Op::VectorGammaLogP { param_start, n_params, alpha, beta } => {
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
                Op::VectorBetaLogP { param_start, n_params, alpha, beta } => {
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
                Op::VectorUniformLogP { param_start, n_params, .. } => {
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
        self.total_logp = graph
            .logp_terms
            .iter()
            .map(|id| self.scalars[id.0])
            .sum();

        // === Backward pass ===
        // Zero adjoint buffers and gradient
        self.adj_scalars.iter_mut().for_each(|x| *x = 0.0);
        self.adj_vec_buf.iter_mut().for_each(|x| *x = 0.0);
        self.grad.iter_mut().for_each(|x| *x = 0.0);

        // Seed
        for &id in &graph.logp_terms {
            self.adj_scalars[id.0] = 1.0;
        }

        for node in graph.nodes.iter().rev() {
            let idx = node.id.0;
            let a_s = self.adj_scalars[idx];

            match &node.op {
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
                    self.adj_scalars[sigma.0] += a_s * ((nv + 1.0) * z2 / (sv * nv * denom) - 1.0 / sv);
                    // d/dnu
                    self.adj_scalars[nu.0] += a_s * (
                        0.5 * digamma(0.5 * (nv + 1.0)) - 0.5 * digamma(0.5 * nv)
                        - 0.5 / nv
                        - 0.5 * (1.0 + z2 / nv).ln()
                        + 0.5 * (nv + 1.0) * z2 / (nv * nv * denom)
                    );
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
                        self.adj_scalars[alpha.0] += a_s * (digamma(av + bv) - digamma(av) + xv.ln());
                        self.adj_scalars[beta.0] += a_s * (digamma(av + bv) - digamma(bv) + (1.0 - xv).ln());
                    }
                }
                Op::NormalObsLogP {
                    mu_vec,
                    sigma,
                    obs_data_idx,
                } => {
                    let sv = self.scalars[sigma.0];
                    let obs = &graph.obs_vectors[*obs_data_idx];
                    let s2 = sv * sv;
                    let mut dsigma = 0.0f64;

                    let mu_off = match self.node_kind[mu_vec.0] {
                        NodeKind::ComputedVec(o) => Some(o),
                        _ => None,
                    };

                    for i in 0..vl {
                        let m = self.read_vec(mu_vec.0, i, graph);
                        let diff = obs[i] - m;
                        // Adjoint for mu_vec
                        if let Some(off) = mu_off {
                            self.adj_vec_buf[off + i] += a_s * diff / s2;
                        }
                        dsigma += diff * diff / (s2 * sv) - 1.0 / sv;
                    }
                    self.adj_scalars[sigma.0] += a_s * dsigma;
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
                        let data = &graph.data_vectors[data_indices[k]];
                        let mut ds = 0.0f64;
                        for i in 0..vl {
                            ds += adj[i] * data[i];
                        }
                        self.adj_scalars[pn.0] += ds;
                    }
                    if let Some(n) = intercept {
                        let mut ds = 0.0f64;
                        for i in 0..vl {
                            ds += adj[i];
                        }
                        self.adj_scalars[n.0] += ds;
                    }
                }
                Op::MatVecMul { matrix_idx, param_start, n_params, intercept } => {
                    use faer::{col, mat, linalg::matmul::matmul, Parallelism};
                    let out_off = match self.node_kind[idx] {
                        NodeKind::ComputedVec(o) => o,
                        _ => unreachable!(),
                    };
                    let matrix = &graph.data_matrices[*matrix_idx];
                    let x = mat::from_row_major_slice::<f64>(
                        &matrix.data, matrix.n_rows, matrix.n_cols,
                    );
                    let adj_slice = &self.adj_vec_buf[out_off..out_off + vl];
                    let adj_col = col::from_slice::<f64>(adj_slice);
                    let grad_slice = &mut self.grad[*param_start..*param_start + *n_params];
                    let grad_col = col::from_slice_mut::<f64>(grad_slice);
                    let par = if matrix.n_rows * matrix.n_cols >= 100_000 {
                        Parallelism::Rayon(0)
                    } else {
                        Parallelism::None
                    };
                    // grad += X^T @ adj
                    matmul(grad_col.as_2d_mut(), x.transpose(), adj_col.as_2d(), Some(1.0), 1.0, par);
                    if let Some(n) = intercept {
                        let ds: f64 = adj_slice.iter().sum();
                        self.adj_scalars[n.0] += ds;
                    }
                }
                Op::VectorNormalLogP { param_start, n_params, mu, sigma } => {
                    let s2 = sigma * sigma;
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        self.grad[param_start + k] += a_s * (-(v - mu) / s2);
                    }
                }
                Op::VectorHalfNormalLogP { param_start, n_params, sigma } => {
                    let s2 = sigma * sigma;
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        // d/draw = -exp(2·raw)/σ² + 1
                        self.grad[param_start + k] += a_s * (-(2.0 * raw).exp() / s2 + 1.0);
                    }
                }
                Op::VectorStudentTLogP { param_start, n_params, nu, mu, sigma } => {
                    for k in 0..*n_params {
                        let v = params[param_start + k];
                        let z = (v - mu) / sigma;
                        // d/dv = -(ν+1)·z / (σ·ν·(1 + z²/ν))
                        self.grad[param_start + k] += a_s * (-(nu + 1.0) * z / (sigma * nu * (1.0 + z * z / nu)));
                    }
                }
                Op::VectorGammaLogP { param_start, n_params, alpha, beta } => {
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        // d/draw = α - β·exp(raw)
                        self.grad[param_start + k] += a_s * (alpha - beta * raw.exp());
                    }
                }
                Op::VectorBetaLogP { param_start, n_params, alpha, beta } => {
                    for k in 0..*n_params {
                        let raw = params[param_start + k];
                        let s = 1.0 / (1.0 + (-raw).exp());
                        // d/draw = α·(1-s) - β·s
                        self.grad[param_start + k] += a_s * (alpha * (1.0 - s) - beta * s);
                    }
                }
                Op::VectorUniformLogP { param_start, n_params, .. } => {
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

#[derive(Debug, Clone)]
pub enum Value {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl Value {
    pub fn as_scalar(&self) -> f64 {
        match self {
            Value::Scalar(v) => *v,
            Value::Vector(_) => panic!("expected scalar, got vector"),
        }
    }

    pub fn as_vector(&self) -> &[f64] {
        match self {
            Value::Vector(v) => v,
            Value::Scalar(_) => panic!("expected vector, got scalar"),
        }
    }
}

pub fn forward(graph: &Graph, params: &[f64]) -> Vec<Value> {
    let mut values: Vec<Value> = Vec::with_capacity(graph.nodes.len());

    for node in &graph.nodes {
        let val = match &node.op {
            Op::Param(idx) => Value::Scalar(params[*idx]),
            Op::Constant(c) => Value::Scalar(*c),
            Op::Data(idx) => Value::Vector(graph.data_vectors[*idx].clone()),
            Op::Add(a, b) => Value::Scalar(values[a.0].as_scalar() + values[b.0].as_scalar()),
            Op::Sub(a, b) => Value::Scalar(values[a.0].as_scalar() - values[b.0].as_scalar()),
            Op::Mul(a, b) => Value::Scalar(values[a.0].as_scalar() * values[b.0].as_scalar()),
            Op::Div(a, b) => Value::Scalar(values[a.0].as_scalar() / values[b.0].as_scalar()),
            Op::Neg(a) => Value::Scalar(-values[a.0].as_scalar()),
            Op::Exp(a) => Value::Scalar(values[a.0].as_scalar().exp()),
            Op::Log(a) => Value::Scalar(values[a.0].as_scalar().ln()),
            Op::Sigmoid(a) => {
                let v = values[a.0].as_scalar();
                Value::Scalar(1.0 / (1.0 + (-v).exp()))
            }
            Op::Square(a) => {
                let v = values[a.0].as_scalar();
                Value::Scalar(v * v)
            }
            Op::ScalarMulData(scalar, data) => {
                let s = values[scalar.0].as_scalar();
                let d = values[data.0].as_vector();
                Value::Vector(d.iter().map(|x| s * x).collect())
            }
            Op::VectorAdd(a, b) => {
                let va = values[a.0].as_vector();
                let vb = values[b.0].as_vector();
                Value::Vector(va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect())
            }
            Op::ScalarBroadcastAdd(scalar, vec) => {
                let s = values[scalar.0].as_scalar();
                let v = values[vec.0].as_vector();
                Value::Vector(v.iter().map(|x| s + x).collect())
            }
            Op::ScalarBroadcast(scalar) => {
                let s = values[scalar.0].as_scalar();
                let n = graph.obs_vectors.first()
                    .or_else(|| graph.data_vectors.first())
                    .map_or(0, |v| v.len());
                Value::Vector(vec![s; n])
            }
            Op::NormalLogP { x, mu, sigma } => {
                Value::Scalar(normal_logp_scalar(values[x.0].as_scalar(), values[mu.0].as_scalar(), values[sigma.0].as_scalar()))
            }
            Op::HalfNormalLogP { x, sigma } => {
                Value::Scalar(half_normal_logp_scalar(values[x.0].as_scalar(), values[sigma.0].as_scalar()))
            }
            Op::StudentTLogP { x, nu, mu, sigma } => {
                Value::Scalar(student_t_logp_scalar(values[x.0].as_scalar(), values[nu.0].as_scalar(), values[mu.0].as_scalar(), values[sigma.0].as_scalar()))
            }
            Op::UniformLogP { x, lower, upper } => {
                Value::Scalar(uniform_logp_scalar(values[x.0].as_scalar(), values[lower.0].as_scalar(), values[upper.0].as_scalar()))
            }
            Op::BernoulliLogP { x, p } => {
                Value::Scalar(bernoulli_logp_scalar(values[x.0].as_scalar(), values[p.0].as_scalar()))
            }
            Op::PoissonLogP { x, lam } => {
                Value::Scalar(poisson_logp_scalar(values[x.0].as_scalar(), values[lam.0].as_scalar()))
            }
            Op::GammaLogP { x, alpha, beta } => {
                Value::Scalar(gamma_logp_scalar(values[x.0].as_scalar(), values[alpha.0].as_scalar(), values[beta.0].as_scalar()))
            }
            Op::BetaLogP { x, alpha, beta } => {
                Value::Scalar(beta_logp_scalar(values[x.0].as_scalar(), values[alpha.0].as_scalar(), values[beta.0].as_scalar()))
            }
            Op::NormalObsLogP {
                mu_vec,
                sigma,
                obs_data_idx,
            } => {
                let mu = values[mu_vec.0].as_vector();
                let sv = values[sigma.0].as_scalar();
                let obs = &graph.obs_vectors[*obs_data_idx];
                Value::Scalar(normal_obs_logp_sum(mu, sv, obs))
            }
            Op::FusedLinearMu {
                param_nodes,
                data_indices,
                intercept,
            } => {
                let vl = graph.data_vectors[data_indices[0]].len();
                let base = intercept.map_or(0.0, |n| values[n.0].as_scalar());
                let mut result = vec![base; vl];
                for (k, &pn) in param_nodes.iter().enumerate() {
                    let beta = values[pn.0].as_scalar();
                    let data = &graph.data_vectors[data_indices[k]];
                    for i in 0..vl {
                        result[i] += beta * data[i];
                    }
                }
                Value::Vector(result)
            }
            Op::MatVecMul { matrix_idx, param_start, n_params, intercept } => {
                let matrix = &graph.data_matrices[*matrix_idx];
                let base = intercept.map_or(0.0, |n| values[n.0].as_scalar());
                let mut result = vec![base; matrix.n_rows];
                for i in 0..matrix.n_rows {
                    for j in 0..*n_params {
                        result[i] += matrix.data[i * matrix.n_cols + j] * params[param_start + j];
                    }
                }
                Value::Vector(result)
            }
            Op::VectorNormalLogP { param_start, n_params, mu, sigma } => {
                let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
                let s2 = sigma * sigma;
                let sum: f64 = (0..*n_params).map(|k| {
                    let d = params[param_start + k] - mu;
                    log_norm - 0.5 * d * d / s2
                }).sum();
                Value::Scalar(sum)
            }
            Op::VectorHalfNormalLogP { param_start, n_params, sigma } => {
                let log_norm = (2.0 / (sigma * std::f64::consts::TAU.sqrt())).ln();
                let s2 = sigma * sigma;
                let sum: f64 = (0..*n_params).map(|k| {
                    let raw = params[param_start + k];
                    log_norm - (2.0 * raw).exp() / (2.0 * s2) + raw
                }).sum();
                Value::Scalar(sum)
            }
            Op::VectorStudentTLogP { param_start, n_params, nu, mu, sigma } => {
                let log_norm = ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu)
                    - 0.5 * (nu * std::f64::consts::PI * sigma * sigma).ln();
                let sum: f64 = (0..*n_params).map(|k| {
                    let v = params[param_start + k];
                    let z = (v - mu) / sigma;
                    log_norm - 0.5 * (nu + 1.0) * (1.0 + z * z / nu).ln()
                }).sum();
                Value::Scalar(sum)
            }
            Op::VectorGammaLogP { param_start, n_params, alpha, beta } => {
                let log_norm = alpha * beta.ln() - ln_gamma(*alpha);
                let sum: f64 = (0..*n_params).map(|k| {
                    let raw = params[param_start + k];
                    log_norm + alpha * raw - beta * raw.exp()
                }).sum();
                Value::Scalar(sum)
            }
            Op::VectorBetaLogP { param_start, n_params, alpha, beta } => {
                let log_norm = ln_gamma(alpha + beta) - ln_gamma(*alpha) - ln_gamma(*beta);
                let sum: f64 = (0..*n_params).map(|k| {
                    let raw = params[param_start + k];
                    let s = 1.0 / (1.0 + (-raw).exp());
                    log_norm + alpha * s.ln() + beta * (1.0 - s).ln()
                }).sum();
                Value::Scalar(sum)
            }
            Op::VectorUniformLogP { param_start, n_params, .. } => {
                let sum: f64 = (0..*n_params).map(|k| {
                    let raw = params[param_start + k];
                    let s = 1.0 / (1.0 + (-raw).exp());
                    s.ln() + (1.0 - s).ln()
                }).sum();
                Value::Scalar(sum)
            }
        };
        values.push(val);
    }
    values
}

pub fn eval_logp(graph: &Graph, params: &[f64]) -> f64 {
    let values = forward(graph, params);
    graph
        .logp_terms
        .iter()
        .map(|id| values[id.0].as_scalar())
        .sum()
}

pub fn grad_logp(graph: &Graph, params: &[f64]) -> (f64, Vec<f64>) {
    let values = forward(graph, params);
    let n = graph.nodes.len();

    let total_logp: f64 = graph
        .logp_terms
        .iter()
        .map(|id| values[id.0].as_scalar())
        .sum();

    let mut adj_scalar = vec![0.0f64; n];
    let mut adj_vector: Vec<Option<Vec<f64>>> = vec![None; n];
    let mut grad = vec![0.0f64; graph.param_count];

    for &id in &graph.logp_terms {
        adj_scalar[id.0] += 1.0;
    }

    for node in graph.nodes.iter().rev() {
        let idx = node.id.0;
        let a_s = adj_scalar[idx];

        match &node.op {
            Op::Param(_) | Op::Constant(_) | Op::Data(_) => {}
            Op::Add(a, b) => {
                adj_scalar[a.0] += a_s;
                adj_scalar[b.0] += a_s;
            }
            Op::Sub(a, b) => {
                adj_scalar[a.0] += a_s;
                adj_scalar[b.0] -= a_s;
            }
            Op::Mul(a, b) => {
                adj_scalar[a.0] += a_s * values[b.0].as_scalar();
                adj_scalar[b.0] += a_s * values[a.0].as_scalar();
            }
            Op::Div(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                adj_scalar[a.0] += a_s / vb;
                adj_scalar[b.0] -= a_s * va / (vb * vb);
            }
            Op::Neg(a) => adj_scalar[a.0] -= a_s,
            Op::Exp(a) => adj_scalar[a.0] += a_s * values[a.0].as_scalar().exp(),
            Op::Log(a) => adj_scalar[a.0] += a_s / values[a.0].as_scalar(),
            Op::Sigmoid(a) => {
                let s = values[idx].as_scalar();
                adj_scalar[a.0] += a_s * s * (1.0 - s);
            }
            Op::Square(a) => adj_scalar[a.0] += a_s * 2.0 * values[a.0].as_scalar(),
            Op::ScalarMulData(scalar, data) => {
                let s = values[scalar.0].as_scalar();
                let d = values[data.0].as_vector();
                if let Some(ref uv) = adj_vector[idx].take() {
                    let ds: f64 = uv.iter().zip(d.iter()).map(|(u, di)| u * di).sum();
                    adj_scalar[scalar.0] += ds;
                    let dd: Vec<f64> = uv.iter().map(|u| u * s).collect();
                    merge_vec_adj(&mut adj_vector[data.0], &dd);
                }
            }
            Op::VectorAdd(a, b) => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    merge_vec_adj(&mut adj_vector[a.0], uv);
                    merge_vec_adj(&mut adj_vector[b.0], uv);
                }
            }
            Op::ScalarBroadcastAdd(scalar, vec) => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    adj_scalar[scalar.0] += uv.iter().sum::<f64>();
                    merge_vec_adj(&mut adj_vector[vec.0], uv);
                }
            }
            Op::ScalarBroadcast(scalar) => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    adj_scalar[scalar.0] += uv.iter().sum::<f64>();
                }
            }
            Op::NormalLogP { x, mu, sigma } => {
                let xv = values[x.0].as_scalar();
                let mv = values[mu.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                let diff = xv - mv;
                let s2 = sv * sv;
                adj_scalar[x.0] += a_s * (-diff / s2);
                adj_scalar[mu.0] += a_s * (diff / s2);
                adj_scalar[sigma.0] += a_s * (diff * diff / (s2 * sv) - 1.0 / sv);
            }
            Op::HalfNormalLogP { x, sigma } => {
                let xv = values[x.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                if xv >= 0.0 {
                    adj_scalar[x.0] += a_s * (-xv / (sv * sv));
                    adj_scalar[sigma.0] += a_s * (xv * xv / (sv * sv * sv) - 1.0 / sv);
                }
            }
            Op::StudentTLogP { x, nu, mu, sigma } => {
                let xv = values[x.0].as_scalar();
                let nv = values[nu.0].as_scalar();
                let mv = values[mu.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                let z = (xv - mv) / sv;
                let z2 = z * z;
                let denom = 1.0 + z2 / nv;
                adj_scalar[x.0] += a_s * (-(nv + 1.0) * z / (sv * nv * denom));
                adj_scalar[mu.0] += a_s * ((nv + 1.0) * z / (sv * nv * denom));
                adj_scalar[sigma.0] += a_s * ((nv + 1.0) * z2 / (sv * nv * denom) - 1.0 / sv);
                adj_scalar[nu.0] += a_s * (
                    0.5 * digamma(0.5 * (nv + 1.0)) - 0.5 * digamma(0.5 * nv)
                    - 0.5 / nv - 0.5 * denom.ln()
                    + 0.5 * (nv + 1.0) * z2 / (nv * nv * denom)
                );
            }
            Op::UniformLogP { x: _, lower, upper } => {
                let lv = values[lower.0].as_scalar();
                let uv = values[upper.0].as_scalar();
                let range = uv - lv;
                if range > 0.0 {
                    adj_scalar[lower.0] += a_s / range;
                    adj_scalar[upper.0] -= a_s / range;
                }
            }
            Op::BernoulliLogP { x, p } => {
                let xv = values[x.0].as_scalar();
                let pv = values[p.0].as_scalar().clamp(1e-12, 1.0 - 1e-12);
                adj_scalar[p.0] += a_s * (xv / pv - (1.0 - xv) / (1.0 - pv));
            }
            Op::PoissonLogP { x, lam } => {
                let xv = values[x.0].as_scalar();
                let lv = values[lam.0].as_scalar();
                adj_scalar[lam.0] += a_s * (xv / lv - 1.0);
            }
            Op::GammaLogP { x, alpha, beta } => {
                let xv = values[x.0].as_scalar();
                let av = values[alpha.0].as_scalar();
                let bv = values[beta.0].as_scalar();
                if xv > 0.0 {
                    adj_scalar[x.0] += a_s * ((av - 1.0) / xv - bv);
                    adj_scalar[alpha.0] += a_s * (bv.ln() - digamma(av) + xv.ln());
                    adj_scalar[beta.0] += a_s * (av / bv - xv);
                }
            }
            Op::BetaLogP { x, alpha, beta } => {
                let xv = values[x.0].as_scalar();
                let av = values[alpha.0].as_scalar();
                let bv = values[beta.0].as_scalar();
                if xv > 0.0 && xv < 1.0 {
                    adj_scalar[x.0] += a_s * ((av - 1.0) / xv - (bv - 1.0) / (1.0 - xv));
                    adj_scalar[alpha.0] += a_s * (digamma(av + bv) - digamma(av) + xv.ln());
                    adj_scalar[beta.0] += a_s * (digamma(av + bv) - digamma(bv) + (1.0 - xv).ln());
                }
            }
            Op::NormalObsLogP {
                mu_vec,
                sigma,
                obs_data_idx,
            } => {
                let mu = values[mu_vec.0].as_vector();
                let sv = values[sigma.0].as_scalar();
                let obs = &graph.obs_vectors[*obs_data_idx];
                let s2 = sv * sv;
                let dmu: Vec<f64> = mu
                    .iter()
                    .zip(obs.iter())
                    .map(|(m, o)| a_s * (o - m) / s2)
                    .collect();
                merge_vec_adj(&mut adj_vector[mu_vec.0], &dmu);
                let dsigma: f64 = mu
                    .iter()
                    .zip(obs.iter())
                    .map(|(m, o)| {
                        let diff = o - m;
                        diff * diff / (s2 * sv) - 1.0 / sv
                    })
                    .sum::<f64>();
                adj_scalar[sigma.0] += a_s * dsigma;
            }
            Op::FusedLinearMu {
                param_nodes,
                data_indices,
                intercept,
            } => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    for (k, &pn) in param_nodes.iter().enumerate() {
                        let data = &graph.data_vectors[data_indices[k]];
                        let ds: f64 = uv.iter().zip(data.iter()).map(|(u, d)| u * d).sum();
                        adj_scalar[pn.0] += ds;
                    }
                    if let Some(n) = *intercept {
                        adj_scalar[n.0] += uv.iter().sum::<f64>();
                    }
                }
            }
            Op::MatVecMul { matrix_idx, param_start, n_params, intercept } => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    let matrix = &graph.data_matrices[*matrix_idx];
                    // grad[param_start + k] += sum_i X[i,k] * adj[i]
                    for k in 0..*n_params {
                        let mut ds = 0.0f64;
                        for i in 0..matrix.n_rows {
                            ds += uv[i] * matrix.data[i * matrix.n_cols + k];
                        }
                        grad[param_start + k] += ds;
                    }
                    if let Some(n) = *intercept {
                        adj_scalar[n.0] += uv.iter().sum::<f64>();
                    }
                }
            }
            Op::VectorNormalLogP { param_start, n_params, mu, sigma } => {
                let s2 = sigma * sigma;
                for k in 0..*n_params {
                    let v = params[param_start + k];
                    grad[param_start + k] += a_s * (-(v - mu) / s2);
                }
            }
            Op::VectorHalfNormalLogP { param_start, n_params, sigma } => {
                let s2 = sigma * sigma;
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    grad[param_start + k] += a_s * (-(2.0 * raw).exp() / s2 + 1.0);
                }
            }
            Op::VectorStudentTLogP { param_start, n_params, nu, mu, sigma } => {
                for k in 0..*n_params {
                    let v = params[param_start + k];
                    let z = (v - mu) / sigma;
                    grad[param_start + k] += a_s * (-(nu + 1.0) * z / (sigma * nu * (1.0 + z * z / nu)));
                }
            }
            Op::VectorGammaLogP { param_start, n_params, alpha, beta } => {
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    grad[param_start + k] += a_s * (alpha - beta * raw.exp());
                }
            }
            Op::VectorBetaLogP { param_start, n_params, alpha, beta } => {
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    let s = 1.0 / (1.0 + (-raw).exp());
                    grad[param_start + k] += a_s * (alpha * (1.0 - s) - beta * s);
                }
            }
            Op::VectorUniformLogP { param_start, n_params, .. } => {
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    let s = 1.0 / (1.0 + (-raw).exp());
                    grad[param_start + k] += a_s * (1.0 - 2.0 * s);
                }
            }
        }
    }

    for node in &graph.nodes {
        if let Op::Param(pidx) = node.op {
            grad[pidx] = adj_scalar[node.id.0];
        }
    }
    (total_logp, grad)
}

fn merge_vec_adj(slot: &mut Option<Vec<f64>>, incoming: &[f64]) {
    match slot {
        Some(ref mut existing) => {
            for (e, i) in existing.iter_mut().zip(incoming.iter()) {
                *e += i;
            }
        }
        None => {
            *slot = Some(incoming.to_vec());
        }
    }
}

fn normal_logp_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    let diff = x - mu;
    -0.5 * (diff * diff) / (sigma * sigma)
        - sigma.ln()
        - 0.5 * std::f64::consts::TAU.ln()
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

fn half_normal_logp_scalar(x: f64, sigma: f64) -> f64 {
    if x < 0.0 {
        return f64::NEG_INFINITY;
    }
    (2.0 / (sigma * std::f64::consts::TAU.sqrt())).ln() - x * x / (2.0 * sigma * sigma)
}

fn student_t_logp_scalar(x: f64, nu: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu)
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

/// Lanczos approximation to ln(Γ(x)) for x > 0.
fn ln_gamma(x: f64) -> f64 {
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
    let tmp = tmp - (y - 0.5) * tmp.ln();
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
    use crate::graph::Graph;

    #[test]
    fn test_normal_logp_gradient() {
        let mut g = Graph::new();
        let x = g.add_param("x");
        let mu = g.add_constant(0.0);
        let sigma = g.add_constant(1.0);
        g.normal_logp(x, mu, sigma);

        let params = vec![1.5];
        let (logp, grad) = grad_logp(&g, &params);
        assert!(
            (logp - (-0.5 * 1.5_f64.powi(2) - 0.5 * std::f64::consts::TAU.ln())).abs() < 1e-10
        );
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
        let num = (eval_logp(&g, &[params[0] + eps]) - eval_logp(&g, &[params[0] - eps]))
            / (2.0 * eps);
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
        for i in 0..3 {
            assert!(
                (eval.grad[i] - grad_old[i]).abs() < 1e-10,
                "grad[{}] mismatch: {} vs {}",
                i,
                eval.grad[i],
                grad_old[i]
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
                i, grad[i], num, (grad[i] - num).abs()
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
            eval.total_logp, logp_ref
        );
        for i in 0..params.len() {
            assert!(
                (eval.grad[i] - grad_ref[i]).abs() < 1e-8,
                "grad[{}] mismatch: Evaluator={} grad_logp={}",
                i, eval.grad[i], grad_ref[i]
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
                i, eval.grad[i], num
            );
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
            eval.total_logp, logp_ref
        );
        let eps = 1e-6;
        for i in 0..params.len() {
            assert!(
                (eval.grad[i] - grad_ref[i]).abs() < 1e-8,
                "grad[{}] mismatch: Evaluator={} grad_logp={}",
                i, eval.grad[i], grad_ref[i]
            );
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let num = (eval_logp(g, &p_plus) - eval_logp(g, &p_minus)) / (2.0 * eps);
            assert!(
                (eval.grad[i] - num).abs() < tol,
                "Evaluator grad[{}]: analytic={}, numerical={}",
                i, eval.grad[i], num
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
        let param_start = g.add_vector_params_with_transform("x", 3, ParamTransform::BoundedSigmoid { lower: 0.0, upper: 1.0 });
        g.vector_uniform_logp(param_start, 3, 0.0, 1.0);
        let params = vec![0.5, -0.3, 1.2];
        full_finite_diff_check(&g, &params, 1e-4);
    }
}
