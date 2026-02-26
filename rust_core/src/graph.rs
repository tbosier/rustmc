use std::collections::HashMap;

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Operations supported in the computation graph.
#[derive(Debug, Clone)]
pub enum Op {
    /// A free parameter to be sampled (index into the parameter vector).
    Param(usize),
    /// A constant scalar value baked into the graph.
    Constant(f64),
    /// Observed data vector (index into the data table).
    Data(usize),
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Div(NodeId, NodeId),
    Neg(NodeId),
    Exp(NodeId),
    Log(NodeId),
    /// 1 / (1 + exp(-x))
    Sigmoid(NodeId),
    Square(NodeId),
    /// Element-wise multiply: scalar * data vector.
    ScalarMulData(NodeId, NodeId),
    /// Element-wise addition of two vectors.
    VectorAdd(NodeId, NodeId),
    /// Broadcast scalar + vector → vector.
    ScalarBroadcastAdd(NodeId, NodeId),
    /// Log-probability of a Normal distribution: logp(x | mu, sigma).
    NormalLogP {
        x: NodeId,
        mu: NodeId,
        sigma: NodeId,
    },
    /// Sum-of-log-probabilities for observed data under Normal(mu_vec, sigma).
    NormalObsLogP {
        mu_vec: NodeId,
        sigma: NodeId,
        obs_data_idx: usize,
    },
    /// logp(x | sigma) for x >= 0; HalfNormal
    HalfNormalLogP { x: NodeId, sigma: NodeId },
    /// logp(x | nu, mu, sigma); StudentT
    StudentTLogP { x: NodeId, nu: NodeId, mu: NodeId, sigma: NodeId },
    /// logp(x | lower, upper); Uniform
    UniformLogP { x: NodeId, lower: NodeId, upper: NodeId },
    /// logp(x | p); Bernoulli (x in {0, 1})
    BernoulliLogP { x: NodeId, p: NodeId },
    /// logp(x | lam); Poisson
    PoissonLogP { x: NodeId, lam: NodeId },
    /// logp(x | alpha, beta); Gamma
    GammaLogP { x: NodeId, alpha: NodeId, beta: NodeId },
    /// logp(x | alpha, beta); Beta
    BetaLogP { x: NodeId, alpha: NodeId, beta: NodeId },
    /// Fused linear combination: mu[i] = intercept + Σ_k params[k] * data[k][i]
    ///
    /// Replaces a chain of ScalarMulData + VectorAdd + ScalarBroadcastAdd with
    /// a single pass over the data, dramatically improving cache utilization.
    FusedLinearMu {
        param_nodes: Vec<NodeId>,
        data_indices: Vec<usize>,
        intercept: Option<NodeId>,
    },
}

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub name: Option<String>,
}

/// Transform applied to a parameter so NUTS samples on unconstrained space.
#[derive(Debug, Clone)]
pub enum ParamTransform {
    /// No transform — parameter is unconstrained.
    Identity,
    /// x = exp(raw). For parameters that must be > 0.
    Exp,
    /// x = sigmoid(raw). For parameters in (0, 1).
    Sigmoid,
    /// x = lower + (upper - lower) * sigmoid(raw). For parameters in (lower, upper).
    BoundedSigmoid { lower: f64, upper: f64 },
}

impl ParamTransform {
    pub fn apply(&self, raw: f64) -> f64 {
        match self {
            ParamTransform::Identity => raw,
            ParamTransform::Exp => raw.exp(),
            ParamTransform::Sigmoid => 1.0 / (1.0 + (-raw).exp()),
            ParamTransform::BoundedSigmoid { lower, upper } => {
                let s = 1.0 / (1.0 + (-raw).exp());
                lower + (upper - lower) * s
            }
        }
    }
}

/// The computational graph representing a probabilistic model.
///
/// Stores nodes in topological order (each node only references earlier nodes).
/// Data vectors and observed values are stored separately from the graph
/// structure so the graph itself stays lightweight and shareable across threads.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub param_count: usize,
    pub data_vectors: Vec<Vec<f64>>,
    pub obs_vectors: Vec<Vec<f64>>,
    pub param_names: Vec<String>,
    pub param_transforms: Vec<ParamTransform>,
    pub logp_terms: Vec<NodeId>,
    name_to_node: HashMap<String, NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            param_count: 0,
            data_vectors: Vec::new(),
            obs_vectors: Vec::new(),
            param_names: Vec::new(),
            param_transforms: Vec::new(),
            logp_terms: Vec::new(),
            name_to_node: HashMap::new(),
        }
    }

    fn add_node(&mut self, op: Op, name: Option<String>) -> NodeId {
        let id = NodeId(self.nodes.len());
        if let Some(ref n) = name {
            self.name_to_node.insert(n.clone(), id);
        }
        self.nodes.push(Node { id, op, name });
        id
    }

    pub fn add_param(&mut self, name: &str) -> NodeId {
        self.add_param_with_transform(name, ParamTransform::Identity)
    }

    pub fn add_param_with_transform(&mut self, name: &str, transform: ParamTransform) -> NodeId {
        let idx = self.param_count;
        self.param_count += 1;
        self.param_names.push(name.to_string());
        self.param_transforms.push(transform);
        self.add_node(Op::Param(idx), Some(name.to_string()))
    }

    pub fn add_constant(&mut self, value: f64) -> NodeId {
        self.add_node(Op::Constant(value), None)
    }

    pub fn add_data(&mut self, name: &str, values: Vec<f64>) -> NodeId {
        let idx = self.data_vectors.len();
        self.data_vectors.push(values);
        self.add_node(Op::Data(idx), Some(name.to_string()))
    }

    pub fn add_obs_data(&mut self, values: Vec<f64>) -> usize {
        let idx = self.obs_vectors.len();
        self.obs_vectors.push(values);
        idx
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Add(a, b), None)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Mul(a, b), None)
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Sub(a, b), None)
    }

    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::Div(a, b), None)
    }

    pub fn neg(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Neg(a), None)
    }

    pub fn exp(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Exp(a), None)
    }

    pub fn log(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Log(a), None)
    }

    pub fn sigmoid(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Sigmoid(a), None)
    }

    pub fn square(&mut self, a: NodeId) -> NodeId {
        self.add_node(Op::Square(a), None)
    }

    pub fn scalar_mul_data(&mut self, scalar: NodeId, data: NodeId) -> NodeId {
        self.add_node(Op::ScalarMulData(scalar, data), None)
    }

    pub fn vector_add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(Op::VectorAdd(a, b), None)
    }

    pub fn scalar_broadcast_add(&mut self, scalar: NodeId, vec: NodeId) -> NodeId {
        self.add_node(Op::ScalarBroadcastAdd(scalar, vec), None)
    }

    pub fn normal_logp(&mut self, x: NodeId, mu: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::NormalLogP { x, mu, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn normal_obs_logp(
        &mut self,
        mu_vec: NodeId,
        sigma: NodeId,
        obs_data_idx: usize,
    ) -> NodeId {
        let node = self.add_node(
            Op::NormalObsLogP {
                mu_vec,
                sigma,
                obs_data_idx,
            },
            None,
        );
        self.logp_terms.push(node);
        node
    }

    pub fn half_normal_logp(&mut self, x: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::HalfNormalLogP { x, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn student_t_logp(&mut self, x: NodeId, nu: NodeId, mu: NodeId, sigma: NodeId) -> NodeId {
        let node = self.add_node(Op::StudentTLogP { x, nu, mu, sigma }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn uniform_logp(&mut self, x: NodeId, lower: NodeId, upper: NodeId) -> NodeId {
        let node = self.add_node(Op::UniformLogP { x, lower, upper }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn bernoulli_logp(&mut self, x: NodeId, p: NodeId) -> NodeId {
        let node = self.add_node(Op::BernoulliLogP { x, p }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn poisson_logp(&mut self, x: NodeId, lam: NodeId) -> NodeId {
        let node = self.add_node(Op::PoissonLogP { x, lam }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn gamma_logp(&mut self, x: NodeId, alpha: NodeId, beta: NodeId) -> NodeId {
        let node = self.add_node(Op::GammaLogP { x, alpha, beta }, None);
        self.logp_terms.push(node);
        node
    }

    pub fn beta_logp(&mut self, x: NodeId, alpha: NodeId, beta: NodeId) -> NodeId {
        let node = self.add_node(Op::BetaLogP { x, alpha, beta }, None);
        self.logp_terms.push(node);
        node
    }

    /// Mark an existing node as a log-probability term (adds its value to total logp).
    pub fn add_logp_term(&mut self, node: NodeId) {
        self.logp_terms.push(node);
    }

    /// Convenience: add a node's value directly as a logp term (used for Jacobians).
    pub fn add_node_as_logp(&mut self, node: NodeId) -> NodeId {
        self.logp_terms.push(node);
        node
    }

    /// Store a data vector without creating a graph node (used by FusedLinearMu).
    pub fn store_data_vec(&mut self, values: Vec<f64>) -> usize {
        let idx = self.data_vectors.len();
        self.data_vectors.push(values);
        idx
    }

    pub fn fused_linear_mu(
        &mut self,
        param_nodes: Vec<NodeId>,
        data_indices: Vec<usize>,
        intercept: Option<NodeId>,
    ) -> NodeId {
        self.add_node(
            Op::FusedLinearMu {
                param_nodes,
                data_indices,
                intercept,
            },
            None,
        )
    }

    pub fn node_by_name(&self, name: &str) -> Option<NodeId> {
        self.name_to_node.get(name).copied()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
