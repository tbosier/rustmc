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
        let idx = self.param_count;
        self.param_count += 1;
        self.param_names.push(name.to_string());
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
