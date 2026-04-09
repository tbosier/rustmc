use crate::graph::{Graph, NodeId, Op, ParamTransform};
use crate::sampler::{self, SampleResult, SamplerConfig};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

const ARTIFACT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializableObsFamily {
    Normal,
    BernoulliLogit,
    PoissonLog,
    ExponentialLog,
    LogNormal,
    NegativeBinomialLog,
}

impl From<&crate::graph::ObsFamily> for SerializableObsFamily {
    fn from(value: &crate::graph::ObsFamily) -> Self {
        match value {
            crate::graph::ObsFamily::Normal => Self::Normal,
            crate::graph::ObsFamily::BernoulliLogit => Self::BernoulliLogit,
            crate::graph::ObsFamily::PoissonLog => Self::PoissonLog,
            crate::graph::ObsFamily::ExponentialLog => Self::ExponentialLog,
            crate::graph::ObsFamily::LogNormal => Self::LogNormal,
            crate::graph::ObsFamily::NegativeBinomialLog => Self::NegativeBinomialLog,
        }
    }
}

impl From<&SerializableObsFamily> for crate::graph::ObsFamily {
    fn from(value: &SerializableObsFamily) -> Self {
        match value {
            SerializableObsFamily::Normal => crate::graph::ObsFamily::Normal,
            SerializableObsFamily::BernoulliLogit => crate::graph::ObsFamily::BernoulliLogit,
            SerializableObsFamily::PoissonLog => crate::graph::ObsFamily::PoissonLog,
            SerializableObsFamily::ExponentialLog => crate::graph::ObsFamily::ExponentialLog,
            SerializableObsFamily::LogNormal => crate::graph::ObsFamily::LogNormal,
            SerializableObsFamily::NegativeBinomialLog => crate::graph::ObsFamily::NegativeBinomialLog,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SerializableParamTransform {
    Identity,
    Exp,
    Sigmoid,
    BoundedSigmoid { lower: f64, upper: f64 },
}

impl From<&ParamTransform> for SerializableParamTransform {
    fn from(value: &ParamTransform) -> Self {
        match value {
            ParamTransform::Identity => Self::Identity,
            ParamTransform::Exp => Self::Exp,
            ParamTransform::Sigmoid => Self::Sigmoid,
            ParamTransform::BoundedSigmoid { lower, upper } => {
                Self::BoundedSigmoid {
                    lower: *lower,
                    upper: *upper,
                }
            }
        }
    }
}

impl From<&SerializableParamTransform> for ParamTransform {
    fn from(value: &SerializableParamTransform) -> Self {
        match value {
            SerializableParamTransform::Identity => ParamTransform::Identity,
            SerializableParamTransform::Exp => ParamTransform::Exp,
            SerializableParamTransform::Sigmoid => ParamTransform::Sigmoid,
            SerializableParamTransform::BoundedSigmoid { lower, upper } => {
                ParamTransform::BoundedSigmoid {
                    lower: *lower,
                    upper: *upper,
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub created_by: String,
    pub format_version: u32,
    pub param_count: usize,
    pub node_count: usize,
    pub notes: Vec<String>,
}

impl ArtifactMetadata {
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            created_by: "rustmc_core".to_string(),
            format_version: ARTIFACT_FORMAT_VERSION,
            param_count: 0,
            node_count: 0,
            notes: Vec::new(),
        }
    }
}

impl Default for ArtifactMetadata {
    fn default() -> Self {
        Self::new()
    }
}

pub use ArtifactMetadata as ModelMetadata;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterBlock {
    Scalar {
        name: String,
        transform: SerializableParamTransform,
    },
    Vector {
        name: String,
        len: usize,
        transform: SerializableParamTransform,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRef {
    Param(String),
    Node(usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelStep {
    Constant {
        value: f64,
    },
    Data {
        name: String,
        values: Vec<f64>,
    },
    Add {
        lhs: NodeRef,
        rhs: NodeRef,
    },
    Mul {
        lhs: NodeRef,
        rhs: NodeRef,
    },
    Sub {
        lhs: NodeRef,
        rhs: NodeRef,
    },
    Div {
        lhs: NodeRef,
        rhs: NodeRef,
    },
    Neg {
        input: NodeRef,
    },
    Exp {
        input: NodeRef,
    },
    Log {
        input: NodeRef,
    },
    Sigmoid {
        input: NodeRef,
    },
    Square {
        input: NodeRef,
    },
    ScalarMulData {
        scalar: NodeRef,
        data: NodeRef,
    },
    VectorAdd {
        lhs: NodeRef,
        rhs: NodeRef,
    },
    ScalarBroadcastAdd {
        scalar: NodeRef,
        vector: NodeRef,
    },
    ScalarBroadcast {
        scalar: NodeRef,
    },
    NormalLogP {
        x: NodeRef,
        mu: NodeRef,
        sigma: NodeRef,
    },
    Observation {
        family: SerializableObsFamily,
        linpred: NodeRef,
        aux: Option<NodeRef>,
        obs: Vec<f64>,
    },
    HalfNormalLogP {
        x: NodeRef,
        sigma: NodeRef,
    },
    StudentTLogP {
        x: NodeRef,
        nu: NodeRef,
        mu: NodeRef,
        sigma: NodeRef,
    },
    UniformLogP {
        x: NodeRef,
        lower: NodeRef,
        upper: NodeRef,
    },
    BernoulliLogP {
        x: NodeRef,
        p: NodeRef,
    },
    PoissonLogP {
        x: NodeRef,
        lam: NodeRef,
    },
    GammaLogP {
        x: NodeRef,
        alpha: NodeRef,
        beta: NodeRef,
    },
    BetaLogP {
        x: NodeRef,
        alpha: NodeRef,
        beta: NodeRef,
    },
    FusedLinearMu {
        param_names: Vec<String>,
        term_data: Vec<Vec<f64>>,
        intercept: Option<NodeRef>,
    },
    MatVecMul {
        param_name: String,
        matrix: Vec<f64>,
        n_rows: usize,
        n_cols: usize,
        intercept: Option<NodeRef>,
    },
    VectorNormalLogP {
        param_name: String,
        n_params: usize,
        mu: f64,
        sigma: f64,
    },
    VectorHalfNormalLogP {
        param_name: String,
        n_params: usize,
        sigma: f64,
    },
    VectorStudentTLogP {
        param_name: String,
        n_params: usize,
        nu: f64,
        mu: f64,
        sigma: f64,
    },
    VectorGammaLogP {
        param_name: String,
        n_params: usize,
        alpha: f64,
        beta: f64,
    },
    VectorBetaLogP {
        param_name: String,
        n_params: usize,
        alpha: f64,
        beta: f64,
    },
    VectorUniformLogP {
        param_name: String,
        n_params: usize,
        lower: f64,
        upper: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledModelArtifact {
    pub format_version: u32,
    pub metadata: ArtifactMetadata,
    pub parameter_blocks: Vec<ParameterBlock>,
    pub steps: Vec<ModelStep>,
}

impl CompiledModelArtifact {
    pub fn from_graph(graph: &Graph) -> Result<Self, ArtifactError> {
        let (parameter_blocks, param_index_to_block_name) = group_parameter_blocks(graph)?;
        let mut original_to_step_ref: HashMap<usize, NodeRef> = HashMap::new();
        let mut steps = Vec::new();

        for node in &graph.nodes {
            match &node.op {
                Op::Param(pidx) => {
                    let name = graph.param_names[*pidx].clone();
                    original_to_step_ref.insert(node.id.0, NodeRef::Param(name));
                }
                _ => {
                    let step = match &node.op {
                        Op::Constant(value) => ModelStep::Constant { value: *value },
                        Op::Data(idx) => ModelStep::Data {
                            name: node
                                .name
                                .clone()
                                .unwrap_or_else(|| format!("data_{}", idx)),
                            values: graph.data_vectors[*idx].clone(),
                        },
                        Op::Add(lhs, rhs) => ModelStep::Add {
                            lhs: translate_ref(*lhs, &original_to_step_ref)?,
                            rhs: translate_ref(*rhs, &original_to_step_ref)?,
                        },
                        Op::Mul(lhs, rhs) => ModelStep::Mul {
                            lhs: translate_ref(*lhs, &original_to_step_ref)?,
                            rhs: translate_ref(*rhs, &original_to_step_ref)?,
                        },
                        Op::Sub(lhs, rhs) => ModelStep::Sub {
                            lhs: translate_ref(*lhs, &original_to_step_ref)?,
                            rhs: translate_ref(*rhs, &original_to_step_ref)?,
                        },
                        Op::Div(lhs, rhs) => ModelStep::Div {
                            lhs: translate_ref(*lhs, &original_to_step_ref)?,
                            rhs: translate_ref(*rhs, &original_to_step_ref)?,
                        },
                        Op::Neg(input) => ModelStep::Neg {
                            input: translate_ref(*input, &original_to_step_ref)?,
                        },
                        Op::Exp(input) => ModelStep::Exp {
                            input: translate_ref(*input, &original_to_step_ref)?,
                        },
                        Op::Log(input) => ModelStep::Log {
                            input: translate_ref(*input, &original_to_step_ref)?,
                        },
                        Op::Sigmoid(input) => ModelStep::Sigmoid {
                            input: translate_ref(*input, &original_to_step_ref)?,
                        },
                        Op::Square(input) => ModelStep::Square {
                            input: translate_ref(*input, &original_to_step_ref)?,
                        },
                        Op::ScalarMulData(scalar, data) => ModelStep::ScalarMulData {
                            scalar: translate_ref(*scalar, &original_to_step_ref)?,
                            data: translate_ref(*data, &original_to_step_ref)?,
                        },
                        Op::VectorAdd(lhs, rhs) => ModelStep::VectorAdd {
                            lhs: translate_ref(*lhs, &original_to_step_ref)?,
                            rhs: translate_ref(*rhs, &original_to_step_ref)?,
                        },
                        Op::ScalarBroadcastAdd(scalar, vector) => {
                            ModelStep::ScalarBroadcastAdd {
                                scalar: translate_ref(*scalar, &original_to_step_ref)?,
                                vector: translate_ref(*vector, &original_to_step_ref)?,
                            }
                        }
                        Op::ScalarBroadcast(scalar) => ModelStep::ScalarBroadcast {
                            scalar: translate_ref(*scalar, &original_to_step_ref)?,
                        },
                        Op::NormalLogP { x, mu, sigma } => ModelStep::NormalLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            mu: translate_ref(*mu, &original_to_step_ref)?,
                            sigma: translate_ref(*sigma, &original_to_step_ref)?,
                        },
                        Op::ObsLogP {
                            family,
                            linpred_vec,
                            aux,
                            obs_data_idx,
                        } => ModelStep::Observation {
                            family: SerializableObsFamily::from(family),
                            linpred: translate_ref(*linpred_vec, &original_to_step_ref)?,
                            aux: aux
                                .as_ref()
                                .map(|node| translate_ref(*node, &original_to_step_ref))
                                .transpose()?,
                            obs: graph.obs_vectors[*obs_data_idx].clone(),
                        },
                        Op::HalfNormalLogP { x, sigma } => ModelStep::HalfNormalLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            sigma: translate_ref(*sigma, &original_to_step_ref)?,
                        },
                        Op::StudentTLogP { x, nu, mu, sigma } => ModelStep::StudentTLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            nu: translate_ref(*nu, &original_to_step_ref)?,
                            mu: translate_ref(*mu, &original_to_step_ref)?,
                            sigma: translate_ref(*sigma, &original_to_step_ref)?,
                        },
                        Op::UniformLogP { x, lower, upper } => ModelStep::UniformLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            lower: translate_ref(*lower, &original_to_step_ref)?,
                            upper: translate_ref(*upper, &original_to_step_ref)?,
                        },
                        Op::BernoulliLogP { x, p } => ModelStep::BernoulliLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            p: translate_ref(*p, &original_to_step_ref)?,
                        },
                        Op::PoissonLogP { x, lam } => ModelStep::PoissonLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            lam: translate_ref(*lam, &original_to_step_ref)?,
                        },
                        Op::GammaLogP { x, alpha, beta } => ModelStep::GammaLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            alpha: translate_ref(*alpha, &original_to_step_ref)?,
                            beta: translate_ref(*beta, &original_to_step_ref)?,
                        },
                        Op::BetaLogP { x, alpha, beta } => ModelStep::BetaLogP {
                            x: translate_ref(*x, &original_to_step_ref)?,
                            alpha: translate_ref(*alpha, &original_to_step_ref)?,
                            beta: translate_ref(*beta, &original_to_step_ref)?,
                        },
                        Op::FusedLinearMu {
                            param_nodes,
                            data_indices,
                            intercept,
                        } => {
                            let param_names = param_nodes
                                .iter()
                                .map(|nid| {
                                    match graph.nodes[nid.0].op {
                                        Op::Param(pidx) => Ok(graph.param_names[pidx].clone()),
                                        _ => Err(ArtifactError::unsupported(
                                            "FusedLinearMu contained a non-parameter node",
                                        )),
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let term_data = data_indices
                                .iter()
                                .map(|idx| graph.data_vectors[*idx].clone())
                                .collect();
                            ModelStep::FusedLinearMu {
                                param_names,
                                term_data,
                                intercept: intercept
                                    .as_ref()
                                    .map(|n| translate_ref(*n, &original_to_step_ref))
                                    .transpose()?,
                            }
                        }
                        Op::MatVecMul {
                            matrix_idx,
                            param_start,
                            n_params: _,
                            intercept,
                        } => {
                            let param_name = param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?;
                            let matrix = graph.data_matrices[*matrix_idx].data.clone();
                            ModelStep::MatVecMul {
                                param_name,
                                matrix,
                                n_rows: graph.data_matrices[*matrix_idx].n_rows,
                                n_cols: graph.data_matrices[*matrix_idx].n_cols,
                                intercept: intercept
                                    .as_ref()
                                    .map(|n| translate_ref(*n, &original_to_step_ref))
                                    .transpose()?,
                            }
                        }
                        Op::VectorNormalLogP {
                            param_start,
                            n_params,
                            mu,
                            sigma,
                        } => ModelStep::VectorNormalLogP {
                            param_name: param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?,
                            n_params: *n_params,
                            mu: *mu,
                            sigma: *sigma,
                        },
                        Op::VectorHalfNormalLogP {
                            param_start,
                            n_params,
                            sigma,
                        } => ModelStep::VectorHalfNormalLogP {
                            param_name: param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?,
                            n_params: *n_params,
                            sigma: *sigma,
                        },
                        Op::VectorStudentTLogP {
                            param_start,
                            n_params,
                            nu,
                            mu,
                            sigma,
                        } => ModelStep::VectorStudentTLogP {
                            param_name: param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?,
                            n_params: *n_params,
                            nu: *nu,
                            mu: *mu,
                            sigma: *sigma,
                        },
                        Op::VectorGammaLogP {
                            param_start,
                            n_params,
                            alpha,
                            beta,
                        } => ModelStep::VectorGammaLogP {
                            param_name: param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?,
                            n_params: *n_params,
                            alpha: *alpha,
                            beta: *beta,
                        },
                        Op::VectorBetaLogP {
                            param_start,
                            n_params,
                            alpha,
                            beta,
                        } => ModelStep::VectorBetaLogP {
                            param_name: param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?,
                            n_params: *n_params,
                            alpha: *alpha,
                            beta: *beta,
                        },
                        Op::VectorUniformLogP {
                            param_start,
                            n_params,
                            lower,
                            upper,
                        } => ModelStep::VectorUniformLogP {
                            param_name: param_index_to_block_name
                                .get(param_start)
                                .cloned()
                                .ok_or_else(|| {
                                    ArtifactError::invalid(format!(
                                        "missing vector parameter block for param_start {}",
                                        param_start
                                    ))
                                })?,
                            n_params: *n_params,
                            lower: *lower,
                            upper: *upper,
                        },
                        Op::Param(_) => unreachable!("parameter nodes are handled separately"),
                    };

                    let artifact_index = steps.len();
                    original_to_step_ref.insert(node.id.0, NodeRef::Node(artifact_index));
                    steps.push(step);
                }
            }
        }

        Ok(Self {
            format_version: ARTIFACT_FORMAT_VERSION,
            metadata: ArtifactMetadata {
                param_count: graph.param_count,
                node_count: graph.nodes.len(),
                ..ArtifactMetadata::new()
            },
            parameter_blocks,
            steps,
        })
    }

    pub fn validate(&self) -> Result<(), ArtifactError> {
        if self.format_version != ARTIFACT_FORMAT_VERSION {
            return Err(ArtifactError::VersionMismatch {
                found: self.format_version,
                expected: ARTIFACT_FORMAT_VERSION,
            });
        }

        let mut seen_names = HashSet::new();
        for block in &self.parameter_blocks {
            let name = match block {
                ParameterBlock::Scalar { name, .. } | ParameterBlock::Vector { name, .. } => {
                    name
                }
            };
            if !seen_names.insert(name) {
                return Err(ArtifactError::invalid(format!(
                    "duplicate parameter block name '{}'",
                    name
                )));
            }
        }

        let scalar_names: HashSet<&str> = self
            .parameter_blocks
            .iter()
            .filter_map(|block| match block {
                ParameterBlock::Scalar { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let vector_lengths: HashMap<&str, usize> = self
            .parameter_blocks
            .iter()
            .filter_map(|block| match block {
                ParameterBlock::Vector { name, len, .. } => Some((name.as_str(), *len)),
                _ => None,
            })
            .collect();
        let vector_names: HashSet<&str> = vector_lengths.keys().copied().collect();

        for (idx, step) in self.steps.iter().enumerate() {
            validate_step(
                step,
                idx,
                &scalar_names,
                &vector_names,
                &vector_lengths,
            )?;
        }

        Ok(())
    }

    pub fn to_json_pretty(&self) -> Result<String, ArtifactError> {
        serde_json::to_string_pretty(self).map_err(|err| ArtifactError::serialization(err.to_string()))
    }

    pub fn from_json_str(input: &str) -> Result<Self, ArtifactError> {
        let artifact: Self = serde_json::from_str(input)
            .map_err(|err| ArtifactError::serialization(err.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }
}

#[derive(Debug, Clone)]
pub struct CompiledModelRuntime {
    artifact: CompiledModelArtifact,
}

impl CompiledModelRuntime {
    pub fn from_graph(graph: &Graph) -> Result<Self, ArtifactError> {
        let artifact = CompiledModelArtifact::from_graph(graph)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: CompiledModelArtifact) -> Result<Self, ArtifactError> {
        artifact.validate()?;
        Ok(Self { artifact })
    }

    pub fn artifact(&self) -> &CompiledModelArtifact {
        &self.artifact
    }

    pub fn into_artifact(self) -> CompiledModelArtifact {
        self.artifact
    }

    pub fn to_graph(&self) -> Result<Graph, ArtifactError> {
        build_graph(&self.artifact)
    }

    pub fn sample(&self, config: SamplerConfig) -> Result<SampleResult, ArtifactError> {
        let graph = self.to_graph()?;
        sampler::sample(graph, config).map_err(ArtifactError::sampler)
    }

    pub fn to_json_pretty(&self) -> Result<String, ArtifactError> {
        self.artifact.to_json_pretty()
    }

    pub fn from_json_str(input: &str) -> Result<Self, ArtifactError> {
        Self::from_artifact(CompiledModelArtifact::from_json_str(input)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    VersionMismatch { found: u32, expected: u32 },
    Invalid(String),
    Unsupported(String),
    MissingParameter(String),
    MissingNode(usize),
    Serialization(String),
    Sampler(String),
}

impl ArtifactError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    fn missing_parameter(message: impl Into<String>) -> Self {
        Self::MissingParameter(message.into())
    }

    fn missing_node(idx: usize) -> Self {
        Self::MissingNode(idx)
    }

    fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization(message.into())
    }

    fn sampler(message: String) -> Self {
        Self::Sampler(message)
    }
}

impl Display for ArtifactError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactError::VersionMismatch { found, expected } => write!(
                f,
                "artifact format version mismatch: found {}, expected {}",
                found, expected
            ),
            ArtifactError::Invalid(message) => write!(f, "invalid artifact: {}", message),
            ArtifactError::Unsupported(message) => write!(f, "unsupported artifact: {}", message),
            ArtifactError::MissingParameter(message) => {
                write!(f, "missing parameter reference: {}", message)
            }
            ArtifactError::MissingNode(idx) => write!(f, "missing node reference {}", idx),
            ArtifactError::Serialization(message) => {
                write!(f, "artifact serialization error: {}", message)
            }
            ArtifactError::Sampler(message) => write!(f, "sampler error: {}", message),
        }
    }
}

impl Error for ArtifactError {}

fn translate_ref(
    node: NodeId,
    map: &HashMap<usize, NodeRef>,
) -> Result<NodeRef, ArtifactError> {
    map.get(&node.0)
        .cloned()
        .ok_or_else(|| ArtifactError::missing_node(node.0))
}

fn build_graph(artifact: &CompiledModelArtifact) -> Result<Graph, ArtifactError> {
    let mut graph = Graph::new();
    let mut scalar_param_nodes: HashMap<String, NodeId> = HashMap::new();
    let mut vector_param_starts: HashMap<String, usize> = HashMap::new();
    let mut op_nodes: Vec<NodeId> = Vec::with_capacity(artifact.steps.len());

    for block in &artifact.parameter_blocks {
        match block {
            ParameterBlock::Scalar { name, transform } => {
                let node = graph.add_param_with_transform(name, transform.into());
                scalar_param_nodes.insert(name.clone(), node);
            }
            ParameterBlock::Vector {
                name,
                len,
                transform,
            } => {
                let start = graph.add_vector_params_with_transform(name, *len, transform.into());
                vector_param_starts.insert(name.clone(), start);
            }
        }
    }

    for (idx, step) in artifact.steps.iter().enumerate() {
        let node = match step {
            ModelStep::Constant { value } => graph.add_constant(*value),
            ModelStep::Data { name, values } => graph.add_data(name, values.clone()),
            ModelStep::Add { lhs, rhs } => graph.add(
                resolve_node_ref(lhs, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(rhs, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::Mul { lhs, rhs } => graph.mul(
                resolve_node_ref(lhs, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(rhs, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::Sub { lhs, rhs } => graph.sub(
                resolve_node_ref(lhs, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(rhs, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::Div { lhs, rhs } => graph.div(
                resolve_node_ref(lhs, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(rhs, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::Neg { input } => {
                graph.neg(resolve_node_ref(input, &scalar_param_nodes, &op_nodes)?)
            }
            ModelStep::Exp { input } => {
                graph.exp(resolve_node_ref(input, &scalar_param_nodes, &op_nodes)?)
            }
            ModelStep::Log { input } => {
                graph.log(resolve_node_ref(input, &scalar_param_nodes, &op_nodes)?)
            }
            ModelStep::Sigmoid { input } => {
                graph.sigmoid(resolve_node_ref(input, &scalar_param_nodes, &op_nodes)?)
            }
            ModelStep::Square { input } => {
                graph.square(resolve_node_ref(input, &scalar_param_nodes, &op_nodes)?)
            }
            ModelStep::ScalarMulData { scalar, data } => graph.scalar_mul_data(
                resolve_node_ref(scalar, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(data, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::VectorAdd { lhs, rhs } => graph.vector_add(
                resolve_node_ref(lhs, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(rhs, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::ScalarBroadcastAdd { scalar, vector } => graph.scalar_broadcast_add(
                resolve_node_ref(scalar, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(vector, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::ScalarBroadcast { scalar } => {
                graph.scalar_broadcast(resolve_node_ref(scalar, &scalar_param_nodes, &op_nodes)?)
            }
            ModelStep::NormalLogP { x, mu, sigma } => graph.normal_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(mu, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(sigma, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::Observation {
                family,
                linpred,
                aux,
                obs,
            } => {
                let obs_idx = graph.add_obs_data(obs.clone());
                let linpred = resolve_node_ref(linpred, &scalar_param_nodes, &op_nodes)?;
                match family {
                    SerializableObsFamily::Normal => {
                        let sigma = aux.as_ref().ok_or_else(|| {
                            ArtifactError::invalid(
                                "normal observation step is missing sigma",
                            )
                        })?;
                        graph.obs_logp_normal(
                            linpred,
                            resolve_node_ref(sigma, &scalar_param_nodes, &op_nodes)?,
                            obs_idx,
                        )
                    }
                    SerializableObsFamily::BernoulliLogit => {
                        if aux.is_some() {
                            return Err(ArtifactError::invalid(
                                "bernoulli-logit observation must not carry aux data",
                            ));
                        }
                        graph.obs_logp_bernoulli_logit(linpred, obs_idx)
                    }
                    SerializableObsFamily::PoissonLog => {
                        if aux.is_some() {
                            return Err(ArtifactError::invalid(
                                "poisson-log observation must not carry aux data",
                            ));
                        }
                        graph.obs_logp_poisson_log(linpred, obs_idx)
                    }
                    SerializableObsFamily::ExponentialLog => {
                        if aux.is_some() {
                            return Err(ArtifactError::invalid(
                                "exponential-log observation must not carry aux data",
                            ));
                        }
                        graph.obs_logp_exponential_log(linpred, obs_idx)
                    }
                    SerializableObsFamily::LogNormal => {
                        let sigma = aux.as_ref().ok_or_else(|| {
                            ArtifactError::invalid(
                                "log-normal observation step is missing sigma",
                            )
                        })?;
                        graph.obs_logp_lognormal(
                            linpred,
                            resolve_node_ref(sigma, &scalar_param_nodes, &op_nodes)?,
                            obs_idx,
                        )
                    }
                    SerializableObsFamily::NegativeBinomialLog => {
                        let alpha = aux.as_ref().ok_or_else(|| {
                            ArtifactError::invalid(
                                "negative-binomial observation step is missing alpha",
                            )
                        })?;
                        graph.obs_logp_negative_binomial_log(
                            linpred,
                            resolve_node_ref(alpha, &scalar_param_nodes, &op_nodes)?,
                            obs_idx,
                        )
                    }
                }
            }
            ModelStep::HalfNormalLogP { x, sigma } => graph.half_normal_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(sigma, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::StudentTLogP { x, nu, mu, sigma } => graph.student_t_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(nu, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(mu, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(sigma, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::UniformLogP { x, lower, upper } => graph.uniform_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(lower, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(upper, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::BernoulliLogP { x, p } => graph.bernoulli_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(p, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::PoissonLogP { x, lam } => graph.poisson_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(lam, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::GammaLogP { x, alpha, beta } => graph.gamma_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(alpha, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(beta, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::BetaLogP { x, alpha, beta } => graph.beta_logp(
                resolve_node_ref(x, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(alpha, &scalar_param_nodes, &op_nodes)?,
                resolve_node_ref(beta, &scalar_param_nodes, &op_nodes)?,
            ),
            ModelStep::FusedLinearMu {
                param_names,
                term_data,
                intercept,
            } => {
                let param_nodes = param_names
                    .iter()
                    .map(|name| {
                        scalar_param_nodes.get(name).copied().ok_or_else(|| {
                            ArtifactError::missing_parameter(format!(
                                "unknown scalar parameter '{}'",
                                name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let data_indices = term_data
                    .iter()
                    .cloned()
                    .map(|values| graph.store_data_vec(values))
                    .collect();
                graph.fused_linear_mu(
                    param_nodes,
                    data_indices,
                    intercept
                        .as_ref()
                        .map(|r| resolve_node_ref(r, &scalar_param_nodes, &op_nodes))
                        .transpose()?,
                )
            }
            ModelStep::MatVecMul {
                param_name,
                matrix,
                n_rows,
                n_cols,
                intercept,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                let matrix_idx = graph.store_matrix(matrix.clone(), *n_rows, *n_cols);
                graph.mat_vec_mul(
                    matrix_idx,
                    param_start,
                    *n_cols,
                    intercept
                        .as_ref()
                        .map(|r| resolve_node_ref(r, &scalar_param_nodes, &op_nodes))
                        .transpose()?,
                )
            }
            ModelStep::VectorNormalLogP {
                param_name,
                n_params,
                mu,
                sigma,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                graph.vector_normal_logp(param_start, *n_params, *mu, *sigma)
            }
            ModelStep::VectorHalfNormalLogP {
                param_name,
                n_params,
                sigma,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                graph.vector_half_normal_logp(param_start, *n_params, *sigma)
            }
            ModelStep::VectorStudentTLogP {
                param_name,
                n_params,
                nu,
                mu,
                sigma,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                graph.vector_student_t_logp(param_start, *n_params, *nu, *mu, *sigma)
            }
            ModelStep::VectorGammaLogP {
                param_name,
                n_params,
                alpha,
                beta,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                graph.vector_gamma_logp(param_start, *n_params, *alpha, *beta)
            }
            ModelStep::VectorBetaLogP {
                param_name,
                n_params,
                alpha,
                beta,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                graph.vector_beta_logp(param_start, *n_params, *alpha, *beta)
            }
            ModelStep::VectorUniformLogP {
                param_name,
                n_params,
                lower,
                upper,
            } => {
                let param_start = vector_param_starts.get(param_name).copied().ok_or_else(|| {
                    ArtifactError::missing_parameter(format!(
                        "unknown vector parameter '{}'",
                        param_name
                    ))
                })?;
                graph.vector_uniform_logp(param_start, *n_params, *lower, *upper)
            }
        };

        op_nodes.push(node);
        if op_nodes.len() != idx + 1 {
            return Err(ArtifactError::invalid("internal node replay mismatch"));
        }
    }

    Ok(graph)
}

fn resolve_node_ref(
    reference: &NodeRef,
    scalar_param_nodes: &HashMap<String, NodeId>,
    op_nodes: &[NodeId],
) -> Result<NodeId, ArtifactError> {
    match reference {
        NodeRef::Param(name) => scalar_param_nodes.get(name).copied().ok_or_else(|| {
            ArtifactError::missing_parameter(format!("unknown scalar parameter '{}'", name))
        }),
        NodeRef::Node(idx) => op_nodes
            .get(*idx)
            .copied()
            .ok_or_else(|| ArtifactError::missing_node(*idx)),
    }
}

fn validate_step(
    step: &ModelStep,
    step_idx: usize,
    scalar_names: &HashSet<&str>,
    vector_names: &HashSet<&str>,
    vector_lengths: &HashMap<&str, usize>,
) -> Result<(), ArtifactError> {
    let check_ref = |r: &NodeRef| -> Result<(), ArtifactError> {
        match r {
            NodeRef::Param(name) => {
                if scalar_names.contains(name.as_str()) {
                    Ok(())
                } else {
                    Err(ArtifactError::missing_parameter(format!(
                        "unknown scalar parameter '{}'",
                        name
                    )))
                }
            }
            NodeRef::Node(idx) => {
                if *idx < step_idx {
                    Ok(())
                } else {
                    Err(ArtifactError::missing_node(*idx))
                }
            }
        }
    };

    match step {
        ModelStep::Constant { .. } | ModelStep::Data { .. } => Ok(()),
        ModelStep::Add { lhs, rhs }
        | ModelStep::Mul { lhs, rhs }
        | ModelStep::Sub { lhs, rhs }
        | ModelStep::Div { lhs, rhs }
        | ModelStep::VectorAdd { lhs, rhs } => {
            check_ref(lhs)?;
            check_ref(rhs)
        }
        ModelStep::Neg { input }
        | ModelStep::Exp { input }
        | ModelStep::Log { input }
        | ModelStep::Sigmoid { input }
        | ModelStep::Square { input }
        | ModelStep::ScalarBroadcast { scalar: input } => check_ref(input),
        ModelStep::ScalarMulData { scalar, data } => {
            check_ref(scalar)?;
            check_ref(data)
        }
        ModelStep::ScalarBroadcastAdd { scalar, vector } => {
            check_ref(scalar)?;
            check_ref(vector)
        }
        ModelStep::NormalLogP { x, mu, sigma } => {
            check_ref(x)?;
            check_ref(mu)?;
            check_ref(sigma)
        }
        ModelStep::Observation {
            family,
            linpred,
            aux,
            obs,
        } => {
            if obs.is_empty() {
                return Err(ArtifactError::invalid(format!(
                    "observation step {} has no observed values",
                    step_idx
                )));
            }
            check_ref(linpred)?;
            match family {
                SerializableObsFamily::Normal => {
                    let sigma = aux.as_ref().ok_or_else(|| {
                        ArtifactError::invalid(format!(
                            "normal observation step {} is missing sigma",
                            step_idx
                        ))
                    })?;
                    check_ref(sigma)
                }
                SerializableObsFamily::BernoulliLogit => {
                    if aux.is_some() {
                        return Err(ArtifactError::invalid(format!(
                            "bernoulli-logit observation step {} must not have aux data",
                            step_idx
                        )));
                    }
                    Ok(())
                }
                SerializableObsFamily::PoissonLog => {
                    if aux.is_some() {
                        return Err(ArtifactError::invalid(format!(
                            "poisson-log observation step {} must not have aux data",
                            step_idx
                        )));
                    }
                    Ok(())
                }
                SerializableObsFamily::ExponentialLog => {
                    if aux.is_some() {
                        return Err(ArtifactError::invalid(format!(
                            "exponential-log observation step {} must not have aux data",
                            step_idx
                        )));
                    }
                    Ok(())
                }
                SerializableObsFamily::LogNormal => {
                    let sigma = aux.as_ref().ok_or_else(|| {
                        ArtifactError::invalid(format!(
                            "log-normal observation step {} is missing sigma",
                            step_idx
                        ))
                    })?;
                    check_ref(sigma)
                }
                SerializableObsFamily::NegativeBinomialLog => {
                    let alpha = aux.as_ref().ok_or_else(|| {
                        ArtifactError::invalid(format!(
                            "negative-binomial observation step {} is missing alpha",
                            step_idx
                        ))
                    })?;
                    check_ref(alpha)
                }
            }
        }
        ModelStep::HalfNormalLogP { x, sigma } => {
            check_ref(x)?;
            check_ref(sigma)
        }
        ModelStep::StudentTLogP { x, nu, mu, sigma } => {
            check_ref(x)?;
            check_ref(nu)?;
            check_ref(mu)?;
            check_ref(sigma)
        }
        ModelStep::UniformLogP { x, lower, upper } => {
            check_ref(x)?;
            check_ref(lower)?;
            check_ref(upper)
        }
        ModelStep::BernoulliLogP { x, p } => {
            check_ref(x)?;
            check_ref(p)
        }
        ModelStep::PoissonLogP { x, lam } => {
            check_ref(x)?;
            check_ref(lam)
        }
        ModelStep::GammaLogP { x, alpha, beta } => {
            check_ref(x)?;
            check_ref(alpha)?;
            check_ref(beta)
        }
        ModelStep::BetaLogP { x, alpha, beta } => {
            check_ref(x)?;
            check_ref(alpha)?;
            check_ref(beta)
        }
        ModelStep::FusedLinearMu {
            param_names,
            term_data,
            intercept,
        } => {
            if param_names.len() != term_data.len() {
                return Err(ArtifactError::invalid(format!(
                    "fused linear op at step {} has {} params but {} terms",
                    step_idx,
                    param_names.len(),
                    term_data.len()
                )));
            }
            for name in param_names {
                if !scalar_names.contains(name.as_str()) {
                    return Err(ArtifactError::missing_parameter(format!(
                        "unknown scalar parameter '{}'",
                        name
                    )));
                }
            }
            if let Some(r) = intercept {
                check_ref(r)?;
            }
            Ok(())
        }
        ModelStep::MatVecMul {
            param_name,
            matrix,
            n_rows,
            n_cols,
            intercept,
        } => {
            if matrix.len() != n_rows * n_cols {
                return Err(ArtifactError::invalid(format!(
                    "matrix payload at step {} has inconsistent shape {}x{} for {} values",
                    step_idx,
                    n_rows,
                    n_cols,
                    matrix.len()
                )));
            }
            if !vector_names.contains(param_name.as_str()) {
                return Err(ArtifactError::missing_parameter(format!(
                    "unknown vector parameter '{}'",
                    param_name
                )));
            }
            if let Some(expected) = vector_lengths.get(param_name.as_str()) {
                if *expected != *n_cols {
                    return Err(ArtifactError::invalid(format!(
                        "matvec step {} expects {} vector parameters but block '{}' has {}",
                        step_idx, n_cols, param_name, expected
                    )));
                }
            }
            if let Some(r) = intercept {
                check_ref(r)?;
            }
            Ok(())
        }
        ModelStep::VectorNormalLogP {
            param_name,
            n_params,
            ..
        }
        | ModelStep::VectorHalfNormalLogP {
            param_name,
            n_params,
            ..
        }
        | ModelStep::VectorStudentTLogP {
            param_name,
            n_params,
            ..
        }
        | ModelStep::VectorGammaLogP {
            param_name,
            n_params,
            ..
        }
        | ModelStep::VectorBetaLogP {
            param_name,
            n_params,
            ..
        }
        | ModelStep::VectorUniformLogP {
            param_name,
            n_params,
            ..
        } => {
            if !vector_names.contains(param_name.as_str()) {
                return Err(ArtifactError::missing_parameter(format!(
                    "unknown vector parameter '{}'",
                    param_name
                )));
            }
            if let Some(expected) = vector_lengths.get(param_name.as_str()) {
                if *expected != *n_params {
                    return Err(ArtifactError::invalid(format!(
                        "vector step {} expects {} parameters but block '{}' has {}",
                        step_idx, n_params, param_name, expected
                    )));
                }
            }
            Ok(())
        }
    }
}

fn group_parameter_blocks(
    graph: &Graph,
) -> Result<(Vec<ParameterBlock>, HashMap<usize, String>), ArtifactError> {
    let mut blocks = Vec::new();
    let mut index_to_block_name = HashMap::new();
    let mut i = 0usize;

    while i < graph.param_names.len() {
        let name = &graph.param_names[i];
        if let Some((base, len)) = parse_vector_block(graph, i) {
            let transform = SerializableParamTransform::from(&graph.param_transforms[i]);
            for offset in 0..len {
                index_to_block_name.insert(i + offset, base.clone());
            }
            blocks.push(ParameterBlock::Vector {
                name: base,
                len,
                transform,
            });
            i += len;
        } else {
            index_to_block_name.insert(i, name.clone());
            blocks.push(ParameterBlock::Scalar {
                name: name.clone(),
                transform: SerializableParamTransform::from(&graph.param_transforms[i]),
            });
            i += 1;
        }
    }

    Ok((blocks, index_to_block_name))
}

fn parse_vector_block(graph: &Graph, start_idx: usize) -> Option<(String, usize)> {
    let name = &graph.param_names[start_idx];
    let open = name.rfind('[')?;
    let close = name.rfind(']')?;
    if close != name.len() - 1 || close <= open + 1 {
        return None;
    }

    let base = name[..open].to_string();
    let mut len = 0usize;
    while start_idx + len < graph.param_names.len() {
        let candidate = &graph.param_names[start_idx + len];
        let expected = format!("{}[{}]", base, len);
        if candidate != &expected {
            break;
        }
        len += 1;
    }

    if len > 0 { Some((base, len)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SamplerConfig;

    fn simple_graph() -> Graph {
        let mut g = Graph::new();
        let beta = g.add_param("beta");
        let mu0 = g.add_constant(0.0);
        let sigma = g.add_constant(1.0);
        g.normal_logp(beta, mu0, sigma);

        let x = g.add_data("x", vec![1.0, 2.0, 3.0]);
        let mu = g.scalar_mul_data(beta, x);
        let obs = g.add_obs_data(vec![1.2, 2.1, 2.9]);
        g.normal_obs_logp(mu, sigma, obs);
        g
    }

    fn vector_graph() -> Graph {
        let mut g = Graph::new();
        let start = g.add_vector_params("beta", 2);
        g.vector_normal_logp(start, 2, 0.0, 1.0);

        let matrix_idx = g.store_matrix(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let mu = g.mat_vec_mul(matrix_idx, start, 2, None);
        let sigma = g.add_constant(1.0);
        let obs = g.add_obs_data(vec![0.5, -0.2]);
        g.normal_obs_logp(mu, sigma, obs);
        g
    }

    #[test]
    fn artifact_round_trip_serialization() {
        let graph = simple_graph();
        let runtime = CompiledModelRuntime::from_graph(&graph).unwrap();

        let json = runtime.to_json_pretty().unwrap();
        let restored = CompiledModelRuntime::from_json_str(&json).unwrap();
        let rebuilt = restored.to_graph().unwrap();

        assert_eq!(rebuilt.param_count, graph.param_count);
        assert_eq!(rebuilt.nodes.len(), graph.nodes.len());
        assert_eq!(rebuilt.obs_vectors.len(), graph.obs_vectors.len());
    }

    #[test]
    fn runtime_can_sample_rebuilt_graph() {
        let graph = simple_graph();
        let runtime = CompiledModelRuntime::from_graph(&graph).unwrap();

        let fit = runtime
            .sample(SamplerConfig {
                num_chains: 1,
                num_draws: 10,
                num_warmup: 10,
                show_progress: false,
                seed: 42,
                ..SamplerConfig::default()
            })
            .unwrap();

        assert_eq!(fit.param_names, graph.param_names);
        assert_eq!(fit.samples.len(), 1);
        assert_eq!(fit.samples[0].len(), 10);
    }

    #[test]
    fn vector_runtime_round_trip() {
        let graph = vector_graph();
        let artifact = CompiledModelArtifact::from_graph(&graph).unwrap();
        let rebuilt = CompiledModelRuntime::from_artifact(artifact)
            .unwrap()
            .to_graph()
            .unwrap();

        assert_eq!(rebuilt.param_count, graph.param_count);
        assert_eq!(rebuilt.param_names, graph.param_names);
        assert_eq!(rebuilt.nodes.len(), graph.nodes.len());
    }
}
