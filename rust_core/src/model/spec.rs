//! The declarative model definition: priors, likelihoods, expressions and
//! the errors every model operation reports.
use crate::graph::Graph;
use crate::param_ref::ParamRefError;
use std::collections::HashMap;

pub type Data1d = HashMap<String, Vec<f64>>;
pub type Data2d = HashMap<String, (Vec<f64>, usize, usize)>;
pub(super) type LinearTerms = Vec<(String, String)>;
pub type ModelResult<T> = Result<T, ModelError>;
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Invalid(String),
    Parameter(String),
}
impl ModelError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
    pub fn parameter(message: impl Into<String>) -> Self {
        Self::Parameter(message.into())
    }
}
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) | Self::Parameter(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for ModelError {}
pub(super) fn param_error(error: ParamRefError) -> ModelError {
    ModelError::parameter(error.to_string())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    #[serde(serialize_with = "serialize_sorted")]
    pub dimensions: HashMap<String, String>,
    pub potentials: Vec<(String, MuExpr)>,
    pub deterministics: Vec<(String, MuExpr)>,
    pub priors: Vec<PriorSpec>,
    pub likelihoods: Vec<LikelihoodSpec>,
    #[serde(skip)]
    pub bound_data_1d: HashMap<String, Vec<f64>>,
    #[serde(skip)]
    pub bound_data_2d: HashMap<String, (Vec<f64>, usize, usize)>,
}

/// Serialise a map in key order, so the same model always writes the same
/// artifact bytes.
fn serialize_sorted<S: serde::Serializer>(
    map: &HashMap<String, String>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(
        &map.iter().collect::<std::collections::BTreeMap<_, _>>(),
        serializer,
    )
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum DisplayParamSpec {
    Raw {
        name: String,
        raw_index: usize,
    },
    DerivedNonCenteredNormal {
        name: String,
        raw_index: usize,
        mu: HyperParam,
        sigma: HyperParam,
    },
}

#[derive(Debug, Clone)]
pub struct CompiledDefinition {
    pub graph: Graph,
    pub likelihood_names: Vec<String>,
    pub display_params: Vec<DisplayParamSpec>,
    pub auto_vector_params: HashMap<String, usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HyperParam {
    Const(f64),
    /// Name of a parameter whose value node (post-transform) is used as the hyperparameter.
    Param(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PriorSpec {
    Normal {
        name: String,
        mu: HyperParam,
        sigma: HyperParam,
    },
    HalfNormal {
        name: String,
        sigma: HyperParam,
    },
    Exponential {
        name: String,
        rate: HyperParam,
    },
    LogNormal {
        name: String,
        mu: HyperParam,
        sigma: HyperParam,
    },
    StudentT {
        name: String,
        nu: f64,
        mu: f64,
        sigma: f64,
    },
    Uniform {
        name: String,
        lower: f64,
        upper: f64,
    },
    Bernoulli {
        name: String,
        p: f64,
    },
    Poisson {
        name: String,
        lam: f64,
    },
    Gamma {
        name: String,
        alpha: f64,
        beta: f64,
    },
    Beta {
        name: String,
        alpha: f64,
        beta: f64,
    },
    VectorNormal {
        name: String,
        n: usize,
        mu: f64,
        sigma: f64,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SigmaSpec {
    Const(f64),
    Param(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LikelihoodFamily {
    Normal,
    BernoulliLogit,
    PoissonLog,
    ExponentialLog,
    LogNormal,
    NegativeBinomialLog,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LikelihoodSpec {
    pub family: LikelihoodFamily,
    pub name: String,
    pub mu_expr: MuExpr,
    pub sigma: Option<SigmaSpec>,
    pub observed_key: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MuExpr {
    Data(String),
    Gather {
        param_name: String,
        data_key: String,
    },
    Unary(crate::graph::ElementwiseOp, Box<MuExpr>),
    Binary(crate::graph::ElementwiseOp, Box<MuExpr>, Box<MuExpr>),
    Sum(Box<MuExpr>),
    Const(f64),
    ParamTimesData {
        param_name: String,
        data_key: String,
    },
    /// Element-wise sum of two vector expressions.
    Add(Box<MuExpr>, Box<MuExpr>),
    /// Bare parameter broadcast-added to a vector expression.
    Param(String),
    /// faer-backed matrix-vector multiply: matrix_data_key @ vector_param.
    MatVec {
        param_name: String,
        data_key: String,
    },
}

impl MuExpr {
    pub fn is_scalar(&self) -> bool {
        match self {
            MuExpr::Const(_) | MuExpr::Sum(_) => true,
            MuExpr::Data(_) | MuExpr::Gather { .. } => false,
            MuExpr::Unary(_, a) => a.is_scalar(),
            MuExpr::Binary(_, a, b) => a.is_scalar() && b.is_scalar(),
            MuExpr::Param(_) => true,
            MuExpr::ParamTimesData { .. } => false,
            MuExpr::MatVec { .. } => false,
            MuExpr::Add(a, b) => a.is_scalar() && b.is_scalar(),
        }
    }
}

pub fn collect_expr_param_names(expr: &MuExpr, out: &mut Vec<String>) {
    match expr {
        MuExpr::Const(_) | MuExpr::Data(_) => {}
        MuExpr::Unary(_, a) | MuExpr::Sum(a) => collect_expr_param_names(a, out),
        MuExpr::Param(name) => out.push(name.clone()),
        MuExpr::ParamTimesData { param_name, .. }
        | MuExpr::MatVec { param_name, .. }
        | MuExpr::Gather { param_name, .. } => out.push(param_name.clone()),
        MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
            collect_expr_param_names(a, out);
            collect_expr_param_names(b, out);
        }
    }
}

impl ModelSpec {
    pub fn structure_definition(&self) -> Self {
        let mut definition = self.clone();
        definition.bound_data_1d.clear();
        definition.bound_data_2d.clear();
        definition
    }
}
