use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal as NormalDist};
use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::{DataBinding as CoreDataBinding, DataInputs, MatrixBinding};
use rustmc_core::diagnostics::inv_normal_cdf;
use rustmc_core::distributions::{
    Bernoulli, BetaDist, Exponential, Gamma, HalfNormal, LogNormal, Normal, Poisson, StudentT,
    Uniform,
};
use rustmc_core::graph::{Graph, NodeId, ParamTransform};
use rustmc_core::param_ref::{validate_param_references, ParamRefError, ParamReference};
use rustmc_core::sampler::{self, SampleResult, SamplerConfig, SamplerType};
use rustmc_core::state_space::{
    ForecastResult as CoreForecastResult, KalmanFilterResult as CoreKalmanFilterResult,
    KalmanSmootherResult as CoreKalmanSmootherResult,
    LinearGaussianStateSpace as CoreLinearGaussianStateSpace,
    StateSpaceError as CoreStateSpaceError,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

type Data1d = HashMap<String, Vec<f64>>;
type Data2d = HashMap<String, (Vec<f64>, usize, usize)>;
type LinearTerms = Vec<(String, String)>;
type PyIntervalArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

create_exception!(
    rustmc,
    ParameterError,
    PyValueError,
    "Raised when a parameter reference in a model cannot be resolved.\n\n\
     Subclasses ``ValueError`` for backwards compatibility."
);

create_exception!(
    rustmc,
    StateSpaceError,
    PyValueError,
    "Raised when state-space inputs or numerical updates are invalid."
);

/// Convert a core parameter-resolution failure into the Python exception.
fn param_error(err: ParamRefError) -> PyErr {
    ParameterError::new_err(err.to_string())
}

/// Monotonic id handed to each `ModelBuilder` so that a `ParamRef` produced by
/// one model can never be silently consumed by another.
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_model_id() -> u64 {
    NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed)
}

/// Error for a `ParamRef`/`Expr` that belongs to a different `ModelBuilder`.
fn foreign_param_error(name: &str, context: &str) -> PyErr {
    ParameterError::new_err(format!(
        "parameter '{}' used in {} belongs to a different model. \
         A ParamRef returned by one ModelBuilder cannot be used in another.",
        name, context
    ))
}

#[pyclass]
#[derive(Debug, Clone)]
struct ModelSpec {
    priors: Vec<PriorSpec>,
    likelihoods: Vec<LikelihoodSpec>,
    bound_data_1d: HashMap<String, Vec<f64>>,
    bound_data_2d: HashMap<String, (Vec<f64>, usize, usize)>,
}

#[derive(Debug, Clone)]
enum DisplayParamSpec {
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
struct CompiledPythonModel {
    graph: Graph,
    likelihood_names: Vec<String>,
    display_params: Vec<DisplayParamSpec>,
    auto_vector_params: HashMap<String, usize>,
}

#[pyclass(name = "BoundModel")]
#[derive(Clone)]
struct PyBoundModel {
    structure: Arc<Graph>,
    binding: CoreDataBinding,
}

#[pyclass(name = "CompiledModel")]
#[derive(Clone)]
struct PyCompiledModel {
    structure: Arc<Graph>,
    likelihood_names: Vec<String>,
    display_params: Vec<DisplayParamSpec>,
    default_data_1d: Data1d,
    default_data_2d: Data2d,
}

fn core_binding_from_maps(
    schema: &rustmc_core::DataSchema,
    data_1d: &Data1d,
    data_2d: &Data2d,
    id: String,
    strict: bool,
    check_finite: bool,
) -> PyResult<CoreDataBinding> {
    let inputs = data_inputs_from_maps(data_1d, data_2d);
    CoreDataBinding::bind(schema, inputs, id, strict, check_finite)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn validate_core_binding(graph: &Graph, binding: CoreDataBinding) -> PyResult<CoreDataBinding> {
    binding
        .validate_for(graph)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(binding)
}

fn data_inputs_from_maps(data_1d: &Data1d, data_2d: &Data2d) -> DataInputs {
    DataInputs {
        vectors: data_1d
            .iter()
            .map(|(key, values)| (key.clone(), Arc::<[f64]>::from(values.clone())))
            .collect(),
        matrices: data_2d
            .iter()
            .map(|(key, (values, n_rows, n_cols))| {
                (
                    key.clone(),
                    MatrixBinding {
                        data: Arc::from(values.clone()),
                        n_rows: *n_rows,
                        n_cols: *n_cols,
                    },
                )
            })
            .collect(),
    }
}

impl PyCompiledModel {
    fn bind_any(
        &self,
        value: &Bound<'_, PyAny>,
        id: String,
        shared: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<CoreDataBinding> {
        if let Ok(bound) = value.downcast::<PyBoundModel>() {
            let bound = bound.borrow();
            if !Arc::ptr_eq(&bound.structure, &self.structure) {
                return Err(PyValueError::new_err(
                    "BoundModel belongs to a different CompiledModel",
                ));
            }
            let mut binding = bound.binding.clone();
            binding.set_id(id);
            return validate_core_binding(&self.structure, binding);
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("data must be a dict or BoundModel from this compiled model")
        })?;
        let mut one_d = self.default_data_1d.clone();
        let mut two_d = self.default_data_2d.clone();
        let mut shared_keys = std::collections::HashSet::new();
        if let Some(shared) = shared {
            let (shared_1d, shared_2d) = parse_data_dict(shared)?;
            shared_keys.extend(shared_1d.keys().cloned());
            shared_keys.extend(shared_2d.keys().cloned());
            merge_data_overrides(&mut one_d, &mut two_d, shared_1d, shared_2d);
        }
        let (extra_1d, extra_2d) = parse_data_dict(dict)?;
        for key in extra_1d.keys().chain(extra_2d.keys()) {
            if shared_keys.contains(key) {
                return Err(PyValueError::new_err(format!(
                    "data key '{}' appears in both shared and per-dataset inputs",
                    key
                )));
            }
        }
        merge_data_overrides(&mut one_d, &mut two_d, extra_1d, extra_2d);
        validate_core_binding(
            &self.structure,
            core_binding_from_maps(&self.structure.schema, &one_d, &two_d, id, true, true)?,
        )
    }
}

#[pymethods]
impl PyBoundModel {
    #[getter]
    fn id(&self) -> &str {
        self.binding.id()
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.binding.n_obs()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }
}

fn template_data_for_spec(spec: &ModelSpec) -> PyResult<(Data1d, Data2d)> {
    let mut one_d = spec.bound_data_1d.clone();
    let mut two_d = spec.bound_data_2d.clone();
    let vector_sizes: HashMap<&str, usize> = spec
        .priors
        .iter()
        .filter_map(|prior| match prior {
            PriorSpec::VectorNormal { name, n, .. } => Some((name.as_str(), *n)),
            _ => None,
        })
        .collect();
    fn visit(
        expr: &MuExpr,
        one_d: &mut Data1d,
        two_d: &mut Data2d,
        vector_sizes: &HashMap<&str, usize>,
    ) -> PyResult<()> {
        match expr {
            MuExpr::ParamTimesData { data_key, .. } => {
                one_d.entry(data_key.clone()).or_insert_with(|| vec![1.0]);
            }
            MuExpr::MatVec {
                param_name,
                data_key,
            } => {
                if !two_d.contains_key(data_key) {
                    let n_cols = vector_sizes.get(param_name.as_str()).copied().ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "cannot compile matrix '{}' without bound data: declare '{}' with vector_normal_prior so its column count is structural",
                            data_key, param_name
                        ))
                    })?;
                    two_d.insert(data_key.clone(), (vec![1.0; n_cols], 1, n_cols));
                }
            }
            MuExpr::Add(a, b) => {
                visit(a, one_d, two_d, vector_sizes)?;
                visit(b, one_d, two_d, vector_sizes)?;
            }
            MuExpr::Const(_) | MuExpr::Param(_) => {}
        }
        Ok(())
    }
    for likelihood in &spec.likelihoods {
        visit(&likelihood.mu_expr, &mut one_d, &mut two_d, &vector_sizes)?;
        one_d
            .entry(likelihood.observed_key.clone())
            .or_insert_with(|| vec![1.0]);
    }
    Ok((one_d, two_d))
}

/// A hyperparameter value: either a scalar constant or a reference to another
/// already-declared parameter (for hierarchical / multilevel models).
#[derive(Debug, Clone)]
enum HyperParam {
    Const(f64),
    /// Name of a parameter whose value node (post-transform) is used as the hyperparameter.
    Param(String),
}

#[derive(Debug, Clone)]
enum PriorSpec {
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

#[derive(Debug, Clone)]
enum SigmaSpec {
    Const(f64),
    Param(String),
}

#[derive(Debug, Clone)]
enum LikelihoodFamily {
    Normal,
    BernoulliLogit,
    PoissonLog,
    ExponentialLog,
    LogNormal,
    NegativeBinomialLog,
}

#[derive(Debug, Clone)]
struct LikelihoodSpec {
    family: LikelihoodFamily,
    name: String,
    mu_expr: MuExpr,
    sigma: Option<SigmaSpec>,
    observed_key: String,
}

/// Recursive expression tree built on the Python side, compiled to graph
/// nodes at sampling time.
#[derive(Debug, Clone)]
enum MuExpr {
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
    fn is_scalar(&self) -> bool {
        match self {
            MuExpr::Const(_) => true,
            MuExpr::Param(_) => true,
            MuExpr::ParamTimesData { .. } => false,
            MuExpr::MatVec { .. } => false,
            MuExpr::Add(a, b) => a.is_scalar() && b.is_scalar(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct VectorParamRef {
    name: String,
    _n: usize,
    /// Id of the `ModelBuilder` that created this reference.
    owner: u64,
}

#[pymethods]
impl VectorParamRef {
    fn __matmul__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::MatVec {
                param_name: self.name.clone(),
                data_key: data_key.to_string(),
            },
            owner: Some(self.owner),
        }
    }
}

/// Combine the owning-model ids of two sub-expressions, rejecting mixtures.
fn merge_owners(a: Option<u64>, b: Option<u64>, a_name: &str) -> PyResult<Option<u64>> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => Err(ParameterError::new_err(format!(
            "expression mixes parameters from two different models \
             (offending parameter: '{}'). Build the whole linear predictor \
             from a single ModelBuilder.",
            a_name
        ))),
        (Some(x), _) => Ok(Some(x)),
        (None, other) => Ok(other),
    }
}

/// First parameter name appearing in an expression, for error messages.
fn first_param_name(expr: &MuExpr) -> String {
    match expr {
        MuExpr::Const(_) => "<constant>".to_string(),
        MuExpr::Param(name) => name.clone(),
        MuExpr::ParamTimesData { param_name, .. } | MuExpr::MatVec { param_name, .. } => {
            param_name.clone()
        }
        MuExpr::Add(a, b) => {
            let left = first_param_name(a);
            if left == "<constant>" {
                first_param_name(b)
            } else {
                left
            }
        }
    }
}

/// Collect every parameter name referenced by an expression tree.
fn collect_expr_param_names(expr: &MuExpr, out: &mut Vec<String>) {
    match expr {
        MuExpr::Const(_) => {}
        MuExpr::Param(name) => out.push(name.clone()),
        MuExpr::ParamTimesData { param_name, .. } | MuExpr::MatVec { param_name, .. } => {
            out.push(param_name.clone())
        }
        MuExpr::Add(a, b) => {
            collect_expr_param_names(a, out);
            collect_expr_param_names(b, out);
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct ModelBuilder {
    id: u64,
    priors: Vec<PriorSpec>,
    likelihoods: Vec<LikelihoodSpec>,
    param_names: Vec<String>,
    bound_data_1d: HashMap<String, Vec<f64>>,
    bound_data_2d: HashMap<String, (Vec<f64>, usize, usize)>,
}

#[pyclass]
#[derive(Debug, Clone)]
struct ParamRef {
    name: String,
    /// Id of the `ModelBuilder` that created this reference.
    owner: u64,
}

#[pyclass]
#[derive(Debug, Clone)]
struct Expr {
    inner: MuExpr,
    /// Id of the `ModelBuilder` whose parameters this expression uses, if any.
    /// `None` for constant-only expressions.
    owner: Option<u64>,
}

#[pymethods]
impl ParamRef {
    fn __mul__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::ParamTimesData {
                param_name: self.name.clone(),
                data_key: data_key.to_string(),
            },
            owner: Some(self.owner),
        }
    }

    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        if let Ok(other_expr) = other.downcast::<Expr>() {
            let (rhs, rhs_owner) = {
                let b = other_expr.borrow();
                (b.inner.clone(), b.owner)
            };
            let owner = merge_owners(Some(self.owner), rhs_owner, &self.name)?;
            Ok(Expr {
                inner: MuExpr::Add(Box::new(MuExpr::Param(self.name.clone())), Box::new(rhs)),
                owner,
            })
        } else if let Ok(other_param) = other.downcast::<ParamRef>() {
            let (rhs_name, rhs_owner) = {
                let b = other_param.borrow();
                (b.name.clone(), b.owner)
            };
            let owner = merge_owners(Some(self.owner), Some(rhs_owner), &self.name)?;
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(MuExpr::Param(self.name.clone())),
                    Box::new(MuExpr::Param(rhs_name)),
                ),
                owner,
            })
        } else if let Ok(value) = other.extract::<f64>() {
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(MuExpr::Param(self.name.clone())),
                    Box::new(MuExpr::Const(value)),
                ),
                owner: Some(self.owner),
            })
        } else {
            Err(PyValueError::new_err(
                "unsupported operand type for + with ParamRef",
            ))
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        self.__add__(other)
    }

    fn __matmul__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::MatVec {
                param_name: self.name.clone(),
                data_key: data_key.to_string(),
            },
            owner: Some(self.owner),
        }
    }
}

#[pymethods]
impl Expr {
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        if let Ok(other_expr) = other.downcast::<Expr>() {
            let (rhs, rhs_owner) = {
                let b = other_expr.borrow();
                (b.inner.clone(), b.owner)
            };
            let owner = merge_owners(self.owner, rhs_owner, &first_param_name(&self.inner))?;
            Ok(Expr {
                inner: MuExpr::Add(Box::new(self.inner.clone()), Box::new(rhs)),
                owner,
            })
        } else if let Ok(other_param) = other.downcast::<ParamRef>() {
            let (rhs_name, rhs_owner) = {
                let b = other_param.borrow();
                (b.name.clone(), b.owner)
            };
            let owner = merge_owners(self.owner, Some(rhs_owner), &rhs_name)?;
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(self.inner.clone()),
                    Box::new(MuExpr::Param(rhs_name)),
                ),
                owner,
            })
        } else if let Ok(value) = other.extract::<f64>() {
            Ok(Expr {
                inner: MuExpr::Add(Box::new(self.inner.clone()), Box::new(MuExpr::Const(value))),
                owner: self.owner,
            })
        } else {
            Err(PyValueError::new_err(
                "unsupported operand type for + with Expr",
            ))
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        if let Ok(other_param) = other.downcast::<ParamRef>() {
            let (lhs_name, lhs_owner) = {
                let b = other_param.borrow();
                (b.name.clone(), b.owner)
            };
            let owner = merge_owners(Some(lhs_owner), self.owner, &lhs_name)?;
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(MuExpr::Param(lhs_name)),
                    Box::new(self.inner.clone()),
                ),
                owner,
            })
        } else if let Ok(value) = other.extract::<f64>() {
            Ok(Expr {
                inner: MuExpr::Add(Box::new(MuExpr::Const(value)), Box::new(self.inner.clone())),
                owner: self.owner,
            })
        } else {
            self.__add__(other)
        }
    }
}

/// Name a `PriorSpec` declares.
fn prior_name(prior: &PriorSpec) -> &str {
    match prior {
        PriorSpec::Normal { name, .. }
        | PriorSpec::HalfNormal { name, .. }
        | PriorSpec::Exponential { name, .. }
        | PriorSpec::LogNormal { name, .. }
        | PriorSpec::StudentT { name, .. }
        | PriorSpec::Uniform { name, .. }
        | PriorSpec::Bernoulli { name, .. }
        | PriorSpec::Poisson { name, .. }
        | PriorSpec::Gamma { name, .. }
        | PriorSpec::Beta { name, .. }
        | PriorSpec::VectorNormal { name, .. } => name,
    }
}

/// Hyperparameter references a `PriorSpec` makes, as `(role, name)` pairs.
fn prior_hyper_refs(prior: &PriorSpec) -> Vec<(&'static str, &str)> {
    let mut out = Vec::new();
    fn push<'a>(out: &mut Vec<(&'static str, &'a str)>, role: &'static str, hp: &'a HyperParam) {
        if let HyperParam::Param(name) = hp {
            out.push((role, name.as_str()));
        }
    }
    match prior {
        PriorSpec::Normal { mu, sigma, .. } | PriorSpec::LogNormal { mu, sigma, .. } => {
            push(&mut out, "mu", mu);
            push(&mut out, "sigma", sigma);
        }
        PriorSpec::HalfNormal { sigma, .. } => push(&mut out, "sigma", sigma),
        PriorSpec::Exponential { rate, .. } => push(&mut out, "rate", rate),
        PriorSpec::StudentT { .. }
        | PriorSpec::Uniform { .. }
        | PriorSpec::Bernoulli { .. }
        | PriorSpec::Poisson { .. }
        | PriorSpec::Gamma { .. }
        | PriorSpec::Beta { .. }
        | PriorSpec::VectorNormal { .. } => {}
    }
    out
}

/// The ordered list of parameter names a model declares, plus the full set of
/// references into it. This is the single source of truth for reference
/// validation, shared by `ModelBuilder.build()` and `compile_python_model`.
fn model_reference_set(
    priors: &[PriorSpec],
    likelihoods: &[LikelihoodSpec],
) -> (Vec<String>, Vec<ParamReference>) {
    let declared: Vec<String> = priors.iter().map(|p| prior_name(p).to_string()).collect();
    let mut refs = Vec::new();

    for (idx, prior) in priors.iter().enumerate() {
        for (role, name) in prior_hyper_refs(prior) {
            refs.push(ParamReference::ordered(
                name,
                format!("prior '{}' hyperparameter {}", prior_name(prior), role),
                idx,
            ));
        }
    }

    for lik in likelihoods {
        let mut names = Vec::new();
        collect_expr_param_names(&lik.mu_expr, &mut names);
        for name in names {
            refs.push(ParamReference::unordered(
                name,
                format!("the linear predictor of likelihood '{}'", lik.name),
            ));
        }
        if let Some(SigmaSpec::Param(name)) = &lik.sigma {
            refs.push(ParamReference::unordered(
                name.clone(),
                format!("the scale parameter of likelihood '{}'", lik.name),
            ));
        }
    }

    (declared, refs)
}

/// Validate every parameter reference in a model up front, before any graph is
/// built. Fails loudly on unknown names, out-of-order hyperparameters and
/// duplicate declarations.
fn validate_model_references(priors: &[PriorSpec], likelihoods: &[LikelihoodSpec]) -> PyResult<()> {
    let (declared, refs) = model_reference_set(priors, likelihoods);
    validate_param_references(&declared, &refs).map_err(param_error)
}

/// HMC and NUTS evolve a continuous Euclidean state.  Discrete latent
/// parameters therefore need marginalisation or a discrete transition kernel;
/// treating them as continuous values produces invalid posterior draws.
/// Keep these priors available for prior-predictive simulation, but reject
/// every posterior-sampling entry point until such a kernel exists.
fn reject_discrete_priors_for_gradient_sampling(priors: &[PriorSpec]) -> PyResult<()> {
    let discrete: Vec<&str> = priors
        .iter()
        .filter_map(|prior| match prior {
            PriorSpec::Bernoulli { name, .. } | PriorSpec::Poisson { name, .. } => {
                Some(name.as_str())
            }
            _ => None,
        })
        .collect();
    if discrete.is_empty() {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "Discrete prior parameter(s) [{}] cannot be sampled with HMC/NUTS. \
         Bernoulli and Poisson priors are currently supported only by \
         sample_prior_predictive(); posterior inference requires continuous \
         parameters or explicit marginalisation.",
        discrete.join(", ")
    )))
}

impl ModelBuilder {
    /// A reference to one of this model's parameters, tagged with the model id.
    fn param_ref(&self, name: &str) -> ParamRef {
        ParamRef {
            name: name.to_string(),
            owner: self.id,
        }
    }

    /// Names declared so far, in declaration order.
    fn declared_names(&self) -> Vec<String> {
        self.priors
            .iter()
            .map(|p| prior_name(p).to_string())
            .collect()
    }

    /// Parse a hyperparameter argument, rejecting references that belong to a
    /// different model or that are not yet declared in this one.
    fn hyper_arg(
        &self,
        obj: &Bound<'_, PyAny>,
        arg_name: &str,
        new_prior_name: &str,
    ) -> PyResult<HyperParam> {
        let hp = extract_hyper(obj, arg_name)?;
        if let HyperParam::Const(value) = &hp {
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{} must be finite",
                    arg_name
                )));
            }
            if matches!(arg_name, "sigma" | "rate") && *value <= 0.0 {
                return Err(PyValueError::new_err(format!("{} must be > 0", arg_name)));
            }
        }
        if let HyperParam::Param(ref name) = hp {
            let context = format!("prior '{}' hyperparameter {}", new_prior_name, arg_name);
            if let Ok(p) = obj.downcast::<ParamRef>() {
                if p.borrow().owner != self.id {
                    return Err(foreign_param_error(name, &context));
                }
            }
            let declared = self.declared_names();
            let position = declared.len();
            validate_param_references(
                &declared,
                &[ParamReference::ordered(name.clone(), context, position)],
            )
            .map_err(param_error)?;
        }
        Ok(hp)
    }

    /// Parse a likelihood predictor argument (`Expr` or bare `ParamRef`),
    /// rejecting references that belong to a different model.
    fn likelihood_expr(
        &self,
        value: &Bound<'_, PyAny>,
        arg_name: &str,
        lik_name: &str,
    ) -> PyResult<MuExpr> {
        let (expr, owner) = if let Ok(e) = value.downcast::<Expr>() {
            let b = e.borrow();
            (b.inner.clone(), b.owner)
        } else if let Ok(p) = value.downcast::<ParamRef>() {
            let b = p.borrow();
            (MuExpr::Param(b.name.clone()), Some(b.owner))
        } else {
            return Err(PyValueError::new_err(format!(
                "{} must be an Expr (e.g. beta * 'x') or a ParamRef",
                arg_name
            )));
        };
        let context = format!("the linear predictor of likelihood '{}'", lik_name);
        self.check_owner(owner, &first_param_name(&expr), &context)?;
        Ok(expr)
    }

    /// Parse a likelihood scale argument (float or `ParamRef`), rejecting
    /// references that belong to a different model.
    fn scale_spec(
        &self,
        value: &Bound<'_, PyAny>,
        arg_name: &str,
        lik_name: &str,
    ) -> PyResult<SigmaSpec> {
        if let Ok(v) = value.extract::<f64>() {
            validate_positive_finite(arg_name, v)?;
            Ok(SigmaSpec::Const(v))
        } else if let Ok(p) = value.downcast::<ParamRef>() {
            let (name, owner) = {
                let b = p.borrow();
                (b.name.clone(), b.owner)
            };
            let context = format!("the {} of likelihood '{}'", arg_name, lik_name);
            self.check_owner(Some(owner), &name, &context)?;
            Ok(SigmaSpec::Param(name))
        } else {
            Err(PyValueError::new_err(format!(
                "{} must be a float or a ParamRef (e.g. from half_normal_prior)",
                arg_name
            )))
        }
    }

    /// Reject a `ParamRef`/`Expr` produced by a different `ModelBuilder`.
    fn check_owner(&self, owner: Option<u64>, name: &str, context: &str) -> PyResult<()> {
        match owner {
            Some(id) if id != self.id => Err(foreign_param_error(name, context)),
            _ => Ok(()),
        }
    }
}

#[pymethods]
impl ModelBuilder {
    #[new]
    #[pyo3(signature = (data=None))]
    fn new(data: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let (bound_data_1d, bound_data_2d) = match data {
            Some(d) => parse_data_dict(d)?,
            None => (HashMap::new(), HashMap::new()),
        };
        Ok(Self {
            id: next_model_id(),
            priors: Vec::new(),
            likelihoods: Vec::new(),
            param_names: Vec::new(),
            bound_data_1d,
            bound_data_2d,
        })
    }

    /// Support scoped model construction without changing builder semantics.
    /// No ambient builder is installed and exceptions are never suppressed.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (name, mu, sigma))]
    fn normal_prior(
        &mut self,
        name: &str,
        mu: &Bound<'_, PyAny>,
        sigma: &Bound<'_, PyAny>,
    ) -> PyResult<ParamRef> {
        let mu_hp = self.hyper_arg(mu, "mu", name)?;
        let sigma_hp = self.hyper_arg(sigma, "sigma", name)?;
        self.priors.push(PriorSpec::Normal {
            name: name.to_string(),
            mu: mu_hp,
            sigma: sigma_hp,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, sigma))]
    fn half_normal_prior(&mut self, name: &str, sigma: &Bound<'_, PyAny>) -> PyResult<ParamRef> {
        let sigma_hp = self.hyper_arg(sigma, "sigma", name)?;
        self.priors.push(PriorSpec::HalfNormal {
            name: name.to_string(),
            sigma: sigma_hp,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, rate))]
    fn exponential_prior(&mut self, name: &str, rate: &Bound<'_, PyAny>) -> PyResult<ParamRef> {
        let rate_hp = self.hyper_arg(rate, "rate", name)?;
        self.priors.push(PriorSpec::Exponential {
            name: name.to_string(),
            rate: rate_hp,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, mu, sigma))]
    fn log_normal_prior(
        &mut self,
        name: &str,
        mu: &Bound<'_, PyAny>,
        sigma: &Bound<'_, PyAny>,
    ) -> PyResult<ParamRef> {
        let mu_hp = self.hyper_arg(mu, "mu", name)?;
        let sigma_hp = self.hyper_arg(sigma, "sigma", name)?;
        self.priors.push(PriorSpec::LogNormal {
            name: name.to_string(),
            mu: mu_hp,
            sigma: sigma_hp,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, nu, mu=0.0, sigma=1.0))]
    fn student_t_prior(&mut self, name: &str, nu: f64, mu: f64, sigma: f64) -> PyResult<ParamRef> {
        validate_positive_finite("nu", nu)?;
        validate_finite("mu", mu)?;
        validate_positive_finite("sigma", sigma)?;
        self.priors.push(PriorSpec::StudentT {
            name: name.to_string(),
            nu,
            mu,
            sigma,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, lower=0.0, upper=1.0))]
    fn uniform_prior(&mut self, name: &str, lower: f64, upper: f64) -> PyResult<ParamRef> {
        validate_finite("lower", lower)?;
        validate_finite("upper", upper)?;
        if lower >= upper {
            return Err(PyValueError::new_err("lower must be less than upper"));
        }
        self.priors.push(PriorSpec::Uniform {
            name: name.to_string(),
            lower,
            upper,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, p=0.5))]
    fn bernoulli_prior(&mut self, name: &str, p: f64) -> PyResult<ParamRef> {
        validate_finite("p", p)?;
        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err("p must be between 0 and 1"));
        }
        self.priors.push(PriorSpec::Bernoulli {
            name: name.to_string(),
            p,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, lam))]
    fn poisson_prior(&mut self, name: &str, lam: f64) -> PyResult<ParamRef> {
        validate_positive_finite("lam", lam)?;
        self.priors.push(PriorSpec::Poisson {
            name: name.to_string(),
            lam,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, alpha, beta))]
    fn gamma_prior(&mut self, name: &str, alpha: f64, beta: f64) -> PyResult<ParamRef> {
        validate_positive_finite("alpha", alpha)?;
        validate_positive_finite("beta", beta)?;
        self.priors.push(PriorSpec::Gamma {
            name: name.to_string(),
            alpha,
            beta,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, alpha, beta))]
    fn beta_prior(&mut self, name: &str, alpha: f64, beta: f64) -> PyResult<ParamRef> {
        validate_positive_finite("alpha", alpha)?;
        validate_positive_finite("beta", beta)?;
        self.priors.push(PriorSpec::Beta {
            name: name.to_string(),
            alpha,
            beta,
        });
        self.param_names.push(name.to_string());
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, n, mu=0.0, sigma=1.0))]
    fn vector_normal_prior(
        &mut self,
        name: &str,
        n: usize,
        mu: f64,
        sigma: f64,
    ) -> PyResult<VectorParamRef> {
        if n == 0 {
            return Err(PyValueError::new_err("n must be >= 1"));
        }
        validate_finite("mu", mu)?;
        validate_positive_finite("sigma", sigma)?;
        self.priors.push(PriorSpec::VectorNormal {
            name: name.to_string(),
            n,
            mu,
            sigma,
        });
        Ok(VectorParamRef {
            name: name.to_string(),
            _n: n,
            owner: self.id,
        })
    }

    #[pyo3(signature = (name, mu_expr, sigma, observed_key))]
    fn normal_likelihood(
        &mut self,
        name: &str,
        mu_expr: &Bound<'_, PyAny>,
        sigma: &Bound<'_, PyAny>,
        observed_key: &str,
    ) -> PyResult<()> {
        let inner_expr = self.likelihood_expr(mu_expr, "mu_expr", name)?;
        let sigma_spec = self.scale_spec(sigma, "sigma", name)?;
        if !self.bound_data_1d.is_empty() || !self.bound_data_2d.is_empty() {
            validate_data_keys(
                &inner_expr,
                observed_key,
                &self.bound_data_1d,
                &self.bound_data_2d,
            )?;
        }
        self.likelihoods.push(LikelihoodSpec {
            family: LikelihoodFamily::Normal,
            name: name.to_string(),
            mu_expr: inner_expr,
            sigma: Some(sigma_spec),
            observed_key: observed_key.to_string(),
        });
        Ok(())
    }

    #[pyo3(signature = (name, eta_expr, observed_key))]
    fn bernoulli_logit_likelihood(
        &mut self,
        name: &str,
        eta_expr: &Bound<'_, PyAny>,
        observed_key: &str,
    ) -> PyResult<()> {
        let inner_expr = self.likelihood_expr(eta_expr, "eta_expr", name)?;
        if !self.bound_data_1d.is_empty() || !self.bound_data_2d.is_empty() {
            validate_data_keys(
                &inner_expr,
                observed_key,
                &self.bound_data_1d,
                &self.bound_data_2d,
            )?;
        }
        self.likelihoods.push(LikelihoodSpec {
            family: LikelihoodFamily::BernoulliLogit,
            name: name.to_string(),
            mu_expr: inner_expr,
            sigma: None,
            observed_key: observed_key.to_string(),
        });
        Ok(())
    }

    #[pyo3(signature = (name, eta_expr, observed_key))]
    fn poisson_log_likelihood(
        &mut self,
        name: &str,
        eta_expr: &Bound<'_, PyAny>,
        observed_key: &str,
    ) -> PyResult<()> {
        let inner_expr = self.likelihood_expr(eta_expr, "eta_expr", name)?;
        if !self.bound_data_1d.is_empty() || !self.bound_data_2d.is_empty() {
            validate_data_keys(
                &inner_expr,
                observed_key,
                &self.bound_data_1d,
                &self.bound_data_2d,
            )?;
        }
        self.likelihoods.push(LikelihoodSpec {
            family: LikelihoodFamily::PoissonLog,
            name: name.to_string(),
            mu_expr: inner_expr,
            sigma: None,
            observed_key: observed_key.to_string(),
        });
        Ok(())
    }

    #[pyo3(signature = (name, eta_expr, observed_key))]
    fn exponential_likelihood(
        &mut self,
        name: &str,
        eta_expr: &Bound<'_, PyAny>,
        observed_key: &str,
    ) -> PyResult<()> {
        let inner_expr = self.likelihood_expr(eta_expr, "eta_expr", name)?;
        if !self.bound_data_1d.is_empty() || !self.bound_data_2d.is_empty() {
            validate_data_keys(
                &inner_expr,
                observed_key,
                &self.bound_data_1d,
                &self.bound_data_2d,
            )?;
        }
        self.likelihoods.push(LikelihoodSpec {
            family: LikelihoodFamily::ExponentialLog,
            name: name.to_string(),
            mu_expr: inner_expr,
            sigma: None,
            observed_key: observed_key.to_string(),
        });
        Ok(())
    }

    #[pyo3(signature = (name, mu_expr, sigma, observed_key))]
    fn log_normal_likelihood(
        &mut self,
        name: &str,
        mu_expr: &Bound<'_, PyAny>,
        sigma: &Bound<'_, PyAny>,
        observed_key: &str,
    ) -> PyResult<()> {
        let inner_expr = self.likelihood_expr(mu_expr, "mu_expr", name)?;
        let sigma_spec = self.scale_spec(sigma, "sigma", name)?;
        if !self.bound_data_1d.is_empty() || !self.bound_data_2d.is_empty() {
            validate_data_keys(
                &inner_expr,
                observed_key,
                &self.bound_data_1d,
                &self.bound_data_2d,
            )?;
        }
        self.likelihoods.push(LikelihoodSpec {
            family: LikelihoodFamily::LogNormal,
            name: name.to_string(),
            mu_expr: inner_expr,
            sigma: Some(sigma_spec),
            observed_key: observed_key.to_string(),
        });
        Ok(())
    }

    #[pyo3(signature = (name, eta_expr, alpha, observed_key))]
    fn negative_binomial_likelihood(
        &mut self,
        name: &str,
        eta_expr: &Bound<'_, PyAny>,
        alpha: &Bound<'_, PyAny>,
        observed_key: &str,
    ) -> PyResult<()> {
        let inner_expr = self.likelihood_expr(eta_expr, "eta_expr", name)?;
        let alpha_spec = self.scale_spec(alpha, "alpha", name)?;
        if !self.bound_data_1d.is_empty() || !self.bound_data_2d.is_empty() {
            validate_data_keys(
                &inner_expr,
                observed_key,
                &self.bound_data_1d,
                &self.bound_data_2d,
            )?;
        }
        self.likelihoods.push(LikelihoodSpec {
            family: LikelihoodFamily::NegativeBinomialLog,
            name: name.to_string(),
            mu_expr: inner_expr,
            sigma: Some(alpha_spec),
            observed_key: observed_key.to_string(),
        });
        Ok(())
    }

    /// Finalise the model. Validates every parameter reference up front so an
    /// unresolvable name fails here rather than mid-sample.
    fn build(&self) -> PyResult<ModelSpec> {
        validate_model_references(&self.priors, &self.likelihoods)?;
        Ok(ModelSpec {
            priors: self.priors.clone(),
            likelihoods: self.likelihoods.clone(),
            bound_data_1d: self.bound_data_1d.clone(),
            bound_data_2d: self.bound_data_2d.clone(),
        })
    }

    /// Compile immutable model structure once. Dataset payloads supplied here
    /// are used only to establish structural matrix widths and as bind defaults.
    fn compile(&self) -> PyResult<PyCompiledModel> {
        let spec = self.build()?;
        reject_discrete_priors_for_gradient_sampling(&spec.priors)?;
        let (template_1d, template_2d) = template_data_for_spec(&spec)?;
        validate_bound_vector_lengths(&template_1d, &template_2d)?;
        let compiled = compile_python_model(&spec, &template_1d, &template_2d)?;
        Ok(PyCompiledModel {
            structure: Arc::new(compiled.graph.structure_only()),
            likelihood_names: compiled.likelihood_names,
            display_params: compiled.display_params,
            default_data_1d: self.bound_data_1d.clone(),
            default_data_2d: self.bound_data_2d.clone(),
        })
    }
}

/// Extract numpy arrays from a Python dict into typed Rust maps.
fn parse_data_dict(data: &Bound<'_, PyDict>) -> PyResult<(Data1d, Data2d)> {
    let mut data_1d = HashMap::new();
    let mut data_2d = HashMap::new();
    for (key, value) in data.iter() {
        let key_str: String = key.extract()?;
        if let Ok(arr) = value.downcast::<PyArray2<f64>>() {
            let shape = arr.shape().to_vec();
            let slice = unsafe { arr.as_slice()? };
            ensure_finite_data(&key_str, slice)?;
            data_2d.insert(key_str, (slice.to_vec(), shape[0], shape[1]));
        } else {
            let arr: &Bound<'_, PyArray1<f64>> = value.downcast()?;
            let vec: Vec<f64> = unsafe { arr.as_slice()?.to_vec() };
            ensure_finite_data(&key_str, &vec)?;
            data_1d.insert(key_str, vec);
        }
    }
    Ok((data_1d, data_2d))
}

fn ensure_finite_data(key: &str, values: &[f64]) -> PyResult<()> {
    if values.is_empty() {
        return Err(PyValueError::new_err(format!(
            "data key '{}' must contain at least one value",
            key
        )));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "data key '{}' contains non-finite value {} at flat index {}",
            key, value, index
        )));
    }
    Ok(())
}

fn validate_finite(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{} must be finite", name)))
    }
}

fn validate_positive_finite(name: &str, value: f64) -> PyResult<()> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{} must be > 0", name)))
    }
}

/// Merge call-site data over bound data while ensuring a key has exactly one
/// dimensional kind. A 1-D override removes a stale 2-D binding and vice versa.
fn merge_data_overrides(
    data_1d: &mut HashMap<String, Vec<f64>>,
    data_2d: &mut HashMap<String, (Vec<f64>, usize, usize)>,
    extra_1d: HashMap<String, Vec<f64>>,
    extra_2d: HashMap<String, (Vec<f64>, usize, usize)>,
) {
    for (key, value) in extra_1d {
        data_2d.remove(&key);
        data_1d.insert(key, value);
    }
    for (key, value) in extra_2d {
        data_1d.remove(&key);
        data_2d.insert(key, value);
    }
}

/// Validate that all bound vector-like inputs share the same length.
fn validate_bound_vector_lengths(
    data_1d: &HashMap<String, Vec<f64>>,
    data_2d: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<usize> {
    let mut expected_len: Option<usize> = None;
    let mut expected_label: Option<String> = None;

    let mut record = |actual_len: usize, label: String| -> PyResult<()> {
        match expected_len {
            None => {
                expected_len = Some(actual_len);
                expected_label = Some(label);
                Ok(())
            }
            Some(expected) if expected == actual_len => Ok(()),
            Some(expected) => Err(PyValueError::new_err(format!(
                "shape mismatch: '{}' has length {}, but '{}' has length {}. \
                 rustmc currently requires one shared vector length across all \
                 vectorized data, observations, and matrices.",
                label,
                actual_len,
                expected_label.as_deref().unwrap_or("previous input"),
                expected
            ))),
        }
    };

    for (key, values) in data_1d {
        record(values.len(), format!("data key '{}'", key))?;
    }

    for (key, (values, n_rows, n_cols)) in data_2d {
        if values.len() != n_rows * n_cols {
            return Err(PyValueError::new_err(format!(
                "matrix key '{}' has shape {}x{} but contains {} values",
                key,
                n_rows,
                n_cols,
                values.len()
            )));
        }
        record(*n_rows, format!("matrix key '{}'", key))?;
    }

    Ok(expected_len.unwrap_or(0))
}

/// Validate that every data key referenced in `expr` and `observed_key` exists in the
/// bound data maps.  Called eagerly at `normal_likelihood()` time when data is bound.
fn validate_data_keys(
    expr: &MuExpr,
    observed_key: &str,
    data_1d: &HashMap<String, Vec<f64>>,
    data_2d: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<()> {
    if !data_1d.contains_key(observed_key) {
        let available: Vec<&str> = data_1d.keys().map(String::as_str).collect();
        return Err(PyValueError::new_err(format!(
            "observed key '{}' not found in bound data. Available 1-D keys: [{}]",
            observed_key,
            available.join(", ")
        )));
    }
    validate_expr_keys(expr, data_1d, data_2d)
}

fn validate_expr_keys(
    expr: &MuExpr,
    data_1d: &HashMap<String, Vec<f64>>,
    data_2d: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<()> {
    match expr {
        MuExpr::Const(_) => Ok(()),
        MuExpr::ParamTimesData { data_key, .. } => {
            if !data_1d.contains_key(data_key) {
                let available: Vec<&str> = data_1d.keys().map(String::as_str).collect();
                return Err(PyValueError::new_err(format!(
                    "data key '{}' not found in bound data. Available 1-D keys: [{}]",
                    data_key,
                    available.join(", ")
                )));
            }
            Ok(())
        }
        MuExpr::MatVec { data_key, .. } => {
            if !data_2d.contains_key(data_key) {
                let available: Vec<&str> = data_2d.keys().map(String::as_str).collect();
                return Err(PyValueError::new_err(format!(
                    "matrix key '{}' not found in bound data. Available 2-D keys: [{}]",
                    data_key,
                    available.join(", ")
                )));
            }
            Ok(())
        }
        MuExpr::Param(_) => Ok(()),
        MuExpr::Add(a, b) => {
            validate_expr_keys(a, data_1d, data_2d)?;
            validate_expr_keys(b, data_1d, data_2d)
        }
    }
}

fn validate_binary_observations(obs: &[f64], name: &str) -> PyResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || (*v != 0.0 && *v != 1.0))
    {
        return Err(PyValueError::new_err(format!(
            "Bernoulli-logit likelihood '{}' requires binary observed values; found {} at index {}",
            name, value, idx
        )));
    }
    Ok(())
}

fn validate_positive_observations(obs: &[f64], name: &str, strict: bool) -> PyResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || if strict { *v <= 0.0 } else { *v < 0.0 })
    {
        let relation = if strict {
            "strictly positive"
        } else {
            "non-negative"
        };
        return Err(PyValueError::new_err(format!(
            "{} likelihood '{}' requires {} observed values; found {} at index {}",
            if strict { "LogNormal" } else { "Exponential" },
            name,
            relation,
            value,
            idx
        )));
    }
    Ok(())
}

fn validate_integer_observations(obs: &[f64], name: &str) -> PyResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || *v < 0.0 || v.fract() != 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "NegativeBinomial likelihood '{}' requires non-negative integer observed values; found {} at index {}",
            name, value, idx
        )));
    }
    Ok(())
}

fn validate_count_observations(obs: &[f64], name: &str) -> PyResult<()> {
    if let Some((idx, value)) = obs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || *v < 0.0 || v.fract() != 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "Poisson-log likelihood '{}' requires non-negative integer observed values; found {} at index {}",
            name, value, idx
        )));
    }
    Ok(())
}

fn sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (1.0 + (-x).exp()).ln()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn logit_stable(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    (p / (1.0 - p)).ln()
}

fn invert_param_transform(transform: &ParamTransform, value: f64) -> f64 {
    match transform {
        ParamTransform::Identity => value,
        ParamTransform::Exp => value.max(1e-12).ln(),
        ParamTransform::Sigmoid => logit_stable(value),
        ParamTransform::BoundedSigmoid { lower, upper } => {
            let span = (upper - lower).max(1e-12);
            logit_stable(((value - lower) / span).clamp(1e-12, 1.0 - 1e-12))
        }
    }
}

fn constrained_draw_to_raw(draw: &[f64], transforms: &[ParamTransform]) -> Vec<f64> {
    draw.iter()
        .zip(transforms.iter())
        .map(|(&value, transform)| invert_param_transform(transform, value))
        .collect()
}

fn pointwise_log_likelihood_for_draw(
    graph: &Graph,
    draw: &[f64],
    heads: &[rustmc_core::graph::ObservationHead],
) -> PyResult<Vec<Vec<f64>>> {
    let mut evaluator = Evaluator::new(graph);
    let raw_draw = constrained_draw_to_raw(draw, &graph.param_transforms);
    evaluator.compute(graph, &raw_draw);

    let mut out = Vec::with_capacity(heads.len());
    for head in heads {
        let values = match head.family {
            rustmc_core::graph::ObsFamily::Normal => {
                let sigma_node = head.aux.expect("Normal observation head requires sigma");
                let sigma = evaluator.scalar_at(sigma_node).abs().max(1e-12);
                let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
                let s2 = sigma * sigma;
                let obs = &graph.obs_vectors[head.obs_data_idx];
                let mut vals = Vec::with_capacity(head.n_obs);
                for (i, &observation) in obs.iter().enumerate().take(head.n_obs) {
                    let mu = evaluator.vec_elem(head.linpred, i, graph);
                    let diff = observation - mu;
                    vals.push(log_norm - 0.5 * diff * diff / s2);
                }
                vals
            }
            rustmc_core::graph::ObsFamily::BernoulliLogit => {
                let obs = &graph.obs_vectors[head.obs_data_idx];
                let mut vals = Vec::with_capacity(head.n_obs);
                for (i, &observation) in obs.iter().enumerate().take(head.n_obs) {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    vals.push(observation * eta - softplus(eta));
                }
                vals
            }
            rustmc_core::graph::ObsFamily::PoissonLog => {
                let obs = &graph.obs_vectors[head.obs_data_idx];
                let mut vals = Vec::with_capacity(head.n_obs);
                for (i, &observation) in obs.iter().enumerate().take(head.n_obs) {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    vals.push(
                        observation * eta
                            - eta.exp()
                            - rustmc_core::autodiff::ln_gamma(observation + 1.0),
                    );
                }
                vals
            }
            rustmc_core::graph::ObsFamily::ExponentialLog => {
                let obs = &graph.obs_vectors[head.obs_data_idx];
                let mut vals = Vec::with_capacity(head.n_obs);
                for (i, &observation) in obs.iter().enumerate().take(head.n_obs) {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    vals.push(eta - observation * eta.exp());
                }
                vals
            }
            rustmc_core::graph::ObsFamily::LogNormal => {
                let obs = &graph.obs_vectors[head.obs_data_idx];
                let sigma_node = head.aux.expect("LogNormal observation head requires sigma");
                let sigma = evaluator.scalar_at(sigma_node).abs().max(1e-12);
                let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
                let s2 = sigma * sigma;
                let mut vals = Vec::with_capacity(head.n_obs);
                for (i, &observation) in obs.iter().enumerate().take(head.n_obs) {
                    let mu = evaluator.vec_elem(head.linpred, i, graph);
                    let y = observation.max(1e-300);
                    let ly = y.ln();
                    let diff = ly - mu;
                    vals.push(log_norm - ly - 0.5 * diff * diff / s2);
                }
                vals
            }
            rustmc_core::graph::ObsFamily::NegativeBinomialLog => {
                let obs = &graph.obs_vectors[head.obs_data_idx];
                let alpha_node = head
                    .aux
                    .expect("NegativeBinomial observation head requires alpha");
                let alpha = evaluator.scalar_at(alpha_node).abs().max(1e-12);
                let mut vals = Vec::with_capacity(head.n_obs);
                for (i, &y) in obs.iter().enumerate().take(head.n_obs) {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    let mu = eta.exp();
                    vals.push(
                        rustmc_core::autodiff::ln_gamma(y + alpha)
                            - rustmc_core::autodiff::ln_gamma(alpha)
                            - rustmc_core::autodiff::ln_gamma(y + 1.0)
                            + alpha * (alpha.ln() - (alpha + mu).ln())
                            + y * (eta - (alpha + mu).ln()),
                    );
                }
                vals
            }
        };
        out.push(values);
    }
    Ok(out)
}

/// Parse a Python value (float or ParamRef) into a HyperParam.
fn extract_hyper(obj: &Bound<'_, PyAny>, arg_name: &str) -> PyResult<HyperParam> {
    if let Ok(v) = obj.extract::<f64>() {
        Ok(HyperParam::Const(v))
    } else if let Ok(p) = obj.downcast::<ParamRef>() {
        Ok(HyperParam::Param(p.borrow().name.clone()))
    } else {
        Err(PyValueError::new_err(format!(
            "'{}' must be a float or a ParamRef (e.g. from normal_prior / half_normal_prior)",
            arg_name
        )))
    }
}

/// Resolve a HyperParam to a graph NodeId.
/// Constant → adds a constant node; Param → looks up the value node for a prior parameter.
fn resolve_hyper(
    hp: &HyperParam,
    graph: &mut Graph,
    value_node_map: &HashMap<String, NodeId>,
) -> Result<NodeId, PyErr> {
    match hp {
        HyperParam::Const(v) => Ok(graph.add_constant(*v)),
        HyperParam::Param(name) => value_node_map.get(name.as_str()).copied().ok_or_else(|| {
            ParameterError::new_err(format!(
                "hyperparameter '{}' has no value node. Declare it before the prior \
                 that references it.",
                name
            ))
        }),
    }
}

/// Resolve a `HyperParam` against already-computed parameter values.
///
/// `context` names the model location doing the referencing, so the error can
/// say *which* prior or derived parameter is broken. There is deliberately no
/// default value: a missing hyperparameter must never be silently replaced.
fn resolve_hyper_value(
    hp: &HyperParam,
    values: &HashMap<String, f64>,
    context: &str,
) -> Result<f64, PyErr> {
    match hp {
        HyperParam::Const(v) => Ok(*v),
        HyperParam::Param(name) => values.get(name).copied().ok_or_else(|| {
            let mut available: Vec<&str> = values.keys().map(String::as_str).collect();
            available.sort_unstable();
            ParameterError::new_err(format!(
                "parameter '{}' referenced by {} has no value yet. It must be \
                 declared before the parameter that depends on it. \
                 Available at this point: [{}]",
                name,
                context,
                available.join(", ")
            ))
        }),
    }
}

fn should_auto_noncenter(prior: &PriorSpec, auto_vector_params: &HashMap<String, usize>) -> bool {
    match prior {
        PriorSpec::Normal { name, mu, sigma } => {
            !auto_vector_params.contains_key(name)
                && (matches!(mu, HyperParam::Param(_)) || matches!(sigma, HyperParam::Param(_)))
        }
        _ => false,
    }
}

fn append_raw_display_params(
    display_params: &mut Vec<DisplayParamSpec>,
    graph: &Graph,
    start_idx: usize,
) {
    for raw_index in start_idx..graph.param_count {
        display_params.push(DisplayParamSpec::Raw {
            name: graph.param_names[raw_index].clone(),
            raw_index,
        });
    }
}

fn build_likelihood_into_graph(
    graph: &mut Graph,
    lik: &LikelihoodSpec,
    data_map: &HashMap<String, Vec<f64>>,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
    vector_param_map: &HashMap<String, (usize, usize)>,
    value_node_map: &HashMap<String, NodeId>,
) -> PyResult<()> {
    let linpred_node = build_mu_expr(
        graph,
        &lik.mu_expr,
        data_map,
        matrix_map,
        vector_param_map,
        value_node_map,
    )?;
    let linpred_node = if lik.mu_expr.is_scalar() {
        graph.scalar_broadcast(linpred_node)
    } else {
        linpred_node
    };

    let obs_vec = data_map
        .get(&lik.observed_key)
        .ok_or_else(|| {
            PyValueError::new_err(format!("Missing observed data key: {}", lik.observed_key))
        })?
        .clone();
    let obs_idx = graph.add_named_obs_data(&lik.observed_key, &lik.name, obs_vec.clone());

    match lik.family {
        LikelihoodFamily::Normal => {
            let sigma_spec = lik.sigma.as_ref().ok_or_else(|| {
                PyValueError::new_err(format!("Normal likelihood '{}' is missing sigma", lik.name))
            })?;
            let sigma_node = resolve_sigma(sigma_spec, graph, value_node_map)?;
            graph.normal_obs_logp(linpred_node, sigma_node, obs_idx);
        }
        LikelihoodFamily::BernoulliLogit => {
            validate_binary_observations(&obs_vec, &lik.name)?;
            graph.obs_logp_bernoulli_logit(linpred_node, obs_idx);
        }
        LikelihoodFamily::PoissonLog => {
            validate_count_observations(&obs_vec, &lik.name)?;
            graph.obs_logp_poisson_log(linpred_node, obs_idx);
        }
        LikelihoodFamily::ExponentialLog => {
            validate_positive_observations(&obs_vec, &lik.name, false)?;
            graph.obs_logp_exponential_log(linpred_node, obs_idx);
        }
        LikelihoodFamily::LogNormal => {
            validate_positive_observations(&obs_vec, &lik.name, true)?;
            let sigma_spec = lik.sigma.as_ref().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "LogNormal likelihood '{}' is missing sigma",
                    lik.name
                ))
            })?;
            let sigma_node = resolve_sigma(sigma_spec, graph, value_node_map)?;
            graph.obs_logp_lognormal(linpred_node, sigma_node, obs_idx);
        }
        LikelihoodFamily::NegativeBinomialLog => {
            validate_integer_observations(&obs_vec, &lik.name)?;
            let alpha_spec = lik.sigma.as_ref().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "NegativeBinomial likelihood '{}' is missing alpha",
                    lik.name
                ))
            })?;
            let alpha_node = resolve_sigma(alpha_spec, graph, value_node_map)?;
            graph.obs_logp_negative_binomial_log(linpred_node, alpha_node, obs_idx);
        }
    }
    Ok(())
}

/// Build a single prior into the graph. Used by both `sample()` and `batch_sample()`
/// to avoid duplicating the large match.
fn build_prior_into_graph(
    prior: &PriorSpec,
    graph: &mut Graph,
    vector_param_map: &mut HashMap<String, (usize, usize)>,
    value_node_map: &mut HashMap<String, NodeId>,
    auto_vector_params: &HashMap<String, usize>,
    display_params: &mut Vec<DisplayParamSpec>,
) -> Result<(), PyErr> {
    match prior {
        PriorSpec::Normal { name, mu, sigma } => {
            let start_idx = graph.param_count;
            if should_auto_noncenter(prior, auto_vector_params) {
                let raw_name = format!("{}__raw", name);
                let raw = graph.add_param(&raw_name);
                let zero = graph.add_constant(0.0);
                let one = graph.add_constant(1.0);
                graph.normal_logp(raw, zero, one);
                let mu_node = resolve_hyper(mu, graph, value_node_map)?;
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let scaled = graph.mul(sigma_node, raw);
                let v = graph.add(mu_node, scaled);
                value_node_map.insert(name.clone(), v);
                display_params.push(DisplayParamSpec::DerivedNonCenteredNormal {
                    name: name.clone(),
                    raw_index: start_idx,
                    mu: mu.clone(),
                    sigma: sigma.clone(),
                });
            } else if let Some(&n) = auto_vector_params.get(name) {
                // MatVec auto-promotion: constant hyperparams only
                let (mu_f, sigma_f) = match (mu, sigma) {
                    (HyperParam::Const(m), HyperParam::Const(s)) => (*m, *s),
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Parameter '{}' is used in a matrix multiply (@) but has hierarchical \
                         hyperparameters. Hierarchical vector params are not yet supported.",
                            name
                        )))
                    }
                };
                let param_start = graph.add_vector_params(name, n);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_normal_logp(param_start, n, mu_f, sigma_f);
                append_raw_display_params(display_params, graph, start_idx);
            } else {
                let mu_node = resolve_hyper(mu, graph, value_node_map)?;
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let v = Normal::prior_with_nodes(graph, name, mu_node, sigma_node);
                value_node_map.insert(name.clone(), v);
                append_raw_display_params(display_params, graph, start_idx);
            }
        }
        PriorSpec::HalfNormal { name, sigma } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let sigma_f = match sigma {
                    HyperParam::Const(s) => *s,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                        "Parameter '{}' is used in a matrix multiply (@) but has a hierarchical \
                         sigma. Hierarchical vector params are not yet supported.",
                        name
                    )))
                    }
                };
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_half_normal_logp(param_start, n, sigma_f);
            } else {
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let v = HalfNormal::prior_with_node_sigma(graph, name, sigma_node);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Exponential { name, rate } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let rate_f = match rate {
                    HyperParam::Const(r) => *r,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                        "Parameter '{}' is used in a matrix multiply (@) but has a hierarchical \
                         rate. Hierarchical vector params are not yet supported.",
                        name
                    )))
                    }
                };
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_gamma_logp(param_start, n, 1.0, rate_f);
            } else {
                let rate_node = resolve_hyper(rate, graph, value_node_map)?;
                let v = Exponential::prior_with_node_rate(graph, name, rate_node);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::LogNormal { name, mu, sigma } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let (mu_f, sigma_f) = match (mu, sigma) {
                    (HyperParam::Const(m), HyperParam::Const(s)) => (*m, *s),
                    _ => return Err(PyValueError::new_err(format!(
                        "Parameter '{}' is used in a matrix multiply (@) but has hierarchical \
                         LogNormal hyperparameters. Hierarchical vector params are not yet supported.",
                        name
                    ))),
                };
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_normal_logp(param_start, n, mu_f, sigma_f);
            } else {
                let mu_node = resolve_hyper(mu, graph, value_node_map)?;
                let sigma_node = resolve_hyper(sigma, graph, value_node_map)?;
                let v = LogNormal::prior_with_nodes(graph, name, mu_node, sigma_node);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::StudentT {
            name,
            nu,
            mu,
            sigma,
        } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start = graph.add_vector_params(name, n);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_student_t_logp(param_start, n, *nu, *mu, *sigma);
            } else {
                let v = StudentT::prior(graph, name, *nu, *mu, *sigma);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Uniform { name, lower, upper } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start = graph.add_vector_params_with_transform(
                    name,
                    n,
                    ParamTransform::BoundedSigmoid {
                        lower: *lower,
                        upper: *upper,
                    },
                );
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_uniform_logp(param_start, n, *lower, *upper);
            } else {
                let v = Uniform::prior(graph, name, *lower, *upper);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Bernoulli { name, p } => {
            let start_idx = graph.param_count;
            if auto_vector_params.contains_key(name) {
                return Err(PyValueError::new_err(format!(
                    "Parameter '{}' is used with @ but has a Bernoulli prior. \
                     Discrete distributions cannot be auto-promoted to vector params.",
                    name
                )));
            }
            let v = Bernoulli::prior(graph, name, *p);
            value_node_map.insert(name.clone(), v);
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Poisson { name, lam } => {
            let start_idx = graph.param_count;
            if auto_vector_params.contains_key(name) {
                return Err(PyValueError::new_err(format!(
                    "Parameter '{}' is used with @ but has a Poisson prior. \
                     Discrete distributions cannot be auto-promoted to vector params.",
                    name
                )));
            }
            let v = Poisson::prior(graph, name, *lam);
            value_node_map.insert(name.clone(), v);
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Gamma { name, alpha, beta } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Exp);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_gamma_logp(param_start, n, *alpha, *beta);
            } else {
                let v = Gamma::prior(graph, name, *alpha, *beta);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::Beta { name, alpha, beta } => {
            let start_idx = graph.param_count;
            if let Some(&n) = auto_vector_params.get(name) {
                let param_start =
                    graph.add_vector_params_with_transform(name, n, ParamTransform::Sigmoid);
                vector_param_map.insert(name.clone(), (param_start, n));
                graph.vector_beta_logp(param_start, n, *alpha, *beta);
            } else {
                let v = BetaDist::prior(graph, name, *alpha, *beta);
                value_node_map.insert(name.clone(), v);
            }
            append_raw_display_params(display_params, graph, start_idx);
        }
        PriorSpec::VectorNormal { name, n, mu, sigma } => {
            let start_idx = graph.param_count;
            let param_start = graph.add_vector_params(name, *n);
            vector_param_map.insert(name.clone(), (param_start, *n));
            graph.vector_normal_logp(param_start, *n, *mu, *sigma);
            append_raw_display_params(display_params, graph, start_idx);
        }
    }
    Ok(())
}

/// Resolve a SigmaSpec to a graph NodeId.
fn resolve_sigma(
    spec: &SigmaSpec,
    graph: &mut Graph,
    value_node_map: &HashMap<String, NodeId>,
) -> Result<NodeId, PyErr> {
    match spec {
        SigmaSpec::Const(v) => Ok(graph.add_constant(*v)),
        SigmaSpec::Param(name) => value_node_map.get(name.as_str()).copied().ok_or_else(|| {
            let mut available: Vec<&str> = value_node_map.keys().map(String::as_str).collect();
            available.sort_unstable();
            ParameterError::new_err(format!(
                "scale parameter '{}' is not a scalar parameter of this model. \
                 Scalar parameters: [{}]",
                name,
                available.join(", ")
            ))
        }),
    }
}

/// Try to decompose a MuExpr tree into a flat linear combination:
/// ([(param_name, data_key), ...], optional_intercept_param_name)
fn try_extract_linear(expr: &MuExpr) -> Option<(LinearTerms, Option<String>)> {
    let mut terms = Vec::new();
    let mut intercept: Option<String> = None;

    fn walk(e: &MuExpr, terms: &mut Vec<(String, String)>, intercept: &mut Option<String>) -> bool {
        match e {
            MuExpr::Const(value) => {
                if intercept.is_none() {
                    *intercept = Some(format!("__const__{}", value));
                    true
                } else {
                    false
                }
            }
            MuExpr::ParamTimesData {
                param_name,
                data_key,
            } => {
                terms.push((param_name.clone(), data_key.clone()));
                true
            }
            MuExpr::Add(a, b) => walk(a, terms, intercept) && walk(b, terms, intercept),
            MuExpr::Param(name) => {
                if intercept.is_none() {
                    *intercept = Some(name.clone());
                    true
                } else {
                    false
                }
            }
            // MatVec uses faer GEMV — never fuse into scalar linear combination
            MuExpr::MatVec { .. } => false,
        }
    }

    if walk(expr, &mut terms, &mut intercept) && !terms.is_empty() {
        Some((terms, intercept))
    } else {
        None
    }
}

/// Walk all likelihood MuExpr trees and collect param names used in MatVec ops.
/// Returns a set of param names that should be auto-promoted to vector params.
fn collect_matvec_params(
    likelihoods: &[LikelihoodSpec],
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> Result<HashMap<String, usize>, PyErr> {
    let mut result = HashMap::new();

    fn walk(
        expr: &MuExpr,
        matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
        out: &mut HashMap<String, usize>,
    ) -> Result<(), PyErr> {
        match expr {
            MuExpr::MatVec {
                param_name,
                data_key,
            } => {
                let (_data, _n_rows, n_cols) =
                    matrix_map.get(data_key.as_str()).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Missing matrix key '{}' in data dict",
                            data_key
                        ))
                    })?;
                out.insert(param_name.clone(), *n_cols);
                Ok(())
            }
            MuExpr::Add(a, b) => {
                walk(a, matrix_map, out)?;
                walk(b, matrix_map, out)?;
                Ok(())
            }
            MuExpr::Const(_) | MuExpr::ParamTimesData { .. } | MuExpr::Param(_) => Ok(()),
        }
    }

    for lik in likelihoods {
        walk(&lik.mu_expr, matrix_map, &mut result)?;
    }

    Ok(result)
}

/// Look up the post-transform value node for a scalar parameter.
///
/// Fails loudly — never substitutes a default — when the name is not a scalar
/// parameter of this model.
fn lookup_param_value_node(
    name: &str,
    value_node_map: &HashMap<String, NodeId>,
    context: &str,
) -> Result<NodeId, PyErr> {
    value_node_map.get(name).copied().ok_or_else(|| {
        let mut available: Vec<&str> = value_node_map.keys().map(String::as_str).collect();
        available.sort_unstable();
        ParameterError::new_err(format!(
            "parameter '{}' used in {} is not a scalar parameter of this model. \
             Scalar parameters: [{}]",
            name,
            context,
            available.join(", ")
        ))
    })
}

/// Compile a MuExpr tree into graph nodes.
///
/// Parameters are resolved through `value_node_map`, which holds the
/// *post-transform* value node for every scalar parameter. Resolving via
/// `Graph::node_by_name` instead would return the unconstrained raw node for
/// any transformed prior (HalfNormal, Exponential, LogNormal, Uniform, Gamma,
/// Beta), silently putting a log-scale value into the linear predictor.
///
/// When the tree is a pure linear combination (Σ βₖ xₖ + optional intercept),
/// this emits a single FusedLinearMu op instead of individual
/// ScalarMulData / VectorAdd / ScalarBroadcastAdd nodes.
fn build_mu_expr(
    graph: &mut Graph,
    expr: &MuExpr,
    data_map: &HashMap<String, Vec<f64>>,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
    vector_param_map: &HashMap<String, (usize, usize)>,
    value_node_map: &HashMap<String, NodeId>,
) -> Result<NodeId, PyErr> {
    // Fast path: fuse linear combinations into a single op
    if let Some((terms, intercept_name)) = try_extract_linear(expr) {
        let mut param_nodes = Vec::with_capacity(terms.len());
        let mut data_indices = Vec::with_capacity(terms.len());

        for (param_name, data_key) in &terms {
            let pn = lookup_param_value_node(param_name, value_node_map, "a linear predictor")?;
            param_nodes.push(pn);

            let data_vec = data_map
                .get(data_key)
                .ok_or_else(|| PyValueError::new_err(format!("Missing data key: {}", data_key)))?
                .clone();
            data_indices.push(graph.store_named_data_vec(data_key, data_vec));
        }

        let intercept_node = match intercept_name {
            Some(ref name) if name.starts_with("__const__") => {
                let value = name
                    .trim_start_matches("__const__")
                    .parse::<f64>()
                    .map_err(|_| {
                        PyValueError::new_err(format!(
                            "Invalid constant intercept encoding: {}",
                            name
                        ))
                    })?;
                Some(graph.add_constant(value))
            }
            Some(ref name) => Some(lookup_param_value_node(
                name,
                value_node_map,
                "the intercept of a linear predictor",
            )?),
            None => None,
        };

        return Ok(graph.fused_linear_mu(param_nodes, data_indices, intercept_node));
    }

    // Fallback: individual ops
    match expr {
        MuExpr::Const(value) => Ok(graph.add_constant(*value)),
        MuExpr::ParamTimesData {
            param_name,
            data_key,
        } => {
            let param_node =
                lookup_param_value_node(param_name, value_node_map, "a linear predictor")?;
            let data_vec = data_map
                .get(data_key)
                .ok_or_else(|| PyValueError::new_err(format!("Missing data key: {}", data_key)))?
                .clone();
            let data_node = graph.add_data(data_key, data_vec);
            Ok(graph.scalar_mul_data(param_node, data_node))
        }
        MuExpr::Param(name) => lookup_param_value_node(name, value_node_map, "a linear predictor"),
        MuExpr::MatVec {
            param_name,
            data_key,
        } => {
            let &(param_start, n_params) =
                vector_param_map.get(param_name.as_str()).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Unknown vector param '{}' — did you call vector_normal_prior?",
                        param_name
                    ))
                })?;
            let (data, n_rows, n_cols) = matrix_map.get(data_key.as_str()).ok_or_else(|| {
                PyValueError::new_err(format!("Missing matrix key '{}' in data dict", data_key))
            })?;
            let matrix_idx = graph.store_named_matrix(data_key, data.clone(), *n_rows, *n_cols);
            Ok(graph.mat_vec_mul(matrix_idx, param_start, n_params, None))
        }
        MuExpr::Add(a, b) => {
            let na = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            let nb = build_mu_expr(
                graph,
                b,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            let a_scalar = a.is_scalar();
            let b_scalar = b.is_scalar();
            if a_scalar && !b_scalar {
                Ok(graph.scalar_broadcast_add(na, nb))
            } else if !a_scalar && b_scalar {
                Ok(graph.scalar_broadcast_add(nb, na))
            } else if !a_scalar && !b_scalar {
                Ok(graph.vector_add(na, nb))
            } else {
                Ok(graph.add(na, nb))
            }
        }
    }
}

fn select_posterior_draw_indices(
    total_draws: usize,
    n_samples: Option<usize>,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let n = n_samples.unwrap_or(total_draws).min(total_draws);
    if n >= total_draws {
        return (0..total_draws).collect();
    }

    let mut indices: Vec<usize> = (0..total_draws).collect();
    indices.shuffle(rng);
    indices.truncate(n);
    indices.sort_unstable();
    indices
}

fn compile_python_model(
    model_spec: &ModelSpec,
    data_map: &HashMap<String, Vec<f64>>,
    matrix_map: &HashMap<String, (Vec<f64>, usize, usize)>,
) -> PyResult<CompiledPythonModel> {
    let mut graph = Graph::new();
    let mut vector_param_map: HashMap<String, (usize, usize)> = HashMap::new();
    let mut value_node_map: HashMap<String, NodeId> = HashMap::new();
    let mut display_params = Vec::new();

    // Defence in depth: `ModelBuilder.build()` already validated these, but a
    // `ModelSpec` can reach here by other routes (pickling, batch_sample).
    // Validating before touching the graph guarantees an unresolvable reference
    // never becomes a silently-defaulted value at sampling time.
    validate_model_references(&model_spec.priors, &model_spec.likelihoods)?;

    let auto_vector_params = collect_matvec_params(&model_spec.likelihoods, matrix_map)?;

    for prior in &model_spec.priors {
        build_prior_into_graph(
            prior,
            &mut graph,
            &mut vector_param_map,
            &mut value_node_map,
            &auto_vector_params,
            &mut display_params,
        )?;
    }

    for lik in &model_spec.likelihoods {
        build_likelihood_into_graph(
            &mut graph,
            lik,
            data_map,
            matrix_map,
            &vector_param_map,
            &value_node_map,
        )?;
    }

    graph
        .validate_shapes()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(CompiledPythonModel {
        graph,
        likelihood_names: model_spec
            .likelihoods
            .iter()
            .map(|l| l.name.clone())
            .collect(),
        display_params,
        auto_vector_params,
    })
}

fn derive_display_draw(raw_draw: &[f64], specs: &[DisplayParamSpec]) -> PyResult<Vec<f64>> {
    let mut values = HashMap::new();
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let value = match spec {
            DisplayParamSpec::Raw { name, raw_index } => {
                let value = raw_draw[*raw_index];
                values.insert(name.clone(), value);
                value
            }
            DisplayParamSpec::DerivedNonCenteredNormal {
                name,
                raw_index,
                mu,
                sigma,
            } => {
                let context = format!("non-centered parameter '{}'", name);
                let mu_v = resolve_hyper_value(mu, &values, &context)?;
                let sigma_v = resolve_hyper_value(sigma, &values, &context)?;
                let value = mu_v + sigma_v * raw_draw[*raw_index];
                values.insert(name.clone(), value);
                value
            }
        };
        out.push(value);
    }
    Ok(out)
}

fn derive_display_sample_result(
    raw_result: &SampleResult,
    specs: &[DisplayParamSpec],
) -> PyResult<SampleResult> {
    let mut samples = Vec::with_capacity(raw_result.samples.len());
    for chain in &raw_result.samples {
        let mut chain_out = Vec::with_capacity(chain.len());
        for draw in chain {
            chain_out.push(derive_display_draw(draw, specs)?);
        }
        samples.push(chain_out);
    }
    let param_names = specs
        .iter()
        .map(|spec| match spec {
            DisplayParamSpec::Raw { name, .. } => name.clone(),
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name.clone(),
        })
        .collect();

    Ok(SampleResult {
        samples,
        accept_rates: raw_result.accept_rates.clone(),
        step_sizes: raw_result.step_sizes.clone(),
        divergences: raw_result.divergences.clone(),
        transitions: raw_result.transitions.clone(),
        param_names,
    })
}

fn derive_display_batch_result(
    raw_result: &sampler::BatchModelResult,
    specs: &[DisplayParamSpec],
) -> PyResult<sampler::BatchModelResult> {
    let mut samples = Vec::with_capacity(raw_result.samples.len());
    for draw in &raw_result.samples {
        samples.push(derive_display_draw(draw, specs)?);
    }
    let param_names = specs
        .iter()
        .map(|spec| match spec {
            DisplayParamSpec::Raw { name, .. } => name.clone(),
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name.clone(),
        })
        .collect();

    Ok(sampler::BatchModelResult {
        samples,
        param_names,
        num_chains: raw_result.num_chains,
        num_draws: raw_result.num_draws,
        accept_rates: raw_result.accept_rates.clone(),
        step_sizes: raw_result.step_sizes.clone(),
        divergences: raw_result.divergences.clone(),
        transitions: raw_result.transitions.clone(),
    })
}

fn validate_sample_config(
    chains: usize,
    draws: usize,
    warmup: usize,
    step_size: f64,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
) -> PyResult<()> {
    if chains == 0 {
        return Err(PyValueError::new_err("chains must be >= 1"));
    }
    if draws == 0 {
        return Err(PyValueError::new_err("draws must be >= 1"));
    }
    if warmup == 0 {
        return Err(PyValueError::new_err("warmup must be >= 1"));
    }
    if !step_size.is_finite() || step_size < 0.0 {
        return Err(PyValueError::new_err(
            "step_size must be finite and >= 0 (0 enables adaptation)",
        ));
    }
    if !(1..=63).contains(&max_tree_depth) {
        return Err(PyValueError::new_err(
            "max_tree_depth must be between 1 and 63",
        ));
    }
    if num_leapfrog_steps == 0 {
        return Err(PyValueError::new_err("num_leapfrog_steps must be >= 1"));
    }
    Ok(())
}

fn validate_transition_chain_count(
    transition_chains: usize,
    expected_chains: usize,
) -> Result<(), String> {
    if transition_chains == expected_chains {
        Ok(())
    } else {
        Err(format!(
            "Sampler telemetry has {} chains, but posterior samples have {expected_chains} chains",
            transition_chains
        ))
    }
}

#[pyclass]
struct FitResult {
    raw_result: SampleResult,
    display_result: SampleResult,
    /// A clone of the compiled graph — used for predictive sampling.
    graph: Graph,
    /// Name of each likelihood, in the order they appear in the graph.
    likelihood_names: Vec<String>,
}

#[pymethods]
impl FitResult {
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (pidx, name) in self.display_result.param_names.iter().enumerate() {
            let mut all_samples = Vec::new();
            for chain in &self.display_result.samples {
                for draw in chain {
                    all_samples.push(draw[pidx]);
                }
            }
            let arr = PyArray1::from_vec(py, all_samples);
            dict.set_item(name, arr)?;
        }
        Ok(dict)
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (pidx, name) in self.display_result.param_names.iter().enumerate() {
            let n_chains = self.display_result.samples.len();
            let n_draws = self.display_result.samples[0].len();
            let mut arr = Array2::<f64>::zeros((n_chains, n_draws));
            for (ci, chain) in self.display_result.samples.iter().enumerate() {
                for (di, draw) in chain.iter().enumerate() {
                    arr[[ci, di]] = draw[pidx];
                }
            }
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let means = self.display_result.mean();
        let dict = PyDict::new(py);
        for (name, val) in self.display_result.param_names.iter().zip(means.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stds = self.display_result.std();
        let dict = PyDict::new(py);
        for (name, val) in self.display_result.param_names.iter().zip(stds.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn accept_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.display_result.accept_rates)?;
        Ok(list)
    }

    /// Print a formatted diagnostics table (R-hat, ESS, MCSE, HDI, divergences).
    fn summary(&self) -> String {
        self.display_result.diagnostics().to_table()
    }

    /// Return per-parameter diagnostics as a list of dicts.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let report = self.display_result.diagnostics();
        let items: Vec<Bound<'py, PyDict>> = report
            .params
            .iter()
            .map(|p| {
                let d = PyDict::new(py);
                d.set_item("name", &p.name).unwrap();
                d.set_item("mean", p.mean).unwrap();
                d.set_item("std", p.std).unwrap();
                d.set_item("hdi_3%", p.hdi_3).unwrap();
                d.set_item("hdi_97%", p.hdi_97).unwrap();
                d.set_item("ess_bulk", p.ess_bulk).unwrap();
                d.set_item("ess_tail", p.ess_tail).unwrap();
                d.set_item("r_hat", p.r_hat).unwrap();
                d.set_item("mcse_mean", p.mcse_mean).unwrap();
                d
            })
            .collect();
        let list = PyList::new(py, &items)?;
        Ok(list)
    }

    /// Per-chain adapted step sizes.
    fn step_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.display_result.step_sizes)?;
        Ok(list)
    }

    /// Per-chain divergence counts.
    fn divergences<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.display_result.divergences)?;
        Ok(list)
    }

    /// Draw samples from the posterior predictive distribution.
    ///
    /// For each posterior draw (or a random subsample of `n_samples`), runs a
    /// forward pass through the model graph and samples
    ///     ŷ ~ Normal(mu(params), sigma(params))
    /// for every observation.
    ///
    /// Parameters
    /// ----------
    /// n_samples : int or None
    ///     How many posterior draws to use.  None = use all (chains × draws).
    /// seed : int
    ///     RNG seed for the noise draws.
    ///
    /// Returns
    /// -------
    /// dict[str, ndarray(n_samples, n_obs)]
    ///     One key per likelihood (the name passed to normal_likelihood).
    #[pyo3(signature = (n_samples=None, seed=42))]
    fn posterior_predictive<'py>(
        &self,
        py: Python<'py>,
        n_samples: Option<usize>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut evaluator = Evaluator::new(&self.graph);
        let heads = self.graph.observation_heads();

        // Flatten all chain draws in order, then subsample without replacement
        // when the caller requests fewer draws than are available.
        let all_draws: Vec<&Vec<f64>> = self
            .raw_result
            .samples
            .iter()
            .flat_map(|c| c.iter())
            .collect();
        let chosen_indices = select_posterior_draw_indices(all_draws.len(), n_samples, &mut rng);
        let n = chosen_indices.len();

        // Pre-allocate: predictions[likelihood_idx] = flat Vec of n * n_obs values
        let mut preds: Vec<Vec<f64>> = heads
            .iter()
            .map(|head| Vec::with_capacity(n * head.n_obs))
            .collect();

        for draw_idx in chosen_indices {
            let draw = all_draws[draw_idx];
            let raw_draw = constrained_draw_to_raw(draw, &self.graph.param_transforms);
            evaluator.compute(&self.graph, &raw_draw);
            for (li, head) in heads.iter().enumerate() {
                match head.family {
                    rustmc_core::graph::ObsFamily::Normal => {
                        let sigma_node = head.aux.ok_or_else(|| {
                            PyValueError::new_err("Normal observation head is missing sigma")
                        })?;
                        let sigma = evaluator.scalar_at(sigma_node);
                        validate_positive_finite("likelihood sigma", sigma)?;
                        let noise_dist = NormalDist::new(0.0_f64, sigma)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        for i in 0..head.n_obs {
                            let mu = evaluator.vec_elem(head.linpred, i, &self.graph);
                            preds[li].push(mu + noise_dist.sample(&mut rng));
                        }
                    }
                    rustmc_core::graph::ObsFamily::BernoulliLogit => {
                        for i in 0..head.n_obs {
                            let eta = evaluator.vec_elem(head.linpred, i, &self.graph);
                            let p = sigmoid_stable(eta).clamp(1e-12, 1.0 - 1e-12);
                            preds[li].push(if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
                        }
                    }
                    rustmc_core::graph::ObsFamily::PoissonLog => {
                        for i in 0..head.n_obs {
                            let eta = evaluator.vec_elem(head.linpred, i, &self.graph);
                            let lam = eta.exp();
                            validate_positive_finite("Poisson posterior predictive rate", lam)?;
                            let draw = rand_distr::Poisson::new(lam)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?
                                .sample(&mut rng);
                            preds[li].push(draw);
                        }
                    }
                    rustmc_core::graph::ObsFamily::ExponentialLog => {
                        for i in 0..head.n_obs {
                            let eta = evaluator.vec_elem(head.linpred, i, &self.graph);
                            let rate = eta.exp().max(1e-12);
                            let u = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
                            preds[li].push((-u.ln() / rate).max(1e-12));
                        }
                    }
                    rustmc_core::graph::ObsFamily::LogNormal => {
                        let sigma_node = head.aux.ok_or_else(|| {
                            PyValueError::new_err("LogNormal observation head is missing sigma")
                        })?;
                        let sigma = evaluator.scalar_at(sigma_node);
                        validate_positive_finite("likelihood sigma", sigma)?;
                        let noise_dist = NormalDist::new(0.0_f64, sigma)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        for i in 0..head.n_obs {
                            let mu = evaluator.vec_elem(head.linpred, i, &self.graph);
                            preds[li].push((mu + noise_dist.sample(&mut rng)).exp());
                        }
                    }
                    rustmc_core::graph::ObsFamily::NegativeBinomialLog => {
                        let alpha_node = head.aux.ok_or_else(|| {
                            PyValueError::new_err(
                                "NegativeBinomial observation head is missing alpha",
                            )
                        })?;
                        let alpha = evaluator.scalar_at(alpha_node);
                        validate_positive_finite("negative-binomial alpha", alpha)?;
                        for i in 0..head.n_obs {
                            let eta = evaluator.vec_elem(head.linpred, i, &self.graph);
                            let mu = eta.exp();
                            validate_positive_finite("negative-binomial mean", mu)?;
                            let gamma_scale = mu / alpha;
                            let lambda = rand_distr::Gamma::new(alpha, gamma_scale)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?
                                .sample(&mut rng);
                            let draw = rand_distr::Poisson::new(lambda)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?
                                .sample(&mut rng);
                            preds[li].push(draw);
                        }
                    }
                }
            }
        }

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            let n_obs = heads[li].n_obs;
            let arr = Array2::from_shape_vec((n, n_obs), preds[li].clone())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    /// Pointwise log-likelihood for each observation in each posterior draw.
    ///
    /// Returns a dict of arrays with shape (chain, draw, obs), one per
    /// likelihood. This is the group ArviZ uses for LOO/WAIC workflows.
    fn log_likelihood<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let heads = self.graph.observation_heads();
        let n_chains = self.raw_result.samples.len();
        let n_draws = self.raw_result.samples.first().map_or(0, |c| c.len());

        let mut arrays: Vec<Array3<f64>> = heads
            .iter()
            .map(|head| Array3::<f64>::zeros((n_chains, n_draws, head.n_obs)))
            .collect();

        for (chain_idx, chain) in self.raw_result.samples.iter().enumerate() {
            for (draw_idx, draw) in chain.iter().enumerate() {
                let per_head = pointwise_log_likelihood_for_draw(&self.graph, draw, &heads)?;
                for (li, values) in per_head.iter().enumerate() {
                    for (obs_idx, &value) in values.iter().enumerate() {
                        arrays[li][[chain_idx, draw_idx, obs_idx]] = value;
                    }
                }
            }
        }

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            dict.set_item(name, arrays[li].clone().into_pyarray(py))?;
        }
        Ok(dict)
    }

    /// Convert to an ArviZ InferenceData object.
    ///
    /// Requires ArviZ: `pip install arviz`
    ///
    /// Returns an `arviz.InferenceData` with:
    ///   - `posterior`             — (n_chains × n_draws) arrays for every parameter
    ///   - `sample_stats`          — `diverging` (bool) and `step_size` per draw
    ///   - `observed_data`         — the fitted response vector for each likelihood
    ///   - `posterior_predictive`  — ŷ samples (only when include_ppc=True)
    ///
    /// Example
    /// -------
    ///     idata = fit.to_arviz()
    ///     az.plot_trace(idata)
    ///     az.plot_pair(idata, divergences=True)
    ///     idata = fit.to_arviz(include_ppc=True)
    ///     az.plot_ppc(idata)
    #[pyo3(signature = (include_ppc=false, ppc_samples=None, ppc_seed=42, include_log_likelihood=true))]
    fn to_arviz<'py>(
        &self,
        py: Python<'py>,
        include_ppc: bool,
        ppc_samples: Option<usize>,
        ppc_seed: u64,
        include_log_likelihood: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Preserve ArviZ's actual import failure. This distinguishes a missing
        // optional package from a broken transitive dependency or import-time
        // runtime error, all of which previously looked "not installed".
        let az = py.import("arviz")?;

        let n_chains = self.display_result.samples.len();
        let n_draws = self.display_result.samples.first().map_or(0, |c| c.len());

        // ── posterior ────────────────────────────────────────────────────
        let posterior = self.get_samples_2d(py)?;

        // ── sample_stats ─────────────────────────────────────────────────
        // Transitions include warmup for auditability. ArviZ sample_stats is
        // aligned with posterior draws, so export only post-warmup telemetry.
        validate_transition_chain_count(self.raw_result.transitions.len(), n_chains)
            .map_err(PyValueError::new_err)?;
        let sample_stats = PyDict::new(py);
        let mut step_size_arr = Array2::<f64>::zeros((n_chains, n_draws));
        let mut diverging_arr = Array2::<bool>::from_elem((n_chains, n_draws), false);
        for (ci, transitions) in self.raw_result.transitions.iter().enumerate() {
            let post_warmup: Vec<_> = transitions
                .iter()
                .filter(|transition| !transition.is_warmup)
                .collect();
            if post_warmup.len() != n_draws {
                return Err(PyValueError::new_err(format!(
                    "Sampler telemetry for chain {ci} has {} posterior transitions, expected {n_draws}",
                    post_warmup.len()
                )));
            }
            for (di, transition) in post_warmup.into_iter().enumerate() {
                step_size_arr[[ci, di]] = transition.step_size;
                diverging_arr[[ci, di]] = transition.divergent;
            }
        }
        sample_stats.set_item("step_size", step_size_arr.into_pyarray(py))?;
        sample_stats.set_item("diverging", diverging_arr.into_pyarray(py))?;

        // ── posterior predictive (optional) ──────────────────────────────
        let kwargs = PyDict::new(py);
        kwargs.set_item("posterior", posterior)?;
        kwargs.set_item("sample_stats", sample_stats)?;

        if !self.likelihood_names.is_empty() {
            let heads = self.graph.observation_heads();
            let observed_data = PyDict::new(py);
            for (li, name) in self.likelihood_names.iter().enumerate() {
                let head = heads.get(li).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "observation metadata for likelihood '{}' is unavailable",
                        name
                    ))
                })?;
                let observed = self
                    .graph
                    .obs_vectors
                    .get(head.obs_data_idx)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "observed payload for likelihood '{}' is unavailable",
                            name
                        ))
                    })?;
                observed_data.set_item(name, PyArray1::from_vec(py, observed.clone()))?;
            }
            kwargs.set_item("observed_data", observed_data)?;
        }

        if include_log_likelihood && !self.likelihood_names.is_empty() {
            let log_likelihood = self.log_likelihood(py)?;
            kwargs.set_item("log_likelihood", log_likelihood)?;
        }

        if include_ppc && !self.likelihood_names.is_empty() {
            let ppc_dict = self.posterior_predictive(py, ppc_samples, ppc_seed)?;
            // Reshape (n_samples, n_obs) → (1, n_samples, n_obs) for ArviZ convention
            // ArviZ expects posterior_predictive as (chain, draw, obs)
            // We treat all samples as a single chain.
            let ppc_reshaped = PyDict::new(py);
            let np = py.import("numpy")?;
            for (key, arr) in ppc_dict.iter() {
                // arr is (n_samples, n_obs); expand_dims to (1, n_samples, n_obs)
                let expanded = np.call_method1("expand_dims", (arr, 0))?;
                ppc_reshaped.set_item(key, expanded)?;
            }
            kwargs.set_item("posterior_predictive", ppc_reshaped)?;
        }

        az.call_method("from_dict", (), Some(&kwargs))
    }

    fn __repr__(&self) -> String {
        let means = self.display_result.mean();
        let stds = self.display_result.std();
        let mut parts = Vec::new();
        for (i, name) in self.display_result.param_names.iter().enumerate() {
            parts.push(format!(
                "  {}: mean={:.4}, std={:.4}",
                name, means[i], stds[i]
            ));
        }
        let n_chains = self.display_result.samples.len();
        let n_draws = if self.display_result.samples.is_empty() {
            0
        } else {
            self.display_result.samples[0].len()
        };
        format!(
            "rustmc FitResult ({} chains × {} draws)\n{}",
            n_chains,
            n_draws,
            parts.join("\n")
        )
    }
}

#[pyfunction]
#[pyo3(signature = (model_spec, data=None, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true))]
#[allow(clippy::too_many_arguments)]
fn sample(
    py: Python<'_>,
    model_spec: &ModelSpec,
    data: Option<&Bound<'_, PyDict>>,
    chains: usize,
    draws: usize,
    warmup: usize,
    seed: u64,
    threads: usize,
    step_size: f64,
    sampler: &str,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
    show_progress: bool,
) -> PyResult<FitResult> {
    validate_sample_config(
        chains,
        draws,
        warmup,
        step_size,
        max_tree_depth,
        num_leapfrog_steps,
    )?;
    reject_discrete_priors_for_gradient_sampling(&model_spec.priors)?;
    // Start from data bound at build time, then let call-site data override/extend.
    let mut data_map: HashMap<String, Vec<f64>> = model_spec.bound_data_1d.clone();
    let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> =
        model_spec.bound_data_2d.clone();

    if let Some(data_dict) = data {
        let (extra_1d, extra_2d) = parse_data_dict(data_dict)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, extra_1d, extra_2d);
    }

    validate_bound_vector_lengths(&data_map, &matrix_map)?;

    if data_map.is_empty() && matrix_map.is_empty() && !model_spec.likelihoods.is_empty() {
        return Err(PyValueError::new_err(
            "No data provided. Pass data= to sample() or bind it via ModelBuilder(data=...).",
        ));
    }

    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;

    let sampler_type = match sampler {
        "nuts" | "NUTS" => SamplerType::Nuts,
        "hmc" | "HMC" => SamplerType::Hmc,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
                sampler
            )))
        }
    };

    let config = SamplerConfig {
        sampler: sampler_type,
        num_chains: chains,
        num_draws: draws,
        num_warmup: warmup,
        step_size,
        num_leapfrog_steps,
        max_tree_depth,
        seed,
        num_threads: threads,
        show_progress,
    };

    let graph_for_predict = compiled.graph.clone();

    let result = py
        .allow_threads(|| sampler::sample(compiled.graph, config))
        .map_err(PyValueError::new_err)?;
    let display_result = derive_display_sample_result(&result, &compiled.display_params)?;

    Ok(FitResult {
        raw_result: result,
        display_result,
        graph: graph_for_predict,
        likelihood_names: compiled.likelihood_names,
    })
}

/// Result for a single model in a batch run.
#[pymethods]
impl PyCompiledModel {
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.structure.param_names.clone()
    }

    #[getter]
    fn required_keys(&self) -> Vec<String> {
        self.structure
            .schema
            .required_keys()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Stable for this process and useful for verifying Arc structure sharing.
    #[getter]
    fn structure_id(&self) -> usize {
        Arc::as_ptr(&self.structure) as usize
    }

    #[pyo3(signature = (data, id="0", strict=true, check_finite=true))]
    fn bind(
        &self,
        data: &Bound<'_, PyDict>,
        id: &str,
        strict: bool,
        check_finite: bool,
    ) -> PyResult<PyBoundModel> {
        let mut one_d = self.default_data_1d.clone();
        let mut two_d = self.default_data_2d.clone();
        let (extra_1d, extra_2d) = parse_data_dict(data)?;
        merge_data_overrides(&mut one_d, &mut two_d, extra_1d, extra_2d);
        let binding = core_binding_from_maps(
            &self.structure.schema,
            &one_d,
            &two_d,
            id.to_string(),
            strict,
            check_finite,
        )?;
        Ok(PyBoundModel {
            structure: Arc::clone(&self.structure),
            binding: validate_core_binding(&self.structure, binding)?,
        })
    }

    #[pyo3(signature = (data, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        chains: usize,
        draws: usize,
        warmup: usize,
        seed: u64,
        threads: usize,
        step_size: f64,
        sampler: &str,
        max_tree_depth: usize,
        num_leapfrog_steps: usize,
        show_progress: bool,
    ) -> PyResult<FitResult> {
        validate_sample_config(
            chains,
            draws,
            warmup,
            step_size,
            max_tree_depth,
            num_leapfrog_steps,
        )?;
        let binding = self.bind_any(data, "0".to_string(), None)?;
        let sampler_type = parse_sampler_type(sampler)?;
        let config = SamplerConfig {
            sampler: sampler_type,
            num_chains: chains,
            num_draws: draws,
            num_warmup: warmup,
            step_size,
            num_leapfrog_steps,
            max_tree_depth,
            seed,
            num_threads: threads,
            show_progress,
        };
        let hydrated_graph = self.structure.with_binding(&binding);
        let result = py
            .allow_threads(|| sampler::sample_bound(Arc::clone(&self.structure), binding, config))
            .map_err(PyValueError::new_err)?;
        let display_result = derive_display_sample_result(&result, &self.display_params)?;
        Ok(FitResult {
            raw_result: result,
            display_result,
            graph: hydrated_graph,
            likelihood_names: self.likelihood_names.clone(),
        })
    }

    #[pyo3(signature = (datasets, ids=None, shared=None, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn sample_batch(
        &self,
        py: Python<'_>,
        datasets: Vec<Bound<'_, PyAny>>,
        ids: Option<Vec<String>>,
        shared: Option<&Bound<'_, PyDict>>,
        chains: usize,
        draws: usize,
        warmup: usize,
        seed: u64,
        sampler: &str,
        step_size: f64,
        max_tree_depth: usize,
        num_leapfrog_steps: usize,
        show_progress: bool,
    ) -> PyResult<PyBatchFit> {
        validate_sample_config(
            chains,
            draws,
            warmup,
            step_size,
            max_tree_depth,
            num_leapfrog_steps,
        )?;
        let ids = ids.unwrap_or_else(|| (0..datasets.len()).map(|i| i.to_string()).collect());
        if ids.len() != datasets.len() {
            return Err(PyValueError::new_err(
                "ids length must equal datasets length",
            ));
        }
        let mut unique = std::collections::HashSet::new();
        if ids.iter().any(|id| !unique.insert(id)) {
            return Err(PyValueError::new_err("dataset ids must be unique"));
        }
        // Convert defaults/shared payloads once. Cloning this map only clones
        // Arc handles, so a shared design matrix remains one allocation.
        let mut base_1d = self.default_data_1d.clone();
        let mut base_2d = self.default_data_2d.clone();
        let mut shared_keys = std::collections::HashSet::new();
        if let Some(shared) = shared {
            let (shared_1d, shared_2d) = parse_data_dict(shared)?;
            shared_keys.extend(shared_1d.keys().cloned());
            shared_keys.extend(shared_2d.keys().cloned());
            merge_data_overrides(&mut base_1d, &mut base_2d, shared_1d, shared_2d);
        }
        let base_inputs = data_inputs_from_maps(&base_1d, &base_2d);
        let bindings = datasets
            .iter()
            .zip(&ids)
            .map(|(data, id)| {
                if let Ok(bound) = data.downcast::<PyBoundModel>() {
                    let bound = bound.borrow();
                    if !Arc::ptr_eq(&bound.structure, &self.structure) {
                        return Err(PyValueError::new_err(
                            "BoundModel belongs to a different CompiledModel",
                        ));
                    }
                    let mut binding = bound.binding.clone();
                    binding.set_id(id.clone());
                    return validate_core_binding(&self.structure, binding);
                }
                let dict = data.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("datasets must contain dicts or BoundModel objects")
                })?;
                let (extra_1d, extra_2d) = parse_data_dict(dict)?;
                if let Some(key) = extra_1d
                    .keys()
                    .chain(extra_2d.keys())
                    .find(|key| shared_keys.contains(*key))
                {
                    return Err(PyValueError::new_err(format!(
                        "data key '{}' appears in both shared and per-dataset inputs",
                        key
                    )));
                }
                let mut inputs = base_inputs.clone();
                for (key, values) in extra_1d {
                    inputs.matrices.remove(&key);
                    inputs.vectors.insert(key, Arc::from(values));
                }
                for (key, (values, n_rows, n_cols)) in extra_2d {
                    inputs.vectors.remove(&key);
                    inputs.matrices.insert(
                        key,
                        MatrixBinding {
                            data: Arc::from(values),
                            n_rows,
                            n_cols,
                        },
                    );
                }
                let binding =
                    CoreDataBinding::bind(&self.structure.schema, inputs, id.clone(), true, true)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                validate_core_binding(&self.structure, binding)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let config = sampler::BatchSampleConfig {
            sampler: parse_sampler_type(sampler)?,
            num_chains: chains,
            num_draws: draws,
            num_warmup: warmup,
            step_size,
            num_leapfrog_steps,
            max_tree_depth,
            seed,
            show_progress,
        };
        let raw = py
            .allow_threads(|| {
                sampler::sample_batch_bound(Arc::clone(&self.structure), bindings, config)
            })
            .map_err(PyValueError::new_err)?;
        let mut results = Vec::with_capacity(raw.len());
        for item in raw {
            results.push(BatchResult {
                inner: derive_display_batch_result(&item.result, &self.display_params)?,
            });
        }
        Ok(PyBatchFit { ids, results })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledModel(params={}, required_keys={:?})",
            self.structure.param_count,
            self.required_keys()
        )
    }
}

fn parse_sampler_type(sampler: &str) -> PyResult<SamplerType> {
    match sampler {
        "nuts" | "NUTS" => Ok(SamplerType::Nuts),
        "hmc" | "HMC" => Ok(SamplerType::Hmc),
        _ => Err(PyValueError::new_err(format!(
            "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
            sampler
        ))),
    }
}

#[pyclass]
#[derive(Clone)]
struct BatchResult {
    inner: sampler::BatchModelResult,
}

#[pymethods]
impl BatchResult {
    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let n_chains = self.inner.num_chains;
        let n_draws = self.inner.num_draws;
        for (pidx, name) in self.inner.param_names.iter().enumerate() {
            let mut arr = Array2::<f64>::zeros((n_chains, n_draws));
            for chain_idx in 0..n_chains {
                for draw_idx in 0..n_draws {
                    let flat_idx = chain_idx * n_draws + draw_idx;
                    arr[[chain_idx, draw_idx]] = self.inner.samples[flat_idx][pidx];
                }
            }
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let means = self.inner.mean();
        let dict = PyDict::new(py);
        for (name, val) in self.inner.param_names.iter().zip(means.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stds = self.inner.std();
        let dict = PyDict::new(py);
        for (name, val) in self.inner.param_names.iter().zip(stds.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (pidx, name) in self.inner.param_names.iter().enumerate() {
            let vals: Vec<f64> = self.inner.samples.iter().map(|d| d[pidx]).collect();
            let arr = PyArray1::from_vec(py, vals);
            dict.set_item(name, arr)?;
        }
        Ok(dict)
    }

    #[getter]
    fn chains(&self) -> usize {
        self.inner.num_chains
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.num_draws
    }

    #[getter]
    fn accept_rate(&self) -> f64 {
        self.inner.mean_accept_rate()
    }

    #[getter]
    fn accept_rates(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.accept_rates.clone())
    }

    #[getter]
    fn divergences(&self) -> usize {
        self.inner.total_divergences()
    }

    #[getter]
    fn divergences_per_chain(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.divergences.clone())
    }

    fn __repr__(&self) -> String {
        let means = self.inner.mean();
        let parts: Vec<String> = self
            .inner
            .param_names
            .iter()
            .zip(means.iter())
            .map(|(n, m)| format!("{}={:.4}", n, m))
            .collect();
        format!(
            "BatchResult({} chains × {} draws, {})",
            self.inner.num_chains,
            self.inner.num_draws,
            parts.join(", ")
        )
    }
}

#[pyclass(name = "BatchFit")]
struct PyBatchFit {
    ids: Vec<String>,
    results: Vec<BatchResult>,
}

#[pymethods]
impl PyBatchFit {
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }

    fn __len__(&self) -> usize {
        self.results.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<BatchResult>> {
        let len = self.results.len() as isize;
        let normalized = if index < 0 { len + index } else { index };
        if normalized < 0 || normalized >= len {
            return Err(PyIndexError::new_err("batch index out of range"));
        }
        let result = self
            .results
            .get(normalized as usize)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("batch index out of range"))?;
        Py::new(py, result)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }

    fn __repr__(&self) -> String {
        format!("BatchFit({} datasets, 0 failed)", self.results.len())
    }
}

/// Run thousands of independent models in parallel through Rayon.
///
/// Each entry in `models` is a (ModelSpec, data_dict) pair. By default each gets
/// 1 NUTS chain for throughput, but the batch runner can be configured to use
/// multiple chains or fixed-step HMC when reliability matters more.
#[pyfunction]
#[pyo3(signature = (models, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true))]
// The Python API intentionally exposes each sampler option as a named argument.
#[allow(clippy::too_many_arguments)]
fn batch_sample(
    py: Python<'_>,
    models: Vec<(Bound<'_, ModelSpec>, Bound<'_, PyDict>)>,
    chains: usize,
    draws: usize,
    warmup: usize,
    seed: u64,
    sampler: &str,
    step_size: f64,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
    show_progress: bool,
) -> PyResult<Vec<BatchResult>> {
    validate_sample_config(
        chains,
        draws,
        warmup,
        step_size,
        max_tree_depth,
        num_leapfrog_steps,
    )?;

    let mut compiled_models = Vec::with_capacity(models.len());

    for (spec_bound, data_bound) in &models {
        let spec = spec_bound.borrow();
        reject_discrete_priors_for_gradient_sampling(&spec.priors)?;

        // Bound data from ModelSpec is the base; call-site dict overrides/extends.
        let mut data_map: HashMap<String, Vec<f64>> = spec.bound_data_1d.clone();
        let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> = spec.bound_data_2d.clone();
        let (extra_1d, extra_2d) = parse_data_dict(data_bound)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, extra_1d, extra_2d);

        validate_bound_vector_lengths(&data_map, &matrix_map)?;

        compiled_models.push(compile_python_model(&spec, &data_map, &matrix_map)?);
    }

    let sampler = match sampler {
        "nuts" | "NUTS" => SamplerType::Nuts,
        "hmc" | "HMC" => SamplerType::Hmc,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown sampler '{}'. Use 'nuts' or 'hmc'.",
                sampler
            )))
        }
    };

    let config = sampler::BatchSampleConfig {
        sampler,
        num_chains: chains,
        num_draws: draws,
        num_warmup: warmup,
        step_size,
        num_leapfrog_steps,
        max_tree_depth,
        seed,
        show_progress,
    };

    let graphs: Vec<(Graph, Vec<f64>)> = compiled_models
        .iter()
        .map(|compiled| (compiled.graph.clone(), vec![]))
        .collect();

    let results = py
        .allow_threads(|| sampler::batch_sample(graphs, config))
        .map_err(PyValueError::new_err)?;

    results
        .into_iter()
        .zip(compiled_models.iter())
        .map(|(raw_result, compiled)| {
            Ok(BatchResult {
                inner: derive_display_batch_result(&raw_result, &compiled.display_params)?,
            })
        })
        .collect()
}

/// Draw samples from the **prior predictive** distribution.
///
/// Samples parameters from the model priors using their analytic distributions,
/// then runs a forward pass to generate predicted observations.
/// Use this to check whether your priors make sense before fitting.
///
/// Parameters
/// ----------
/// model_spec : ModelSpec
///     A compiled model (from `builder.build()`).  Must have at least one likelihood.
/// data : dict or None
///     Data dict (same as `sample()`).  Needed for the predictor covariates (x values).
/// n_samples : int
///     Number of prior predictive draws.
/// seed : int
///     RNG seed.
///
/// Returns
/// -------
/// dict
///     ``"<param_name>"`` → 1-D array of n_samples prior samples.
///     ``"<likelihood_name>"`` → 2-D array (n_samples, n_obs) of predicted y.
#[pyfunction]
#[pyo3(signature = (model_spec, data=None, n_samples=500, seed=42))]
fn sample_prior_predictive<'py>(
    py: Python<'py>,
    model_spec: &ModelSpec,
    data: Option<&Bound<'py, PyDict>>,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be >= 1"));
    }
    // ── Build data maps ───────────────────────────────────────────────────────
    let mut data_map: HashMap<String, Vec<f64>> = model_spec.bound_data_1d.clone();
    let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> =
        model_spec.bound_data_2d.clone();
    if let Some(d) = data {
        let (e1, e2) = parse_data_dict(d)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, e1, e2);
    }

    validate_bound_vector_lengths(&data_map, &matrix_map)?;

    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;
    let graph = compiled.graph.clone();
    let likelihood_names = compiled.likelihood_names.clone();
    let heads = graph.observation_heads();

    // ── Sample from priors and run forward passes ─────────────────────────────
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut evaluator = Evaluator::new(&graph);

    let mut param_prior_draws: Vec<Vec<f64>> =
        vec![Vec::with_capacity(n_samples); compiled.display_params.len()];
    // predictions[lik_idx] = flat Vec (n_samples * n_obs)
    let mut preds: Vec<Vec<f64>> = heads
        .iter()
        .map(|head| Vec::with_capacity(n_samples * head.n_obs))
        .collect();

    for _ in 0..n_samples {
        // Sample raw parameters from priors (in declaration order)
        let raw = sample_prior_raw(&model_spec.priors, &compiled.auto_vector_params, &mut rng)?;
        if raw.len() != graph.param_count {
            return Err(PyValueError::new_err(format!(
                "prior sampler produced {} raw values, but the compiled model requires {}",
                raw.len(),
                graph.param_count
            )));
        }
        let constrained_raw: Vec<f64> = raw
            .iter()
            .enumerate()
            .map(|(pi, &r)| graph.param_transforms[pi].apply(r))
            .collect();
        let display_draw = derive_display_draw(&constrained_raw, &compiled.display_params)?;
        for (pi, &value) in display_draw.iter().enumerate() {
            param_prior_draws[pi].push(value);
        }

        // Forward pass to get predictions
        evaluator.compute(&graph, &raw);
        for (li, head) in heads.iter().enumerate() {
            match head.family {
                rustmc_core::graph::ObsFamily::Normal => {
                    let sigma_node = head.aux.ok_or_else(|| {
                        PyValueError::new_err("Normal observation head is missing sigma")
                    })?;
                    let sigma = evaluator.scalar_at(sigma_node);
                    validate_positive_finite("likelihood sigma", sigma)?;
                    let noise_dist = NormalDist::new(0.0_f64, sigma)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    for i in 0..head.n_obs {
                        let mu = evaluator.vec_elem(head.linpred, i, &graph);
                        preds[li].push(mu + noise_dist.sample(&mut rng));
                    }
                }
                rustmc_core::graph::ObsFamily::BernoulliLogit => {
                    for i in 0..head.n_obs {
                        let eta = evaluator.vec_elem(head.linpred, i, &graph);
                        let p = sigmoid_stable(eta).clamp(1e-12, 1.0 - 1e-12);
                        preds[li].push(if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
                    }
                }
                rustmc_core::graph::ObsFamily::PoissonLog => {
                    for i in 0..head.n_obs {
                        let eta = evaluator.vec_elem(head.linpred, i, &graph);
                        let lam = eta.exp();
                        validate_positive_finite("Poisson prior predictive rate", lam)?;
                        let draw = rand_distr::Poisson::new(lam)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?
                            .sample(&mut rng);
                        preds[li].push(draw);
                    }
                }
                rustmc_core::graph::ObsFamily::ExponentialLog => {
                    for i in 0..head.n_obs {
                        let eta = evaluator.vec_elem(head.linpred, i, &graph);
                        let rate = eta.exp().max(1e-12);
                        let u = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
                        preds[li].push((-u.ln() / rate).max(1e-12));
                    }
                }
                rustmc_core::graph::ObsFamily::LogNormal => {
                    let sigma_node = head.aux.ok_or_else(|| {
                        PyValueError::new_err("LogNormal observation head is missing sigma")
                    })?;
                    let sigma = evaluator.scalar_at(sigma_node);
                    validate_positive_finite("likelihood sigma", sigma)?;
                    let noise_dist = NormalDist::new(0.0_f64, sigma)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    for i in 0..head.n_obs {
                        let mu = evaluator.vec_elem(head.linpred, i, &graph);
                        preds[li].push((mu + noise_dist.sample(&mut rng)).exp());
                    }
                }
                rustmc_core::graph::ObsFamily::NegativeBinomialLog => {
                    let alpha_node = head.aux.ok_or_else(|| {
                        PyValueError::new_err("NegativeBinomial observation head is missing alpha")
                    })?;
                    let alpha = evaluator.scalar_at(alpha_node);
                    validate_positive_finite("negative-binomial alpha", alpha)?;
                    for i in 0..head.n_obs {
                        let eta = evaluator.vec_elem(head.linpred, i, &graph);
                        let mu = eta.exp();
                        validate_positive_finite("negative-binomial mean", mu)?;
                        let gamma_scale = mu / alpha;
                        let lambda = rand_distr::Gamma::new(alpha, gamma_scale)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?
                            .sample(&mut rng);
                        let draw = rand_distr::Poisson::new(lambda)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?
                            .sample(&mut rng);
                        preds[li].push(draw);
                    }
                }
            }
        }
    }

    // ── Package results ───────────────────────────────────────────────────────
    let dict = PyDict::new(py);
    for (pi, spec) in compiled.display_params.iter().enumerate() {
        let name = match spec {
            DisplayParamSpec::Raw { name, .. } => name,
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name,
        };
        let arr = PyArray1::from_vec(py, param_prior_draws[pi].clone());
        dict.set_item(name, arr)?;
    }
    for (li, name) in likelihood_names.iter().enumerate() {
        let n_obs = heads[li].n_obs;
        let arr = Array2::from_shape_vec((n_samples, n_obs), preds[li].clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray(py))?;
    }
    Ok(dict)
}

/// Sample raw (unconstrained) parameters from the model priors.
/// Processes priors in declaration order so hierarchical hyperpriors work.
fn sample_prior_raw(
    priors: &[PriorSpec],
    auto_vector_params: &HashMap<String, usize>,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>, PyErr> {
    use rand_distr::{
        Beta, Gamma as GammaDist, Poisson as PoissonDist, StandardNormal, StudentT as StudentTDist,
        Uniform as UniformDist,
    };

    let mut raw: Vec<f64> = Vec::new();
    // Track post-transform values for HyperParam::Param resolution
    let mut sampled_values: HashMap<String, f64> = HashMap::new();

    // A hyperparameter that is not yet available is a broken model, not a
    // reason to substitute 1.0: doing so returns plausible-but-wrong prior
    // predictive draws with no warning.
    let resolve = |hp: &HyperParam, sv: &HashMap<String, f64>, owner: &str| -> Result<f64, PyErr> {
        resolve_hyper_value(hp, sv, &format!("prior '{}'", owner))
    };

    for prior in priors {
        match prior {
            PriorSpec::Normal { name, mu, sigma } => {
                if let Some(&n) = auto_vector_params.get(name) {
                    let mu_v = resolve(mu, &sampled_values, name)?;
                    let sigma_v = resolve(sigma, &sampled_values, name)?;
                    validate_positive_finite("sigma", sigma_v)?;
                    let dist = NormalDist::new(mu_v, sigma_v)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    for k in 0..n {
                        let x = dist.sample(rng);
                        if k == 0 {
                            sampled_values.insert(name.clone(), x);
                        }
                        raw.push(x);
                    }
                } else if should_auto_noncenter(prior, auto_vector_params) {
                    let z: f64 = StandardNormal.sample(rng);
                    let mu_v = resolve(mu, &sampled_values, name)?;
                    let sigma_v = resolve(sigma, &sampled_values, name)?;
                    validate_positive_finite("sigma", sigma_v)?;
                    sampled_values.insert(name.clone(), mu_v + sigma_v * z);
                    raw.push(z);
                } else {
                    let mu_v = resolve(mu, &sampled_values, name)?;
                    let sigma_v = resolve(sigma, &sampled_values, name)?;
                    validate_positive_finite("sigma", sigma_v)?;
                    let x = NormalDist::new(mu_v, sigma_v)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .sample(rng);
                    sampled_values.insert(name.clone(), x);
                    raw.push(x); // identity transform
                }
            }
            PriorSpec::HalfNormal { name, sigma } => {
                let sigma_v = resolve(sigma, &sampled_values, name)?;
                validate_positive_finite("sigma", sigma_v)?;
                let dist = NormalDist::new(0.0_f64, sigma_v)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let x = dist.sample(rng).abs().max(1e-12);
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    raw.push(x.ln()); // Exp transform: raw = log(x)
                }
            }
            PriorSpec::Exponential { name, rate } => {
                if let Some(&n) = auto_vector_params.get(name) {
                    let rate_v = resolve(rate, &sampled_values, name)?;
                    validate_positive_finite("rate", rate_v)?;
                    for k in 0..n {
                        let u = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
                        let x = (-u.ln() / rate_v).max(1e-12);
                        if k == 0 {
                            sampled_values.insert(name.clone(), x);
                        }
                        raw.push(x.ln());
                    }
                } else {
                    let rate_v = resolve(rate, &sampled_values, name)?;
                    validate_positive_finite("rate", rate_v)?;
                    let u = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
                    let x = (-u.ln() / rate_v).max(1e-12);
                    sampled_values.insert(name.clone(), x);
                    raw.push(x.ln());
                }
            }
            PriorSpec::LogNormal { name, mu, sigma } => {
                if let Some(&n) = auto_vector_params.get(name) {
                    let mu_v = resolve(mu, &sampled_values, name)?;
                    let sigma_v = resolve(sigma, &sampled_values, name)?;
                    validate_positive_finite("sigma", sigma_v)?;
                    let dist = NormalDist::new(mu_v, sigma_v)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    for k in 0..n {
                        let raw_draw = dist.sample(rng);
                        if k == 0 {
                            sampled_values.insert(name.clone(), raw_draw.exp());
                        }
                        raw.push(raw_draw);
                    }
                } else {
                    let mu_v = resolve(mu, &sampled_values, name)?;
                    let sigma_v = resolve(sigma, &sampled_values, name)?;
                    validate_positive_finite("sigma", sigma_v)?;
                    let raw_draw = NormalDist::new(mu_v, sigma_v)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .sample(rng);
                    let x = raw_draw.exp();
                    sampled_values.insert(name.clone(), x);
                    raw.push(raw_draw);
                }
            }
            PriorSpec::StudentT {
                name,
                nu,
                mu,
                sigma,
            } => {
                let dist =
                    StudentTDist::new(*nu).map_err(|e| PyValueError::new_err(e.to_string()))?;
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let x = mu + sigma * dist.sample(rng);
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    raw.push(x);
                }
            }
            PriorSpec::Uniform { name, lower, upper } => {
                let dist = UniformDist::new(*lower, *upper);
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let x = dist.sample(rng);
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    let p = ((x - lower) / (upper - lower)).clamp(1e-12, 1.0 - 1e-12);
                    raw.push((p / (1.0 - p)).ln());
                }
            }
            PriorSpec::Gamma { name, alpha, beta } => {
                let dist = GammaDist::new(*alpha, 1.0 / beta)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let x = dist.sample(rng).max(1e-12);
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    raw.push(x.ln()); // Exp transform
                }
            }
            PriorSpec::Beta { name, alpha, beta } => {
                let dist =
                    Beta::new(*alpha, *beta).map_err(|e| PyValueError::new_err(e.to_string()))?;
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let x = dist.sample(rng).clamp(1e-12, 1.0 - 1e-12);
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    raw.push((x / (1.0 - x)).ln());
                }
            }
            PriorSpec::Bernoulli { name, p } => {
                let x: f64 = if rng.gen::<f64>() < *p { 1.0 } else { 0.0 };
                sampled_values.insert(name.clone(), x);
                raw.push(x);
            }
            PriorSpec::Poisson { name, lam } => {
                let x = PoissonDist::new(*lam)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .sample(rng);
                sampled_values.insert(name.clone(), x);
                raw.push(x);
            }
            PriorSpec::VectorNormal { name, n, mu, sigma } => {
                let dist = NormalDist::new(*mu, *sigma)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                for k in 0..*n {
                    let x = dist.sample(rng);
                    // Store only the first component for HyperParam resolution (rare case)
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    raw.push(x); // identity transform
                }
            }
        }
    }
    Ok(raw)
}

fn state_space_error(error: CoreStateSpaceError) -> PyErr {
    StateSpaceError::new_err(error.to_string())
}

fn state_space_matrix(name: &str, value: PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize)> {
    let shape = value.shape();
    if shape[0] != shape[1] {
        return Err(StateSpaceError::new_err(format!(
            "invalid dimension: {name} must be a square matrix"
        )));
    }
    Ok((value.as_array().iter().copied().collect(), shape[0]))
}

fn state_space_vector(value: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    value.as_array().iter().copied().collect()
}

fn state_means_array<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
    dimension: usize,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((values.len(), dimension), |(time, state)| {
        values[time][state]
    })
    .into_pyarray(py)
}

fn state_covariances_array<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
    dimension: usize,
) -> Bound<'py, PyArray3<f64>> {
    Array3::from_shape_fn(
        (values.len(), dimension, dimension),
        |(time, row, column)| values[time][row * dimension + column],
    )
    .into_pyarray(py)
}

/// A time-homogeneous linear Gaussian state-space model with scalar observations.
/// Initial moments describe the state immediately before the first observation;
/// filtering performs one prediction before updating on observations[0].
#[pyclass(name = "LinearGaussianStateSpace")]
#[derive(Clone)]
struct PyLinearGaussianStateSpace {
    inner: CoreLinearGaussianStateSpace,
}

#[pymethods]
impl PyLinearGaussianStateSpace {
    #[new]
    #[pyo3(signature = (transition, observation, process_covariance, observation_variance, initial_mean, initial_covariance))]
    fn new(
        transition: PyReadonlyArray2<'_, f64>,
        observation: PyReadonlyArray1<'_, f64>,
        process_covariance: PyReadonlyArray2<'_, f64>,
        observation_variance: f64,
        initial_mean: PyReadonlyArray1<'_, f64>,
        initial_covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let (transition, dimension) = state_space_matrix("transition", transition)?;
        let (process_covariance, process_dimension) =
            state_space_matrix("process_covariance", process_covariance)?;
        let (initial_covariance, initial_dimension) =
            state_space_matrix("initial_covariance", initial_covariance)?;
        if process_dimension != dimension || initial_dimension != dimension {
            return Err(StateSpaceError::new_err(
                "invalid dimension: covariance matrices must match the transition matrix",
            ));
        }
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::new(
                dimension,
                transition,
                state_space_vector(observation),
                process_covariance,
                observation_variance,
                state_space_vector(initial_mean),
                initial_covariance,
            )
            .map_err(state_space_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (process_variance, observation_variance, initial_mean=0.0, initial_variance=1.0))]
    fn local_level(
        process_variance: f64,
        observation_variance: f64,
        initial_mean: f64,
        initial_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::local_level(
                process_variance,
                observation_variance,
                initial_mean,
                initial_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (level_variance, trend_variance, observation_variance, initial_level=0.0, initial_trend=0.0, initial_level_variance=1.0, initial_trend_variance=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn local_linear_trend(
        level_variance: f64,
        trend_variance: f64,
        observation_variance: f64,
        initial_level: f64,
        initial_trend: f64,
        initial_level_variance: f64,
        initial_trend_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::local_linear_trend(
                level_variance,
                trend_variance,
                observation_variance,
                initial_level,
                initial_trend,
                initial_level_variance,
                initial_trend_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    /// Construct a zero-mean stationary AR(1) latent process observed with
    /// independent Gaussian noise.
    #[staticmethod]
    fn stationary_ar1(
        coefficient: f64,
        process_variance: f64,
        observation_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLinearGaussianStateSpace::stationary_ar1(
                coefficient,
                process_variance,
                observation_variance,
            )
            .map_err(state_space_error)?,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn filter(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyKalmanFilterResult> {
        let observations = state_space_vector(observations);
        let result = py
            .allow_threads(|| self.inner.filter(&observations))
            .map_err(state_space_error)?;
        Ok(PyKalmanFilterResult::new(result, self.inner.dimension()))
    }

    fn smooth(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyKalmanSmootherResult> {
        let observations = state_space_vector(observations);
        let result = py
            .allow_threads(|| self.inner.smooth(&observations))
            .map_err(state_space_error)?;
        Ok(PyKalmanSmootherResult::new(result, self.inner.dimension()))
    }

    fn forecast(
        &self,
        py: Python<'_>,
        observations: PyReadonlyArray1<'_, f64>,
        steps: usize,
    ) -> PyResult<PyForecastResult> {
        let observations = state_space_vector(observations);
        let result = py
            .allow_threads(|| self.inner.forecast(&observations, steps))
            .map_err(state_space_error)?;
        Ok(PyForecastResult::new(result, self.inner.dimension()))
    }
}

#[pyclass(name = "KalmanFilterResult")]
struct PyKalmanFilterResult {
    inner: CoreKalmanFilterResult,
    dimension: usize,
}

impl PyKalmanFilterResult {
    fn new(inner: CoreKalmanFilterResult, dimension: usize) -> Self {
        Self { inner, dimension }
    }
}

#[pymethods]
impl PyKalmanFilterResult {
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    #[getter]
    fn predicted_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.predicted_means, self.dimension)
    }

    #[getter]
    fn predicted_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.predicted_covariances, self.dimension)
    }

    #[getter]
    fn filtered_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.filtered_means, self.dimension)
    }

    #[getter]
    fn filtered_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.filtered_covariances, self.dimension)
    }
}

#[pyclass(name = "KalmanSmootherResult")]
struct PyKalmanSmootherResult {
    inner: CoreKalmanSmootherResult,
    dimension: usize,
}

impl PyKalmanSmootherResult {
    fn new(inner: CoreKalmanSmootherResult, dimension: usize) -> Self {
        Self { inner, dimension }
    }
}

#[pymethods]
impl PyKalmanSmootherResult {
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.filter.log_likelihood
    }

    #[getter]
    fn filtered_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.filter.filtered_means, self.dimension)
    }

    #[getter]
    fn filtered_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.filter.filtered_covariances, self.dimension)
    }

    #[getter]
    fn smoothed_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.smoothed_means, self.dimension)
    }

    #[getter]
    fn smoothed_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.smoothed_covariances, self.dimension)
    }
}

#[pyclass(name = "ForecastResult")]
struct PyForecastResult {
    inner: CoreForecastResult,
    dimension: usize,
}

impl PyForecastResult {
    fn new(inner: CoreForecastResult, dimension: usize) -> Self {
        Self { inner, dimension }
    }
}

#[pymethods]
impl PyForecastResult {
    #[getter]
    fn state_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        state_means_array(py, &self.inner.state_means, self.dimension)
    }

    #[getter]
    fn state_covariances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        state_covariances_array(py, &self.inner.state_covariances, self.dimension)
    }

    #[getter]
    fn observation_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.observation_means.clone().into_pyarray(py)
    }

    #[getter]
    fn observation_variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.observation_variances.clone().into_pyarray(py)
    }

    /// Pointwise Gaussian predictive interval conditional on the fixed model
    /// parameters. This does not include parameter-estimation uncertainty.
    #[pyo3(signature = (level=0.95))]
    fn interval<'py>(&self, py: Python<'py>, level: f64) -> PyResult<PyIntervalArrays<'py>> {
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(PyValueError::new_err(
                "level must be finite and strictly between 0 and 1",
            ));
        }
        let critical = inv_normal_cdf(0.5 + level / 2.0);
        let mut lower = Vec::with_capacity(self.inner.observation_means.len());
        let mut upper = Vec::with_capacity(self.inner.observation_means.len());
        for (&mean, &variance) in self
            .inner
            .observation_means
            .iter()
            .zip(&self.inner.observation_variances)
        {
            let half_width = critical * variance.sqrt();
            lower.push(mean - half_width);
            upper.push(mean + half_width);
        }
        Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
    }

    #[getter]
    fn uncertainty_kind(&self) -> &'static str {
        "conditional_fixed_parameters"
    }
}

#[pymodule]
fn rustmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<ModelBuilder>()?;
    m.add_class::<ModelSpec>()?;
    m.add_class::<ParamRef>()?;
    m.add_class::<VectorParamRef>()?;
    m.add_class::<Expr>()?;
    m.add_class::<FitResult>()?;
    m.add_class::<BatchResult>()?;
    m.add_class::<PyCompiledModel>()?;
    m.add_class::<PyBoundModel>()?;
    m.add_class::<PyBatchFit>()?;
    m.add_class::<PyLinearGaussianStateSpace>()?;
    m.add_class::<PyKalmanFilterResult>()?;
    m.add_class::<PyKalmanSmootherResult>()?;
    m.add_class::<PyForecastResult>()?;
    m.add("ParameterError", m.py().get_type::<ParameterError>())?;
    m.add("StateSpaceError", m.py().get_type::<StateSpaceError>())?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_prior_predictive, m)?)?;
    Ok(())
}
