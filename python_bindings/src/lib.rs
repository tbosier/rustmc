use rustmc_core::model::{
    prior_name, CompiledDefinition as CompiledPythonModel, DisplayParamSpec, HyperParam,
    LikelihoodFamily, LikelihoodSpec, MuExpr, PriorSpec, SigmaSpec,
};
fn model_error(error: rustmc_core::model::ModelError) -> PyErr {
    match error {
        rustmc_core::model::ModelError::Invalid(s) => PyValueError::new_err(s),
        rustmc_core::model::ModelError::Parameter(s) => ParameterError::new_err(s),
    }
}
mod generic_results;
use generic_results::StoredBatchFit;
mod fit_artifact;
mod model_artifact;
mod prediction_binding;
use prediction_binding::prediction_graph;
mod expressions;
use expressions::{extract_expr, first_param_name, Expr, ParamRef, VectorParamRef};
mod ar;
mod dynamic_glm;
mod forecast_batch;
mod forecast_diagnostics;
mod forecast_support;
mod hierarchical;
mod hurdle;
mod local_level;
mod regression;
mod runoff;
mod seasonal;
mod state_space;
mod structural;
mod trend;
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::{DataBinding as CoreDataBinding, DataInputs, MatrixBinding};
use rustmc_core::graph::{Graph, ParamTransform};
use rustmc_core::param_ref::{validate_param_references, ParamRefError, ParamReference};
use rustmc_core::sampler::{self, SampleResult, SamplerConfig, SamplerType};
use rustmc_core::seeding::{stream_seed, POSTERIOR_PREDICT_SEED_DOMAIN, PRIOR_PREDICT_SEED_DOMAIN};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

type Data1d = HashMap<String, Vec<f64>>;
type Data2d = HashMap<String, (Vec<f64>, usize, usize)>;

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

create_exception!(
    rustmc,
    InferenceError,
    PyValueError,
    "Raised when a fitted Bayesian model has invalid inputs or a numerical failure."
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

#[pyclass(module = "rustmc")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
struct ModelSpec(rustmc_core::model::ModelSpec);
impl std::ops::Deref for ModelSpec {
    type Target = rustmc_core::model::ModelSpec;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for ModelSpec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ModelSpec {
    fn structure_definition(&self) -> Self {
        let mut definition = self.clone();
        definition.bound_data_1d.clear();
        definition.bound_data_2d.clear();
        definition
    }
}

#[pyclass(name = "BoundModel", module = "rustmc")]
#[derive(Clone)]
struct PyBoundModel {
    structure: Arc<Graph>,
    binding: CoreDataBinding,
}

#[pyclass(name = "CompiledModel", module = "rustmc")]
#[derive(Clone)]
struct PyCompiledModel {
    definition: ModelSpec,
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
    fn bind_any(&self, value: &Bound<'_, PyAny>, id: String) -> PyResult<CoreDataBinding> {
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
        let (extra_1d, extra_2d) = parse_data_dict(dict)?;
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
    rustmc_core::model::template_data_for_spec(&spec.0).map_err(model_error)
}

#[pyclass(module = "rustmc")]
#[derive(Debug, Clone)]
struct ModelBuilder {
    dimensions: HashMap<String, String>,
    potentials: Vec<(String, MuExpr)>,
    deterministics: Vec<(String, MuExpr)>,
    id: u64,
    priors: Vec<PriorSpec>,
    likelihoods: Vec<LikelihoodSpec>,
    bound_data_1d: HashMap<String, Vec<f64>>,
    bound_data_2d: HashMap<String, (Vec<f64>, usize, usize)>,
}

/// Validate every parameter reference in a model up front, before any graph is
/// built. Fails loudly on unknown names, out-of-order hyperparameters and
/// duplicate declarations.
fn validate_model_references(priors: &[PriorSpec], likelihoods: &[LikelihoodSpec]) -> PyResult<()> {
    rustmc_core::model::validate_model_references(priors, likelihoods).map_err(model_error)
}

/// HMC and NUTS evolve a continuous Euclidean state.  Discrete latent
/// parameters therefore need marginalisation or a discrete transition kernel;
/// treating them as continuous values produces invalid posterior draws.
/// Keep these priors available for prior-predictive simulation, but reject
/// every posterior-sampling entry point until such a kernel exists.
fn reject_discrete_priors_for_gradient_sampling(priors: &[PriorSpec]) -> PyResult<()> {
    rustmc_core::model::reject_discrete_priors_for_gradient_sampling(priors).map_err(model_error)
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

    /// Parse a likelihood predictor argument (`Expr`, bare `ParamRef`, bare
    /// data key or constant), rejecting references that belong to a different
    /// model.
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
        } else if let Ok(data_key) = value.extract::<String>() {
            // A bare "x" is the data column keyed x, as everywhere else in the
            // DSL. Checked before `f64`, as in `extract_expr`, so that a string
            // is never coerced to a number. Unowned, like a constant: it names
            // no parameter, so it means the same thing in any model.
            (MuExpr::Data(data_key), None)
        } else if let Ok(value) = value.extract::<f64>() {
            validate_finite(arg_name, value)?;
            (MuExpr::Const(value), None)
        } else {
            return Err(PyValueError::new_err(format!(
                "{} must be an Expr (e.g. beta * 'x'), a ParamRef, or a data \
                 key string naming one column (e.g. 'x')",
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

    /// Reject a data key this builder's bound data does not carry.
    ///
    /// The likelihood families do this through `validate_data_keys`, which
    /// also checks the observed key. Potentials and deterministics have no
    /// observed key, so they get the expression half on its own.
    ///
    /// With nothing bound there is nothing to check against, and the key is
    /// named later at bind time -- the same deferral the likelihood path makes.
    fn check_data_keys(&self, expr: &MuExpr) -> PyResult<()> {
        if self.bound_data_1d.is_empty() && self.bound_data_2d.is_empty() {
            return Ok(());
        }
        validate_expr_keys(expr, &self.bound_data_1d, &self.bound_data_2d)
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
    #[pyo3(signature = (data=None, dims=None))]
    fn new(
        data: Option<&Bound<'_, PyDict>>,
        dims: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let (bound_data_1d, bound_data_2d) = match data {
            Some(d) => parse_data_dict(d)?,
            None => (HashMap::new(), HashMap::new()),
        };
        Ok(Self {
            dimensions: dims.unwrap_or_default(),
            potentials: Vec::new(),
            deterministics: Vec::new(),
            id: next_model_id(),
            priors: Vec::new(),
            likelihoods: Vec::new(),
            bound_data_1d,
            bound_data_2d,
        })
    }

    /// Declare a numeric data expression and its observation dimension.
    #[pyo3(signature = (name, dim=None))]
    fn data(&mut self, name: &str, dim: Option<&str>) -> Expr {
        if let Some(dim) = dim {
            self.dimensions.insert(name.into(), dim.into());
        }
        Expr {
            inner: MuExpr::Data(name.into()),
            owner: Some(self.id),
        }
    }
    /// Add a scalar custom log-density term; reduce vector expressions with sum().
    fn potential(&mut self, name: &str, expression: &Bound<'_, PyAny>) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyValueError::new_err("potential name must not be empty"));
        }
        let expr = extract_expr(expression)?;
        self.check_owner(expr.owner, &first_param_name(&expr.inner), "potential")?;
        self.check_data_keys(&expr.inner)?;
        if !expr.inner.is_scalar() {
            return Err(PyValueError::new_err(
                "potential requires a scalar expression; use .sum()",
            ));
        }
        if self.potentials.iter().any(|(n, _)| n == name) {
            return Err(PyValueError::new_err("duplicate potential name"));
        }
        self.potentials.push((name.into(), expr.inner));
        Ok(())
    }
    /// Record a named scalar or vector expression at every posterior draw.
    fn deterministic(&mut self, name: &str, expression: &Bound<'_, PyAny>) -> PyResult<Expr> {
        let expr = extract_expr(expression)?;
        self.check_owner(expr.owner, &first_param_name(&expr.inner), "deterministic")?;
        self.check_data_keys(&expr.inner)?;
        if name.is_empty()
            || self.deterministics.iter().any(|(n, _)| n == name)
            || self.priors.iter().any(|p| prior_name(p) == name)
            || self.likelihoods.iter().any(|l| l.name == name)
        {
            return Err(PyValueError::new_err("deterministic name must be unique"));
        }
        self.deterministics.push((name.into(), expr.inner.clone()));
        Ok(expr)
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
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, sigma))]
    fn half_normal_prior(&mut self, name: &str, sigma: &Bound<'_, PyAny>) -> PyResult<ParamRef> {
        let sigma_hp = self.hyper_arg(sigma, "sigma", name)?;
        self.priors.push(PriorSpec::HalfNormal {
            name: name.to_string(),
            sigma: sigma_hp,
        });
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, rate))]
    fn exponential_prior(&mut self, name: &str, rate: &Bound<'_, PyAny>) -> PyResult<ParamRef> {
        let rate_hp = self.hyper_arg(rate, "rate", name)?;
        self.priors.push(PriorSpec::Exponential {
            name: name.to_string(),
            rate: rate_hp,
        });
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
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, lower=0.0, upper=1.0))]
    fn uniform_prior(&mut self, name: &str, lower: f64, upper: f64) -> PyResult<ParamRef> {
        validate_finite("lower", lower)?;
        validate_finite("upper", upper)?;
        if lower >= upper {
            return Err(PyValueError::new_err("lower must be less than upper"));
        }
        validate_positive_finite("uniform width", upper - lower)?;
        self.priors.push(PriorSpec::Uniform {
            name: name.to_string(),
            lower,
            upper,
        });
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
        Ok(self.param_ref(name))
    }

    #[pyo3(signature = (name, lam))]
    fn poisson_prior(&mut self, name: &str, lam: f64) -> PyResult<ParamRef> {
        validate_positive_finite("lam", lam)?;
        self.priors.push(PriorSpec::Poisson {
            name: name.to_string(),
            lam,
        });
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
        Ok(ModelSpec(rustmc_core::model::ModelSpec {
            dimensions: self.dimensions.clone(),
            potentials: self.potentials.clone(),
            deterministics: self.deterministics.clone(),
            priors: self.priors.clone(),
            likelihoods: self.likelihoods.clone(),
            bound_data_1d: self.bound_data_1d.clone(),
            bound_data_2d: self.bound_data_2d.clone(),
        }))
    }

    /// Compile immutable model structure once. Dataset payloads supplied here
    /// are used only to establish structural matrix widths and as bind defaults.
    fn compile(&self) -> PyResult<PyCompiledModel> {
        let spec = self.build()?;
        reject_discrete_priors_for_gradient_sampling(&spec.priors)?;
        let (template_1d, template_2d) = template_data_for_spec(&spec)?;
        validate_matrix_storage(&template_2d)?;
        let compiled = compile_python_model(&spec, &template_1d, &template_2d)?;
        let mut definition = spec;
        definition.bound_data_1d.clear();
        definition.bound_data_2d.clear();
        Ok(PyCompiledModel {
            definition,
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
        // Core matrices use row-major storage. NumPy's slice API also accepts
        // Fortran-contiguous arrays, so dtype alone cannot justify bypassing
        // this normalization. It also accepts strided views and Python lists.
        let value = data
            .py()
            .import("numpy")?
            .call_method1("ascontiguousarray", (&value, "float64"))?;
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

/// Adapters over the core validators. The rule and its wording live in
/// `rustmc_core::model` so the Python surface and the Rust core cannot drift.
fn validate_finite(name: &str, value: f64) -> PyResult<()> {
    rustmc_core::model::validate_finite(name, value).map_err(model_error)
}

fn validate_positive_finite(name: &str, value: f64) -> PyResult<()> {
    rustmc_core::model::validate_positive_finite(name, value).map_err(model_error)
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

/// Validate that each matrix's row-major storage matches its shape.
fn validate_matrix_storage(data_2d: &HashMap<String, (Vec<f64>, usize, usize)>) -> PyResult<()> {
    for (key, (values, rows, cols)) in data_2d {
        if rows.checked_mul(*cols) != Some(values.len()) {
            return Err(PyValueError::new_err(format!(
                "invalid matrix shape for {key}"
            )));
        }
    }
    Ok(())
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
        MuExpr::Unary(_, a) | MuExpr::Sum(a) => validate_expr_keys(a, data_1d, data_2d),
        MuExpr::ParamTimesData { data_key, .. }
        | MuExpr::Data(data_key)
        | MuExpr::Gather { data_key, .. } => {
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
        MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
            validate_expr_keys(a, data_1d, data_2d)?;
            validate_expr_keys(b, data_1d, data_2d)
        }
    }
}

fn logit_stable(p: f64) -> f64 {
    rustmc_core::prior_sampling::logit_stable(p)
}

fn invert_param_transform(transform: &ParamTransform, value: f64) -> f64 {
    match transform {
        ParamTransform::Identity => value,
        ParamTransform::Exp => value.ln(),
        ParamTransform::Sigmoid => logit_stable(value),
        ParamTransform::BoundedSigmoid { lower, upper } => {
            let span = upper - lower;
            logit_stable((value - lower) / span)
        }
    }
}

fn constrained_draw_to_raw(draw: &[f64], transforms: &[ParamTransform]) -> Vec<f64> {
    draw.iter()
        .zip(transforms.iter())
        .map(|(&value, transform)| invert_param_transform(transform, value))
        .collect()
}

/// Prefer the sampler's exact position: constrained values may round to a
/// transform boundary, making their inverse infinite or otherwise lossy.
fn posterior_position<'a>(
    result: &'a SampleResult,
    graph: &Graph,
    chain: usize,
    draw: usize,
) -> std::borrow::Cow<'a, [f64]> {
    if let Some(positions) = &result.unconstrained_samples {
        std::borrow::Cow::Borrowed(&positions[chain][draw])
    } else {
        std::borrow::Cow::Owned(constrained_draw_to_raw(
            &result.samples[chain][draw],
            &graph.param_transforms,
        ))
    }
}

fn pointwise_log_likelihood_for_draw(
    graph: &Graph,
    raw_draw: &[f64],
    heads: &[rustmc_core::graph::ObservationHead],
) -> PyResult<Vec<Vec<f64>>> {
    let mut evaluator =
        Evaluator::try_new(graph).map_err(|error| PyValueError::new_err(error.to_string()))?;
    evaluator.forward(graph, raw_draw);

    heads
        .iter()
        .map(|head| {
            let aux = head.aux.map(|node| evaluator.scalar_at(node));
            graph.obs_vectors[head.obs_data_idx]
                .iter()
                .enumerate()
                .map(|(i, &observed)| {
                    rustmc_core::observation::log_density(
                        head.family,
                        observed,
                        evaluator.vec_elem(head.linpred, i, graph),
                        aux,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))
                })
                .collect()
        })
        .collect()
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
    spec: &ModelSpec,
    data: &Data1d,
    matrices: &Data2d,
) -> PyResult<CompiledPythonModel> {
    rustmc_core::model::compile(&spec.0, data, matrices).map_err(model_error)
}

fn derive_display_draw(draw: &[f64], specs: &[DisplayParamSpec]) -> PyResult<Vec<f64>> {
    rustmc_core::model::derive_display_draw(draw, specs).map_err(model_error)
}

fn derive_display_sample_result(
    raw_result: &SampleResult,
    specs: &[DisplayParamSpec],
) -> PyResult<SampleResult> {
    rustmc_core::model::derive_display_sample_result(raw_result, specs).map_err(model_error)
}

/// True when the display layer is a pure pass-through of the raw draws: every
/// parameter is reported as sampled, in the order it was sampled.
fn display_specs_are_identity(raw_result: &SampleResult, specs: &[DisplayParamSpec]) -> bool {
    specs.len() == raw_result.param_names.len()
        && specs.iter().enumerate().all(|(index, spec)| match spec {
            DisplayParamSpec::Raw { name, raw_index } => {
                *raw_index == index && *name == raw_result.param_names[index]
            }
            DisplayParamSpec::DerivedNonCenteredNormal { .. } => false,
        })
}

/// Display draws for a fit, sharing the raw posterior when nothing is derived.
///
/// `derive_display_sample_result` allocates a second copy of every draw. When
/// no parameter is non-centred, that copy is bit-identical to the raw draws, so
/// a retained batch cell paid for two posteriors to hold one. Sharing the `Arc`
/// keeps the display and raw views distinguishable without duplicating them.
fn display_sample_result(
    raw_result: &Arc<SampleResult>,
    specs: &[DisplayParamSpec],
) -> PyResult<Arc<SampleResult>> {
    if display_specs_are_identity(raw_result, specs) {
        // The copying path rejects nonfinite display values; run the same check
        // so sharing can never accept a fit that copying would have refused.
        for chain in &raw_result.samples {
            for draw in chain {
                if draw.iter().any(|value| !value.is_finite()) {
                    derive_display_draw(draw, specs)?;
                }
            }
        }
        return Ok(Arc::clone(raw_result));
    }
    Ok(Arc::new(derive_display_sample_result(raw_result, specs)?))
}

fn validate_sample_config(
    chains: usize,
    draws: usize,
    warmup: usize,
    step_size: f64,
    target_accept: f64,
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
    if !target_accept.is_finite() || target_accept <= 0.0 || target_accept >= 1.0 {
        return Err(PyValueError::new_err(
            "target_accept must be finite and strictly between 0 and 1",
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

#[pyclass(module = "rustmc")]
#[derive(Clone)]
struct FitResult {
    definition: ModelSpec,
    /// The posterior draws dominate a fit's memory, so both views are shared
    /// handles: cloning a `FitResult` never duplicates them, and when no
    /// parameter is derived the two point at the same allocation.
    raw_result: Arc<SampleResult>,
    display_result: Arc<SampleResult>,
    /// A clone of the compiled graph — used for predictive sampling.
    graph: Graph,
    /// Name of each likelihood, in the order they appear in the graph.
    likelihood_names: Vec<String>,
}

impl FitResult {
    /// Forward-simulate the observation model at the given `(chain, draw)`
    /// posterior coordinates.
    ///
    /// Returns one flat `coordinates.len() * n_obs` vector per likelihood, in
    /// the order the coordinates were supplied.
    fn simulate_predictive(
        &self,
        graph: &Graph,
        heads: &[rustmc_core::graph::ObservationHead],
        coordinates: &[(usize, usize)],
        expected: bool,
        rng: &mut ChaCha8Rng,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mut evaluator =
            Evaluator::try_new(graph).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut preds: Vec<Vec<f64>> = heads
            .iter()
            .map(|head| Vec::with_capacity(coordinates.len() * head.n_obs))
            .collect();

        for &(chain_idx, draw_idx) in coordinates {
            let position = posterior_position(&self.raw_result, graph, chain_idx, draw_idx);
            evaluator.forward(graph, &position);
            for (li, head) in heads.iter().enumerate() {
                for i in 0..head.n_obs {
                    let eta = evaluator.vec_elem(head.linpred, i, graph);
                    let aux = head.aux.map(|node| evaluator.scalar_at(node));
                    preds[li].push(
                        if expected {
                            rustmc_core::observation::mean(head.family, eta, aux)
                        } else {
                            rustmc_core::observation::sample(head.family, eta, aux, rng)
                        }
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                    );
                }
            }
        }
        Ok(preds)
    }

    /// Posterior-predictive draws laid out on the posterior's own
    /// `(chain, draw, obs)` grid, so every predictive draw stays paired with the
    /// parameter draw that produced it.
    ///
    /// When `n_samples` asks for fewer draws than were sampled, the thinning is
    /// chain-stratified: one shared set of per-chain draw indices is retained in
    /// every chain. That keeps the exported block rectangular (ArviZ groups must
    /// be dense arrays), represents every chain equally, and leaves a single
    /// `draw` coordinate vector that identifies exactly which posterior draws
    /// were kept. The second return value holds those retained draw indices, or
    /// `None` when nothing was thinned away.
    fn posterior_predictive_grid<'py>(
        &self,
        py: Python<'py>,
        n_samples: Option<usize>,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyDict>, Option<Vec<i64>>)> {
        let graph = prediction_graph(&self.graph, None, None)?;
        graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, POSTERIOR_PREDICT_SEED_DOMAIN));
        let heads = graph.observation_heads();

        let n_chains = self.raw_result.samples.len();
        let n_draws = self.raw_result.samples.first().map_or(0, Vec::len);
        let per_chain = match n_samples {
            // A request of fewer draws than there are chains still keeps one
            // draw per chain: dropping whole chains would be worse than
            // overshooting the budget by a handful of draws.
            Some(requested) if n_chains > 0 => (requested / n_chains).max(1).min(n_draws),
            _ => n_draws,
        };
        let retained = select_posterior_draw_indices(n_draws, Some(per_chain), &mut rng);

        let coordinates: Vec<(usize, usize)> = (0..n_chains)
            .flat_map(|chain_idx| retained.iter().map(move |&draw_idx| (chain_idx, draw_idx)))
            .collect();
        let mut preds = self.simulate_predictive(&graph, &heads, &coordinates, false, &mut rng)?;

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            let n_obs = heads[li].n_obs;
            let arr = Array3::from_shape_vec(
                (n_chains, retained.len(), n_obs),
                std::mem::take(&mut preds[li]),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            dict.set_item(name, arr.into_pyarray(py))?;
        }

        let thinned = retained.len() < n_draws;
        Ok((
            dict,
            thinned.then(|| retained.iter().map(|&index| index as i64).collect()),
        ))
    }
}

#[pymethods]
impl FitResult {
    /// Versioned JSON including bound training data, stored graph draws, and sampler telemetry.
    fn to_json(&self) -> PyResult<String> {
        fit_artifact::encode(self)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        fit_artifact::decode(text)
    }
    /// Declarative compiled model with the fitted training data available as bind defaults.
    #[getter]
    fn model(&self) -> PyResult<PyCompiledModel> {
        fit_artifact::model(self)
    }
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
        generic_results::diagnostics(&self.display_result, py)
    }

    /// Structured sampler telemetry, including integrator work and tree depth.
    fn transition_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::transition_diagnostics(&self.raw_result, py)
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

    /// Prediction preserving (chain, draw, observation) axes.
    #[pyo3(signature = (data=None, seed=42, expected=false, sizes=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        seed: u64,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let flat = self.posterior_predictive(py, None, seed, data, expected, sizes)?;
        let result = PyDict::new(py);
        let chains = self.raw_result.samples.len();
        let draws = self.raw_result.samples.first().map_or(0, Vec::len);
        for (name, value) in flat.iter() {
            let arr = value.downcast::<PyArray2<f64>>()?;
            let n = arr.shape()[1];
            let values = unsafe { arr.as_slice()? }.to_vec();
            result.set_item(
                name,
                Array3::from_shape_vec((chains, draws, n), values)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?;
        }
        Ok(result)
    }
    /// Named deterministic draws, with (chain, draw[, observation]) axes.
    #[pyo3(signature = (data=None, sizes=None))]
    fn deterministics<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let graph = prediction_graph(&self.graph, data, sizes)?;
        let mut evaluator =
            Evaluator::try_new(&graph).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let chains = self.raw_result.samples.len();
        let draws = self.raw_result.samples.first().map_or(0, Vec::len);
        let result = PyDict::new(py);
        for (name, node) in &graph.deterministics {
            let n = evaluator.node_len(*node);
            let mut values = Vec::with_capacity(chains * draws * n.max(1));
            for (chain_idx, chain) in self.raw_result.samples.iter().enumerate() {
                for draw_idx in 0..chain.len() {
                    let position =
                        posterior_position(&self.raw_result, &graph, chain_idx, draw_idx);
                    evaluator.forward(&graph, &position);
                    // Same standard the prior predictive holds deterministics
                    // to, and the same one `sampler` holds the parameters to:
                    // a nonfinite value is a failed computation, not a result.
                    for i in 0..n.max(1) {
                        let value = evaluator.vec_elem(*node, i, &graph);
                        if !value.is_finite() {
                            return Err(PyValueError::new_err(format!(
                                "deterministic '{name}' is nonfinite at chain {chain_idx}, \
                                 draw {draw_idx}"
                            )));
                        }
                        values.push(value);
                    }
                }
            }
            if n == 0 {
                result.set_item(
                    name,
                    Array2::from_shape_vec((chains, draws), values)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
            } else {
                result.set_item(
                    name,
                    Array3::from_shape_vec((chains, draws, n), values)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
            }
        }
        Ok(result)
    }
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("kernel", "graph_mcmc")?;
        d.set_item("chains", self.raw_result.samples.len())?;
        d.set_item("draws", self.raw_result.samples.first().map_or(0, Vec::len))?;
        d.set_item("prediction_axes", ("chain", "draw", "observation"))?;
        let dimensions = PyDict::new(py);
        for (slot, obs) in self
            .graph
            .schema
            .observations
            .iter()
            .zip(&self.graph.obs_vectors)
        {
            dimensions.set_item(&slot.dim, obs.len())?;
        }
        d.set_item("dimensions", dimensions)?;
        Ok(d)
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
    #[pyo3(signature = (n_samples=None, seed=42, data=None, expected=false, sizes=None))]
    fn posterior_predictive<'py>(
        &self,
        py: Python<'py>,
        n_samples: Option<usize>,
        seed: u64,
        data: Option<&Bound<'_, PyDict>>,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let graph = prediction_graph(&self.graph, data, sizes)?;
        let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, POSTERIOR_PREDICT_SEED_DOMAIN));
        graph
            .validate_shapes()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let heads = graph.observation_heads();

        // Flatten all chain draws in order, then subsample without replacement
        // when the caller requests fewer draws than are available.
        let all_draws: Vec<(usize, usize)> = self
            .raw_result
            .samples
            .iter()
            .enumerate()
            .flat_map(|(chain_idx, chain)| {
                (0..chain.len()).map(move |draw_idx| (chain_idx, draw_idx))
            })
            .collect();
        let chosen_indices = select_posterior_draw_indices(all_draws.len(), n_samples, &mut rng);
        let n = chosen_indices.len();
        let coordinates: Vec<(usize, usize)> = chosen_indices
            .into_iter()
            .map(|index| all_draws[index])
            .collect();

        let mut preds =
            self.simulate_predictive(&graph, &heads, &coordinates, expected, &mut rng)?;

        let dict = PyDict::new(py);
        for (li, name) in self.likelihood_names.iter().enumerate() {
            let n_obs = heads[li].n_obs;
            let arr = Array2::from_shape_vec((n, n_obs), std::mem::take(&mut preds[li]))
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
            for draw_idx in 0..chain.len() {
                let position =
                    posterior_position(&self.raw_result, &self.graph, chain_idx, draw_idx);
                let per_head = pointwise_log_likelihood_for_draw(&self.graph, &position, &heads)?;
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

    /// Convert to ArviZ's version-native inference container.
    ///
    /// Requires ArviZ: `pip install arviz`
    ///
    /// Returns an `arviz.InferenceData` on ArviZ 0.x or an `xarray.DataTree`
    /// on ArviZ 1.x, with:
    ///   - `posterior`             — (n_chains × n_draws) arrays for every parameter
    ///   - `sample_stats`          — `diverging` (bool) and `step_size` per draw
    ///   - `observed_data`         — the fitted response vector for each likelihood
    ///   - `log_likelihood`        — (n_chains × n_draws × n_obs) pointwise values
    ///   - `posterior_predictive`  — ŷ samples (only when include_ppc=True)
    ///
    /// `posterior_predictive` is exported on the posterior's own
    /// `(chain, draw, obs)` axes, so predictive draw `(c, d)` is the one
    /// generated from posterior draw `(c, d)`. LOO/PSIS and per-chain
    /// predictive diagnostics need that pairing.
    ///
    /// `ppc_samples` thins the draw axis rather than the flattened sample list:
    /// the same `ppc_samples // n_chains` draw indices are retained in every
    /// chain, and the `posterior_predictive` group's `draw` coordinate records
    /// which posterior draws they were, so
    /// `idata.posterior.sel(draw=idata.posterior_predictive.draw)` recovers the
    /// matching parameters. (Before this, `ppc_samples` subsampled a flattened
    /// pool and the export was collapsed to a single fake chain.)
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
        let groups = PyDict::new(py);
        groups.set_item("posterior", posterior)?;
        groups.set_item("sample_stats", sample_stats)?;

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
            groups.set_item("observed_data", observed_data)?;
        }

        if include_log_likelihood && !self.likelihood_names.is_empty() {
            let log_likelihood = self.log_likelihood(py)?;
            groups.set_item("log_likelihood", log_likelihood)?;
        }

        // Posterior-predictive draws keep the posterior's own (chain, draw)
        // axes so a consumer can pair a predictive draw with the parameters
        // that produced it. `retained_draws` is Some only when `ppc_samples`
        // thinned the draw axis, and then carries the kept draw indices.
        let mut retained_draws = None;
        if include_ppc && !self.likelihood_names.is_empty() {
            let (ppc_dict, retained) = self.posterior_predictive_grid(py, ppc_samples, ppc_seed)?;
            groups.set_item("posterior_predictive", ppc_dict)?;
            retained_draws = retained;
        }

        let arviz_major = arviz_api_generation(&az)?;
        let container = arviz_from_groups_versioned(&az, arviz_major, groups)?;
        if let Some(retained) = retained_draws {
            // Label the thinned axis with the posterior draw indices it came
            // from, so `posterior.sel(draw=ppc.draw)` lines the groups back up.
            assign_posterior_predictive_draw_coords(py, arviz_major, &container, &retained)?;
        }
        Ok(container)
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
#[pyo3(signature = (model_spec, data=None, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, target_accept=0.8, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true, init=None, metric="auto"))]
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
    target_accept: f64,
    sampler: &str,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
    show_progress: bool,
    init: Option<Vec<Vec<f64>>>,
    metric: &str,
) -> PyResult<FitResult> {
    let metric = parse_metric(metric)?;
    validate_sample_config(
        chains,
        draws,
        warmup,
        step_size,
        target_accept,
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

    validate_matrix_storage(&matrix_map)?;

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
        target_accept,
        num_leapfrog_steps,
        max_tree_depth,
        seed,
        num_threads: threads,
        show_progress,
        metric,
    };

    let graph_for_predict = compiled.graph.clone();

    let result = py
        .allow_threads(|| {
            sampler::sample_bound_with_init(
                Arc::new(compiled.graph.structure_only()),
                CoreDataBinding::from_graph(&compiled.graph).map_err(|e| e.to_string())?,
                config,
                init,
            )
        })
        .map_err(PyValueError::new_err)?;
    let raw_result = Arc::new(result);
    let display_result = display_sample_result(&raw_result, &compiled.display_params)?;

    Ok(FitResult {
        definition: model_spec.structure_definition(),
        raw_result,
        display_result,
        graph: graph_for_predict,
        likelihood_names: compiled.likelihood_names,
    })
}

/// Result for a single model in a batch run.
#[pymethods]
impl PyCompiledModel {
    /// Versioned declarative artifact; excludes all bound training data and defaults.
    fn to_json(&self) -> PyResult<String> {
        model_artifact::encode(self)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        model_artifact::decode(text)
    }

    /// Evaluate the native graph target and gradient in unconstrained coordinates.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        position: Vec<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        if position.len() != self.structure.param_count || position.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "position must be a finite vector matching the parameter dimension",
            ));
        }
        let binding = self.bind_any(data, "0".into())?;
        let mut evaluator = Evaluator::try_with_binding(&self.structure, binding)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        evaluator.compute(&self.structure, &position);
        Ok((evaluator.total_logp, evaluator.grad.into_pyarray(py)))
    }
    #[getter]
    fn dimensions(&self) -> HashMap<String, String> {
        self.structure
            .schema
            .observations
            .iter()
            .chain(&self.structure.schema.vectors)
            .chain(&self.structure.schema.matrices)
            .map(|s| (s.key.clone(), s.dim.clone()))
            .collect()
    }
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

    #[pyo3(signature = (data, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, target_accept=0.8, sampler="nuts", max_tree_depth=10, num_leapfrog_steps=15, show_progress=true, init=None, metric="auto"))]
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
        target_accept: f64,
        sampler: &str,
        max_tree_depth: usize,
        num_leapfrog_steps: usize,
        show_progress: bool,
        init: Option<Vec<Vec<f64>>>,
        metric: &str,
    ) -> PyResult<FitResult> {
        let metric = parse_metric(metric)?;
        validate_sample_config(
            chains,
            draws,
            warmup,
            step_size,
            target_accept,
            max_tree_depth,
            num_leapfrog_steps,
        )?;
        let binding = self.bind_any(data, "0".to_string())?;
        let sampler_type = parse_sampler_type(sampler)?;
        let config = SamplerConfig {
            sampler: sampler_type,
            num_chains: chains,
            num_draws: draws,
            num_warmup: warmup,
            step_size,
            target_accept,
            num_leapfrog_steps,
            max_tree_depth,
            seed,
            num_threads: threads,
            show_progress,
            metric,
        };
        let hydrated_graph = self.structure.with_binding(&binding);
        let result = py
            .allow_threads(|| {
                sampler::sample_bound_with_init(Arc::clone(&self.structure), binding, config, init)
            })
            .map_err(PyValueError::new_err)?;
        let raw_result = Arc::new(result);
        let display_result = display_sample_result(&raw_result, &self.display_params)?;
        Ok(FitResult {
            definition: self.definition.clone(),
            raw_result,
            display_result,
            graph: hydrated_graph,
            likelihood_names: self.likelihood_names.clone(),
        })
    }

    #[pyo3(signature = (datasets, ids=None, shared=None, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, target_accept=0.8, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true, threads=1, chunk_size=64, errors="raise", seed_policy="cell_id_v1", init=None, metric="auto"))]
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
        target_accept: f64,
        max_tree_depth: usize,
        num_leapfrog_steps: usize,
        show_progress: bool,
        threads: usize,
        chunk_size: usize,
        errors: &str,
        seed_policy: &str,
        init: Option<&Bound<'_, PyDict>>,
        metric: &str,
    ) -> PyResult<PyBatchFit> {
        let metric = parse_metric(metric)?;
        validate_sample_config(
            chains,
            draws,
            warmup,
            step_size,
            target_accept,
            max_tree_depth,
            num_leapfrog_steps,
        )?;
        let collect_errors = match errors {
            "raise" => false,
            "collect" => true,
            _ => return Err(PyValueError::new_err("errors must be 'raise' or 'collect'")),
        };
        let seed_policy = match seed_policy {
            "cell_id_v1" => sampler::BatchSeedPolicy::CellIdV1,
            "position_v0" => sampler::BatchSeedPolicy::PositionV0,
            _ => {
                return Err(PyValueError::new_err(
                    "seed_policy must be 'cell_id_v1' or 'position_v0'",
                ))
            }
        };
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
        let mut initial_positions = HashMap::new();
        let mut initial_errors = HashMap::new();
        if let Some(initial) = init {
            for (key, value) in initial.iter() {
                let id: String = key.extract()?;
                if !ids.contains(&id) {
                    return Err(PyValueError::new_err(format!(
                        "initialization supplied for unknown dataset ID '{id}'"
                    )));
                }
                match value.extract::<Vec<Vec<f64>>>() {
                    Ok(positions) => {
                        initial_positions.insert(id, positions);
                    }
                    Err(error) => {
                        initial_errors.insert(id, format!("invalid init: {error}"));
                    }
                }
            }
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
                if let Some(error) = initial_errors.get(id) {
                    return Err(PyValueError::new_err(error.clone()));
                }
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
            .map(|value| value.map_err(|error| error.to_string()))
            .collect::<Vec<_>>();
        let config = sampler::BatchSampleConfig {
            sampler: parse_sampler_type(sampler)?,
            num_chains: chains,
            num_draws: draws,
            num_warmup: warmup,
            step_size,
            target_accept,
            num_leapfrog_steps,
            max_tree_depth,
            seed,
            show_progress,
            metric,
        };
        let raw = py
            .allow_threads(|| {
                sampler::sample_batch_bound_with_initial(
                    Arc::clone(&self.structure),
                    ids.iter().cloned().zip(bindings.clone()).collect(),
                    config,
                    sampler::BoundBatchOptions {
                        threads,
                        chunk_size,
                        collect_errors,
                        seed_policy,
                    },
                    initial_positions,
                )
            })
            .map_err(PyValueError::new_err)?;
        let mut results = Vec::with_capacity(raw.len());
        for (item, binding) in raw.into_iter().zip(bindings) {
            results.push(match item {
                Err(error) => Err(error),
                Ok(raw_result) => {
                    let raw_result = Arc::new(raw_result);
                    let display_result = display_sample_result(&raw_result, &self.display_params)?;
                    let binding = binding.map_err(PyValueError::new_err)?;
                    Ok(BatchResult {
                        full_fit: StoredBatchFit::Bound(Arc::new(generic_results::BoundBatchFit {
                            structure: Arc::clone(&self.structure),
                            binding,
                            raw_result,
                            display_result,
                            likelihood_names: self.likelihood_names.clone(),
                            definition: self.definition.clone(),
                        })),
                    })
                }
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

fn parse_metric(metric: &str) -> PyResult<sampler::MetricKind> {
    sampler::MetricKind::parse(metric).map_err(PyValueError::new_err)
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

/// One cell of a batch run.
///
/// Everything this exposes is read off the retained fit, which every cell
/// has: both construction sites supply one, so the accessors are infallible.
/// The fit used to be optional, and the `None` arm manufactured an error for
/// a "legacy" cell that no code path could produce. It used to also hold a
/// flattened `BatchModelResult` copy of the display draws, which made a third
/// posterior per cell alongside the raw and display trees.
#[pyclass(module = "rustmc")]
#[derive(Clone)]
struct BatchResult {
    full_fit: StoredBatchFit,
}

impl BatchResult {
    /// Display draws for this cell.
    fn display(&self) -> &SampleResult {
        self.full_fit.display()
    }
}
#[pymethods]
impl BatchResult {
    /// Internal regression-test hook: compare immutable payload ownership without exposing addresses.
    fn _shares_data(&self, other: &BatchResult, key: &str) -> bool {
        match (&self.full_fit, &other.full_fit) {
            (StoredBatchFit::Bound(a), StoredBatchFit::Bound(b)) => {
                a.binding.shares_payload_with(&b.binding, key)
            }
            _ => false,
        }
    }
    /// Internal regression-test hook: how many distinct posterior sample trees
    /// this cell retains. One when the display layer passes the raw draws
    /// through unchanged, two when a parameter is genuinely derived.
    fn _posterior_allocations(&self) -> usize {
        if std::ptr::eq(self.full_fit.raw(), self.full_fit.display()) {
            1
        } else {
            2
        }
    }

    /// Internal regression-test hook: whether `fit` reuses this cell's retained
    /// posterior rather than holding a copy of it. Compares ownership without
    /// exposing addresses.
    fn _shares_posterior_with(&self, fit: &FitResult) -> bool {
        std::ptr::eq(self.full_fit.raw(), &*fit.raw_result)
            && std::ptr::eq(self.full_fit.display(), &*fit.display_result)
    }

    #[getter]
    fn fit(&self) -> FitResult {
        self.full_fit.materialize()
    }

    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        generic_results::diagnostics(self.full_fit.display(), py)
    }

    fn summary(&self) -> String {
        self.full_fit.display().diagnostics().to_table()
    }

    fn transition_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        generic_results::transition_diagnostics(self.full_fit.raw(), py)
    }

    #[pyo3(signature = (data=None, seed=42, expected=false, sizes=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'_, PyDict>>,
        seed: u64,
        expected: bool,
        sizes: Option<HashMap<String, usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.fit().predict(py, data, seed, expected, sizes)
    }

    fn get_samples_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let dict = PyDict::new(py);
        let n_chains = display.samples.len();
        let n_draws = display.samples.first().map_or(0, Vec::len);
        for (pidx, name) in display.param_names.iter().enumerate() {
            let mut arr = Array2::<f64>::zeros((n_chains, n_draws));
            for (chain_idx, chain) in display.samples.iter().enumerate() {
                for (draw_idx, draw) in chain.iter().enumerate() {
                    arr[[chain_idx, draw_idx]] = draw[pidx];
                }
            }
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let means = display.mean();
        let dict = PyDict::new(py);
        for (name, val) in display.param_names.iter().zip(means.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let stds = display.std();
        let dict = PyDict::new(py);
        for (name, val) in display.param_names.iter().zip(stds.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let display = self.display();
        let dict = PyDict::new(py);
        for (pidx, name) in display.param_names.iter().enumerate() {
            let vals: Vec<f64> = display
                .samples
                .iter()
                .flatten()
                .map(|draw| draw[pidx])
                .collect();
            let arr = PyArray1::from_vec(py, vals);
            dict.set_item(name, arr)?;
        }
        Ok(dict)
    }

    #[getter]
    fn chains(&self) -> usize {
        self.display().samples.len()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.display().samples.first().map_or(0, Vec::len)
    }

    #[getter]
    fn accept_rate(&self) -> f64 {
        let rates = &self.display().accept_rates;
        if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        }
    }

    #[getter]
    fn accept_rates(&self) -> Vec<f64> {
        self.display().accept_rates.clone()
    }

    #[getter]
    fn divergences(&self) -> usize {
        self.display().total_divergences()
    }

    #[getter]
    fn divergences_per_chain(&self) -> Vec<usize> {
        self.display().divergences.clone()
    }

    fn __repr__(&self) -> String {
        let display = self.display();
        let means = display.mean();
        let parts: Vec<String> = display
            .param_names
            .iter()
            .zip(means.iter())
            .map(|(n, m)| format!("{}={:.4}", n, m))
            .collect();
        format!(
            "BatchResult({} chains × {} draws, {})",
            display.samples.len(),
            display.samples.first().map_or(0, Vec::len),
            parts.join(", ")
        )
    }
}

#[pyclass(name = "BatchFit", module = "rustmc")]
struct PyBatchFit {
    ids: Vec<String>,
    results: Vec<Result<BatchResult, String>>,
}

#[pymethods]
impl PyBatchFit {
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }

    #[getter]
    fn errors(&self) -> HashMap<String, String> {
        self.ids
            .iter()
            .zip(&self.results)
            .filter_map(|(id, value)| {
                value
                    .as_ref()
                    .err()
                    .map(|error| (id.clone(), error.clone()))
            })
            .collect()
    }

    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Py<BatchResult>> {
        let index = self
            .ids
            .iter()
            .position(|value| value == id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown dataset ID '{id}'")))?;
        self.__getitem__(py, index as isize)
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
        let result = result.map_err(|error| {
            PyValueError::new_err(format!(
                "dataset '{}': {error}",
                self.ids[normalized as usize]
            ))
        })?;
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
        format!(
            "BatchFit({} datasets, {} failed)",
            self.results.len(),
            self.errors().len()
        )
    }
}

/// Run thousands of independent models in parallel through Rayon.
///
/// Each entry in `models` is a (ModelSpec, data_dict) pair. By default each gets
/// 1 NUTS chain for throughput, but the batch runner can be configured to use
/// multiple chains or fixed-step HMC when reliability matters more.
#[pyfunction]
#[pyo3(signature = (models, chains=1, draws=500, warmup=300, seed=42, sampler="nuts", step_size=0.0, target_accept=0.8, max_tree_depth=8, num_leapfrog_steps=15, show_progress=true, metric="auto"))]
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
    target_accept: f64,
    max_tree_depth: usize,
    num_leapfrog_steps: usize,
    show_progress: bool,
    metric: &str,
) -> PyResult<Vec<BatchResult>> {
    let metric = parse_metric(metric)?;
    validate_sample_config(
        chains,
        draws,
        warmup,
        step_size,
        target_accept,
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

        validate_matrix_storage(&matrix_map)?;

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
        target_accept,
        num_leapfrog_steps,
        max_tree_depth,
        seed,
        show_progress,
        metric,
    };

    let graphs: Vec<Graph> = compiled_models
        .iter()
        .map(|compiled| compiled.graph.clone())
        .collect();

    let results = py
        .allow_threads(|| sampler::batch_sample_graphs(graphs, config))
        .map_err(PyValueError::new_err)?;

    results
        .into_iter()
        .zip(compiled_models.iter())
        .zip(models.iter())
        .map(|((raw_result, compiled), (spec, _))| {
            let num_draws = raw_result.num_draws;
            let raw = Arc::new(SampleResult {
                samples: regroup_draws_by_chain(raw_result.samples, num_draws),
                unconstrained_samples: raw_result.unconstrained_samples,
                param_names: raw_result.param_names,
                accept_rates: raw_result.accept_rates,
                step_sizes: raw_result.step_sizes,
                divergences: raw_result.divergences,
                transitions: raw_result.transitions,
            });
            let display_result = display_sample_result(&raw, &compiled.display_params)?;
            Ok(BatchResult {
                full_fit: StoredBatchFit::Ready(Arc::new(FitResult {
                    raw_result: raw,
                    display_result,
                    graph: compiled.graph.clone(),
                    likelihood_names: compiled.likelihood_names.clone(),
                    definition: spec.borrow().structure_definition(),
                })),
            })
        })
        .collect()
}

/// Regroup a flat, chain-major draw list into per-chain blocks.
///
/// The draw buffers are moved rather than copied, so regrouping a batch result
/// does not duplicate the posterior. Draining the source is what keeps that
/// true: `split_off` would leave each chain holding the capacity of the whole
/// remaining suffix, which costs O(chains² × draws) descriptor slots.
fn regroup_draws_by_chain(flat: Vec<Vec<f64>>, num_draws: usize) -> Vec<Vec<Vec<f64>>> {
    if num_draws == 0 {
        return Vec::new();
    }
    let mut remaining = flat.into_iter();
    let mut chains = Vec::with_capacity(remaining.len().div_ceil(num_draws));
    loop {
        let chain: Vec<Vec<f64>> = remaining.by_ref().take(num_draws).collect();
        if chain.is_empty() {
            return chains;
        }
        chains.push(chain);
    }
}

/// Draw samples from the **prior predictive** distribution.
///
/// Samples parameters from the model priors using their analytic distributions,
/// then runs a forward pass to generate predicted observations.
/// Use this to check whether your priors make sense before fitting.
///
/// A likelihood is not required.  With none declared, the result carries the
/// prior draws of the parameters and of any deterministic, and no predicted
/// observations -- which is exactly what "check whether your priors make sense
/// before fitting" means for a model whose likelihood is not written yet.
/// Potentials *are* refused: a custom density term supplies no random
/// generator, so a model carrying one has no prior to simulate from.
///
/// Parameters
/// ----------
/// model_spec : ModelSpec
///     A model definition, from `builder.build()`.
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
    rustmc_core::model::reject_potentials_for_prior_predictive(&model_spec.potentials)
        .map_err(model_error)?;
    // ── Build data maps ───────────────────────────────────────────────────────
    let mut data_map: HashMap<String, Vec<f64>> = model_spec.bound_data_1d.clone();
    let mut matrix_map: HashMap<String, (Vec<f64>, usize, usize)> =
        model_spec.bound_data_2d.clone();
    if let Some(d) = data {
        let (e1, e2) = parse_data_dict(d)?;
        merge_data_overrides(&mut data_map, &mut matrix_map, e1, e2);
    }

    validate_matrix_storage(&matrix_map)?;

    let compiled = compile_python_model(model_spec, &data_map, &matrix_map)?;
    let graph = compiled.graph.clone();
    let likelihood_names = compiled.likelihood_names.clone();
    let heads = graph.observation_heads();

    // ── Sample from priors and run forward passes ─────────────────────────────
    // The generator itself lives in the core so a `GraphModel` loaded outside
    // Python simulates from exactly the same code.
    let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, PRIOR_PREDICT_SEED_DOMAIN));
    let draws = rustmc_core::prior_sampling::prior_predictive(
        &graph,
        &model_spec.priors,
        &compiled.display_params,
        &compiled.auto_vector_params,
        n_samples,
        &mut rng,
    )
    .map_err(model_error)?;

    // ── Package results ───────────────────────────────────────────────────────
    let dict = PyDict::new(py);
    for (pi, spec) in compiled.display_params.iter().enumerate() {
        let name = match spec {
            DisplayParamSpec::Raw { name, .. } => name,
            DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name,
        };
        let arr = PyArray1::from_vec(py, draws.params[pi].clone());
        dict.set_item(name, arr)?;
    }
    for (li, name) in likelihood_names.iter().enumerate() {
        let n_obs = heads[li].n_obs;
        let arr = Array2::from_shape_vec((n_samples, n_obs), draws.predictions[li].clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray(py))?;
    }
    for (j, (name, _)) in graph.deterministics.iter().enumerate() {
        let n = draws.deterministic_lens[j];
        if n == 0 {
            dict.set_item(
                name,
                PyArray1::from_vec(py, draws.deterministics[j].clone()),
            )?;
        } else {
            dict.set_item(
                name,
                Array2::from_shape_vec((n_samples, n), draws.deterministics[j].clone())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(dict)
}

/// Major version of the installed ArviZ, which selects its conversion API.
fn arviz_api_generation(az: &Bound<'_, PyModule>) -> PyResult<u64> {
    let arviz_version: String = az.getattr("__version__")?.extract()?;
    arviz_version
        .split('.')
        .next()
        .and_then(|part| part.parse::<u64>().ok())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "cannot determine the ArviZ API generation from version '{arviz_version}'"
            ))
        })
}

/// Call the version-native ArviZ dictionary converter.
fn arviz_from_groups<'py>(
    az: &Bound<'py, PyModule>,
    groups: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let arviz_major = arviz_api_generation(az)?;
    arviz_from_groups_versioned(az, arviz_major, groups)
}

fn arviz_from_groups_versioned<'py>(
    az: &Bound<'py, PyModule>,
    arviz_major: u64,
    groups: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    // ArviZ 1.0 moved conversion into arviz-base and changed `from_dict`
    // from one keyword per group to a single nested group dictionary.
    if arviz_major >= 1 {
        az.call_method1("from_dict", (groups,))
    } else {
        az.call_method("from_dict", (), Some(&groups))
    }
}

/// The dataset for one group of an ArviZ container.
///
/// ArviZ 0.x returns an `InferenceData` whose groups are Dataset attributes;
/// 1.x returns an xarray `DataTree` whose children are nodes wrapping one.
fn arviz_group<'py>(
    arviz_major: u64,
    container: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if arviz_major >= 1 {
        container.get_item(name)?.getattr("dataset")
    } else {
        container.getattr(name)
    }
}

/// Relabel the `posterior_predictive` draw axis with the posterior draw labels
/// that survived thinning.
///
/// ArviZ groups are independent datasets, so a shorter predictive draw axis is
/// legal — but without labels it is anonymous, and the pairing between a
/// predictive draw and its parameter draw is lost. Writing the retained draws'
/// labels as the `draw` coordinate restores it: xarray can then align or
/// `.sel()` the posterior down to exactly the draws that were simulated.
///
/// `retained` holds positions along the posterior's draw axis, and the labels
/// are read back off the posterior group rather than assumed: ArviZ's
/// `data.index_origin` decides where the `draw` coordinate starts, so writing
/// bare indices would silently offset the two groups wherever it is not zero.
fn assign_posterior_predictive_draw_coords(
    py: Python<'_>,
    arviz_major: u64,
    container: &Bound<'_, PyAny>,
    retained: &[i64],
) -> PyResult<()> {
    let positions = PyArray1::from_slice(py, retained);
    let labels = arviz_group(arviz_major, container, "posterior")?
        .getattr("draw")?
        .getattr("values")?
        .get_item(positions)?;
    let coords = PyDict::new(py);
    coords.set_item("draw", labels)?;
    let relabelled = arviz_group(arviz_major, container, "posterior_predictive")?.call_method(
        "assign_coords",
        (),
        Some(&coords),
    )?;
    if arviz_major >= 1 {
        container.set_item("posterior_predictive", relabelled)?;
    } else {
        container.setattr("posterior_predictive", relabelled)?;
    }
    Ok(())
}

#[pymodule]
fn _rustmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dynamic_glm::register(m)?;
    hurdle::register(m)?;
    regression::register(m)?;
    structural::register(m)?;
    runoff::register(m)?;
    forecast_batch::register(m)?;
    state_space::register(m)?;
    forecast_support::register(m)?;
    hierarchical::register(m)?;
    local_level::register(m)?;
    seasonal::register(m)?;
    trend::register(m)?;
    ar::register(m)?;
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
    m.add("ParameterError", m.py().get_type::<ParameterError>())?;
    m.add("StateSpaceError", m.py().get_type::<StateSpaceError>())?;
    m.add("InferenceError", m.py().get_type::<InferenceError>())?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_prior_predictive, m)?)?;
    Ok(())
}
