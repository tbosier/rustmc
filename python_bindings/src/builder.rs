//! `ModelBuilder` and the `ModelSpec` it produces.
use crate::compiled::PyCompiledModel;
use crate::data_input::{
    merge_data_overrides, parse_data_dict, validate_data_keys, validate_expr_keys,
    validate_matrix_storage, Data1d, Data2d,
};
use crate::expressions::{extract_expr, first_param_name, Expr, ParamRef, VectorParamRef};
use crate::{model_error, param_error, ParameterError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustmc_core::model::{
    prior_name, CompiledDefinition as CompiledPythonModel, HyperParam, LikelihoodFamily,
    LikelihoodSpec, MuExpr, PriorSpec, SigmaSpec,
};
use rustmc_core::param_ref::{validate_param_references, ParamReference};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic id handed to each `ModelBuilder` so that a `ParamRef` produced by
/// one model can never be silently consumed by another.
pub(crate) static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_model_id() -> u64 {
    NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed)
}

/// Error for a `ParamRef`/`Expr` that belongs to a different `ModelBuilder`.
pub(crate) fn foreign_param_error(name: &str, context: &str) -> PyErr {
    ParameterError::new_err(format!(
        "parameter '{}' used in {} belongs to a different model. \
         A ParamRef returned by one ModelBuilder cannot be used in another.",
        name, context
    ))
}

#[pyclass(module = "rustmc")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub(crate) struct ModelSpec(pub(crate) rustmc_core::model::ModelSpec);
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
    /// The data bound at build time, overridden and extended by call-site
    /// `data`.
    pub(crate) fn data_with(&self, data: Option<&Bound<'_, PyDict>>) -> PyResult<(Data1d, Data2d)> {
        let mut data_1d = self.bound_data_1d.clone();
        let mut data_2d = self.bound_data_2d.clone();
        if let Some(data) = data {
            let (extra_1d, extra_2d) = parse_data_dict(data)?;
            merge_data_overrides(&mut data_1d, &mut data_2d, extra_1d, extra_2d);
        }
        validate_matrix_storage(&data_2d)?;
        Ok((data_1d, data_2d))
    }
}

pub(crate) fn template_data_for_spec(spec: &ModelSpec) -> PyResult<(Data1d, Data2d)> {
    rustmc_core::model::template_data_for_spec(&spec.0).map_err(model_error)
}

#[pyclass(module = "rustmc")]
#[derive(Debug, Clone)]
pub(crate) struct ModelBuilder {
    pub(crate) dimensions: HashMap<String, String>,
    pub(crate) potentials: Vec<(String, MuExpr)>,
    pub(crate) deterministics: Vec<(String, MuExpr)>,
    pub(crate) id: u64,
    pub(crate) priors: Vec<PriorSpec>,
    pub(crate) likelihoods: Vec<LikelihoodSpec>,
    pub(crate) bound_data_1d: HashMap<String, Vec<f64>>,
    pub(crate) bound_data_2d: HashMap<String, (Vec<f64>, usize, usize)>,
}

/// Validate every parameter reference in a model up front, before any graph is
/// built. Fails loudly on unknown names, out-of-order hyperparameters and
/// duplicate declarations.
pub(crate) fn validate_model_references(
    priors: &[PriorSpec],
    likelihoods: &[LikelihoodSpec],
) -> PyResult<()> {
    rustmc_core::model::validate_model_references(priors, likelihoods).map_err(model_error)
}

/// HMC and NUTS evolve a continuous Euclidean state.  Discrete latent
/// parameters therefore need marginalisation or a discrete transition kernel;
/// treating them as continuous values produces invalid posterior draws.
/// Keep these priors available for prior-predictive simulation, but reject
/// every posterior-sampling entry point until such a kernel exists.
pub(crate) fn reject_discrete_priors_for_gradient_sampling(priors: &[PriorSpec]) -> PyResult<()> {
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
        Ok(PyCompiledModel {
            model: compiled.into_model(&spec),
            default_data_1d: self.bound_data_1d.clone(),
            default_data_2d: self.bound_data_2d.clone(),
        })
    }
}

/// Adapters over the core validators. The rule and its wording live in
/// `rustmc_core::model` so the Python surface and the Rust core cannot drift.
pub(crate) fn validate_finite(name: &str, value: f64) -> PyResult<()> {
    rustmc_core::model::validate_finite(name, value).map_err(model_error)
}

pub(crate) fn validate_positive_finite(name: &str, value: f64) -> PyResult<()> {
    rustmc_core::model::validate_positive_finite(name, value).map_err(model_error)
}

/// Parse a Python value (float or ParamRef) into a HyperParam.
pub(crate) fn extract_hyper(obj: &Bound<'_, PyAny>, arg_name: &str) -> PyResult<HyperParam> {
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

pub(crate) fn compile_python_model(
    spec: &ModelSpec,
    data: &Data1d,
    matrices: &Data2d,
) -> PyResult<CompiledPythonModel> {
    rustmc_core::model::compile(&spec.0, data, matrices).map_err(model_error)
}
