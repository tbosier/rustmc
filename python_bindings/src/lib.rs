use ndarray::Array2;
use numpy::PyArrayMethods;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustmc_core::distributions::Normal;
use rustmc_core::graph::{Graph, NodeId};
use rustmc_core::sampler::{self, SampleResult, SamplerConfig};
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Clone)]
struct ModelSpec {
    priors: Vec<PriorSpec>,
    likelihoods: Vec<LikelihoodSpec>,
}

#[derive(Debug, Clone)]
struct PriorSpec {
    name: String,
    mu: f64,
    sigma: f64,
}

#[derive(Debug, Clone)]
struct LikelihoodSpec {
    mu_expr: MuExpr,
    sigma: f64,
    observed_key: String,
}

/// Recursive expression tree built on the Python side, compiled to graph
/// nodes at sampling time.
#[derive(Debug, Clone)]
enum MuExpr {
    ParamTimesData {
        param_name: String,
        data_key: String,
    },
    /// Element-wise sum of two vector expressions.
    Add(Box<MuExpr>, Box<MuExpr>),
    /// Bare parameter broadcast-added to a vector expression.
    Param(String),
}

impl MuExpr {
    fn is_scalar(&self) -> bool {
        match self {
            MuExpr::Param(_) => true,
            MuExpr::ParamTimesData { .. } => false,
            MuExpr::Add(a, b) => a.is_scalar() && b.is_scalar(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct ModelBuilder {
    priors: Vec<PriorSpec>,
    likelihoods: Vec<LikelihoodSpec>,
    param_names: Vec<String>,
}

#[pyclass]
#[derive(Debug, Clone)]
struct ParamRef {
    name: String,
}

#[pyclass]
#[derive(Debug, Clone)]
struct Expr {
    inner: MuExpr,
}

#[pymethods]
impl ParamRef {
    fn __mul__(&self, data_key: &str) -> Expr {
        Expr {
            inner: MuExpr::ParamTimesData {
                param_name: self.name.clone(),
                data_key: data_key.to_string(),
            },
        }
    }

    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        if let Ok(other_expr) = other.downcast::<Expr>() {
            let rhs = other_expr.borrow().inner.clone();
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(MuExpr::Param(self.name.clone())),
                    Box::new(rhs),
                ),
            })
        } else if let Ok(other_param) = other.downcast::<ParamRef>() {
            let rhs_name = other_param.borrow().name.clone();
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(MuExpr::Param(self.name.clone())),
                    Box::new(MuExpr::Param(rhs_name)),
                ),
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
}

#[pymethods]
impl Expr {
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        if let Ok(other_expr) = other.downcast::<Expr>() {
            let rhs = other_expr.borrow().inner.clone();
            Ok(Expr {
                inner: MuExpr::Add(Box::new(self.inner.clone()), Box::new(rhs)),
            })
        } else if let Ok(other_param) = other.downcast::<ParamRef>() {
            let rhs_name = other_param.borrow().name.clone();
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(self.inner.clone()),
                    Box::new(MuExpr::Param(rhs_name)),
                ),
            })
        } else {
            Err(PyValueError::new_err(
                "unsupported operand type for + with Expr",
            ))
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Expr> {
        if let Ok(other_param) = other.downcast::<ParamRef>() {
            let lhs_name = other_param.borrow().name.clone();
            Ok(Expr {
                inner: MuExpr::Add(
                    Box::new(MuExpr::Param(lhs_name)),
                    Box::new(self.inner.clone()),
                ),
            })
        } else {
            self.__add__(other)
        }
    }
}

#[pymethods]
impl ModelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            priors: Vec::new(),
            likelihoods: Vec::new(),
            param_names: Vec::new(),
        }
    }

    #[pyo3(signature = (name, mu, sigma))]
    fn normal_prior(&mut self, name: &str, mu: f64, sigma: f64) -> ParamRef {
        self.priors.push(PriorSpec {
            name: name.to_string(),
            mu,
            sigma,
        });
        self.param_names.push(name.to_string());
        ParamRef {
            name: name.to_string(),
        }
    }

    #[pyo3(signature = (name, mu_expr, sigma, observed_key))]
    fn normal_likelihood(
        &mut self,
        name: &str,
        mu_expr: &Expr,
        sigma: f64,
        observed_key: &str,
    ) {
        let _ = name;
        self.likelihoods.push(LikelihoodSpec {
            mu_expr: mu_expr.inner.clone(),
            sigma,
            observed_key: observed_key.to_string(),
        });
    }

    fn build(&self) -> ModelSpec {
        ModelSpec {
            priors: self.priors.clone(),
            likelihoods: self.likelihoods.clone(),
        }
    }
}

/// Try to decompose a MuExpr tree into a flat linear combination:
/// ([(param_name, data_key), ...], optional_intercept_param_name)
fn try_extract_linear(expr: &MuExpr) -> Option<(Vec<(String, String)>, Option<String>)> {
    let mut terms = Vec::new();
    let mut intercept: Option<String> = None;

    fn walk(
        e: &MuExpr,
        terms: &mut Vec<(String, String)>,
        intercept: &mut Option<String>,
    ) -> bool {
        match e {
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
        }
    }

    if walk(expr, &mut terms, &mut intercept) && !terms.is_empty() {
        Some((terms, intercept))
    } else {
        None
    }
}

/// Compile a MuExpr tree into graph nodes.
///
/// When the tree is a pure linear combination (Σ βₖ xₖ + optional intercept),
/// this emits a single FusedLinearMu op instead of individual
/// ScalarMulData / VectorAdd / ScalarBroadcastAdd nodes.
fn build_mu_expr(
    graph: &mut Graph,
    expr: &MuExpr,
    data_map: &HashMap<String, Vec<f64>>,
) -> Result<NodeId, PyErr> {
    // Fast path: fuse linear combinations into a single op
    if let Some((terms, intercept_name)) = try_extract_linear(expr) {
        let mut param_nodes = Vec::with_capacity(terms.len());
        let mut data_indices = Vec::with_capacity(terms.len());

        for (param_name, data_key) in &terms {
            let pn = graph
                .node_by_name(param_name)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown param: {}", param_name)))?;
            param_nodes.push(pn);

            let data_vec = data_map
                .get(data_key)
                .ok_or_else(|| PyValueError::new_err(format!("Missing data key: {}", data_key)))?
                .clone();
            data_indices.push(graph.store_data_vec(data_vec));
        }

        let intercept_node = match intercept_name {
            Some(ref name) => Some(
                graph
                    .node_by_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("Unknown param: {}", name)))?,
            ),
            None => None,
        };

        return Ok(graph.fused_linear_mu(param_nodes, data_indices, intercept_node));
    }

    // Fallback: individual ops
    match expr {
        MuExpr::ParamTimesData {
            param_name,
            data_key,
        } => {
            let param_node = graph
                .node_by_name(param_name)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown param: {}", param_name)))?;
            let data_vec = data_map
                .get(data_key)
                .ok_or_else(|| PyValueError::new_err(format!("Missing data key: {}", data_key)))?
                .clone();
            let data_node = graph.add_data(data_key, data_vec);
            Ok(graph.scalar_mul_data(param_node, data_node))
        }
        MuExpr::Param(name) => {
            let param_node = graph
                .node_by_name(name)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown param: {}", name)))?;
            Ok(param_node)
        }
        MuExpr::Add(a, b) => {
            let na = build_mu_expr(graph, a, data_map)?;
            let nb = build_mu_expr(graph, b, data_map)?;
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

#[pyclass]
struct FitResult {
    result: SampleResult,
}

#[pymethods]
impl FitResult {
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (pidx, name) in self.result.param_names.iter().enumerate() {
            let mut all_samples = Vec::new();
            for chain in &self.result.samples {
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
        for (pidx, name) in self.result.param_names.iter().enumerate() {
            let n_chains = self.result.samples.len();
            let n_draws = self.result.samples[0].len();
            let mut arr = Array2::<f64>::zeros((n_chains, n_draws));
            for (ci, chain) in self.result.samples.iter().enumerate() {
                for (di, draw) in chain.iter().enumerate() {
                    arr[[ci, di]] = draw[pidx];
                }
            }
            dict.set_item(name, arr.into_pyarray(py))?;
        }
        Ok(dict)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let means = self.result.mean();
        let dict = PyDict::new(py);
        for (name, val) in self.result.param_names.iter().zip(means.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stds = self.result.std();
        let dict = PyDict::new(py);
        for (name, val) in self.result.param_names.iter().zip(stds.iter()) {
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn accept_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, &self.result.accept_rates)?;
        Ok(list)
    }

    fn __repr__(&self) -> String {
        let means = self.result.mean();
        let stds = self.result.std();
        let mut parts = Vec::new();
        for (i, name) in self.result.param_names.iter().enumerate() {
            parts.push(format!(
                "  {}: mean={:.4}, std={:.4}",
                name, means[i], stds[i]
            ));
        }
        let n_chains = self.result.samples.len();
        let n_draws = if self.result.samples.is_empty() {
            0
        } else {
            self.result.samples[0].len()
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
#[pyo3(signature = (model_spec, data, chains=4, draws=1000, warmup=500, seed=42, threads=0, step_size=0.0, num_leapfrog_steps=15))]
#[allow(clippy::too_many_arguments)]
fn sample(
    py: Python<'_>,
    model_spec: &ModelSpec,
    data: &Bound<'_, PyDict>,
    chains: usize,
    draws: usize,
    warmup: usize,
    seed: u64,
    threads: usize,
    step_size: f64,
    num_leapfrog_steps: usize,
) -> PyResult<FitResult> {
    let mut data_map: HashMap<String, Vec<f64>> = HashMap::new();
    for (key, value) in data.iter() {
        let key_str: String = key.extract()?;
        let arr: &Bound<'_, PyArray1<f64>> = value.downcast()?;
        let vec: Vec<f64> = unsafe { arr.as_slice()?.to_vec() };
        data_map.insert(key_str, vec);
    }

    let mut graph = Graph::new();

    for prior in &model_spec.priors {
        Normal::prior(&mut graph, &prior.name, prior.mu, prior.sigma);
    }

    for lik in &model_spec.likelihoods {
        let mu_node = build_mu_expr(&mut graph, &lik.mu_expr, &data_map)?;

        let obs_vec = data_map
            .get(&lik.observed_key)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Missing observed data key: {}",
                    lik.observed_key
                ))
            })?
            .clone();
        Normal::observed(&mut graph, mu_node, lik.sigma, obs_vec);
    }

    let config = SamplerConfig {
        num_chains: chains,
        num_draws: draws,
        num_warmup: warmup,
        step_size,
        num_leapfrog_steps,
        seed,
        num_threads: threads,
    };

    let result = py.allow_threads(|| sampler::sample(graph, config));

    Ok(FitResult { result })
}

#[pymodule]
fn rustmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ModelBuilder>()?;
    m.add_class::<ModelSpec>()?;
    m.add_class::<ParamRef>()?;
    m.add_class::<Expr>()?;
    m.add_class::<FitResult>()?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    Ok(())
}
