//! Lower typed Python expressions into native graph operations, retaining fused linear paths.
use super::{LinearTerms, ModelSpec, MuExpr, ParameterError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustmc_core::graph::{Graph, NodeId};
use std::collections::HashMap;

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
            MuExpr::Param(name) if intercept.is_none() => {
                *intercept = Some(name.clone());
                true
            }
            // MatVec uses faer GEMV — never fuse into scalar linear combination
            _ => false,
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
pub(super) fn collect_matvec_params(
    spec: &ModelSpec,
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
            MuExpr::Add(a, b) | MuExpr::Binary(_, a, b) => {
                walk(a, matrix_map, out)?;
                walk(b, matrix_map, out)?;
                Ok(())
            }
            MuExpr::Unary(_, a) | MuExpr::Sum(a) => walk(a, matrix_map, out),
            _ => Ok(()),
        }
    }

    for expr in spec.likelihoods.iter().map(|lik| &lik.mu_expr).chain(
        spec.potentials
            .iter()
            .chain(&spec.deterministics)
            .map(|(_, expr)| expr),
    ) {
        walk(expr, matrix_map, &mut result)?;
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
pub(super) fn build_mu_expr(
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
        MuExpr::Unary(op, a) => {
            let a = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            Ok(graph.elementwise(*op, a, None))
        }
        MuExpr::Sum(a) => {
            let a = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            Ok(graph.sum(a))
        }
        MuExpr::Binary(op, a, b) => {
            let a = build_mu_expr(
                graph,
                a,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            let b = build_mu_expr(
                graph,
                b,
                data_map,
                matrix_map,
                vector_param_map,
                value_node_map,
            )?;
            Ok(graph.elementwise(*op, a, Some(b)))
        }
        MuExpr::Data(key) => Ok(graph.add_data(
            key,
            data_map
                .get(key)
                .ok_or_else(|| PyValueError::new_err(format!("missing data key {key}")))?
                .clone(),
        )),
        MuExpr::Gather {
            param_name,
            data_key,
        } => {
            let (start, n) = *vector_param_map
                .get(param_name)
                .ok_or_else(|| PyValueError::new_err("indexing requires a vector parameter"))?;
            let data = graph.add_data(
                data_key,
                data_map
                    .get(data_key)
                    .ok_or_else(|| PyValueError::new_err(format!("missing index key {data_key}")))?
                    .clone(),
            );
            Ok(graph.gather(start, n, data))
        }
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
