use crate::graph::{Graph, Op};

/// Value produced by evaluating a node. Scalars and vectors are tracked
/// separately so the graph can mix element-wise data operations with
/// scalar parameter operations without unnecessary heap allocation for
/// the common scalar case.
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

/// Forward-evaluate every node in the graph and return the per-node values.
pub fn forward(graph: &Graph, params: &[f64]) -> Vec<Value> {
    let mut values: Vec<Value> = Vec::with_capacity(graph.nodes.len());

    for node in &graph.nodes {
        let val = match &node.op {
            Op::Param(idx) => Value::Scalar(params[*idx]),
            Op::Constant(c) => Value::Scalar(*c),
            Op::Data(idx) => Value::Vector(graph.data_vectors[*idx].clone()),
            Op::Add(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                Value::Scalar(va + vb)
            }
            Op::Sub(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                Value::Scalar(va - vb)
            }
            Op::Mul(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                Value::Scalar(va * vb)
            }
            Op::Div(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                Value::Scalar(va / vb)
            }
            Op::Neg(a) => Value::Scalar(-values[a.0].as_scalar()),
            Op::Exp(a) => Value::Scalar(values[a.0].as_scalar().exp()),
            Op::Log(a) => Value::Scalar(values[a.0].as_scalar().ln()),
            Op::Square(a) => {
                let v = values[a.0].as_scalar();
                Value::Scalar(v * v)
            }
            Op::ScalarMulData(scalar, data) => {
                let s = values[scalar.0].as_scalar();
                let d = values[data.0].as_vector();
                Value::Vector(d.iter().map(|x| s * x).collect())
            }
            Op::NormalLogP { x, mu, sigma } => {
                let xv = values[x.0].as_scalar();
                let mv = values[mu.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                Value::Scalar(normal_logp_scalar(xv, mv, sv))
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
        };
        values.push(val);
    }

    values
}

/// Compute the total log-probability (sum of all logp_terms).
pub fn eval_logp(graph: &Graph, params: &[f64]) -> f64 {
    let values = forward(graph, params);
    graph
        .logp_terms
        .iter()
        .map(|id| values[id.0].as_scalar())
        .sum()
}

/// Reverse-mode autodiff: compute gradient of total log-probability w.r.t. params.
pub fn grad_logp(graph: &Graph, params: &[f64]) -> (f64, Vec<f64>) {
    let values = forward(graph, params);
    let n = graph.nodes.len();

    let total_logp: f64 = graph
        .logp_terms
        .iter()
        .map(|id| values[id.0].as_scalar())
        .sum();

    // Adjoint for each node. Vectors nodes get vector adjoints via a separate table.
    let mut adj_scalar = vec![0.0f64; n];
    let mut adj_vector: Vec<Option<Vec<f64>>> = vec![None; n];

    // Seed: d(total_logp)/d(logp_term) = 1.0
    for &id in &graph.logp_terms {
        adj_scalar[id.0] += 1.0;
    }

    // Reverse pass
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
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                adj_scalar[a.0] += a_s * vb;
                adj_scalar[b.0] += a_s * va;
            }
            Op::Div(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                adj_scalar[a.0] += a_s / vb;
                adj_scalar[b.0] -= a_s * va / (vb * vb);
            }
            Op::Neg(a) => {
                adj_scalar[a.0] -= a_s;
            }
            Op::Exp(a) => {
                let va = values[a.0].as_scalar().exp();
                adj_scalar[a.0] += a_s * va;
            }
            Op::Log(a) => {
                let va = values[a.0].as_scalar();
                adj_scalar[a.0] += a_s / va;
            }
            Op::Square(a) => {
                let va = values[a.0].as_scalar();
                adj_scalar[a.0] += a_s * 2.0 * va;
            }
            Op::ScalarMulData(scalar, data) => {
                let s = values[scalar.0].as_scalar();
                let d = values[data.0].as_vector();

                // This node produces a vector. Its adjoint comes from downstream
                // consumers (e.g. NormalObsLogP) via adj_vector.
                let upstream = adj_vector[idx].take();
                if let Some(ref uv) = upstream {
                    // d(output_i)/d(scalar) = data_i  =>  adj(scalar) += sum(upstream_i * data_i)
                    let ds: f64 = uv.iter().zip(d.iter()).map(|(u, di)| u * di).sum();
                    adj_scalar[scalar.0] += ds;

                    // d(output_i)/d(data_i) = scalar  =>  adj(data_i) += upstream_i * scalar
                    let dd: Vec<f64> = uv.iter().map(|u| u * s).collect();
                    merge_vec_adj(&mut adj_vector[data.0], &dd);
                }
            }
            Op::NormalLogP { x, mu, sigma } => {
                let xv = values[x.0].as_scalar();
                let mv = values[mu.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                let diff = xv - mv;
                let s2 = sv * sv;
                // d logp / d x = -(x - mu) / sigma^2
                adj_scalar[x.0] += a_s * (-diff / s2);
                // d logp / d mu = (x - mu) / sigma^2
                adj_scalar[mu.0] += a_s * (diff / s2);
                // d logp / d sigma = (x - mu)^2 / sigma^3 - 1/sigma
                adj_scalar[sigma.0] += a_s * (diff * diff / (s2 * sv) - 1.0 / sv);
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
                let n_obs = obs.len() as f64;

                // Gradient w.r.t. mu_vec (vector)
                let dmu: Vec<f64> = mu
                    .iter()
                    .zip(obs.iter())
                    .map(|(m, o)| a_s * (o - m) / s2)
                    .collect();
                merge_vec_adj(&mut adj_vector[mu_vec.0], &dmu);

                // Gradient w.r.t. sigma (scalar)
                let dsigma: f64 = mu
                    .iter()
                    .zip(obs.iter())
                    .map(|(m, o)| {
                        let diff = o - m;
                        diff * diff / (s2 * sv) - 1.0 / sv
                    })
                    .sum::<f64>();
                adj_scalar[sigma.0] += a_s * dsigma;

                // obs data is constant — no gradient needed.
                let _ = n_obs;
            }
        }
    }

    // Extract parameter gradients
    let mut grad = vec![0.0; graph.param_count];
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

        // logp = -0.5 * 1.5^2 - 0.5*ln(2pi) ≈ -1.125 - 0.9189
        assert!((logp - (-0.5 * 1.5_f64.powi(2) - 0.5 * std::f64::consts::TAU.ln())).abs() < 1e-10);
        // d logp / d x = -x = -1.5
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
        let (logp, grad) = grad_logp(&g, &params);

        let eps = 1e-6;
        let logp_plus = eval_logp(&g, &[params[0] + eps]);
        let logp_minus = eval_logp(&g, &[params[0] - eps]);
        let numerical_grad = (logp_plus - logp_minus) / (2.0 * eps);

        assert!(
            (grad[0] - numerical_grad).abs() < 1e-4,
            "analytic={}, numerical={}",
            grad[0],
            numerical_grad
        );

        let _ = logp;
    }
}
