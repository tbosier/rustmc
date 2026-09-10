//! Allocating reference evaluator retained for independent checks of optimized kernels.
use super::*;
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

pub fn forward(graph: &Graph, params: &[f64]) -> Vec<Value> {
    let mut values: Vec<Value> = Vec::with_capacity(graph.nodes.len());

    for node in &graph.nodes {
        let val = match &node.op {
            Op::Elementwise { operator, a, b } => {
                let av = &values[a.0];
                let bv = b.map(|b| &values[b.0]);
                let n = match (av, bv) {
                    (Value::Vector(v), _) => v.len(),
                    (_, Some(Value::Vector(v))) => v.len(),
                    _ => 0,
                };
                let read = |v: &Value, i: usize| match v {
                    Value::Scalar(x) => *x,
                    Value::Vector(v) => v[i],
                };
                if n == 0 {
                    Value::Scalar(operator.value(read(av, 0), bv.map_or(0.0, |v| read(v, 0))))
                } else {
                    Value::Vector(
                        (0..n)
                            .map(|i| operator.value(read(av, i), bv.map_or(0.0, |v| read(v, i))))
                            .collect(),
                    )
                }
            }
            Op::Gather {
                param_start,
                indices,
                ..
            } => Value::Vector(
                values[indices.0]
                    .as_vector()
                    .iter()
                    .map(|i| {
                        let k = *param_start + *i as usize;
                        graph.param_transforms[k].apply(params[k])
                    })
                    .collect(),
            ),
            Op::Sum(a) => Value::Scalar(match &values[a.0] {
                Value::Scalar(x) => *x,
                Value::Vector(v) => v.iter().sum(),
            }),
            Op::BroadcastObservation {
                scalar,
                obs_data_idx,
            } => Value::Vector(vec![
                values[scalar.0].as_scalar();
                graph.obs_vectors[*obs_data_idx].len()
            ]),
            Op::Param(idx) => Value::Scalar(params[*idx]),
            Op::Constant(c) => Value::Scalar(*c),
            Op::Data(idx) => Value::Vector(graph.data_vectors[*idx].clone()),
            Op::Add(a, b) => Value::Scalar(values[a.0].as_scalar() + values[b.0].as_scalar()),
            Op::Sub(a, b) => Value::Scalar(values[a.0].as_scalar() - values[b.0].as_scalar()),
            Op::Mul(a, b) => Value::Scalar(values[a.0].as_scalar() * values[b.0].as_scalar()),
            Op::Div(a, b) => Value::Scalar(values[a.0].as_scalar() / values[b.0].as_scalar()),
            Op::Neg(a) => Value::Scalar(-values[a.0].as_scalar()),
            Op::Exp(a) => Value::Scalar(values[a.0].as_scalar().exp()),
            Op::Log(a) => Value::Scalar(values[a.0].as_scalar().ln()),
            Op::Sigmoid(a) => {
                let v = values[a.0].as_scalar();
                Value::Scalar(1.0 / (1.0 + (-v).exp()))
            }
            Op::Square(a) => {
                let v = values[a.0].as_scalar();
                Value::Scalar(v * v)
            }
            Op::ScalarMulData(scalar, data) => {
                let s = values[scalar.0].as_scalar();
                let d = values[data.0].as_vector();
                Value::Vector(d.iter().map(|x| s * x).collect())
            }
            Op::VectorAdd(a, b) => {
                let va = values[a.0].as_vector();
                let vb = values[b.0].as_vector();
                Value::Vector(va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect())
            }
            Op::ScalarBroadcastAdd(scalar, vec) => {
                let s = values[scalar.0].as_scalar();
                let v = values[vec.0].as_vector();
                Value::Vector(v.iter().map(|x| s + x).collect())
            }
            Op::ScalarBroadcast(scalar) => {
                let s = values[scalar.0].as_scalar();
                let n = graph
                    .obs_vectors
                    .first()
                    .or_else(|| graph.data_vectors.first())
                    .map_or(0, |v| v.len());
                Value::Vector(vec![s; n])
            }
            Op::NormalLogP { x, mu, sigma } => Value::Scalar(normal_logp_scalar(
                values[x.0].as_scalar(),
                values[mu.0].as_scalar(),
                values[sigma.0].as_scalar(),
            )),
            Op::HalfNormalLogP { x, sigma } => Value::Scalar(half_normal_logp_scalar(
                values[x.0].as_scalar(),
                values[sigma.0].as_scalar(),
            )),
            Op::StudentTLogP { x, nu, mu, sigma } => Value::Scalar(student_t_logp_scalar(
                values[x.0].as_scalar(),
                values[nu.0].as_scalar(),
                values[mu.0].as_scalar(),
                values[sigma.0].as_scalar(),
            )),
            Op::UniformLogP { x, lower, upper } => Value::Scalar(uniform_logp_scalar(
                values[x.0].as_scalar(),
                values[lower.0].as_scalar(),
                values[upper.0].as_scalar(),
            )),
            Op::BernoulliLogP { x, p } => Value::Scalar(bernoulli_logp_scalar(
                values[x.0].as_scalar(),
                values[p.0].as_scalar(),
            )),
            Op::PoissonLogP { x, lam } => Value::Scalar(poisson_logp_scalar(
                values[x.0].as_scalar(),
                values[lam.0].as_scalar(),
            )),
            Op::GammaLogP { x, alpha, beta } => Value::Scalar(gamma_logp_scalar(
                values[x.0].as_scalar(),
                values[alpha.0].as_scalar(),
                values[beta.0].as_scalar(),
            )),
            Op::BetaLogP { x, alpha, beta } => Value::Scalar(beta_logp_scalar(
                values[x.0].as_scalar(),
                values[alpha.0].as_scalar(),
                values[beta.0].as_scalar(),
            )),
            Op::ObsLogP {
                family,
                linpred_vec,
                aux,
                obs_data_idx,
            } => {
                let obs = &graph.obs_vectors[*obs_data_idx];
                match family {
                    crate::graph::ObsFamily::Normal => {
                        let mu = values[linpred_vec.0].as_vector();
                        let sigma_node = aux.expect("Normal obs logp requires sigma");
                        let sv = values[sigma_node.0].as_scalar();
                        Value::Scalar(normal_obs_logp_sum(mu, sv, obs))
                    }
                    crate::graph::ObsFamily::BernoulliLogit => {
                        let eta = values[linpred_vec.0].as_vector();
                        Value::Scalar(bernoulli_logit_obs_logp_sum(eta, obs))
                    }
                    crate::graph::ObsFamily::PoissonLog => {
                        let eta = values[linpred_vec.0].as_vector();
                        Value::Scalar(poisson_log_obs_logp_sum(eta, obs))
                    }
                    crate::graph::ObsFamily::ExponentialLog => {
                        let eta = values[linpred_vec.0].as_vector();
                        Value::Scalar(exponential_log_obs_logp_sum(eta, obs))
                    }
                    crate::graph::ObsFamily::LogNormal => {
                        let mu = values[linpred_vec.0].as_vector();
                        let sigma_node = aux.expect("LogNormal obs logp requires sigma");
                        let sv = values[sigma_node.0].as_scalar();
                        Value::Scalar(log_normal_obs_logp_sum(mu, sv, obs))
                    }
                    crate::graph::ObsFamily::NegativeBinomialLog => {
                        let eta = values[linpred_vec.0].as_vector();
                        let alpha_node = aux.expect("NegativeBinomial obs logp requires alpha");
                        let av = values[alpha_node.0].as_scalar();
                        Value::Scalar(negative_binomial_log_obs_logp_sum(eta, av, obs))
                    }
                }
            }
            Op::FusedLinearMu {
                param_nodes,
                data_indices,
                intercept,
            } => {
                let vl = graph.data_vectors[data_indices[0]].len();
                let base = intercept.map_or(0.0, |n| values[n.0].as_scalar());
                let mut result = vec![base; vl];
                for (k, &pn) in param_nodes.iter().enumerate() {
                    let beta = values[pn.0].as_scalar();
                    let data = &graph.data_vectors[data_indices[k]];
                    for i in 0..vl {
                        result[i] += beta * data[i];
                    }
                }
                Value::Vector(result)
            }
            Op::MatVecMul {
                matrix_idx,
                param_start,
                n_params,
                intercept,
            } => {
                let matrix = &graph.data_matrices[*matrix_idx];
                let base = intercept.map_or(0.0, |n| values[n.0].as_scalar());
                let mut result = vec![base; matrix.n_rows];
                for (i, value) in result.iter_mut().enumerate().take(matrix.n_rows) {
                    for j in 0..*n_params {
                        *value += matrix.data[i * matrix.n_cols + j]
                            * graph.param_transforms[param_start + j]
                                .apply(params[param_start + j]);
                    }
                }
                Value::Vector(result)
            }
            Op::VectorNormalLogP {
                param_start,
                n_params,
                mu,
                sigma,
            } => {
                let log_norm = -0.5 * std::f64::consts::TAU.ln() - sigma.ln();
                let s2 = sigma * sigma;
                let sum: f64 = (0..*n_params)
                    .map(|k| {
                        let d = params[param_start + k] - mu;
                        log_norm - 0.5 * d * d / s2
                    })
                    .sum();
                Value::Scalar(sum)
            }
            Op::VectorHalfNormalLogP {
                param_start,
                n_params,
                sigma,
            } => {
                let log_norm = (2.0 / (sigma * std::f64::consts::TAU.sqrt())).ln();
                let s2 = sigma * sigma;
                let sum: f64 = (0..*n_params)
                    .map(|k| {
                        let raw = params[param_start + k];
                        log_norm - (2.0 * raw).exp() / (2.0 * s2) + raw
                    })
                    .sum();
                Value::Scalar(sum)
            }
            Op::VectorStudentTLogP {
                param_start,
                n_params,
                nu,
                mu,
                sigma,
            } => {
                let log_norm = ln_gamma(0.5 * (nu + 1.0))
                    - ln_gamma(0.5 * nu)
                    - 0.5 * (nu * std::f64::consts::PI * sigma * sigma).ln();
                let sum: f64 = (0..*n_params)
                    .map(|k| {
                        let v = params[param_start + k];
                        let z = (v - mu) / sigma;
                        log_norm - 0.5 * (nu + 1.0) * (1.0 + z * z / nu).ln()
                    })
                    .sum();
                Value::Scalar(sum)
            }
            Op::VectorGammaLogP {
                param_start,
                n_params,
                alpha,
                beta,
            } => {
                let log_norm = alpha * beta.ln() - ln_gamma(*alpha);
                let sum: f64 = (0..*n_params)
                    .map(|k| {
                        let raw = params[param_start + k];
                        log_norm + alpha * raw - beta * raw.exp()
                    })
                    .sum();
                Value::Scalar(sum)
            }
            Op::VectorBetaLogP {
                param_start,
                n_params,
                alpha,
                beta,
            } => {
                let log_norm = ln_gamma(alpha + beta) - ln_gamma(*alpha) - ln_gamma(*beta);
                let sum: f64 = (0..*n_params)
                    .map(|k| {
                        let raw = params[param_start + k];
                        let s = 1.0 / (1.0 + (-raw).exp());
                        log_norm + alpha * s.ln() + beta * (1.0 - s).ln()
                    })
                    .sum();
                Value::Scalar(sum)
            }
            Op::VectorUniformLogP {
                param_start,
                n_params,
                ..
            } => {
                let sum: f64 = (0..*n_params)
                    .map(|k| {
                        let raw = params[param_start + k];
                        let s = 1.0 / (1.0 + (-raw).exp());
                        s.ln() + (1.0 - s).ln()
                    })
                    .sum();
                Value::Scalar(sum)
            }
        };
        values.push(val);
    }
    values
}

pub fn eval_logp(graph: &Graph, params: &[f64]) -> f64 {
    let values = forward(graph, params);
    graph
        .logp_terms
        .iter()
        .map(|id| values[id.0].as_scalar())
        .sum()
}

pub fn grad_logp(graph: &Graph, params: &[f64]) -> (f64, Vec<f64>) {
    let values = forward(graph, params);
    let n = graph.nodes.len();

    let total_logp: f64 = graph
        .logp_terms
        .iter()
        .map(|id| values[id.0].as_scalar())
        .sum();

    let mut adj_scalar = vec![0.0f64; n];
    let mut adj_vector: Vec<Option<Vec<f64>>> = vec![None; n];
    let mut grad = vec![0.0f64; graph.param_count];

    for &id in &graph.logp_terms {
        adj_scalar[id.0] += 1.0;
    }

    for node in graph.nodes.iter().rev() {
        let idx = node.id.0;
        let a_s = adj_scalar[idx];

        match &node.op {
            Op::Elementwise { operator, a, b } => {
                let upstream = adj_vector[idx].take().unwrap_or_else(|| vec![a_s]);
                let read = |v: &Value, i: usize| match v {
                    Value::Scalar(x) => *x,
                    Value::Vector(v) => v[i],
                };
                let mut da = Vec::new();
                let mut db = Vec::new();
                for (i, u) in upstream.iter().enumerate() {
                    let (x, y) = operator.derivatives(
                        read(&values[a.0], i),
                        b.map_or(0.0, |b| read(&values[b.0], i)),
                    );
                    da.push(u * x);
                    db.push(u * y);
                }
                if matches!(values[a.0], Value::Scalar(_)) {
                    adj_scalar[a.0] += da.iter().sum::<f64>();
                } else {
                    merge_vec_adj(&mut adj_vector[a.0], &da);
                }
                if let Some(b) = b {
                    if matches!(values[b.0], Value::Scalar(_)) {
                        adj_scalar[b.0] += db.iter().sum::<f64>();
                    } else {
                        merge_vec_adj(&mut adj_vector[b.0], &db);
                    }
                }
            }
            Op::Gather {
                param_start,
                indices,
                ..
            } => {
                if let Some(upstream) = adj_vector[idx].take() {
                    for (i, u) in upstream.iter().enumerate() {
                        let k = *param_start + values[indices.0].as_vector()[i] as usize;
                        grad[k] += u * graph.param_transforms[k].derivative(params[k]);
                    }
                }
            }
            Op::Sum(a) => match &values[a.0] {
                Value::Scalar(_) => adj_scalar[a.0] += a_s,
                Value::Vector(v) => merge_vec_adj(&mut adj_vector[a.0], &vec![a_s; v.len()]),
            },
            Op::BroadcastObservation { scalar, .. } => {
                if let Some(v) = adj_vector[idx].take() {
                    adj_scalar[scalar.0] += v.iter().sum::<f64>();
                }
            }
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
                adj_scalar[a.0] += a_s * values[b.0].as_scalar();
                adj_scalar[b.0] += a_s * values[a.0].as_scalar();
            }
            Op::Div(a, b) => {
                let va = values[a.0].as_scalar();
                let vb = values[b.0].as_scalar();
                adj_scalar[a.0] += a_s / vb;
                adj_scalar[b.0] -= a_s * va / (vb * vb);
            }
            Op::Neg(a) => adj_scalar[a.0] -= a_s,
            Op::Exp(a) => adj_scalar[a.0] += a_s * values[a.0].as_scalar().exp(),
            Op::Log(a) => adj_scalar[a.0] += a_s / values[a.0].as_scalar(),
            Op::Sigmoid(a) => {
                let s = values[idx].as_scalar();
                adj_scalar[a.0] += a_s * s * (1.0 - s);
            }
            Op::Square(a) => adj_scalar[a.0] += a_s * 2.0 * values[a.0].as_scalar(),
            Op::ScalarMulData(scalar, data) => {
                let s = values[scalar.0].as_scalar();
                let d = values[data.0].as_vector();
                if let Some(ref uv) = adj_vector[idx].take() {
                    let ds: f64 = uv.iter().zip(d.iter()).map(|(u, di)| u * di).sum();
                    adj_scalar[scalar.0] += ds;
                    let dd: Vec<f64> = uv.iter().map(|u| u * s).collect();
                    merge_vec_adj(&mut adj_vector[data.0], &dd);
                }
            }
            Op::VectorAdd(a, b) => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    merge_vec_adj(&mut adj_vector[a.0], uv);
                    merge_vec_adj(&mut adj_vector[b.0], uv);
                }
            }
            Op::ScalarBroadcastAdd(scalar, vec) => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    adj_scalar[scalar.0] += uv.iter().sum::<f64>();
                    merge_vec_adj(&mut adj_vector[vec.0], uv);
                }
            }
            Op::ScalarBroadcast(scalar) => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    adj_scalar[scalar.0] += uv.iter().sum::<f64>();
                }
            }
            Op::NormalLogP { x, mu, sigma } => {
                let xv = values[x.0].as_scalar();
                let mv = values[mu.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                let diff = xv - mv;
                let s2 = sv * sv;
                adj_scalar[x.0] += a_s * (-diff / s2);
                adj_scalar[mu.0] += a_s * (diff / s2);
                adj_scalar[sigma.0] += a_s * (diff * diff / (s2 * sv) - 1.0 / sv);
            }
            Op::HalfNormalLogP { x, sigma } => {
                let xv = values[x.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                if xv >= 0.0 {
                    adj_scalar[x.0] += a_s * (-xv / (sv * sv));
                    adj_scalar[sigma.0] += a_s * (xv * xv / (sv * sv * sv) - 1.0 / sv);
                }
            }
            Op::StudentTLogP { x, nu, mu, sigma } => {
                let xv = values[x.0].as_scalar();
                let nv = values[nu.0].as_scalar();
                let mv = values[mu.0].as_scalar();
                let sv = values[sigma.0].as_scalar();
                let z = (xv - mv) / sv;
                let z2 = z * z;
                let denom = 1.0 + z2 / nv;
                adj_scalar[x.0] += a_s * (-(nv + 1.0) * z / (sv * nv * denom));
                adj_scalar[mu.0] += a_s * ((nv + 1.0) * z / (sv * nv * denom));
                adj_scalar[sigma.0] += a_s * ((nv + 1.0) * z2 / (sv * nv * denom) - 1.0 / sv);
                adj_scalar[nu.0] += a_s
                    * (0.5 * digamma(0.5 * (nv + 1.0))
                        - 0.5 * digamma(0.5 * nv)
                        - 0.5 / nv
                        - 0.5 * denom.ln()
                        + 0.5 * (nv + 1.0) * z2 / (nv * nv * denom));
            }
            Op::UniformLogP { x: _, lower, upper } => {
                let lv = values[lower.0].as_scalar();
                let uv = values[upper.0].as_scalar();
                let range = uv - lv;
                if range > 0.0 {
                    adj_scalar[lower.0] += a_s / range;
                    adj_scalar[upper.0] -= a_s / range;
                }
            }
            Op::BernoulliLogP { x, p } => {
                let xv = values[x.0].as_scalar();
                let pv = values[p.0].as_scalar().clamp(1e-12, 1.0 - 1e-12);
                adj_scalar[p.0] += a_s * (xv / pv - (1.0 - xv) / (1.0 - pv));
            }
            Op::PoissonLogP { x, lam } => {
                let xv = values[x.0].as_scalar();
                let lv = values[lam.0].as_scalar();
                adj_scalar[lam.0] += a_s * (xv / lv - 1.0);
            }
            Op::GammaLogP { x, alpha, beta } => {
                let xv = values[x.0].as_scalar();
                let av = values[alpha.0].as_scalar();
                let bv = values[beta.0].as_scalar();
                if xv > 0.0 {
                    adj_scalar[x.0] += a_s * ((av - 1.0) / xv - bv);
                    adj_scalar[alpha.0] += a_s * (bv.ln() - digamma(av) + xv.ln());
                    adj_scalar[beta.0] += a_s * (av / bv - xv);
                }
            }
            Op::BetaLogP { x, alpha, beta } => {
                let xv = values[x.0].as_scalar();
                let av = values[alpha.0].as_scalar();
                let bv = values[beta.0].as_scalar();
                if xv > 0.0 && xv < 1.0 {
                    adj_scalar[x.0] += a_s * ((av - 1.0) / xv - (bv - 1.0) / (1.0 - xv));
                    adj_scalar[alpha.0] += a_s * (digamma(av + bv) - digamma(av) + xv.ln());
                    adj_scalar[beta.0] += a_s * (digamma(av + bv) - digamma(bv) + (1.0 - xv).ln());
                }
            }
            Op::ObsLogP {
                family,
                linpred_vec,
                aux,
                obs_data_idx,
            } => {
                let obs = &graph.obs_vectors[*obs_data_idx];
                match family {
                    crate::graph::ObsFamily::Normal => {
                        let mu = values[linpred_vec.0].as_vector();
                        let sigma_node = aux.expect("Normal obs logp requires sigma");
                        let sv = values[sigma_node.0].as_scalar();
                        let s2 = sv * sv;
                        let dmu: Vec<f64> = mu
                            .iter()
                            .zip(obs.iter())
                            .map(|(m, o)| a_s * (o - m) / s2)
                            .collect();
                        merge_vec_adj(&mut adj_vector[linpred_vec.0], &dmu);
                        let dsigma: f64 = mu
                            .iter()
                            .zip(obs.iter())
                            .map(|(m, o)| {
                                let diff = o - m;
                                diff * diff / (s2 * sv) - 1.0 / sv
                            })
                            .sum::<f64>();
                        adj_scalar[sigma_node.0] += a_s * dsigma;
                    }
                    crate::graph::ObsFamily::BernoulliLogit => {
                        let eta = values[linpred_vec.0].as_vector();
                        let deta: Vec<f64> = eta
                            .iter()
                            .zip(obs.iter())
                            .map(|(e, y)| a_s * (y - sigmoid_stable(*e)))
                            .collect();
                        merge_vec_adj(&mut adj_vector[linpred_vec.0], &deta);
                    }
                    crate::graph::ObsFamily::PoissonLog => {
                        let eta = values[linpred_vec.0].as_vector();
                        let deta: Vec<f64> = eta
                            .iter()
                            .zip(obs.iter())
                            .map(|(e, y)| a_s * (y - e.exp()))
                            .collect();
                        merge_vec_adj(&mut adj_vector[linpred_vec.0], &deta);
                    }
                    crate::graph::ObsFamily::ExponentialLog => {
                        let eta = values[linpred_vec.0].as_vector();
                        let deta: Vec<f64> = eta
                            .iter()
                            .zip(obs.iter())
                            .map(|(e, y)| a_s * (1.0 - y * e.exp()))
                            .collect();
                        merge_vec_adj(&mut adj_vector[linpred_vec.0], &deta);
                    }
                    crate::graph::ObsFamily::LogNormal => {
                        let mu = values[linpred_vec.0].as_vector();
                        let sigma_node = aux.expect("LogNormal obs logp requires sigma");
                        let sv = values[sigma_node.0].as_scalar();
                        let s2 = sv * sv;
                        let dmu: Vec<f64> = mu
                            .iter()
                            .zip(obs.iter())
                            .map(|(m, y)| {
                                let ly = y.max(1e-300).ln();
                                a_s * (ly - m) / s2
                            })
                            .collect();
                        merge_vec_adj(&mut adj_vector[linpred_vec.0], &dmu);
                        let dsigma: f64 = mu
                            .iter()
                            .zip(obs.iter())
                            .map(|(m, y)| {
                                let ly = y.max(1e-300).ln();
                                let d = ly - m;
                                d * d / (s2 * sv) - 1.0 / sv
                            })
                            .sum::<f64>();
                        adj_scalar[sigma_node.0] += a_s * dsigma;
                    }
                    crate::graph::ObsFamily::NegativeBinomialLog => {
                        let eta = values[linpred_vec.0].as_vector();
                        let alpha_node = aux.expect("NegativeBinomial obs logp requires alpha");
                        let av = values[alpha_node.0].as_scalar();
                        let deta: Vec<f64> = eta
                            .iter()
                            .zip(obs.iter())
                            .map(|(e, y)| {
                                let mu = e.exp();
                                a_s * av * (y - mu) / (av + mu)
                            })
                            .collect();
                        merge_vec_adj(&mut adj_vector[linpred_vec.0], &deta);
                        let dalpha: f64 = eta
                            .iter()
                            .zip(obs.iter())
                            .map(|(e, y)| {
                                let mu = e.exp();
                                let denom = av + mu;
                                digamma(y + av) - digamma(av) + av.ln() + 1.0
                                    - denom.ln()
                                    - (y + av) / denom
                            })
                            .sum::<f64>();
                        adj_scalar[alpha_node.0] += a_s * dalpha;
                    }
                }
            }
            Op::FusedLinearMu {
                param_nodes,
                data_indices,
                intercept,
            } => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    for (k, &pn) in param_nodes.iter().enumerate() {
                        let data = &graph.data_vectors[data_indices[k]];
                        let ds: f64 = uv.iter().zip(data.iter()).map(|(u, d)| u * d).sum();
                        adj_scalar[pn.0] += ds;
                    }
                    if let Some(n) = *intercept {
                        adj_scalar[n.0] += uv.iter().sum::<f64>();
                    }
                }
            }
            Op::MatVecMul {
                matrix_idx,
                param_start,
                n_params,
                intercept,
            } => {
                if let Some(ref uv) = adj_vector[idx].take() {
                    let matrix = &graph.data_matrices[*matrix_idx];
                    // grad[param_start + k] += sum_i X[i,k] * adj[i]
                    for k in 0..*n_params {
                        let mut ds = 0.0f64;
                        ds += uv
                            .iter()
                            .enumerate()
                            .take(matrix.n_rows)
                            .map(|(i, u)| u * matrix.data[i * matrix.n_cols + k])
                            .sum::<f64>();
                        grad[param_start + k] += ds
                            * graph.param_transforms[param_start + k]
                                .derivative(params[param_start + k]);
                    }
                    if let Some(n) = *intercept {
                        adj_scalar[n.0] += uv.iter().sum::<f64>();
                    }
                }
            }
            Op::VectorNormalLogP {
                param_start,
                n_params,
                mu,
                sigma,
            } => {
                let s2 = sigma * sigma;
                for k in 0..*n_params {
                    let v = params[param_start + k];
                    grad[param_start + k] += a_s * (-(v - mu) / s2);
                }
            }
            Op::VectorHalfNormalLogP {
                param_start,
                n_params,
                sigma,
            } => {
                let s2 = sigma * sigma;
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    grad[param_start + k] += a_s * (-(2.0 * raw).exp() / s2 + 1.0);
                }
            }
            Op::VectorStudentTLogP {
                param_start,
                n_params,
                nu,
                mu,
                sigma,
            } => {
                for k in 0..*n_params {
                    let v = params[param_start + k];
                    let z = (v - mu) / sigma;
                    grad[param_start + k] +=
                        a_s * (-(nu + 1.0) * z / (sigma * nu * (1.0 + z * z / nu)));
                }
            }
            Op::VectorGammaLogP {
                param_start,
                n_params,
                alpha,
                beta,
            } => {
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    grad[param_start + k] += a_s * (alpha - beta * raw.exp());
                }
            }
            Op::VectorBetaLogP {
                param_start,
                n_params,
                alpha,
                beta,
            } => {
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    let s = 1.0 / (1.0 + (-raw).exp());
                    grad[param_start + k] += a_s * (alpha * (1.0 - s) - beta * s);
                }
            }
            Op::VectorUniformLogP {
                param_start,
                n_params,
                ..
            } => {
                for k in 0..*n_params {
                    let raw = params[param_start + k];
                    let s = 1.0 / (1.0 + (-raw).exp());
                    grad[param_start + k] += a_s * (1.0 - 2.0 * s);
                }
            }
        }
    }

    for node in &graph.nodes {
        if let Op::Param(pidx) = node.op {
            grad[pidx] += adj_scalar[node.id.0];
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
