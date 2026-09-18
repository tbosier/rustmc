//! Analytic prior draws in unconstrained coordinates, without arbitrary floors.
//!
//! The model-level generator lives here rather than in the Python binding
//! crate so that a `GraphModel` loaded from an artifact can be simulated from
//! Rust, and so that the prior predictive and the posterior cannot disagree
//! about a prior's meaning.
use crate::autodiff::Evaluator;
use crate::graph::Graph;
use crate::model::{
    derive_display_draw, resolve_hyper_value, should_auto_noncenter, validate_positive_finite,
    DisplayParamSpec, HyperParam, ModelError, ModelResult, PriorSpec,
};
use crate::observation;
use rand::{distributions::Open01, Rng};
use rand_distr::{Distribution, Gamma, StandardNormal};
use std::collections::HashMap;

fn positive(value: f64) -> Result<f64, String> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("prior parameters must be finite and positive".into())
    }
}

fn finite(value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("unconstrained prior draw is not representable".into())
    }
}

/// Log of a Gamma(shape, rate) draw. For shape below one, the identity
/// G(shape) = G(shape + 1) U^(1/shape) avoids underflow before taking the log.
pub fn log_gamma<R: Rng + ?Sized>(shape: f64, rate: f64, rng: &mut R) -> Result<f64, String> {
    positive(shape)?;
    positive(rate)?;
    let boosted = if shape < 1.0 { shape + 1.0 } else { shape };
    let unit: f64 = Gamma::new(boosted, 1.0)
        .map_err(|e| e.to_string())?
        .sample(rng);
    let mut draw = positive(unit)?.ln() - rate.ln();
    if shape < 1.0 {
        let u: f64 = rng.sample(Open01);
        draw += u.ln() / shape;
    }
    finite(draw)
}

/// The logit of a Beta draw is the difference of independent log Gamma draws.
pub fn logit_beta<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64, String> {
    finite(log_gamma(alpha, 1.0, rng)? - log_gamma(beta, 1.0, rng)?)
}

pub fn log_half_normal<R: Rng + ?Sized>(sigma: f64, rng: &mut R) -> Result<f64, String> {
    positive(sigma)?;
    let standard: f64 = StandardNormal.sample(rng);
    finite(positive(standard.abs())?.ln() + sigma.ln())
}

/// The logit, written so neither tail loses precision.
///
/// This is the one implementation: the Uniform prior draw below and the
/// binding's sigmoid/bounded-sigmoid transform inverses all call it, so they
/// cannot drift apart. It is not an exact inverse of the forward transform -
/// a sigmoid saturates, so a raw value far from zero does not survive the
/// round trip.
pub fn logit_stable(p: f64) -> f64 {
    p.ln() - (-p).ln_1p()
}

/// Sample raw (unconstrained) parameters from the model priors.
/// Processes priors in declaration order so hierarchical hyperpriors work.
///
/// Expects priors that have already passed `model::validate_definition`, which
/// is what `model::compile` and every loaded `GraphModel` guarantee. It does
/// not re-check prior parameters, so a hand-built `PriorSpec` with, say, a
/// Bernoulli probability outside [0, 1] yields draws rather than an error.
pub fn sample_prior_raw<R: Rng + ?Sized>(
    priors: &[PriorSpec],
    auto_vector_params: &HashMap<String, usize>,
    rng: &mut R,
) -> ModelResult<Vec<f64>> {
    use crate::graph::ParamTransform;
    use rand_distr::{Normal as NormalDist, StudentT as StudentTDist};

    let mut raw: Vec<f64> = Vec::new();
    // Track post-transform values for HyperParam::Param resolution
    let mut sampled_values: HashMap<String, f64> = HashMap::new();

    // A hyperparameter that is not yet available is a broken model, not a
    // reason to substitute 1.0: doing so returns plausible-but-wrong prior
    // predictive draws with no warning.
    let resolve = |hp: &HyperParam, sv: &HashMap<String, f64>, owner: &str| -> ModelResult<f64> {
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
                        .map_err(|e| ModelError::invalid(e.to_string()))?;
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
                        .map_err(|e| ModelError::invalid(e.to_string()))?
                        .sample(rng);
                    sampled_values.insert(name.clone(), x);
                    raw.push(x); // identity transform
                }
            }
            PriorSpec::HalfNormal { name, sigma } => {
                let sigma_v = resolve(sigma, &sampled_values, name)?;
                validate_positive_finite("sigma", sigma_v)?;
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let draw = log_half_normal(sigma_v, rng).map_err(ModelError::invalid)?;
                    if k == 0 {
                        sampled_values.insert(name.clone(), draw.exp());
                    }
                    raw.push(draw);
                }
            }
            PriorSpec::Exponential { name, rate } => {
                let rate_v = resolve(rate, &sampled_values, name)?;
                validate_positive_finite("rate", rate_v)?;
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let draw = log_gamma(1.0, rate_v, rng).map_err(ModelError::invalid)?;
                    if k == 0 {
                        sampled_values.insert(name.clone(), draw.exp());
                    }
                    raw.push(draw);
                }
            }
            PriorSpec::LogNormal { name, mu, sigma } => {
                if let Some(&n) = auto_vector_params.get(name) {
                    let mu_v = resolve(mu, &sampled_values, name)?;
                    let sigma_v = resolve(sigma, &sampled_values, name)?;
                    validate_positive_finite("sigma", sigma_v)?;
                    let dist = NormalDist::new(mu_v, sigma_v)
                        .map_err(|e| ModelError::invalid(e.to_string()))?;
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
                        .map_err(|e| ModelError::invalid(e.to_string()))?
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
                    StudentTDist::new(*nu).map_err(|e| ModelError::invalid(e.to_string()))?;
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
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let p: f64 = rng.sample(Open01);
                    let draw = logit_stable(p);
                    let x = ParamTransform::BoundedSigmoid {
                        lower: *lower,
                        upper: *upper,
                    }
                    .apply(draw);
                    if k == 0 {
                        sampled_values.insert(name.clone(), x);
                    }
                    raw.push(draw);
                }
            }
            PriorSpec::Gamma { name, alpha, beta } => {
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let draw = log_gamma(*alpha, *beta, rng).map_err(ModelError::invalid)?;
                    if k == 0 {
                        sampled_values.insert(name.clone(), draw.exp());
                    }
                    raw.push(draw);
                }
            }
            PriorSpec::Beta { name, alpha, beta } => {
                let n = auto_vector_params.get(name).copied().unwrap_or(1);
                for k in 0..n {
                    let draw = logit_beta(*alpha, *beta, rng).map_err(ModelError::invalid)?;
                    if k == 0 {
                        sampled_values.insert(name.clone(), ParamTransform::Sigmoid.apply(draw));
                    }
                    raw.push(draw);
                }
            }
            PriorSpec::Bernoulli { name, p } => {
                let x: f64 = if rng.gen::<f64>() < *p { 1.0 } else { 0.0 };
                sampled_values.insert(name.clone(), x);
                raw.push(x);
            }
            PriorSpec::Poisson { name, lam } => {
                let x = if *lam == 0.0 {
                    0.0
                } else {
                    observation::sample(crate::graph::ObsFamily::PoissonLog, lam.ln(), None, rng)
                        .map_err(ModelError::invalid)?
                };
                sampled_values.insert(name.clone(), x);
                raw.push(x);
            }
            PriorSpec::VectorNormal { name, n, mu, sigma } => {
                let dist =
                    NormalDist::new(*mu, *sigma).map_err(|e| ModelError::invalid(e.to_string()))?;
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

/// One prior draw, in both coordinate systems the rest of the library uses.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorDraw {
    /// Unconstrained values in the graph's parameter order.
    pub raw: Vec<f64>,
    /// Constrained values in `display_params` order, with non-centred Normals
    /// already resolved back to the parameter the user declared.
    pub display: Vec<f64>,
}

/// Draw one set of prior values for a compiled graph.
///
/// Carries the same validated-input expectation as `sample_prior_raw`.
pub fn sample_prior_draw<R: Rng + ?Sized>(
    graph: &Graph,
    priors: &[PriorSpec],
    display_params: &[DisplayParamSpec],
    auto_vector_params: &HashMap<String, usize>,
    rng: &mut R,
) -> ModelResult<PriorDraw> {
    let raw = sample_prior_raw(priors, auto_vector_params, rng)?;
    if raw.len() != graph.param_count {
        return Err(ModelError::invalid(format!(
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
    let display = derive_display_draw(&constrained_raw, display_params)?;
    if raw.iter().chain(&display).any(|value| !value.is_finite()) {
        return Err(ModelError::invalid("prior draw is not representable"));
    }
    Ok(PriorDraw { raw, display })
}

/// Prior predictive draws: parameters, simulated observations, and any
/// deterministics, all from the priors alone.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorPredictive {
    /// `params[display_index][sample]`.
    pub params: Vec<Vec<f64>>,
    /// `predictions[likelihood_index]`, flattened `(n_samples, n_obs)`.
    pub predictions: Vec<Vec<f64>>,
    /// Observation count per likelihood, parallel to `predictions`.
    pub n_obs: Vec<usize>,
    /// `deterministics[index]`, flattened `(n_samples, len.max(1))`.
    pub deterministics: Vec<Vec<f64>>,
    /// Node length per deterministic; zero marks a scalar.
    pub deterministic_lens: Vec<usize>,
}

/// Simulate from the prior and push each draw forward through `graph`.
///
/// `graph` must already carry its data, because the linear predictor needs the
/// covariates. Bind first, then call this.
pub fn prior_predictive<R: Rng + ?Sized>(
    graph: &Graph,
    priors: &[PriorSpec],
    display_params: &[DisplayParamSpec],
    auto_vector_params: &HashMap<String, usize>,
    n_samples: usize,
    rng: &mut R,
) -> ModelResult<PriorPredictive> {
    let heads = graph.observation_heads();
    let mut evaluator = Evaluator::new(graph);

    let mut params: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); display_params.len()];
    let mut predictions: Vec<Vec<f64>> = heads
        .iter()
        .map(|head| Vec::with_capacity(n_samples * head.n_obs))
        .collect();
    let deterministic_lens: Vec<usize> = graph
        .deterministics
        .iter()
        .map(|(_, node)| evaluator.node_len(*node))
        .collect();
    let mut deterministics: Vec<Vec<f64>> = deterministic_lens
        .iter()
        .map(|len| Vec::with_capacity(n_samples * (*len).max(1)))
        .collect();

    for _ in 0..n_samples {
        let draw = sample_prior_draw(graph, priors, display_params, auto_vector_params, rng)?;
        for (pi, &value) in draw.display.iter().enumerate() {
            params[pi].push(value);
        }

        // Forward pass to get predictions
        evaluator.compute(graph, &draw.raw);
        for (j, (_, node)) in graph.deterministics.iter().enumerate() {
            for i in 0..deterministic_lens[j].max(1) {
                deterministics[j].push(evaluator.vec_elem(*node, i, graph));
            }
        }
        for (li, head) in heads.iter().enumerate() {
            for i in 0..head.n_obs {
                let eta = evaluator.vec_elem(head.linpred, i, graph);
                let aux = head.aux.map(|node| evaluator.scalar_at(node));
                predictions[li].push(
                    observation::sample(head.family, eta, aux, rng).map_err(ModelError::invalid)?,
                );
            }
        }
    }

    Ok(PriorPredictive {
        params,
        predictions,
        n_obs: heads.iter().map(|head| head.n_obs).collect(),
        deterministics,
        deterministic_lens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn small_gamma_shapes_retain_unconstrained_tail_mass() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(951);
        let draws = 50_000;
        let count = (0..draws)
            .filter(|_| log_gamma(0.001, 1.0, &mut rng).unwrap() < -1000.0)
            .count();
        // Gamma CDF below exp(-1000) is exp(-1)/Gamma(1.001), to negligible error.
        let expected = (-1.0 - crate::autodiff::ln_gamma(1.001)).exp();
        assert!((count as f64 / draws as f64 - expected).abs() < 0.012);
    }

    #[test]
    fn beta_draws_retain_both_logit_tails() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(952);
        let values: Vec<_> = (0..50_000)
            .map(|_| logit_beta(0.01, 0.01, &mut rng).unwrap())
            .collect();
        for sign in [-1.0, 1.0] {
            let fraction =
                values.iter().filter(|&&x| sign * x > 40.0).count() as f64 / values.len() as f64;
            assert!((fraction - 0.3352143658).abs() < 0.012);
        }
    }
}
