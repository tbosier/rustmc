//! Composable structural models with exact Gaussian FFBS and conjugate variance
//! updates. Student-t observations use Gamma precision mixtures with fixed df.
//! Initial state priors are independent of innovation variances; x[-1] is included
//! in every state draw so every transition contributes to the variance update.
use crate::forecast_common::{
    checked_value_count, overdispersed_positive, run_gibbs_chains, sample_inverse_gamma,
    simulate_draws, GibbsSchedule, MAX_MATERIALIZED_VALUES,
};
use crate::seeding::chain_seed;
use crate::state_space::{LinearGaussianStateSpace, StateSpaceError};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use serde::{Deserialize, Serialize};
type Result<T> = std::result::Result<T, StateSpaceError>;
fn invalid(s: &str) -> StateSpaceError {
    StateSpaceError::InvalidParameter(s.into())
}
fn allocation(what: &'static str, factors: &[usize]) -> Result<()> {
    checked_value_count(what, factors, MAX_MATERIALIZED_VALUES)?;
    Ok(())
}

/// `deny_unknown_fields` reaches the `InverseGamma` struct variant; the
/// `Fixed` newtype variant carries no field names for it to act on, and serde
/// already refuses an unrecognised variant name.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum VarianceParameter {
    Fixed(f64),
    InverseGamma { shape: f64, scale: f64 },
}
impl VarianceParameter {
    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Fixed(v) if v.is_finite() && v >= 0.0 => Ok(()),
            Self::InverseGamma {shape, scale} if shape.is_finite() && scale.is_finite() && shape > 0.0 && scale > 0.0 => Ok(()),
            _ => Err(invalid("variance requires a finite nonnegative fixed value or positive inverse-gamma shape and scale"))
        }
    }
    fn initial(&self) -> f64 {
        match *self {
            Self::Fixed(v) => v,
            Self::InverseGamma { shape, scale } => scale / (shape + 1.0),
        }
    }
    /// A chain's starting value: the fixed value, or the prior mode spread
    /// over a log-uniform window so that chains start apart.
    fn start<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match *self {
            Self::Fixed(v) => v,
            Self::InverseGamma { .. } => overdispersed_positive(self.initial(), rng),
        }
    }
    fn sample<R: Rng + ?Sized>(&self, n: usize, ss: f64, rng: &mut R) -> Result<f64> {
        match *self {
            Self::Fixed(v) => Ok(v),
            Self::InverseGamma { shape, scale } => Ok(sample_inverse_gamma(
                shape + n as f64 / 2.0,
                scale + ss / 2.0,
                rng,
            )?),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub name: String,
    pub transition: Vec<f64>,
    pub observation: Vec<f64>,
    pub initial_mean: Vec<f64>,
    pub initial_covariance: Vec<f64>,
    /// One independent variance per coordinate. Equal values do not imply pooling.
    pub innovations: Vec<VarianceParameter>,
    /// Regression observation rows are supplied by the design matrix, in component order.
    pub regression: bool,
}
impl Component {
    pub fn level(name: String, innovation: VarianceParameter, mean: f64, variance: f64) -> Self {
        Self {
            name,
            transition: vec![1.0],
            observation: vec![1.0],
            initial_mean: vec![mean],
            initial_covariance: vec![variance],
            innovations: vec![innovation],
            regression: false,
        }
    }
    pub fn trend(
        name: String,
        damping: f64,
        level: VarianceParameter,
        slope: VarianceParameter,
        mean: Vec<f64>,
        covariance: Vec<f64>,
    ) -> Result<Self> {
        if !damping.is_finite() || damping <= 0.0 || damping > 1.0 {
            return Err(invalid("trend damping must lie in (0,1]"));
        }
        Ok(Self {
            name,
            transition: vec![1.0, damping, 0.0, damping],
            observation: vec![1.0, 0.0],
            initial_mean: mean,
            initial_covariance: covariance,
            innovations: vec![level, slope],
            regression: false,
        })
    }
    /// Harmonic pairs rotate by 2*pi*k/period. Fractional periods are supported;
    /// Nyquist and aliased harmonics are rejected (2*harmonics must be < period).
    pub fn seasonal(
        name: String,
        period: f64,
        harmonics: usize,
        innovation: VarianceParameter,
        initial_variance: f64,
    ) -> Result<Self> {
        if !period.is_finite() || harmonics == 0 || 2.0 * harmonics as f64 >= period {
            return Err(invalid(
                "seasonality requires finite period > 2*harmonics > 0",
            ));
        }
        let d = harmonics
            .checked_mul(2)
            .ok_or_else(|| invalid("too many harmonics"))?;
        let square = d
            .checked_mul(d)
            .ok_or_else(|| invalid("too many harmonics"))?;
        allocation("seasonal component", &[square, 3])?;
        let mut t = vec![0.0; square];
        let mut h = vec![0.0; d];
        let mut p = vec![0.0; square];
        for k in 0..harmonics {
            let (s, c) = (std::f64::consts::TAU * (k + 1) as f64 / period).sin_cos();
            let i = 2 * k;
            t[i * d + i] = c;
            t[i * d + i + 1] = s;
            t[(i + 1) * d + i] = -s;
            t[(i + 1) * d + i + 1] = c;
            h[i] = 1.0;
        }
        for i in 0..d {
            p[i * d + i] = initial_variance;
        }
        Ok(Self {
            name,
            transition: t,
            observation: h,
            initial_mean: vec![0.0; d],
            initial_covariance: p,
            innovations: vec![innovation; d],
            regression: false,
        })
    }
    pub fn regression(
        name: String,
        mean: Vec<f64>,
        covariance: Vec<f64>,
        innovations: Vec<VarianceParameter>,
    ) -> Self {
        let d = mean.len();
        let mut t = vec![0.0; d * d];
        for i in 0..d {
            t[i * d + i] = 1.0;
        }
        Self {
            name,
            transition: t,
            observation: vec![0.0; d],
            initial_mean: mean,
            initial_covariance: covariance,
            innovations,
            regression: true,
        }
    }
    /// Fixed stable AR(p) residual coefficients; the explicit initial prior need
    /// not be stationary. Only the leading coordinate receives innovations.
    pub fn ar(
        name: String,
        coefficients: Vec<f64>,
        innovation: VarianceParameter,
        mean: Vec<f64>,
        covariance: Vec<f64>,
    ) -> Result<Self> {
        let d = coefficients.len();
        if d == 0 || coefficients.iter().any(|v| !v.is_finite()) {
            return Err(invalid("AR coefficients must be finite and nonempty"));
        }
        allocation("AR component", &[d, d, 3])?;
        let mut reduced = coefficients.clone();
        while !reduced.is_empty() {
            let k = *reduced.last().unwrap();
            if k.abs() >= 1.0 {
                return Err(invalid("AR coefficients must define a stable process"));
            }
            let n = reduced.len() - 1;
            reduced = (0..n)
                .map(|j| (reduced[j] + k * reduced[n - 1 - j]) / (1.0 - k * k))
                .collect();
        }
        let mut t = vec![0.0; d * d];
        t[..d].copy_from_slice(&coefficients);
        for i in 1..d {
            t[i * d + i - 1] = 1.0;
        }
        let mut h = vec![0.0; d];
        h[0] = 1.0;
        let mut q = vec![VarianceParameter::Fixed(0.0); d];
        q[0] = innovation;
        Ok(Self {
            name,
            transition: t,
            observation: h,
            initial_mean: mean,
            initial_covariance: covariance,
            innovations: q,
            regression: false,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralConfig {
    pub components: Vec<Component>,
    pub observation_variance: VarianceParameter,
    pub student_df: Option<f64>,
}
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub chains: usize,
    pub draws: usize,
    pub warmup: usize,
    pub thinning: usize,
    pub seed: u64,
    pub store_states: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralDraw {
    pub variances: Vec<f64>,
    pub observation_variance: f64,
    pub terminal_state: Vec<f64>,
    pub states: Option<Vec<Vec<f64>>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralPosterior {
    pub config: StructuralConfig,
    pub chains: Vec<Vec<StructuralDraw>>,
    pub training_rows: Vec<Vec<f64>>,
}
/// Not reachable from either structural loader today - it is a forecast/smoother
/// result, not part of an artifact - but it derives `Deserialize`, so it is held
/// to the same rule in case it is ever embedded in one.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralPaths {
    /// [chain][draw][time][state]
    pub states: Vec<Vec<Vec<Vec<f64>>>>,
    /// [chain][draw][time][component], sums to means.
    pub components: Vec<Vec<Vec<Vec<f64>>>>,
    pub means: Vec<Vec<Vec<f64>>>,
    pub observations: Vec<Vec<Vec<f64>>>,
    pub cumulative: Vec<Vec<Vec<f64>>>,
}
impl StructuralConfig {
    pub fn dimension(&self) -> usize {
        self.components.iter().map(|c| c.initial_mean.len()).sum()
    }
    pub fn features(&self) -> usize {
        self.components
            .iter()
            .filter(|c| c.regression)
            .map(|c| c.initial_mean.len())
            .sum()
    }
    pub fn variance_names(&self) -> Vec<String> {
        self.components
            .iter()
            .flat_map(|c| (0..c.innovations.len()).map(|i| format!("{}.innovation[{i}]", c.name)))
            .collect()
    }
    pub fn validate(&self) -> Result<()> {
        if self.components.is_empty() {
            return Err(invalid("at least one component is required"));
        }
        let mut names = std::collections::HashSet::new();
        for c in &self.components {
            let d = c.initial_mean.len();
            if c.name.is_empty() || !names.insert(&c.name) || d == 0 || c.innovations.len() != d {
                return Err(invalid("components require unique nonempty names, positive dimensions and one variance per coordinate"));
            }
            allocation("structural component", &[d, d, 4])?;
            for q in &c.innovations {
                q.validate()?;
            }
            let mut q = vec![0.0; d * d];
            for i in 0..d {
                q[i * d + i] = c.innovations[i].initial();
            }
            LinearGaussianStateSpace::new(
                d,
                c.transition.clone(),
                c.observation.clone(),
                q,
                1.0,
                c.initial_mean.clone(),
                c.initial_covariance.clone(),
            )?;
        }
        allocation("structural state", &[self.dimension(), self.dimension(), 4])?;
        self.observation_variance.validate()?;
        if self.observation_variance.initial() <= 0.0 {
            return Err(invalid("observation variance must be positive"));
        }
        if self.student_df.is_some_and(|v| !v.is_finite() || v <= 1.0) {
            return Err(invalid(
                "Student-t degrees of freedom must be finite and greater than one for conditional mean forecasts",
            ));
        }
        Ok(())
    }
    pub fn observation_rows(
        &self,
        count: usize,
        design: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<f64>>> {
        let p = self.features();
        if (p > 0 && design.is_none())
            || design.is_some_and(|x| {
                x.len() != count
                    || x.iter()
                        .any(|r| r.len() != p || r.iter().any(|v| !v.is_finite()))
            })
        {
            return Err(invalid(
                "design must have one finite row per time and match total regression features",
            ));
        }
        let mut rows = Vec::with_capacity(count);
        for time in 0..count {
            let mut row = vec![];
            let mut j = 0;
            for c in &self.components {
                if c.regression {
                    let d = c.initial_mean.len();
                    row.extend_from_slice(&design.unwrap()[time][j..j + d]);
                    j += d;
                } else {
                    row.extend_from_slice(&c.observation);
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }
    pub fn build(&self, variances: &[f64], noise: f64) -> Result<LinearGaussianStateSpace> {
        self.validate()?;
        let d = self.dimension();
        if variances.len() != d {
            return Err(invalid("innovation variance dimension mismatch"));
        }
        let mut t = vec![0.0; d * d];
        let mut p = vec![0.0; d * d];
        let mut q = vec![0.0; d * d];
        let mut h = vec![];
        let mut m = vec![];
        let mut off = 0;
        for c in &self.components {
            let n = c.initial_mean.len();
            for i in 0..n {
                for j in 0..n {
                    t[(off + i) * d + off + j] = c.transition[i * n + j];
                    p[(off + i) * d + off + j] = c.initial_covariance[i * n + j];
                }
            }
            h.extend_from_slice(&c.observation);
            m.extend_from_slice(&c.initial_mean);
            off += n;
        }
        for i in 0..d {
            q[i * d + i] = variances[i];
        }
        LinearGaussianStateSpace::new(d, t, h, q, noise, m, p)
    }
    /// The assembled model with every variance at its fixed value or prior mode.
    ///
    /// `build` validates the configuration and factors every covariance, which
    /// is O(d^3) work that depends only on the configuration. Samplers build
    /// this once and then overwrite the diagonal variances with
    /// [`with_variances`], rather than rebuilding per sweep or per draw.
    fn template(&self) -> Result<LinearGaussianStateSpace> {
        let q: Vec<f64> = self.priors().iter().map(|p| p.initial()).collect();
        self.build(&q, self.observation_variance.initial())
    }
    fn priors(&self) -> Vec<&VarianceParameter> {
        self.components
            .iter()
            .flat_map(|c| &c.innovations)
            .collect()
    }
    pub fn prior_predict(
        &self,
        steps: usize,
        design: Option<&[Vec<f64>]>,
        draws: usize,
        seed: u64,
    ) -> Result<StructuralPaths> {
        self.validate()?;
        if draws == 0 || steps == 0 {
            return Err(invalid("draws and steps must be positive"));
        }
        allocation("structural prior draws", &[draws, self.dimension(), 3])?;
        allocation(
            "structural prior predictive",
            &[draws, steps, self.dimension() + self.components.len() + 3],
        )?;
        let template = self.template()?;
        let prior_seed = chain_seed(seed, 0, STRUCTURAL_PRIOR_SEED_DOMAIN);
        let mut rng = ChaCha8Rng::seed_from_u64(prior_seed);
        let mut chain = vec![];
        for _ in 0..draws {
            let q = self
                .priors()
                .iter()
                .map(|p| p.sample(0, 0.0, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let r = self.observation_variance.sample(0, 0.0, &mut rng)?;
            // The initial state distribution does not depend on the variances.
            let state = template.simulate_initial(&mut rng)?;
            chain.push(StructuralDraw {
                variances: q,
                observation_variance: r,
                terminal_state: state,
                states: None,
            });
        }
        StructuralPosterior {
            config: self.clone(),
            chains: vec![chain],
            training_rows: vec![],
        }
        // `forecast` re-keys whatever it is given through FORECAST_SEED_DOMAIN,
        // so handing it the prior-predictive key rather than the caller's own
        // separates these paths from both the prior draws above and a posterior
        // forecast made with the same caller seed.
        .forecast(steps, design, prior_seed)
    }
}
fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    StandardNormal.sample(rng)
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
/// Overwrite a [`StructuralConfig::template`]'s diagonal innovation variances
/// and its observation variance. Every structural innovation covariance is
/// diagonal, so this reproduces `build(variances, noise)` exactly.
fn with_variances(model: &mut LinearGaussianStateSpace, variances: &[f64], noise: f64) {
    let indices: Vec<usize> = (0..variances.len()).collect();
    model.set_variances(&indices, variances, noise);
}
const FIT_SEED_DOMAIN: u64 = 0x4649_545F_5354_5243;
const FORECAST_SEED_DOMAIN: u64 = 0x4652_4353_545F_5354;
const STRUCTURAL_PRIOR_SEED_DOMAIN: u64 = 0x5052_494F_525F_5354;

pub fn fit(
    y: &[f64],
    design: Option<&[Vec<f64>]>,
    config: &StructuralConfig,
    sampling: &SamplingConfig,
) -> Result<StructuralPosterior> {
    config.validate()?;
    if y.is_empty() || y.iter().any(|v| v.is_infinite()) {
        return Err(invalid(
            "nonempty finite/NaN data and positive chains, draws and thinning required",
        ));
    }
    let schedule = GibbsSchedule::new(
        sampling.chains,
        sampling.warmup,
        sampling.draws,
        sampling.thinning,
    )?;
    let d = config.dimension();
    allocation(
        "structural FFBS working state",
        &[sampling.chains, y.len() + 1, d, d, 6],
    )?;
    allocation(
        "structural posterior",
        &[
            sampling.chains,
            sampling.draws,
            d,
            if sampling.store_states {
                y.len() + 3
            } else {
                3
            },
        ],
    )?;
    let rows = config.observation_rows(y.len(), design)?;
    let template = config.template()?.with_observation_rows(rows.clone())?;
    let priors = config.priors();
    let observed = y.iter().filter(|v| v.is_finite()).count();
    let chains = run_gibbs_chains(
        &schedule,
        sampling.seed,
        FIT_SEED_DOMAIN,
        |rng| {
            Ok::<_, StateSpaceError>(ChainState {
                model: template.clone(),
                q: priors.iter().map(|p| p.start(rng)).collect(),
                r: config.observation_variance.start(rng),
                lambda: vec![1.0; y.len()],
            })
        },
        |ChainState {
             model,
             q,
             r,
             lambda,
         },
         rng,
         retain| {
            with_variances(model, q, *r);
            if config.student_df.is_some() {
                model.set_observation_variances(lambda.iter().map(|l| *r / l).collect())?;
            }
            let states = model.sample_states_ffbs(y, rng)?;
            for i in 0..d {
                if matches!(priors[i], VarianceParameter::Fixed(_)) {
                    continue;
                }
                let ss = states
                    .windows(2)
                    .map(|w| {
                        let e = w[1][i] - dot(&model.transition()[i * d..(i + 1) * d], &w[0]);
                        e * e
                    })
                    .sum();
                q[i] = priors[i].sample(y.len(), ss, rng)?;
            }
            let ss = y
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(t, v)| lambda[t] * (v - dot(&rows[t], &states[t + 1])).powi(2))
                .sum();
            *r = config.observation_variance.sample(observed, ss, rng)?;
            if let Some(df) = config.student_df {
                for (t, v) in y.iter().enumerate() {
                    if v.is_finite() {
                        let e = v - dot(&rows[t], &states[t + 1]);
                        lambda[t] = Gamma::new((df + 1.0) / 2.0, 2.0 / (df + e * e / *r))
                            .map_err(|_| invalid("invalid Student-t conditional"))?
                            .sample(rng);
                        if !lambda[t].is_finite() || lambda[t] <= 0.0 {
                            return Err(invalid(
                                "Student-t precision draw overflowed or underflowed",
                            ));
                        }
                    }
                }
            }
            Ok(retain.then(|| StructuralDraw {
                variances: q.clone(),
                observation_variance: *r,
                terminal_state: states[y.len()].clone(),
                states: if sampling.store_states {
                    Some(states)
                } else {
                    None
                },
            }))
        },
    )?;
    Ok(StructuralPosterior {
        config: config.clone(),
        chains,
        training_rows: rows,
    })
}

/// One draw's forecast, indexed `[step]` (and then `[state]` or
/// `[component]`).
#[derive(Default)]
struct DrawPath {
    states: Vec<Vec<f64>>,
    components: Vec<Vec<f64>>,
    means: Vec<f64>,
    observations: Vec<f64>,
    cumulative: Vec<f64>,
}

/// The Gibbs state carried between sweeps: innovation variances, the
/// observation variance and, for Student-t observations, the per-time
/// precision multipliers.
struct ChainState {
    model: LinearGaussianStateSpace,
    q: Vec<f64>,
    r: f64,
    lambda: Vec<f64>,
}

impl StructuralPosterior {
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names = self.config.variance_names();
        names.push("observation_variance".into());
        names.extend((0..self.config.dimension()).map(|i| format!("terminal_state[{i}]")));
        names
    }
    pub fn parameter_samples(&self) -> Vec<Vec<Vec<f64>>> {
        self.chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        let mut values = d.variances.clone();
                        values.push(d.observation_variance);
                        values.extend_from_slice(&d.terminal_state);
                        values
                    })
                    .collect()
            })
            .collect()
    }
    pub fn diagnostics(&self) -> crate::diagnostics::DiagnosticsReport {
        crate::forecast_diagnostics::parameter_diagnostics(
            &self.parameter_samples(),
            &self.parameter_names(),
        )
    }
    pub fn forecast(
        &self,
        steps: usize,
        design: Option<&[Vec<f64>]>,
        seed: u64,
    ) -> Result<StructuralPaths> {
        self.validate()?;
        if steps == 0 || self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(invalid("steps and posterior chains must be nonempty"));
        }
        allocation(
            "structural forecast",
            &[
                self.chains.len(),
                self.chains[0].len(),
                steps,
                self.config.dimension() + self.config.components.len() + 3,
            ],
        )?;
        let rows = self.config.observation_rows(steps, design)?;
        let template = self.config.template()?;
        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &StructuralDraw, rng| {
                let mut model = template.clone();
                with_variances(&mut model, &draw.variances, draw.observation_variance);
                let mut state = draw.terminal_state.clone();
                let mut path = DrawPath::default();
                let mut total = 0.0;
                for row in &rows {
                    state = model.simulate_transition(&state, rng)?;
                    let mut off = 0;
                    let comps = self
                        .config
                        .components
                        .iter()
                        .map(|c| {
                            let end = off + c.initial_mean.len();
                            let v = dot(&row[off..end], &state[off..end]);
                            off = end;
                            v
                        })
                        .collect::<Vec<_>>();
                    let mean = comps.iter().sum::<f64>();
                    let precision = if let Some(df) = self.config.student_df {
                        Gamma::new(df / 2.0, 2.0 / df)
                            .map_err(|_| invalid("invalid Student-t degrees of freedom"))?
                            .sample(rng)
                    } else {
                        1.0
                    };
                    if !precision.is_finite() || precision <= 0.0 {
                        return Err(invalid(
                            "Student-t predictive precision overflowed or underflowed",
                        ));
                    }
                    let observation =
                        mean + normal(rng) * (draw.observation_variance / precision).sqrt();
                    total += observation;
                    if !total.is_finite() || !mean.is_finite() {
                        return Err(invalid("predictive simulation overflowed"));
                    }
                    path.states.push(state.clone());
                    path.components.push(comps);
                    path.means.push(mean);
                    path.observations.push(observation);
                    path.cumulative.push(total);
                }
                Ok(path)
            },
        )?;
        let mut out = StructuralPaths::default();
        for chain in per_draw {
            let (mut states, mut components, mut means, mut observations, mut cumulative) =
                (vec![], vec![], vec![], vec![], vec![]);
            for path in chain {
                states.push(path.states);
                components.push(path.components);
                means.push(path.means);
                observations.push(path.observations);
                cumulative.push(path.cumulative);
            }
            out.states.push(states);
            out.components.push(components);
            out.means.push(means);
            out.observations.push(observations);
            out.cumulative.push(cumulative);
        }
        Ok(out)
    }
    /// Contributions to fitted observation means. Stored states include x[-1],
    /// while returned decompositions start at the first observation x[0].
    pub fn historical_components(&self) -> Result<Vec<Vec<Vec<Vec<f64>>>>> {
        self.validate()?;
        self.chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|draw| {
                        let states = draw.states.as_ref().ok_or_else(|| {
                            invalid("fit with store_states=true to obtain historical components")
                        })?;
                        Ok(self
                            .training_rows
                            .iter()
                            .zip(states.iter().skip(1))
                            .map(|(row, state)| {
                                let mut off = 0;
                                self.config
                                    .components
                                    .iter()
                                    .map(|c| {
                                        let end = off + c.initial_mean.len();
                                        let v = dot(&row[off..end], &state[off..end]);
                                        off = end;
                                        v
                                    })
                                    .collect()
                            })
                            .collect())
                    })
                    .collect()
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedPosterior {
    format: String,
    version: u32,
    posterior: StructuralPosterior,
}
impl StructuralPosterior {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        let d = self.config.dimension();
        let n = self.training_rows.len();
        if self.chains.is_empty()
            || self.chains[0].is_empty()
            || self.chains.iter().any(|c| c.len() != self.chains[0].len())
            || self
                .training_rows
                .iter()
                .any(|r| r.len() != d || r.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid(
                "invalid posterior chain or training row dimensions",
            ));
        }
        allocation(
            "structural posterior",
            &[self.chains.len(), self.chains[0].len(), d, 3],
        )?;
        if self.chains.iter().flatten().any(|d| d.states.is_some()) {
            allocation(
                "structural state history",
                &[self.chains.len(), self.chains[0].len(), d, n + 3],
            )?;
        }
        for draw in self.chains.iter().flatten() {
            if draw.terminal_state.len() != d
                || draw.terminal_state.iter().any(|v| !v.is_finite())
                || draw.variances.len() != d
                || !draw.observation_variance.is_finite()
                || draw.observation_variance <= 0.0
            {
                return Err(invalid("invalid posterior state or variance dimensions"));
            }
            for (value, prior) in draw.variances.iter().zip(self.config.priors()) {
                if !value.is_finite()
                    || *value < 0.0
                    || match prior {
                        VarianceParameter::Fixed(v) => *value != *v,
                        VarianceParameter::InverseGamma { .. } => *value == 0.0,
                    }
                {
                    return Err(invalid(
                        "posterior innovation violates variance specification",
                    ));
                }
            }
            if let VarianceParameter::Fixed(v) = self.config.observation_variance {
                if draw.observation_variance != v {
                    return Err(invalid(
                        "posterior noise violates fixed variance specification",
                    ));
                }
            }
            if let Some(states) = &draw.states {
                if states.len() != n + 1
                    || states
                        .iter()
                        .any(|s| s.len() != d || s.iter().any(|v| !v.is_finite()))
                    || states.last() != Some(&draw.terminal_state)
                {
                    return Err(invalid("invalid historical states"));
                }
            }
        }
        Ok(())
    }
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string(&SavedPosterior {
            format: "rustmc.structural".into(),
            version: 1,
            posterior: self.clone(),
        })
        .map_err(|e| invalid(&e.to_string()))
    }
    pub fn from_json(value: &str) -> Result<Self> {
        let saved: SavedPosterior =
            serde_json::from_str(value).map_err(|e| invalid(&e.to_string()))?;
        if saved.version != 1 || saved.format != "rustmc.structural" {
            return Err(invalid("unsupported structural posterior format/version"));
        }
        saved.posterior.validate()?;
        Ok(saved.posterior)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedModel {
    format: String,
    version: u32,
    model: StructuralConfig,
}
impl StructuralConfig {
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string(&SavedModel {
            format: "rustmc.structural.model".into(),
            version: 1,
            model: self.clone(),
        })
        .map_err(|e| invalid(&e.to_string()))
    }
    pub fn from_json(value: &str) -> Result<Self> {
        let saved: SavedModel = serde_json::from_str(value).map_err(|e| invalid(&e.to_string()))?;
        if saved.version != 1 || saved.format != "rustmc.structural.model" {
            return Err(invalid("unsupported structural model format/version"));
        }
        saved.model.validate()?;
        Ok(saved.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_forecast_seed_domains_are_distinct() {
        assert_ne!(
            chain_seed(42, 0, FIT_SEED_DOMAIN),
            chain_seed(42, 0, FORECAST_SEED_DOMAIN)
        );
        // The keys the three sampling roles actually build, for one caller seed:
        // `fit` and `forecast` key per chain, while `prior_predict` keys its
        // prior draws once and then hands that key to `forecast`.
        for seed in [0, 1, 42, 491, u64::MAX] {
            let prior = chain_seed(seed, 0, STRUCTURAL_PRIOR_SEED_DOMAIN);
            for chain in 0..4 {
                let keys = [
                    chain_seed(seed, chain, FIT_SEED_DOMAIN),
                    chain_seed(seed, chain, FORECAST_SEED_DOMAIN),
                    prior,
                    chain_seed(prior, chain, FORECAST_SEED_DOMAIN),
                ];
                for (i, left) in keys.iter().enumerate() {
                    for right in &keys[i + 1..] {
                        assert_ne!(
                            left, right,
                            "seed {seed} chain {chain} reuses one stream for two roles"
                        );
                    }
                }
            }
        }
    }

    fn fixed(v: f64) -> VarianceParameter {
        VarianceParameter::Fixed(v)
    }
    fn settings(draws: usize) -> SamplingConfig {
        SamplingConfig {
            chains: 1,
            draws,
            warmup: 100,
            thinning: 1,
            seed: 491,
            store_states: true,
        }
    }
    fn level(q: f64, r: f64) -> StructuralConfig {
        StructuralConfig {
            components: vec![Component::level("level".into(), fixed(q), 0.0, 2.0)],
            observation_variance: fixed(r),
            student_df: None,
        }
    }
    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }
    fn covariance(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len() as f64 - mean(a) * mean(b)
    }
    #[test]
    fn thinning_keeps_the_last_sweep_of_each_block_like_every_other_sampler() {
        // With warmup w and thinning k, every Gibbs sampler in the crate keeps
        // sweeps w + k - 1, w + 2k - 1, ..., ending on the final sweep. A
        // thinned fit is therefore every k-th draw of the unthinned fit with
        // the same seed, starting from the (k-1)-th.
        let mut c = level(0.1, 0.5);
        c.components[0].innovations[0] = VarianceParameter::InverseGamma {
            shape: 3.0,
            scale: 0.2,
        };
        let y = [0.3, -0.1, 0.4, f64::NAN, 0.8, 0.5];
        let mut every = settings(12);
        every.warmup = 2;
        every.chains = 2;
        let mut thinned = every.clone();
        thinned.draws = 4;
        thinned.thinning = 3;
        let all = fit(&y, None, &c, &every).unwrap();
        let kept = fit(&y, None, &c, &thinned).unwrap();
        for (all, kept) in all.chains.iter().zip(&kept.chains) {
            let expected: Vec<_> = [2, 5, 8, 11]
                .iter()
                .map(|&index| all[index].variances.clone())
                .collect();
            let actual: Vec<_> = kept.iter().map(|draw| draw.variances.clone()).collect();
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn gaussian_posterior_and_joint_forecast_match_independent_kalman_moments() {
        // The reference used to be `model.smooth` and `model.forecast`, which is
        // the same `build` and the same filter the sampler runs: a shared error
        // in either would have moved the draws and the expected answer together,
        // so the test showed internal consistency and not the independence its
        // name claims. The fixture is small enough to write the Kalman recursion
        // out in closed form instead.
        //
        // `x0 ~ N(0, 2)`, `x_t = x_{t-1} + N(0, 3/10)`, `y_t = x_t + N(0, 7/10)`,
        // with `y = [1, missing, 2]`. Predicting to the first observation gives
        // variance 2 + 3/10 = 23/10 and innovation variance 23/10 + 7/10 = 3, so
        // the filtered mean is (23/30) * 1 and the filtered variance is
        // (23/10)(7/10)/3 = 161/300. Two further transitions with no update in
        // between carry that to mean 23/30 and variance 161/300 + 6/10 = 341/300;
        // the second update has innovation variance 341/300 + 7/10 = 551/300, so
        //
        //   terminal mean     = 23/30 + (341/551)(2 - 23/30) = 843/551
        //   terminal variance = (341/300)(7/10)/(551/300)    = 2387/5510
        //
        // and, the state being a driftless random walk, every predictive mean is
        // the terminal mean while
        //
        //   Cov(y_{T+i}, y_{T+j}) = P_T + (3/10) min(i, j) + (7/10) [i = j].
        const TERMINAL_MEAN: f64 = 843.0 / 551.0;
        const TERMINAL_VARIANCE: f64 = 2387.0 / 5510.0;
        let c = level(0.3, 0.7);
        let y = [1.0, f64::NAN, 2.0];
        let post = fit(&y, None, &c, &settings(12000)).unwrap();
        let final_states = post.chains[0]
            .iter()
            .map(|d| d.terminal_state[0])
            .collect::<Vec<_>>();
        assert!((mean(&final_states) - TERMINAL_MEAN).abs() < 0.025);
        assert!((covariance(&final_states, &final_states) - TERMINAL_VARIANCE).abs() < 0.025);
        let paths = post.forecast(3, None, 827).unwrap();
        for i in 0..3 {
            let a = paths.observations[0]
                .iter()
                .map(|p| p[i])
                .collect::<Vec<_>>();
            assert!((mean(&a) - TERMINAL_MEAN).abs() < 0.055);
            for j in 0..3 {
                let b = paths.observations[0]
                    .iter()
                    .map(|p| p[j])
                    .collect::<Vec<_>>();
                let expected = TERMINAL_VARIANCE
                    + 0.3 * (i.min(j) + 1) as f64
                    + if i == j { 0.7 } else { 0.0 };
                assert!((covariance(&a, &b) - expected).abs() < 0.065);
            }
        }
        let saved = post.to_json().unwrap();
        let loaded = StructuralPosterior::from_json(&saved).unwrap();
        assert!(
            paths == loaded.forecast(3, None, 827).unwrap(),
            "JSON persistence must preserve seeded predictions exactly"
        );
    }
    #[test]
    fn harmonic_damping_regression_components_add_and_retain_static_parameters() {
        let c = StructuralConfig {
            components: vec![
                Component::trend(
                    "trend".into(),
                    0.8,
                    fixed(0.1),
                    fixed(0.01),
                    vec![1.0, 0.2],
                    vec![1.0, 0.0, 0.0, 0.1],
                )
                .unwrap(),
                Component::seasonal("weekly".into(), 7.2, 2, fixed(0.0), 0.3).unwrap(),
                Component::seasonal("yearly".into(), 365.25, 1, fixed(0.002), 0.1).unwrap(),
                Component::regression("static".into(), vec![2.0], vec![0.3], vec![fixed(0.0)]),
                Component::regression("dynamic".into(), vec![-1.0], vec![0.2], vec![fixed(0.05)]),
                Component::ar(
                    "residual".into(),
                    vec![0.5, -0.2],
                    fixed(0.2),
                    vec![0.0; 2],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .unwrap(),
            ],
            observation_variance: fixed(0.2),
            student_df: None,
        };
        let exog = vec![vec![1.0, 2.0]; 5];
        let p = c.prior_predict(5, Some(&exog), 200, 8).unwrap();
        for draw in 0..200 {
            for t in 0..5 {
                assert!(
                    (p.components[0][draw][t].iter().sum::<f64>() - p.means[0][draw][t]).abs()
                        < 1e-12
                );
                assert_eq!(p.states[0][draw][t][8], p.states[0][draw][0][8]);
            }
        }
        let posterior = fit(&[1.0, 0.5, 0.4, 1.2, 2.0], Some(&exog), &c, &settings(5)).unwrap();
        assert_eq!(posterior.historical_components().unwrap()[0][0].len(), 5);
    }
    #[test]
    fn student_mixture_resists_an_outlier_and_has_heavy_predictive_tails() {
        // The bulk of the series sits at 2.0, not at 0.0. With a bulk of zero
        // the robust check below was `|mean| < 0.15` around a location prior of
        // Normal(0, 2), whose mean is 0: a Student branch that ignored its
        // observations outright passed it, since the average of 1200 prior
        // draws lands inside that window with probability 0.9998.
        let mut c = level(0.0, 1.0);
        let mut y = vec![2.0; 20];
        y[9] = 37.0;
        let gaussian = fit(&y, None, &c, &settings(1200)).unwrap();
        c.student_df = Some(4.0);
        let robust = fit(&y, None, &c, &settings(1200)).unwrap();
        let average = |p: &StructuralPosterior| {
            p.chains[0].iter().map(|d| d.terminal_state[0]).sum::<f64>() / 1200.0
        };
        // The Gaussian fit is dragged well above the bulk by the outlier; the
        // Student fit stays on it. Both bounds exclude the prior mean of 0.
        assert!(average(&gaussian) > 3.0, "{}", average(&gaussian));
        assert!(
            (average(&robust) - 2.0).abs() < 0.15,
            "{}",
            average(&robust)
        );
        let p = c.prior_predict(1, None, 18000, 942).unwrap();
        let residuals = p.observations[0]
            .iter()
            .zip(&p.means[0])
            .map(|(y, m)| y[0] - m[0])
            .collect::<Vec<_>>();
        assert!((covariance(&residuals, &residuals) - 2.0).abs() < 0.2);
        assert!(residuals.iter().filter(|x| x.abs() > 4.0).count() > 140);
    }
    #[test]
    fn inverse_gamma_prior_prediction_integrates_parameter_uncertainty() {
        let mut c = level(0.0, 1.0);
        c.components[0].innovations[0] = VarianceParameter::InverseGamma {
            shape: 5.0,
            scale: 2.0,
        };
        c.observation_variance = VarianceParameter::InverseGamma {
            shape: 4.0,
            scale: 3.0,
        };
        let paths = c.prior_predict(2, None, 120000, 48).unwrap();
        let a = paths.observations[0]
            .iter()
            .map(|p| p[0])
            .collect::<Vec<_>>();
        let b = paths.observations[0]
            .iter()
            .map(|p| p[1])
            .collect::<Vec<_>>();
        // These three are linear in q and r, so they equal 2 + E[q] + E[r],
        // 2 + E[q] and 2 + 2 E[q] + E[r] whether the variances are drawn from
        // their priors or pinned at their means (1/2 and 1). On their own they
        // do not test what the name claims.
        assert!((covariance(&a, &a) - 3.5).abs() < 0.14);
        assert!((covariance(&a, &b) - 2.5).abs() < 0.14);
        assert!((covariance(&b, &b) - 4.0).abs() < 0.14);
        // The statistic that separates the two is any nonlinear one. The latent
        // increment between the two steps is exactly Normal(0, q), so
        //
        //   E|dx| = sqrt(2/pi) E[sqrt q] = sqrt(2/pi) sqrt(b) G(a - 1/2)/G(a)
        //         = (2/sqrt pi)(105/16) sqrt(pi) / 24 = 35/64 = 0.546875,
        //
        // against sqrt(E[q] 2/pi) = 1/sqrt(pi) = 0.564190 if q were pinned at
        // its prior mean of 1/2. The window below excludes the pinned value by
        // 1.9 tolerance-widths, and is 4.6 Monte Carlo standard errors wide at
        // this draw count. The fourth moment separates them further but its own
        // eighth moment is 70, so it cannot be calibrated this tightly.
        let increments = paths.means[0]
            .iter()
            .map(|m| m[1] - m[0])
            .collect::<Vec<_>>();
        let absolute = increments.iter().map(|d| d.abs()).sum::<f64>() / increments.len() as f64;
        assert!(
            (absolute - 35.0 / 64.0).abs() < 0.006,
            "mean absolute increment {absolute} does not separate an integrated q from one \
             pinned at its prior mean, which would give 0.564190"
        );
    }
    #[test]
    fn static_regression_posterior_matches_analytic_normal_update() {
        let c = StructuralConfig {
            components: vec![Component::regression(
                "beta".into(),
                vec![0.0],
                vec![4.0],
                vec![fixed(0.0)],
            )],
            observation_variance: fixed(1.0),
            student_df: None,
        };
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let p = fit(&[2.0, 4.0, 6.0], Some(&x), &c, &settings(7000)).unwrap();
        let beta = p.chains[0]
            .iter()
            .map(|d| d.terminal_state[0])
            .collect::<Vec<_>>();
        assert!((mean(&beta) - 28.0 / 14.25).abs() < 0.015);
        assert!((covariance(&beta, &beta) - 1.0 / 14.25).abs() < 0.004);
    }
    #[test]
    fn inferred_variance_matches_independent_marginal_quadrature() {
        // One observation lets us integrate out the latent initial/terminal state:
        // y ~ Normal(0, initial_variance + q + r). Integrate the remaining
        // inverse-gamma variance in log coordinates without using the sampler.
        //
        // The observation is 4.0 rather than 2.0 because at 2.0 the quadrature
        // answers were 0.760 and 0.751 against an InverseGamma(3, 1.5) prior
        // whose mean is 0.75: both sat inside the +/- 0.045 window below, so a
        // Gibbs step that ignored the observation entirely would have passed.
        // At 4.0 they are 1.082 and 0.985. The negative control below keeps it
        // that way.
        for infer_process in [false, true] {
            let mut c = level(0.0, 0.4);
            let prior = VarianceParameter::InverseGamma {
                shape: 3.0,
                scale: 1.5,
            };
            let known = if infer_process {
                c.components[0].innovations[0] = prior;
                2.4
            } else {
                c.observation_variance = prior;
                2.0
            };
            let mut weights = 0.0;
            let mut moment = 0.0;
            for k in 0..20000 {
                let logv = -12.0 + 20.0 * k as f64 / 19999.0;
                let v = logv.exp();
                let logw =
                    -3.0 * logv - 1.5 / v - 0.5 * (known + v).ln() - 16.0 / (2.0 * (known + v));
                let w = logw.exp();
                weights += w;
                moment += w * v;
            }
            let mut sampling = settings(18000);
            sampling.warmup = 500;
            let p = fit(&[4.0], None, &c, &sampling).unwrap();
            let values = p.chains[0]
                .iter()
                .map(|d| {
                    if infer_process {
                        d.variances[0]
                    } else {
                        d.observation_variance
                    }
                })
                .collect::<Vec<_>>();
            let reference = moment / weights;
            let prior_mean = 1.5 / (3.0 - 1.0);
            assert!(
                (reference - prior_mean).abs() > 2.0 * 0.045,
                "infer_process={infer_process}: the quadrature reference {reference} is within \
                 two tolerances of the prior mean {prior_mean}, so the check below would pass \
                 on a sampler that ignored the observation"
            );
            assert!(
                (mean(&values) - reference).abs() < 0.045,
                "infer_process={infer_process}, mean={}, expected={reference}",
                mean(&values),
            );
        }
    }
    #[test]
    fn dynamic_regression_tracks_simulated_coefficients_and_chains_ignore_pool_size() {
        let c = StructuralConfig {
            components: vec![Component::regression(
                "dynamic".into(),
                vec![0.0],
                vec![0.2],
                vec![fixed(0.015)],
            )],
            observation_variance: fixed(0.02),
            student_df: None,
        };
        // The series and the truth it is scored against used to come from
        // `StructuralConfig::prior_predict`, so the module was being compared
        // against its own forward simulator: a config misread shared by both
        // would have cancelled out. The generative model is three lines, so it
        // is written out here instead. `Component::regression` above declares a
        // coefficient starting at N(0, 0.2) and random-walking with innovation
        // variance 0.015, observed through a design column of ones with
        // observation variance 0.02.
        const STEPS: usize = 120;
        let mut rng = ChaCha8Rng::seed_from_u64(908);
        let mut coefficient = normal(&mut rng) * 0.2_f64.sqrt();
        let mut truth = Vec::with_capacity(STEPS);
        let mut series = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            coefficient += normal(&mut rng) * 0.015_f64.sqrt();
            truth.push(coefficient);
            series.push(coefficient + normal(&mut rng) * 0.02_f64.sqrt());
        }
        let x = vec![vec![1.0]; STEPS];
        let y = &series;
        let mut sampling = settings(200);
        sampling.chains = 2;
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| fit(y, Some(&x), &c, &sampling))
            .unwrap();
        let two = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap()
            .install(|| fit(y, Some(&x), &c, &sampling))
            .unwrap();
        assert!(one.to_json().unwrap() == two.to_json().unwrap());
        let smoothed = (0..STEPS)
            .map(|t| {
                one.chains
                    .iter()
                    .flatten()
                    .map(|d| d.states.as_ref().unwrap()[t + 1][0])
                    .sum::<f64>()
                    / 400.0
            })
            .collect::<Vec<_>>();
        let rmse = |estimates: &[f64]| {
            (estimates
                .iter()
                .zip(&truth)
                .map(|(e, t)| (e - t).powi(2))
                .sum::<f64>()
                / STEPS as f64)
                .sqrt()
        };
        let error = rmse(&smoothed);
        // Negative control: a fit that ignored the series would report the
        // coefficient's prior mean of zero at every step. The smoother has to
        // keep well under a quarter of that error, so the bound below cannot be
        // met without reading the data.
        let uninformed = rmse(&vec![0.0; STEPS]);
        assert!(
            error < 0.15 && error < 0.25 * uninformed,
            "state RMSE {error} against an uninformed RMSE of {uninformed}"
        );
    }
    #[test]
    fn validation_rejects_aliases_invalid_variances_and_corrupt_persistence() {
        assert!(Component::seasonal("s".into(), 4.0, 2, fixed(0.0), 1.0).is_err());
        assert!(Component::ar("ar".into(), vec![1.2], fixed(1.0), vec![0.0], vec![1.0]).is_err());
        let c = level(0.1, 1.0);
        let mut p = fit(&[0.0], None, &c, &settings(2)).unwrap();
        p.chains[0][0].variances[0] = 0.2;
        assert!(p.forecast(1, None, 1).is_err());
        assert!(StructuralConfig::from_json(
            &c.to_json()
                .unwrap()
                .replace("\"version\":1", "\"version\":2")
        )
        .is_err());
    }
    #[test]
    fn student_static_location_matches_independent_likelihood_quadrature() {
        let mut c = level(0.0, 0.8);
        let df = 3.5;
        c.student_df = Some(df);
        let y = [0.0, 1.0, 4.0, 10.0];
        let mut mass = 0.;
        let mut first = 0.;
        let mut second = 0.;
        for i in 0..20001 {
            let x = -10. + i as f64 * 0.001;
            let log_weight = -x * x / 4.
                - y.iter()
                    .map(|value| 0.5 * (df + 1.) * ((value - x).powi(2) / (df * 0.8)).ln_1p())
                    .sum::<f64>();
            let weight = log_weight.exp();
            mass += weight;
            first += weight * x;
            second += weight * x * x;
        }
        let expected_mean = first / mass;
        let expected_variance = second / mass - expected_mean.powi(2);
        let mut sampling = settings(25000);
        sampling.warmup = 1000;
        let posterior = fit(&y, None, &c, &sampling).unwrap();
        let values: Vec<_> = posterior.chains[0]
            .iter()
            .map(|d| d.terminal_state[0])
            .collect();
        assert!((mean(&values) - expected_mean).abs() < 0.03);
        assert!((covariance(&values, &values) - expected_variance).abs() < 0.035);
        assert_eq!(posterior.diagnostics().params.len(), 3);
    }
    #[test]
    fn oversized_structural_requests_and_undefined_student_means_are_rejected() {
        assert!(Component::seasonal("large".into(), 1e20, 1 << 20, fixed(0.1), 1.).is_err());
        assert!(Component::ar("large".into(), vec![0.; 4000], fixed(0.1), vec![], vec![]).is_err());
        let mut c = level(0.1, 1.);
        for df in [0., 0.5, 1., f64::INFINITY] {
            c.student_df = Some(df);
            assert!(c.validate().is_err());
        }
        c.student_df = Some(1.01);
        assert!(c.validate().is_ok());
        assert!(c.prior_predict(usize::MAX, None, 2, 1).is_err());
        let mut sampling = settings(usize::MAX);
        sampling.warmup = 0;
        assert!(fit(&[0.], None, &c, &sampling).is_err());
        let posterior = fit(&[0.], None, &c, &settings(2)).unwrap();
        assert!(posterior.forecast(usize::MAX, None, 1).is_err());
    }
}
