//! Composable structural models with exact Gaussian FFBS and conjugate variance
//! updates. Student-t observations use Gamma precision mixtures with fixed df.
//! Initial state priors are independent of innovation variances; x[-1] is included
//! in every state draw so every transition contributes to the variance update.
use crate::state_space::{LinearGaussianStateSpace, StateSpaceError};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
type Result<T> = std::result::Result<T, StateSpaceError>;
const MAX_VALUES: usize = 25_000_000;
fn invalid(s: &str) -> StateSpaceError {
    StateSpaceError::InvalidParameter(s.into())
}
fn allocation(factors: &[usize]) -> Result<()> {
    let count = factors
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| invalid("structural allocation size overflow"))?;
    if count > MAX_VALUES {
        return Err(invalid(
            "structural request exceeds 25 million working or retained values",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    fn sample<R: Rng + ?Sized>(&self, n: usize, ss: f64, rng: &mut R) -> Result<f64> {
        let value = match *self {
            Self::Fixed(v) => v,
            Self::InverseGamma { shape, scale } => {
                let g = Gamma::new(shape + n as f64 / 2.0, 1.0 / (scale + ss / 2.0))
                    .map_err(|_| invalid("invalid variance conditional"))?;
                1.0 / g.sample(rng)
            }
        };
        if !value.is_finite() || value < 0.0 || (value == 0.0 && !matches!(self, Self::Fixed(0.0)))
        {
            return Err(invalid("variance draw overflowed or underflowed"));
        }
        Ok(value)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        allocation(&[square, 3])?;
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
        allocation(&[d, d, 3])?;
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
pub struct StructuralDraw {
    pub variances: Vec<f64>,
    pub observation_variance: f64,
    pub terminal_state: Vec<f64>,
    pub states: Option<Vec<Vec<f64>>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralPosterior {
    pub config: StructuralConfig,
    pub chains: Vec<Vec<StructuralDraw>>,
    pub training_rows: Vec<Vec<f64>>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
            allocation(&[d, d, 4])?;
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
        allocation(&[self.dimension(), self.dimension(), 4])?;
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
        allocation(&[draws, self.dimension(), 3])?;
        allocation(&[draws, steps, self.dimension() + self.components.len() + 3])?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut chain = vec![];
        for _ in 0..draws {
            let q = self
                .priors()
                .iter()
                .map(|p| p.sample(0, 0.0, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let r = self.observation_variance.sample(0, 0.0, &mut rng)?;
            let state = self.build(&q, r)?.simulate_initial(&mut rng)?;
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
        .forecast(steps, design, seed.wrapping_add(0x5052494f52))
    }
}
fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    StandardNormal.sample(rng)
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn chain_seed(seed: u64, chain: usize) -> u64 {
    let mut x = seed.wrapping_add((chain as u64).wrapping_mul(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
pub fn fit(
    y: &[f64],
    design: Option<&[Vec<f64>]>,
    config: &StructuralConfig,
    sampling: &SamplingConfig,
) -> Result<StructuralPosterior> {
    config.validate()?;
    if y.is_empty()
        || y.iter().any(|v| v.is_infinite())
        || sampling.chains == 0
        || sampling.draws == 0
        || sampling.thinning == 0
    {
        return Err(invalid(
            "nonempty finite/NaN data and positive chains, draws and thinning required",
        ));
    }
    let iterations = sampling
        .draws
        .checked_mul(sampling.thinning)
        .and_then(|n| n.checked_add(sampling.warmup))
        .ok_or_else(|| invalid("iteration count overflow"))?;
    let d = config.dimension();
    allocation(&[sampling.chains, y.len() + 1, d, d, 6])?;
    allocation(&[
        sampling.chains,
        sampling.draws,
        d,
        if sampling.store_states {
            y.len() + 3
        } else {
            3
        },
    ])?;
    let rows = config.observation_rows(y.len(), design)?;
    let priors = config.priors();
    let observed = y.iter().filter(|v| v.is_finite()).count();
    let chains = (0..sampling.chains)
        .into_par_iter()
        .map(|chain| -> Result<Vec<StructuralDraw>> {
            let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(sampling.seed, chain));
            let mut q = priors.iter().map(|p| p.initial()).collect::<Vec<_>>();
            let mut r = config.observation_variance.initial();
            let mut lambda = vec![1.0; y.len()];
            let mut draws = vec![];
            for iteration in 0..iterations {
                let model = config
                    .build(&q, r)?
                    .with_observation_rows(rows.clone())?
                    .with_observation_variances(lambda.iter().map(|l| r / l).collect())?;
                let states = model.sample_states_ffbs(y, &mut rng)?;
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
                    q[i] = priors[i].sample(y.len(), ss, &mut rng)?;
                }
                let ss = y
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(t, v)| lambda[t] * (v - dot(&rows[t], &states[t + 1])).powi(2))
                    .sum();
                r = config.observation_variance.sample(observed, ss, &mut rng)?;
                if let Some(df) = config.student_df {
                    for (t, v) in y.iter().enumerate() {
                        if v.is_finite() {
                            let e = v - dot(&rows[t], &states[t + 1]);
                            lambda[t] = Gamma::new((df + 1.0) / 2.0, 2.0 / (df + e * e / r))
                                .map_err(|_| invalid("invalid Student-t conditional"))?
                                .sample(&mut rng);
                            if !lambda[t].is_finite() || lambda[t] <= 0.0 {
                                return Err(invalid(
                                    "Student-t precision draw overflowed or underflowed",
                                ));
                            }
                        }
                    }
                }
                if iteration >= sampling.warmup
                    && (iteration - sampling.warmup).is_multiple_of(sampling.thinning)
                {
                    draws.push(StructuralDraw {
                        variances: q.clone(),
                        observation_variance: r,
                        terminal_state: states[y.len()].clone(),
                        states: if sampling.store_states {
                            Some(states)
                        } else {
                            None
                        },
                    });
                }
            }
            Ok(draws)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StructuralPosterior {
        config: config.clone(),
        chains,
        training_rows: rows,
    })
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
        allocation(&[
            self.chains.len(),
            self.chains[0].len(),
            steps,
            self.config.dimension() + self.config.components.len() + 3,
        ])?;
        let rows = self.config.observation_rows(steps, design)?;
        let mut out = StructuralPaths::default();
        for (chain, draws) in self.chains.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(seed, chain));
            let (mut all_s, mut all_c, mut all_m, mut all_o, mut all_t) =
                (vec![], vec![], vec![], vec![], vec![]);
            for draw in draws {
                let model = self
                    .config
                    .build(&draw.variances, draw.observation_variance)?;
                let mut state = draw.terminal_state.clone();
                let (mut ss, mut cc, mut mm, mut oo, mut tt) =
                    (vec![], vec![], vec![], vec![], vec![]);
                let mut total = 0.0;
                for row in &rows {
                    state = model.simulate_transition(&state, &mut rng)?;
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
                            .sample(&mut rng)
                    } else {
                        1.0
                    };
                    if !precision.is_finite() || precision <= 0.0 {
                        return Err(invalid(
                            "Student-t predictive precision overflowed or underflowed",
                        ));
                    }
                    let observation =
                        mean + normal(&mut rng) * (draw.observation_variance / precision).sqrt();
                    total += observation;
                    if !total.is_finite() || !mean.is_finite() {
                        return Err(invalid("predictive simulation overflowed"));
                    }
                    ss.push(state.clone());
                    cc.push(comps);
                    mm.push(mean);
                    oo.push(observation);
                    tt.push(total);
                }
                all_s.push(ss);
                all_c.push(cc);
                all_m.push(mm);
                all_o.push(oo);
                all_t.push(tt);
            }
            out.states.push(all_s);
            out.components.push(all_c);
            out.means.push(all_m);
            out.observations.push(all_o);
            out.cumulative.push(all_t);
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
        allocation(&[self.chains.len(), self.chains[0].len(), d, 3])?;
        if self.chains.iter().flatten().any(|d| d.states.is_some()) {
            allocation(&[self.chains.len(), self.chains[0].len(), d, n + 3])?;
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
    fn gaussian_posterior_and_joint_forecast_match_independent_kalman_moments() {
        let c = level(0.3, 0.7);
        let y = [1.0, f64::NAN, 2.0];
        let model = c.build(&[0.3], 0.7).unwrap();
        let smooth = model.smooth(&y).unwrap();
        let f = model.forecast(&y, 3).unwrap();
        let post = fit(&y, None, &c, &settings(12000)).unwrap();
        let final_states = post.chains[0]
            .iter()
            .map(|d| d.terminal_state[0])
            .collect::<Vec<_>>();
        assert!((mean(&final_states) - smooth.smoothed_means[2][0]).abs() < 0.025);
        assert!(
            (covariance(&final_states, &final_states) - smooth.smoothed_covariances[2][0]).abs()
                < 0.025
        );
        let paths = post.forecast(3, None, 827).unwrap();
        for i in 0..3 {
            let a = paths.observations[0]
                .iter()
                .map(|p| p[i])
                .collect::<Vec<_>>();
            assert!((mean(&a) - f.observation_means[i]).abs() < 0.055);
            for j in 0..3 {
                let b = paths.observations[0]
                    .iter()
                    .map(|p| p[j])
                    .collect::<Vec<_>>();
                assert!((covariance(&a, &b) - f.observation_covariance[i * 3 + j]).abs() < 0.065);
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
        let mut c = level(0.0, 1.0);
        let mut y = vec![0.0; 20];
        y[9] = 35.0;
        let gaussian = fit(&y, None, &c, &settings(1200)).unwrap();
        c.student_df = Some(4.0);
        let robust = fit(&y, None, &c, &settings(1200)).unwrap();
        let average = |p: &StructuralPosterior| {
            p.chains[0].iter().map(|d| d.terminal_state[0]).sum::<f64>() / 1200.0
        };
        assert!(average(&gaussian) > 1.5);
        assert!(average(&robust).abs() < 0.15);
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
        let paths = c.prior_predict(2, None, 18000, 48).unwrap();
        let a = paths.observations[0]
            .iter()
            .map(|p| p[0])
            .collect::<Vec<_>>();
        let b = paths.observations[0]
            .iter()
            .map(|p| p[1])
            .collect::<Vec<_>>();
        assert!((covariance(&a, &a) - 3.5).abs() < 0.14);
        assert!((covariance(&a, &b) - 2.5).abs() < 0.14);
        assert!((covariance(&b, &b) - 4.0).abs() < 0.14);
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
                    -3.0 * logv - 1.5 / v - 0.5 * (known + v).ln() - 4.0 / (2.0 * (known + v));
                let w = logw.exp();
                weights += w;
                moment += w * v;
            }
            let mut sampling = settings(18000);
            sampling.warmup = 500;
            let p = fit(&[2.0], None, &c, &sampling).unwrap();
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
            assert!(
                (mean(&values) - moment / weights).abs() < 0.045,
                "infer_process={infer_process}, mean={}, expected={}",
                mean(&values),
                moment / weights
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
        let x = vec![vec![1.0]; 60];
        let simulated = c.prior_predict(60, Some(&x), 1, 908).unwrap();
        let y = &simulated.observations[0][0];
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
        let error = (0..60)
            .map(|t| {
                let estimate = one
                    .chains
                    .iter()
                    .flatten()
                    .map(|d| d.states.as_ref().unwrap()[t + 1][0])
                    .sum::<f64>()
                    / 400.0;
                (estimate - simulated.states[0][0][t][0]).powi(2)
            })
            .sum::<f64>()
            / 60.0;
        assert!(error.sqrt() < 0.15);
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
