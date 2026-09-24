//! Joint latent-Gaussian dynamic regressions, with block elliptical slice inference.
//! Gaussian prior scales and NB dispersion are fixed model specifications.
//! Group coefficients equal an uncertain population coefficient plus a Gaussian
//! deviation; group random walks can additionally share a common random walk.
use crate::autodiff::softplus;
use crate::bayesian_forecast::BayesianForecastError as Error;
use crate::observation::sigmoid;
use crate::seeding::chain_seed;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub type Panel = Vec<Vec<f64>>;
pub type Design = Vec<Vec<Vec<f64>>>;
pub type Paths = Vec<Vec<Panel>>;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Family {
    Poisson,
    NegativeBinomial,
    HurdleLogNormal,
    Gaussian,
}

/// `Family` above is left alone deliberately: all its variants are unit
/// variants, so `deny_unknown_fields` is inert on it, and serde already refuses
/// an unrecognised family name.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicGlmConfig {
    pub family: Family,
    pub initial_mean: f64,
    pub occurrence_initial_mean: f64,
    pub coefficient_sd: f64,
    pub group_sd: f64,
    pub process_sd: f64,
    pub shared_process_sd: f64,
    pub observation_sd: f64,
    pub dispersion: f64,
    pub chains: usize,
    pub draws: usize,
    pub warmup: usize,
    pub thin: usize,
    pub seed: u64,
}
impl Default for DynamicGlmConfig {
    fn default() -> Self {
        Self {
            family: Family::Poisson,
            initial_mean: 0.,
            occurrence_initial_mean: 0.,
            coefficient_sd: 1.,
            group_sd: 0.5,
            process_sd: 0.1,
            shared_process_sd: 0.,
            observation_sd: 1.,
            dispersion: 5.,
            chains: 4,
            draws: 1000,
            warmup: 1000,
            thin: 1,
            seed: 42,
        }
    }
}
impl DynamicGlmConfig {
    pub fn validate(&self) -> Result<(), Error> {
        for (name, x) in [
            ("coefficient_sd", self.coefficient_sd),
            ("observation_sd", self.observation_sd),
            ("dispersion", self.dispersion),
        ] {
            if !x.is_finite() || x <= 0. {
                return Err(invalid(format!("{name} must be finite and positive")));
            }
        }
        for (name, x) in [
            ("group_sd", self.group_sd),
            ("process_sd", self.process_sd),
            ("shared_process_sd", self.shared_process_sd),
        ] {
            if !x.is_finite() || x < 0. {
                return Err(invalid(format!("{name} must be finite and nonnegative")));
            }
        }
        if !self.initial_mean.is_finite()
            || !self.occurrence_initial_mean.is_finite()
            || self.chains == 0
            || self.draws == 0
            || self.thin == 0
        {
            return Err(invalid(
                "finite initial means and positive chains/draws/thin required",
            ));
        }
        self.draws
            .checked_mul(self.thin)
            .and_then(|n| n.checked_add(self.warmup))
            .ok_or_else(|| invalid("iteration count overflow"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicGlmDraw {
    /// [component][coefficient], including intercept at column zero.
    pub population_coefficients: Vec<Vec<f64>>,
    /// [component][group][coefficient].
    pub coefficients: Vec<Panel>,
    /// [component][group][training time], excludes regression contribution.
    pub states: Vec<Panel>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicGlmPosterior {
    pub config: DynamicGlmConfig,
    pub chains: Vec<Vec<DynamicGlmDraw>>,
    pub groups: usize,
    pub time_count: usize,
    pub features: usize,
    /// Includes warmup and all thinning transitions.
    pub likelihood_evaluations: Vec<usize>,
    pub observed_count: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicGlmForecast {
    /// All arrays are [chain][draw][group][horizon].
    pub mean_paths: Paths,
    pub observation_paths: Paths,
    /// Hurdle only; other families return empty arrays.
    pub occurrence_paths: Paths,
    pub positive_mean_paths: Paths,
}

/// One elliptical-slice block and the likelihood terms it can move.
struct Block {
    range: std::ops::Range<usize>,
    component: usize,
    /// `None` for the population and shared-walk blocks, which touch every
    /// group; a group's own blocks touch only that group's observations.
    group: Option<usize>,
}

struct Layout {
    groups: usize,
    times: usize,
    k: usize,
    components: usize,
}
impl Layout {
    fn checked(groups: usize, times: usize, k: usize, components: usize) -> Result<Self, Error> {
        groups
            .checked_add(1)
            .and_then(|blocks| {
                k.checked_add(times)
                    .and_then(|width| width.checked_mul(blocks))
            })
            .and_then(|width| width.checked_mul(components))
            .ok_or_else(|| invalid("latent layout dimension overflow"))?;
        Ok(Self {
            groups,
            times,
            k,
            components,
        })
    }
    fn width(&self) -> usize {
        self.k * (1 + self.groups) + self.times * (1 + self.groups)
    }
    fn size(&self) -> usize {
        self.width() * self.components
    }
    /// Per component: the population coefficients, each group's coefficient
    /// deviations, the shared walk's innovations, then each group's.
    fn blocks(&self) -> Vec<Block> {
        let mut out = Vec::new();
        for component in 0..self.components {
            let mut offset = component * self.width();
            for width in [self.k, self.times] {
                for slot in 0..=self.groups {
                    out.push(Block {
                        range: offset..offset + width,
                        component,
                        group: slot.checked_sub(1),
                    });
                    offset += width;
                }
            }
        }
        out
    }
    fn decode(&self, z: &[f64], cfg: &DynamicGlmConfig) -> DynamicGlmDraw {
        let mut population_coefficients = Vec::new();
        let mut coefficients = Vec::new();
        let mut states = Vec::new();
        for c in 0..self.components {
            let base = c * self.width();
            let mut population: Vec<f64> = z[base..base + self.k]
                .iter()
                .map(|x| x * cfg.coefficient_sd)
                .collect();
            population[0] += if c == 1 {
                cfg.occurrence_initial_mean
            } else {
                cfg.initial_mean
            };
            let group_beta = (0..self.groups)
                .map(|g| {
                    (0..self.k)
                        .map(|j| population[j] + cfg.group_sd * z[base + self.k * (g + 1) + j])
                        .collect()
                })
                .collect();
            let rw = base + self.k * (1 + self.groups);
            let group_states = (0..self.groups)
                .map(|g| {
                    let mut level = 0.;
                    (0..self.times)
                        .map(|t| {
                            level += cfg.shared_process_sd * z[rw + t]
                                + cfg.process_sd * z[rw + (g + 1) * self.times + t];
                            level
                        })
                        .collect()
                })
                .collect();
            population_coefficients.push(population);
            coefficients.push(group_beta);
            states.push(group_states);
        }
        DynamicGlmDraw {
            population_coefficients,
            coefficients,
            states,
        }
    }
}

pub fn fit_dynamic_glm(
    y: &Panel,
    exog: Option<&Design>,
    exposure: Option<&Panel>,
    config: &DynamicGlmConfig,
) -> Result<DynamicGlmPosterior, Error> {
    config.validate()?;
    let groups = y.len();
    let times = y.first().map_or(0, Vec::len);
    if groups == 0 || times == 0 || y.iter().any(|row| row.len() != times) {
        return Err(invalid(
            "y must be a nonempty rectangular group-by-time panel",
        ));
    }
    let features = exog.and_then(|x| x.first()?.first()).map_or(0, Vec::len);
    validate_design(exog, exposure, groups, times, features, config.family)?;
    let count = matches!(config.family, Family::Poisson | Family::NegativeBinomial);
    let mut observed_count = 0;
    for (g, row) in y.iter().enumerate() {
        for (t, &value) in row.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            if !supported_observation(value, config.family) {
                return Err(invalid("observations must be supported finite values or NaN; counts must be exact nonnegative integers below 2^53"));
            }
            if count && exposure.is_some_and(|e| e[g][t] == 0.) && value != 0. {
                return Err(invalid("positive counts are impossible at zero exposure"));
            }
            observed_count += 1;
        }
    }
    let layout = Layout::checked(
        groups,
        times,
        features
            .checked_add(1)
            .ok_or_else(|| invalid("feature count overflow"))?,
        if config.family == Family::HurdleLogNormal {
            2
        } else {
            1
        },
    )?;
    allocation(&[config.chains, config.draws, layout.size()])?;
    let panel = PanelLikelihood {
        y,
        exog,
        exposure,
        config,
        layout: &layout,
    };
    let blocks = layout.blocks();
    let results = (0..config.chains)
        .into_par_iter()
        .map(|chain| {
            let mut rng =
                ChaCha8Rng::seed_from_u64(chain_seed(config.seed, chain, FIT_SEED_DOMAIN));
            let mut z = vec![0.; layout.size()];
            // Independent modest starts; ESS immediately samples prior-only blocks.
            for x in &mut z {
                *x = normal(&mut rng) * 0.1;
            }
            let mut cache = PanelCache::new(&panel, &z);
            if !cache.group_lp.iter().sum::<f64>().is_finite() {
                return Err(numerical(
                    "initial likelihood is not finite; review data and prior scales",
                ));
            }
            let mut evaluations = 1;
            let mut samples = Vec::with_capacity(config.draws);
            for iteration in 0..config.warmup + config.draws * config.thin {
                for block in &blocks {
                    evaluations += cache.update(&panel, &mut z, block, &mut rng)?;
                }
                if iteration >= config.warmup
                    && (iteration + 1 - config.warmup).is_multiple_of(config.thin)
                {
                    samples.push(layout.decode(&z, config));
                }
            }
            Ok((samples, evaluations))
        })
        // Collected in chain order first, so the error reported when several
        // chains fail does not depend on scheduling.
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>, Error>>()?;
    let (chains, likelihood_evaluations) = results.into_iter().unzip();
    let posterior = DynamicGlmPosterior {
        config: config.clone(),
        chains,
        groups,
        time_count: times,
        features,
        likelihood_evaluations,
        observed_count,
    };
    posterior.validate()?;
    Ok(posterior)
}

/// The observation model over the training panel, evaluated one group at a
/// time: every latent block moves either one group's linear predictor or all
/// of them, so a block's slice evaluations need only the groups it touches.
struct PanelLikelihood<'a> {
    y: &'a Panel,
    exog: Option<&'a Design>,
    exposure: Option<&'a Panel>,
    config: &'a DynamicGlmConfig,
    layout: &'a Layout,
}

impl PanelLikelihood<'_> {
    /// Linear predictor of one component for one group at every training
    /// time, read straight from the latent vector with the arithmetic
    /// [`Layout::decode`] and [`predictor`] use.
    fn fill_predictor(
        &self,
        z: &[f64],
        component: usize,
        group: usize,
        beta: &mut [f64],
        eta: &mut [f64],
    ) {
        let (layout, cfg) = (self.layout, self.config);
        let base = component * layout.width();
        let initial = if component == 1 {
            cfg.occurrence_initial_mean
        } else {
            cfg.initial_mean
        };
        let deviations = base + layout.k * (group + 1);
        for (j, b) in beta.iter_mut().enumerate() {
            let mut population = z[base + j] * cfg.coefficient_sd;
            if j == 0 {
                population += initial;
            }
            *b = population + cfg.group_sd * z[deviations + j];
        }
        let shared = base + layout.k * (1 + layout.groups);
        let own = shared + (group + 1) * layout.times;
        let mut level = 0.;
        for (t, out) in eta.iter_mut().enumerate() {
            level += cfg.shared_process_sd * z[shared + t] + cfg.process_sd * z[own + t];
            *out = predictor(beta, self.exog.map(|x| &x[group][t]), level);
        }
    }

    fn group_log_likelihood(&self, group: usize, eta: &[f64], occurrence: Option<&[f64]>) -> f64 {
        let mut lp = 0.;
        for (t, &value) in self.y[group].iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            lp += log_likelihood(
                value,
                eta[t],
                occurrence.map_or(0., |o| o[t]),
                self.exposure.map_or(1., |e| e[group][t]),
                self.config,
            );
        }
        lp
    }
}

/// Linear predictors and per-group log likelihoods at the chain's current
/// latent state, kept in step with it block by block.
struct PanelCache {
    /// [component][group * times + time].
    eta: Vec<Vec<f64>>,
    group_lp: Vec<f64>,
    proposed_eta: Vec<f64>,
    proposed_lp: Vec<f64>,
    beta: Vec<f64>,
}

impl PanelCache {
    fn new(panel: &PanelLikelihood, z: &[f64]) -> Self {
        let layout = panel.layout;
        let times = layout.times;
        let mut beta = vec![0.; layout.k];
        let eta: Vec<Vec<f64>> = (0..layout.components)
            .map(|c| {
                let mut eta = vec![0.; layout.groups * times];
                for (g, row) in eta.chunks_mut(times).enumerate() {
                    panel.fill_predictor(z, c, g, &mut beta, row);
                }
                eta
            })
            .collect();
        let group_lp = (0..layout.groups)
            .map(|g| {
                let rows = g * times..(g + 1) * times;
                panel.group_log_likelihood(g, &eta[0][rows.clone()], eta.get(1).map(|o| &o[rows]))
            })
            .collect();
        Self {
            eta,
            group_lp,
            proposed_eta: vec![0.; layout.groups * times],
            proposed_lp: vec![0.; layout.groups],
            beta,
        }
    }

    /// One elliptical slice update of `block`; returns the likelihood
    /// evaluations it took. The slice sees only the touched groups' terms,
    /// which differ from the full panel log likelihood by a constant.
    fn update<R: Rng + ?Sized>(
        &mut self,
        panel: &PanelLikelihood,
        z: &mut [f64],
        block: &Block,
        rng: &mut R,
    ) -> Result<usize, Error> {
        let layout = panel.layout;
        let times = layout.times;
        let groups = block.group.map_or(0..layout.groups, |g| g..g + 1);
        let current = groups.clone().map(|g| self.group_lp[g]).sum::<f64>();
        let c = block.component;
        let Self {
            eta,
            proposed_eta,
            proposed_lp,
            beta,
            ..
        } = self;
        let (_, evaluations) = crate::elliptical_slice::update(
            z,
            block.range.clone(),
            current,
            |z| {
                let mut lp = 0.;
                for g in groups.clone() {
                    let rows = g * times..(g + 1) * times;
                    let row = &mut proposed_eta[rows.clone()];
                    panel.fill_predictor(z, c, g, beta, row);
                    let row = &proposed_eta[rows.clone()];
                    proposed_lp[g] = if c == 0 {
                        panel.group_log_likelihood(g, row, eta.get(1).map(|o| &o[rows]))
                    } else {
                        panel.group_log_likelihood(g, &eta[0][rows], Some(row))
                    };
                    lp += proposed_lp[g];
                }
                lp
            },
            rng,
        )
        .map_err(|e| numerical(e.to_string()))?;
        // The accepted point was the last one evaluated.
        let rows = groups.start * times..groups.end * times;
        self.eta[c][rows.clone()].copy_from_slice(&self.proposed_eta[rows]);
        self.group_lp[groups.clone()].copy_from_slice(&self.proposed_lp[groups]);
        Ok(evaluations)
    }
}

fn predictor(beta: &[f64], x: Option<&Vec<f64>>, state: f64) -> f64 {
    beta[0] + state + x.map_or(0., |x| x.iter().zip(&beta[1..]).map(|(x, b)| x * b).sum())
}

/// Independent Gaussian prior draws, followed by recursive predictive simulation.
pub fn prior_predictive(
    config: &DynamicGlmConfig,
    groups: usize,
    steps: usize,
    exog: Option<&Design>,
    exposure: Option<&Panel>,
) -> Result<DynamicGlmForecast, Error> {
    config.validate()?;
    if groups == 0 || steps == 0 {
        return Err(invalid("groups and steps must be positive"));
    }
    let features = exog.and_then(|x| x.first()?.first()).map_or(0, Vec::len);
    validate_design(exog, exposure, groups, steps, features, config.family)?;
    allocation(&[config.chains, config.draws, groups, steps, 4])?;
    let layout = Layout::checked(
        groups,
        1,
        features
            .checked_add(1)
            .ok_or_else(|| invalid("feature count overflow"))?,
        if config.family == Family::HurdleLogNormal {
            2
        } else {
            1
        },
    )?;
    allocation(&[config.chains, config.draws, layout.size()])?;
    let chains = (0..config.chains)
        .into_par_iter()
        .map(|c| {
            let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(config.seed, c, PRIOR_SEED_DOMAIN));
            (0..config.draws)
                .map(|_| {
                    let z = (0..layout.size())
                        .map(|_| normal(&mut rng))
                        .collect::<Vec<_>>();
                    let mut draw = layout.decode(&z, config);
                    for component in &mut draw.states {
                        for group in component {
                            group[0] = 0.;
                        }
                    }
                    draw
                })
                .collect()
        })
        .collect();
    DynamicGlmPosterior {
        config: config.clone(),
        chains,
        groups,
        time_count: 1,
        features,
        likelihood_evaluations: vec![0; config.chains],
        observed_count: 0,
    }
    .forecast(steps, exog, exposure, config.seed)
}
fn log_likelihood(y: f64, eta: f64, occurrence: f64, exposure: f64, cfg: &DynamicGlmConfig) -> f64 {
    match cfg.family {
        Family::Poisson | Family::NegativeBinomial => {
            if exposure == 0. {
                return if y == 0. { 0. } else { f64::NEG_INFINITY };
            }
            let log_mu = eta + exposure.ln();
            if cfg.family == Family::Poisson {
                return crate::count_sampling::log_mass_from_log_rate(y, log_mu);
            }
            crate::negative_binomial::log_mass(y, log_mu, cfg.dispersion)
        }
        Family::Gaussian => {
            -0.5 * ((y - eta) / cfg.observation_sd).powi(2)
                - cfg.observation_sd.ln()
                - 0.5 * std::f64::consts::TAU.ln()
        }
        Family::HurdleLogNormal => {
            if y == 0. {
                -softplus(occurrence)
            } else {
                -softplus(-occurrence)
                    - y.ln()
                    - cfg.observation_sd.ln()
                    - 0.5 * std::f64::consts::TAU.ln()
                    - 0.5 * ((y.ln() - eta) / cfg.observation_sd).powi(2)
            }
        }
    }
}

impl DynamicGlmPosterior {
    /// Versioned fitted state with missing observations encoded as JSON null.
    pub fn to_json(&self, observations: &Panel) -> Result<String, Error> {
        self.validate()?;
        validate_saved_observations(observations, self)?;
        serde_json::to_string(&Artifact {
            version: 1,
            posterior: self.clone(),
            observations: observations
                .iter()
                .map(|g| g.iter().map(|x| x.is_finite().then_some(*x)).collect())
                .collect(),
        })
        .map_err(|e| invalid(e.to_string()))
    }
    pub fn from_json(json: &str) -> Result<(Self, Panel), Error> {
        let artifact: Artifact = serde_json::from_str(json).map_err(|e| invalid(e.to_string()))?;
        if artifact.version != 1 {
            return Err(invalid("unsupported dynamic GLM artifact version"));
        }
        artifact.posterior.validate()?;
        let observations = artifact
            .observations
            .into_iter()
            .map(|g| g.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
            .collect();
        validate_saved_observations(&observations, &artifact.posterior)?;
        Ok((artifact.posterior, observations))
    }
    fn validate(&self) -> Result<(), Error> {
        self.config.validate()?;
        if self.groups == 0
            || self.time_count == 0
            || self.chains.len() != self.config.chains
            || self.likelihood_evaluations.len() != self.chains.len()
            || self.features == usize::MAX
        {
            return Err(invalid("invalid posterior dimensions"));
        }
        let components = if self.config.family == Family::HurdleLogNormal {
            2
        } else {
            1
        };
        allocation(&[
            self.groups,
            self.time_count,
            self.config.chains,
            self.config.draws,
            components,
        ])?;
        allocation(&[
            self.groups,
            self.features + 1,
            self.config.chains,
            self.config.draws,
            components,
        ])?;
        for chain in &self.chains {
            if chain.len() != self.config.draws {
                return Err(invalid("invalid posterior draw count"));
            }
            for d in chain {
                if d.population_coefficients.len() != components
                    || d.coefficients.len() != components
                    || d.states.len() != components
                {
                    return Err(invalid("invalid component count"));
                }
                for c in 0..components {
                    if d.population_coefficients[c].len() != self.features + 1
                        || d.population_coefficients[c].iter().any(|x| !x.is_finite())
                        || d.coefficients[c].len() != self.groups
                        || d.states[c].len() != self.groups
                    {
                        return Err(invalid("invalid population/group shape or values"));
                    }
                    for g in 0..self.groups {
                        if d.coefficients[c][g].len() != self.features + 1
                            || d.states[c][g].len() != self.time_count
                            || d.coefficients[c][g]
                                .iter()
                                .chain(&d.states[c][g])
                                .any(|x| !x.is_finite())
                        {
                            return Err(invalid("invalid coefficients/state shape or values"));
                        }
                        if self.config.group_sd == 0.0
                            && d.coefficients[c][g] != d.population_coefficients[c]
                        {
                            return Err(invalid("zero group_sd requires group coefficients to equal population coefficients"));
                        }
                        if self.config.process_sd == 0.0 {
                            if self.config.shared_process_sd == 0.0
                                && d.states[c][g].iter().any(|v| *v != 0.0)
                            {
                                return Err(invalid(
                                    "zero process scales require identically zero states",
                                ));
                            }
                            if d.states[c][g] != d.states[c][0] {
                                return Err(invalid("zero group process_sd requires identical shared state paths across groups"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn parameter_names(&self) -> Vec<String> {
        let components = if self.config.family == Family::HurdleLogNormal {
            2
        } else {
            1
        };
        let mut names = Vec::new();
        for c in 0..components {
            for j in 0..=self.features {
                names.push(format!("population_beta[{c},{j}]"));
            }
            for g in 0..self.groups {
                for j in 0..=self.features {
                    names.push(format!("beta[{c},{g},{j}]"));
                }
                names.push(format!("terminal_state[{c},{g}]"));
            }
        }
        names
    }
    pub fn parameter_samples(&self) -> Vec<Vec<Vec<f64>>> {
        self.chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        let mut values = Vec::new();
                        for c in 0..d.coefficients.len() {
                            values.extend_from_slice(&d.population_coefficients[c]);
                            for g in 0..self.groups {
                                values.extend_from_slice(&d.coefficients[c][g]);
                                values.push(d.states[c][g][self.time_count - 1]);
                            }
                        }
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
        exog: Option<&Design>,
        exposure: Option<&Panel>,
        forecast_seed: u64,
    ) -> Result<DynamicGlmForecast, Error> {
        self.validate()?;
        if steps == 0 {
            return Err(invalid("steps must be positive"));
        }
        validate_design(
            exog,
            exposure,
            self.groups,
            steps,
            self.features,
            self.config.family,
        )?;
        allocation(&[self.chains.len(), self.config.draws, self.groups, steps, 4])?;
        // Each chain has its own stream, so the result does not depend on how
        // rayon schedules chains; errors are reported in chain order for the
        // same reason.
        let chains: Vec<Result<ChainPaths, Error>> = self
            .chains
            .par_iter()
            .enumerate()
            .map(|(chain, draws)| {
                let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(
                    forecast_seed,
                    chain,
                    FORECAST_SEED_DOMAIN,
                ));
                self.forecast_chain(draws, steps, exog, exposure, &mut rng)
            })
            .collect();
        let hurdle = self.config.family == Family::HurdleLogNormal;
        let mut out = DynamicGlmForecast {
            mean_paths: Vec::new(),
            observation_paths: Vec::new(),
            occurrence_paths: Vec::new(),
            positive_mean_paths: Vec::new(),
        };
        for chain in chains {
            let (means, observations, occurrences, positives) = chain?;
            out.mean_paths.push(means);
            out.observation_paths.push(observations);
            if hurdle {
                out.occurrence_paths.push(occurrences);
                out.positive_mean_paths.push(positives);
            }
        }
        Ok(out)
    }

    fn forecast_chain(
        &self,
        draws: &[DynamicGlmDraw],
        steps: usize,
        exog: Option<&Design>,
        exposure: Option<&Panel>,
        rng: &mut ChaCha8Rng,
    ) -> Result<ChainPaths, Error> {
        let hurdle = self.config.family == Family::HurdleLogNormal;
        let mut means = Vec::new();
        let mut observations = Vec::new();
        let mut occurrences = Vec::new();
        let mut positives = Vec::new();
        for draw in draws {
            let mut levels: Vec<Vec<f64>> = draw
                .states
                .iter()
                .map(|groups| groups.iter().map(|s| s[self.time_count - 1]).collect())
                .collect();
            let mut mean = vec![vec![0.; steps]; self.groups];
            let mut observation = mean.clone();
            let mut occurrence = mean.clone();
            let mut positive = mean.clone();
            for t in 0..steps {
                for component in &mut levels {
                    let shared = self.config.shared_process_sd * normal(rng);
                    for level in component {
                        *level += shared + self.config.process_sd * normal(rng);
                    }
                }
                for g in 0..self.groups {
                    let eta = predictor(
                        &draw.coefficients[0][g],
                        exog.map(|x| &x[g][t]),
                        levels[0][g],
                    );
                    let e = exposure.map_or(1., |e| e[g][t]);
                    match self.config.family {
                        Family::Gaussian => {
                            mean[g][t] = eta;
                            observation[g][t] = eta + self.config.observation_sd * normal(rng);
                        }
                        Family::Poisson | Family::NegativeBinomial => {
                            let mu = if e == 0. { 0. } else { (eta + e.ln()).exp() };
                            if e > 0. && mu == 0. {
                                return Err(numerical(
                                    "positive count mean underflowed; no draws clipped or removed",
                                ));
                            }
                            mean[g][t] = mu;
                            let rate = if mu == 0. || self.config.family == Family::Poisson {
                                mu
                            } else {
                                Gamma::new(self.config.dispersion, mu / self.config.dispersion)
                                    .map_err(|e| numerical(e.to_string()))?
                                    .sample(rng)
                            };
                            if mu > 0. && rate == 0. {
                                return Err(numerical("positive Gamma intensity underflowed; no draws clipped or removed"));
                            }
                            observation[g][t] = poisson(rate, rng)?;
                        }
                        Family::HurdleLogNormal => {
                            let p = sigmoid(predictor(
                                &draw.coefficients[1][g],
                                exog.map(|x| &x[g][t]),
                                levels[1][g],
                            ));
                            let positive_mean =
                                (eta + 0.5 * self.config.observation_sd.powi(2)).exp();
                            occurrence[g][t] = p;
                            positive[g][t] = positive_mean;
                            mean[g][t] = p * positive_mean;
                            observation[g][t] = if rng.gen::<f64>() < p {
                                let amount = (eta + self.config.observation_sd * normal(rng)).exp();
                                if amount == 0. {
                                    return Err(numerical("positive severity underflowed; no draws clipped or removed"));
                                }
                                amount
                            } else {
                                0.
                            };
                            if !positive_mean.is_finite() || positive_mean == 0. {
                                return Err(numerical("positive predictive mean overflowed; no draws clipped or removed"));
                            }
                        }
                    }
                    if !mean[g][t].is_finite() || !observation[g][t].is_finite() {
                        return Err(numerical(
                            "predictive value overflowed; no draws clipped or removed",
                        ));
                    }
                }
            }
            means.push(mean);
            observations.push(observation);
            if hurdle {
                occurrences.push(occurrence);
                positives.push(positive);
            }
        }
        Ok((means, observations, occurrences, positives))
    }
}

/// One chain's mean, observation, occurrence and positive-mean paths.
type ChainPaths = (Vec<Panel>, Vec<Panel>, Vec<Panel>, Vec<Panel>);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    version: u32,
    posterior: DynamicGlmPosterior,
    observations: Vec<Vec<Option<f64>>>,
}
fn supported_observation(value: f64, family: Family) -> bool {
    value.is_finite()
        && (family == Family::Gaussian || value >= 0.0)
        && (!matches!(family, Family::Poisson | Family::NegativeBinomial)
            || (value.fract() == 0.0 && value <= 9_007_199_254_740_991.0))
}
fn validate_saved_observations(y: &Panel, p: &DynamicGlmPosterior) -> Result<(), Error> {
    if y.len() != p.groups
        || y.iter()
            .any(|g| g.len() != p.time_count || g.iter().any(|x| x.is_infinite()))
        || y.iter().flatten().filter(|x| x.is_finite()).count() != p.observed_count
        || y.iter()
            .flatten()
            .any(|&x| !x.is_nan() && !supported_observation(x, p.config.family))
    {
        return Err(invalid("saved observations do not match fitted metadata"));
    }
    Ok(())
}

fn validate_design(
    exog: Option<&Design>,
    exposure: Option<&Panel>,
    groups: usize,
    times: usize,
    features: usize,
    family: Family,
) -> Result<(), Error> {
    if features > 0 && exog.is_none() {
        return Err(invalid("future exog required for every fitted feature"));
    }
    if exog.is_some_and(|x| {
        x.len() != groups
            || x.iter().any(|g| {
                g.len() != times
                    || g.iter()
                        .any(|r| r.len() != features || r.iter().any(|x| !x.is_finite()))
            })
    }) {
        return Err(invalid(
            "exog must have shape [group,time,feature] with finite values",
        ));
    }
    if let Some(e) = exposure {
        if !matches!(family, Family::Poisson | Family::NegativeBinomial) {
            return Err(invalid("exposure is supported only for count families"));
        }
        if e.len() != groups
            || e.iter()
                .any(|g| g.len() != times || g.iter().any(|x| !x.is_finite() || *x < 0.))
        {
            return Err(invalid(
                "exposure must have shape [group,time] with finite nonnegative values",
            ));
        }
    }
    Ok(())
}
fn poisson<R: Rng + ?Sized>(rate: f64, rng: &mut R) -> Result<f64, Error> {
    if rate == 0. {
        return Ok(0.);
    }
    // The sampler and f64 output cannot represent arbitrary integer counts.
    if !rate.is_finite() || rate > 1e15 {
        return Err(numerical(
            "Poisson rate outside supported numerical range; no draws clipped or removed",
        ));
    }
    crate::count_sampling::poisson(rate, rng).map_err(|e| numerical(e.to_string()))
}
fn allocation(factors: &[usize]) -> Result<(), Error> {
    let n = factors
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| invalid("allocation overflow"))?;
    if n > 25_000_000 {
        return Err(invalid(
            "requested retained arrays exceed 25 million values",
        ));
    }
    Ok(())
}
fn normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    StandardNormal.sample(rng)
}
const FIT_SEED_DOMAIN: u64 = 0x4649_545F_4447_4C4D; // "FIT_DGLM"
const FORECAST_SEED_DOMAIN: u64 = 0x4643_5354_4447_4C4D; // "FCSTDGLM"
const PRIOR_SEED_DOMAIN: u64 = 0x5052_4952_4447_4C4D; // "PRIRDGLM"
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfiguration(message.into())
}
fn numerical(message: impl Into<String>) -> Error {
    Error::NumericalFailure(message.into())
}

#[cfg(test)]
mod incremental_likelihood_tests {
    use super::*;

    /// The full-panel log likelihood through the nested draw, which is how the
    /// sampler evaluated every slice proposal before it went group by group.
    fn full_log_likelihood(panel: &PanelLikelihood, z: &[f64]) -> f64 {
        let draw = panel.layout.decode(z, panel.config);
        let mut lp = 0.;
        for (g, row) in panel.y.iter().enumerate() {
            for (t, &value) in row.iter().enumerate() {
                if value.is_nan() {
                    continue;
                }
                let x = panel.exog.map(|x| &x[g][t]);
                let eta = predictor(&draw.coefficients[0][g], x, draw.states[0][g][t]);
                let occurrence = if panel.layout.components == 2 {
                    predictor(&draw.coefficients[1][g], x, draw.states[1][g][t])
                } else {
                    0.
                };
                lp += log_likelihood(value, eta, occurrence, 1., panel.config);
            }
        }
        lp
    }

    #[test]
    fn block_updates_keep_the_cache_equal_to_a_fresh_full_panel_evaluation() {
        let config = DynamicGlmConfig {
            family: Family::HurdleLogNormal,
            group_sd: 0.4,
            process_sd: 0.2,
            shared_process_sd: 0.3,
            ..Default::default()
        };
        let y = vec![
            vec![0., 1.5, f64::NAN, 2.5, 3., 0.],
            vec![1., 0., 0., 4., f64::NAN, 0.5],
            vec![0., 0., 2., 1., 1., 7.],
        ];
        let exog: Design = (0..3)
            .map(|g| {
                (0..6)
                    .map(|t| vec![(g * 6 + t) as f64 * 0.1 - 0.8])
                    .collect()
            })
            .collect();
        let layout = Layout::checked(3, 6, 2, 2).unwrap();
        let panel = PanelLikelihood {
            y: &y,
            exog: Some(&exog),
            exposure: None,
            config: &config,
            layout: &layout,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut z: Vec<f64> = (0..layout.size()).map(|_| normal(&mut rng)).collect();
        let mut cache = PanelCache::new(&panel, &z);
        let blocks = layout.blocks();
        let mut covered: Vec<usize> = blocks.iter().flat_map(|b| b.range.clone()).collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..layout.size()).collect::<Vec<_>>());
        for _ in 0..10 {
            for block in &blocks {
                cache.update(&panel, &mut z, block, &mut rng).unwrap();
                // A block that moved a group it does not declare would leave
                // that group's cached terms stale.
                let fresh = PanelCache::new(&panel, &z);
                assert_eq!(cache.eta, fresh.eta);
                assert_eq!(cache.group_lp, fresh.group_lp);
                let full = full_log_likelihood(&panel, &z);
                let cached = cache.group_lp.iter().sum::<f64>();
                assert!(
                    (cached - full).abs() <= 1e-12 * full.abs(),
                    "{cached} vs {full}"
                );
            }
        }
    }

    #[test]
    fn seed_domains_separate_fit_forecast_and_prior_streams() {
        let seeds = [FIT_SEED_DOMAIN, FORECAST_SEED_DOMAIN, PRIOR_SEED_DOMAIN]
            .map(|domain| chain_seed(42, 0, domain));
        assert!(seeds[0] != seeds[1] && seeds[1] != seeds[2] && seeds[0] != seeds[2]);
        assert!(seeds.iter().all(|&s| s != 42));
    }
}

#[cfg(test)]
mod poisson_density_tests {
    use super::*;

    #[test]
    fn poisson_large_count_curvature_is_preserved_with_exposure() {
        let config = DynamicGlmConfig::default();
        for count in [1e14_f64, 1e15, 8e15] {
            let expected_mode = -0.5 * (std::f64::consts::TAU.ln() + count.ln());
            for exposure in [0.5_f64, 1.0, 10.0] {
                for z in [-1.0, 0.0, 1.0] {
                    let eta = count.ln() - exposure.ln() + z / count.sqrt();
                    let actual = log_likelihood(count, eta, 0.0, exposure, &config);
                    assert!((actual - expected_mode + 0.5 * z * z).abs() < 3e-6);
                }
            }
        }
    }
}

#[cfg(test)]
mod negative_binomial_density_tests {
    use super::*;

    #[test]
    fn large_dispersion_likelihood_preserves_local_curvature() {
        let config = DynamicGlmConfig {
            family: Family::NegativeBinomial,
            dispersion: 1e14,
            ..Default::default()
        };
        let eta = 1e14_f64.ln();
        let center = log_likelihood(1e14, eta, 0.0, 1.0, &config);
        let upper = log_likelihood(1e14, eta + 1e-7, 0.0, 1.0, &config);
        let lower = log_likelihood(1e14, eta - 1e-7, 0.0, 1.0, &config);
        assert!((center + 17.383607774442968).abs() < 2e-8);
        assert!((upper + lower - 2.0 * center + 0.5).abs() < 5e-8);
    }
}
