//! One rule for which fits to distrust, shared by every engine and binding.
//!
//! [`fit_warnings`] turns a fit's parameter diagnostics and, for HMC and NUTS,
//! its transition telemetry into records a caller can filter: what was
//! checked, the value, the threshold, and whether the fit failed the check or
//! the check could not be made. A check that passes produces no record. A
//! check the sampler has no statistic for is reported once as
//! [`WarningStatus::Unavailable`], never left out and never reported as a
//! pass, so an empty list means every check ran and passed.
//!
//! Thresholds:
//! - R-hat above 1.01, and bulk or tail ESS below 400: Vehtari, Gelman,
//!   Simpson, Carpenter and Bürkner (2021), "Rank-normalization, folding, and
//!   localization: an improved R-hat for assessing convergence of MCMC".
//! - E-BFMI below 0.3 in any chain: Betancourt (2016), "Diagnosing suboptimal
//!   cotangent disintegrations in Hamiltonian Monte Carlo"; 0.3 is the value
//!   Stan and ArviZ warn at.
//! - Any divergence, and any draw transition that reached the maximum tree
//!   depth: the conditions Stan's diagnostics report.
//! - R-hat and ESS need at least
//!   [`MIN_DIAGNOSTIC_DRAWS_PER_CHAIN`](super::MIN_DIAGNOSTIC_DRAWS_PER_CHAIN)
//!   finite draws per chain; a fit with fewer, or with a non-finite draw,
//!   fails `diagnostics_unavailable`. So does an HMC or NUTS parameter whose
//!   draws do not vary (a stuck chain); for other samplers that is a value
//!   the model fixes, and the check is only unavailable.
use super::{DiagnosticsReport, TransitionDiagnosticsReport, MIN_DIAGNOSTIC_DRAWS_PER_CHAIN};

/// Rank-normalized split R-hat above this fails.
pub const R_HAT_THRESHOLD: f64 = 1.01;
/// Bulk or tail effective sample size below this fails.
pub const ESS_THRESHOLD: f64 = 400.0;
/// A chain's E-BFMI below this fails.
pub const E_BFMI_THRESHOLD: f64 = 0.3;
/// Divergences and saturated trees above this count fail.
const COUNT_THRESHOLD: f64 = 0.0;

/// What a [`FitWarning`] checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningCode {
    RHat,
    EssBulk,
    EssTail,
    Divergences,
    EBfmi,
    TreeDepthSaturation,
    /// R-hat and ESS could not be computed for a parameter.
    DiagnosticsUnavailable,
    /// A documented problem of the model's sampler that its diagnostics may
    /// not show on a given fit.
    KnownLimitation,
}

impl WarningCode {
    /// Every code, in the order [`fit_warnings`] reports them.
    pub const ALL: [WarningCode; 8] = [
        WarningCode::RHat,
        WarningCode::EssBulk,
        WarningCode::EssTail,
        WarningCode::DiagnosticsUnavailable,
        WarningCode::Divergences,
        WarningCode::EBfmi,
        WarningCode::TreeDepthSaturation,
        WarningCode::KnownLimitation,
    ];

    /// The code's name in records and in the Python API.
    pub fn as_str(self) -> &'static str {
        match self {
            WarningCode::RHat => "r_hat",
            WarningCode::EssBulk => "ess_bulk",
            WarningCode::EssTail => "ess_tail",
            WarningCode::Divergences => "divergences",
            WarningCode::EBfmi => "e_bfmi",
            WarningCode::TreeDepthSaturation => "tree_depth_saturation",
            WarningCode::DiagnosticsUnavailable => "diagnostics_unavailable",
            WarningCode::KnownLimitation => "known_limitation",
        }
    }

    /// The code named `name`, if there is one.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|code| code.as_str() == name)
    }
}

/// Whether a check failed or could not be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningStatus {
    /// The fit failed the check: distrust it until the cause is understood.
    Fail,
    /// The check could not be made: the sampler has no such statistic, or
    /// the fit does not carry what it needs. Not a pass.
    Unavailable,
}

impl WarningStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            WarningStatus::Fail => "fail",
            WarningStatus::Unavailable => "unavailable",
        }
    }
}

/// One failed or unavailable check of one fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitWarning {
    pub code: WarningCode,
    pub status: WarningStatus,
    /// The parameter the check is about; `None` for a whole-fit or per-chain
    /// check.
    pub parameter: Option<String>,
    /// The chain a per-chain check (E-BFMI) is about.
    pub chain: Option<usize>,
    /// The value checked, NaN when there is none.
    pub value: f64,
    /// The threshold the value is compared with.
    pub threshold: f64,
    /// One line stating what was found and the threshold.
    pub message: String,
}

impl FitWarning {
    fn new(code: WarningCode, status: WarningStatus, value: f64, threshold: f64) -> Self {
        Self {
            code,
            status,
            parameter: None,
            chain: None,
            value,
            threshold,
            message: String::new(),
        }
    }

    fn parameter(mut self, name: &str) -> Self {
        self.parameter = Some(name.to_string());
        self
    }

    fn chain(mut self, chain: usize) -> Self {
        self.chain = Some(chain);
        self
    }

    fn message(mut self, message: String) -> Self {
        self.message = message;
        self
    }
}

/// The sampler statistics a fit has beyond its draws.
#[derive(Debug, Clone, Copy)]
pub enum SamplerTelemetry<'a> {
    /// An HMC or NUTS fit's transition telemetry, from
    /// [`crate::sampler::SampleResult::transition_diagnostics`].
    Hamiltonian(&'a TransitionDiagnosticsReport),
    /// A sampler without Hamiltonian trajectories (Gibbs, elliptical slice,
    /// conjugate draws), named as the messages should name it.
    NonHamiltonian { sampler: &'a str },
}

/// Every failed or unavailable check of one fit, in a fixed order: each
/// parameter's R-hat, bulk ESS, tail ESS and availability records, then
/// divergences, E-BFMI per chain and tree-depth saturation.
pub fn fit_warnings(
    report: &DiagnosticsReport,
    telemetry: SamplerTelemetry<'_>,
) -> Vec<FitWarning> {
    let mut warnings = Vec::new();
    let hamiltonian = matches!(telemetry, SamplerTelemetry::Hamiltonian(_));
    for parameter in &report.params {
        parameter_warnings(report.num_draws, parameter, hamiltonian, &mut warnings);
    }
    match telemetry {
        SamplerTelemetry::Hamiltonian(transitions) => {
            hamiltonian_warnings(transitions, &mut warnings);
        }
        SamplerTelemetry::NonHamiltonian { sampler } => {
            non_hamiltonian_warnings(sampler, &mut warnings);
        }
    }
    warnings
}

fn parameter_warnings(
    num_draws: usize,
    parameter: &super::ParamDiagnostics,
    hamiltonian: bool,
    warnings: &mut Vec<FitWarning>,
) {
    use WarningCode::*;
    use WarningStatus::*;
    let name = parameter.name.as_str();
    let minimum = MIN_DIAGNOSTIC_DRAWS_PER_CHAIN;
    let statistics = [
        (RHat, "R-hat", parameter.r_hat),
        (EssBulk, "bulk ESS", parameter.ess_bulk),
        (EssTail, "tail ESS", parameter.ess_tail),
    ];
    if num_draws < minimum {
        warnings.push(
            FitWarning::new(
                DiagnosticsUnavailable,
                Fail,
                num_draws as f64,
                minimum as f64,
            )
            .parameter(name)
            .message(format!(
                "R-hat and ESS of '{name}' are unavailable: {num_draws} draws per chain, \
                     fewer than the {minimum} they need"
            )),
        );
    } else if !parameter.mean.is_finite() {
        // compute_diagnostics blanks every statistic, the mean included, of a
        // parameter with a non-finite draw or of a ragged sample.
        warnings.push(
            FitWarning::new(DiagnosticsUnavailable, Fail, f64::NAN, minimum as f64)
                .parameter(name)
                .message(format!(
                    "R-hat and ESS of '{name}' are unavailable: its draws include a \
                     non-finite value (they need at least {minimum} finite draws per chain)"
                )),
        );
    } else {
        let missing: Vec<&str> = statistics
            .iter()
            .filter(|(_, _, value)| value.is_nan())
            .map(|(_, label, _)| *label)
            .collect();
        if !missing.is_empty() {
            // Enough finite draws, but too uniform for a variance-based
            // statistic. Every parameter of an HMC or NUTS fit is sampled (or
            // derived from sampled ones), so there a chain is stuck and the fit
            // fails; a Gibbs model can fix a value, which is only unavailable.
            let statistic = match missing.len() {
                1 => format!("{} is", missing[0]),
                _ => format!("{} are", missing.join(" and ")),
            };
            let (status, cause) = if hamiltonian {
                (Fail, "a chain is stuck")
            } else {
                (Unavailable, "as for a fixed value")
            };
            warnings.push(
                FitWarning::new(DiagnosticsUnavailable, status, f64::NAN, minimum as f64)
                    .parameter(name)
                    .message(format!(
                        "{statistic} of '{name}' unavailable: its draws do not vary enough to \
                         estimate it, {cause} (at least {minimum} varying draws per chain are \
                         needed)"
                    )),
            );
        }
    }

    // A statistic that was computed is checked even when another was not:
    // chains stuck at different values have no ESS but an infinite R-hat.
    let r_hat = parameter.r_hat;
    if r_hat > R_HAT_THRESHOLD {
        warnings.push(
            FitWarning::new(RHat, Fail, r_hat, R_HAT_THRESHOLD)
                .parameter(name)
                .message(format!(
                    "R-hat of '{name}' is {r_hat:.4}, above the {R_HAT_THRESHOLD} threshold; \
                     the chains do not agree"
                )),
        );
    }
    for (code, label, value) in [
        (EssBulk, "bulk ESS", parameter.ess_bulk),
        (EssTail, "tail ESS", parameter.ess_tail),
    ] {
        if value < ESS_THRESHOLD {
            warnings.push(
                FitWarning::new(code, Fail, value, ESS_THRESHOLD)
                    .parameter(name)
                    .message(format!(
                        "{label} of '{name}' is {value:.0}, below the {ESS_THRESHOLD} \
                         threshold; run more draws or reparameterize"
                    )),
            );
        }
    }
}

fn hamiltonian_warnings(transitions: &TransitionDiagnosticsReport, warnings: &mut Vec<FitWarning>) {
    use WarningCode::*;
    use WarningStatus::*;
    let draws = transitions.total_draw_transitions;

    let divergences = transitions.total_divergences;
    if divergences > 0 {
        warnings.push(
            FitWarning::new(Divergences, Fail, divergences as f64, COUNT_THRESHOLD).message(
                format!(
                    "{divergences} of {draws} draw transitions diverged, above the threshold \
                     of {COUNT_THRESHOLD}; the sampler could not follow the posterior's \
                     curvature, so the draws may be biased"
                ),
            ),
        );
    }

    for chain in &transitions.chains {
        let index = chain.chain_index;
        let value = chain.e_bfmi;
        if value.is_nan() {
            warnings.push(
                FitWarning::new(EBfmi, Unavailable, f64::NAN, E_BFMI_THRESHOLD)
                    .chain(index)
                    .message(format!(
                        "E-BFMI of chain {index} is unavailable: it needs at least two draws \
                         with finite, varying energies, which fits loaded from artifacts \
                         older than version 3 do not record (the check fails below \
                         {E_BFMI_THRESHOLD})"
                    )),
            );
        } else if value < E_BFMI_THRESHOLD {
            warnings.push(
                FitWarning::new(EBfmi, Fail, value, E_BFMI_THRESHOLD)
                    .chain(index)
                    .message(format!(
                        "E-BFMI of chain {index} is {value:.3}, below the {E_BFMI_THRESHOLD} \
                         threshold; momentum resampling moves the chain through the \
                         posterior's energy levels too slowly"
                    )),
            );
        }
    }

    match (
        transitions.total_tree_depth_saturation,
        transitions.max_tree_depth_limit,
    ) {
        (Some(saturated), Some(limit)) => {
            if saturated > 0 {
                warnings.push(
                    FitWarning::new(TreeDepthSaturation, Fail, saturated as f64, COUNT_THRESHOLD)
                        .message(format!(
                            "{saturated} of {draws} draw transitions reached the maximum tree \
                             depth of {limit}, above the threshold of {COUNT_THRESHOLD}; those \
                             trees used every doubling the limit allows, so the limit may have \
                             cut their trajectories short (raise max_tree_depth or \
                             reparameterize)"
                        )),
                );
            }
        }
        _ => {
            let is_nuts = transitions
                .chains
                .iter()
                .any(|chain| chain.max_tree_depth.is_some());
            let message = if is_nuts {
                format!(
                    "tree-depth saturation is unavailable: the maximum tree depth this fit ran \
                     with was not recorded, as in fits loaded from artifacts older than \
                     version 3 (the check fails above {COUNT_THRESHOLD})"
                )
            } else {
                format!(
                    "tree-depth saturation is not checked: static HMC builds no tree (for \
                     NUTS fits the check fails above {COUNT_THRESHOLD})"
                )
            };
            warnings.push(
                FitWarning::new(TreeDepthSaturation, Unavailable, f64::NAN, COUNT_THRESHOLD)
                    .message(message),
            );
        }
    }
}

fn non_hamiltonian_warnings(sampler: &str, warnings: &mut Vec<FitWarning>) {
    use WarningCode::*;
    use WarningStatus::*;
    warnings.push(
        FitWarning::new(Divergences, Unavailable, f64::NAN, COUNT_THRESHOLD).message(format!(
            "divergences are not checked: the {sampler} sampler has no Hamiltonian \
             trajectory (for HMC and NUTS fits the check fails above {COUNT_THRESHOLD})"
        )),
    );
    warnings.push(
        FitWarning::new(EBfmi, Unavailable, f64::NAN, E_BFMI_THRESHOLD).message(format!(
            "E-BFMI is not checked: the {sampler} sampler has no Hamiltonian energy (for HMC \
             and NUTS fits the check fails below {E_BFMI_THRESHOLD})"
        )),
    );
    warnings.push(
        FitWarning::new(TreeDepthSaturation, Unavailable, f64::NAN, COUNT_THRESHOLD).message(
            format!(
                "tree-depth saturation is not checked: the {sampler} sampler builds no NUTS \
                 tree (for NUTS fits the check fails above {COUNT_THRESHOLD})"
            ),
        ),
    );
}

/// Whether `warnings` hold a failed check whose code is not in `ignore`:
/// the rule batch triage flags a fit by. An unavailable check never flags.
pub fn has_failure(warnings: &[FitWarning], ignore: &[WarningCode]) -> bool {
    warnings
        .iter()
        .any(|warning| warning.status == WarningStatus::Fail && !ignore.contains(&warning.code))
}

/// A known limitation of a model's sampler, reported as a failed check so
/// that triage flags every such fit until the user rules it out by code.
///
/// `threshold` is the threshold the limitation is stated against, and
/// `message` must state it.
pub(crate) fn known_limitation(threshold: f64, message: String) -> FitWarning {
    FitWarning::new(
        WarningCode::KnownLimitation,
        WarningStatus::Fail,
        f64::NAN,
        threshold,
    )
    .message(message)
}
