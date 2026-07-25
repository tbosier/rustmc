//! Shared helpers for the statistical validation suite.
//!
//! # Tolerance philosophy
//!
//! Every assertion in this suite compares a Monte Carlo estimate against a
//! value that is either analytically known or known by construction. The
//! tolerance is therefore derived from the Monte Carlo standard error (MCSE)
//! of the estimator rather than hand-tuned until the test happens to pass.
//!
//! * **Posterior mean.** For a posterior with standard deviation `s` and bulk
//!   effective sample size `S`, the MCSE of the sample mean is `s / sqrt(S)`.
//!   `DiagnosticsReport` reports this directly as `mcse_mean`. We assert
//!   `|mean_hat - mean_true| <= K_MEAN * mcse_mean`. With `K_MEAN = 4` a single
//!   assertion has a ~6e-5 two-sided false-positive rate under a normal CLT
//!   approximation; across the ~120 mean assertions in this suite the
//!   family-wise flake probability is well under 1%.
//!
//! * **Posterior standard deviation.** The MCSE of a standard-deviation
//!   estimate from `S` effectively independent draws of a roughly Gaussian
//!   quantity is `s / sqrt(2 S)`, i.e. a relative error of `1 / sqrt(2 S)`.
//!   We use `K_SD = 5` because the sd estimator's sampling distribution is
//!   more skewed than the mean's and because `ess_bulk` is tuned for the mean.
//!   The *detectable* effect size is therefore `K_SD / sqrt(2 S)` relative:
//!   at `S = 2000` that is ~7.9%, so a scale error of 10% or more (and any
//!   sqrt(2)- or 2x-style scale bug) is caught, while a 1% error is not.
//!   Where a tighter bound is wanted, the test raises the draw count instead
//!   of lowering `K_SD`.
//!
//! * **Floors.** Some assertions add a small absolute floor so that a
//!   pathologically small reported MCSE (which itself is an estimate) cannot
//!   make a test infinitely strict. Floors are always stated in the caller.
//!
//! * **Determinism.** Every test seeds both the data-generating RNG and the
//!   sampler. Reruns are bit-identical, so a failure is a real failure and
//!   not a reseed away from passing.

#![allow(dead_code)]

use rustmc_core::diagnostics::{DiagnosticsReport, ParamDiagnostics};
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{sample as run_sample, SampleResult, SamplerConfig, SamplerType};

/// Multiplier on `mcse_mean` for posterior-mean assertions.
pub const K_MEAN: f64 = 4.0;
/// Multiplier on the relative sd MCSE `1/sqrt(2 S)` for posterior-sd assertions.
pub const K_SD: f64 = 5.0;

/// Run NUTS with a fixed configuration. `num_threads = 1` keeps chain results
/// reproducible independently of machine core count.
pub fn nuts(graph: Graph, seed: u64, chains: usize, draws: usize, warmup: usize) -> SampleResult {
    run_sample(
        graph,
        SamplerConfig {
            sampler: SamplerType::Nuts,
            num_chains: chains,
            num_draws: draws,
            num_warmup: warmup,
            step_size: 0.0,
            num_leapfrog_steps: 15,
            max_tree_depth: 10,
            seed,
            num_threads: 1,
            show_progress: false,
        },
    )
    .expect("sampling failed")
}

pub fn diag<'a>(report: &'a DiagnosticsReport, name: &str) -> &'a ParamDiagnostics {
    report
        .params
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| {
            panic!(
                "missing diagnostic for {name}; available: {:?}",
                report.params.iter().map(|p| &p.name).collect::<Vec<_>>()
            )
        })
}

/// Assert the sampled posterior mean of `name` matches `truth` to within
/// `K_MEAN` reported Monte Carlo standard errors (plus an absolute `floor`).
///
/// Also requires `ess_bulk >= min_ess`: without adequate ESS the reported MCSE
/// is itself unreliable and the assertion would be vacuous.
pub fn assert_mean_within_mcse(
    report: &DiagnosticsReport,
    name: &str,
    truth: f64,
    min_ess: f64,
    floor: f64,
) {
    let p = diag(report, name);
    assert!(
        p.ess_bulk.is_finite() && p.ess_bulk >= min_ess,
        "{name}: ess_bulk {} < required {min_ess}; MCSE-based tolerance would be meaningless",
        p.ess_bulk
    );
    let tol = K_MEAN * p.mcse_mean + floor;
    assert!(
        (p.mean - truth).abs() <= tol,
        "{name}: posterior mean {:.6} differs from analytic {:.6} by {:.6}, \
         tolerance {:.6} = {K_MEAN} * mcse({:.6}) + floor({floor}) [ess_bulk {:.0}]",
        p.mean,
        truth,
        (p.mean - truth).abs(),
        tol,
        p.mcse_mean,
        p.ess_bulk
    );
}

/// Assert the sampled posterior sd of `name` matches `truth` to within
/// `K_SD / sqrt(2 * ess_bulk)` in relative terms (plus an absolute `floor`).
///
/// Assumes an approximately Gaussian target; use
/// [`assert_sd_within_mcse_kurtosis`] for skewed or heavy-tailed targets.
pub fn assert_sd_within_mcse(
    report: &DiagnosticsReport,
    name: &str,
    truth: f64,
    min_ess: f64,
    floor: f64,
) {
    assert_sd_within_mcse_kurtosis(report, name, truth, 3.0, min_ess, floor)
}

/// As [`assert_sd_within_mcse`], but with the target's standardized fourth
/// moment `kurtosis = mu4 / sigma^4` supplied explicitly.
///
/// The delta-method standard error of a sample standard deviation from `S`
/// effectively independent draws is `sigma * sqrt((kurtosis - 1) / (4 S))`.
/// For a Gaussian (`kurtosis = 3`) this reduces to `sigma / sqrt(2 S)`. Using
/// the true kurtosis keeps heavy-tailed targets (LogNormal, Exponential,
/// StudentT) from producing spurious failures without loosening the bound for
/// well-behaved ones.
pub fn assert_sd_within_mcse_kurtosis(
    report: &DiagnosticsReport,
    name: &str,
    truth: f64,
    kurtosis: f64,
    min_ess: f64,
    floor: f64,
) {
    let p = diag(report, name);
    assert!(
        p.ess_bulk.is_finite() && p.ess_bulk >= min_ess,
        "{name}: ess_bulk {} < required {min_ess}",
        p.ess_bulk
    );
    let rel = K_SD * ((kurtosis - 1.0) / (4.0 * p.ess_bulk)).sqrt();
    let tol = rel * truth.abs() + floor;
    assert!(
        (p.std - truth).abs() <= tol,
        "{name}: posterior sd {:.6} differs from analytic {:.6} by {:.6} \
         ({:.2}% relative), tolerance {:.6} ({:.2}% relative, K_SD={K_SD}, ess_bulk {:.0})",
        p.std,
        truth,
        (p.std - truth).abs(),
        100.0 * (p.std - truth).abs() / truth.abs(),
        tol,
        100.0 * rel,
        p.ess_bulk
    );
}

/// Split the divergence count into (warmup, post-warmup).
///
/// NOTE: `SampleResult::total_divergences()` — and therefore
/// `DiagnosticsReport::divergences` and the Python `FitResult.divergences()` —
/// counts divergences over *warmup as well as sampling* iterations. Stan and
/// PyMC report post-warmup divergences only. Divergences during step-size
/// adaptation are expected and benign; divergences after warmup indicate
/// posterior bias. Statistical tests here gate on the post-warmup count.
pub fn divergence_split(result: &SampleResult) -> (usize, usize) {
    let mut warm = 0;
    let mut post = 0;
    for chain in &result.transitions {
        for t in chain {
            if t.divergent {
                if t.is_warmup {
                    warm += 1;
                } else {
                    post += 1;
                }
            }
        }
    }
    (warm, post)
}

/// Assert that no more than `max` divergences occurred *after* warmup.
pub fn assert_post_warmup_divergences(result: &SampleResult, max: usize) {
    let (warm, post) = divergence_split(result);
    assert!(
        post <= max,
        "{post} post-warmup divergences > allowed {max} (plus {warm} during warmup)"
    );
}

/// Assert the post-warmup divergence *rate* stays under `max_rate`.
///
/// Used for targets whose unconstrained geometry is genuinely stiff (anything
/// on a log scale has a doubly-exponential right tail), where a small residual
/// rate is expected from the step size rather than from a correctness defect.
/// The accompanying moment assertions are what establish the absence of bias;
/// this gate only catches a rate blowing up.
pub fn assert_post_warmup_divergence_rate(result: &SampleResult, max_rate: f64) {
    let (warm, post) = divergence_split(result);
    let draws: usize = result.samples.iter().map(|c| c.len()).sum();
    let rate = post as f64 / draws.max(1) as f64;
    assert!(
        rate <= max_rate,
        "post-warmup divergence rate {:.4}% ({post}/{draws}) > allowed {:.4}% \
         (plus {warm} during warmup)",
        100.0 * rate,
        100.0 * max_rate
    );
}

/// Convergence / health gate applied to every statistical test.
pub fn assert_converged(report: &DiagnosticsReport, max_rhat: f64) {
    for p in &report.params {
        assert!(
            p.r_hat.is_finite() && p.r_hat < max_rhat,
            "{}: r_hat {} exceeded {max_rhat}",
            p.name,
            p.r_hat
        );
    }
}

// ── Small dense linear algebra (avoids adding a dependency) ─────────────────

/// Invert a symmetric positive-definite matrix via Gauss-Jordan with partial
/// pivoting. Panics if the matrix is singular to working precision.
pub fn invert(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for col in 0..n {
        let (pivot_row, _) = (col..n)
            .map(|r| (r, m[r][col].abs()))
            .fold((col, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        assert!(
            m[pivot_row][col].abs() > 1e-14,
            "matrix is singular at column {col}"
        );
        m.swap(col, pivot_row);
        let pivot = m[col][col];
        for v in m[col].iter_mut() {
            *v /= pivot;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..2 * n {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    m.into_iter().map(|row| row[n..].to_vec()).collect()
}

pub fn mat_vec(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(b).map(|(x, y)| x * y).sum())
        .collect()
}

/// Posterior of a linear-Gaussian model expressed through its information form:
/// prior precision `lambda0`, prior information vector `h0`, plus the
/// likelihood contribution `X' Sigma_y^{-1} X` and `X' Sigma_y^{-1} y`.
///
/// Returns `(mean, covariance)`.
pub struct GaussianPosterior {
    pub mean: Vec<f64>,
    pub cov: Vec<Vec<f64>>,
}

impl GaussianPosterior {
    pub fn sd(&self, i: usize) -> f64 {
        self.cov[i][i].sqrt()
    }
}

/// Build the exact posterior of `y ~ N(X beta, sigma^2 I)` with independent
/// Normal priors `beta_k ~ N(prior_mean[k], prior_sd[k])`.
///
/// `x` is `n` rows of `p` columns.
pub fn linear_gaussian_posterior(
    x: &[Vec<f64>],
    y: &[f64],
    sigma: f64,
    prior_mean: &[f64],
    prior_sd: &[f64],
) -> GaussianPosterior {
    let p = prior_mean.len();
    let inv_var = 1.0 / (sigma * sigma);
    let mut lambda = vec![vec![0.0; p]; p];
    let mut h = vec![0.0; p];

    for (k, item) in lambda.iter_mut().enumerate().take(p) {
        item[k] += 1.0 / (prior_sd[k] * prior_sd[k]);
        h[k] += prior_mean[k] / (prior_sd[k] * prior_sd[k]);
    }
    for (row, yi) in x.iter().zip(y) {
        for j in 0..p {
            h[j] += inv_var * row[j] * yi;
            for k in 0..p {
                lambda[j][k] += inv_var * row[j] * row[k];
            }
        }
    }

    let cov = invert(&lambda);
    let mean = mat_vec(&cov, &h);
    GaussianPosterior { mean, cov }
}

// ── Deterministic RNG utilities ────────────────────────────────────────────

/// Split-mix style deterministic uniform stream. Used where we want reproducible
/// synthetic data without depending on a particular `rand` version's stream.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(2685821657736338717).wrapping_add(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1).
    pub fn uniform(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v.clamp(1e-15, 1.0 - 1e-15)
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn normal_with(&mut self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * self.normal()
    }
}

// ── Rank / uniformity statistics for SBC ───────────────────────────────────

/// Chi-square statistic for `counts` against a uniform expectation, plus the
/// number of degrees of freedom.
pub fn chi_square_uniform(counts: &[usize]) -> (f64, usize) {
    let total: usize = counts.iter().sum();
    let expected = total as f64 / counts.len() as f64;
    let stat = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum();
    (stat, counts.len() - 1)
}

/// Upper-tail probability of a chi-square distribution with `df` degrees of
/// freedom, via the regularized upper incomplete gamma function Q(df/2, x/2).
pub fn chi_square_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df as f64 / 2.0, x / 2.0)
}

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..500 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Lanczos log-gamma.
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, &g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}
