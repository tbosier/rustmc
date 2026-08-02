use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{
    Bernoulli as BernoulliDist, Distribution, Normal as NormalDist, Poisson as PoissonDist,
};
use rustmc_core::diagnostics::{DiagnosticsReport, ParamDiagnostics};
use rustmc_core::distributions::{HalfNormal, Normal};
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{sample as run_sample, SamplerConfig, SamplerType};

const CHAIN_COUNT: usize = 4;
const DEFAULT_DRAWS: usize = 600;
const DEFAULT_WARMUP: usize = 600;
const FUNNEL_DRAWS: usize = 400;
const FUNNEL_WARMUP: usize = 1200;

fn sample_graph(
    graph: Graph,
    seed: u64,
    draws: usize,
    warmup: usize,
    max_tree_depth: usize,
) -> rustmc_core::sampler::SampleResult {
    run_sample(
        graph,
        SamplerConfig {
            sampler: SamplerType::Nuts,
            num_chains: CHAIN_COUNT,
            num_draws: draws,
            num_warmup: warmup,
            step_size: 0.0,
            target_accept: 0.80,
            num_leapfrog_steps: 15,
            max_tree_depth,
            seed,
            num_threads: 1,
            show_progress: false,
        },
    )
    .expect("sampling failed")
}

fn diag<'a>(report: &'a DiagnosticsReport, name: &str) -> &'a ParamDiagnostics {
    report
        .params
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("missing diagnostic for {name}"))
}

fn assert_health(report: &DiagnosticsReport, max_rhat: f64, min_ess: f64, max_divergences: usize) {
    assert!(
        report.divergences <= max_divergences,
        "divergences {} > {}",
        report.divergences,
        max_divergences
    );
    assert!(
        report
            .params
            .iter()
            .all(|p| p.r_hat.is_finite() && p.r_hat < max_rhat),
        "some r_hat values exceeded {max_rhat}: {:?}",
        report
            .params
            .iter()
            .map(|p| (&p.name, p.r_hat))
            .collect::<Vec<_>>()
    );
    assert!(
        report
            .params
            .iter()
            .all(|p| p.ess_bulk.is_finite() && p.ess_bulk >= min_ess),
        "some ESS values fell below {min_ess}: {:?}",
        report
            .params
            .iter()
            .map(|p| (&p.name, p.ess_bulk))
            .collect::<Vec<_>>()
    );
}

fn assert_scalar(report: &DiagnosticsReport, name: &str, truth: f64, tol: f64) {
    let p = diag(report, name);
    assert!(
        (p.mean - truth).abs() <= tol,
        "{name} mean {} not within {tol} of truth {truth}",
        p.mean
    );
}

fn assert_vector_rmse(report: &DiagnosticsReport, prefix: &str, truth: &[f64], tol: f64) {
    let mut sum_sq = 0.0;
    for (i, truth_i) in truth.iter().enumerate() {
        let bracket = format!("{prefix}[{i}]");
        let underscore = format!("{prefix}_{i}");
        let p = report
            .params
            .iter()
            .find(|p| p.name == bracket || p.name == underscore)
            .unwrap_or_else(|| panic!("missing diagnostic for {prefix} element {i}"));
        sum_sq += (p.mean - truth_i).powi(2);
    }
    let rmse = (sum_sq / truth.len() as f64).sqrt();
    assert!(rmse <= tol, "{prefix} RMSE {rmse} > {tol}");
}

fn one_hot_columns(groups: &[usize], n_groups: usize) -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0; groups.len()]; n_groups];
    for (row, &group) in groups.iter().enumerate() {
        cols[group][row] = 1.0;
    }
    cols
}

fn fused_linear_mu(
    graph: &mut Graph,
    params: &[rustmc_core::graph::NodeId],
    columns: &[Vec<f64>],
    intercept: Option<rustmc_core::graph::NodeId>,
) -> rustmc_core::graph::NodeId {
    let data_indices = columns
        .iter()
        .map(|col| graph.store_data_vec(col.clone()))
        .collect::<Vec<_>>();
    graph.fused_linear_mu(params.to_vec(), data_indices, intercept)
}

#[test]
fn intercept_only_gaussian_recovers_location_and_scale() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let mu_true = 1.5;
    let sigma_true = 0.6;
    let n = 120;
    let dist = NormalDist::new(mu_true, sigma_true).unwrap();
    let y: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

    let mut graph = Graph::new();
    let mu = Normal::prior(&mut graph, "mu", 0.0, 5.0);
    let sigma = HalfNormal::prior(&mut graph, "sigma", 1.0);
    let mu_vec = graph.scalar_broadcast(mu);
    let obs_idx = graph.add_obs_data(y);
    graph.normal_obs_logp(mu_vec, sigma, obs_idx);

    let result = sample_graph(graph, 101, DEFAULT_DRAWS, DEFAULT_WARMUP, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.02, 100.0, 15);
    assert_scalar(&report, "mu", mu_true, 0.15);
    assert_scalar(&report, "sigma", sigma_true, 0.15);
}

#[test]
fn linear_regression_recovers_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    let alpha_true = 0.8;
    let beta_true = -1.7;
    let sigma_true = 0.5;
    let n = 140;
    let x_dist = NormalDist::new(0.0, 1.0).unwrap();
    let noise_dist = NormalDist::new(0.0, sigma_true).unwrap();
    let x: Vec<f64> = (0..n).map(|_| x_dist.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| alpha_true + beta_true * xi + noise_dist.sample(&mut rng))
        .collect();

    let mut graph = Graph::new();
    let alpha = Normal::prior(&mut graph, "alpha", 0.0, 5.0);
    let beta = Normal::prior(&mut graph, "beta", 0.0, 2.0);
    let sigma = HalfNormal::prior(&mut graph, "sigma", 1.0);
    let mu = fused_linear_mu(&mut graph, &[alpha, beta], &[vec![1.0; n], x], None);
    let obs_idx = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, sigma, obs_idx);

    let result = sample_graph(graph, 102, DEFAULT_DRAWS, DEFAULT_WARMUP, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.02, 100.0, 25);
    assert_scalar(&report, "alpha", alpha_true, 0.15);
    assert_scalar(&report, "beta", beta_true, 0.15);
    assert_scalar(&report, "sigma", sigma_true, 0.15);
}

#[test]
fn logistic_regression_recovers_linear_predictor() {
    let mut rng = ChaCha8Rng::seed_from_u64(33);
    let alpha_true = -0.4;
    let beta_true = 1.3;
    let n = 160;
    let x_dist = NormalDist::new(0.0, 1.0).unwrap();
    let x: Vec<f64> = (0..n).map(|_| x_dist.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let p = 1.0 / (1.0 + (-(alpha_true + beta_true * xi)).exp());
            BernoulliDist::new(p).unwrap().sample(&mut rng) as u8 as f64
        })
        .collect();

    let mut graph = Graph::new();
    let alpha = Normal::prior(&mut graph, "alpha", 0.0, 3.0);
    let beta = Normal::prior(&mut graph, "beta", 0.0, 3.0);
    let mu = fused_linear_mu(&mut graph, &[alpha, beta], &[vec![1.0; n], x], None);
    let obs_idx = graph.add_obs_data(y);
    graph.obs_logp_bernoulli_logit(mu, obs_idx);

    let result = sample_graph(graph, 103, DEFAULT_DRAWS, DEFAULT_WARMUP, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.03, 80.0, 15);
    assert_scalar(&report, "alpha", alpha_true, 0.25);
    assert_scalar(&report, "beta", beta_true, 0.4);
}

#[test]
fn poisson_glm_recovers_rate_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(44);
    let alpha_true = 0.2;
    let beta_true = 0.4;
    let n = 150;
    let x_dist = NormalDist::new(0.0, 1.0).unwrap();
    let x: Vec<f64> = (0..n).map(|_| x_dist.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let lam = (alpha_true + beta_true * xi).exp().max(1e-12);
            PoissonDist::new(lam).unwrap().sample(&mut rng)
        })
        .collect();

    let mut graph = Graph::new();
    let alpha = Normal::prior(&mut graph, "alpha", 0.0, 3.0);
    let beta = Normal::prior(&mut graph, "beta", 0.0, 2.0);
    let eta = fused_linear_mu(&mut graph, &[alpha, beta], &[vec![1.0; n], x], None);
    let obs_idx = graph.add_obs_data(y);
    graph.obs_logp_poisson_log(eta, obs_idx);

    let result = sample_graph(graph, 104, DEFAULT_DRAWS, DEFAULT_WARMUP, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.03, 80.0, 20);
    assert_scalar(&report, "alpha", alpha_true, 0.25);
    assert_scalar(&report, "beta", beta_true, 0.25);
}

#[test]
fn ar1_style_regression_recovers_lag_coefficient() {
    let mut rng = ChaCha8Rng::seed_from_u64(55);
    let alpha_true = 0.3;
    let phi_true = 0.7;
    let sigma_true = 0.4;
    let n = 160;
    let noise_dist = NormalDist::new(0.0, sigma_true).unwrap();
    let mut series = vec![0.8];
    for _ in 1..n {
        let prev = *series.last().unwrap();
        series.push(alpha_true + phi_true * prev + noise_dist.sample(&mut rng));
    }
    let lag_y = series[..n - 1].to_vec();
    let y = series[1..].to_vec();

    let mut graph = Graph::new();
    let alpha = Normal::prior(&mut graph, "alpha", 0.0, 2.0);
    let phi = Normal::prior(&mut graph, "phi", 0.0, 1.0);
    let sigma = HalfNormal::prior(&mut graph, "sigma", 1.0);
    let mu = fused_linear_mu(&mut graph, &[alpha, phi], &[vec![1.0; n - 1], lag_y], None);
    let obs_idx = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, sigma, obs_idx);

    let result = sample_graph(graph, 105, DEFAULT_DRAWS, DEFAULT_WARMUP, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.03, 80.0, 20);
    assert_scalar(&report, "alpha", alpha_true, 0.15);
    assert_scalar(&report, "phi", phi_true, 0.15);
    assert_scalar(&report, "sigma", sigma_true, 0.15);
}

#[test]
fn ridge_regression_recovers_high_dimensional_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(66);
    let n = 180;
    let p = 6;
    let sigma_true = 0.5;
    let x_dist = NormalDist::new(0.0, 1.0).unwrap();
    let beta_dist = NormalDist::new(0.0, 0.8).unwrap();
    let noise_dist = NormalDist::new(0.0, sigma_true).unwrap();

    let mut x_cols = vec![vec![0.0; n]; p];
    for col in &mut x_cols {
        for v in col.iter_mut() {
            *v = x_dist.sample(&mut rng);
        }
    }
    let beta_true: Vec<f64> = (0..p).map(|_| beta_dist.sample(&mut rng)).collect();
    let mut y = vec![0.0; n];
    for (i, observation) in y.iter_mut().enumerate().take(n) {
        let mut mu = 0.0;
        for j in 0..p {
            mu += x_cols[j][i] * beta_true[j];
        }
        *observation = mu + noise_dist.sample(&mut rng);
    }

    let mut graph = Graph::new();
    let beta_start = graph.add_vector_params("beta", p);
    graph.vector_normal_logp(beta_start, p, 0.0, 1.0);
    let x_cols_ref = &x_cols;
    let matrix_idx = graph.store_matrix(
        (0..n)
            .flat_map(|row| (0..p).map(move |col| x_cols_ref[col][row]))
            .collect(),
        n,
        p,
    );
    let mu = graph.mat_vec_mul(matrix_idx, beta_start, p, None);
    let obs_idx = graph.add_obs_data(y);
    let sigma = graph.add_constant(sigma_true);
    graph.normal_obs_logp(mu, sigma, obs_idx);

    let result = sample_graph(graph, 106, 200, 200, 8);
    let report = result.diagnostics();
    assert_health(&report, 1.08, 40.0, 20);
    assert_vector_rmse(&report, "beta", &beta_true, 0.20);
}

#[test]
fn partial_pooling_panel_recovers_group_effects() {
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let groups = 5;
    let per_group = 30;
    let n = groups * per_group;
    let mu_alpha_true = 1.0;
    let sigma_alpha_true = 0.6;
    let beta_true = 0.7;
    let sigma_true = 0.4;
    let group_dist = NormalDist::new(mu_alpha_true, sigma_alpha_true).unwrap();
    let alpha_true: Vec<f64> = (0..groups).map(|_| group_dist.sample(&mut rng)).collect();
    let x_dist = NormalDist::new(0.0, 1.0).unwrap();
    let noise_dist = NormalDist::new(0.0, sigma_true).unwrap();

    let mut group_idx = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for (g, &alpha) in alpha_true.iter().enumerate().take(groups) {
        for _ in 0..per_group {
            let xi = x_dist.sample(&mut rng);
            x.push(xi);
            group_idx.push(g);
            y.push(alpha + beta_true * xi + noise_dist.sample(&mut rng));
        }
    }
    let group_cols = one_hot_columns(&group_idx, groups);

    let mut graph = Graph::new();
    let mu_alpha = Normal::prior(&mut graph, "mu_alpha", 0.0, 5.0);
    let sigma_alpha = HalfNormal::prior(&mut graph, "sigma_alpha", 1.0);
    let alpha_nodes: Vec<_> = (0..groups)
        .map(|g| Normal::prior_with_nodes(&mut graph, &format!("alpha_{g}"), mu_alpha, sigma_alpha))
        .collect();
    let beta = Normal::prior(&mut graph, "beta", 0.0, 2.0);
    let sigma = HalfNormal::prior(&mut graph, "sigma", 1.0);
    let mut columns = group_cols;
    columns.push(x);
    let mut params = alpha_nodes.clone();
    params.push(beta);
    let mu = fused_linear_mu(&mut graph, &params, &columns, None);
    let obs_idx = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, sigma, obs_idx);

    let result = sample_graph(graph, 107, 2000, 1000, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.50, 20.0, 40);
    assert_scalar(&report, "mu_alpha", mu_alpha_true, 0.6);
    assert_scalar(&report, "sigma_alpha", sigma_alpha_true, 0.4);
    assert_scalar(&report, "beta", beta_true, 0.25);
    assert_scalar(&report, "sigma", sigma_true, 0.15);
    assert_vector_rmse(&report, "alpha", &alpha_true, 0.7);
}

#[test]
fn noncentered_hierarchical_poisson_recovers_partial_pooling_counts() {
    let mut rng = ChaCha8Rng::seed_from_u64(88);
    let groups = 4;
    let per_group = 35;
    let n = groups * per_group;
    let mu_alpha_true = 0.25;
    let sigma_alpha_true = 0.5;
    let beta_true = 0.35;
    let group_dist = NormalDist::new(mu_alpha_true, sigma_alpha_true).unwrap();
    let alpha_true: Vec<f64> = (0..groups).map(|_| group_dist.sample(&mut rng)).collect();
    let x_dist = NormalDist::new(0.0, 1.0).unwrap();

    let mut group_idx = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for (g, &alpha) in alpha_true.iter().enumerate().take(groups) {
        for _ in 0..per_group {
            let xi = x_dist.sample(&mut rng);
            x.push(xi);
            group_idx.push(g);
            let lam = (alpha + beta_true * xi).exp().max(1e-12);
            y.push(PoissonDist::new(lam).unwrap().sample(&mut rng));
        }
    }
    let group_cols = one_hot_columns(&group_idx, groups);

    let mut graph = Graph::new();
    let mu_alpha = Normal::prior(&mut graph, "mu_alpha", 0.0, 2.0);
    let sigma_alpha = HalfNormal::prior(&mut graph, "sigma_alpha", 1.0);
    let alpha_nodes: Vec<_> = (0..groups)
        .map(|g| {
            let z = Normal::prior(&mut graph, &format!("z_{g}"), 0.0, 1.0);
            let scaled = graph.mul(sigma_alpha, z);
            graph.add(mu_alpha, scaled)
        })
        .collect();
    let beta = Normal::prior(&mut graph, "beta", 0.0, 2.0);
    let mut columns = group_cols;
    columns.push(x);
    let mut params = alpha_nodes.clone();
    params.push(beta);
    let eta = fused_linear_mu(&mut graph, &params, &columns, None);
    let obs_idx = graph.add_obs_data(y);
    graph.obs_logp_poisson_log(eta, obs_idx);

    let result = sample_graph(graph, 108, DEFAULT_DRAWS, DEFAULT_WARMUP, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.02, 80.0, 20);
    assert_scalar(&report, "mu_alpha", mu_alpha_true, 0.35);
    assert_scalar(&report, "sigma_alpha", sigma_alpha_true, 0.45);
    assert_scalar(&report, "beta", beta_true, 0.20);
    let z_true: Vec<f64> = alpha_true
        .iter()
        .map(|alpha| (alpha - mu_alpha_true) / sigma_alpha_true)
        .collect();
    assert_vector_rmse(&report, "z", &z_true, 0.75);
}

#[test]
fn centered_eight_schools_recovers_or_reports_bad_geometry() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mu_true = 5.0;
    let tau_true = 2.0;
    let sigma = [2.0, 2.5, 3.0, 2.2, 2.8, 3.3, 2.1, 2.6];
    let theta_dist = NormalDist::new(mu_true, tau_true).unwrap();
    let theta_true: Vec<f64> = (0..8).map(|_| theta_dist.sample(&mut rng)).collect();
    let y: Vec<f64> = theta_true
        .iter()
        .zip(sigma.iter())
        .map(|(&theta, &s)| NormalDist::new(theta, s).unwrap().sample(&mut rng))
        .collect();

    let mut graph = Graph::new();
    let mu = Normal::prior(&mut graph, "mu", 0.0, 10.0);
    let tau = HalfNormal::prior(&mut graph, "tau", 5.0);
    for i in 0..8 {
        let theta = Normal::prior_with_nodes(&mut graph, &format!("theta_{i}"), mu, tau);
        let yi = graph.add_constant(y[i]);
        let si = graph.add_constant(sigma[i]);
        graph.normal_logp(yi, theta, si);
    }

    let result = sample_graph(graph, 109, 800, 800, 10);
    let report = result.diagnostics();
    let theta_rmse = (theta_true
        .iter()
        .enumerate()
        .map(|(i, truth)| (diag(&report, &format!("theta_{i}")).mean - truth).powi(2))
        .sum::<f64>()
        / theta_true.len() as f64)
        .sqrt();
    // A centered funnel is a deliberate negative control: it may either be
    // sampled accurately, or it must raise a convergence/geometry signal.
    // This lets a future sampler improvement pass without hiding a bad fit.
    let recovered = (diag(&report, "mu").mean - mu_true).abs() <= 1.0
        && (diag(&report, "tau").mean - tau_true).abs() <= 0.8
        && theta_rmse <= 1.8
        && report.divergences == 0
        && report.params.iter().all(|param| {
            param.r_hat.is_finite()
                && param.r_hat <= 1.01
                && param.ess_bulk.is_finite()
                && param.ess_bulk >= 100.0
        });
    let problem_reported = report.divergences > 0
        || report.params.iter().any(|param| {
            !param.r_hat.is_finite()
                || param.r_hat > 1.01
                || !param.ess_bulk.is_finite()
                || param.ess_bulk < 100.0
        });
    assert!(
        recovered || problem_reported,
        "centered eight-schools failed recovery without a diagnostic warning"
    );
}

#[test]
fn eight_schools_noncentered_recovers_hyperparameters() {
    let mut rng = ChaCha8Rng::seed_from_u64(100);
    let mu_true = 5.0;
    let tau_true = 2.0;
    let sigma = [2.0, 2.5, 3.0, 2.2, 2.8, 3.3, 2.1, 2.6];
    let theta_dist = NormalDist::new(mu_true, tau_true).unwrap();
    let theta_true: Vec<f64> = (0..8).map(|_| theta_dist.sample(&mut rng)).collect();
    let y: Vec<f64> = theta_true
        .iter()
        .zip(sigma.iter())
        .map(|(&theta, &s)| NormalDist::new(theta, s).unwrap().sample(&mut rng))
        .collect();

    let mut graph = Graph::new();
    let mu = Normal::prior(&mut graph, "mu", 0.0, 10.0);
    let tau = HalfNormal::prior(&mut graph, "tau", 5.0);
    for i in 0..8 {
        let z = Normal::prior(&mut graph, &format!("z_{i}"), 0.0, 1.0);
        let tau_z = graph.mul(tau, z);
        let theta = graph.add(mu, tau_z);
        let yi = graph.add_constant(y[i]);
        let si = graph.add_constant(sigma[i]);
        graph.normal_logp(yi, theta, si);
    }

    let result = sample_graph(graph, 110, 800, 800, 10);
    let report = result.diagnostics();
    assert_health(&report, 1.04, 60.0, 30);
    assert_scalar(&report, "mu", mu_true, 0.75);
    assert_scalar(&report, "tau", tau_true, 1.3);
    for i in 0..8 {
        let name = format!("z_{i}");
        let p = diag(&report, &name);
        assert!(
            p.mean.abs() < 1.25,
            "{name} mean too far from zero: {}",
            p.mean
        );
    }
}

#[test]
fn centered_funnel_recovers_or_reports_bad_geometry() {
    let mut graph = Graph::new();
    let y = Normal::prior(&mut graph, "y", 0.0, 3.0);
    let x = graph.add_param("x");
    let half = graph.add_constant(0.5);
    let y_half = graph.mul(y, half);
    let sigma = graph.exp(y_half);
    let zero = graph.add_constant(0.0);
    graph.normal_logp(x, zero, sigma);

    let result = sample_graph(graph, 111, FUNNEL_DRAWS, FUNNEL_WARMUP, 12);
    let report = result.diagnostics();
    let y_diag = diag(&report, "y");
    // As above, the test guards against silent failure while allowing a
    // genuinely improved sampler to turn this negative control healthy.
    let recovered = y_diag.mean.abs() <= 0.75
        && (y_diag.std - 3.0).abs() < 1.0
        && report.divergences == 0
        && y_diag.r_hat.is_finite()
        && y_diag.r_hat <= 1.01;
    let problem_reported = report.divergences > 0
        || !y_diag.r_hat.is_finite()
        || y_diag.r_hat > 1.01
        || !y_diag.ess_bulk.is_finite()
        || y_diag.ess_bulk < 100.0;
    assert!(
        recovered || problem_reported,
        "centered funnel failed recovery without a diagnostic warning: {y_diag:?}"
    );
}

#[test]
fn noncentered_funnel_is_stable_and_recovers_latent_scale() {
    let mut graph = Graph::new();
    let y = Normal::prior(&mut graph, "y", 0.0, 3.0);
    let z = Normal::prior(&mut graph, "z", 0.0, 1.0);
    let half = graph.add_constant(0.5);
    let y_half = graph.mul(y, half);
    let scale = graph.exp(y_half);
    let _x = graph.mul(scale, z);

    let result = sample_graph(graph, 112, FUNNEL_DRAWS, FUNNEL_WARMUP, 12);
    let report = result.diagnostics();
    assert_health(&report, 1.05, 50.0, 25);
    assert_scalar(&report, "y", 0.0, 0.5);
    assert_scalar(&report, "z", 0.0, 0.35);
    let y_diag = diag(&report, "y");
    assert!((y_diag.std - 3.0).abs() < 0.75, "y std {}", y_diag.std);
    let z_diag = diag(&report, "z");
    assert!((z_diag.std - 1.0).abs() < 0.35, "z std {}", z_diag.std);
}
