// SCRATCH — NOT WIRED INTO THE BUILD.
//
// Measurement harness used to quantify the per-dataset graph-rebuild cost for
// the Task 5 design (docs/architecture/compiled-model.md). To run it:
//
//     cp docs/architecture/scratch/rebind_cost_bench.rs rust_core/examples/bench.rs
//     cargo run --release -p rustmc_core --example bench
//     rm rust_core/examples/bench.rs
//
// Recorded results are in the "Measurements" section of the design doc.

use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{self, BatchSampleConfig, SamplerType};
use std::time::Instant;

const N_OBS: usize = 60;
const N_COLS: usize = 200;
const N_DATASETS: usize = 2000;

fn make_dataset(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(N_OBS * N_COLS);
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    for _ in 0..(N_OBS * N_COLS) {
        x.push(next());
    }
    let y: Vec<f64> = (0..N_OBS).map(|i| x[i * N_COLS] * 1.5 + 0.3).collect();
    (x, y)
}

// Mirrors what compile_python_model does for a matvec model.
fn build_graph(x: Vec<f64>, y: Vec<f64>) -> Graph {
    let mut g = Graph::new();
    let start = g.add_vector_params("beta", N_COLS);
    g.vector_normal_logp(start, N_COLS, 0.0, 1.0);
    let midx = g.store_matrix(x, N_OBS, N_COLS);
    let mu = g.mat_vec_mul(midx, start, N_COLS, None);
    let sigma = g.add_constant(1.0);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sigma, obs);
    g
}

fn main() {
    let datasets: Vec<(Vec<f64>, Vec<f64>)> = (0..N_DATASETS as u64).map(make_dataset).collect();

    // (1) construction cost, per dataset
    let t = Instant::now();
    let graphs: Vec<Graph> = datasets
        .iter()
        .map(|(x, y)| build_graph(x.clone(), y.clone()))
        .collect();
    let build_time = t.elapsed();

    // (2) the extra clone batch_sample() does
    let t = Instant::now();
    let cloned: Vec<(Graph, Vec<f64>)> = graphs.iter().map(|g| (g.clone(), vec![])).collect();
    let clone_time = t.elapsed();

    // (3) Evaluator construction (per chain)
    let t = Instant::now();
    let mut sink = 0usize;
    for g in &graphs {
        let e = Evaluator::new(g);
        sink += e.grad.len();
    }
    let eval_time = t.elapsed();

    // (4) actual sampling
    let cfg = BatchSampleConfig {
        sampler: SamplerType::Nuts,
        num_chains: 1,
        num_draws: 50,
        num_warmup: 50,
        step_size: 0.0,
        num_leapfrog_steps: 15,
        max_tree_depth: 8,
        seed: 42,
        show_progress: false,
    };
    let _ = (&cfg, &cloned);
    let sample_time = std::time::Duration::ZERO;
    let res: Vec<u8> = vec![];

    // (5) forward-pass-only cost (the prediction / prior-predictive path)
    let params = vec![0.1f64; graphs[0].param_count];
    let t = Instant::now();
    let mut acc = 0.0f64;
    for g in &graphs {
        let mut e = Evaluator::new(g);
        e.compute(g, &params);
        acc += e.total_logp;
    }
    let fwd_time = t.elapsed();
    println!("fwd(new+compute)   = {:?}  per-dataset = {:?} acc={:.3}", fwd_time, fwd_time / N_DATASETS as u32, acc);
    let mut e = Evaluator::new(&graphs[0]);
    let t = Instant::now();
    for g in &graphs { e.compute(g, &params); }
    let fwd2 = t.elapsed();
    println!("compute only       = {:?}  per-dataset = {:?}", fwd2, fwd2 / N_DATASETS as u32);

    let bytes_per_graph = (N_OBS * N_COLS + N_OBS) * 8;
    println!("datasets           = {}", N_DATASETS);
    println!("n_obs x n_cols     = {} x {}", N_OBS, N_COLS);
    println!("payload per dataset= {:.2} MiB", bytes_per_graph as f64 / 1048576.0);
    println!("--");
    println!("graph build  total = {:?}  per-dataset = {:?}", build_time, build_time / N_DATASETS as u32);
    println!("graph clone  total = {:?}  per-dataset = {:?}", clone_time, clone_time / N_DATASETS as u32);
    println!("Evaluator::new tot = {:?}  per-dataset = {:?}", eval_time, eval_time / N_DATASETS as u32);
    println!("batch_sample total = {:?}  per-dataset = {:?}", sample_time, sample_time / N_DATASETS as u32);
    println!("sink={} results={}", sink, res.len());
}
