// SCRATCH — NOT WIRED INTO THE BUILD.
//
// Measurement harness for the Task 5 design (docs/architecture/compiled-model.md).
// Every number in that document's "Measurements" section is produced by this file
// in a single run, with no constants to edit.
//
//     cp docs/architecture/scratch/rebind_cost_bench.rs rust_core/examples/bench.rs
//     cargo run --release -p rustmc_core --example bench            # all shapes
//     cargo run --release -p rustmc_core --example bench -- small    # one shape
//     rm rust_core/examples/bench.rs
//
// Runtime is ~10 minutes for the full sweep, dominated by the wide-k shape.
//
// RECORD THE MACHINE LOAD alongside any results you quote (`uptime` before and
// after). The fitN column is a parallel wall clock and inflates badly on a
// contended box — which makes the setup-overhead fraction look SMALLER than it
// is, i.e. contention biases in favour of this design's conclusion. The fit1
// column is single-threaded and is the honest denominator. Numbers in the design
// doc were taken at load 0.28 (wide-n, small) and ~25 (wide-k); the doc reports
// both and flags which is which.
//
// What it measures, per shape:
//   build    — Graph construction, mirroring compile_python_model. SERIAL.
//   clone    — the extra compiled.graph.clone() batch_sample does. SERIAL.
//   evalnew  — Evaluator::new. SERIAL.
//   compute  — one logp+gradient evaluation (the forward/backward hot path).
//   fit1     — ONE dataset through batch_sample. Effectively single-threaded,
//              so it is directly comparable to the serial setup costs above and
//              is immune to how loaded the machine is. THIS IS THE DENOMINATOR
//              the design doc's overhead percentages use.
//   fitN     — the full batch through batch_sample. PARALLEL over rayon's pool.
//              Reported for context only; on a contended machine this inflates,
//              which would make the overhead fraction look *better* than it is.
//
// Why two fit numbers: build/clone/evalnew are single-threaded costs paid in the
// caller's loop before sampling starts. Comparing them against a parallel wall
// clock is apples-to-oranges and biased in favour of the conclusion this harness
// exists to test. fit1 is the honest comparison; fitN is what a user's wall
// clock actually looks like. The doc quotes fit1 and reports both.

use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{self, BatchSampleConfig, SamplerType};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    n_obs: usize,
    n_cols: usize,
    n_datasets: usize,
    draws: usize,
    warmup: usize,
}

const SHAPES: &[Shape] = &[
    Shape { label: "wide-n", n_obs: 2000, n_cols: 8,   n_datasets: 500,  draws: 200, warmup: 200 },
    Shape { label: "small",  n_obs: 60,   n_cols: 8,   n_datasets: 5000, draws: 200, warmup: 200 },
    Shape { label: "wide-k", n_obs: 60,   n_cols: 200, n_datasets: 2000, draws: 50,  warmup: 50  },
];

fn make_dataset(shape: &Shape, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let x: Vec<f64> = (0..shape.n_obs * shape.n_cols).map(|_| next()).collect();
    let y: Vec<f64> = (0..shape.n_obs)
        .map(|i| x[i * shape.n_cols] * 1.5 + 0.3)
        .collect();
    (x, y)
}

/// Mirrors what compile_python_model builds for a matvec model.
fn build_graph(shape: &Shape, x: Vec<f64>, y: Vec<f64>) -> Graph {
    let mut g = Graph::new();
    let start = g.add_vector_params("beta", shape.n_cols);
    g.vector_normal_logp(start, shape.n_cols, 0.0, 1.0);
    let midx = g.store_matrix(x, shape.n_obs, shape.n_cols);
    let mu = g.mat_vec_mul(midx, start, shape.n_cols, None);
    let sigma = g.add_constant(1.0);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sigma, obs);
    g
}

fn per(d: Duration, n: usize) -> Duration {
    d / n as u32
}

fn run_shape(shape: &Shape) {
    let n = shape.n_datasets;
    println!(
        "\n=== {} : n_obs={} n_cols={} datasets={} draws={} warmup={} ===",
        shape.label, shape.n_obs, shape.n_cols, n, shape.draws, shape.warmup
    );

    let datasets: Vec<(Vec<f64>, Vec<f64>)> =
        (0..n as u64).map(|i| make_dataset(shape, i)).collect();

    // (1) construction — what compile_python_model costs, per dataset
    let t = Instant::now();
    let graphs: Vec<Graph> = datasets
        .iter()
        .map(|(x, y)| build_graph(shape, x.clone(), y.clone()))
        .collect();
    let build = t.elapsed();

    // (2) the extra clone batch_sample performs at lib.rs:2300-2303
    let t = Instant::now();
    let cloned: Vec<(Graph, Vec<f64>)> = graphs.iter().map(|g| (g.clone(), vec![])).collect();
    let clone = t.elapsed();

    // (3) Evaluator construction
    let t = Instant::now();
    let mut sink = 0usize;
    for g in &graphs {
        sink += Evaluator::new(g).grad.len();
    }
    let evalnew = t.elapsed();

    // (4) one logp+gradient evaluation — the prediction / forward-pass unit of work
    let params = vec![0.1f64; graphs[0].param_count];
    let mut ev = Evaluator::new(&graphs[0]);
    let t = Instant::now();
    let mut acc = 0.0f64;
    for g in &graphs {
        ev.compute(g, &params);
        acc += ev.total_logp;
    }
    let compute = t.elapsed();

    // (5) the actual fit — THESE MUST ACTUALLY RUN
    let cfg = BatchSampleConfig {
        sampler: SamplerType::Nuts,
        num_chains: 1,
        num_draws: shape.draws,
        num_warmup: shape.warmup,
        step_size: 0.0,
        num_leapfrog_steps: 15,
        max_tree_depth: 8,
        seed: 42,
        show_progress: false,
    };

    // (5a) ONE dataset — effectively serial, comparable to the setup costs,
    //      and unaffected by how contended the machine is.
    let single = vec![(graphs[0].clone(), Vec::<f64>::new())];
    let t = Instant::now();
    let one = sampler::batch_sample(single, cfg.clone()).expect("single batch_sample failed");
    let fit1 = t.elapsed();
    assert_eq!(one.len(), 1);
    assert!(!one[0].samples.is_empty(), "single fit produced no draws");

    // (5b) the whole batch, parallel.
    let t = Instant::now();
    let results = sampler::batch_sample(cloned, cfg).expect("batch_sample failed");
    let fitn = t.elapsed();
    assert_eq!(results.len(), n, "batch_sample returned the wrong count");
    let total_draws: usize = results.iter().map(|r| r.samples.len()).sum();
    assert!(total_draws > 0, "batch_sample produced no draws");

    let payload = (shape.n_obs * shape.n_cols + shape.n_obs) * 8;
    let setup_per = per(build, n) + per(clone, n);
    // Honest denominator: serial setup vs serial fit.
    let overhead_pct = 100.0 * setup_per.as_secs_f64() / fit1.as_secs_f64();
    // Context only: serial setup total vs parallel batch wall clock.
    let overhead_pct_wall = 100.0 * (build + clone).as_secs_f64() / fitn.as_secs_f64();
    let rebuild_vs_compute = setup_per.as_secs_f64() / per(compute, n).as_secs_f64();

    println!(
        "payload/dataset  {:.3} MiB   (total {:.1} MiB)",
        payload as f64 / 1048576.0,
        (payload * n) as f64 / 1048576.0
    );
    println!("{:<9} {:>14} {:>16}", "phase", "total", "per-dataset");
    println!("{:<9} {:>14?} {:>16?}", "build", build, per(build, n));
    println!("{:<9} {:>14?} {:>16?}", "clone", clone, per(clone, n));
    println!("{:<9} {:>14?} {:>16?}", "evalnew", evalnew, per(evalnew, n));
    println!("{:<9} {:>14?} {:>16?}", "compute", compute, per(compute, n));
    println!("{:<9} {:>14?} {:>16?}", "fit1", fit1, fit1);
    println!("{:<9} {:>14?} {:>16?}", "fitN", fitn, per(fitn, n));
    println!("--");
    println!("setup/dataset vs ONE serial fit  : {:.3} %   <-- headline", overhead_pct);
    println!("setup total   vs parallel batch  : {:.3} %   (context; contention-sensitive)",
        overhead_pct_wall);
    println!("rebuild vs one compute()         : {:.2} x", rebuild_vs_compute);
    println!("parallel speedup (fit1*n/fitN)   : {:.1} x",
        fit1.as_secs_f64() * n as f64 / fitn.as_secs_f64());
    println!("(checksums: sink={} acc={:.3} draws={})", sink, acc, total_draws);
}

fn main() {
    let filter: Vec<String> = std::env::args().skip(1).collect();
    println!("rayon threads = {}", rayon::current_num_threads());
    for shape in SHAPES {
        if filter.is_empty() || filter.iter().any(|f| f == shape.label) {
            run_shape(shape);
        }
    }
}
