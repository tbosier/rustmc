//! Independent forecasting execution with stable cell identity and bounded workers.
use rayon::prelude::*;
use std::collections::HashSet;

/// Version-one FNV-1a encoding: ASCII `rustmc-cell-v1`, little-endian seed,
/// little-endian UTF-8 ID byte length, ID bytes, then domain bytes.
/// Kernels further derive their existing model/domain/chain streams from this seed.
/// IDs are exact UTF-8 strings; normalization is the caller's responsibility.
pub fn stable_cell_seed(seed: u64, id: &str, domain: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in b"rustmc-cell-v1"
        .iter()
        .copied()
        .chain(seed.to_le_bytes())
        .chain((id.len() as u64).to_le_bytes())
        .chain(id.bytes())
        .chain(domain.bytes())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Run cells in input order on one private pool, also used by nested chain work.
/// At most `chunk_size` cells are dispatched together. Outputs are retained; for
/// bounded total retention, call this on caller-managed chunks and persist/drop
/// each result. IDs and seeds must remain unchanged when resuming.
/// Per-cell errors are collected; pool/configuration errors are outer errors.
pub fn execute_batch<T: Sync, R: Send, E: Send>(
    cells: &[(String, T)],
    seed: u64,
    threads: usize,
    chunk_size: usize,
    fit: impl Fn(&T, u64) -> Result<R, E> + Sync,
) -> Result<Vec<Result<R, E>>, String> {
    if threads == 0 || chunk_size == 0 {
        return Err("threads and chunk_size must be positive".into());
    }
    let mut ids = HashSet::with_capacity(cells.len());
    if cells.iter().any(|(id, _)| !ids.insert(id)) {
        return Err("cell IDs must be unique".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| format!("could not build batch worker pool: {error}"))?;
    let mut results = Vec::with_capacity(cells.len());
    for chunk in cells.chunks(chunk_size) {
        let output = pool.install(|| {
            chunk
                .par_iter()
                .map(|(id, cell)| fit(cell, stable_cell_seed(seed, id, "fit")))
                .collect::<Vec<_>>()
        });
        results.extend(output);
    }
    Ok(results)
}

/// A batch setup failure or the first observed cell failure. Concurrent work
/// already running may finish; later chunks are never dispatched after failure.
#[derive(Debug)]
pub enum BatchError<E> {
    Configuration(String),
    Cell { id: String, error: E },
}

/// Fail-fast variant of `execute_batch`, with cooperative Rayon cancellation.
pub fn execute_batch_fail_fast<T: Sync, R: Send, E: Send>(
    cells: &[(String, T)],
    seed: u64,
    threads: usize,
    chunk_size: usize,
    fit: impl Fn(&T, u64) -> Result<R, E> + Sync,
) -> Result<Vec<R>, BatchError<E>> {
    if threads == 0 || chunk_size == 0 {
        return Err(BatchError::Configuration(
            "threads and chunk_size must be positive".into(),
        ));
    }
    let mut ids = HashSet::with_capacity(cells.len());
    if cells.iter().any(|(id, _)| !ids.insert(id)) {
        return Err(BatchError::Configuration("cell IDs must be unique".into()));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| {
            BatchError::Configuration(format!("could not build batch worker pool: {error}"))
        })?;
    let mut results = Vec::with_capacity(cells.len());
    for chunk in cells.chunks(chunk_size) {
        let output = pool.install(|| {
            chunk
                .par_iter()
                .map(|(id, cell)| {
                    fit(cell, stable_cell_seed(seed, id, "fit")).map_err(|error| BatchError::Cell {
                        id: id.clone(),
                        error,
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        results.extend(output);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayesian_forecast::*;
    fn config(seed: u64) -> BayesianLocalLevelConfig {
        BayesianLocalLevelConfig {
            initial_mean: 0.,
            initial_variance: 2.,
            process_variance_prior: InverseGammaPrior::new(3., 1.).unwrap(),
            observation_variance_prior: InverseGammaPrior::new(3., 1.).unwrap(),
            num_chains: 2,
            num_draws: 12,
            num_warmup: 5,
            thinning: 1,
            seed,
        }
    }
    #[test]
    fn seeded_fits_survive_order_chunks_resume_threads_and_bad_cells() {
        let cells = vec![
            ("alpha".into(), vec![1., 2., 3.]),
            ("β".into(), vec![2., f64::NAN, 1., 2.]),
            ("bad".into(), vec![]),
        ];
        let run = |cells: &[(String, Vec<f64>)], threads, chunk| {
            execute_batch(cells, 91, threads, chunk, |y, seed| {
                fit_bayesian_local_level(y, &config(seed))
            })
            .unwrap()
        };
        let first = run(&cells, 1, 3);
        let other = run(&cells, 3, 1);
        assert_eq!(first[0].as_ref().unwrap(), other[0].as_ref().unwrap());
        assert_eq!(first[1].as_ref().unwrap(), other[1].as_ref().unwrap());
        assert!(first[2].is_err());
        let reversed = run(&[cells[1].clone(), cells[0].clone()], 2, 2);
        assert_eq!(first[0].as_ref().unwrap(), reversed[1].as_ref().unwrap());
        assert_eq!(
            first[1].as_ref().unwrap(),
            run(&cells[1..2], 1, 1)[0].as_ref().unwrap()
        );
        assert_ne!(
            stable_cell_seed(91, "alpha", "fit"),
            stable_cell_seed(91, "alpha", "forecast")
        );
        assert_ne!(
            stable_cell_seed(91, "alpha", "fit"),
            stable_cell_seed(91, "β", "fit")
        );
    }
    #[test]
    fn fail_fast_does_not_dispatch_later_chunks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let cells = vec![("bad".into(), false), ("later".into(), true)];
        let result = execute_batch_fail_fast(&cells, 3, 2, 1, |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
            Err::<(), _>("failed")
        });
        assert!(matches!(result, Err(BatchError::Cell { id, .. }) if id == "bad"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn version_one_seed_encoding_is_fixed() {
        assert_eq!(stable_cell_seed(91, "β", "fit"), 5_264_580_120_131_183_224);
    }
    #[test]
    fn invalid_execution_options_fail_before_work() {
        let run =
            |cells, threads, chunk| execute_batch(cells, 1, threads, chunk, |_, _| Ok::<_, ()>(()));
        let duplicate = [("x".into(), ()), ("x".into(), ())];
        assert!(run(&duplicate, 1, 1).is_err());
        assert!(run(&[], 0, 1).is_err());
        assert!(run(&[], 1, 0).is_err());
    }
}
