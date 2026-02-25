use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shared progress state updated atomically by all sampler chains.
///
/// A dedicated thread reads these counters and renders a live progress
/// bar to stderr, completely independent of the sampling threads.
pub struct ProgressState {
    pub total_iters: usize,
    pub completed: AtomicUsize,
    pub divergences: AtomicUsize,
    pub done: AtomicBool,
    pub start_time: Instant,
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub num_leapfrog_steps: usize,
}

impl ProgressState {
    pub fn new(
        num_chains: usize,
        num_draws: usize,
        num_warmup: usize,
        num_leapfrog_steps: usize,
    ) -> Self {
        Self {
            total_iters: num_chains * (num_warmup + num_draws),
            completed: AtomicUsize::new(0),
            divergences: AtomicUsize::new(0),
            done: AtomicBool::new(false),
            start_time: Instant::now(),
            num_chains,
            num_draws,
            num_warmup,
            num_leapfrog_steps,
        }
    }

    pub fn increment(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_divergence(&self) {
        self.divergences.fetch_add(1, Ordering::Relaxed);
    }

    pub fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
    }
}

fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn fmt_speed(n: f64) -> String {
    if n >= 1_000_000.0 {
        format!("{:.1}M", n / 1_000_000.0)
    } else if n >= 1_000.0 {
        format!("{:.1}k", n / 1_000.0)
    } else {
        format!("{:.0}", n)
    }
}

fn fmt_time(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else {
        let mins = (secs / 60.0) as usize;
        let s = (secs % 60.0) as usize;
        format!("{}:{:02}", mins, s)
    }
}

fn render(state: &ProgressState) {
    let completed = state.completed.load(Ordering::Relaxed);
    let total = state.total_iters;
    let divs = state.divergences.load(Ordering::Relaxed);
    let elapsed = state.start_time.elapsed().as_secs_f64();

    let pct = if total > 0 {
        (completed * 100 / total).min(100)
    } else {
        0
    };
    let speed = if elapsed > 0.05 {
        completed as f64 / elapsed
    } else {
        0.0
    };
    let remaining = if speed > 0.0 && completed < total {
        (total - completed) as f64 / speed
    } else {
        0.0
    };
    let grad_evals = completed * (state.num_leapfrog_steps + 1);

    let bar_width = 30;
    let filled = if total > 0 {
        (bar_width * completed).min(bar_width * total) / total
    } else {
        0
    };
    let bar: String = "━".repeat(filled) + &"╌".repeat(bar_width - filled);

    let is_done = state.done.load(Ordering::Relaxed);
    let mut err = std::io::stderr().lock();

    if is_done {
        let _ = write!(
            err,
            "\rSampling {} chains {} {:>3}% │ {}/{} │ {} divergences │ {} grad evals │ {}\x1b[K\n",
            state.num_chains,
            bar,
            pct,
            fmt_count(completed),
            fmt_count(total),
            divs,
            fmt_count(grad_evals),
            fmt_time(elapsed),
        );
    } else {
        let _ = write!(
            err,
            "\rSampling {} chains {} {:>3}% │ {}/{} │ {} divs │ {} it/s │ {} < ~{}\x1b[K",
            state.num_chains,
            bar,
            pct,
            fmt_count(completed),
            fmt_count(total),
            divs,
            fmt_speed(speed),
            fmt_time(elapsed),
            fmt_time(remaining),
        );
    }
    let _ = err.flush();
}

/// Spawn a background thread that renders the progress bar at ~10 Hz.
/// Returns a join handle; call `state.finish()` then `handle.join()` to
/// clean up after sampling.
pub fn spawn_progress_thread(state: Arc<ProgressState>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !state.done.load(Ordering::Relaxed) {
            render(&state);
            std::thread::sleep(Duration::from_millis(100));
        }
        render(&state);
    })
}
