use std::io::{IsTerminal, Write};
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

fn fmt_speed(n: f64) -> String {
    if n >= 1_000.0 {
        format!("{:.1}k/s", n / 1_000.0)
    } else if n >= 1.0 {
        format!("{:.1}/s", n)
    } else if n > 0.0 {
        format!("{:.2}/s", n)
    } else {
        "-.--/s".to_string()
    }
}

fn fmt_time(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        let mins = (secs / 60.0) as usize;
        let s = (secs % 60.0) as usize;
        format!("{}:{:02}", mins, s)
    } else {
        let hrs = (secs / 3600.0) as usize;
        let mins = ((secs % 3600.0) / 60.0) as usize;
        format!("{}h{:02}m", hrs, mins)
    }
}

/// Render one in-place update to a TTY using `\r`.
/// The line is kept short enough (≤ 79 chars) to avoid wrapping.
fn render_tty(state: &ProgressState, err: &mut impl Write) {
    let completed = state.completed.load(Ordering::Relaxed);
    let total = state.total_iters;
    let divs = state.divergences.load(Ordering::Relaxed);
    let elapsed = state.start_time.elapsed().as_secs_f64();
    let is_done = state.done.load(Ordering::Relaxed);

    let pct = if total > 0 { (completed * 100 / total).min(100) } else { 0 };
    let speed = if elapsed > 0.1 { completed as f64 / elapsed } else { 0.0 };
    let remaining = if speed > 0.0 && completed < total {
        (total - completed) as f64 / speed
    } else {
        0.0
    };

    // Bar: 20 chars wide — keeps total line ≤ 79 chars for standard terminals.
    let bar_width = 20usize;
    let filled = if total > 0 { (bar_width * completed).min(bar_width * total) / total } else { 0 };
    let bar = format!("{}{}", "━".repeat(filled), "╌".repeat(bar_width - filled));

    if is_done {
        let _ = write!(
            err,
            "\rSampling {ch}ch {bar} {pct:>3}% {done}/{tot} | {divs} div | {t}\x1b[K\n",
            ch = state.num_chains,
            bar = bar,
            pct = pct,
            done = completed,
            tot = total,
            divs = divs,
            t = fmt_time(elapsed),
        );
    } else {
        let _ = write!(
            err,
            "\rSampling {ch}ch {bar} {pct:>3}% {done}/{tot} | {divs} div | {spd} | ~{eta}\x1b[K",
            ch = state.num_chains,
            bar = bar,
            pct = pct,
            done = completed,
            tot = total,
            divs = divs,
            spd = fmt_speed(speed),
            eta = fmt_time(remaining),
        );
    }
    let _ = err.flush();
}

/// Render a plain timestamped line (for non-TTY output like pipes/files).
fn render_plain(state: &ProgressState, err: &mut impl Write) {
    let completed = state.completed.load(Ordering::Relaxed);
    let total = state.total_iters;
    let divs = state.divergences.load(Ordering::Relaxed);
    let elapsed = state.start_time.elapsed().as_secs_f64();
    let is_done = state.done.load(Ordering::Relaxed);

    let pct = if total > 0 { (completed * 100 / total).min(100) } else { 0 };
    let speed = if elapsed > 0.1 { completed as f64 / elapsed } else { 0.0 };
    let remaining = if speed > 0.0 && completed < total {
        (total - completed) as f64 / speed
    } else {
        0.0
    };

    if is_done {
        let _ = writeln!(
            err,
            "Sampling done: {}/{} | {} div | elapsed {}",
            completed, total, divs, fmt_time(elapsed),
        );
    } else {
        let _ = writeln!(
            err,
            "Sampling: {}/{} ({}%) | {} div | {} | ~{} remaining",
            completed, total, pct, divs, fmt_speed(speed), fmt_time(remaining),
        );
    }
    let _ = err.flush();
}

/// Spawn a background thread that renders the progress bar at ~10 Hz (TTY)
/// or every ~10 s / 10% milestone (non-TTY).
pub fn spawn_progress_thread(state: Arc<ProgressState>) -> std::thread::JoinHandle<()> {
    let is_tty = std::io::stderr().is_terminal();
    std::thread::spawn(move || {
        let mut last_plain_pct = 0usize;
        let mut last_plain_time = std::time::Instant::now();
        let mut err = std::io::stderr();

        while !state.done.load(Ordering::Relaxed) {
            if is_tty {
                render_tty(&state, &mut err);
                std::thread::sleep(Duration::from_millis(100));
            } else {
                // For non-TTY: print every 10% progress or every 10 seconds.
                let completed = state.completed.load(Ordering::Relaxed);
                let total = state.total_iters;
                let pct = if total > 0 { completed * 100 / total } else { 0 };
                let secs_since = last_plain_time.elapsed().as_secs_f64();
                if pct >= last_plain_pct + 10 || secs_since >= 10.0 {
                    render_plain(&state, &mut err);
                    last_plain_pct = (pct / 10) * 10;
                    last_plain_time = std::time::Instant::now();
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        // Final line
        if is_tty {
            render_tty(&state, &mut err);
        } else {
            render_plain(&state, &mut err);
        }
    })
}
