#!/usr/bin/env python3
"""Plant bugs in the Gibbs samplers and check that their exact-posterior tests catch them.

Each mutation replaces one piece of sampler code, runs the matching unit test in
release mode, and restores the file. A bug counts as caught only when the test
ran and failed one of its moment checks; any other failure, such as a build
error, stops the script, and so does a failure caught only by the ESS floor or a
control. The unmutated tests must pass first. Before each mutation the original
file is copied beside it as `*.unmutated`, and the copy is removed once the file
is restored. If the script is killed mid-mutation, the copy remains and the next
run refuses to start until someone restores or deletes it by hand, so a later
edit to the file is never overwritten.

    python3 scripts/plant_gibbs_bugs.py            # the planted bugs
    python3 scripts/plant_gibbs_bugs.py --seeds 100 [FIRST]  # false alarms

`--seeds N` plants nothing: it reruns each test with `RUSTMC_EXACT_SEED_OFFSET`
set to FIRST (default 1) through FIRST + N - 1, counts the runs that fail, and reports the largest of each
run's worst comparison, in MCSE, against the test's tolerance. Exit status 1 means a planted
bug escaped, or an unmutated test failed. Run from the repository root.
"""
import os
from pathlib import Path
import subprocess
import sys

LOCAL_LEVEL = ("rust_core/src/bayesian_forecast.rs",
               "draws_match_the_exact_posterior_of_both_variances_and_the_terminal_level")
SEASONAL = ("rust_core/src/bayesian_seasonal.rs",
            "draws_match_the_exact_posterior_of_every_variance_and_the_terminal_state")
HIERARCHY = ("rust_core/src/hierarchical.rs",
             "draws_match_the_exact_posterior_of_every_variance_and_mean")

# (description, test, file or None for the test's own, old text, new text)
MUTATIONS = [
    ("process variance counts one transition too few", LOCAL_LEVEL, None,
     "                observations.len(),\n                process_sum_sq,",
     "                observations.len() - 1,\n                process_sum_sq,"),
    ("observation variance counts missing steps", LOCAL_LEVEL, None,
     "                observed_count,\n                observation_sum_sq,",
     "                observations.len(),\n                observation_sum_sq,"),
    ("observation variance updated from the process variance's prior", LOCAL_LEVEL, None,
     "*observation_variance = config.observation_variance_prior.conditional_draw(",
     "*observation_variance = config.process_variance_prior.conditional_draw("),
    ("process sum of squares 15% low", LOCAL_LEVEL, None,
     "                process_sum_sq,\n                rng,",
     "                0.85 * process_sum_sq,\n                rng,"),
    ("seasonal residual uses 90% of the previous effects", SEASONAL, None,
     "let seasonal_residual = pair[1][1] + pair[0][1..].iter().sum::<f64>();",
     "let seasonal_residual = pair[1][1] + 0.9 * pair[0][1..].iter().sum::<f64>();"),
    ("seasonal variance updated from the level variance's prior", SEASONAL, None,
     "*seasonal_variance = config.seasonal_variance_prior.conditional_draw(",
     "*seasonal_variance = config.level_variance_prior.conditional_draw("),
    ("seasonal variance counts one transition too few", SEASONAL, None,
     "                transitions,\n                seasonal_sum_sq,",
     "                transitions - 1,\n                seasonal_sum_sq,"),
    ("level variance counts only observed steps", SEASONAL, None,
     ".conditional_draw(transitions, level_sum_sq, rng)?;",
     ".conditional_draw(observed_count, level_sum_sq, rng)?;"),
    ("initial seasonal effects read in reverse", SEASONAL, "rust_core/src/state_space.rs",
     "initial_mean[state_index] = initial_seasonal_effects[period - state_index];",
     "initial_mean[state_index] = initial_seasonal_effects[state_index];"),
    ("program variance counts one program too few", HIERARCHY, None,
     "config.program_variance_prior.shape + program_count as f64 / 2.0",
     "config.program_variance_prior.shape + (program_count as f64 - 1.0) / 2.0"),
    ("observation sum of squares 20% low", HIERARCHY, None,
     "config.observation_variance_prior.scale + observation_ss / 2.0",
     "config.observation_variance_prior.scale + 0.8 * observation_ss / 2.0"),
    ("group sum of squares 20% low", HIERARCHY, None,
     "config.group_variance_prior.scale + group_ss / 2.0",
     "config.group_variance_prior.scale + 0.8 * group_ss / 2.0"),
    ("group-mean precision 10% low in its program term", HIERARCHY, None,
     "1.0 / *group_variance + member_count as f64 / *program_variance",
     "1.0 / *group_variance + member_count as f64 / *program_variance * 0.9"),
    ("group variance uses the observation prior's shape", HIERARCHY, None,
     "config.group_variance_prior.shape + group_count as f64 / 2.0",
     "config.observation_variance_prior.shape + group_count as f64 / 2.0"),
    ("population prior mean ignored", HIERARCHY, None,
     "config.population_mean_prior / config.population_variance_prior",
     "0.0 / config.population_variance_prior"),
    ("program's group taken from its position", HIERARCHY, None,
     "group_means[group_index[program]] / *program_variance",
     "group_means[program % group_count] / *program_variance"),
]


BACKUP_SUFFIX = ".unmutated"

# What a failed moment check prints: a missed exact value, too little bulk ESS,
# or a control that could not fail.
CHECK_FAILURES = (" MCSE from the exact ", "bulk ESS", "the checks cannot")


def run(test, environment=None):
    """Run one test; return whether it passed and its output. Raise unless the
    test ran: a build or harness failure is not a caught bug."""
    # --nocapture shows the margin report a passing test prints.
    command = ["cargo", "test", "--release", "-p", "rustmc_core", "--lib", test, "--", "--nocapture"]
    result = subprocess.run(command, capture_output=True, text=True,
                            env=None if environment is None else {**os.environ, **environment})
    output = result.stdout + result.stderr
    if "running 1 test" not in output or "test result:" not in output:
        raise RuntimeError(f"{test} did not run:\n{output[-3000:]}")
    passed = result.returncode == 0 and "test result: ok. 1 passed" in output
    if not passed and not any(marker in output for marker in CHECK_FAILURES):
        raise RuntimeError(f"{test} failed for a reason other than a moment check:\n"
                           f"{output[-3000:]}")
    return passed, output


def leftover_backups():
    """Copies left by a run that was killed mid-mutation."""
    return sorted(Path("rust_core/src").glob("*" + BACKUP_SUFFIX))


def worst_miss(output):
    """The failure furthest from its exact value, in MCSE."""
    misses = [
        (float(line.split(" is ")[1].split(" MCSE")[0]), line.strip().strip('",'))
        for line in output.splitlines()
        if " MCSE from the exact " in line
    ]
    if not misses:
        return next((line.strip() for line in output.splitlines()
                     if any(marker in line for marker in CHECK_FAILURES)), "")
    return f"{len(misses)} checks missed, worst {max(misses)[1][:120]}"


def seed_study(first, last):
    """Rerun each test on other seeds; report failures and the worst margin."""
    failing = 0
    seeds = last - first + 1
    for _, test in (LOCAL_LEVEL, SEASONAL, HIERARCHY):
        failures, margins, tolerance = [], [], None
        for offset in range(first, last + 1):
            environment = {"RUSTMC_EXACT_SEED_OFFSET": str(offset), "RUSTMC_EXACT_REPORT": "1"}
            passed, output = run(test, environment)
            # The first report is the sampler's draws; later ones are controls.
            report = next(line for line in output.splitlines() if line.startswith("worst comparison:"))
            margins.append(float(report.split()[2]))
            tolerance = float(report.split("tolerance ")[1].split()[0])
            if not passed:
                failures.append(offset)
        margins.sort()
        print(f"{test}: {len(failures)} of {seeds} seeds failed {failures}; worst comparison "
              f"median {margins[len(margins) // 2]:.2f}, largest {margins[-1]:.2f} MCSE, "
              f"tolerance {tolerance:.2f}", flush=True)
        failing += bool(failures)
    return 1 if failing else 0


def main():
    leftovers = leftover_backups()
    if leftovers:
        print("an earlier run was interrupted; compare each file with its copy, restore it "
              "with `mv COPY FILE` if it is still mutated, or delete the copy:")
        for backup in leftovers:
            print(f"  {backup}")
        return 1
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--seeds":
        first = int(sys.argv[3]) if len(sys.argv) == 4 else 1
        return seed_study(first, first + int(sys.argv[2]) - 1)
    for _, test in (LOCAL_LEVEL, SEASONAL, HIERARCHY):
        passed, output = run(test)
        if not passed:
            print(f"{test} fails without any bug planted:\n{output[-2000:]}")
            return 1
    escaped = []
    for description, (test_file, test), target, old, new in MUTATIONS:
        path = Path(target or test_file)
        original = path.read_text()
        if original.count(old) != 1:
            print(f"{description}: the text to mutate is not in {path} exactly once")
            return 1
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        backup.write_text(original)
        try:
            path.write_text(original.replace(old, new))
            passed, output = run(test)
        finally:
            path.write_text(original)
            backup.unlink()
        if not passed and " MCSE from the exact " not in output:
            print(f"{description}: failed only on the ESS floor or a control, not a moment:\n"
                  f"{output[-3000:]}")
            return 1
        print(f"{'ESCAPED' if passed else 'caught '}  {description}: {worst_miss(output)}")
        if passed:
            escaped.append(description)
    if escaped:
        print(f"{len(escaped)} of {len(MUTATIONS)} planted bugs passed the tests")
        return 1
    print(f"all {len(MUTATIONS)} planted bugs failed the tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
