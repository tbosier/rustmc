#!/usr/bin/env python3
"""Plant bugs in the Gibbs samplers and check that their exact-posterior tests fail.

Each mutation replaces one line of sampler code, runs the matching unit test in
release mode, and restores the file, whatever happens. The unmutated tests must
pass first. Exit status 1 means a test passed with a bug planted, or failed
without one. Run from the repository root:

    python3 scripts/plant_gibbs_bugs.py

It takes a few minutes, most of it rebuilding the test binary once per mutation.
"""
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
    ("process sum of squares 15% low", LOCAL_LEVEL, None,
     "                process_sum_sq,\n                rng,",
     "                0.85 * process_sum_sq,\n                rng,"),
    ("seasonal residual uses 90% of the previous effects", SEASONAL, None,
     "let seasonal_residual = pair[1][1] + pair[0][1..].iter().sum::<f64>();",
     "let seasonal_residual = pair[1][1] + 0.9 * pair[0][1..].iter().sum::<f64>();"),
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


def run(test):
    command = ["cargo", "test", "--release", "-p", "rustmc_core", "--lib", test]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if "error[" in output:
        raise RuntimeError(f"the build failed:\n{output[-2000:]}")
    return result.returncode == 0, output


def worst_miss(output):
    """The failure furthest from its exact value, in MCSE."""
    misses = [
        (float(line.split(" is ")[1].split(" MCSE")[0]), line.strip().strip('",'))
        for line in output.splitlines()
        if " MCSE from the exact " in line
    ]
    if not misses:
        return next((line.strip() for line in output.splitlines() if "ESS" in line), "no MCSE line")
    count = len(misses)
    largest = max(misses)[1]
    return f"{count} checks failed, worst {largest[:120]}"


def main():
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
        try:
            path.write_text(original.replace(old, new))
            passed, output = run(test)
        finally:
            path.write_text(original)
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
