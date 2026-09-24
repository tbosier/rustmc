# Releasing rustmc

The checklist for cutting a release: which versions must agree, what to verify before
tagging, and how the two packages are published. It is for maintainers only.

`rustmc_core` on crates.io and `rustmc` on PyPI share one version. The internal
`python_bindings` crate is also named `rustmc`, but it has `publish = false`:
that name on crates.io belongs to an unrelated project.

Before publishing, synchronize the package versions in `rust_core/Cargo.toml`,
`python_bindings/Cargo.toml`, and `pyproject.toml`; the binding's `rustmc_core` path
**and version** dependency; and both workspace package entries in `Cargo.lock`.
Update the changelog and crate README dependency example. Check the release tag:

```bash
python3 scripts/verify_version.py v0.13.0
cargo metadata --locked --offline --no-deps --format-version 1
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --release
cargo package -p rustmc_core --locked
```

The version verifier also runs without a tag on every CI change. It supports
Python 3.9+ without `tomllib` or third-party packages. Its narrow parser reads the
repository's literal version fields; Cargo checks full manifest/lock validity.
Explicit malformed tags and mismatched dependency/lock versions fail validation.

Both registries publish from CI when the matching `vX.Y.Z` tag is pushed; neither is
published by hand. The tag triggers `.github/workflows/ci.yml`, whose release path
requires Rust tests/package verification, version checks (including the tag), and
source/wheel installs across CPython 3.9–3.14. Each release wheel is then built and
installed on a matching native platform; a clean temporary environment runs the
API smoke check and full Python tests against that exact wheel before upload.
The source archive is likewise installed and tested before upload. Pull requests run
these same native artifact checks before merge. Publishing remains restricted to
matching release tags.

Once every check passes, two jobs publish, in this order:

1. `publish-crate` runs `cargo publish -p rustmc_core --locked`, which packages and
   builds the crate once more before uploading it to crates.io.
2. `publish` uploads the verified wheels and source archive to PyPI. It runs only
   after `publish-crate` succeeded, so a Python release never goes out without its
   Rust crate.

Registry releases are immutable, so inspect package contents and test the final
revision before tagging. If `publish` fails after `publish-crate` succeeded, fix the
cause and re-run only the failed job; re-running `publish-crate` fails because the
version already exists on crates.io. See
[Cargo's publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).

Release builds use CPython 3.11 and retain the `cp39-abi3` compatibility tag. Native
Linux ARM and Intel/ARM macOS jobs use standard GitHub-hosted runner labels; see
[GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The earlier Linux matrix separately verifies the advertised Python compatibility
floor. Test commands can also be run locally after building exactly one artifact:

```bash
python scripts/test_release_artifact.py wheel --dist-dir dist
python scripts/test_release_artifact.py sdist --dist-dir dist
```

### One-time repository setup

The workflow names two GitHub environments. A maintainer creates them under
**Settings → Environments** before the first tagged release, since a job that names
a missing environment creates it without protection:

- `crates-io`: add required reviewers, restrict deployments to `v*` tags, and add a
  `CARGO_REGISTRY_TOKEN` environment secret holding a crates.io API token scoped to
  `publish-update` for `rustmc_core` only. `publish-crate` reads nothing else.
- `pypi`: add required reviewers and restrict deployments to `v*` tags. `publish`
  uses PyPI trusted publishing through `pypa/gh-action-pypi-publish`, with
  job-scoped `id-token: write` and no API token. Add `pypi` as the environment in
  the project's trusted-publisher entry on PyPI so uploads are accepted only from
  this job; see
  [PyPI's trusted publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

With required reviewers in place, each publish job waits for approval, so pushing a
tag alone does not publish anything. The third-party actions on the release path
(`PyO3/maturin-action`, `pypa/gh-action-pypi-publish`, and `dtolnay/rust-toolchain`
in the source-archive and crate jobs) are pinned to full commit SHAs; update a pin by
resolving the new tag to its commit, not by editing the version comment alone.

The same tag deploys the documentation site (`.github/workflows/docs.yml`), so the
site describes the version that was just released. Pushes to `main` and pull requests
build the site without deploying it; to redeploy, run that workflow by hand on the
release tag.

A green build is not itself evidence that publishing succeeded: verify both registry
versions and install the published distribution before reporting the release complete.
