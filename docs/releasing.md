# Releasing rustmc

`rustmc_core` on crates.io and `rustmc` on PyPI share one version. The internal
`python_bindings` crate is also named `rustmc`, but it has `publish = false`:
that name on crates.io belongs to an unrelated project.

Before publishing, synchronize the package versions in `rust_core/Cargo.toml`,
`python_bindings/Cargo.toml`, and `pyproject.toml`; the binding's `rustmc_core` path
**and version** dependency; and both workspace package entries in `Cargo.lock`.
Update the changelog and crate README dependency example. Check the release tag:

```bash
python3 scripts/verify_version.py v0.11.0
cargo metadata --locked --offline --no-deps --format-version 1
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cargo package -p rustmc_core --locked
```

The version verifier also runs without a tag on every CI change. It supports
Python 3.9+ without `tomllib` or third-party packages. Its narrow parser reads the
repository's literal version fields; Cargo checks full manifest/lock validity.
Explicit malformed tags and mismatched dependency/lock versions fail validation.

After the final revision passes CI and publishing is authorized, publish only the
core Rust crate from that revision:

```bash
cargo publish -p rustmc_core --locked
```

Cargo packages and verifies the crate before uploading. Registry releases are
immutable, so inspect package contents and test the final revision first. See
[Cargo's publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).
Wait until the new core version is visible in the public registry index before
announcing the synchronized release.

For PyPI, pushing the matching `vX.Y.Z` tag triggers `.github/workflows/ci.yml`.
The release path requires Rust tests/package verification, version checks, and
source/wheel installs across CPython 3.9–3.13. Each release wheel is then built and
installed on a matching native platform; a clean temporary environment runs the
API smoke check and full Python tests against that exact wheel before upload.
The source archive is likewise installed and tested before upload. Network-marked
packaging tests remain deselected; these jobs perform archive installation directly.

Release builds use CPython 3.11 and retain the `cp39-abi3` compatibility tag. Native
Linux ARM and Intel/ARM macOS jobs use standard GitHub-hosted runner labels; see
[GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The earlier Linux matrix separately verifies the advertised Python compatibility
floor. Test commands can also be run locally after building exactly one artifact:

```bash
python scripts/test_release_artifact.py wheel --dist-dir dist
python scripts/test_release_artifact.py sdist --dist-dir dist
```

The existing `publish` job downloads these verified artifacts and uses PyPI trusted
publishing through `pypa/gh-action-pypi-publish`, with job-scoped `id-token: write`.
No explicit API token or new GitHub environment is required by this workflow. Keep
the configured workflow identity intact; see
[PyPI's trusted publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
A green build is not itself evidence that publishing succeeded: verify both registry
versions and install the published distribution before reporting the release complete.
