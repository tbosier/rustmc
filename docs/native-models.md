# Use a model from Rust

Python and Rust share the `rustmc.graph-model` version 1 format. The definition,
expression compiler, schema validation, and loader live in `rustmc_core::model`.
Python models saved by 0.12 remain readable.

Save a compiled Python model with `Path("model.json").write_text(compiled.to_json())`.
In Rust, use `GraphModel::from_json`, bind `DataInputs`, then call `sample` or
`log_density`. `ModelFit::predict` accepts future inputs without response placeholders.
Its output is indexed by response, chain, draw, and observation. Displayed samples
preserve transformed and noncentered parameter identities.

The runnable `load_model` example accepts four JSON files: the model artifact,
training data, an unconstrained parameter vector, and future predictors. Data files
use the same keyed vectors and row-major matrices as Python dictionaries.

```bash
cargo run -p rustmc_core --release --example load_model -- \
  model.json data.json position.json future.json
```

The example prints a log density, gradient, posterior means, and conditional-mean
predictions. It uses fixed demonstration sampling controls; applications should
supply their own `SamplerConfig` and inspect diagnostics.

Compiled artifacts contain structure, not training data. Fitted Python artifacts
still contain training data and draws. Neither format checkpoints sampler state.
The older `CompiledModelArtifact` format remains a separate legacy API. The Rust
API is alpha; the current-format loader rejects unknown versions and schema changes.
