# Use a model from Rust

This page covers saving a model built in Python and loading it in a Rust program, so
inference runs without a Python interpreter. You need it if you build models in Python
but deploy into a Rust service; if you work only in Python, skip it.

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

Compiled artifacts contain structure, not training data. Fitted artifacts
(`rustmc.graph-fit`) contain training data and draws, and load on either side: a fit
saved with `fit.to_json()` in Python opens in Rust with `ModelFit::from_json`, which
validates every stored draw against the model, and `ModelFit::to_json` writes a file
Python's `FitResult.from_json` reads. Both write keys in a fixed order, so the same
fit saves to the same bytes. `ModelFit` also provides `log_likelihood`,
`deterministics` and `posterior_predictive`. `GraphModel::sample_batch` fits many
datasets against one model. Neither format checkpoints sampler state.
The older data-owning `CompiledModelArtifact` format and its
`rustmc_core::compiled_model` module have been removed, so `rustmc.graph-model` is
the only compiled-model artifact.

The Rust API is alpha. The loader rejects unknown versions, and every artifact type
it deserializes — `ModelArtifact`, `ModelSpec`, `PriorSpec`, `LikelihoodSpec`,
`MuExpr`, `DataSchema`, `DataSlot`, and `SlotKind` — denies unknown fields. A field
this version does not recognise is an error rather than a silent drop, so an
artifact written by a newer version fails to load instead of loading as a different
model.
