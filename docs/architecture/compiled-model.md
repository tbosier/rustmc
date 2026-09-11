# Compile once, bind many datasets

Status: updated for 0.12.

`ModelBuilder.compile()` constructs the parameter and operation graph once and
returns a Python `CompiledModel`. The compiled object owns one `Arc<Graph>` whose
dataset payload vectors are empty; its structural `DataSchema` records required
predictor, response, and matrix keys. Matrix column counts and parameter/group
cardinalities are structural, while the observation row count belongs to each
`DataBinding` and may differ from one fit to the next.

`CompiledModel.bind(data)` eagerly validates missing and unexpected keys, 1-D
versus 2-D kind, non-empty payloads, row counts within each named dimension,
matrix storage and column count, and finite values. Independent dimensions may
have different lengths; indexed group arrays retain fixed parameter cardinality.
It returns `BoundModel`. NumPy inputs are currently
copied into `Arc<[f64]>`; this implementation intentionally makes no zero-copy
claim.

`ModelBuilder`, `CompiledModel`, `BoundModel`, and `BatchFit` all support optional
context-manager syntax. The contract is deliberately nonambient and
non-owning: `__enter__` returns the same object, `__exit__` never suppresses an
exception, and leaving the block does not close or invalidate the object. A
builder block therefore does not install a PyMC-style "current model"; calls
remain explicit on the builder. Likewise, compiled models and bindings can be
reused after a block. The syntax provides lexical scoping only, not compilation,
data binding, cleanup, or thread-local state.

`CompiledModel.sample()` evaluates one binding against the shared structure.
`CompiledModel.sample_batch()` preserves input order and caller-supplied IDs and
uses the same structure for every dataset. It shares the forecasting batch executor,
uses stable ID seeds by default, controls native worker count, and can collect failures.
See [execution options](../forecasting-workflows.md#custom-models-and-independent-batches).
Legacy `sample()` and `batch_sample()` remain available; the latter retains positional
seeds. Generic fits can predict on a new binding without response placeholders.

The pre-existing JSON `CompiledModelArtifact` remains a legacy, data-owning
format and is not emitted or accepted by the new Python `CompiledModel` API.
This explicit boundary prevents a data-owning v1 artifact from being mistaken
for a re-bindable compiled model. Python `CompiledModel.to_json/from_json` uses a
separate versioned declarative format, preserving schema widths/dimensions and omitting
bound training payloads. Loading rebuilds through the validated compiler. Generic fit
artifacts additionally retain training inputs and posterior/telemetry arrays for
prediction; see [artifact semantics](../custom-models.md).

Future work includes streaming input/result retention and borrowed read-only NumPy
buffers with pinned ownership. Chunked execution still retains all submitted payloads
and posterior results; it is not a streaming memory guarantee.

## Native kernel boundaries

The declarative model definition, expression compiler, prediction binder, and artifact
loader live in `rustmc_core::model`. Python provides construction and input adapters.
See [standalone Rust execution](../native-models.md). The core evaluator owns per-node lengths so that
named dimensions affect computation. One native observation simulator supplies prior
and posterior generation. Reference autodiff and the older data-owning artifact remain
isolated for validation and compatibility.

Structural components compile into validated state blocks handled by Gaussian
FFBS/Gibbs (with Student-t latent precision updates when requested). Dynamic GLMs
use an explicitly selected elliptical-slice kernel for their Gaussian latent priors.
The `LogDensity` trait is an independent Rust extension boundary sharing the HMC/NUTS
integrators. These are explicit inference choices; there is no automatic kernel planner.
