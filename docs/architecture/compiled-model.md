# Compile once, bind many datasets

This page explains how a `ModelBuilder` model becomes something you can fit repeatedly:
what `compile()` fixes, what `bind()` checks, what each artifact format keeps, and where
the boundary between the Rust core and the Python adapter falls. It is for contributors
and for anyone deciding whether the compile-once model fits their deployment. To *use*
the API, read [custom models](../custom-models.md) instead.

Status: current as of 0.13.

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

The earlier data-owning JSON `CompiledModelArtifact`, and the
`rustmc_core::compiled_model` module that defined it, have been removed. It was never
emitted or accepted by the Python `CompiledModel` API, so nothing that used that API
changes. Python `CompiledModel.to_json/from_json` uses a
versioned declarative format, preserving schema widths/dimensions and omitting
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
and posterior generation.

A second, allocating reference evaluator in `autodiff_reference.rs` cross-checks the
production evaluator. It is compiled only under `#[cfg(test)]`, so it is test
scaffolding rather than public API. It still walks the graph independently, but it is
no longer an independent implementation of the derivatives:
for elementwise operators, Student-t, Bernoulli, and Poisson it deliberately calls the
same `ElementwiseOp::adjoints`, `student_t_derivatives`, `bernoulli_logp_dp`, and
`poisson_logp_dlam` that production uses. The two had drifted apart and the oracle was
the one that was wrong in the tails, so sharing them was the fix. The consequence is
that this comparison checks graph traversal, broadcasting, and adjoint accumulation in
the optimized evaluator — not the derivative formulas themselves, which the two now
agree on by construction.

The formulas are covered separately, in `rust_core/tests/numerical_stability.rs`.
Almost every expectation there is a closed form, a central finite difference, or a
constant computed outside the crate with Python's `decimal` at 400 or 1500 significant
digits. Two are deliberate exceptions, and the file marks both: one compares the
evaluator against `transform.apply` and one compares `mean()` against `diagnostics()`,
and in each case both sides call the same helper. Those two are tripwires against a
second formula being reintroduced, not checks on the formula — which is the same
caveat that applies to the reference evaluator above.

Structural components compile into validated state blocks handled by Gaussian
FFBS/Gibbs (with Student-t latent precision updates when requested). Dynamic GLMs
use an explicitly selected elliptical-slice kernel for their Gaussian latent priors.
The `LogDensity` trait is an independent Rust extension boundary sharing the HMC/NUTS
integrators. These are explicit inference choices; there is no automatic kernel planner.
