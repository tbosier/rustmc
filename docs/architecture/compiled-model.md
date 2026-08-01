# Compile once, bind many datasets

Status: implemented foundational slice in 0.8 development.

`ModelBuilder.compile()` constructs the parameter and operation graph once and
returns a Python `CompiledModel`. The compiled object owns one `Arc<Graph>` whose
dataset payload vectors are empty; its structural `DataSchema` records required
predictor, response, and matrix keys. Matrix column counts and parameter/group
cardinalities are structural, while the observation row count belongs to each
`DataBinding` and may differ from one fit to the next.

`CompiledModel.bind(data)` eagerly validates missing and unexpected keys, 1-D
versus 2-D kind, non-empty payloads, common row count, matrix storage and column
count, and finite values. It returns `BoundModel`. NumPy inputs are currently
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
uses the same structure for every dataset. Legacy `sample()` and `batch_sample()`
remain available and retain their prior behavior.

The pre-existing JSON `CompiledModelArtifact` remains a legacy, data-owning
format and is not emitted or accepted by the new Python `CompiledModel` API.
This explicit boundary prevents a data-owning v1 artifact from being mistaken
for a re-bindable compiled model; a future slot-only v2 format must use a version
bump and reject v1 rather than silently migrating embedded data.

Future work includes streaming partial-failure batches, borrowed read-only NumPy
buffers with pinned ownership, and slot-only artifact serialization. The current
slice focuses on a safe input contract, structure reuse, variable row counts,
stable batch ordering, and backward compatibility.
