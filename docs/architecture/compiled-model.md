# Compile Once, Bind Many Datasets

Design for RustMC Task 5. **Status: proposed, awaiting review.** No implementation
has been done; this document is the contract that Tasks 6–10 and 14 build against.

Target workflow:

```python
compiled = model.compile()
fits = compiled.sample_batch([
    {"x": x_1, "y": y_1},
    {"x": x_2, "y": y_2},
    {"x": x_3, "y": y_3},
])
```

---

## 1. The problem, verified

### 1.1 Findings confirmed against the code

All three findings handed to this design hold. Verified at
`8687933` (v0.8.0):

1. **`batch_sample` rebuilds per dataset.** `python_bindings/src/lib.rs:2274`
   calls `compile_python_model(&spec, &data_map, &matrix_map)` inside the
   per-dataset loop, and `python_bindings/src/lib.rs:2300-2303` then does
   `compiled.graph.clone()` for every model. Every dataset therefore pays a full
   graph construction *plus* a deep clone. Upstream of both, `parse_data_dict`
   copies every NumPy array into a fresh `Vec<f64>`, and
   `spec.bound_data_1d.clone()` copies the builder-bound data again — so a
   dataset's bytes are materialised **three times** before sampling starts, and
   all copies stay resident because `sampler::batch_sample` takes an owned
   `Vec<(Graph, Vec<f64>)>`.

2. **`Graph` entangles structure and data.** `rust_core/src/graph.rs:223-234`:

   ```rust
   pub struct Graph {
       pub nodes: Vec<Node>,
       pub param_count: usize,
       pub data_vectors: Vec<Vec<f64>>,     // <- data
       pub obs_vectors: Vec<Vec<f64>>,      // <- data
       pub data_matrices: Vec<MatrixData>,  // <- data
       pub param_names: Vec<String>,
       pub param_transforms: Vec<ParamTransform>,
       pub param_spans: Vec<ParamSpan>,
       pub logp_terms: Vec<NodeId>,
       name_to_node: HashMap<String, NodeId>,
   }
   ```

   The doc comment above it ("Data vectors and observed values are stored
   separately from the graph structure so the graph itself stays lightweight and
   shareable across threads") describes an intent the type does not implement.
   Everything else in the struct is dataset-independent.

3. **`compiled_model.rs` is not usable as-is.** `CompiledModelArtifact` is not
   exposed to Python — `grep -n '#\[pyclass\]' python_bindings/src/lib.rs`
   returns `ModelSpec`, `VectorParamRef`, `ModelBuilder`, `ParamRef`, `Expr`,
   `FitResult`, `BatchResult` and nothing else. And
   `CompiledModelRuntime::sample(&self, config: SamplerConfig)`
   (`rust_core/src/compiled_model.rs:703`) takes no data argument; internally it
   calls `self.to_graph()`, which replays a graph *with the data that was baked
   into the artifact at `from_graph` time*. See §3 for the full verdict.

### 1.2 One crucial structural fact

**`Op` already references data by index, never by value.** `Op::Data(usize)`,
`Op::ObsLogP { obs_data_idx }`, `Op::FusedLinearMu { data_indices }`,
`Op::MatVecMul { matrix_idx }`. Nothing in the op payloads holds `f64` data.

This means the separation is *mechanically* a move of three fields plus a
rename of ~15 field accesses. It does **not** require touching the `Op` enum,
which is exactly the file three other worktrees are editing. The migration is
low-conflict by construction, and the only forward-compatibility rule new op
variants must respect is: **reference data by slot index, never inline it.**

### 1.3 Measurements — what the rebuild actually costs

Run with the harness in `docs/architecture/scratch/rebind_cost_bench.rs`
(release build, this machine). "build" = `Graph` construction mirroring
`compile_python_model`; "clone" = the extra `compiled.graph.clone()` that
`batch_sample` performs.

| shape | build/dataset | clone/dataset | one NUTS fit | build+clone as % of fit |
|---|---|---|---|---|
| n=2000, k=8, 200+200 draws | 67.6 µs | 66.7 µs | 64.7 ms | 0.21 % |
| n=60, k=8, 200+200 draws | 3.2 µs | 3.2 µs | 2.85 ms | 0.22 % |
| n=60, k=200, 50+50 draws | 61.9 µs | 54.5 µs | 155.7 ms | 0.07 % |

`Evaluator::new` is 0.15–0.9 µs and is not a factor.

**Honest conclusion: rebuild-per-dataset is not a meaningful CPU cost for full
NUTS fits.** Any design pitch that leads with "10,000 graph constructions are
slow" is wrong, and the benchmark plan in §12 must not claim a win that isn't
there. The three costs that *are* real:

- **Forward-pass rebinding (prediction) is dominated by rebuild.** One
  `Evaluator::compute` on the n=60, k=200 model is **8.7 µs**; rebuilding and
  cloning the graph to get there is **116 µs — 13× the work being done.** Task 14
  (prediction on new data) and `sample_prior_predictive` are pure
  forward-pass workloads. Under the current architecture they are ~93 % overhead.
  This is the sharpest quantitative argument for the split.

- **Memory residency is a hard wall.** 10,000 datasets at n=2000, k=8 is 1.4 GiB
  of payload. Today that is held ~3× (Python NumPy + `data_map` + `Graph`,
  then ×2 again at the `graph.clone()` step) — ≈4 GiB resident before a single
  draw is taken, and it must *all* be resident, because `sampler::batch_sample`
  consumes an owned `Vec` of fully-built graphs. Streaming is impossible.

- **Shared covariates are duplicated N times.** The headline use case —
  "one model, thousands of related datasets" — very often means *the same design
  matrix, different response series* (per-store demand, per-SKU forecast,
  per-subject trial). Today X is copied once per dataset: 10,000 × 0.14 MiB =
  1.4 GiB where 0.14 MiB would do. Under the design in §4 this becomes a single
  `Arc` — a 10,000× reduction, and it is *only* expressible once data lives
  outside the structure.

So the justification for this work is **capability and memory, not fit
throughput**. Stated plainly here so the reviewer can hold the benchmark plan to
it.

---

## 2. The central decision: what separates from what

Three types replace today's one.

```
                 built once, immutable, Arc-shared across all datasets
   ┌──────────────────────────────────────────────────────────────┐
   │ CompiledModel                                                │
   │   structure : Graph      (nodes, ops, logp_terms)            │
   │   params    : ParamTable (names, transforms, spans)          │
   │   schema    : DataSchema (slot i <-> user-facing key + kind)  │
   └──────────────────────────────────────────────────────────────┘
                                  │ .bind(dict) -> validated
                                  v
   ┌──────────────────────────────────────────────────────────────┐
   │ DataBinding             one per dataset, cheap, Arc-holding   │
   │   vectors      : Vec<Arc<[f64]>>      (slot-indexed)          │
   │   observations : Vec<Arc<[f64]>>      (slot-indexed)          │
   │   matrices     : Vec<Arc<MatrixData>> (slot-indexed)          │
   │   n_obs        : usize                                        │
   └──────────────────────────────────────────────────────────────┘
                                  │ (&CompiledModel, &DataBinding)
                                  v
   ┌──────────────────────────────────────────────────────────────┐
   │ Evaluator      per worker thread, reused across datasets      │
   │   buffers sized by max n_obs seen; rebind() grows, never      │
   │   shrinks; zero allocation in the hot loop                    │
   └──────────────────────────────────────────────────────────────┘
```

**The boundary rule, stated once:** *the compiled model holds everything that is
the same for every dataset; the binding holds everything that is not; the
`Op` payload is the join key.* Slot index `i` in `Op::Data(i)` is the same `i`
that indexes `DataBinding::vectors`. `DataSchema` is the only thing that knows
the user-facing name of slot `i`.

Consequences that fall out for free:

- `Graph` becomes genuinely `Arc`-shareable — the doc comment at
  `graph.rs:217-221` becomes true.
- Prediction is a rebind (§9.6), not a recompile.
- A shared design matrix is one `Arc` cloned N times (8 bytes each).
- The batch runner can *stream*: bindings can be constructed lazily per
  work item instead of all up front.

### 2.1 What `Graph` becomes

Keep the type name `Graph`, delete three fields, add one. A rename to
`ModelStructure` is a large mechanical diff across four files that three other
worktrees are actively editing; it buys nothing this quarter. Do it later if at
all.

```rust
// rust_core/src/graph.rs
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub param_count: usize,
    pub param_names: Vec<String>,
    pub param_transforms: Vec<ParamTransform>,
    pub param_spans: Vec<ParamSpan>,
    pub logp_terms: Vec<NodeId>,
    /// Declares slot i for every data/obs/matrix index referenced by an Op.
    pub schema: DataSchema,
    name_to_node: HashMap<String, NodeId>,
}
```

`add_data`, `add_obs_data`, `store_data_vec`, `store_matrix` change from
"push values, return index" to "declare a slot, return index":

```rust
impl Graph {
    /// Declare a 1-D data slot bound to `key`. Idempotent: a repeat `key`
    /// returns the existing slot (this de-duplicates the FusedLinearMu case
    /// where the same column feeds two terms).
    pub fn declare_data(&mut self, key: &str) -> usize;
    pub fn declare_obs(&mut self, key: &str, likelihood: &str) -> usize;
    /// `n_cols` is structural (it fixes the parameter-vector length);
    /// `n_rows` is per-dataset and lives in the binding.
    pub fn declare_matrix(&mut self, key: &str, n_cols: usize) -> usize;

    /// Node-creating wrapper, unchanged in spirit.
    pub fn add_data_node(&mut self, key: &str) -> NodeId; // Op::Data(declare_data(key))
}
```

`Graph::validate_shapes()` is **deleted**. Shape validation is a bind-time
concern (§5) — a structure alone has no shapes to validate. `Graph::schema` gains
`fn structural_check(&self) -> Result<(), SchemaError>` for the things that *are*
structural (every `Op` slot index is declared; every declared slot is reachable;
matrix `n_cols` matches the vector-param span length).

`Graph::observation_heads()` currently reads `self.obs_vectors[i].len()` for
`n_obs`. `ObservationHead::n_obs` moves off the head; callers that need it read
`binding.n_obs`. `observation_heads()` otherwise unchanged.

### 2.2 The new types

```rust
// rust_core/src/data.rs  (new module)
use std::sync::Arc;

/// Row-major dense matrix. n_rows is a property of the *binding*, not the
/// structure; n_cols is fixed by the structure and re-checked at bind time.
#[derive(Debug, Clone)]
pub struct MatrixData {
    pub data: Arc<[f64]>,
    pub n_rows: usize,
    pub n_cols: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotKind {
    /// 1-D predictor of length n_obs.
    Vector,
    /// 1-D observed response of length n_obs, owned by a named likelihood.
    Observation { likelihood: String },
    /// 2-D design matrix, n_obs rows x `n_cols` columns.
    Matrix { n_cols: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataSlot {
    /// User-facing key, e.g. "x" or "y". Unique within its kind.
    pub key: String,
    pub kind: SlotKind,
    /// Names the dimension this slot is indexed along. Always "obs" today;
    /// Task 6 populates it with a real coordinate name.
    pub dim: String,
}

/// The declared input surface of a compiled model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataSchema {
    pub vectors: Vec<DataSlot>,      // index == Op::Data(i) / data_indices[k]
    pub observations: Vec<DataSlot>, // index == obs_data_idx
    pub matrices: Vec<DataSlot>,     // index == matrix_idx
}

impl DataSchema {
    /// Every user-facing key that must be present in a bind() dict.
    pub fn required_keys(&self) -> Vec<&str>;
    pub fn slot_of(&self, key: &str) -> Option<(SlotKind, usize)>;
}

/// One dataset, validated against a DataSchema.
#[derive(Debug, Clone)]
pub struct DataBinding {
    pub vectors: Vec<Arc<[f64]>>,
    pub observations: Vec<Arc<[f64]>>,
    pub matrices: Vec<MatrixData>,
    /// The single length every vector/observation has and every matrix has rows of.
    pub n_obs: usize,
    /// Stable identity, preserved through results. See §7.
    pub id: DatasetId,
}
```

`DataBinding` is *cheap*: for a 10-slot model it is 10 `Arc` clones (8 bytes
each) plus three `Vec` headers. Binding 10,000 datasets that share their design
matrix costs 10,000 × ~200 bytes of `DataBinding`, not 10,000 × 0.14 MiB.

---

## 3. Verdict on the existing `compiled_model.rs` artifact layer

**Rework in place. Do not discard; do not keep as-is.** Roughly 70 % survives.

**What is right and must be kept:**

- The *shape* of the artifact: a flat `Vec<ModelStep>` in topological order,
  with `NodeRef::Param(String) | NodeRef::Node(usize)`. Name-based parameter
  refs and ordinal node refs are exactly the right choice — they survive graph
  renumbering and they make `validate()` a single forward scan.
- `ParameterBlock::{Scalar, Vector}` and `group_parameter_blocks` — this is the
  named-parameter-block concept Task 6 needs, arrived at early.
- `validate()` as a total, pre-`build_graph` check with a real error enum
  (`ArtifactError`), and explicit `format_version` gating.
- `build_graph` as a replay function. It stays, minus the data.

**What is wrong and must change — every one of these is a data/structure
entanglement:**

| current | problem | fix |
|---|---|---|
| `ModelStep::Data { name, values: Vec<f64> }` | inlines a dataset into the model | `ModelStep::Data { slot: usize }` |
| `ModelStep::Observation { obs: Vec<f64> }` | inlines the response | `Observation { obs_slot: usize }` |
| `ModelStep::MatVecMul { matrix: Vec<f64>, n_rows, n_cols }` | inlines X **and bakes `n_rows`** — an outright bug for rebinding, since n_rows is per-dataset | `MatVecMul { matrix_slot: usize, n_cols: usize }` |
| `ModelStep::FusedLinearMu { term_data: Vec<Vec<f64>> }` | inlines every predictor column | `FusedLinearMu { term_slots: Vec<usize> }` |
| `CompiledModelRuntime::sample(&self, config)` | no data argument; can only refit the baked data | `sample(&self, data: &DataBinding, config)` |
| `CompiledModelRuntime::to_graph()` called per `sample()` | replays the whole graph on every fit | build once in `from_artifact`, store `Arc<Graph>` |
| `validate_step`'s `obs.is_empty()` check | validates data, in a structure validator | moves to `bind()` |

Plus: add `pub schema: DataSchema` to `CompiledModelArtifact`, and bump
`ARTIFACT_FORMAT_VERSION` 1 → 2. Version 1 artifacts are rejected with the
existing `VersionMismatch` error; **no v1→v2 migration is written**, because v1
artifacts contain data and there is nowhere sensible to put it, and — critically
— **v1 has zero users**: nothing in `python_bindings` or `rust_core/tests`
constructs one outside `compiled_model.rs`'s own three unit tests. This makes the
rework cheap and safe, which is the other half of the verdict.

Net: ~300 lines of mechanical edits inside one file that nobody else is
currently touching. Cheaper than a greenfield artifact and it preserves a
serialization design that is already correct.

**Naming.** `CompiledModelRuntime` becomes the Python-visible `CompiledModel`:

```rust
#[derive(Debug, Clone)]
pub struct CompiledModel {
    artifact: CompiledModelArtifact,
    /// Replayed once at construction, shared by every fit.
    structure: Arc<Graph>,
}

impl CompiledModel {
    pub fn from_graph(graph: &Graph) -> Result<Self, ArtifactError>;
    pub fn from_artifact(a: CompiledModelArtifact) -> Result<Self, ArtifactError>;
    pub fn artifact(&self) -> &CompiledModelArtifact;
    pub fn structure(&self) -> &Arc<Graph>;
    pub fn schema(&self) -> &DataSchema;
    pub fn param_names(&self) -> &[String];

    pub fn bind(&self, inputs: DataInputs) -> Result<DataBinding, BindError>;
    pub fn sample(&self, data: &DataBinding, cfg: SamplerConfig) -> Result<SampleResult, SamplerError>;
    pub fn sample_batch(&self, data: Vec<DataBinding>, cfg: BatchConfig) -> BatchFit;

    pub fn to_json_pretty(&self) -> Result<String, ArtifactError>;
    pub fn from_json_str(s: &str) -> Result<Self, ArtifactError>;
}
```

Keep `pub type CompiledModelRuntime = CompiledModel;` for one release.

---

## 4. Graph reuse across datasets, and the Evaluator hot path

This is the hardest part of the design. Addressed directly.

### 4.1 What is shared vs. per-dataset

| | lives in | lifetime | cost per dataset |
|---|---|---|---|
| `nodes`, `logp_terms`, `Op` payloads | `Arc<Graph>` | whole batch | 8 bytes (Arc clone) |
| `param_names`, `param_transforms`, `param_spans` | `Arc<Graph>` | whole batch | 0 |
| `DataSchema` | `Arc<Graph>` | whole batch | 0 |
| predictor / obs / matrix bytes | `DataBinding` (`Arc<[f64]>`) | per dataset, or shared | 8 bytes/slot if shared, else the payload |
| `n_obs` | `DataBinding` | per dataset | 8 bytes |
| `vec_buf`, `adj_vec_buf`, `scalars`, `adj_scalars`, `grad` | `Evaluator` | **per worker thread**, reused | amortised to 0 |
| step size, mass matrix, RNG, sample store | chain-local | per chain | unchanged |

The `Evaluator` is the one that needs care, because its buffer sizes depend on
`n_obs`, which is now per-dataset.

### 4.2 How the Evaluator works when data leaves the Graph

Today `Evaluator::compute(&mut self, graph: &Graph, params: &[f64])` reads data
off the graph in exactly three places per pass:

- `graph.data_vectors[di][i]` — `read_vec`, `autodiff.rs:99`
- `graph.obs_vectors[*obs_data_idx]` — `autodiff.rs:232` (fwd), `:623` (bwd)
- `graph.data_matrices[*matrix_idx]` — `autodiff.rs:333` (fwd), `:766` (bwd)
- `graph.data_vectors[data_indices[k]]` — `FusedLinearMu`, `:321` / `:745`

**The change is a signature change and a receiver rename, nothing more:**

```rust
pub fn compute(&mut self, g: &Graph, d: &DataBinding, params: &[f64]);
//                        ^ was the sole source of both              ^

#[inline(always)]
fn read_vec(&self, node_id: usize, i: usize, d: &DataBinding) -> f64 {
    match self.node_kind[node_id] {
        NodeKind::DataRef(di)      => d.vectors[di][i],   // was g.data_vectors[di][i]
        NodeKind::ComputedVec(off) => self.vec_buf[off + i],
        NodeKind::Scalar           => unreachable!(),
    }
}
```

There is no indirection cost. `Arc<[f64]>` derefs to `&[f64]` with one pointer
load — the same load pattern as `Vec<Vec<f64>>` indexing, which also costs one
pointer load. The hot loop's inner arithmetic is untouched. **No allocation is
introduced anywhere in `compute`.** The zero-alloc property is preserved
verbatim.

### 4.3 Buffer sizing: `for_structure` + `rebind`

`vec_len` currently comes from `Graph::validate_shapes()`, whose fallback chain
(`data_vectors → obs_vectors → data_matrices`, `autodiff.rs:47`) is a symptom of
the entanglement — it is guessing the dataset length off whichever data field
happens to be populated. Under the split it is not a guess: **`vec_len ==
binding.n_obs`, full stop.** That fallback chain is deleted.

The `Evaluator` splits construction into an n-independent part and an
n-dependent part:

```rust
impl Evaluator {
    /// n-independent: node_kind classification, scalar/adjoint arrays,
    /// grad, param_node_ids, vec_slot_count. Depends only on the structure.
    pub fn for_structure(g: &Graph) -> Self;

    /// Point this evaluator at a dataset. Grows vec_buf/adj_vec_buf if
    /// `d.n_obs` exceeds current capacity; never shrinks. O(1) when the
    /// buffers already fit, which after the first few datasets is always.
    pub fn rebind(&mut self, d: &DataBinding) -> Result<(), ShapeError> {
        let need = self.vec_slot_count * d.n_obs;
        if need > self.vec_buf.len() {
            self.vec_buf.resize(need, 0.0);
            self.adj_vec_buf.resize(need, 0.0);
        }
        self.vec_len = d.n_obs;
        Ok(())
    }

    pub fn compute(&mut self, g: &Graph, d: &DataBinding, params: &[f64]);
}
```

Two rules the implementer must not get wrong:

1. **`vec_buf` is addressed as `slot * self.vec_len + i`, and `vec_len` changes
   on rebind.** Slot offsets can therefore no longer be precomputed into
   `NodeKind::ComputedVec(byte_offset)` as they are today
   (`autodiff.rs:61-63`). Change `NodeKind::ComputedVec(usize)` to hold the
   **slot ordinal**, and compute `slot * vec_len + i` at access time. That is one
   extra `imul` per vector element access. Measure it (§12, bench B4); if it
   regresses single-fit throughput by more than 2 %, fall back to recomputing
   the offset table inside `rebind` (an O(n_nodes) pass, ~ns, still amortised
   away). *Prefer the offset table if in doubt — it is strictly safer for
   throughput and `rebind` is not hot.*

2. **Buffers grow but never shrink.** A worker that sees one 10⁶-row dataset
   holds that buffer for the rest of the batch. Bound it: if
   `need < self.vec_buf.len() / 4`, shrink to `need`. This caps waste at 4× and
   costs one realloc on a large drop.

### 4.4 Threading model for `sample_batch`

```
rayon par_iter over datasets
  └─ thread-local Evaluator (rayon `map_init`), reused across datasets
       for chain in 0..chains:
          evaluator.rebind(&binding)
          nuts::run_chain(&structure, &binding, &mut evaluator, ...)
```

`nuts::run_chain` and `hmc::run_chain` gain a `data: &DataBinding` parameter and
take `evaluator: &mut Evaluator` instead of constructing one internally
(`nuts.rs:98`, `hmc.rs:71`). Everything downstream (`nuts.rs:287/383/492/553`,
`hmc.rs:249/326`) already threads `graph: &Graph, evaluator: &mut Evaluator`
together; add `data: &DataBinding` alongside. Purely additive.

`mass_matrix.rs`'s `from_graph(graph)` / `identity(graph)` only read
`param_count`; they are unaffected.

`sampler::batch_sample` changes signature from
`Vec<(Graph, Vec<f64>)>` to `(&Arc<Graph>, Vec<DataBinding>)` — one structure,
many bindings. The `let _ = obs_y; // data is already baked into the graph` line
at `sampler.rs:358` (a vestigial second data channel) is deleted.

---

## 5. Data binding lifecycle

```rust
/// Whatever the caller hands us, before validation.
pub struct DataInputs {
    pub vectors: HashMap<String, Arc<[f64]>>,
    pub matrices: HashMap<String, (Arc<[f64]>, usize, usize)>, // data, n_rows, n_cols
}

#[derive(Debug, Clone, PartialEq)]
pub enum BindError {
    MissingKey { key: String, kind: SlotKind },
    UnexpectedKey { key: String, did_you_mean: Option<String> },
    WrongKind { key: String, expected: SlotKind, found: SlotKind },
    LengthMismatch { key: String, len: usize, expected: usize, expected_from: String },
    MatrixColsMismatch { key: String, n_cols: usize, expected: usize, param_block: String },
    RaggedMatrix { key: String, n_rows: usize, n_cols: usize, values: usize },
    NonFinite { key: String, index: usize, value: f64 },
    Empty { key: String },
}
```

`bind` is **total and eager**: it either returns a `DataBinding` that the
sampler is guaranteed to be able to run, or it returns the *first* error in a
deterministic order. Ordering is fixed so error messages are reproducible:

1. missing required keys (schema order: observations, then vectors, then matrices)
2. unexpected keys (sorted; `did_you_mean` = Levenshtein ≤ 2 against required keys)
3. kind mismatches (1-D given where 2-D expected, etc.)
4. emptiness
5. length agreement — the first slot bound fixes `n_obs` and its key is
   reported as `expected_from` in every subsequent mismatch, so the message
   reads *"'x' has length 200, expected 150 (from 'y')"*
6. matrix `n_cols` vs. the declared vector-param span
7. non-finite scan (NaN/±inf) — **on by default**, `O(n)`, ~0.2 ns/element;
   at 10,000 × 16,000 elements that is ~30 ms for the whole batch, which is
   noise against 10,000 fits. Opt out with `check_finite=False`.

Bind-time cost with shared inputs is ~10 pointer copies. Binding is therefore
cheap enough to do lazily per work item, which is what makes streaming (§13,
R4) possible later.

**Extra keys are an error, not a warning.** A typo'd key that is silently
ignored produces a model fit to the wrong data, which is the worst possible
failure mode for this product. `bind(..., strict=False)` exists for the
notebook workflow where one dict feeds several models.

---

## 6. Per-dataset variation

### 6.1 Parameter initialization

```rust
pub enum InitStrategy {
    /// Stan-compatible: U(-scale, scale) on the unconstrained scale. Default scale = 2.0.
    Jitter { scale: f64 },
    /// Same unconstrained start for every dataset and chain.
    Fixed(Vec<f64>),
    /// Per-dataset start, indexed by position in the batch. Length must equal
    /// the batch length. This is the warm-start / rolling-forecast path:
    /// feed in the previous period's posterior means.
    PerDataset(Vec<Vec<f64>>),
    /// Computed from the bound data — e.g. an OLS start for a GLM.
    Derived(Arc<dyn Fn(&Graph, &DataBinding) -> Vec<f64> + Send + Sync>),
}
```

Default is `Jitter { scale: 2.0 }`. `Fixed` and `PerDataset` values are in
**unconstrained space** (matching what `run_chain` consumes and what
`ParamTransform::apply` inverts); this must be documented loudly, and the Python
layer should accept constrained values and invert them, since users think in
constrained space. Add `ParamTransform::unapply(&self, constrained) -> f64` for
this — it does not exist today and is needed for warm starts regardless.

### 6.2 Deterministic seeding

Current derivation (`sampler.rs:367-370`) is
`seed + (model_idx << 32) + chain_idx`. **It is positional**: insert one dataset
at the front and every downstream fit changes. For a workflow whose whole point
is "re-fit 10,000 series nightly, some added, some dropped", that is a
reproducibility bug.

Replace with an identity-derived seed:

```rust
/// Stable across batch composition, ordering, and thread count.
/// Versioned: bump SEED_SCHEME if the derivation ever changes.
const SEED_SCHEME: u64 = 1;

pub fn dataset_seed(base: u64, id: &DatasetId, chain: usize) -> u64 {
    // FNV-1a over the tuple, then a SplitMix64 finaliser for avalanche.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in base.to_le_bytes().iter()
        .chain(SEED_SCHEME.to_le_bytes().iter())
        .chain(id.as_bytes())
        .chain((chain as u64).to_le_bytes().iter())
    {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut z = h.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
```

No new dependency. Property to test: same `(base, id, chain)` ⇒ identical draws,
regardless of batch size, batch order, `threads=`, or whether the dataset was
fit alone or in a batch of 10,000.

---

## 7. Result identity and ordering

```rust
/// A stable, user-meaningful dataset name. Newtype so it cannot be confused
/// with an index. Defaults to the decimal position when the caller gives none.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DatasetId(pub String);
```

Rules:

- **`BatchFit.outcomes` is in input order. Always.** `rayon`'s
  `par_iter().enumerate().map(...).collect()` already guarantees this
  (`sampler.rs:354-418`); the design keeps it and it becomes a tested invariant,
  not an accident.
- Every `DatasetFit` and `DatasetError` carries both `index: usize` and
  `id: DatasetId`.
- IDs come from, in precedence order: an explicit `ids=[...]` argument, a
  reserved `"__id__"` string entry in the dataset dict, else `str(index)`.
- Duplicate IDs are an error at batch construction. Silent collision would make
  `fits["store_42"]` non-deterministic.
- `BatchFit` carries `by_id: HashMap<DatasetId, usize>` for O(1) lookup.

---

## 8. Partial failure

One bad dataset must not kill 9,999 good ones.

```rust
pub struct BatchFit {
    pub outcomes: Vec<DatasetOutcome>,   // input order, one per input dataset
    by_id: HashMap<DatasetId, usize>,
    pub param_names: Arc<Vec<String>>,   // shared, not cloned per dataset
    pub config: BatchConfig,
}

pub enum DatasetOutcome {
    Ok(Box<DatasetFit>),
    Err(DatasetError),
}

pub struct DatasetFit {
    pub id: DatasetId,
    pub index: usize,
    pub seed: u64,
    /// [chain][draw][param] — chain structure preserved, see §9.5.
    pub samples: Vec<Vec<Vec<f64>>>,
    pub accept_rates: Vec<f64>,
    pub step_sizes: Vec<f64>,
    pub divergences: Vec<usize>,
    pub transitions: Vec<Vec<TransitionStats>>,
    pub n_obs: usize,
}

pub struct DatasetError {
    pub id: DatasetId,
    pub index: usize,
    pub kind: DatasetErrorKind,
    pub message: String,
}

pub enum DatasetErrorKind {
    /// bind() rejected the inputs. Detected before any sampling.
    Bind(BindError),
    /// Initial point produced non-finite logp after N retries.
    Initialization,
    /// logp or gradient went non-finite mid-chain in a way the sampler
    /// could not recover from.
    NonFinite,
    /// Structured sampler failure.
    Sampler(SamplerError),
    /// A panic was caught. Should never happen; recorded rather than
    /// propagated so one bug does not lose a 6-hour batch.
    Panic(String),
}
```

**Divergences are not errors.** A fit with 400 divergences is `Ok` with
`divergences: [400]`. Only hard failures produce `Err`. This distinction must be
in the docstring; conflating them would train users to ignore `errors()`.

Each dataset's work is wrapped in `std::panic::catch_unwind(AssertUnwindSafe(..))`
inside the rayon closure. Rayon propagates panics out of `par_iter` by default,
which would abort the batch — catching per item is the whole mechanism.

Python surface:

```python
fits.ok            # list[Fit]           — successes, input order
fits.errors        # list[DatasetError]  — failures, input order
fits[3]            # Fit, or raises DatasetFailed if index 3 failed
fits.get(3)        # Fit | None
len(fits)          # == number of input datasets, successes + failures
fits.n_failed
```

`sample_batch(..., on_error="collect")` (default) or `on_error="raise"`
(raise on the first failure — useful in tests and CI).

---

## 9. Forward compatibility

Each downstream task gets a named extension point. Guesses are flagged.

### 9.1 Task 6 — named dimensions and coordinates

`DataSlot.dim: String` exists from day one, hardcoded to `"obs"`. Task 6 makes
it real and adds to `DataSchema`:

```rust
pub struct DimDecl { pub name: String, pub size: DimSize }
pub enum DimSize { FixedByStructure(usize), FromBinding }  // "obs" is FromBinding
pub struct DataSchema { /* ... */ pub dims: Vec<DimDecl> }
```

`DataBinding.n_obs: usize` generalises to
`dim_sizes: HashMap<String, usize>` with `n_obs()` kept as an accessor for the
`"obs"` dim. Coordinate *labels* (`coords: HashMap<String, Vec<String>>`) attach
to `DataBinding`, not the structure — labels are per-dataset (store 1 has
different SKUs than store 2). **Assumption: dimension *names and count* are
structural, dimension *sizes and labels* are per-binding.** If Task 6 needs
per-dataset dimension *names*, this needs revisiting; flag as OQ-3.

### 9.2 Task 7 — richer expression graph

No change required. New `Op` variants are added to the same enum with the same
one rule: **reference data by slot index, never inline values.** The
structure/data boundary is orthogonal to expression richness. The one thing
Task 7 must also do is add a matching `ModelStep` variant and `validate_step`
arm, exactly as today.

### 9.3 Task 8 — group indexing for hierarchical effects

Group indices are per-dataset integer data. Add a third slot kind:

```rust
SlotKind::Index { n_groups: IndexCardinality }
pub enum IndexCardinality {
    /// n_groups fixed by the structure — the parameter vector is sized to it.
    Fixed(usize),
    /// n_groups read from the binding — requires a *ragged* parameter block.
    FromBinding,
}
```

and `DataBinding.indices: Vec<Arc<[u32]>>`. `bind()` validates
`0 <= idx < n_groups` for `Fixed`, and derives `n_groups = max+1` for
`FromBinding`.

**`Fixed` works today; `FromBinding` does not.** A per-dataset group count means
a per-dataset parameter-vector length, which breaks the assumption that
`param_count`, `param_names`, and the mass matrix are structural. That is a real
and separate piece of work (§13, OQ-4). **Recommendation: Task 8 ships
`Fixed` only** — "3,000 stores, each with the same 12 product categories" — and
`FromBinding` is a follow-on. This design does not preclude it; it just does not
solve it.

### 9.4 Task 9 — borrowed NumPy and shared design matrices

**Shared matrices are already solved** by `Arc<[f64]>`: bind N datasets against
one `Arc` and X is stored once. The Python API exposes it as
`sample_batch(datasets, shared={"X": X})` — `shared` is bound once and merged
into every dataset's inputs, and a key appearing in both is an error.

**Borrowing** (true zero-copy from a NumPy buffer) is a further step and is
*not* in this design, deliberately. The blocker: `sample_batch` releases the GIL
via `py.allow_threads`, during which Python can resize or free the array behind
the pointer. The safe versions are (a) hold a `Py<PyArray1<f64>>` reference for
the duration to prevent deallocation *and* require `WRITEABLE=False`, or (b)
document `borrow=True` as an unsafe opt-in. The type that makes either possible
is the one change to make now:

```rust
pub enum DataBuffer {
    Owned(Arc<[f64]>),
    // Task 9 adds: Borrowed(BorrowedBuffer) — a lifetime-erased, refcount-pinned view.
}
```

**Recommendation: define `DataBuffer` as a one-variant enum now** so Task 9 is
additive, but implement only `Owned`. `DataBinding` fields become
`Vec<DataBuffer>` with a `#[inline] fn as_slice(&self) -> &[f64]`.

### 9.5 Task 10 — chain-preserving diagnostics

Today `BatchModelResult.samples` is `Vec<Vec<f64>>` — chains are **flattened**
(`sampler.rs:395`, `samples.extend(...)`), and `BatchResult.get_samples_2d`
reconstructs the `(chain, draw)` split by integer division
(`lib.rs:2151`). That is lossy in principle and fragile in practice.

`DatasetFit.samples` is `Vec<Vec<Vec<f64>>>` = `[chain][draw][param]` from the
start (§8), matching `SampleResult`. R-hat and ESS then work per dataset with no
reshaping, and `to_arviz()` maps directly onto the `(chain, draw, *dims)`
InferenceData layout. `BatchResult.get_samples_2d`'s index arithmetic is
deleted.

### 9.6 Task 14 — prediction on new data

Prediction is a rebind. It requires no new abstraction:

```rust
impl CompiledModel {
    /// `posterior` is [draw][param] in *unconstrained* space. `data` may have a
    /// different n_obs than the data the posterior was fit on — that is the point.
    pub fn predict(
        &self,
        data: &DataBinding,
        posterior: &[Vec<f64>],
        kind: PredictKind,
    ) -> Result<HashMap<String, Vec<Vec<f64>>>, PredictError>;
}

pub enum PredictKind { LinearPredictor, PosteriorPredictive { seed: u64 } }
```

One caveat the implementer must handle: **a prediction binding legitimately has
no observation slots.** `bind()` must accept `required = vectors + matrices`
only, when called via `bind_for_prediction()`. Response slots are marked
optional in that mode. This is the only place the schema is partially bound, and
it is why `SlotKind::Observation` is a distinct kind rather than just another
vector.

Given §1.3 (one forward pass = 8.7 µs, one rebuild = 116 µs), this is where the
design pays off most visibly: prediction on 10,000 new datasets goes from
~93 % overhead to ~0 %.

`sample_prior_predictive` (`lib.rs:2344`) is rewritten on the same path and
stops calling `compile_python_model`.

---

## 10. Migration and backward compatibility

Both existing entry points survive as thin shims over the new core. **No user
code breaks in 0.9.**

```rust
// python_bindings — unchanged signatures
#[pyfunction] fn sample(model_spec, data=None, chains=4, ...) -> FitResult
#[pyfunction] fn batch_sample(models: Vec<(ModelSpec, PyDict)>, ...) -> Vec<BatchResult>
```

- `sample(spec, data, ...)` becomes `spec.compile().sample(spec.bind(data), cfg)`.
  Behaviour is identical, including the seed (single fit, `model_idx = 0`).
- `batch_sample(models, ...)` groups the `(spec, dict)` pairs by spec identity
  (`Py::as_ptr` equality — in practice callers pass the same spec N times),
  compiles each distinct spec **once**, binds each dict, and dispatches. For the
  overwhelmingly common single-spec case this is a pure win with no API change.
- **`batch_sample` keeps the legacy positional seed derivation**
  (`seed + (idx<<32) + chain`) so existing scripts reproduce bit-for-bit. Only
  the new `sample_batch` uses identity-derived seeds. Flagged as OQ-2 — the
  reviewer may prefer to break this and take the clean derivation everywhere.
- `batch_sample` continues to return `Vec<BatchResult>` and continues to
  *raise* on the first failure, preserving today's all-or-nothing semantics.
  `sample_batch` is where partial failure lives.

Deprecation path:

| release | `sample()` | `batch_sample()` | new API |
|---|---|---|---|
| 0.9 | shim, no warning | shim, no warning | `compile()` / `sample_batch()` ship |
| 0.10 | shim, no warning | `DeprecationWarning` | docs and examples migrated |
| 1.0 | **kept forever** as sugar for `compile().sample()` | removed | canonical |

`sample()` is good API and there is no reason to remove it — one model, one
dataset is a real use case and `compile()` for it is ceremony. `batch_sample`'s
`(spec, data)`-pairs shape is the thing that is actually wrong (it invites N
builders), and that one goes.

---

## 11. Python API

```python
class ModelBuilder:
    def compile(self) -> CompiledModel: ...
    # build(), sample() etc. unchanged

class CompiledModel:
    param_names: list[str]
    schema: DataSchema           # .required_keys, .describe()

    def bind(self, data: dict, *, strict: bool = True,
             check_finite: bool = True) -> DataBinding: ...

    def sample(self, data: dict | DataBinding, *, chains: int = 4,
               draws: int = 1000, warmup: int = 500, seed: int = 42,
               threads: int = 0, init: str | dict | list = "jitter",
               sampler: str = "nuts", step_size: float = 0.0,
               max_tree_depth: int = 10, num_leapfrog_steps: int = 15,
               show_progress: bool = True) -> Fit: ...

    def sample_batch(self, datasets: Sequence[dict | DataBinding], *,
                     ids: Sequence[str] | None = None,
                     shared: dict | None = None,
                     chains: int = 1, draws: int = 500, warmup: int = 300,
                     seed: int = 42, threads: int = 0,
                     init: str | dict | list = "jitter",
                     on_error: str = "collect",
                     sampler: str = "nuts", step_size: float = 0.0,
                     max_tree_depth: int = 8, num_leapfrog_steps: int = 15,
                     show_progress: bool = True) -> BatchFit: ...

    def predict(self, data: dict | DataBinding, posterior: Fit | dict, *,
                kind: str = "posterior_predictive",
                n_samples: int | None = None, seed: int = 42) -> dict: ...

    def to_json(self) -> str: ...
    @staticmethod
    def from_json(s: str) -> "CompiledModel": ...

class BatchFit:
    ids: list[str]
    ok: list[Fit]
    errors: list[DatasetError]
    n_failed: int
    def __len__(self) -> int: ...
    def __getitem__(self, key: int | str) -> Fit: ...   # raises DatasetFailed
    def get(self, key: int | str) -> Fit | None: ...
    def __iter__(self) -> Iterator[Fit | DatasetError]: ...
    def summary(self) -> "pandas.DataFrame": ...        # one row per dataset x param
    def to_arviz(self) -> "arviz.InferenceData": ...    # Task 10
```

### 11.1 How this avoids thousands of `ModelBuilder` objects

Today the only batch entry point is
`batch_sample([(spec, data), (spec, data), ...])`, which forces the caller to
carry N spec references and forces the binding to compile N times. The new shape
inverts it: **one `CompiledModel`, a sequence of plain dicts.** There is no
per-dataset Python object at all on the way in — no `ModelBuilder`, no
`ModelSpec`, not even a `DataBinding` unless the caller wants one. The dicts are
consumed and dropped.

`DataBinding` *is* exposed, for the case where the same dataset is fit under
several models or re-used for prediction:

```python
compiled = model.compile()
b = compiled.bind({"x": x, "y": y})   # validate once
fit = compiled.sample(b)
pred = compiled.predict(b_new, fit)
```

`shared={"X": X}` is the memory lever: bind the design matrix once, hand each
dataset only its response.

---

## 12. Staged implementation plan

Seven stages. Each is independently reviewable, each leaves `cargo test --all`
green, and stages 1–3 are invisible to Python users.

**S1 — Introduce `DataSchema`, `DataBinding`, `DataBuffer` (additive only).**
New module `rust_core/src/data.rs`; `MatrixData` moves there with an `Arc`
payload and a re-export from `graph.rs` for compatibility. `Graph` gains
`schema: DataSchema`, populated in parallel with the existing data fields, which
stay. Nothing reads the schema yet. Unit tests for `bind()` error ordering and
messages.
*Files: `rust_core/src/data.rs` (new), `rust_core/src/graph.rs`, `rust_core/src/lib.rs`.*

**S2 — Thread `&DataBinding` through the evaluator and samplers.**
`Evaluator::for_structure` / `rebind` / `compute(g, d, params)`; `read_vec` reads
the binding. `run_chain` in `nuts.rs` and `hmc.rs` takes `data: &DataBinding` and
`evaluator: &mut Evaluator`. `Graph`'s data fields are still present; a temporary
`DataBinding::from_graph(&Graph)` shim keeps every caller working. **This is the
riskiest stage — do it alone, and gate it on bench B4 showing no single-fit
regression.**
*Files: `rust_core/src/autodiff.rs`, `nuts.rs`, `hmc.rs`, `sampler.rs`.*

**S3 — Delete the data fields from `Graph`.**
Remove `data_vectors` / `obs_vectors` / `data_matrices` and
`validate_shapes()`; delete `DataBinding::from_graph`. Fix the ~15 call sites in
`autodiff.rs` and the 6 in `python_bindings/src/lib.rs:890-943`. Move
`ObservationHead::n_obs` onto the binding. Add `Graph::structural_check()`.
*Files: `graph.rs`, `autodiff.rs`, `sampler.rs`, `python_bindings/src/lib.rs`.*
**Merge-conflict note: coordinate S3 with whichever worktree owns `graph.rs`.**

**S4 — Rework the artifact (§3).**
Slot-referencing `ModelStep` variants, `schema` on the artifact,
`ARTIFACT_FORMAT_VERSION = 2`, `CompiledModel` with `Arc<Graph>`,
`sample(&DataBinding, cfg)`. Update the three existing tests; add a round-trip
test that binds *two different* datasets to one deserialised artifact.
*Files: `rust_core/src/compiled_model.rs`.*

**S5 — Batch core: identity, ordering, partial failure.**
`DatasetId`, `dataset_seed`, `BatchFit` / `DatasetOutcome` / `DatasetError`,
`catch_unwind` per dataset, `map_init` evaluator reuse, chain-preserving
`DatasetFit.samples`. New `sampler::sample_batch(&Arc<Graph>, Vec<DataBinding>, BatchConfig)`;
legacy `batch_sample` reimplemented on top of it.
*Files: `rust_core/src/sampler.rs`, `rust_core/tests/`.*

**S6 — Python surface.**
`#[pyclass] CompiledModel`, `DataBinding`, `BatchFit`, `DatasetError`;
`ModelBuilder::compile()`; `sample_batch` with `ids` / `shared` / `on_error` /
`init`; rewrite `sample` and `batch_sample` as shims. `parse_data_dict` produces
`Arc<[f64]>` directly.
*Files: `python_bindings/src/lib.rs`, `examples/`, `docs/reference.md`.*

**S7 — Prediction on the same path.**
`CompiledModel::predict`, `bind_for_prediction`, and a rewrite of
`sample_prior_predictive` and `FitResult`'s predictive helpers to stop
recompiling. Delete `FitResult.graph` (the cloned graph kept for prediction) in
favour of `Arc<Graph>` + the binding.
*Files: `rust_core/src/compiled_model.rs`, `python_bindings/src/lib.rs`.*

S1–S3 must be sequential. S4 and S5 can proceed in parallel once S3 lands. S6
needs S4+S5. S7 needs S6.

---

## 13. Benchmark plan

Committed as `rust_core/benches/` (Criterion) plus one Python end-to-end script.
**The acceptance bar is set to what §1.3 says is actually achievable — do not
claim a fit-throughput win.**

| id | what | baseline (today) | target |
|---|---|---|---|
| **B1** | Setup time only: N datasets from dict to sampler-ready, N ∈ {100, 1k, 10k}, n=2000, k=8 | `compile_python_model` × N + `graph.clone()` × N | ≥ **20× faster**; must be O(N × slots), not O(N × payload) |
| **B2** | Peak RSS for the same, measured at the moment sampling starts | ≈ 3–4× payload | ≤ **1.1× payload**; with `shared={"X":X}` ≤ **payload/k + ε** |
| **B3** | Shared design matrix, 10k datasets: resident bytes for X | 10,000 copies | **1 copy** (assert via pointer identity, not just RSS) |
| **B4** | **Single-fit throughput guard.** ns/leapfrog for one n=2000 k=8 NUTS fit | current | **within ±2 %.** A regression here fails the PR — this is the load-bearing check on §4.3 |
| **B5** | Prediction rebind: 10k forward passes over new data | rebuild+clone+compute = 125 µs/dataset | ≤ **12 µs/dataset** (≥ 10×) |
| **B6** | End-to-end `sample_batch` wall clock, 1k datasets, 1 chain, 500 draws | current `batch_sample` | **within ±3 %** — parity is the goal, not a win |
| **B7** | Evaluator reuse: allocations per dataset in a batch (via a counting allocator) | 1 `Evaluator` per chain | **0** after the first dataset per worker |

Correctness benchmarks — these gate the PR as hard as the timings:

- **B8** Bit-identical draws: `compiled.sample(d, seed=s)` ≡ legacy
  `sample(spec, d, seed=s)` for every example model in `examples/`.
- **B9** Seed stability: fitting dataset `"abc"` alone, in a batch of 10, and in
  a batch of 10,000 with different neighbours and `threads` ∈ {1, 8} yields
  identical draws.
- **B10** Order preservation and partial failure: a 1,000-dataset batch with
  datasets 7, 300, 999 deliberately malformed returns 1,000 outcomes in input
  order with exactly those three as `Err`.

---

## 14. Risks and open questions

### Risks

**R1 — S2/S3 touch the hot path.** `Evaluator::compute` is the single most
performance-critical function in the project and the change touches its inner
loop's addressing. *Mitigation:* B4 as a hard gate; the offset-table fallback in
§4.3(1) is pre-designed so there is a known answer if it regresses.

**R2 — Merge conflict with three concurrent worktrees.** S3 rewrites parts of
`graph.rs` and `autodiff.rs`. *Mitigation:* the `Op` enum is untouched, which is
where most concurrent edits will land; S1 is purely additive and can merge early
to shrink the S3 diff; S3 should be scheduled after the other worktrees' graph
work merges, not before.

**R3 — Artifact v1 is silently broken by v2.** Anyone with a serialised v1
artifact gets a hard `VersionMismatch`. *Mitigation:* v1 has no users outside
its own tests (verified). Accept the break; do not write a migration.

**R4 — 10k `DataBinding`s still means 10k live datasets.** The design removes
the *duplicate* copies, not the originals. A truly memory-bounded run needs a
streaming/lazy-bind API. *Mitigation:* out of scope, but §5's cheap `bind()` is
what makes it a later addition rather than a redesign. Note it in the docs so
nobody promises unbounded scale.

### Open questions — **reviewer decision needed before implementation starts**

**OQ-1 — Does `Graph` get renamed to `ModelStructure`?**
§2.1 recommends *no* (large diff, active concurrent edits, no functional gain).
But "Graph" will read oddly once it holds a schema and no data. Decide now;
renaming later is worse. *Recommendation: keep `Graph`.*

**OQ-2 — Do we break `batch_sample`'s seed reproducibility?**
§10 preserves the legacy positional derivation in the shim so existing scripts
reproduce bit-for-bit, at the cost of two seed schemes coexisting until 1.0. The
alternative is one clean scheme and a documented one-time reproducibility break
in 0.9. *Recommendation: preserve, and accept the duplication — but this is a
product call, not an engineering one.*

**OQ-3 — Are dimension names structural or per-binding?**
§9.1 assumes names+count are structural, sizes+labels per-binding. If Task 6
needs per-dataset dimension *names*, `DataSchema` must move partly into
`DataBinding` and §2's boundary shifts. **Whoever owns Task 6 should confirm
this before S1 lands.**

**OQ-4 — Does Task 8 need per-dataset group counts?**
§9.3: `Fixed` (same group count for every dataset) works within this design;
`FromBinding` (per-dataset group count) implies a per-dataset parameter-vector
length, which breaks `param_count` / `param_names` / mass-matrix as structural
properties. That is a substantially larger change and would alter §2's boundary.
*Recommendation: Task 8 ships `Fixed` only.* Needs confirmation that the
intended use cases fit.

**OQ-5 — Is `check_finite=True` the right default?**
It costs an O(total payload) scan at bind time (~30 ms per 10k datasets) and
catches the single most common user error. *Recommendation: on by default.*

**OQ-6 — Is `on_error="collect"` the right default for `sample_batch`?**
Collecting is right for production batch runs and wrong for a notebook, where a
silent `n_failed=1` may go unnoticed. *Recommendation: collect by default, and
have `BatchFit.__repr__` print failure count prominently in red.*

**OQ-7 — Should `Fit`/`DatasetFit` retain a reference to its `DataBinding`?**
Convenient (`fit.predict(new_x)` needs the model; posterior-predictive checks
need the original data) but it pins every dataset in memory for the lifetime of
the results — directly against R4. *Recommendation: `BatchFit` does **not**
retain bindings; single `Fit` does, since there is only one.*
