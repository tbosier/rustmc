# Stream repeated fits

`compiled.sample_iter()` consumes `(stable_id, data)` pairs lazily. It fits one
bounded chunk at a time and yields a `BatchItem` for each dataset in input order.

```python
with compiled.sample_iter(
    datasets, chunk_size=16, retention="summary", errors="collect",
    chains=4, warmup=1000, draws=1000, threads=4, seed=42,
) as results:
    for item in results:
        save(item.id, item.error, item.diagnostics, item.warnings)
```

`datasets` can be a generator. `save` is your own result sink. Each dataset, and the
positions an `init` callable returns, are deep-copied when they are read, so a
generator may refill and yield the same buffers; the copy keeps types and masks, so
`sample_batch` validates what the source gave. Memoryviews are copied too. A bound model
is passed as given, being immutable, and so is a value that cannot be copied at all,
such as a generator. With `retention="full"` (the default),
each successful item also has a `.fit` for prediction. Summary retention releases
posterior draws after their chunk; `parameters=["beta[0]"]` selects diagnostic rows.
Sampling options are those of `sample_batch` and are checked when the stream is
created, so a bad option fails at once even when every job is already complete.

A failed fit has an `.error`, no fit and empty `.warnings`: no check ran on it, so test
`.error` first. For a successful fit, under either retention, `.warnings` holds the
fit's failed or unavailable checks as `BatchResult.warnings()` reports them (see
[fit warnings](reference.md#fit-warnings-and-batch-triage)); an empty tuple then means
every check passed, not that the posterior is right. Inspect the diagnostics too.

Memory is bounded by the current input chunk, its fits, and worker scratch space: the
previous chunk's data, initial positions and fits are released before the next chunk
is read, unless you keep them. An exception you keep after the stream stops holds
none of them either. An input list you build also occupies memory. The stream
tracks IDs in a SQLite file with a bounded page cache, so duplicate detection does not
retain every ID in Python memory. The file uses space proportional to the job count in
the temporary directory (`TMPDIR`); where that directory is in memory, as with tmpfs,
so is the ledger. Duplicate IDs abort the stream, including duplicates across chunks.

Each chunk is sampled in full before any of its items is yielded. Under the default
`errors="raise"`, a dataset that fails to bind or sample raises before the rest of its
chunk is yielded; `errors="collect"` instead yields it as a failed item. Either way, an
exception from the input, from an `init` callable, or from `shared` data that cannot
be converted stops the stream, and the jobs of the current chunk that were read but not
yet yielded must be run again.

Persist completed results as you consume them. On restart, supply `completed_ids`
as an iterable and keep the original dataset IDs, model, sampling options, and seed.
Completed IDs are skipped and yield no item; other jobs reproduce the eager batch's
per-ID streams. This reruns jobs from their seeds, not from partially sampled chains.
Failed jobs should not be marked completed unless you intend to skip them.

Use the context manager or call `.close()` when stopping early. It releases the current
batch, removes the temporary ID ledger, closes a closeable input iterator, and drops the
stream's references to both; it closes the input even if releasing the batch fails. It
may be called from another thread than the one iterating, and then waits for a
`next()` in progress to return. A failure to close the input is raised, also when the
stream ends normally; when the stream stops on another error, that error is raised
with a note about the failed close, also when leaving a `with` block, and calling
`.close()` again retries it. A stream that is garbage
collected unclosed closes itself, but when that
happens is up to Python. No next chunk starts until the current chunk is consumed.
Native worker count applies to each chunk; pool creation currently occurs once per
chunk.

`init` accepts a mapping from dataset ID to positions, whose keys must be strings, or
a callable `(id, data) -> positions` that is called as each dataset is read and not
for completed ones. An ID the mapping lacks gets a random start; positions of `None`
are passed on, and are an error for that dataset as in `sample_batch`. Draws reproduce across chunk sizes, input order and restarts only
if the callable gives each ID the same positions whatever the order of the calls: a
function of the ID and data, not of a shared random generator. A supplied mapping
stays in caller memory; use a callable for an unbounded source. Mapping IDs that no
dataset had, counting only datasets that were read, are reported when the stream ends.
