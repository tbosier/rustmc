# Stream repeated fits

`compiled.sample_iter()` consumes `(stable_id, data)` pairs lazily. It fits one
bounded chunk at a time and yields a `BatchItem` for each dataset in input order.

```python
with compiled.sample_iter(
    datasets, chunk_size=16, retention="summary", errors="collect",
    chains=4, warmup=1000, draws=1000, threads=4, seed=42,
) as results:
    for item in results:
        save(item.id, item.diagnostics, item.error)
```

`datasets` can be a generator. `save` is your own result sink. With `retention="full"`
(the default), each successful item also has a `.fit` for prediction. Summary retention
releases posterior draws after their chunk; `parameters=["beta[0]"]` selects diagnostic
rows. Failed fits have an `.error` and no fit. Computational success does not imply
adequate convergence; inspect the diagnostics.

Memory is bounded by the current input chunk, its fits, and worker scratch space.
Results you keep and an input list you build still occupy memory. The stream tracks
IDs in a temporary SQLite file with a bounded page cache, so duplicate detection does
not retain every ID in Python memory. The ledger uses disk space proportional to job
count. Duplicate IDs abort the stream, including duplicates across chunks.

Persist completed results as you consume them. On restart, supply `completed_ids`
as an iterable and keep the original dataset IDs, model, sampling options, and seed.
Completed IDs are skipped; other jobs reproduce the eager batch's per-ID streams.
This reruns jobs from their seeds, not from partially sampled chains. Failed jobs should
not be marked completed unless you intend to skip them.

Use the context manager or call `.close()` when stopping early. It releases the current
batch, closes a closeable input iterator, and removes the temporary ID ledger. No next
chunk starts until the current chunk is consumed. Native worker count applies to each
chunk; pool creation currently occurs once per chunk.

`init` accepts an ID-to-positions mapping, or a callable `(id, data) -> positions`
that creates initial values for the current chunk. A supplied mapping stays in
caller memory; use a callable for an unbounded source. Unknown mapping IDs are
reported when the stream ends. Exceptions from input and initialization callbacks
stop iteration; `errors="collect"` isolates native data-binding and sampling errors.
