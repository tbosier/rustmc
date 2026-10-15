"""Bounded input and result streaming for repeated compiled models."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import copy
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from ._rustmc import BoundModel


@dataclass(frozen=True)
class BatchItem:
    """One dataset's fit or error. Summary retention releases posterior draws.

    ``warnings`` holds a successful fit's failed or unavailable checks, as
    ``BatchResult.warnings()`` reports them, under either retention. A failed
    item has an ``error``, no fit and no warnings: no check ran.
    """
    id: str
    fit: Any = None
    diagnostics: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    warnings: tuple[dict[str, Any], ...] = ()


_NO_INIT = object()


def _copied(value: Any) -> Any:
    """A deep copy of ``value``, or ``value`` itself if it cannot be copied:
    sample_batch then validates it, and collects the error, as given."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot(data: Any) -> Any:
    """Copy a dataset when it is read, so that a source may reuse its
    buffers: binding happens only when the whole chunk is sampled. A deep copy
    keeps every type, mask and malformed entry, so that sample_batch validates
    what the source gave. A value that cannot be deep-copied, such as a
    memoryview, is passed as given and must not be reused before its chunk is
    sampled. A bound model is immutable and is passed as given."""
    if isinstance(data, Mapping):
        return {key: _copied(value) for key, value in data.items()}
    if isinstance(data, BoundModel):
        return data
    return _copied(data)


class BatchStream(Iterator[BatchItem]):
    """Lazy chunks; close the stream to release its current batch and input iterator."""
    def __init__(self, compiled: Any, datasets: Iterable[tuple[str, Any]], *,
                 chunk_size: int = 16, retention: str = "full", shared: Any = None,
                 completed_ids: Iterable[str] = (), parameters: Iterable[str] | None = None,
                 **sampling: Any):
        self._closed = True  # until construction succeeds, there is nothing to close
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if retention not in ("full", "summary"):
            raise ValueError("retention must be 'full' or 'summary'")
        if parameters is not None and retention != "summary":
            raise ValueError("parameters selects diagnostics only; use retention='summary'")
        forbidden = {"ids", "chunk_size", "shared", "seed_policy"} & sampling.keys()
        if forbidden:
            raise ValueError(f"stream controls cannot be overridden: {sorted(forbidden)}")
        if sampling.get("errors", "raise") not in ("raise", "collect"):
            raise ValueError("errors must be 'raise' or 'collect'")
        initial = sampling.get("init")
        if initial is not None and not isinstance(initial, Mapping) and not callable(initial):
            raise TypeError("init must be an ID mapping or a callable (id, data) -> positions")
        if isinstance(initial, Mapping):
            for key in initial:
                self._validate_id(key, "init keys")
        selected = None if parameters is None else tuple(parameters)
        if selected is not None and (not selected or len(set(selected)) != len(selected)):
            raise ValueError("parameters must be a nonempty list of unique names")
        # Check every sampling option now, as sample_batch does, rather than
        # when the first unfinished job arrives, which a restart may never reach.
        options = {"show_progress": False, **sampling}
        options.pop("init", None)
        compiled.sample_batch([], ids=[], shared=shared, chunk_size=chunk_size,
                              seed_policy="cell_id_v1", **options)
        self._source = iter(datasets)
        # Serialises next() and close(): a close from another thread waits
        # for a next() in progress to return.
        self._lock = threading.RLock()
        self._iterator = self._iterate(compiled, chunk_size, retention, shared,
                                       completed_ids, selected, sampling)
        self._closed = False

    def __iter__(self) -> BatchStream:
        return self

    def __next__(self) -> BatchItem:
        with self._lock:
            if self._closed:
                raise StopIteration
            try:
                return next(self._iterator)
            except StopIteration:
                # Exhausted normally: a failure to clean up is the news.
                self.close()
                raise
            except BaseException as error:
                try:
                    self.close()
                except Exception as cleanup:
                    # The error that stopped iteration is the one raised; the
                    # input stays open, and close() can be called again.
                    if hasattr(error, "add_note"):
                        error.add_note(f"closing the input also failed: {cleanup!r}")
                raise

    def close(self) -> None:
        """Release the current chunk and its ledger, and close the input
        iterator if it has a ``close`` method, then drop the stream's
        references to both. The input is closed even if releasing the chunk
        fails, and if closing the input fails, calling this again retries it.
        From another thread, this waits for a ``next()`` in progress to return."""
        lock = getattr(self, "_lock", None)
        if lock is None:  # construction failed; there is nothing to close
            return
        with lock:
            self._closed = True
            iterator, self._iterator = self._iterator, None
            try:
                if iterator is not None:
                    iterator.close()
            finally:
                source = self._source
                close = getattr(source, "close", None)
                if close is not None:
                    close()  # on failure the source is kept for a retry
                self._source = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> BatchStream:
        return self

    def __exit__(self, *_: Any) -> bool:
        self.close()
        return False

    def _iterate(self, compiled, chunk_size, retention, shared, completed_ids, selected, sampling):
        # IDs live on disk: an in-memory "seen" set would grow with the job count.
        # The ledger is temporary. Callers persist results and supply completed IDs
        # on restart; no adaptation or partially sampled chain is resumed.
        with tempfile.TemporaryDirectory(prefix="rustmc-batch-") as directory:
            # The stream may be advanced on one thread and closed on another.
            connection = sqlite3.connect(Path(directory)/"ids.sqlite", check_same_thread=False)
            try:
                connection.execute("PRAGMA cache_size=-1024")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY, seen INTEGER NOT NULL)")
                for name in completed_ids:
                    self._validate_id(name)
                    connection.execute("INSERT OR IGNORE INTO ids VALUES (?, 0)", (name,))
                connection.commit()
                initial = sampling.get("init")
                while True:
                    chunk = []
                    # Skipped completed jobs do not occupy a sampling slot.
                    while len(chunk) < chunk_size:
                        try:
                            pair = next(self._source)
                        except StopIteration:
                            break
                        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                            raise ValueError("datasets must yield (stable_id, data) pairs")
                        name, data = pair
                        self._validate_id(name)
                        previous = connection.execute("SELECT seen FROM ids WHERE id=?", (name,)).fetchone()
                        if previous is not None:
                            if previous[0]:
                                raise ValueError(f"duplicate dataset ID: {name!r}")
                            connection.execute("UPDATE ids SET seen=1 WHERE id=?", (name,))
                            continue
                        connection.execute("INSERT INTO ids VALUES (?, 1)", (name,))
                        # Read and copy now, like the data, so that a reused
                        # buffer cannot change an earlier job's start.
                        # An explicit None is passed on, for sample_batch to
                        # reject; only an ID the mapping lacks gets no start.
                        if callable(initial):
                            positions = _copied(initial(name, data))
                        elif initial is not None and name in initial:
                            positions = _copied(initial[name])
                        else:
                            positions = _NO_INIT
                        chunk.append((name, _snapshot(data), positions))
                    data = pair = positions = None
                    connection.commit()
                    if not chunk:
                        if isinstance(initial, Mapping):
                            for key in initial:
                                # Completed IDs are in the ledger too, but only
                                # those a dataset had are marked seen.
                                seen = connection.execute(
                                    "SELECT 1 FROM ids WHERE id=? AND seen=1", (key,)).fetchone()
                                if seen is None:
                                    raise ValueError(f"initialization supplied for unknown dataset ID {key!r}")
                        return
                    options = {"show_progress": False, **sampling}
                    options.pop("init", None)
                    starts = {name: positions for name, _, positions in chunk
                              if positions is not _NO_INIT}
                    if starts:
                        options["init"] = starts
                    batch = compiled.sample_batch(
                        [data for _, data, _ in chunk], ids=[name for name, _, _ in chunk],
                        shared=shared, chunk_size=chunk_size, seed_policy="cell_id_v1", **options,
                    )
                    # Nothing of this chunk's input outlives its sampling.
                    del options, starts
                    names = [name for name, _, _ in chunk]
                    chunk = None
                    result = item = diagnostics = None
                    try:
                        errors = batch.errors
                        for name in names:
                            if name in errors:
                                yield BatchItem(name, error=errors[name])
                                continue
                            result = batch.get(name)
                            diagnostics = result.diagnostics()
                            if selected is not None:
                                lookup = {entry["name"]: entry for entry in diagnostics}
                                unknown = set(selected)-lookup.keys()
                                if unknown:
                                    raise ValueError(f"unknown diagnostic parameters: {sorted(unknown)}")
                                diagnostics = [lookup[key] for key in selected]
                            item = BatchItem(name, result.fit if retention == "full" else None,
                                             tuple(diagnostics), warnings=tuple(result.warnings()))
                            yield item
                            result = item = diagnostics = None
                    finally:
                        # A traceback kept by the caller keeps this frame; it
                        # must not keep the chunk's fits with it.
                        batch = errors = names = result = item = diagnostics = None
            finally:
                connection.close()

    @staticmethod
    def _validate_id(name, what="dataset IDs"):
        if not isinstance(name, str) or not name:
            raise ValueError(f"{what} must be nonempty strings")


def sample_iter(compiled: Any, datasets: Iterable[tuple[str, Any]], **kwargs: Any) -> BatchStream:
    """Fit a lazy sequence of (ID, data) pairs with bounded chunk retention.

    Sampling options match CompiledModel.sample_batch and are checked when the
    stream is created. Keep IDs and the base seed unchanged, and make an ``init``
    callable a function of the ID and data, to reproduce jobs across chunk sizes,
    order, and resumed runs. Each dataset, and the positions an ``init`` callable
    returns, are deep-copied when read, so a source may reuse its buffers. Caller-retained BatchItems and a caller-materialized input
    list still use memory.
    """
    return BatchStream(compiled, datasets, **kwargs)
