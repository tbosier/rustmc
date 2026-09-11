"""Bounded input and result streaming for repeated compiled models."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class BatchItem:
    """One dataset's fit or error. Summary retention releases posterior draws."""
    id: str
    fit: Any = None
    diagnostics: tuple[dict[str, Any], ...] = ()
    error: str | None = None


class BatchStream(Iterator[BatchItem]):
    """Lazy chunks; close the stream to release its current batch and input iterator."""
    def __init__(self, compiled: Any, datasets: Iterable[tuple[str, Any]], *,
                 chunk_size: int = 16, retention: str = "full", shared: Any = None,
                 completed_ids: Iterable[str] = (), parameters: Iterable[str] | None = None,
                 **sampling: Any):
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
        self._source = iter(datasets)
        self._closed = False
        selected = None if parameters is None else tuple(parameters)
        if selected is not None and (not selected or len(set(selected)) != len(selected)):
            raise ValueError("parameters must be a nonempty list of unique names")
        self._iterator = self._iterate(compiled, chunk_size, retention, shared,
                                       completed_ids, selected, sampling)

    def __iter__(self) -> BatchStream:
        return self

    def __next__(self) -> BatchItem:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._iterator.close()
            close = getattr(self._source, "close", None)
            if close is not None:
                close()

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
            connection = sqlite3.connect(Path(directory)/"ids.sqlite")
            try:
                connection.execute("PRAGMA cache_size=-1024")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY, seen INTEGER NOT NULL)")
                for name in completed_ids:
                    self._validate_id(name)
                    connection.execute("INSERT OR IGNORE INTO ids VALUES (?, 0)", (name,))
                connection.commit()
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
                        chunk.append((name, data))
                    connection.commit()
                    if not chunk:
                        initial = sampling.get("init")
                        if isinstance(initial, Mapping):
                            for key in initial:
                                if connection.execute("SELECT 1 FROM ids WHERE id=?", (key,)).fetchone() is None:
                                    raise ValueError(f"initialization supplied for unknown dataset ID {key!r}")
                        return
                    options = {"show_progress": False, **sampling}
                    initial = options.pop("init", None)
                    if callable(initial):
                        options["init"] = {name: initial(name, data) for name, data in chunk}
                    elif initial is not None:
                        options["init"] = {name: initial[name] for name, _ in chunk if name in initial}
                    batch = compiled.sample_batch(
                        [data for _, data in chunk], ids=[name for name, _ in chunk],
                        shared=shared, chunk_size=chunk_size, seed_policy="cell_id_v1", **options,
                    )
                    errors = batch.errors
                    for index in range(len(chunk)):
                        name = chunk[index][0]
                        if name in errors:
                            yield BatchItem(name, error=errors[name])
                            continue
                        result = batch.get(name)
                        diagnostics = result.diagnostics()
                        if selected is not None:
                            lookup = {item["name"]: item for item in diagnostics}
                            unknown = set(selected)-lookup.keys()
                            if unknown:
                                raise ValueError(f"unknown diagnostic parameters: {sorted(unknown)}")
                            diagnostics = [lookup[key] for key in selected]
                        item = BatchItem(name, result.fit if retention == "full" else None,
                                         tuple(diagnostics))
                        yield item
                        del result, item, diagnostics
                    del batch, chunk, errors
                    # Avoid holding the final source payload across the next chunk.
                    data = pair = None
            finally:
                connection.close()

    @staticmethod
    def _validate_id(name):
        if not isinstance(name, str) or not name:
            raise ValueError("dataset IDs must be nonempty strings")


def sample_iter(compiled: Any, datasets: Iterable[tuple[str, Any]], **kwargs: Any) -> BatchStream:
    """Fit a lazy sequence of (ID, data) pairs with bounded chunk retention.

    Sampling options match CompiledModel.sample_batch. Keep IDs and the base seed
    unchanged to reproduce jobs across chunk sizes, order, and resumed runs.
    Caller-retained BatchItems and a caller-materialized input list still use memory.
    """
    return BatchStream(compiled, datasets, **kwargs)
