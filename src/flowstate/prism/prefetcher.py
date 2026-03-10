"""Double-buffered async prefetch pipeline.

Decouples data loading from consumption by prefetching the next N batches
in a background thread. While the consumer processes batch N, the prefetcher
is already loading batch N+1 (and optionally N+2) into pinned memory.

This eliminates I/O stalls in the training loop and enables overlap of
CPU→GPU transfers with GPU computation when combined with CUDA streams.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from flowstate.prism.pinned_buffer import PinnedBuffer, PinnedBufferPool

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Signals end-of-stream


@dataclass
class PrefetchConfig:
    """Configuration for the prefetch pipeline."""

    depth: int = 2  # Number of batches to prefetch ahead
    timeout_s: float = 30.0  # Timeout for queue operations


@dataclass
class PrefetchStats:
    """Statistics from a prefetch run."""

    batches_prefetched: int = 0
    batches_consumed: int = 0
    stalls: int = 0  # Times consumer had to wait for prefetcher
    errors: int = 0


class PrefetchedBatch:
    """A batch that has been prefetched into pinned memory.

    Holds both the original Arrow batch (for metadata) and pinned
    numpy arrays for numeric columns (for fast GPU transfer).
    """

    def __init__(
        self,
        batch: pa.RecordBatch,
        pinned_arrays: dict[str, PinnedBuffer] | None = None,
    ) -> None:
        self._batch = batch
        self._pinned_arrays = pinned_arrays or {}

    @property
    def batch(self) -> pa.RecordBatch:
        return self._batch

    @property
    def num_rows(self) -> int:
        return self._batch.num_rows

    @property
    def schema(self) -> pa.Schema:
        return self._batch.schema

    def column_numpy(self, name: str) -> np.ndarray:
        """Get a column as a numpy array, from pinned memory if available."""
        if name in self._pinned_arrays:
            col = self._batch.column(name)
            buf = self._pinned_arrays[name]
            if pa.types.is_timestamp(col.type):
                return buf.view(np.int64, (self._batch.num_rows,))
            elif pa.types.is_floating(col.type):
                return buf.view(np.float64, (self._batch.num_rows,))
            elif pa.types.is_integer(col.type):
                return buf.view(np.int64, (self._batch.num_rows,))
        # Fallback to Arrow conversion
        col = self._batch.column(name)
        if pa.types.is_timestamp(col.type):
            return col.cast(pa.int64()).to_numpy(zero_copy_only=False)
        return col.to_numpy(zero_copy_only=False)

    @property
    def pinned_columns(self) -> list[str]:
        return list(self._pinned_arrays.keys())

    def release_to(self, pool: PinnedBufferPool) -> None:
        """Release pinned buffers back to the pool."""
        for buf in self._pinned_arrays.values():
            pool.release(buf)
        self._pinned_arrays.clear()


class PrefetchPipeline:
    """Double-buffered prefetch pipeline with pinned memory staging.

    Runs a background thread that reads from a batch iterator, copies
    numeric columns into pinned memory, and enqueues the results for
    the consumer.

    Example::

        source = replay_engine.replay(filter)
        pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price", "size"])

        for prefetched in pipeline.iter(source):
            prices = prefetched.column_numpy("price")  # Already in pinned memory
            # ... transfer to GPU, train ...
            prefetched.release_to(pool)
    """

    def __init__(
        self,
        pool: PinnedBufferPool | None = None,
        config: PrefetchConfig | None = None,
        numeric_columns: list[str] | None = None,
    ) -> None:
        self._pool = pool or PinnedBufferPool()
        self._config = config or PrefetchConfig()
        self._numeric_columns = numeric_columns
        self._stats = PrefetchStats()
        self._stop_event = threading.Event()

    @property
    def stats(self) -> PrefetchStats:
        return PrefetchStats(
            batches_prefetched=self._stats.batches_prefetched,
            batches_consumed=self._stats.batches_consumed,
            stalls=self._stats.stalls,
            errors=self._stats.errors,
        )

    def iter(self, source: Iterator[pa.RecordBatch]) -> Iterator[PrefetchedBatch]:
        """Prefetch batches from source and yield them with pinned memory staging.

        Args:
            source: Iterator of Arrow RecordBatches (e.g., from ReplayEngine).

        Yields:
            PrefetchedBatch instances with numeric columns in pinned memory.
        """
        self._stop_event.clear()
        self._stats = PrefetchStats()
        buf_queue: queue.Queue[PrefetchedBatch | object] = queue.Queue(
            maxsize=self._config.depth
        )

        producer = threading.Thread(
            target=self._producer_loop,
            args=(source, buf_queue),
            daemon=True,
        )
        producer.start()

        try:
            yield from self._consumer_loop(buf_queue)
        finally:
            self._stop_event.set()
            # Drain remaining items so producer can finish
            while True:
                try:
                    item = buf_queue.get_nowait()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, PrefetchedBatch):
                        item.release_to(self._pool)
                except queue.Empty:
                    break
            producer.join(timeout=5.0)

    def stop(self) -> None:
        """Signal the prefetcher to stop."""
        self._stop_event.set()

    def _producer_loop(
        self,
        source: Iterator[pa.RecordBatch],
        buf_queue: queue.Queue[PrefetchedBatch | object],
    ) -> None:
        """Background thread: read batches, pin, enqueue."""
        try:
            for batch in source:
                if self._stop_event.is_set():
                    break
                if batch.num_rows == 0:
                    continue

                prefetched = self._pin_batch(batch)
                self._stats.batches_prefetched += 1

                # Block if queue is full (backpressure)
                while not self._stop_event.is_set():
                    try:
                        buf_queue.put(prefetched, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            logger.error(f"Prefetch producer error: {e}")
            self._stats.errors += 1
        finally:
            buf_queue.put(_SENTINEL)

    def _consumer_loop(
        self, buf_queue: queue.Queue[PrefetchedBatch | object]
    ) -> Iterator[PrefetchedBatch]:
        """Main thread: dequeue and yield prefetched batches."""
        while True:
            try:
                item = buf_queue.get(timeout=self._config.timeout_s)
            except queue.Empty:
                self._stats.stalls += 1
                continue

            if item is _SENTINEL:
                break

            self._stats.batches_consumed += 1
            yield item  # type: ignore[misc]

    def _pin_batch(self, batch: pa.RecordBatch) -> PrefetchedBatch:
        """Copy numeric columns into pinned memory."""
        pinned: dict[str, PinnedBuffer] = {}

        if self._numeric_columns:
            columns_to_pin = list(self._numeric_columns)
        else:
            # Auto-detect numeric columns
            columns_to_pin = [
                field.name for field in batch.schema
                if pa.types.is_floating(field.type)
                or pa.types.is_integer(field.type)
                or pa.types.is_timestamp(field.type)
            ]

        for name in columns_to_pin:
            if name not in batch.schema.names:
                continue
            col = batch.column(name)
            if pa.types.is_timestamp(col.type):
                np_array = col.cast(pa.int64()).to_numpy(zero_copy_only=False)
            elif pa.types.is_floating(col.type) or pa.types.is_integer(col.type):
                np_array = col.to_numpy(zero_copy_only=False)
            else:
                continue

            buf = self._pool.allocate_like(np_array)
            pinned[name] = buf

        return PrefetchedBatch(batch, pinned)
