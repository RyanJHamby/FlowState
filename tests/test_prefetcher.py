"""Tests for the double-buffered prefetch pipeline."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from flowstate.prism.pinned_buffer import PinnedBufferPool
from flowstate.prism.prefetcher import (
    PrefetchConfig,
    PrefetchedBatch,
    PrefetchPipeline,
)

TS = pa.timestamp("ns", tz="UTC")


def _make_batch(n: int = 100, offset: int = 0) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {
            "timestamp": pa.array(
                [1_000_000_000 * (i + offset) for i in range(n)], type=TS
            ),
            "price": [100.0 + (i + offset) * 0.01 for i in range(n)],
            "size": [10.0] * n,
            "symbol": ["AAPL"] * n,
        }
    )


def _batch_iter(count: int = 5, batch_size: int = 100):
    for i in range(count):
        yield _make_batch(batch_size, offset=i * batch_size)


class TestPrefetchedBatch:
    def test_column_numpy_without_pinning(self):
        batch = _make_batch(10)
        pb = PrefetchedBatch(batch)
        prices = pb.column_numpy("price")
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 10

    def test_properties(self):
        batch = _make_batch(10)
        pb = PrefetchedBatch(batch)
        assert pb.num_rows == 10
        assert "price" in pb.schema.names
        assert pb.pinned_columns == []

    def test_column_numpy_timestamp(self):
        batch = _make_batch(5)
        pb = PrefetchedBatch(batch)
        ts = pb.column_numpy("timestamp")
        assert isinstance(ts, np.ndarray)
        assert ts.dtype == np.int64


class TestPrefetchPipeline:
    def test_basic_prefetch(self):
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(
            pool=pool,
            numeric_columns=["price", "size"],
        )

        batches = list(pipeline.iter(_batch_iter(3)))
        assert len(batches) == 3

        for pb in batches:
            assert pb.num_rows == 100
            assert "price" in pb.pinned_columns
            assert "size" in pb.pinned_columns
            pb.release_to(pool)

    def test_pinned_data_matches_original(self):
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(
            pool=pool,
            numeric_columns=["price"],
        )

        source_batches = list(_batch_iter(1))
        expected = source_batches[0].column("price").to_numpy()

        prefetched = list(pipeline.iter(iter(source_batches)))
        actual = prefetched[0].column_numpy("price")

        np.testing.assert_array_almost_equal(actual, expected)
        prefetched[0].release_to(pool)

    def test_ordering_preserved(self):
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price"])

        prefetched = list(pipeline.iter(_batch_iter(5)))
        assert len(prefetched) == 5

        # Check that batches arrive in order (first batch starts at 100.0)
        prev_first = None
        for pb in prefetched:
            first_price = pb.column_numpy("price")[0]
            if prev_first is not None:
                assert first_price > prev_first
            prev_first = first_price
            pb.release_to(pool)

    def test_empty_source(self):
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool)

        batches = list(pipeline.iter(iter([])))
        assert len(batches) == 0

    def test_stats(self):
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price"])

        batches = list(pipeline.iter(_batch_iter(3)))
        for pb in batches:
            pb.release_to(pool)

        stats = pipeline.stats
        assert stats.batches_prefetched == 3
        assert stats.batches_consumed == 3
        assert stats.errors == 0

    def test_prefetch_depth(self):
        """Pipeline respects prefetch depth (backpressure)."""
        pool = PinnedBufferPool()
        config = PrefetchConfig(depth=1)
        pipeline = PrefetchPipeline(pool=pool, config=config, numeric_columns=["price"])

        # This should still work — just with a smaller queue
        batches = list(pipeline.iter(_batch_iter(5)))
        assert len(batches) == 5
        for pb in batches:
            pb.release_to(pool)

    def test_auto_detect_numeric(self):
        """When no numeric_columns specified, auto-detects them."""
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool)  # No numeric_columns

        batches = list(pipeline.iter(_batch_iter(1)))
        assert len(batches) == 1
        # Should auto-detect price, size, and timestamp as numeric
        assert "price" in batches[0].pinned_columns
        assert "size" in batches[0].pinned_columns
        batches[0].release_to(pool)

    def test_skip_empty_batches(self):
        """Empty batches from source are skipped."""
        def source_with_empty():
            yield _make_batch(10)
            yield pa.RecordBatch.from_pydict(
                {"timestamp": pa.array([], type=TS), "price": [], "size": [], "symbol": []}
            )
            yield _make_batch(10, offset=10)

        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price"])

        batches = list(pipeline.iter(source_with_empty()))
        assert len(batches) == 2
        for pb in batches:
            pb.release_to(pool)

    def test_release_returns_to_pool(self):
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price"])

        batches = list(pipeline.iter(_batch_iter(1)))
        assert pool.stats.current_pool_bytes == 0  # All in use

        batches[0].release_to(pool)
        assert pool.stats.current_pool_bytes > 0  # Returned to pool

    def test_stop(self):
        """Calling stop terminates the pipeline cleanly."""
        pool = PinnedBufferPool()
        pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price"])

        count = 0
        for pb in pipeline.iter(_batch_iter(100)):
            count += 1
            pb.release_to(pool)
            if count >= 3:
                pipeline.stop()
                break

        assert count == 3
