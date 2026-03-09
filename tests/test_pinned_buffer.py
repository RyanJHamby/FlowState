"""Tests for the pinned memory buffer pool."""

from __future__ import annotations

import numpy as np

from flowstate.prism.pinned_buffer import (
    PinnedBuffer,
    PinnedBufferConfig,
    PinnedBufferPool,
)


class TestPinnedBuffer:
    def test_create_buffer(self):
        array = np.zeros(1024, dtype=np.uint8)
        buf = PinnedBuffer(array, pinned=False, size_bytes=1024)
        assert buf.size_bytes == 1024
        assert buf.nbytes == 1024
        assert not buf.is_pinned

    def test_view(self):
        array = np.zeros(800, dtype=np.uint8)
        buf = PinnedBuffer(array, pinned=False, size_bytes=800)
        viewed = buf.view(np.float64, (100,))
        assert viewed.shape == (100,)
        assert viewed.dtype == np.float64

    def test_as_numpy(self):
        array = np.arange(10, dtype=np.uint8)
        buf = PinnedBuffer(array, pinned=False, size_bytes=10)
        assert np.array_equal(buf.as_numpy(), array)


class TestPinnedBufferPool:
    def test_allocate_returns_buffer(self):
        pool = PinnedBufferPool()
        buf = pool.allocate(1024)
        assert buf.size_bytes >= 1024
        assert isinstance(buf.array, np.ndarray)

    def test_alignment(self):
        config = PinnedBufferConfig(alignment=512)
        pool = PinnedBufferPool(config)
        buf = pool.allocate(100)
        assert buf.size_bytes % 512 == 0
        assert buf.size_bytes == 512

    def test_pool_reuse(self):
        pool = PinnedBufferPool()
        buf1 = pool.allocate(1024)
        size = buf1.size_bytes
        pool.release(buf1)

        buf2 = pool.allocate(1024)
        assert buf2.size_bytes == size

        stats = pool.stats
        assert stats.pool_hits == 1
        assert stats.pool_misses == 1  # First allocation was a miss

    def test_pool_miss_on_different_size(self):
        pool = PinnedBufferPool()
        buf1 = pool.allocate(1024)
        pool.release(buf1)

        buf2 = pool.allocate(2048)  # Different size, won't reuse
        assert buf2.size_bytes >= 2048

        stats = pool.stats
        assert stats.pool_hits == 0
        assert stats.pool_misses == 2

    def test_stats_tracking(self):
        pool = PinnedBufferPool()
        buf1 = pool.allocate(512)
        buf2 = pool.allocate(512)
        pool.release(buf1)
        pool.release(buf2)

        stats = pool.stats
        assert stats.allocations == 2
        assert stats.deallocations == 2
        assert stats.current_pool_bytes == buf1.size_bytes + buf2.size_bytes

    def test_pool_cap_enforcement(self):
        config = PinnedBufferConfig(max_pool_bytes=1024, alignment=1)
        pool = PinnedBufferPool(config)

        buf1 = pool.allocate(800)
        buf2 = pool.allocate(800)
        pool.release(buf1)
        pool.release(buf2)  # Should be dropped (800 + 800 > 1024)

        stats = pool.stats
        assert stats.current_pool_bytes == 800  # Only first one kept

    def test_clear(self):
        pool = PinnedBufferPool()
        buf = pool.allocate(1024)
        pool.release(buf)
        assert pool.stats.current_pool_bytes > 0

        pool.clear()
        assert pool.stats.current_pool_bytes == 0

    def test_allocate_like(self):
        pool = PinnedBufferPool()
        source = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buf = pool.allocate_like(source)
        assert buf.size_bytes >= source.nbytes

        viewed = buf.view(np.float64, source.shape)
        assert np.array_equal(viewed, source)

    def test_is_pinned_false_without_cuda(self):
        pool = PinnedBufferPool()
        assert not pool.is_pinned  # No CUDA in test environment
        buf = pool.allocate(1024)
        assert not buf.is_pinned

    def test_multiple_sizes(self):
        pool = PinnedBufferPool()
        buffers = [pool.allocate(size) for size in [256, 512, 1024, 2048]]
        for buf in buffers:
            pool.release(buf)

        stats = pool.stats
        assert stats.allocations == 4
        assert stats.deallocations == 4

    def test_peak_bytes_tracked(self):
        pool = PinnedBufferPool()
        buf1 = pool.allocate(1024)
        pool.release(buf1)
        buf2 = pool.allocate(2048)
        pool.release(buf2)

        stats = pool.stats
        assert stats.peak_pool_bytes >= buf1.size_bytes
