"""CUDA pinned memory allocator with pool and CPU fallback.

Pinned (page-locked) host memory enables asynchronous DMA transfers between
host and device, which is critical for overlapping data loading with GPU
computation. This module provides a pooled allocator that reuses pinned
buffers to avoid the high cost of repeated allocation/deallocation.

When CUDA is not available, falls back to page-aligned numpy arrays that
are compatible with the same API.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def cuda_available() -> bool:
    """Check if CUDA pinned memory is available."""
    if not HAS_CUPY:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


@dataclass
class PinnedBufferConfig:
    """Configuration for the pinned memory pool."""

    enable_pinning: bool = True
    max_pool_bytes: int = 1 << 30  # 1 GB default pool cap
    alignment: int = 512  # Byte alignment for DMA compatibility


@dataclass
class PinnedBufferStats:
    """Statistics for the pinned buffer pool."""

    allocations: int = 0
    deallocations: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    total_bytes_allocated: int = 0
    current_pool_bytes: int = 0
    peak_pool_bytes: int = 0
    is_pinned: bool = False


class PinnedBuffer:
    """A single pinned (or aligned) memory buffer.

    Wraps a numpy array that is either backed by CUDA pinned memory
    or a standard page-aligned allocation.
    """

    def __init__(self, array: np.ndarray, pinned: bool, size_bytes: int) -> None:
        self._array = array
        self._pinned = pinned
        self._size_bytes = size_bytes

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def is_pinned(self) -> bool:
        return self._pinned

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    def as_numpy(self) -> np.ndarray:
        return self._array

    def view(self, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        """Reinterpret the buffer with a different dtype/shape."""
        return np.frombuffer(self._array.data, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


class PinnedBufferPool:
    """Pooled allocator for pinned (page-locked) host memory.

    Maintains free lists keyed by buffer size. When a buffer is released,
    it goes back into the pool for reuse instead of being freed. This
    amortizes the cost of `cudaMallocHost` across many allocations.

    Thread-safe: all pool operations are protected by a lock.

    Example::

        pool = PinnedBufferPool()
        buf = pool.allocate(1024 * 1024)  # 1 MB pinned buffer
        # ... use buf.array as a numpy array ...
        pool.release(buf)                 # Return to pool for reuse
    """

    def __init__(self, config: PinnedBufferConfig | None = None) -> None:
        self._config = config or PinnedBufferConfig()
        self._use_pinned = self._config.enable_pinning and cuda_available()
        self._lock = threading.Lock()

        # Free list: size_bytes -> list of PinnedBuffer
        self._free: dict[int, list[PinnedBuffer]] = defaultdict(list)
        self._stats = PinnedBufferStats(is_pinned=self._use_pinned)

        if self._use_pinned:
            logger.info("PinnedBufferPool: using CUDA pinned memory")
        else:
            logger.info("PinnedBufferPool: using aligned CPU memory (CUDA not available)")

    @property
    def stats(self) -> PinnedBufferStats:
        with self._lock:
            return PinnedBufferStats(
                allocations=self._stats.allocations,
                deallocations=self._stats.deallocations,
                pool_hits=self._stats.pool_hits,
                pool_misses=self._stats.pool_misses,
                total_bytes_allocated=self._stats.total_bytes_allocated,
                current_pool_bytes=self._stats.current_pool_bytes,
                peak_pool_bytes=self._stats.peak_pool_bytes,
                is_pinned=self._stats.is_pinned,
            )

    @property
    def is_pinned(self) -> bool:
        return self._use_pinned

    def allocate(self, size_bytes: int) -> PinnedBuffer:
        """Allocate a pinned buffer, reusing from pool if available.

        Args:
            size_bytes: Minimum size in bytes. Will be rounded up to alignment.

        Returns:
            A PinnedBuffer backed by pinned or aligned memory.
        """
        aligned_size = self._align(size_bytes)

        with self._lock:
            self._stats.allocations += 1
            self._stats.total_bytes_allocated += aligned_size

            # Check free list
            free_list = self._free.get(aligned_size)
            if free_list:
                buf = free_list.pop()
                self._stats.pool_hits += 1
                self._stats.current_pool_bytes -= aligned_size
                return buf

            self._stats.pool_misses += 1

        # Allocate outside the lock
        return self._allocate_new(aligned_size)

    def release(self, buf: PinnedBuffer) -> None:
        """Return a buffer to the pool for reuse.

        If the pool is at capacity, the buffer is discarded instead.
        """
        with self._lock:
            self._stats.deallocations += 1

            if self._stats.current_pool_bytes + buf.size_bytes <= self._config.max_pool_bytes:
                self._free[buf.size_bytes].append(buf)
                self._stats.current_pool_bytes += buf.size_bytes
                self._stats.peak_pool_bytes = max(
                    self._stats.peak_pool_bytes, self._stats.current_pool_bytes
                )
            # else: drop the buffer, let GC handle it

    def allocate_like(self, array: np.ndarray) -> PinnedBuffer:
        """Allocate a pinned buffer matching the given array's size and copy data into it."""
        buf = self.allocate(array.nbytes)
        dest = buf.view(array.dtype, array.shape)
        np.copyto(dest, array)
        return buf

    def clear(self) -> None:
        """Free all pooled buffers."""
        with self._lock:
            self._free.clear()
            self._stats.current_pool_bytes = 0

    def _align(self, size_bytes: int) -> int:
        """Round up to the configured alignment."""
        a = self._config.alignment
        return ((size_bytes + a - 1) // a) * a

    def _allocate_new(self, aligned_size: int) -> PinnedBuffer:
        """Allocate a new buffer (pinned or aligned)."""
        if self._use_pinned:
            return self._allocate_pinned(aligned_size)
        return self._allocate_aligned(aligned_size)

    def _allocate_pinned(self, size_bytes: int) -> PinnedBuffer:
        """Allocate CUDA pinned host memory."""
        try:
            mem = cp.cuda.alloc_pinned_memory(size_bytes)
            array = np.frombuffer(mem, dtype=np.uint8, count=size_bytes)
            return PinnedBuffer(array, pinned=True, size_bytes=size_bytes)
        except Exception as e:
            logger.warning(f"Pinned allocation failed ({e}), falling back to aligned")
            return self._allocate_aligned(size_bytes)

    def _allocate_aligned(self, size_bytes: int) -> PinnedBuffer:
        """Allocate page-aligned CPU memory as fallback."""
        array = np.empty(size_bytes, dtype=np.uint8)
        return PinnedBuffer(array, pinned=False, size_bytes=size_bytes)
