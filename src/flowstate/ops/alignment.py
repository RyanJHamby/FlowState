"""Cache-line aligned allocators for high-performance data structures."""

from __future__ import annotations

import contextlib
import ctypes
import mmap
from typing import Final

# Standard cache line size for modern x86/ARM processors
CACHE_LINE_SIZE: Final[int] = 64


def aligned_size(size: int, alignment: int = CACHE_LINE_SIZE) -> int:
    """Round up a size to the nearest multiple of alignment.

    Args:
        size: The requested size in bytes.
        alignment: Alignment boundary (default: cache line size).

    Returns:
        The aligned size.
    """
    return (size + alignment - 1) & ~(alignment - 1)


def is_aligned(address: int, alignment: int = CACHE_LINE_SIZE) -> bool:
    """Check if an address is aligned to the given boundary."""
    return address % alignment == 0


class AlignedBuffer:
    """A buffer aligned to cache-line boundaries.

    Uses mmap for page-aligned allocation, which guarantees cache-line alignment
    since page boundaries are always at least cache-line aligned.
    """

    def __init__(self, size: int) -> None:
        self._requested_size = size
        self._aligned_size = aligned_size(size)
        self._mmap = mmap.mmap(-1, self._aligned_size)
        # Get the buffer address
        buf = (ctypes.c_char * self._aligned_size).from_buffer(self._mmap)
        self._address = ctypes.addressof(buf)

    @property
    def size(self) -> int:
        return self._aligned_size

    @property
    def address(self) -> int:
        return self._address

    @property
    def is_cache_aligned(self) -> bool:
        return is_aligned(self._address)

    def write(self, data: bytes, offset: int = 0) -> None:
        """Write bytes to the buffer at a given offset."""
        if offset + len(data) > self._aligned_size:
            raise ValueError("Write exceeds buffer size")
        self._mmap.seek(offset)
        self._mmap.write(data)

    def read(self, size: int, offset: int = 0) -> bytes:
        """Read bytes from the buffer at a given offset."""
        if offset + size > self._aligned_size:
            raise ValueError("Read exceeds buffer size")
        self._mmap.seek(offset)
        return self._mmap.read(size)

    def close(self) -> None:
        """Release the underlying memory."""
        self._mmap.close()

    def __enter__(self) -> AlignedBuffer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._mmap.close()
