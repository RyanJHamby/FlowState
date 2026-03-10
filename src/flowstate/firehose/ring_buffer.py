"""Lock-free shared-memory ring buffer for high-throughput message passing."""

from __future__ import annotations

import contextlib
import struct
from multiprocessing import shared_memory
from typing import Final

from flowstate.ops.alignment import CACHE_LINE_SIZE, aligned_size

# Header layout (all cache-line padded):
# [0:64)   - write_pos (int64) + padding
# [64:128) - read_pos (int64) + padding
# [128:192) - capacity (int64) + padding
# [192:256) - slot_size (int64) + padding
HEADER_SIZE: Final[int] = CACHE_LINE_SIZE * 4
WRITE_POS_OFFSET: Final[int] = 0
READ_POS_OFFSET: Final[int] = CACHE_LINE_SIZE
CAPACITY_OFFSET: Final[int] = CACHE_LINE_SIZE * 2
SLOT_SIZE_OFFSET: Final[int] = CACHE_LINE_SIZE * 3

# Each slot: [4 bytes length][payload][padding to slot_size]
LENGTH_PREFIX_SIZE: Final[int] = 4


class RingBufferFullError(Exception):
    """Raised when attempting to write to a full ring buffer."""


class RingBufferEmptyError(Exception):
    """Raised when attempting to read from an empty ring buffer."""


class RingBuffer:
    """Lock-free shared-memory ring buffer with cache-line padded slots.

    Designed for single-producer, single-consumer (SPSC) use across processes.
    Uses shared memory for zero-copy IPC.

    Args:
        name: Shared memory segment name.
        capacity: Number of slots in the ring buffer.
        slot_size: Maximum message size per slot (will be cache-line aligned).
        create: If True, create new shared memory. If False, attach to existing.
    """

    def __init__(
        self,
        name: str,
        capacity: int = 65536,
        slot_size: int = 4096,
        create: bool = True,
    ) -> None:
        self._name = name
        self._slot_size = aligned_size(slot_size)
        self._capacity = capacity
        self._total_size = HEADER_SIZE + (self._capacity * self._slot_size)

        if create:
            try:
                # Clean up any stale segment
                old = shared_memory.SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
            self._init_header()
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._capacity = self._read_int(CAPACITY_OFFSET)
            self._slot_size = self._read_int(SLOT_SIZE_OFFSET)

    def _init_header(self) -> None:
        self._write_int(WRITE_POS_OFFSET, 0)
        self._write_int(READ_POS_OFFSET, 0)
        self._write_int(CAPACITY_OFFSET, self._capacity)
        self._write_int(SLOT_SIZE_OFFSET, self._slot_size)

    def _write_int(self, offset: int, value: int) -> None:
        struct.pack_into("q", self._shm.buf, offset, value)

    def _read_int(self, offset: int) -> int:
        return struct.unpack_from("q", self._shm.buf, offset)[0]

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    def write_pos(self) -> int:
        return self._read_int(WRITE_POS_OFFSET)

    @property
    def read_pos(self) -> int:
        return self._read_int(READ_POS_OFFSET)

    @property
    def size(self) -> int:
        """Number of items currently in the buffer."""
        return self.write_pos - self.read_pos

    @property
    def is_full(self) -> bool:
        return self.size >= self._capacity

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def _slot_offset(self, pos: int) -> int:
        return HEADER_SIZE + (pos % self._capacity) * self._slot_size

    def put(self, data: bytes) -> None:
        """Write data to the next available slot.

        Args:
            data: The bytes to write (must fit in slot_size - LENGTH_PREFIX_SIZE).

        Raises:
            RingBufferFullError: If the buffer is full.
            ValueError: If data exceeds slot capacity.
        """
        max_payload = self._slot_size - LENGTH_PREFIX_SIZE
        if len(data) > max_payload:
            raise ValueError(
                f"Data size {len(data)} exceeds max payload {max_payload}"
            )
        if self.is_full:
            raise RingBufferFullError("Ring buffer is full")

        wp = self.write_pos
        offset = self._slot_offset(wp)

        # Write length prefix + data
        struct.pack_into("I", self._shm.buf, offset, len(data))
        self._shm.buf[offset + LENGTH_PREFIX_SIZE : offset + LENGTH_PREFIX_SIZE + len(data)] = data

        # Advance write position
        self._write_int(WRITE_POS_OFFSET, wp + 1)

    def get(self) -> bytes:
        """Read data from the next available slot.

        Returns:
            The bytes stored in the slot.

        Raises:
            RingBufferEmptyError: If the buffer is empty.
        """
        if self.is_empty:
            raise RingBufferEmptyError("Ring buffer is empty")

        rp = self.read_pos
        offset = self._slot_offset(rp)

        # Read length prefix
        length = struct.unpack_from("I", self._shm.buf, offset)[0]
        end = offset + LENGTH_PREFIX_SIZE + length
        data = bytes(self._shm.buf[offset + LENGTH_PREFIX_SIZE : end])

        # Advance read position
        self._write_int(READ_POS_OFFSET, rp + 1)
        return data

    def close(self) -> None:
        """Close the shared memory handle (does not unlink)."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink (destroy) the shared memory segment."""
        self._shm.unlink()

    def __enter__(self) -> RingBuffer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._shm.close()
