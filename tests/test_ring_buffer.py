"""Tests for lock-free shared-memory ring buffer."""

from __future__ import annotations

import multiprocessing
import uuid

import pytest

from flowstate.firehose.ring_buffer import RingBuffer, RingBufferEmpty, RingBufferFull


@pytest.fixture
def ring_name() -> str:
    return f"test_ring_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def ring(ring_name: str):
    rb = RingBuffer(ring_name, capacity=16, slot_size=256)
    yield rb
    rb.close()
    rb.unlink()


class TestRingBuffer:
    def test_create(self, ring: RingBuffer):
        assert ring.capacity == 16
        assert ring.is_empty
        assert not ring.is_full
        assert ring.size == 0

    def test_put_and_get(self, ring: RingBuffer):
        ring.put(b"hello world")
        assert ring.size == 1
        data = ring.get()
        assert data == b"hello world"
        assert ring.is_empty

    def test_multiple_messages(self, ring: RingBuffer):
        messages = [f"msg_{i}".encode() for i in range(10)]
        for msg in messages:
            ring.put(msg)
        assert ring.size == 10

        for msg in messages:
            assert ring.get() == msg
        assert ring.is_empty

    def test_full_raises(self, ring: RingBuffer):
        for i in range(16):
            ring.put(f"msg_{i}".encode())
        assert ring.is_full
        with pytest.raises(RingBufferFull):
            ring.put(b"overflow")

    def test_empty_raises(self, ring: RingBuffer):
        with pytest.raises(RingBufferEmpty):
            ring.get()

    def test_wraparound(self, ring: RingBuffer):
        # Fill and drain multiple times to test wraparound
        for cycle in range(3):
            for i in range(16):
                ring.put(f"cycle{cycle}_msg{i}".encode())
            for i in range(16):
                data = ring.get()
                assert data == f"cycle{cycle}_msg{i}".encode()

    def test_oversized_data_raises(self, ring: RingBuffer):
        max_payload = ring.slot_size - 4  # 4 bytes for length prefix
        with pytest.raises(ValueError, match="exceeds"):
            ring.put(b"x" * (max_payload + 1))

    def test_context_manager(self, ring_name: str):
        with RingBuffer(ring_name, capacity=4, slot_size=128) as rb:
            rb.put(b"test")
            assert rb.get() == b"test"
        rb.unlink()

    def test_slot_size_alignment(self, ring: RingBuffer):
        assert ring.slot_size % 64 == 0  # Cache-line aligned


def _producer(name: str, count: int):
    """Producer function for multi-process test."""
    rb = RingBuffer(name, create=False)
    for i in range(count):
        while True:
            try:
                rb.put(f"msg_{i}".encode())
                break
            except RingBufferFull:
                pass  # Spin wait
    rb.close()


class TestMultiProcess:
    def test_cross_process_communication(self, ring_name: str):
        count = 100
        rb = RingBuffer(ring_name, capacity=32, slot_size=256)

        proc = multiprocessing.Process(target=_producer, args=(ring_name, count))
        proc.start()

        received = []
        while len(received) < count:
            try:
                data = rb.get()
                received.append(data)
            except RingBufferEmpty:
                pass  # Spin wait

        proc.join(timeout=5)
        assert proc.exitcode == 0

        expected = [f"msg_{i}".encode() for i in range(count)]
        assert received == expected

        rb.close()
        rb.unlink()
