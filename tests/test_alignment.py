"""Tests for cache-line aligned allocators."""

from __future__ import annotations

from flowstate.ops.alignment import (
    CACHE_LINE_SIZE,
    AlignedBuffer,
    aligned_size,
    is_aligned,
)


class TestAlignedSize:
    def test_already_aligned(self):
        assert aligned_size(64) == 64

    def test_round_up(self):
        assert aligned_size(65) == 128

    def test_zero(self):
        assert aligned_size(0) == 0

    def test_small(self):
        assert aligned_size(1) == 64

    def test_custom_alignment(self):
        assert aligned_size(100, 32) == 128


class TestIsAligned:
    def test_aligned(self):
        assert is_aligned(128)
        assert is_aligned(0)

    def test_not_aligned(self):
        assert not is_aligned(63)
        assert not is_aligned(65)


class TestAlignedBuffer:
    def test_create_and_size(self):
        with AlignedBuffer(100) as buf:
            assert buf.size >= 100
            assert buf.size % CACHE_LINE_SIZE == 0

    def test_cache_aligned(self):
        with AlignedBuffer(256) as buf:
            assert buf.is_cache_aligned

    def test_write_and_read(self):
        with AlignedBuffer(128) as buf:
            data = b"hello world"
            buf.write(data)
            result = buf.read(len(data))
            assert result == data

    def test_write_at_offset(self):
        with AlignedBuffer(128) as buf:
            buf.write(b"hello", offset=64)
            result = buf.read(5, offset=64)
            assert result == b"hello"

    def test_write_exceeds_raises(self):
        with AlignedBuffer(64) as buf:
            try:
                buf.write(b"x" * 128)
                assert False, "Should have raised"
            except ValueError:
                pass

    def test_context_manager(self):
        buf = AlignedBuffer(64)
        with buf:
            buf.write(b"data")
        # After exit, buffer should be closed
