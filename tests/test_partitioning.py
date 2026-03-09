"""Tests for deterministic Hive partitioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from flowstate.storage.partitioning import PartitionKey, PartitionScheme


class TestPartitionKey:
    def test_path(self):
        key = PartitionKey(data_type="trade", date="2024-01-15", bucket=42)
        assert key.path == "type=trade/date=2024-01-15/bucket=0042"

    def test_full_path(self):
        key = PartitionKey(data_type="trade", date="2024-01-15", bucket=42)
        full = key.full_path("/data")
        assert full == Path("/data/type=trade/date=2024-01-15/bucket=0042")

    def test_frozen(self):
        key = PartitionKey(data_type="trade", date="2024-01-15", bucket=42)
        with pytest.raises(AttributeError):
            key.bucket = 99


class TestPartitionScheme:
    def test_deterministic_bucketing(self):
        scheme = PartitionScheme(num_buckets=256)
        b1 = scheme.bucket_for("AAPL")
        b2 = scheme.bucket_for("AAPL")
        assert b1 == b2

    def test_different_symbols_distribute(self):
        scheme = PartitionScheme(num_buckets=256)
        buckets = {scheme.bucket_for(f"SYM{i}") for i in range(100)}
        # With 100 symbols and 256 buckets, we should get reasonable distribution
        assert len(buckets) > 50

    def test_bucket_range(self):
        scheme = PartitionScheme(num_buckets=16)
        for i in range(100):
            b = scheme.bucket_for(f"SYM{i}")
            assert 0 <= b < 16

    def test_partition_key(self):
        scheme = PartitionScheme(num_buckets=256)
        # 2024-01-15 12:00:00 UTC in nanoseconds
        ts_ns = 1705320000 * 10**9
        key = scheme.partition_key("AAPL", ts_ns, "trade")
        assert key.data_type == "trade"
        assert key.date == "2024-01-15"
        assert 0 <= key.bucket < 256

    def test_partition_path(self):
        scheme = PartitionScheme(num_buckets=256)
        ts_ns = 1705320000 * 10**9
        path = scheme.partition_path("AAPL", ts_ns, "trade", base="/data")
        assert str(path).startswith("/data/type=trade/date=2024-01-15/bucket=")

    def test_invalid_num_buckets(self):
        with pytest.raises(ValueError):
            PartitionScheme(num_buckets=0)

    def test_num_buckets_property(self):
        scheme = PartitionScheme(num_buckets=128)
        assert scheme.num_buckets == 128

    def test_hot_symbol_prevention(self):
        """High-volume symbols should not cluster in the same bucket."""
        scheme = PartitionScheme(num_buckets=64)
        hot_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "SPY"]
        buckets = [scheme.bucket_for(s) for s in hot_symbols]
        unique_buckets = len(set(buckets))
        # At minimum, these popular symbols should spread across multiple buckets
        assert unique_buckets >= 4
