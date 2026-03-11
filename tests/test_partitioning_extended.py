"""Extended tests for Hive partitioning — edge cases, distribution quality."""

from __future__ import annotations

import statistics

from flowstate.storage.partitioning import PartitionKey, PartitionScheme


class TestPartitionKeyEdgeCases:
    def test_zero_bucket(self):
        key = PartitionKey(data_type="quote", date="2024-12-31", bucket=0)
        assert key.path == "type=quote/date=2024-12-31/bucket=0000"

    def test_max_bucket(self):
        key = PartitionKey(data_type="bar", date="2024-01-01", bucket=9999)
        assert key.path == "type=bar/date=2024-01-01/bucket=9999"

    def test_equality(self):
        a = PartitionKey(data_type="trade", date="2024-01-15", bucket=42)
        b = PartitionKey(data_type="trade", date="2024-01-15", bucket=42)
        assert a == b

    def test_hashable(self):
        key = PartitionKey(data_type="trade", date="2024-01-15", bucket=42)
        s = {key}
        assert key in s


class TestBucketDistribution:
    def test_uniform_distribution(self):
        """Buckets should be roughly uniformly distributed for many symbols."""
        scheme = PartitionScheme(num_buckets=32)
        counts = [0] * 32
        for i in range(10000):
            b = scheme.bucket_for(f"SYM_{i:05d}")
            counts[b] += 1

        # Each bucket should get ~312 (10000/32). Allow 50% deviation.
        expected = 10000 / 32
        for c in counts:
            assert c > expected * 0.5, f"Under-represented bucket: {c}"
            assert c < expected * 1.5, f"Over-represented bucket: {c}"

    def test_low_variance(self):
        scheme = PartitionScheme(num_buckets=16)
        counts = [0] * 16
        for i in range(5000):
            counts[scheme.bucket_for(f"TICKER_{i}")] += 1

        cv = statistics.stdev(counts) / statistics.mean(counts)
        assert cv < 0.2, f"Coefficient of variation too high: {cv}"

    def test_single_bucket(self):
        scheme = PartitionScheme(num_buckets=1)
        for sym in ["AAPL", "MSFT", "GOOG"]:
            assert scheme.bucket_for(sym) == 0


class TestTimestampEdgeCases:
    def test_midnight_boundary(self):
        scheme = PartitionScheme(num_buckets=16)
        # Midnight UTC: 2024-01-15 00:00:00
        ts_ns = 1705276800 * 10**9
        key = scheme.partition_key("AAPL", ts_ns, "trade")
        assert key.date == "2024-01-15"

    def test_end_of_day(self):
        scheme = PartitionScheme(num_buckets=16)
        # 2024-01-15 23:59:00 UTC — safely before midnight
        ts_ns = 1705363140 * 10**9
        key = scheme.partition_key("AAPL", ts_ns, "trade")
        assert key.date == "2024-01-15"

    def test_cross_day_boundary(self):
        scheme = PartitionScheme(num_buckets=16)
        # Just before midnight: 2024-01-15 23:59:59
        ts1 = 1705363199 * 10**9
        # Just after midnight: 2024-01-16 00:00:01
        ts2 = 1705363201 * 10**9
        k1 = scheme.partition_key("AAPL", ts1, "trade")
        k2 = scheme.partition_key("AAPL", ts2, "trade")
        assert k1.date == "2024-01-15"
        assert k2.date == "2024-01-16"

    def test_different_data_types(self):
        scheme = PartitionScheme(num_buckets=16)
        ts_ns = 1705320000 * 10**9
        trade_key = scheme.partition_key("AAPL", ts_ns, "trade")
        quote_key = scheme.partition_key("AAPL", ts_ns, "quote")
        bar_key = scheme.partition_key("AAPL", ts_ns, "bar")
        assert trade_key.data_type == "trade"
        assert quote_key.data_type == "quote"
        assert bar_key.data_type == "bar"
        # Same symbol + timestamp → same bucket
        assert trade_key.bucket == quote_key.bucket == bar_key.bucket


class TestPartitionPath:
    def test_path_with_base(self):
        scheme = PartitionScheme(num_buckets=16)
        ts_ns = 1705320000 * 10**9
        path = scheme.partition_path("AAPL", ts_ns, "trade", base="/data/market")
        parts = str(path).split("/")
        assert "type=trade" in parts
        assert any(p.startswith("date=") for p in parts)
        assert any(p.startswith("bucket=") for p in parts)

    def test_path_without_base(self):
        scheme = PartitionScheme(num_buckets=16)
        ts_ns = 1705320000 * 10**9
        path = scheme.partition_path("MSFT", ts_ns, "quote")
        assert str(path).startswith("type=quote")
