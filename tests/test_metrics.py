"""Tests for P99 latency tracker and throughput counters."""

from __future__ import annotations

import threading

from flowstate.ops.metrics import LatencyTracker, MetricsRegistry, ThroughputCounter


class TestLatencyTracker:
    def test_record_and_count(self):
        lt = LatencyTracker("test")
        lt.record(100.0)
        lt.record(200.0)
        assert lt.count == 2

    def test_percentiles(self):
        lt = LatencyTracker("test")
        for i in range(100):
            lt.record(float(i))
        assert lt.p50 == pytest.approx(50.0, abs=2)
        assert lt.p99 >= 95.0

    def test_mean(self):
        lt = LatencyTracker("test")
        lt.record(100.0)
        lt.record(200.0)
        assert lt.mean == 150.0

    def test_snapshot(self):
        lt = LatencyTracker("test")
        lt.record(50.0)
        snap = lt.snapshot()
        assert snap["name"] == "test"
        assert snap["count"] == 1
        assert snap["min_ns"] == 50.0
        assert snap["max_ns"] == 50.0

    def test_empty_percentile(self):
        lt = LatencyTracker("test")
        assert lt.p99 == 0.0
        assert lt.mean == 0.0

    def test_reset(self):
        lt = LatencyTracker("test")
        lt.record(100.0)
        lt.reset()
        assert lt.count == 0
        assert lt.p50 == 0.0

    def test_reservoir_overflow(self):
        lt = LatencyTracker("test", reservoir_size=10)
        for i in range(100):
            lt.record(float(i))
        assert lt.count == 100
        # Reservoir has at most 10 samples
        assert lt.p50 >= 0  # Just verify it doesn't crash

    def test_thread_safety(self):
        lt = LatencyTracker("test")
        errors = []

        def record_many():
            try:
                for i in range(1000):
                    lt.record(float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert lt.count == 4000


class TestThroughputCounter:
    def test_increment(self):
        tc = ThroughputCounter("test")
        tc.increment()
        tc.increment(5)
        assert tc.total == 6

    def test_rate(self):
        tc = ThroughputCounter("test", window_seconds=10.0)
        tc.increment(100)
        rate = tc.rate()
        assert rate > 0

    def test_snapshot(self):
        tc = ThroughputCounter("test")
        tc.increment(42)
        snap = tc.snapshot()
        assert snap["name"] == "test"
        assert snap["total"] == 42

    def test_reset(self):
        tc = ThroughputCounter("test")
        tc.increment(100)
        tc.reset()
        assert tc.total == 0


class TestMetricsRegistry:
    def test_get_or_create_latency(self):
        reg = MetricsRegistry()
        lt1 = reg.latency("ingest")
        lt2 = reg.latency("ingest")
        assert lt1 is lt2

    def test_get_or_create_throughput(self):
        reg = MetricsRegistry()
        tc1 = reg.throughput("messages")
        tc2 = reg.throughput("messages")
        assert tc1 is tc2

    def test_snapshot(self):
        reg = MetricsRegistry()
        reg.latency("ingest").record(100.0)
        reg.throughput("messages").increment(10)

        snap = reg.snapshot()
        assert len(snap["latency"]) == 1
        assert len(snap["throughput"]) == 1
        assert snap["latency"][0]["name"] == "ingest"


import pytest
