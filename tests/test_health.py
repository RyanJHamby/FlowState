"""Tests for health checks and monitoring."""

from __future__ import annotations

import uuid

import pytest

from flowstate.firehose.ring_buffer import RingBuffer
from flowstate.ops.health import HealthChecker, HealthStatus
from flowstate.ops.metrics import MetricsRegistry

# Reuse the FakeClient from pipeline tests
from tests.test_pipeline import FakeClient


@pytest.fixture
def ring_buffer():
    name = f"test_health_{uuid.uuid4().hex[:8]}"
    rb = RingBuffer(name, capacity=100, slot_size=256)
    yield rb
    rb.close()
    rb.unlink()


class TestHealthChecker:
    def test_empty_is_healthy(self):
        checker = HealthChecker()
        health = checker.check()
        assert health.is_healthy
        assert health.status == HealthStatus.HEALTHY

    def test_ring_buffer_healthy(self, ring_buffer: RingBuffer):
        checker = HealthChecker()
        checker.register_ring_buffer("main", ring_buffer)
        health = checker.check()
        assert health.is_healthy

    def test_ring_buffer_degraded(self, ring_buffer: RingBuffer):
        # Fill to 75%
        for i in range(75):
            ring_buffer.put(f"msg_{i}".encode())

        checker = HealthChecker()
        checker.register_ring_buffer("main", ring_buffer)
        health = checker.check()
        assert health.status == HealthStatus.DEGRADED

    def test_ring_buffer_unhealthy(self, ring_buffer: RingBuffer):
        # Fill to 95%
        for i in range(95):
            ring_buffer.put(f"msg_{i}".encode())

        checker = HealthChecker()
        checker.register_ring_buffer("main", ring_buffer)
        health = checker.check()
        assert health.status == HealthStatus.UNHEALTHY

    def test_client_disconnected(self):
        client = FakeClient([])
        checker = HealthChecker()
        checker.register_client("test", client)
        health = checker.check()
        assert health.status == HealthStatus.UNHEALTHY

    def test_metrics_check(self):
        metrics = MetricsRegistry()
        metrics.latency("test").record(100.0)
        checker = HealthChecker()
        checker.register_metrics(metrics)
        health = checker.check()
        assert health.is_healthy
        assert any(c.name == "metrics" for c in health.components)

    def test_to_dict(self, ring_buffer: RingBuffer):
        checker = HealthChecker()
        checker.register_ring_buffer("main", ring_buffer)
        health = checker.check()
        d = health.to_dict()
        assert d["status"] == "healthy"
        assert len(d["components"]) == 1
        assert d["components"][0]["name"] == "ring_buffer:main"
