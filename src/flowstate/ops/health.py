"""Health checks, heartbeats, and operational monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from flowstate.firehose.client import ConnectionState, MarketDataClient
from flowstate.firehose.ring_buffer import RingBuffer
from flowstate.ops.metrics import MetricsRegistry


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    last_check: float = 0.0


@dataclass
class SystemHealth:
    """Aggregate health status for the entire system."""

    components: list[ComponentHealth] = field(default_factory=list)

    @property
    def status(self) -> HealthStatus:
        if not self.components:
            return HealthStatus.HEALTHY
        statuses = [c.status for c in self.components]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Performs health checks on FlowState components."""

    def __init__(self) -> None:
        self._ring_buffers: list[tuple[str, RingBuffer]] = []
        self._clients: list[tuple[str, MarketDataClient]] = []
        self._metrics: MetricsRegistry | None = None
        self._thresholds = {
            "ring_buffer_fill_pct": 90.0,
            "ring_buffer_warn_pct": 70.0,
        }

    def register_ring_buffer(self, name: str, rb: RingBuffer) -> None:
        self._ring_buffers.append((name, rb))

    def register_client(self, name: str, client: MarketDataClient) -> None:
        self._clients.append((name, client))

    def register_metrics(self, metrics: MetricsRegistry) -> None:
        self._metrics = metrics

    def check(self) -> SystemHealth:
        """Run all health checks and return aggregate status."""
        health = SystemHealth()
        now = time.time()

        for name, rb in self._ring_buffers:
            health.components.append(self._check_ring_buffer(name, rb, now))

        for name, client in self._clients:
            health.components.append(self._check_client(name, client, now))

        if self._metrics:
            health.components.append(self._check_metrics(now))

        return health

    def _check_ring_buffer(
        self, name: str, rb: RingBuffer, now: float
    ) -> ComponentHealth:
        fill_pct = (rb.size / rb.capacity * 100) if rb.capacity > 0 else 0

        if fill_pct >= self._thresholds["ring_buffer_fill_pct"]:
            status = HealthStatus.UNHEALTHY
            msg = f"Ring buffer {name} is {fill_pct:.1f}% full"
        elif fill_pct >= self._thresholds["ring_buffer_warn_pct"]:
            status = HealthStatus.DEGRADED
            msg = f"Ring buffer {name} is {fill_pct:.1f}% full"
        else:
            status = HealthStatus.HEALTHY
            msg = f"Ring buffer {name} OK ({fill_pct:.1f}% full)"

        return ComponentHealth(
            name=f"ring_buffer:{name}",
            status=status,
            message=msg,
            details={"fill_pct": fill_pct, "size": rb.size, "capacity": rb.capacity},
            last_check=now,
        )

    def _check_client(
        self, name: str, client: MarketDataClient, now: float
    ) -> ComponentHealth:
        state = client.state

        if state == ConnectionState.CONNECTED:
            status = HealthStatus.HEALTHY
            msg = f"Client {name} connected"
        elif state in (ConnectionState.CONNECTING, ConnectionState.RECONNECTING):
            status = HealthStatus.DEGRADED
            msg = f"Client {name} is {state.value}"
        else:
            status = HealthStatus.UNHEALTHY
            msg = f"Client {name} is {state.value}"

        return ComponentHealth(
            name=f"client:{name}",
            status=status,
            message=msg,
            details={"state": state.value, "subscriptions": len(client.subscriptions)},
            last_check=now,
        )

    def _check_metrics(self, now: float) -> ComponentHealth:
        snap = self._metrics.snapshot() if self._metrics else {}
        return ComponentHealth(
            name="metrics",
            status=HealthStatus.HEALTHY,
            message="Metrics registry OK",
            details=snap,
            last_check=now,
        )
