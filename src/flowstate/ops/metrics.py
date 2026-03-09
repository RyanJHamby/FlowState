"""P99 latency tracker, throughput counters, and metrics registry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class LatencyTracker:
    """Thread-safe latency tracker with percentile computation.

    Uses a reservoir sampling approach with a fixed-size buffer for memory efficiency.
    """

    def __init__(self, name: str, reservoir_size: int = 10_000) -> None:
        self._name = name
        self._reservoir_size = reservoir_size
        self._lock = threading.Lock()
        self._samples: list[float] = []
        self._count: int = 0
        self._sum: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    @property
    def name(self) -> str:
        return self._name

    @property
    def count(self) -> int:
        return self._count

    def record(self, latency_ns: float) -> None:
        """Record a latency sample in nanoseconds."""
        with self._lock:
            self._count += 1
            self._sum += latency_ns
            self._min = min(self._min, latency_ns)
            self._max = max(self._max, latency_ns)
            if len(self._samples) < self._reservoir_size:
                self._samples.append(latency_ns)
            else:
                # Overwrite oldest in circular fashion
                idx = (self._count - 1) % self._reservoir_size
                self._samples[idx] = latency_ns

    def percentile(self, p: float) -> float:
        """Compute a percentile (0-100) from recorded samples.

        Args:
            p: Percentile value (e.g. 50, 95, 99).

        Returns:
            The latency value at the given percentile, or 0 if no samples.
        """
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_samples = sorted(self._samples)
            idx = int(len(sorted_samples) * p / 100)
            idx = min(idx, len(sorted_samples) - 1)
            return sorted_samples[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def mean(self) -> float:
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0.0

    def snapshot(self) -> dict[str, float]:
        """Return a snapshot of all metrics."""
        return {
            "name": self._name,
            "count": self._count,
            "mean_ns": self.mean,
            "p50_ns": self.p50,
            "p95_ns": self.p95,
            "p99_ns": self.p99,
            "min_ns": self._min if self._count > 0 else 0.0,
            "max_ns": self._max if self._count > 0 else 0.0,
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._count = 0
            self._sum = 0.0
            self._min = float("inf")
            self._max = float("-inf")


class ThroughputCounter:
    """Thread-safe throughput counter with windowed rate computation."""

    def __init__(self, name: str, window_seconds: float = 1.0) -> None:
        self._name = name
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._total: int = 0
        self._window_count: int = 0
        self._window_start: float = time.monotonic()

    @property
    def name(self) -> str:
        return self._name

    @property
    def total(self) -> int:
        return self._total

    def increment(self, count: int = 1) -> None:
        """Increment the counter."""
        with self._lock:
            self._total += count
            self._window_count += count

    def rate(self) -> float:
        """Compute the current rate (events per second)."""
        with self._lock:
            elapsed = time.monotonic() - self._window_start
            if elapsed <= 0:
                return 0.0
            r = self._window_count / elapsed
            if elapsed >= self._window_seconds:
                self._window_count = 0
                self._window_start = time.monotonic()
            return r

    def snapshot(self) -> dict[str, float]:
        return {
            "name": self._name,
            "total": self._total,
            "rate_per_sec": self.rate(),
        }

    def reset(self) -> None:
        with self._lock:
            self._total = 0
            self._window_count = 0
            self._window_start = time.monotonic()


class MetricsRegistry:
    """Central registry for all metrics instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latency: dict[str, LatencyTracker] = {}
        self._throughput: dict[str, ThroughputCounter] = {}

    def latency(self, name: str, **kwargs: int) -> LatencyTracker:
        """Get or create a latency tracker."""
        with self._lock:
            if name not in self._latency:
                self._latency[name] = LatencyTracker(name, **kwargs)
            return self._latency[name]

    def throughput(self, name: str, **kwargs: float) -> ThroughputCounter:
        """Get or create a throughput counter."""
        with self._lock:
            if name not in self._throughput:
                self._throughput[name] = ThroughputCounter(name, **kwargs)
            return self._throughput[name]

    def snapshot(self) -> dict[str, list[dict[str, float]]]:
        """Return a snapshot of all registered metrics."""
        with self._lock:
            return {
                "latency": [t.snapshot() for t in self._latency.values()],
                "throughput": [c.snapshot() for c in self._throughput.values()],
            }
