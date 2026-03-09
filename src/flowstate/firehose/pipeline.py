"""Ingestion pipeline orchestrator: client -> normalize -> validate -> ring buffer."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import orjson

from flowstate.firehose.client import MarketDataClient
from flowstate.firehose.ring_buffer import RingBuffer, RingBufferFull
from flowstate.ops.metrics import MetricsRegistry
from flowstate.schema.normalization import Normalizer
from flowstate.schema.validation import SequenceTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Runtime statistics for the ingestion pipeline."""

    messages_received: int = 0
    messages_normalized: int = 0
    messages_dropped: int = 0
    sequence_gaps: int = 0
    buffer_overflows: int = 0


class IngestionPipeline:
    """Orchestrates the flow: WebSocket client -> normalization -> validation -> ring buffer.

    Reads messages from one or more MarketDataClients, normalizes them using
    the configured Normalizer, validates sequences, and writes to a RingBuffer.
    """

    def __init__(
        self,
        clients: list[MarketDataClient],
        normalizers: dict[str, Normalizer],
        ring_buffer: RingBuffer,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._clients = clients
        self._normalizers = normalizers
        self._ring_buffer = ring_buffer
        self._metrics = metrics or MetricsRegistry()
        self._sequence_tracker = SequenceTracker()
        self._stats = PipelineStats()
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    @property
    def sequence_tracker(self) -> SequenceTracker:
        return self._sequence_tracker

    async def start(self) -> None:
        """Start the ingestion pipeline."""
        self._running = True
        for client in self._clients:
            await client.start()
            task = asyncio.create_task(self._process_client(client))
            self._tasks.append(task)

    async def stop(self) -> None:
        """Stop the ingestion pipeline."""
        self._running = False
        for client in self._clients:
            await client.stop()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def _process_client(self, client: MarketDataClient) -> None:
        """Process messages from a single client."""
        latency = self._metrics.latency("ingest_latency")
        throughput = self._metrics.throughput("ingest_throughput")

        async for msg in client.messages():
            if not self._running:
                break

            start_ns = time.time_ns()
            self._stats.messages_received += 1

            try:
                processed = self._process_message(msg)
                if processed is not None:
                    serialized = orjson.dumps(processed)
                    self._ring_buffer.put(serialized)
                    self._stats.messages_normalized += 1
                    throughput.increment()
            except RingBufferFull:
                self._stats.buffer_overflows += 1
                self._stats.messages_dropped += 1
                logger.warning("Ring buffer full, dropping message")
            except Exception as e:
                self._stats.messages_dropped += 1
                logger.error(f"Pipeline error: {e}")

            latency.record(float(time.time_ns() - start_ns))

    def _process_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize and validate a single message."""
        source = msg.get("source", "")
        normalizer = self._normalizers.get(source)

        if normalizer is not None:
            msg = normalizer.normalize(msg)

        # Track sequences if available
        symbol = msg.get("symbol")
        sequence = msg.get("sequence")
        if symbol and sequence is not None:
            gap = self._sequence_tracker.track(symbol, sequence)
            if gap is not None:
                self._stats.sequence_gaps += 1
                logger.warning(
                    f"Sequence gap for {gap.symbol}: expected {gap.expected}, got {gap.actual}"
                )

        return msg
