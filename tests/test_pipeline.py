"""Tests for the ingestion pipeline orchestrator."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import orjson
import pytest

from flowstate.firehose.client import ClientConfig, MarketDataClient
from flowstate.firehose.pipeline import IngestionPipeline
from flowstate.firehose.ring_buffer import RingBuffer


class FakeClient(MarketDataClient):
    """Fake client that yields pre-loaded messages."""

    def __init__(self, messages: list[dict[str, Any]]):
        super().__init__(ClientConfig(url="wss://fake"))
        self._messages = messages
        self._started = False

    async def _authenticate(self) -> None:
        pass

    async def _subscribe(self, symbols: list[str]) -> None:
        pass

    async def _parse_message(self, raw: str | bytes) -> list[dict[str, Any]]:
        return []

    async def _send_heartbeat(self) -> None:
        pass

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def messages(self):
        for msg in self._messages:
            yield msg


class TestIngestionPipeline:
    @pytest.fixture
    def ring_buffer(self):
        name = f"test_pipe_{uuid.uuid4().hex[:8]}"
        rb = RingBuffer(name, capacity=64, slot_size=1024)
        yield rb
        rb.close()
        rb.unlink()

    def test_stats_initial(self, ring_buffer: RingBuffer):
        pipeline = IngestionPipeline(
            clients=[],
            normalizers={},
            ring_buffer=ring_buffer,
        )
        assert pipeline.stats.messages_received == 0
        assert pipeline.stats.messages_normalized == 0

    @pytest.mark.asyncio
    async def test_process_message(self, ring_buffer: RingBuffer):
        messages = [
            {"symbol": "AAPL", "price": 185.5, "sequence": 1, "source": "test"},
            {"symbol": "AAPL", "price": 185.6, "sequence": 2, "source": "test"},
        ]
        client = FakeClient(messages)
        pipeline = IngestionPipeline(
            clients=[client],
            normalizers={},
            ring_buffer=ring_buffer,
        )

        await pipeline.start()
        # Give async tasks time to process
        import asyncio
        await asyncio.sleep(0.1)
        await pipeline.stop()

        assert pipeline.stats.messages_received == 2
        assert pipeline.stats.messages_normalized == 2
        assert ring_buffer.size == 2

    @pytest.mark.asyncio
    async def test_sequence_gap_tracking(self, ring_buffer: RingBuffer):
        messages = [
            {"symbol": "AAPL", "price": 185.5, "sequence": 1, "source": "test"},
            {"symbol": "AAPL", "price": 185.6, "sequence": 5, "source": "test"},
        ]
        client = FakeClient(messages)
        pipeline = IngestionPipeline(
            clients=[client],
            normalizers={},
            ring_buffer=ring_buffer,
        )

        await pipeline.start()
        import asyncio
        await asyncio.sleep(0.1)
        await pipeline.stop()

        assert pipeline.stats.sequence_gaps == 1
