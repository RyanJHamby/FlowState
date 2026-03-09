"""Tests for the base async WebSocket client."""

from __future__ import annotations

from typing import Any

import pytest

from flowstate.firehose.client import (
    ClientConfig,
    ConnectionState,
    MarketDataClient,
)


class MockClient(MarketDataClient):
    """Concrete implementation for testing."""

    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.auth_called = False
        self.subscribed: list[str] = []
        self.heartbeats_sent = 0

    async def _authenticate(self) -> None:
        self.auth_called = True

    async def _subscribe(self, symbols: list[str]) -> None:
        self.subscribed.extend(symbols)

    async def _parse_message(self, raw: str | bytes) -> list[dict[str, Any]]:
        return [{"data": raw}]

    async def _send_heartbeat(self) -> None:
        self.heartbeats_sent += 1


class TestClientConfig:
    def test_defaults(self):
        config = ClientConfig(url="wss://example.com")
        assert config.url == "wss://example.com"
        assert config.reconnect_delay == 1.0
        assert config.max_reconnect_delay == 60.0
        assert config.heartbeat_interval == 30.0

    def test_custom(self):
        config = ClientConfig(
            url="wss://example.com",
            api_key="test_key",
            reconnect_delay=2.0,
        )
        assert config.api_key == "test_key"
        assert config.reconnect_delay == 2.0


class TestMarketDataClient:
    def test_initial_state(self):
        config = ClientConfig(url="wss://example.com")
        client = MockClient(config)
        assert client.state == ConnectionState.DISCONNECTED
        assert client.subscriptions == set()

    def test_config_accessible(self):
        config = ClientConfig(url="wss://example.com")
        client = MockClient(config)
        assert client.config is config

    @pytest.mark.asyncio
    async def test_subscribe_stores_symbols(self):
        config = ClientConfig(url="wss://example.com")
        client = MockClient(config)
        await client.subscribe(["AAPL", "MSFT"])
        assert client.subscriptions == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_symbols(self):
        config = ClientConfig(url="wss://example.com")
        client = MockClient(config)
        await client.subscribe(["AAPL", "MSFT", "GOOG"])
        await client.unsubscribe(["MSFT"])
        assert client.subscriptions == {"AAPL", "GOOG"}

    @pytest.mark.asyncio
    async def test_stop_sets_closed(self):
        config = ClientConfig(url="wss://example.com")
        client = MockClient(config)
        await client.stop()
        assert client.state == ConnectionState.CLOSED
