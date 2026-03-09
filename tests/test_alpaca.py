"""Tests for Alpaca adapter."""

from __future__ import annotations

import pytest

from flowstate.firehose.alpaca import AlpacaClient, create_alpaca_client
from flowstate.firehose.client import ClientConfig


@pytest.fixture
def alpaca_client() -> AlpacaClient:
    config = ClientConfig(
        url="wss://stream.data.alpaca.markets/v2/iex",
        api_key="test_key",
        extra={"api_secret": "test_secret"},
    )
    return AlpacaClient(config)


class TestAlpacaParsing:
    @pytest.mark.asyncio
    async def test_parse_trade(self, alpaca_client: AlpacaClient):
        raw = b'[{"T":"t","S":"AAPL","p":185.5,"s":100,"t":"2024-01-01T10:00:00Z","x":"V","c":["@"],"i":12345,"z":"C"}]'
        results = await alpaca_client._parse_message(raw)
        assert len(results) == 1
        trade = results[0]
        assert trade["_type"] == "trade"
        assert trade["symbol"] == "AAPL"
        assert trade["price"] == 185.5
        assert trade["source"] == "alpaca"

    @pytest.mark.asyncio
    async def test_parse_quote(self, alpaca_client: AlpacaClient):
        raw = b'[{"T":"q","S":"AAPL","bp":185.4,"bs":200,"ap":185.6,"as":300,"t":"2024-01-01T10:00:00Z","bx":"V","ax":"N"}]'
        results = await alpaca_client._parse_message(raw)
        assert len(results) == 1
        quote = results[0]
        assert quote["_type"] == "quote"
        assert quote["bid_price"] == 185.4
        assert quote["ask_price"] == 185.6

    @pytest.mark.asyncio
    async def test_parse_bar(self, alpaca_client: AlpacaClient):
        raw = b'[{"T":"b","S":"AAPL","o":185.0,"h":186.0,"l":184.0,"c":185.5,"v":10000,"vw":185.25,"t":"2024-01-01T10:00:00Z","n":50}]'
        results = await alpaca_client._parse_message(raw)
        assert len(results) == 1
        bar = results[0]
        assert bar["_type"] == "bar"
        assert bar["open"] == 185.0

    @pytest.mark.asyncio
    async def test_parse_skips_control(self, alpaca_client: AlpacaClient):
        raw = b'[{"T":"success","msg":"connected"}]'
        results = await alpaca_client._parse_message(raw)
        assert len(results) == 0


class TestCreateAlpacaClient:
    def test_default_feed(self):
        client = create_alpaca_client("key", "secret")
        assert client.config.url == "wss://stream.data.alpaca.markets/v2/iex"

    def test_sip_feed(self):
        client = create_alpaca_client("key", "secret", feed="sip")
        assert client.config.url == "wss://stream.data.alpaca.markets/v2/sip"

    def test_credentials_stored(self):
        client = create_alpaca_client("key", "secret")
        assert client.config.api_key == "key"
        assert client.config.extra["api_secret"] == "secret"
