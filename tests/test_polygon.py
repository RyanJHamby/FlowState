"""Tests for Polygon.io adapter."""

from __future__ import annotations

import pytest

from flowstate.firehose.client import ClientConfig
from flowstate.firehose.polygon import PolygonClient, create_polygon_client


@pytest.fixture
def polygon_client() -> PolygonClient:
    config = ClientConfig(url="wss://socket.polygon.io/stocks", api_key="test_key")
    return PolygonClient(config)


class TestPolygonParsing:
    @pytest.mark.asyncio
    async def test_parse_trade(self, polygon_client: PolygonClient):
        raw = b'[{"ev":"T","sym":"AAPL","p":185.5,"s":100,"t":1700000000000,"x":"4","c":[" ","12"],"i":"12345","z":"3","q":1}]'
        results = await polygon_client._parse_message(raw)
        assert len(results) == 1
        trade = results[0]
        assert trade["_type"] == "trade"
        assert trade["symbol"] == "AAPL"
        assert trade["price"] == 185.5
        assert trade["size"] == 100
        assert trade["source"] == "polygon"

    @pytest.mark.asyncio
    async def test_parse_quote(self, polygon_client: PolygonClient):
        raw = b'[{"ev":"Q","sym":"AAPL","bp":185.4,"bs":200,"ap":185.6,"as":300,"t":1700000000000,"bx":"4","ax":"7"}]'
        results = await polygon_client._parse_message(raw)
        assert len(results) == 1
        quote = results[0]
        assert quote["_type"] == "quote"
        assert quote["bid_price"] == 185.4
        assert quote["ask_price"] == 185.6

    @pytest.mark.asyncio
    async def test_parse_bar(self, polygon_client: PolygonClient):
        raw = b'[{"ev":"A","sym":"AAPL","o":185.0,"h":186.0,"l":184.0,"c":185.5,"v":10000,"vw":185.25,"s":1700000000000,"n":50}]'
        results = await polygon_client._parse_message(raw)
        assert len(results) == 1
        bar = results[0]
        assert bar["_type"] == "bar"
        assert bar["open"] == 185.0
        assert bar["close"] == 185.5
        assert bar["volume"] == 10000

    @pytest.mark.asyncio
    async def test_parse_skips_status(self, polygon_client: PolygonClient):
        raw = b'[{"ev":"status","status":"connected","message":"Connected Successfully"}]'
        results = await polygon_client._parse_message(raw)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_parse_multiple_events(self, polygon_client: PolygonClient):
        raw = b'[{"ev":"T","sym":"AAPL","p":185.5,"s":100,"t":1700000000000},{"ev":"T","sym":"MSFT","p":370.0,"s":50,"t":1700000000001}]'
        results = await polygon_client._parse_message(raw)
        assert len(results) == 2
        assert results[0]["symbol"] == "AAPL"
        assert results[1]["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_parse_string_input(self, polygon_client: PolygonClient):
        raw = '[{"ev":"T","sym":"AAPL","p":185.5,"s":100,"t":1700000000000}]'
        results = await polygon_client._parse_message(raw)
        assert len(results) == 1


class TestCreatePolygonClient:
    def test_default_cluster(self):
        client = create_polygon_client("test_key")
        assert client.config.url == "wss://socket.polygon.io/stocks"
        assert client.config.api_key == "test_key"

    def test_crypto_cluster(self):
        client = create_polygon_client("test_key", cluster="crypto")
        assert client.config.url == "wss://socket.polygon.io/crypto"
