"""Polygon.io WebSocket adapter."""

from __future__ import annotations

import logging
from typing import Any

import orjson

from flowstate.firehose.client import ClientConfig, MarketDataClient

logger = logging.getLogger(__name__)


class PolygonClient(MarketDataClient):
    """Polygon.io WebSocket market data client.

    Implements the Polygon.io real-time WebSocket protocol including
    authentication, subscriptions, and message parsing for trades/quotes/bars.
    """

    TRADE_EVENT = "T"
    QUOTE_EVENT = "Q"
    BAR_EVENT = "A"

    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)

    async def _authenticate(self) -> None:
        if self._ws is None:
            return
        auth_msg = orjson.dumps({"action": "auth", "params": self._config.api_key})
        await self._ws.send(auth_msg)
        # Wait for auth response
        raw = await self._ws.recv()
        response = orjson.loads(raw)
        msgs = response if isinstance(response, list) else [response]
        for msg in msgs:
            if msg.get("status") == "auth_failed":
                raise ConnectionError(f"Polygon auth failed: {msg.get('message', '')}")

    async def _subscribe(self, symbols: list[str]) -> None:
        if self._ws is None:
            return
        # Subscribe to trades, quotes, and bars for all symbols
        params = ",".join(
            f"{prefix}.{sym}"
            for sym in symbols
            for prefix in (self.TRADE_EVENT, self.QUOTE_EVENT, self.BAR_EVENT)
        )
        sub_msg = orjson.dumps({"action": "subscribe", "params": params})
        await self._ws.send(sub_msg)

    async def _parse_message(self, raw: str | bytes) -> list[dict[str, Any]]:
        if isinstance(raw, str):
            raw = raw.encode()
        data = orjson.loads(raw)
        messages = data if isinstance(data, list) else [data]

        results: list[dict[str, Any]] = []
        for msg in messages:
            ev = msg.get("ev")
            if ev == self.TRADE_EVENT:
                results.append(self._parse_trade(msg))
            elif ev == self.QUOTE_EVENT:
                results.append(self._parse_quote(msg))
            elif ev == self.BAR_EVENT:
                results.append(self._parse_bar(msg))
            # Skip status/control messages

        return results

    async def _send_heartbeat(self) -> None:
        # Polygon uses WebSocket pings, handled by the websockets library
        if self._ws is not None:
            await self._ws.ping()

    @staticmethod
    def _parse_trade(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": "trade",
            "symbol": msg.get("sym", ""),
            "price": msg.get("p", 0.0),
            "size": msg.get("s", 0.0),
            "timestamp": msg.get("t", 0),
            "exchange": str(msg.get("x", "")),
            "conditions": msg.get("c", []),
            "trade_id": msg.get("i", ""),
            "tape": str(msg.get("z", "")),
            "sequence": msg.get("q"),
            "source": "polygon",
        }

    @staticmethod
    def _parse_quote(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": "quote",
            "symbol": msg.get("sym", ""),
            "bid_price": msg.get("bp", 0.0),
            "bid_size": msg.get("bs", 0.0),
            "ask_price": msg.get("ap", 0.0),
            "ask_size": msg.get("as", 0.0),
            "timestamp": msg.get("t", 0),
            "bid_exchange": str(msg.get("bx", "")),
            "ask_exchange": str(msg.get("ax", "")),
            "conditions": msg.get("c", []),
            "tape": str(msg.get("z", "")),
            "sequence": msg.get("q"),
            "source": "polygon",
        }

    @staticmethod
    def _parse_bar(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": "bar",
            "symbol": msg.get("sym", ""),
            "open": msg.get("o", 0.0),
            "high": msg.get("h", 0.0),
            "low": msg.get("l", 0.0),
            "close": msg.get("c", 0.0),
            "volume": msg.get("v", 0.0),
            "vwap": msg.get("vw"),
            "timestamp": msg.get("s", 0),
            "trade_count": msg.get("n"),
            "source": "polygon",
        }


def create_polygon_client(
    api_key: str,
    cluster: str = "stocks",
) -> PolygonClient:
    """Create a Polygon.io client with default configuration.

    Args:
        api_key: Polygon.io API key.
        cluster: Market cluster ("stocks", "options", "forex", "crypto").

    Returns:
        Configured PolygonClient.
    """
    url = f"wss://socket.polygon.io/{cluster}"
    config = ClientConfig(url=url, api_key=api_key)
    return PolygonClient(config)
