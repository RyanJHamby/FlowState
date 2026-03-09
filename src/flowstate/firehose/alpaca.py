"""Alpaca Markets WebSocket adapter."""

from __future__ import annotations

import logging
from typing import Any

import orjson

from flowstate.firehose.client import ClientConfig, MarketDataClient

logger = logging.getLogger(__name__)


class AlpacaClient(MarketDataClient):
    """Alpaca Markets WebSocket market data client.

    Implements the Alpaca real-time data protocol including
    authentication, subscriptions, and message parsing.
    """

    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)

    async def _authenticate(self) -> None:
        if self._ws is None:
            return
        key = self._config.api_key
        secret = self._config.extra.get("api_secret", "")
        auth_msg = orjson.dumps({
            "action": "auth",
            "key": key,
            "secret": secret,
        })
        await self._ws.send(auth_msg)
        raw = await self._ws.recv()
        response = orjson.loads(raw)
        msgs = response if isinstance(response, list) else [response]
        for msg in msgs:
            if msg.get("T") == "error":
                raise ConnectionError(f"Alpaca auth failed: {msg.get('msg', '')}")

    async def _subscribe(self, symbols: list[str]) -> None:
        if self._ws is None:
            return
        sub_msg = orjson.dumps({
            "action": "subscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars": symbols,
        })
        await self._ws.send(sub_msg)

    async def _parse_message(self, raw: str | bytes) -> list[dict[str, Any]]:
        if isinstance(raw, str):
            raw = raw.encode()
        data = orjson.loads(raw)
        messages = data if isinstance(data, list) else [data]

        results: list[dict[str, Any]] = []
        for msg in messages:
            msg_type = msg.get("T")
            if msg_type == "t":
                results.append(self._parse_trade(msg))
            elif msg_type == "q":
                results.append(self._parse_quote(msg))
            elif msg_type == "b":
                results.append(self._parse_bar(msg))

        return results

    async def _send_heartbeat(self) -> None:
        if self._ws is not None:
            await self._ws.ping()

    @staticmethod
    def _parse_trade(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": "trade",
            "symbol": msg.get("S", ""),
            "price": msg.get("p", 0.0),
            "size": msg.get("s", 0.0),
            "timestamp": msg.get("t", ""),
            "exchange": msg.get("x", ""),
            "conditions": msg.get("c", []),
            "trade_id": str(msg.get("i", "")),
            "tape": msg.get("z", ""),
            "source": "alpaca",
        }

    @staticmethod
    def _parse_quote(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": "quote",
            "symbol": msg.get("S", ""),
            "bid_price": msg.get("bp", 0.0),
            "bid_size": msg.get("bs", 0.0),
            "ask_price": msg.get("ap", 0.0),
            "ask_size": msg.get("as", 0.0),
            "timestamp": msg.get("t", ""),
            "bid_exchange": msg.get("bx", ""),
            "ask_exchange": msg.get("ax", ""),
            "conditions": msg.get("c", []),
            "tape": msg.get("z", ""),
            "source": "alpaca",
        }

    @staticmethod
    def _parse_bar(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "_type": "bar",
            "symbol": msg.get("S", ""),
            "open": msg.get("o", 0.0),
            "high": msg.get("h", 0.0),
            "low": msg.get("l", 0.0),
            "close": msg.get("c", 0.0),
            "volume": msg.get("v", 0.0),
            "vwap": msg.get("vw"),
            "timestamp": msg.get("t", ""),
            "trade_count": msg.get("n"),
            "source": "alpaca",
        }


def create_alpaca_client(
    api_key: str,
    api_secret: str,
    feed: str = "iex",
) -> AlpacaClient:
    """Create an Alpaca client with default configuration.

    Args:
        api_key: Alpaca API key ID.
        api_secret: Alpaca API secret key.
        feed: Data feed ("iex" or "sip").

    Returns:
        Configured AlpacaClient.
    """
    url = f"wss://stream.data.alpaca.markets/v2/{feed}"
    config = ClientConfig(
        url=url,
        api_key=api_key,
        extra={"api_secret": api_secret},
    )
    return AlpacaClient(config)
