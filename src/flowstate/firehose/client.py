"""Base async WebSocket client with reconnection and heartbeat."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class ClientConfig:
    """Configuration for a WebSocket market data client."""

    url: str
    api_key: str = ""
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    reconnect_backoff: float = 2.0
    heartbeat_interval: float = 30.0
    max_reconnect_attempts: int = 0  # 0 = unlimited
    extra: dict[str, Any] = field(default_factory=dict)


class MarketDataClient(ABC):
    """Abstract base class for async WebSocket market data clients.

    Provides reconnection logic, heartbeat management, and a message iterator.
    Subclasses implement protocol-specific authentication and message parsing.
    """

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._ws: Any = None
        self._reconnect_attempts = 0
        self._subscriptions: set[str] = set()
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def config(self) -> ClientConfig:
        return self._config

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def subscriptions(self) -> set[str]:
        return set(self._subscriptions)

    @abstractmethod
    async def _authenticate(self) -> None:
        """Perform protocol-specific authentication after connection."""

    @abstractmethod
    async def _subscribe(self, symbols: list[str]) -> None:
        """Send protocol-specific subscription messages."""

    @abstractmethod
    async def _parse_message(self, raw: str | bytes) -> list[dict[str, Any]]:
        """Parse a raw WebSocket message into normalized dicts."""

    @abstractmethod
    async def _send_heartbeat(self) -> None:
        """Send a protocol-specific heartbeat/ping."""

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        import websockets

        self._state = ConnectionState.CONNECTING
        self._ws = await websockets.connect(self._config.url)
        self._state = ConnectionState.CONNECTED
        self._reconnect_attempts = 0
        await self._authenticate()
        if self._subscriptions:
            await self._subscribe(list(self._subscriptions))

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for the given symbols."""
        self._subscriptions.update(symbols)
        if self._state == ConnectionState.CONNECTED:
            await self._subscribe(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from market data for the given symbols."""
        self._subscriptions -= set(symbols)

    async def start(self) -> None:
        """Start the client's receive and heartbeat loops."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._receive_loop()),
            asyncio.create_task(self._heartbeat_loop()),
        ]

    async def stop(self) -> None:
        """Stop the client and close the connection."""
        self._running = False
        self._state = ConnectionState.CLOSED
        for task in self._tasks:
            task.cancel()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        """Async iterator over received messages."""
        while self._running or not self._message_queue.empty():
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                yield msg
            except asyncio.TimeoutError:
                continue

    async def _receive_loop(self) -> None:
        """Main receive loop with automatic reconnection."""
        while self._running:
            try:
                if self._ws is None or self._state != ConnectionState.CONNECTED:
                    await self._reconnect()
                    continue

                raw = await self._ws.recv()
                parsed = await self._parse_message(raw)
                for msg in parsed:
                    await self._message_queue.put(msg)

            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"Receive error: {e}, will reconnect")
                self._state = ConnectionState.RECONNECTING
                await self._reconnect()

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat sender."""
        while self._running:
            await asyncio.sleep(self._config.heartbeat_interval)
            if self._state == ConnectionState.CONNECTED and self._ws is not None:
                try:
                    await self._send_heartbeat()
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if self._config.max_reconnect_attempts > 0:
            if self._reconnect_attempts >= self._config.max_reconnect_attempts:
                logger.error("Max reconnect attempts reached")
                self._running = False
                return

        self._reconnect_attempts += 1
        delay = min(
            self._config.reconnect_delay * (self._config.reconnect_backoff ** (self._reconnect_attempts - 1)),
            self._config.max_reconnect_delay,
        )
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
        except Exception as e:
            logger.warning(f"Reconnect failed: {e}")
