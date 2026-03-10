"""Arrow-native market data types with nanosecond timestamps."""

from __future__ import annotations

from enum import StrEnum

import pyarrow as pa

# Nanosecond-precision timestamp type used across all schemas
TIMESTAMP_NS = pa.timestamp("ns", tz="UTC")


class MarketDataType(StrEnum):
    """Enumeration of supported market data types."""

    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"


# --- Trade schema ---
TRADE_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.utf8(), nullable=False),
        pa.field("timestamp", TIMESTAMP_NS, nullable=False),
        pa.field("exchange_timestamp", TIMESTAMP_NS, nullable=True),
        pa.field("receive_timestamp", TIMESTAMP_NS, nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("exchange", pa.utf8(), nullable=True),
        pa.field("conditions", pa.list_(pa.utf8()), nullable=True),
        pa.field("tape", pa.utf8(), nullable=True),
        pa.field("sequence", pa.int64(), nullable=True),
        pa.field("trade_id", pa.utf8(), nullable=True),
        pa.field("source", pa.utf8(), nullable=False),
    ],
    metadata={
        b"flowstate.type": b"trade",
        b"flowstate.version": b"1",
    },
)

# --- Quote (NBBO) schema ---
QUOTE_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.utf8(), nullable=False),
        pa.field("timestamp", TIMESTAMP_NS, nullable=False),
        pa.field("exchange_timestamp", TIMESTAMP_NS, nullable=True),
        pa.field("receive_timestamp", TIMESTAMP_NS, nullable=False),
        pa.field("bid_price", pa.float64(), nullable=False),
        pa.field("bid_size", pa.float64(), nullable=False),
        pa.field("ask_price", pa.float64(), nullable=False),
        pa.field("ask_size", pa.float64(), nullable=False),
        pa.field("bid_exchange", pa.utf8(), nullable=True),
        pa.field("ask_exchange", pa.utf8(), nullable=True),
        pa.field("conditions", pa.list_(pa.utf8()), nullable=True),
        pa.field("tape", pa.utf8(), nullable=True),
        pa.field("sequence", pa.int64(), nullable=True),
        pa.field("source", pa.utf8(), nullable=False),
    ],
    metadata={
        b"flowstate.type": b"quote",
        b"flowstate.version": b"1",
    },
)

# --- Aggregate bar schema ---
BAR_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.utf8(), nullable=False),
        pa.field("timestamp", TIMESTAMP_NS, nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("vwap", pa.float64(), nullable=True),
        pa.field("trade_count", pa.int64(), nullable=True),
        pa.field("bar_duration_ns", pa.int64(), nullable=False),
        pa.field("source", pa.utf8(), nullable=False),
    ],
    metadata={
        b"flowstate.type": b"bar",
        b"flowstate.version": b"1",
    },
)

# Lookup table for schema by type
SCHEMAS: dict[MarketDataType, pa.Schema] = {
    MarketDataType.TRADE: TRADE_SCHEMA,
    MarketDataType.QUOTE: QUOTE_SCHEMA,
    MarketDataType.BAR: BAR_SCHEMA,
}


def get_schema(data_type: MarketDataType, *, version: int = 1) -> pa.Schema:
    """Get the canonical Arrow schema for a market data type.

    Args:
        data_type: The market data type.
        version: Schema version (currently only version 1 is supported).

    Returns:
        The PyArrow schema for the requested data type.

    Raises:
        ValueError: If the data type or version is not supported.
    """
    if data_type not in SCHEMAS:
        raise ValueError(f"Unsupported data type: {data_type}")
    schema = SCHEMAS[data_type]
    schema_version = int(schema.metadata.get(b"flowstate.version", b"0"))
    if schema_version != version:
        raise ValueError(f"Schema version {version} not found for {data_type.value}")
    return schema
