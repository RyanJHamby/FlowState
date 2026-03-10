"""Tests for Arrow-native market data types."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flowstate.schema.types import (
    BAR_SCHEMA,
    QUOTE_SCHEMA,
    SCHEMAS,
    TIMESTAMP_NS,
    TRADE_SCHEMA,
    MarketDataType,
    get_schema,
)


class TestTimestampType:
    def test_nanosecond_precision(self):
        assert pa.timestamp("ns", tz="UTC") == TIMESTAMP_NS

    def test_utc_timezone(self):
        assert TIMESTAMP_NS.tz == "UTC"


class TestTradeSchema:
    def test_has_required_fields(self):
        names = TRADE_SCHEMA.names
        assert "symbol" in names
        assert "timestamp" in names
        assert "price" in names
        assert "size" in names
        assert "source" in names

    def test_timestamp_fields_are_nanosecond(self):
        assert TRADE_SCHEMA.field("timestamp").type == TIMESTAMP_NS
        assert TRADE_SCHEMA.field("exchange_timestamp").type == TIMESTAMP_NS
        assert TRADE_SCHEMA.field("receive_timestamp").type == TIMESTAMP_NS

    def test_symbol_not_nullable(self):
        assert not TRADE_SCHEMA.field("symbol").nullable

    def test_metadata(self):
        assert TRADE_SCHEMA.metadata[b"flowstate.type"] == b"trade"
        assert TRADE_SCHEMA.metadata[b"flowstate.version"] == b"1"

    def test_conditions_is_list_of_strings(self):
        assert TRADE_SCHEMA.field("conditions").type == pa.list_(pa.utf8())


class TestQuoteSchema:
    def test_has_bid_ask_fields(self):
        names = QUOTE_SCHEMA.names
        assert "bid_price" in names
        assert "bid_size" in names
        assert "ask_price" in names
        assert "ask_size" in names

    def test_metadata(self):
        assert QUOTE_SCHEMA.metadata[b"flowstate.type"] == b"quote"


class TestBarSchema:
    def test_has_ohlcv_fields(self):
        names = BAR_SCHEMA.names
        for field in ["open", "high", "low", "close", "volume"]:
            assert field in names

    def test_has_bar_duration(self):
        assert BAR_SCHEMA.field("bar_duration_ns").type == pa.int64()

    def test_metadata(self):
        assert BAR_SCHEMA.metadata[b"flowstate.type"] == b"bar"


class TestGetSchema:
    def test_get_trade_schema(self):
        schema = get_schema(MarketDataType.TRADE)
        assert schema == TRADE_SCHEMA

    def test_get_quote_schema(self):
        schema = get_schema(MarketDataType.QUOTE)
        assert schema == QUOTE_SCHEMA

    def test_get_bar_schema(self):
        schema = get_schema(MarketDataType.BAR)
        assert schema == BAR_SCHEMA

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            get_schema(MarketDataType.TRADE, version=99)


class TestSchemaLookup:
    def test_all_types_have_schemas(self):
        for dt in MarketDataType:
            assert dt in SCHEMAS

    def test_schemas_are_valid_arrow_schemas(self):
        for schema in SCHEMAS.values():
            assert isinstance(schema, pa.Schema)
