"""Tests for zero-copy normalization and A/B line arbitration."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flowstate.schema.normalization import (
    ABLineArbiter,
    FieldMapping,
    NormalizationProfile,
    Normalizer,
    _convert_timestamp,
)
from flowstate.schema.types import MarketDataType


@pytest.fixture
def trade_profile() -> NormalizationProfile:
    return NormalizationProfile(
        source_name="test_vendor",
        data_type=MarketDataType.TRADE,
        field_mappings=[
            FieldMapping(source="sym", target="symbol"),
            FieldMapping(source="t", target="timestamp"),
            FieldMapping(source="p", target="price"),
            FieldMapping(source="s", target="size"),
            FieldMapping(source="x", target="exchange"),
        ],
        timestamp_unit="ms",
        defaults={"source": "test_vendor"},
    )


@pytest.fixture
def raw_trade() -> dict:
    return {
        "sym": "AAPL",
        "t": 1700000000000,  # milliseconds
        "p": 185.50,
        "s": 100.0,
        "x": "XNAS",
    }


class TestConvertTimestamp:
    def test_milliseconds(self):
        assert _convert_timestamp(1000, "ms") == 1000 * 10**6

    def test_seconds(self):
        assert _convert_timestamp(1, "s") == 10**9

    def test_microseconds(self):
        assert _convert_timestamp(1000, "us") == 1000 * 10**3

    def test_nanoseconds(self):
        assert _convert_timestamp(1000, "ns") == 1000


class TestNormalizer:
    def test_normalize_single(self, trade_profile: NormalizationProfile, raw_trade: dict):
        normalizer = Normalizer(trade_profile)
        result = normalizer.normalize(raw_trade)

        assert result["symbol"] == "AAPL"
        assert result["price"] == 185.50
        assert result["size"] == 100.0
        assert result["exchange"] == "XNAS"
        assert result["source"] == "test_vendor"

    def test_timestamp_conversion(self, trade_profile: NormalizationProfile, raw_trade: dict):
        normalizer = Normalizer(trade_profile)
        result = normalizer.normalize(raw_trade)

        expected_ns = 1700000000000 * 10**6
        assert result["timestamp"] == expected_ns

    def test_receive_timestamp_auto(self, trade_profile: NormalizationProfile, raw_trade: dict):
        normalizer = Normalizer(trade_profile)
        result = normalizer.normalize(raw_trade)

        assert result["receive_timestamp"] is not None
        assert isinstance(result["receive_timestamp"], int)
        assert result["receive_timestamp"] > 0

    def test_missing_fields_are_none(self, trade_profile: NormalizationProfile, raw_trade: dict):
        normalizer = Normalizer(trade_profile)
        result = normalizer.normalize(raw_trade)

        assert result["conditions"] is None
        assert result["tape"] is None

    def test_normalize_batch(self, trade_profile: NormalizationProfile, raw_trade: dict):
        normalizer = Normalizer(trade_profile)
        batch = normalizer.normalize_batch([raw_trade, raw_trade])

        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 2
        assert batch.column("symbol").to_pylist() == ["AAPL", "AAPL"]

    def test_field_transform(self):
        profile = NormalizationProfile(
            source_name="test",
            data_type=MarketDataType.TRADE,
            field_mappings=[
                FieldMapping(
                    source="sym",
                    target="symbol",
                    transform=lambda x: x.upper(),
                ),
                FieldMapping(source="t", target="timestamp"),
                FieldMapping(source="p", target="price"),
                FieldMapping(source="s", target="size"),
            ],
            timestamp_unit="ns",
            defaults={"source": "test"},
        )
        normalizer = Normalizer(profile)
        result = normalizer.normalize({"sym": "aapl", "t": 100, "p": 1.0, "s": 1.0})
        assert result["symbol"] == "AAPL"

    def test_profile_and_schema_accessible(self, trade_profile: NormalizationProfile):
        normalizer = Normalizer(trade_profile)
        assert normalizer.profile == trade_profile
        assert isinstance(normalizer.target_schema, pa.Schema)


class TestABLineArbiter:
    def test_picks_earlier_timestamp(self):
        arbiter = ABLineArbiter()
        a = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}
        b = {"symbol": "AAPL", "exchange_timestamp": 200, "price": 185.0}

        result = arbiter.arbitrate(a, b)
        assert result is a

    def test_picks_b_when_earlier(self):
        arbiter = ABLineArbiter()
        a = {"symbol": "AAPL", "exchange_timestamp": 200, "price": 185.0}
        b = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}

        result = arbiter.arbitrate(a, b)
        assert result is b

    def test_none_a(self):
        arbiter = ABLineArbiter()
        b = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}
        result = arbiter.arbitrate(None, b)
        assert result is b

    def test_none_b(self):
        arbiter = ABLineArbiter()
        a = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}
        result = arbiter.arbitrate(a, None)
        assert result is a

    def test_both_none(self):
        arbiter = ABLineArbiter()
        assert arbiter.arbitrate(None, None) is None

    def test_dedup_within_window(self):
        arbiter = ABLineArbiter(dedup_window_ns=1000)
        a = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}
        b = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}

        result1 = arbiter.arbitrate(a, None)
        assert result1 is a

        result2 = arbiter.arbitrate(None, b)
        assert result2 is None  # Deduplicated

    def test_clear(self):
        arbiter = ABLineArbiter(dedup_window_ns=1000)
        a = {"symbol": "AAPL", "exchange_timestamp": 100, "price": 185.0}

        arbiter.arbitrate(a, None)
        arbiter.clear()

        result = arbiter.arbitrate(a, None)
        assert result is a  # Not deduped after clear
