"""Tests for schema enforcement and sequence gap detection."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flowstate.schema.types import TRADE_SCHEMA
from flowstate.schema.validation import SchemaValidator, SequenceTracker


@pytest.fixture
def validator() -> SchemaValidator:
    return SchemaValidator(TRADE_SCHEMA)


def _make_trade_batch(**overrides: list) -> pa.RecordBatch:
    """Helper to build a valid trade batch with optional column overrides."""
    defaults = {
        "symbol": pa.array(["AAPL"], type=pa.utf8()),
        "timestamp": pa.array([1700000000000000000], type=pa.timestamp("ns", tz="UTC")),
        "exchange_timestamp": pa.array([None], type=pa.timestamp("ns", tz="UTC")),
        "receive_timestamp": pa.array([1700000000000000000], type=pa.timestamp("ns", tz="UTC")),
        "price": pa.array([185.50], type=pa.float64()),
        "size": pa.array([100.0], type=pa.float64()),
        "exchange": pa.array(["XNAS"], type=pa.utf8()),
        "conditions": pa.array([None], type=pa.list_(pa.utf8())),
        "tape": pa.array(["A"], type=pa.utf8()),
        "sequence": pa.array([1], type=pa.int64()),
        "trade_id": pa.array(["t1"], type=pa.utf8()),
        "source": pa.array(["polygon"], type=pa.utf8()),
    }
    defaults.update(overrides)
    return pa.RecordBatch.from_pydict(defaults, schema=TRADE_SCHEMA)


class TestSchemaValidator:
    def test_valid_batch(self, validator: SchemaValidator):
        batch = _make_trade_batch()
        result = validator.validate(batch)
        assert result.is_valid
        assert result.rows_validated == 1
        assert result.errors == []

    def test_null_in_non_nullable_field(self, validator: SchemaValidator):
        batch = _make_trade_batch(
            symbol=pa.array([None], type=pa.utf8()),
        )
        result = validator.validate(batch)
        assert not result.is_valid
        assert any(e.error_type == "null_violation" for e in result.errors)

    def test_missing_field(self, validator: SchemaValidator):
        # Build a batch with a missing column
        schema = pa.schema([
            pa.field("symbol", pa.utf8(), nullable=False),
            pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
        ])
        batch = pa.RecordBatch.from_pydict(
            {
                "symbol": pa.array(["AAPL"], type=pa.utf8()),
                "timestamp": pa.array([1700000000000000000], type=pa.timestamp("ns", tz="UTC")),
            },
            schema=schema,
        )
        result = validator.validate(batch)
        assert not result.is_valid
        assert any(e.error_type == "missing_field" for e in result.errors)

    def test_schema_property(self, validator: SchemaValidator):
        assert validator.schema == TRADE_SCHEMA


class TestSequenceTracker:
    def test_first_message_no_gap(self):
        tracker = SequenceTracker()
        gap = tracker.track("AAPL", 1)
        assert gap is None
        assert tracker.total_messages == 1

    def test_sequential_no_gap(self):
        tracker = SequenceTracker()
        tracker.track("AAPL", 1)
        gap = tracker.track("AAPL", 2)
        assert gap is None

    def test_gap_detected(self):
        tracker = SequenceTracker()
        tracker.track("AAPL", 1)
        gap = tracker.track("AAPL", 5)
        assert gap is not None
        assert gap.symbol == "AAPL"
        assert gap.expected == 2
        assert gap.actual == 5
        assert gap.gap_size == 3

    def test_multiple_symbols(self):
        tracker = SequenceTracker()
        tracker.track("AAPL", 1)
        tracker.track("MSFT", 1)
        gap = tracker.track("AAPL", 2)
        assert gap is None

    def test_gaps_list(self):
        tracker = SequenceTracker()
        tracker.track("AAPL", 1)
        tracker.track("AAPL", 5)
        tracker.track("AAPL", 10)
        assert len(tracker.gaps) == 2
        assert tracker.total_gaps == 2

    def test_track_batch(self):
        tracker = SequenceTracker()
        batch = pa.RecordBatch.from_pydict(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "sequence": [1, 2, 5],
            },
            schema=pa.schema([
                pa.field("symbol", pa.utf8()),
                pa.field("sequence", pa.int64()),
            ]),
        )
        gaps = tracker.track_batch(batch)
        assert len(gaps) == 1
        assert gaps[0].gap_size == 2

    def test_track_batch_with_nulls(self):
        tracker = SequenceTracker()
        batch = pa.RecordBatch.from_pydict(
            {
                "symbol": ["AAPL", "AAPL"],
                "sequence": [1, None],
            },
            schema=pa.schema([
                pa.field("symbol", pa.utf8()),
                pa.field("sequence", pa.int64()),
            ]),
        )
        gaps = tracker.track_batch(batch)
        assert len(gaps) == 0

    def test_reset_symbol(self):
        tracker = SequenceTracker()
        tracker.track("AAPL", 1)
        tracker.reset("AAPL")
        gap = tracker.track("AAPL", 100)
        assert gap is None

    def test_reset_all(self):
        tracker = SequenceTracker()
        tracker.track("AAPL", 1)
        tracker.track("AAPL", 5)
        tracker.reset()
        assert tracker.total_messages == 0
        assert tracker.total_gaps == 0
        assert tracker.gaps == []
