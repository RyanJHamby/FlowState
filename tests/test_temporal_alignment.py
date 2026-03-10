"""Tests for the temporal alignment engine."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flowstate.prism.alignment import (
    AlignmentSpec,
    AsOfConfig,
    TemporalAligner,
    align_streams,
    as_of_join,
)

TS = pa.timestamp("ns", tz="UTC")


def _ts(seconds: int) -> int:
    """Helper: convert seconds to nanoseconds."""
    return seconds * 10**9


def _make_trades(symbols: list[str], timestamps_s: list[int]) -> pa.Table:
    n = len(symbols)
    return pa.table(
        {
            "symbol": symbols,
            "timestamp": pa.array([_ts(t) for t in timestamps_s], type=TS),
            "price": [100.0 + i for i in range(n)],
            "size": [10.0] * n,
        }
    )


def _make_quotes(symbols: list[str], timestamps_s: list[int]) -> pa.Table:
    n = len(symbols)
    return pa.table(
        {
            "symbol": symbols,
            "timestamp": pa.array([_ts(t) for t in timestamps_s], type=TS),
            "bid_price": [99.0 + i * 0.1 for i in range(n)],
            "ask_price": [101.0 + i * 0.1 for i in range(n)],
            "bid_size": [50.0] * n,
            "ask_size": [50.0] * n,
        }
    )


# ===== as_of_join: backward (default) =====

class TestAsOfJoinBackward:
    def test_basic_backward_join(self):
        """Each trade gets the most recent quote at or before its timestamp."""
        trades = _make_trades(["AAPL"] * 3, [10, 20, 30])
        quotes = _make_quotes(["AAPL"] * 3, [5, 15, 25])

        result, stats = as_of_join(trades, quotes)

        assert result.num_rows == 3
        assert stats.matched_rows == 3
        assert stats.unmatched_rows == 0

        # Trade at t=10 matches quote at t=5
        # Trade at t=20 matches quote at t=15
        # Trade at t=30 matches quote at t=25
        bid_prices = result.column("bid_price").to_pylist()
        assert bid_prices[0] == pytest.approx(99.0)  # quote at t=5
        assert bid_prices[1] == pytest.approx(99.1)  # quote at t=15
        assert bid_prices[2] == pytest.approx(99.2)  # quote at t=25

    def test_no_match_before_first_quote(self):
        """Trade before any quote should have null values."""
        trades = _make_trades(["AAPL"] * 2, [1, 20])
        quotes = _make_quotes(["AAPL"] * 1, [10])

        result, stats = as_of_join(trades, quotes)

        assert result.num_rows == 2
        assert stats.matched_rows == 1
        assert stats.unmatched_rows == 1

        bid_prices = result.column("bid_price").to_pylist()
        assert bid_prices[0] is None  # No quote before t=1
        assert bid_prices[1] == pytest.approx(99.0)  # quote at t=10

    def test_exact_match(self):
        """Exact timestamp match is included by default."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL"], [10])

        result, stats = as_of_join(trades, quotes)
        assert stats.matched_rows == 1
        assert result.column("bid_price").to_pylist()[0] is not None

    def test_exact_match_excluded(self):
        """Exact match can be disabled."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL"], [10])

        cfg = AsOfConfig(allow_exact_match=False)
        result, stats = as_of_join(trades, quotes, config=cfg)
        assert stats.matched_rows == 0

    def test_per_symbol_join(self):
        """Join respects symbol grouping."""
        trades = _make_trades(["AAPL", "MSFT", "AAPL"], [10, 10, 20])
        quotes = _make_quotes(["AAPL", "MSFT"], [5, 8])

        result, stats = as_of_join(trades, quotes)

        assert result.num_rows == 3
        bid_prices = result.column("bid_price").to_pylist()
        # AAPL trade at t=10 → AAPL quote at t=5
        assert bid_prices[0] == pytest.approx(99.0)
        # MSFT trade at t=10 → MSFT quote at t=8
        assert bid_prices[1] == pytest.approx(99.1)
        # AAPL trade at t=20 → AAPL quote at t=5 (still most recent)
        assert bid_prices[2] == pytest.approx(99.0)

    def test_cross_symbol_isolation(self):
        """Quotes from one symbol never leak to another."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["MSFT"], [5])

        result, stats = as_of_join(trades, quotes)
        assert stats.matched_rows == 0
        assert result.column("bid_price").to_pylist()[0] is None


# ===== as_of_join: tolerance =====

class TestAsOfJoinTolerance:
    def test_within_tolerance(self):
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL"], [8])

        cfg = AsOfConfig(tolerance_ns=_ts(5))
        result, stats = as_of_join(trades, quotes, config=cfg)
        assert stats.matched_rows == 1

    def test_exceeds_tolerance(self):
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL"], [1])

        cfg = AsOfConfig(tolerance_ns=_ts(5))
        result, stats = as_of_join(trades, quotes, config=cfg)
        assert stats.matched_rows == 0
        assert result.column("bid_price").to_pylist()[0] is None

    def test_tolerance_boundary_exact(self):
        """Match at exactly the tolerance boundary."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL"], [5])

        cfg = AsOfConfig(tolerance_ns=_ts(5))
        result, stats = as_of_join(trades, quotes, config=cfg)
        assert stats.matched_rows == 1


# ===== as_of_join: forward =====

class TestAsOfJoinForward:
    def test_forward_join(self):
        """Forward join: each trade gets the next future quote."""
        trades = _make_trades(["AAPL"] * 2, [10, 20])
        quotes = _make_quotes(["AAPL"] * 2, [15, 25])

        cfg = AsOfConfig(direction="forward")
        result, stats = as_of_join(trades, quotes, config=cfg)

        assert stats.matched_rows == 2
        bid_prices = result.column("bid_price").to_pylist()
        assert bid_prices[0] == pytest.approx(99.0)  # quote at t=15
        assert bid_prices[1] == pytest.approx(99.1)  # quote at t=25

    def test_forward_no_future_quote(self):
        trades = _make_trades(["AAPL"], [30])
        quotes = _make_quotes(["AAPL"], [10])

        cfg = AsOfConfig(direction="forward")
        result, stats = as_of_join(trades, quotes, config=cfg)
        assert stats.matched_rows == 0


# ===== as_of_join: nearest =====

class TestAsOfJoinNearest:
    def test_nearest_prefers_closer(self):
        trades = _make_trades(["AAPL"], [12])
        quotes = _make_quotes(["AAPL"] * 2, [10, 15])

        cfg = AsOfConfig(direction="nearest")
        result, stats = as_of_join(trades, quotes, config=cfg)

        assert stats.matched_rows == 1
        # t=12 is closer to t=10 (dist=2) than t=15 (dist=3) → picks t=10
        bid_prices = result.column("bid_price").to_pylist()
        assert bid_prices[0] == pytest.approx(99.0)

    def test_nearest_with_tolerance(self):
        trades = _make_trades(["AAPL"], [100])
        quotes = _make_quotes(["AAPL"], [1])

        cfg = AsOfConfig(direction="nearest", tolerance_ns=_ts(10))
        result, stats = as_of_join(trades, quotes, config=cfg)
        assert stats.matched_rows == 0


# ===== as_of_join: edge cases =====

class TestAsOfJoinEdgeCases:
    def test_empty_left(self):
        trades = _make_trades([], [])
        quotes = _make_quotes(["AAPL"], [10])

        result, stats = as_of_join(trades, quotes)
        assert result.num_rows == 0
        assert stats.matched_rows == 0

    def test_empty_right(self):
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes([], [])

        result, stats = as_of_join(trades, quotes)
        assert result.num_rows == 1
        assert stats.matched_rows == 0

    def test_global_join_no_symbol(self):
        """Join without symbol column (global join)."""
        left = pa.table({
            "timestamp": pa.array([_ts(10), _ts(20)], type=TS),
            "value": [1.0, 2.0],
        })
        right = pa.table({
            "timestamp": pa.array([_ts(5), _ts(15)], type=TS),
            "indicator": [100, 200],
        })

        result, stats = as_of_join(left, right, by=None)
        assert result.num_rows == 2
        assert stats.matched_rows == 2

        indicators = result.column("indicator").to_pylist()
        assert indicators[0] == 100  # t=10 → right t=5
        assert indicators[1] == 200  # t=20 → right t=15

    def test_prefix_avoids_collision(self):
        """Column prefix prevents name collisions."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL"], [5])

        cfg = AsOfConfig(right_prefix="quote_")
        result, _ = as_of_join(trades, quotes, config=cfg)

        assert "quote_bid_price" in result.schema.names
        assert "quote_ask_price" in result.schema.names

    def test_multiple_quotes_same_timestamp(self):
        """When multiple right rows have the same timestamp, picks the last one."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes(["AAPL", "AAPL"], [5, 5])

        result, stats = as_of_join(trades, quotes)
        assert stats.matched_rows == 1


# ===== align_streams =====

class TestAlignStreams:
    def test_multi_stream_alignment(self):
        """Align trades with both quotes and bars."""
        trades = _make_trades(["AAPL"] * 3, [10, 20, 30])

        quotes = _make_quotes(["AAPL"] * 3, [5, 15, 25])

        bars = pa.table({
            "symbol": ["AAPL"] * 2,
            "timestamp": pa.array([_ts(0), _ts(15)], type=TS),
            "vwap": [100.5, 101.2],
        })

        secondaries = [
            AlignmentSpec(name="quote", table=quotes, value_columns=["bid_price", "ask_price"]),
            AlignmentSpec(name="bar", table=bars, value_columns=["vwap"]),
        ]

        result, stats = align_streams(trades, secondaries)

        assert result.num_rows == 3
        assert "quote_bid_price" in result.schema.names
        assert "quote_ask_price" in result.schema.names
        assert "bar_vwap" in result.schema.names
        assert "quote" in stats
        assert "bar" in stats

    def test_empty_secondary(self):
        """An empty secondary stream produces null columns."""
        trades = _make_trades(["AAPL"], [10])
        quotes = _make_quotes([], [])

        secondaries = [AlignmentSpec(name="quote", table=quotes)]
        result, stats = align_streams(trades, secondaries)

        assert result.num_rows == 1


# ===== TemporalAligner =====

class TestTemporalAligner:
    def test_basic_alignment(self):
        aligner = TemporalAligner(
            primary_type="trade",
            secondary_specs={"quote": ["bid_price", "ask_price"]},
        )

        aligner.add_data("trade", _make_trades(["AAPL"] * 3, [10, 20, 30]))
        aligner.add_data("quote", _make_quotes(["AAPL"] * 3, [5, 15, 25]))

        result, stats = aligner.flush()

        assert result is not None
        assert result.num_rows == 3
        assert "quote_bid_price" in result.schema.names
        assert aligner.total_aligned == 3

    def test_no_primary_data(self):
        aligner = TemporalAligner(primary_type="trade")
        result, stats = aligner.flush()
        assert result is None
        assert stats == {}

    def test_unknown_stream_raises(self):
        aligner = TemporalAligner(primary_type="trade")
        with pytest.raises(ValueError, match="Unknown stream"):
            aligner.add_data("nonexistent", _make_trades([], []))

    def test_multiple_batches(self):
        """Aligner concatenates multiple add_data calls before flushing."""
        aligner = TemporalAligner(
            primary_type="trade",
            secondary_specs={"quote": None},
        )

        aligner.add_data("trade", _make_trades(["AAPL"] * 2, [10, 20]))
        aligner.add_data("trade", _make_trades(["AAPL"] * 2, [30, 40]))
        aligner.add_data("quote", _make_quotes(["AAPL"] * 2, [5, 25]))

        result, stats = aligner.flush()
        assert result is not None
        assert result.num_rows == 4

    def test_flush_clears_buffers(self):
        aligner = TemporalAligner(primary_type="trade")
        aligner.add_data("trade", _make_trades(["AAPL"], [10]))
        result, _ = aligner.flush()
        assert result is not None

        # Second flush should be empty
        result, _ = aligner.flush()
        assert result is None

    def test_reset(self):
        aligner = TemporalAligner(primary_type="trade")
        aligner.add_data("trade", _make_trades(["AAPL"], [10]))
        aligner.reset()
        result, _ = aligner.flush()
        assert result is None

    def test_tolerance(self):
        aligner = TemporalAligner(
            primary_type="trade",
            secondary_specs={"quote": ["bid_price"]},
            tolerance_ns=_ts(3),
        )

        aligner.add_data("trade", _make_trades(["AAPL"] * 2, [10, 20]))
        # Quote at t=5 is 5s before trade at t=10 → exceeds 3s tolerance
        # Quote at t=18 is 2s before trade at t=20 → within tolerance
        aligner.add_data("quote", _make_quotes(["AAPL"] * 2, [5, 18]))

        result, stats = aligner.flush()
        assert result is not None
        bid_prices = result.column("quote_bid_price").to_pylist()
        assert bid_prices[0] is None  # t=10, no quote within 3s
        assert bid_prices[1] is not None  # t=20, quote at t=18 within 3s

    def test_multi_symbol(self):
        aligner = TemporalAligner(
            primary_type="trade",
            secondary_specs={"quote": ["bid_price"]},
        )

        aligner.add_data("trade", _make_trades(["AAPL", "MSFT"], [10, 10]))
        aligner.add_data("quote", _make_quotes(["AAPL", "MSFT"], [5, 8]))

        result, _ = aligner.flush()
        assert result is not None
        assert result.num_rows == 2
        assert "quote_bid_price" in result.schema.names

    def test_sorts_primary_by_timestamp(self):
        """Primary data is sorted by timestamp even if added out of order."""
        aligner = TemporalAligner(primary_type="trade")

        # Add trades out of timestamp order
        aligner.add_data("trade", _make_trades(["AAPL", "AAPL"], [20, 10]))

        result, _ = aligner.flush()
        assert result is not None
        ts_col = result.column("timestamp").cast(pa.int64()).to_pylist()
        assert ts_col[0] < ts_col[1]  # Should be sorted


# ===== Point-in-time correctness =====

class TestPointInTimeCorrectness:
    """Verify no look-ahead bias in alignment results."""

    def test_no_lookahead_bias(self):
        """Quote arriving AFTER a trade must not be visible at the trade's timestamp."""
        trades = _make_trades(["AAPL"] * 3, [10, 20, 30])
        # Quote at t=25 should NOT appear for trade at t=20
        quotes = _make_quotes(["AAPL"] * 2, [5, 25])

        result, _ = as_of_join(trades, quotes)

        bid_prices = result.column("bid_price").to_pylist()
        # t=10 → quote at t=5
        assert bid_prices[0] == pytest.approx(99.0)
        # t=20 → quote at t=5 (NOT t=25!)
        assert bid_prices[1] == pytest.approx(99.0)
        # t=30 → quote at t=25
        assert bid_prices[2] == pytest.approx(99.1)

    def test_forward_join_has_lookahead(self):
        """Forward join intentionally uses future data (for labels)."""
        trades = _make_trades(["AAPL"], [10])
        # Future price at t=20
        future = pa.table({
            "symbol": ["AAPL"],
            "timestamp": pa.array([_ts(20)], type=TS),
            "future_price": [105.0],
        })

        cfg = AsOfConfig(direction="forward")
        result, stats = as_of_join(trades, future, config=cfg)
        assert stats.matched_rows == 1
        assert result.column("future_price").to_pylist()[0] == pytest.approx(105.0)
