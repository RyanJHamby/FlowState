"""Tests for streaming temporal alignment with watermarks."""

from __future__ import annotations

import pyarrow as pa

from flowstate.prism.streaming import (
    LatePolicy,
    StreamingAlignConfig,
    StreamingAligner,
)


def _trades(timestamps: list[int], symbols: list[str] | None = None) -> pa.Table:
    n = len(timestamps)
    data = {
        "timestamp": pa.array(timestamps, type=pa.int64()),
        "price": [100.0 + i * 0.1 for i in range(n)],
    }
    if symbols:
        data["symbol"] = symbols
    return pa.table(data)


def _quotes(timestamps: list[int], symbols: list[str] | None = None) -> pa.Table:
    n = len(timestamps)
    data = {
        "timestamp": pa.array(timestamps, type=pa.int64()),
        "bid": [99.0 + i * 0.1 for i in range(n)],
        "ask": [101.0 + i * 0.1 for i in range(n)],
    }
    if symbols:
        data["symbol"] = symbols
    return pa.table(data)


class TestBasicEmission:
    def test_push_and_emit(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10, 20, 30]))
        aligner.push_right(_quotes([5, 15, 25]))

        aligner.advance_watermark(35)
        result = aligner.emit()

        assert result is not None
        assert result.num_rows == 3

    def test_no_emission_before_watermark(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10, 20, 30]))
        aligner.push_right(_quotes([5, 15, 25]))

        # Watermark not advanced — nothing sealed
        result = aligner.emit()
        assert result is None or result.num_rows == 0

    def test_partial_emission(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10, 20, 30]))
        aligner.push_right(_quotes([5, 15, 25]))

        # Watermark only seals ts=10
        aligner.advance_watermark(15)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1

    def test_incremental_pushes(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))

        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))
        aligner.advance_watermark(15)
        r1 = aligner.emit()
        assert r1 is not None
        assert r1.num_rows == 1

        aligner.push_left(_trades([20]))
        aligner.push_right(_quotes([15]))
        aligner.advance_watermark(25)
        r2 = aligner.emit()
        assert r2 is not None
        assert r2.num_rows == 1

    def test_empty_push_ignored(self):
        aligner = StreamingAligner()
        empty = pa.table({
            "timestamp": pa.array([], type=pa.int64()),
            "price": pa.array([], type=pa.float64()),
        })
        aligner.push_left(empty)
        assert aligner.stats.left_rows_pushed == 0


class TestWatermark:
    def test_watermark_only_advances(self):
        aligner = StreamingAligner()
        aligner.advance_watermark(100)
        aligner.advance_watermark(50)  # Should be ignored
        assert aligner.stats.watermark_ns == 100

    def test_watermark_with_lateness(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=10))
        aligner.push_left(_trades([10, 20, 30]))
        aligner.push_right(_quotes([5, 15, 25]))

        # Watermark=25, lateness=10 → seal threshold=15 → only ts=10 sealed
        aligner.advance_watermark(25)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1


class TestFlush:
    def test_flush_emits_all(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10, 20, 30]))
        aligner.push_right(_quotes([5, 15, 25]))

        # No watermark advance — flush forces everything
        result = aligner.flush()
        assert result is not None
        assert result.num_rows == 3

    def test_flush_after_partial_emit(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10, 20, 30]))
        aligner.push_right(_quotes([5, 15, 25]))

        aligner.advance_watermark(15)
        r1 = aligner.emit()
        assert r1 is not None
        assert r1.num_rows == 1

        r2 = aligner.flush()
        assert r2 is not None
        assert r2.num_rows == 2

    def test_flush_with_no_data(self):
        aligner = StreamingAligner()
        result = aligner.flush()
        assert result is None


class TestDirections:
    def test_backward_join(self):
        config = StreamingAlignConfig(direction="backward", lateness_ns=0)
        aligner = StreamingAligner(config)
        # Right at ts=5, left at ts=10 → backward finds ts=5
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))
        aligner.advance_watermark(15)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1
        # Check the right-side bid was matched
        col_names = result.schema.names
        bid_col = [c for c in col_names if "bid" in c]
        assert len(bid_col) == 1
        assert result.column(bid_col[0])[0].as_py() is not None

    def test_backward_no_match(self):
        config = StreamingAlignConfig(direction="backward", lateness_ns=0)
        aligner = StreamingAligner(config)
        # Right at ts=15 (after left ts=10) → no backward match
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([15]))
        aligner.advance_watermark(20)
        result = aligner.emit()
        assert result is not None
        col_names = result.schema.names
        bid_col = [c for c in col_names if "bid" in c]
        assert len(bid_col) == 1
        assert result.column(bid_col[0])[0].as_py() is None

    def test_forward_join(self):
        config = StreamingAlignConfig(direction="forward", lateness_ns=0)
        aligner = StreamingAligner(config)
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([15]))
        aligner.advance_watermark(20)
        result = aligner.emit()
        assert result is not None
        col_names = result.schema.names
        bid_col = [c for c in col_names if "bid" in c]
        assert len(bid_col) == 1
        assert result.column(bid_col[0])[0].as_py() is not None

    def test_nearest_join(self):
        config = StreamingAlignConfig(direction="nearest", lateness_ns=0)
        aligner = StreamingAligner(config)
        # Right at 5 and 18; left at 10 → nearest is 5 (dist=5 < dist=8)
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5, 18]))
        aligner.advance_watermark(20)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1


class TestTolerance:
    def test_within_tolerance(self):
        config = StreamingAlignConfig(tolerance_ns=10, lateness_ns=0)
        aligner = StreamingAligner(config)
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))  # 10-5=5 <= tolerance 10
        aligner.advance_watermark(15)
        result = aligner.emit()
        assert result is not None
        col_names = result.schema.names
        bid_col = [c for c in col_names if "bid" in c]
        assert result.column(bid_col[0])[0].as_py() is not None

    def test_exceeds_tolerance(self):
        config = StreamingAlignConfig(tolerance_ns=3, lateness_ns=0)
        aligner = StreamingAligner(config)
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))  # 10-5=5 > tolerance 3
        aligner.advance_watermark(15)
        result = aligner.emit()
        assert result is not None
        col_names = result.schema.names
        bid_col = [c for c in col_names if "bid" in c]
        assert result.column(bid_col[0])[0].as_py() is None


class TestGrouped:
    def test_per_symbol_join(self):
        config = StreamingAlignConfig(group_col="symbol", lateness_ns=0)
        aligner = StreamingAligner(config)

        aligner.push_left(_trades([10, 20], symbols=["AAPL", "MSFT"]))
        aligner.push_right(_quotes([8, 18], symbols=["AAPL", "MSFT"]))
        aligner.advance_watermark(25)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 2

    def test_cross_symbol_isolation(self):
        config = StreamingAlignConfig(group_col="symbol", lateness_ns=0)
        aligner = StreamingAligner(config)

        # AAPL trade at 10, MSFT quote at 8 — should NOT match
        aligner.push_left(_trades([10], symbols=["AAPL"]))
        aligner.push_right(_quotes([8], symbols=["MSFT"]))
        aligner.advance_watermark(15)
        result = aligner.emit()
        assert result is not None
        col_names = result.schema.names
        bid_col = [c for c in col_names if "bid" in c]
        assert result.column(bid_col[0])[0].as_py() is None


class TestLateData:
    def test_late_data_dropped(self):
        config = StreamingAlignConfig(
            lateness_ns=0,
            late_policy=LatePolicy.DROP,
        )
        aligner = StreamingAligner(config)
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))
        aligner.advance_watermark(20)
        aligner.emit()

        # Push data behind watermark
        aligner.push_left(_trades([5]))
        # In Python fallback, late rows are dropped
        if aligner.implementation == "python":
            assert aligner.stats.late_rows_dropped >= 1


class TestStats:
    def test_stats_tracking(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10, 20]))
        aligner.push_right(_quotes([5, 15]))
        aligner.advance_watermark(25)
        aligner.emit()

        s = aligner.stats
        assert s.left_rows_pushed == 2
        assert s.right_rows_pushed == 2
        assert s.rows_emitted == 2
        assert s.batches_emitted == 1
        assert s.watermark_ns == 25

    def test_stats_reset(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))
        aligner.advance_watermark(15)
        aligner.emit()

        aligner.reset()
        s = aligner.stats
        assert s.left_rows_pushed == 0
        assert s.rows_emitted == 0
        assert s.watermark_ns == 0


class TestReset:
    def test_reset_clears_state(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))
        aligner.advance_watermark(15)
        aligner.emit()

        aligner.reset()
        # After reset, no data to emit
        result = aligner.flush()
        assert result is None

    def test_usable_after_reset(self):
        aligner = StreamingAligner(StreamingAlignConfig(lateness_ns=0))
        aligner.push_left(_trades([10]))
        aligner.push_right(_quotes([5]))
        aligner.advance_watermark(15)
        aligner.emit()

        aligner.reset()
        aligner.push_left(_trades([100]))
        aligner.push_right(_quotes([95]))
        aligner.advance_watermark(110)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1


class TestImplementation:
    def test_implementation_property(self):
        aligner = StreamingAligner()
        assert aligner.implementation in ("rust", "python")

    def test_config_preserved(self):
        config = StreamingAlignConfig(
            tolerance_ns=42,
            lateness_ns=7,
            direction="forward",
        )
        aligner = StreamingAligner(config)
        assert aligner.config.tolerance_ns == 42
        assert aligner.config.lateness_ns == 7
        assert aligner.config.direction == "forward"
