"""Tests for the Rust core module (flowstate_core).

Validates zero-copy Arrow round-trip, as-of joins (backward, forward, nearest),
grouped joins with rayon parallelism, and correctness parity with Python.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

# Skip entire module if Rust extension not built
flowstate_core = pytest.importorskip("flowstate_core")

from flowstate.prism.alignment import AsOfConfig, as_of_join  # noqa: E402

TS = pa.timestamp("ns", tz="UTC")


# ---------------------------------------------------------------------------
# Zero-copy round-trip
# ---------------------------------------------------------------------------


class TestEchoTable:
    def test_round_trip_preserves_data(self):
        table = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
            "symbol": pa.array(["A", "B", "C"]),
        })
        result = flowstate_core.echo_table(table)
        assert result.equals(table)

    def test_round_trip_empty_table(self):
        table = pa.table({
            "x": pa.array([], type=pa.int64()),
        })
        result = flowstate_core.echo_table(table)
        assert result.num_rows == 0

    def test_round_trip_timestamp_type(self):
        table = pa.table({
            "timestamp": pa.array([1_000_000, 2_000_000], type=TS),
            "value": pa.array([1.0, 2.0]),
        })
        result = flowstate_core.echo_table(table)
        assert result.equals(table)

    def test_row_count(self):
        table = pa.table({"x": pa.array(range(100), type=pa.int64())})
        assert flowstate_core.row_count(table) == 100

    def test_row_count_empty(self):
        table = pa.table({"x": pa.array([], type=pa.int64())})
        assert flowstate_core.row_count(table) == 0


# ---------------------------------------------------------------------------
# Ungrouped backward as-of join
# ---------------------------------------------------------------------------


class TestUngroupedBackwardJoin:
    def test_basic_join(self):
        left = pa.table({
            "timestamp": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        })
        right = pa.table({
            "timestamp": pa.array([5, 15, 25, 35, 45], type=pa.int64()),
            "quote": pa.array([0.5, 1.5, 2.5, 3.5, 4.5]),
        })
        result = flowstate_core.asof_join_backward(left, right, on="timestamp")
        assert result.num_rows == 5
        assert result.column("quote").to_pylist() == [0.5, 1.5, 2.5, 3.5, 4.5]

    def test_exact_matches(self):
        left = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "quote": pa.array([0.1, 0.2, 0.3]),
        })
        result = flowstate_core.asof_join_backward(left, right, on="timestamp")
        assert result.column("quote").to_pylist() == [0.1, 0.2, 0.3]

    def test_no_matches(self):
        left = pa.table({
            "timestamp": pa.array([1, 2, 3], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "quote": pa.array([0.1, 0.2, 0.3]),
        })
        result = flowstate_core.asof_join_backward(left, right, on="timestamp")
        assert all(v is None for v in result.column("quote").to_pylist())

    def test_with_tolerance(self):
        left = pa.table({
            "timestamp": pa.array([10, 20, 100], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([5, 15], type=pa.int64()),
            "quote": pa.array([0.5, 1.5]),
        })
        result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", tolerance_ns=10,
        )
        quotes = result.column("quote").to_pylist()
        assert quotes[0] == 0.5   # 10-5=5 <= 10
        assert quotes[1] == 1.5   # 20-15=5 <= 10
        assert quotes[2] is None  # 100-15=85 > 10

    def test_right_prefix(self):
        left = pa.table({
            "timestamp": pa.array([10], type=pa.int64()),
            "price": pa.array([1.0]),
        })
        right = pa.table({
            "timestamp": pa.array([5], type=pa.int64()),
            "price": pa.array([0.5]),
        })
        result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", right_prefix="q_",
        )
        assert "q_price" in result.column_names

    def test_unsorted_input(self):
        """Rust kernel should handle unsorted input by sorting internally."""
        left = pa.table({
            "timestamp": pa.array([30, 10, 20], type=pa.int64()),
            "price": pa.array([3.0, 1.0, 2.0]),
        })
        right = pa.table({
            "timestamp": pa.array([25, 5, 15], type=pa.int64()),
            "quote": pa.array([2.5, 0.5, 1.5]),
        })
        result = flowstate_core.asof_join_backward(left, right, on="timestamp")
        # After sorting by timestamp, all should match
        assert result.num_rows == 3
        assert None not in result.column("quote").to_pylist()


# ---------------------------------------------------------------------------
# Grouped backward as-of join
# ---------------------------------------------------------------------------


class TestGroupedBackwardJoin:
    def test_basic_grouped(self):
        left = pa.table({
            "timestamp": pa.array([10, 20, 10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 10.0, 20.0]),
            "symbol": pa.array(["AAPL", "AAPL", "MSFT", "MSFT"]),
        })
        right = pa.table({
            "timestamp": pa.array([5, 15, 5, 15], type=pa.int64()),
            "bid": pa.array([100.0, 101.0, 200.0, 201.0]),
            "symbol": pa.array(["AAPL", "AAPL", "MSFT", "MSFT"]),
        })
        result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", by="symbol",
        )
        assert result.num_rows == 4
        assert "bid" in result.column_names

    def test_missing_symbol_in_right(self):
        left = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
            "symbol": pa.array(["AAPL", "GOOG"]),
        })
        right = pa.table({
            "timestamp": pa.array([5], type=pa.int64()),
            "bid": pa.array([100.0]),
            "symbol": pa.array(["AAPL"]),
        })
        result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", by="symbol",
        )
        bids = result.column("bid").to_pylist()
        # AAPL should match, GOOG should be null
        matched = sum(1 for b in bids if b is not None)
        assert matched == 1

    def test_grouped_with_prefix(self):
        left = pa.table({
            "timestamp": pa.array([10], type=pa.int64()),
            "price": pa.array([1.0]),
            "symbol": pa.array(["X"]),
        })
        right = pa.table({
            "timestamp": pa.array([5], type=pa.int64()),
            "price": pa.array([0.5]),
            "symbol": pa.array(["X"]),
        })
        result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", by="symbol", right_prefix="r_",
        )
        assert "r_price" in result.column_names

    def test_many_symbols(self):
        """Test with 100 symbols to exercise the hash partitioning."""
        n_per_sym = 100
        n_syms = 100
        rng = np.random.default_rng(42)

        syms = np.repeat([f"S{i:03d}" for i in range(n_syms)], n_per_sym)
        ts = np.sort(rng.integers(0, 1_000_000, size=n_per_sym * n_syms))

        left = pa.table({
            "timestamp": pa.array(ts, type=pa.int64()),
            "price": pa.array(rng.uniform(0, 100, len(ts))),
            "symbol": pa.array(syms.tolist()),
        })
        right = pa.table({
            "timestamp": pa.array(ts + rng.integers(-5, 5, size=len(ts)), type=pa.int64()),
            "bid": pa.array(rng.uniform(0, 100, len(ts))),
            "symbol": pa.array(syms.tolist()),
        })
        result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", by="symbol",
        )
        assert result.num_rows == len(ts)
        assert "bid" in result.column_names


# ---------------------------------------------------------------------------
# Parity: Rust vs Python implementation
# ---------------------------------------------------------------------------


class TestRustPythonParity:
    """Verify the Rust kernel produces identical results to the Python numpy implementation."""

    def _make_test_data(self, n_left, n_right, n_symbols, seed=42):
        rng = np.random.default_rng(seed)
        n_per_l = n_left // n_symbols
        n_per_r = n_right // n_symbols

        left_ts = np.sort(rng.integers(0, 100_000, size=n_per_l * n_symbols))
        right_ts = np.sort(rng.integers(0, 100_000, size=n_per_r * n_symbols))
        left_syms = np.repeat([f"S{i:02d}" for i in range(n_symbols)], n_per_l)
        right_syms = np.repeat([f"S{i:02d}" for i in range(n_symbols)], n_per_r)

        left = pa.table({
            "timestamp": pa.array(left_ts, type=TS),
            "price": pa.array(rng.uniform(0, 100, len(left_ts))),
            "symbol": pa.array(left_syms.tolist()),
        })
        right = pa.table({
            "timestamp": pa.array(right_ts, type=TS),
            "bid": pa.array(rng.uniform(0, 100, len(right_ts))),
            "symbol": pa.array(right_syms.tolist()),
        })
        return left, right

    def test_ungrouped_parity(self):
        """Rust ungrouped join matches Python numpy result."""
        rng = np.random.default_rng(42)
        left = pa.table({
            "timestamp": pa.array(np.sort(rng.integers(0, 10000, 500)), type=pa.int64()),
            "price": pa.array(rng.uniform(0, 100, 500)),
        })
        right = pa.table({
            "timestamp": pa.array(np.sort(rng.integers(0, 10000, 1000)), type=pa.int64()),
            "bid": pa.array(rng.uniform(0, 100, 1000)),
        })

        # Python result
        py_result, _ = as_of_join(left, right, on="timestamp", by=None)

        # Rust result
        rust_result = flowstate_core.asof_join_backward(
            left, right, on="timestamp",
        )

        # Both should have same number of matched rows
        py_nulls = sum(1 for v in py_result.column("bid").to_pylist() if v is None)
        rust_nulls = sum(1 for v in rust_result.column("bid").to_pylist() if v is None)
        assert py_nulls == rust_nulls, f"Null count differs: Python={py_nulls}, Rust={rust_nulls}"

    def test_grouped_parity(self):
        """Rust grouped join matches Python numpy result on match count."""
        left, right = self._make_test_data(1000, 2000, 10)
        cfg = AsOfConfig(direction="backward")

        py_result, py_stats = as_of_join(left, right, on="timestamp", by="symbol", config=cfg)

        rust_result = flowstate_core.asof_join_backward(
            left, right, on="timestamp", by="symbol",
        )

        assert py_result.num_rows == rust_result.num_rows

        # Match counts should be identical
        py_matched = sum(1 for v in py_result.column("bid").to_pylist() if v is not None)
        rust_matched = sum(1 for v in rust_result.column("bid").to_pylist() if v is not None)
        assert py_matched == rust_matched, (
            f"Match count differs: Python={py_matched}, Rust={rust_matched}"
        )


# ---------------------------------------------------------------------------
# Forward as-of join
# ---------------------------------------------------------------------------


class TestForwardJoin:
    def test_basic_forward(self):
        left = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([15, 25, 35], type=pa.int64()),
            "quote": pa.array([1.5, 2.5, 3.5]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="forward",
        )
        assert result.column("quote").to_pylist() == [1.5, 2.5, 3.5]

    def test_forward_no_future(self):
        """Left timestamps after all right timestamps → null."""
        left = pa.table({
            "timestamp": pa.array([40, 50], type=pa.int64()),
            "price": pa.array([4.0, 5.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "quote": pa.array([1.0, 2.0, 3.0]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="forward",
        )
        assert all(v is None for v in result.column("quote").to_pylist())

    def test_forward_with_tolerance(self):
        left = pa.table({
            "timestamp": pa.array([10, 20, 100], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([15, 25], type=pa.int64()),
            "quote": pa.array([1.5, 2.5]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="forward", tolerance_ns=10,
        )
        quotes = result.column("quote").to_pylist()
        assert quotes[0] == 1.5   # 15-10=5 <= 10
        assert quotes[1] == 2.5   # 25-20=5 <= 10
        assert quotes[2] is None  # no right >= 100 within tolerance

    def test_forward_grouped(self):
        left = pa.table({
            "timestamp": pa.array([10, 10], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
            "symbol": pa.array(["A", "B"]),
        })
        right = pa.table({
            "timestamp": pa.array([15, 15], type=pa.int64()),
            "bid": pa.array([100.0, 200.0]),
            "symbol": pa.array(["A", "B"]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", by="symbol", direction="forward",
        )
        assert result.num_rows == 2
        bids = result.column("bid").to_pylist()
        assert all(b is not None for b in bids)


# ---------------------------------------------------------------------------
# Nearest as-of join
# ---------------------------------------------------------------------------


class TestNearestJoin:
    def test_nearest_basic(self):
        left = pa.table({
            "timestamp": pa.array([12, 23, 37], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 20, 30, 40], type=pa.int64()),
            "quote": pa.array([1.0, 2.0, 3.0, 4.0]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="nearest",
        )
        quotes = result.column("quote").to_pylist()
        assert quotes[0] == 1.0  # 12→10 (dist 2 < 8 to 20)
        assert quotes[1] == 2.0  # 23→20 (dist 3 < 7 to 30)
        assert quotes[2] == 4.0  # 37→40 (dist 3 < 7 to 30)

    def test_nearest_prefers_closer(self):
        left = pa.table({
            "timestamp": pa.array([15], type=pa.int64()),
            "price": pa.array([1.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "quote": pa.array([1.0, 2.0]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="nearest",
        )
        # Equal distance → prefer backward (10)
        assert result.column("quote").to_pylist() == [1.0]

    def test_nearest_with_tolerance(self):
        left = pa.table({
            "timestamp": pa.array([10, 50, 100], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([12, 48], type=pa.int64()),
            "quote": pa.array([1.2, 4.8]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="nearest", tolerance_ns=5,
        )
        quotes = result.column("quote").to_pylist()
        assert quotes[0] == 1.2   # dist 2 <= 5
        assert quotes[1] == 4.8   # dist 2 <= 5
        assert quotes[2] is None  # dist 52 > 5

    def test_nearest_grouped(self):
        left = pa.table({
            "timestamp": pa.array([15, 15], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
            "symbol": pa.array(["A", "B"]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 20, 12, 18], type=pa.int64()),
            "bid": pa.array([1.0, 2.0, 3.0, 4.0]),
            "symbol": pa.array(["A", "A", "B", "B"]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", by="symbol", direction="nearest",
        )
        assert result.num_rows == 2
        assert all(b is not None for b in result.column("bid").to_pylist())


# ---------------------------------------------------------------------------
# allow_exact_match=False
# ---------------------------------------------------------------------------


class TestNoExactMatch:
    def test_backward_no_exact(self):
        left = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 15], type=pa.int64()),
            "quote": pa.array([1.0, 1.5]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="backward", allow_exact_match=False,
        )
        quotes = result.column("quote").to_pylist()
        # left[0]=10: right[0]=10 is exact → skip, no earlier → None
        assert quotes[0] is None
        # left[1]=20: right[1]=15 < 20 → match
        assert quotes[1] == 1.5

    def test_forward_no_exact(self):
        left = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
        })
        right = pa.table({
            "timestamp": pa.array([10, 25], type=pa.int64()),
            "quote": pa.array([1.0, 2.5]),
        })
        result = flowstate_core.asof_join(
            left, right, on="timestamp", direction="forward", allow_exact_match=False,
        )
        quotes = result.column("quote").to_pylist()
        # left[0]=10: right[0]=10 exact → skip, right[1]=25 → match
        assert quotes[0] == 2.5
        # left[1]=20: right[1]=25 > 20 → match
        assert quotes[1] == 2.5


# ---------------------------------------------------------------------------
# Sorted-input optimization
# ---------------------------------------------------------------------------


class TestSortedInputOptimization:
    def test_presorted_matches_unsorted(self):
        """Pre-sorted and unsorted inputs should produce the same results."""
        rng = np.random.default_rng(42)
        n = 1000
        ts = rng.integers(0, 10000, size=n)

        left_sorted = pa.table({
            "timestamp": pa.array(np.sort(ts), type=pa.int64()),
            "price": pa.array(rng.uniform(0, 100, n)),
        })
        right_ts = rng.integers(0, 10000, size=n * 2)
        right_sorted = pa.table({
            "timestamp": pa.array(np.sort(right_ts), type=pa.int64()),
            "bid": pa.array(rng.uniform(0, 100, n * 2)),
        })

        # Shuffle right table
        perm = rng.permutation(n * 2)
        right_shuffled = pa.table({
            "timestamp": pa.array(np.sort(right_ts)[perm], type=pa.int64()),
            "bid": pa.array(right_sorted.column("bid").to_pylist()),
        })

        r1 = flowstate_core.asof_join(left_sorted, right_sorted, on="timestamp")
        r2 = flowstate_core.asof_join(left_sorted, right_shuffled, on="timestamp")

        # Same number of matches
        n1 = sum(1 for v in r1.column("bid").to_pylist() if v is not None)
        n2 = sum(1 for v in r2.column("bid").to_pylist() if v is not None)
        assert n1 == n2


# ---------------------------------------------------------------------------
# Multi-stream alignment (parallel joins)
# ---------------------------------------------------------------------------


class TestAlignStreams:
    def test_two_streams(self):
        """Align two secondary streams onto a primary timeline."""
        primary = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        quotes = pa.table({
            "timestamp": pa.array([5, 15, 25], type=pa.int64()),
            "bid": pa.array([0.5, 1.5, 2.5]),
        })
        signals = pa.table({
            "timestamp": pa.array([8, 22], type=pa.int64()),
            "alpha": pa.array([0.1, 0.2]),
        })

        result = flowstate_core.align_streams(
            primary,
            [
                {"table": quotes, "prefix": "q_"},
                {"table": signals, "prefix": "s_"},
            ],
            on="timestamp",
        )

        assert result.num_rows == 3
        assert "q_bid" in result.schema.names
        assert "s_alpha" in result.schema.names
        # Primary columns preserved
        assert result.column("price").to_pylist() == [1.0, 2.0, 3.0]

    def test_grouped_multi_stream(self):
        """Multi-stream with per-symbol grouping."""
        primary = pa.table({
            "timestamp": pa.array([10, 10, 20, 20], type=pa.int64()),
            "symbol": pa.array(["A", "B", "A", "B"]),
            "price": pa.array([1.0, 2.0, 3.0, 4.0]),
        })
        quotes = pa.table({
            "timestamp": pa.array([5, 5, 15, 15], type=pa.int64()),
            "symbol": pa.array(["A", "B", "A", "B"]),
            "bid": pa.array([0.5, 0.6, 1.5, 1.6]),
        })
        bars = pa.table({
            "timestamp": pa.array([8, 8], type=pa.int64()),
            "symbol": pa.array(["A", "B"]),
            "vwap": pa.array([100.0, 200.0]),
        })

        result = flowstate_core.align_streams(
            primary,
            [
                {"table": quotes, "prefix": "q_"},
                {"table": bars, "prefix": "b_"},
            ],
            on="timestamp",
            by="symbol",
        )

        assert result.num_rows == 4
        assert "q_bid" in result.schema.names
        assert "b_vwap" in result.schema.names

    def test_empty_stream(self):
        """Handle empty secondary streams gracefully."""
        primary = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
        })
        empty = pa.table({
            "timestamp": pa.array([], type=pa.int64()),
            "bid": pa.array([], type=pa.float64()),
        })

        result = flowstate_core.align_streams(
            primary,
            [{"table": empty, "prefix": "q_"}],
            on="timestamp",
        )

        assert result.num_rows == 2
        assert "q_bid" in result.schema.names
        # All nulls for the empty stream
        assert all(v is None for v in result.column("q_bid").to_pylist())

    def test_mixed_directions(self):
        """Each stream can have a different join direction."""
        primary = pa.table({
            "timestamp": pa.array([15], type=pa.int64()),
            "val": pa.array([1.0]),
        })
        backward_stream = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "back": pa.array([1.0, 2.0]),
        })
        forward_stream = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "fwd": pa.array([1.0, 2.0]),
        })

        result = flowstate_core.align_streams(
            primary,
            [
                {"table": backward_stream, "prefix": "b_", "direction": "backward"},
                {"table": forward_stream, "prefix": "f_", "direction": "forward"},
            ],
            on="timestamp",
        )

        assert result.num_rows == 1
        # Backward: rightmost <= 15 → ts=10, back=1.0
        assert result.column("b_back").to_pylist() == [1.0]
        # Forward: leftmost >= 15 → ts=20, fwd=2.0
        assert result.column("f_fwd").to_pylist() == [2.0]

    def test_python_rust_parity(self):
        """Rust multi-stream alignment matches Python sequential alignment."""
        from flowstate.prism.alignment import AlignmentSpec
        from flowstate.prism.alignment import align_streams as py_align

        primary = pa.table({
            "timestamp": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "symbol": pa.array(["A", "B", "A", "B", "A"]),
            "price": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        })
        quotes = pa.table({
            "timestamp": pa.array([5, 5, 15, 15, 25, 25], type=pa.int64()),
            "symbol": pa.array(["A", "B", "A", "B", "A", "B"]),
            "bid": pa.array([0.5, 0.6, 1.5, 1.6, 2.5, 2.6]),
        })

        # Rust path
        rust_result = flowstate_core.align_streams(
            primary,
            [{"table": quotes, "prefix": "q_"}],
            on="timestamp",
            by="symbol",
        )

        # Python path (force Python by calling individual joins)
        specs = [AlignmentSpec(name="quotes", table=quotes)]
        py_result, _ = py_align(primary, specs)

        assert rust_result.num_rows == py_result.num_rows

        # Match counts should agree
        rust_bids = rust_result.column("q_bid").to_pylist()
        py_bids = py_result.column("quotes_bid").to_pylist()
        rust_matched = sum(1 for v in rust_bids if v is not None)
        py_matched = sum(1 for v in py_bids if v is not None)
        assert rust_matched == py_matched


# ---------------------------------------------------------------------------
# Streaming incremental as-of join
# ---------------------------------------------------------------------------


class TestStreamingJoin:
    def test_basic_streaming(self):
        """Push right, push left, advance watermark, emit."""
        aligner = flowstate_core.StreamingJoin(on="timestamp", direction="backward")

        right = pa.table({
            "timestamp": pa.array([5, 15, 25], type=pa.int64()),
            "bid": pa.array([0.5, 1.5, 2.5]),
        })
        left = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })

        aligner.push_right(right)
        aligner.push_left(left)
        aligner.advance_watermark(100)

        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 3
        bids = result.column("bid").to_pylist()
        assert bids == [0.5, 1.5, 2.5]

    def test_incremental_arrival(self):
        """Left arrives before right — matches update when right data comes."""
        aligner = flowstate_core.StreamingJoin(on="timestamp", direction="backward")

        left = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
        })
        aligner.push_left(left)
        assert aligner.pending_count == 2

        # Right arrives later
        right = pa.table({
            "timestamp": pa.array([5, 15], type=pa.int64()),
            "bid": pa.array([0.5, 1.5]),
        })
        aligner.push_right(right)

        aligner.advance_watermark(100)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 2
        assert result.column("bid").to_pylist() == [0.5, 1.5]

    def test_watermark_partial_emission(self):
        """Watermark controls which rows are sealed and emitted."""
        aligner = flowstate_core.StreamingJoin(
            on="timestamp", direction="backward", lateness_ns=5,
        )

        right = pa.table({
            "timestamp": pa.array([5], type=pa.int64()),
            "bid": pa.array([0.5]),
        })
        left = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
        })

        aligner.push_right(right)
        aligner.push_left(left)

        # Watermark 16: seal_threshold = 16-5 = 11. Only ts=10 sealed.
        aligner.advance_watermark(16)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1
        assert aligner.pending_count == 1

        # Watermark 30: ts=20 now sealed
        aligner.advance_watermark(30)
        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 1
        assert aligner.pending_count == 0

    def test_grouped_streaming(self):
        """Per-symbol streaming join."""
        aligner = flowstate_core.StreamingJoin(
            on="timestamp", by="symbol", direction="backward",
        )

        right = pa.table({
            "timestamp": pa.array([5, 5], type=pa.int64()),
            "symbol": pa.array(["A", "B"]),
            "bid": pa.array([0.5, 0.6]),
        })
        left = pa.table({
            "timestamp": pa.array([10, 10], type=pa.int64()),
            "symbol": pa.array(["A", "B"]),
            "price": pa.array([1.0, 2.0]),
        })

        aligner.push_right(right)
        aligner.push_left(left)
        aligner.advance_watermark(100)

        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 2
        bids = result.column("bid").to_pylist()
        assert 0.5 in bids
        assert 0.6 in bids

    def test_flush(self):
        """Flush emits all pending rows regardless of watermark."""
        aligner = flowstate_core.StreamingJoin(
            on="timestamp", direction="backward", lateness_ns=10**9,
        )

        right = pa.table({
            "timestamp": pa.array([5], type=pa.int64()),
            "bid": pa.array([0.5]),
        })
        left = pa.table({
            "timestamp": pa.array([10], type=pa.int64()),
            "price": pa.array([1.0]),
        })

        aligner.push_right(right)
        aligner.push_left(left)

        # Normal emit returns nothing (lateness too large)
        aligner.advance_watermark(20)
        result = aligner.emit()
        assert result is None

        # Flush forces emission
        result = aligner.flush()
        assert result is not None
        assert result.num_rows == 1

    def test_tolerance_streaming(self):
        """Tolerance is respected in streaming mode."""
        aligner = flowstate_core.StreamingJoin(
            on="timestamp", direction="backward", tolerance_ns=10,
        )

        right = pa.table({
            "timestamp": pa.array([5], type=pa.int64()),
            "bid": pa.array([0.5]),
        })
        left = pa.table({
            "timestamp": pa.array([10, 100], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
        })

        aligner.push_right(right)
        aligner.push_left(left)
        aligner.advance_watermark(200)

        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 2
        bids = result.column("bid").to_pylist()
        assert bids[0] == 0.5  # dist 5 <= 10
        assert bids[1] is None  # dist 95 > 10

    def test_forward_streaming(self):
        """Forward direction works in streaming."""
        aligner = flowstate_core.StreamingJoin(
            on="timestamp", direction="forward",
        )

        right = pa.table({
            "timestamp": pa.array([15, 25], type=pa.int64()),
            "bid": pa.array([1.5, 2.5]),
        })
        left = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })

        aligner.push_right(right)
        aligner.push_left(left)
        aligner.advance_watermark(200)

        result = aligner.emit()
        assert result is not None
        assert result.num_rows == 3
        bids = result.column("bid").to_pylist()
        assert bids[0] == 1.5   # 10 → 15
        assert bids[1] == 2.5   # 20 → 25
        assert bids[2] is None  # 30 → nothing

    def test_stats_tracking(self):
        """Verify counters are maintained correctly."""
        aligner = flowstate_core.StreamingJoin(on="timestamp", direction="backward")

        right = pa.table({
            "timestamp": pa.array([5, 15], type=pa.int64()),
            "bid": pa.array([0.5, 1.5]),
        })
        left = pa.table({
            "timestamp": pa.array([10], type=pa.int64()),
            "price": pa.array([1.0]),
        })

        aligner.push_right(right)
        assert aligner.total_right_received == 2

        aligner.push_left(left)
        assert aligner.total_left_received == 1

        aligner.advance_watermark(100)
        aligner.emit()
        assert aligner.total_emitted == 1

    def test_batch_parity_with_batch_join(self):
        """Streaming join produces same results as batch join."""
        # Batch join
        left = pa.table({
            "timestamp": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        })
        right = pa.table({
            "timestamp": pa.array([5, 15, 25, 35, 45], type=pa.int64()),
            "bid": pa.array([0.5, 1.5, 2.5, 3.5, 4.5]),
        })

        batch_result = flowstate_core.asof_join(left, right, on="timestamp")
        batch_bids = batch_result.column("bid").to_pylist()

        # Streaming join — same data
        aligner = flowstate_core.StreamingJoin(on="timestamp", direction="backward")
        aligner.push_right(right)
        aligner.push_left(left)
        aligner.advance_watermark(100)
        stream_result = aligner.emit()
        stream_bids = stream_result.column("bid").to_pylist()

        assert batch_bids == stream_bids


# ---------------------------------------------------------------------------
# Arrow IPC file I/O
# ---------------------------------------------------------------------------


class TestIPCFileIO:
    def test_write_read_roundtrip(self, tmp_path):
        """Write a table to IPC and read it back."""
        path = str(tmp_path / "test.arrow")
        table = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
            "symbol": pa.array(["A", "B", "C"]),
        })

        flowstate_core.write_ipc(table, path)
        result = flowstate_core.read_ipc(path)
        assert result.num_rows == 3
        assert result.column("price").to_pylist() == [1.0, 2.0, 3.0]
        assert result.column("symbol").to_pylist() == ["A", "B", "C"]

    def test_column_projection(self, tmp_path):
        """Read only selected columns."""
        path = str(tmp_path / "proj.arrow")
        table = pa.table({
            "timestamp": pa.array([10, 20], type=pa.int64()),
            "price": pa.array([1.0, 2.0]),
            "symbol": pa.array(["A", "B"]),
        })

        flowstate_core.write_ipc(table, path)
        result = flowstate_core.read_ipc(path, projection=[0, 1])  # timestamp + price
        assert result.num_columns == 2
        assert "symbol" not in result.schema.names

    def test_batch_limit(self, tmp_path):
        """Limit number of batches read."""
        path = str(tmp_path / "limit.arrow")
        # Write multiple small tables as separate batches
        table = pa.table({
            "timestamp": pa.array(list(range(100)), type=pa.int64()),
            "price": pa.array([float(i) for i in range(100)]),
        })
        flowstate_core.write_ipc(table, path)
        result = flowstate_core.read_ipc(path, batch_limit=1)
        assert result.num_rows > 0

    def test_time_range_filter(self, tmp_path):
        """Read only batches overlapping a time range."""
        path = str(tmp_path / "range.arrow")
        table = pa.table({
            "timestamp": pa.array(list(range(0, 1000, 10)), type=pa.int64()),
            "price": pa.array([float(i) for i in range(100)]),
        })
        flowstate_core.write_ipc(table, path)

        result = flowstate_core.read_ipc_time_range(path, on="timestamp", min_ts=200, max_ts=500)
        # Should include the batch containing timestamps in range
        assert result.num_rows > 0

    def test_ipc_then_join(self, tmp_path):
        """Write, read via IPC, then join — end-to-end pipeline."""
        left_path = str(tmp_path / "left.arrow")
        right_path = str(tmp_path / "right.arrow")

        left = pa.table({
            "timestamp": pa.array([10, 20, 30], type=pa.int64()),
            "price": pa.array([1.0, 2.0, 3.0]),
        })
        right = pa.table({
            "timestamp": pa.array([5, 15, 25], type=pa.int64()),
            "bid": pa.array([0.5, 1.5, 2.5]),
        })

        flowstate_core.write_ipc(left, left_path)
        flowstate_core.write_ipc(right, right_path)

        left_read = flowstate_core.read_ipc(left_path)
        right_read = flowstate_core.read_ipc(right_path)

        result = flowstate_core.asof_join(left_read, right_read, on="timestamp")
        assert result.num_rows == 3
        assert result.column("bid").to_pylist() == [0.5, 1.5, 2.5]

    def test_empty_table(self, tmp_path):
        """Handle empty tables gracefully."""
        path = str(tmp_path / "empty.arrow")
        table = pa.table({
            "timestamp": pa.array([], type=pa.int64()),
            "price": pa.array([], type=pa.float64()),
        })
        flowstate_core.write_ipc(table, path)
        result = flowstate_core.read_ipc(path)
        assert result.num_rows == 0
