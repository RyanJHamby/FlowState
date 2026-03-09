"""Tests for the historical replay engine.

Covers:
- Hive partition pruning (type, date)
- Row-group-level predicate pushdown via Parquet statistics
- Global time ordering across multiple files (k-way merge)
- Streaming column projection
- Edge cases: empty dirs, missing columns, single-row files
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.prism.replay import (
    PartitionInfo,
    ReplayConfig,
    ReplayEngine,
    ReplayFilter,
    ReplayStats,
    TimeRange,
    _parse_partition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS_BASE = 1705320000 * 10**9  # 2024-01-15 12:00 UTC in ns
TS_DAY = 86_400 * 10**9       # 1 day in ns


def _write_partition(
    base: Path,
    data_type: str,
    date: str,
    bucket: int,
    table: pa.Table,
    filename: str = "part-00000.parquet",
    row_group_size: int | None = None,
) -> Path:
    """Write a Parquet file into a Hive partition directory."""
    d = base / f"type={data_type}" / f"date={date}" / f"bucket={bucket:04d}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    pq.write_table(table, path, row_group_size=row_group_size or table.num_rows)
    return path


def _trade_table(
    symbols: list[str],
    ts_offsets_ns: list[int],
    prices: list[float] | None = None,
) -> pa.Table:
    """Build a trade table with explicit timestamps."""
    n = len(symbols)
    assert len(ts_offsets_ns) == n
    return pa.table({
        "symbol": symbols,
        "timestamp": pa.array(ts_offsets_ns, type=pa.timestamp("ns", tz="UTC")),
        "price": prices or [100.0 + i for i in range(n)],
        "size": [100.0] * n,
        "source": ["test"] * n,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data(tmp_path: Path) -> Path:
    """Single-day, single-file dataset with 4 trades and 2 quotes."""
    base = tmp_path / "data"
    _write_partition(
        base, "trade", "2024-01-15", 1,
        _trade_table(
            ["AAPL", "MSFT", "AAPL", "GOOG"],
            [TS_BASE + i * 10**6 for i in range(4)],
        ),
    )
    _write_partition(
        base, "quote", "2024-01-15", 1,
        pa.table({
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pa.array(
                [TS_BASE + i * 10**6 for i in range(2)],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "bid_price": [185.40, 369.90],
            "ask_price": [185.60, 370.10],
            "source": ["test"] * 2,
        }),
    )
    return base


@pytest.fixture
def multi_day_data(tmp_path: Path) -> Path:
    """Three days of trade data across separate partitions."""
    base = tmp_path / "data"
    for day_offset in range(3):
        date = f"2024-01-{15 + day_offset:02d}"
        day_ts = TS_BASE + day_offset * TS_DAY
        _write_partition(
            base, "trade", date, 1,
            _trade_table(
                ["AAPL"] * 5,
                [day_ts + i * 10**6 for i in range(5)],
                [100.0 + day_offset * 10 + i for i in range(5)],
            ),
        )
    return base


@pytest.fixture
def multi_file_merge_data(tmp_path: Path) -> Path:
    """Two files on the same date with interleaved timestamps for merge testing."""
    base = tmp_path / "data"

    # File 1: AAPL at even offsets
    _write_partition(
        base, "trade", "2024-01-15", 1,
        _trade_table(
            ["AAPL"] * 4,
            [TS_BASE + i * 2 * 10**6 for i in range(4)],  # 0, 2, 4, 6 ms
        ),
        filename="part-00000.parquet",
    )
    # File 2: MSFT at odd offsets
    _write_partition(
        base, "trade", "2024-01-15", 2,
        _trade_table(
            ["MSFT"] * 4,
            [TS_BASE + (i * 2 + 1) * 10**6 for i in range(4)],  # 1, 3, 5, 7 ms
        ),
        filename="part-00000.parquet",
    )
    return base


@pytest.fixture
def row_group_data(tmp_path: Path) -> Path:
    """File with multiple row groups for row-group pruning tests."""
    base = tmp_path / "data"
    d = base / "type=trade" / "date=2024-01-15" / "bucket=0001"
    d.mkdir(parents=True)

    # 3 row groups, each with 10 rows at distinct timestamp ranges
    rows_per_rg = 10
    symbols = ["AAPL"] * rows_per_rg
    tables = []
    for rg in range(3):
        rg_start = TS_BASE + rg * 100 * 10**6  # 0ms, 100ms, 200ms
        tables.append(_trade_table(
            symbols,
            [rg_start + i * 10**6 for i in range(rows_per_rg)],
        ))
    full_table = pa.concat_tables(tables)
    pq.write_table(full_table, d / "part-00000.parquet", row_group_size=rows_per_rg)
    return base


# ---------------------------------------------------------------------------
# Partition parsing tests
# ---------------------------------------------------------------------------

class TestParsePartition:
    def test_valid_path(self, tmp_path: Path):
        p = tmp_path / "type=trade" / "date=2024-01-15" / "bucket=0042" / "part.parquet"
        info = _parse_partition(p, tmp_path)
        assert info is not None
        assert info.data_type == "trade"
        assert info.date == "2024-01-15"
        assert info.bucket == 42

    def test_non_hive_path(self, tmp_path: Path):
        p = tmp_path / "random" / "file.parquet"
        info = _parse_partition(p, tmp_path)
        assert info is None


class TestTimeRange:
    def test_contains_date_within_range(self):
        tr = TimeRange(start_ns=TS_BASE, end_ns=TS_BASE + TS_DAY)
        assert tr.contains_date("2024-01-15")

    def test_contains_date_before_range(self):
        tr = TimeRange(start_ns=TS_BASE + TS_DAY, end_ns=TS_BASE + 2 * TS_DAY)
        assert not tr.contains_date("2024-01-14")

    def test_contains_date_after_range(self):
        tr = TimeRange(start_ns=TS_BASE - TS_DAY, end_ns=TS_BASE)
        assert not tr.contains_date("2024-01-16")

    def test_open_start(self):
        tr = TimeRange(end_ns=TS_BASE + TS_DAY)
        assert tr.contains_date("2024-01-15")
        assert not tr.contains_date("2024-01-17")

    def test_open_end(self):
        tr = TimeRange(start_ns=TS_BASE + TS_DAY)
        assert tr.contains_date("2024-01-16")
        assert not tr.contains_date("2024-01-14")

    def test_invalid_date_not_pruned(self):
        tr = TimeRange(start_ns=TS_BASE)
        assert tr.contains_date("not-a-date")


# ---------------------------------------------------------------------------
# File discovery / partition pruning
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_discover_all(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        files = engine.discover_files()
        assert len(files) == 2

    def test_prune_by_type(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        files = engine.discover_files(ReplayFilter(data_types=["trade"]))
        assert len(files) == 1
        assert "type=trade" in str(files[0])

    def test_prune_by_date(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # Only request Jan 16 (exclusive end before Jan 17 starts)
        jan16_start = TS_BASE + TS_DAY
        jan16_end = TS_BASE + TS_DAY + 10 * 10**6  # Just past last row on Jan 16
        tr = TimeRange(start_ns=jan16_start, end_ns=jan16_end)
        files = engine.discover_files(ReplayFilter(time_range=tr))
        assert len(files) == 1
        assert "date=2024-01-16" in str(files[0])

    def test_prune_eliminates_all(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # Time range way in the future
        tr = TimeRange(start_ns=TS_BASE + 100 * TS_DAY)
        files = engine.discover_files(ReplayFilter(time_range=tr))
        assert len(files) == 0

    def test_combined_type_and_date(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        tr = TimeRange(start_ns=TS_BASE, end_ns=TS_BASE + TS_DAY)
        files = engine.discover_files(ReplayFilter(data_types=["quote"], time_range=tr))
        assert len(files) == 1
        assert "type=quote" in str(files[0])


# ---------------------------------------------------------------------------
# Basic replay (backwards-compatible with original tests)
# ---------------------------------------------------------------------------

class TestReplayBasic:
    def test_replay_all(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        total = sum(b.num_rows for b in engine.replay())
        assert total == 6

    def test_replay_with_symbol_filter(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        f = ReplayFilter(symbols=["AAPL"], data_types=["trade"])
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 2

    def test_replay_time_sorted(self, simple_data: Path):
        engine = ReplayEngine(simple_data, ReplayConfig(sort_by="timestamp"))
        f = ReplayFilter(data_types=["trade"])
        all_ts = []
        for b in engine.replay(f):
            all_ts.extend(b.column("timestamp").cast(pa.int64()).to_pylist())
        assert all_ts == sorted(all_ts)

    def test_replay_with_columns(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        f = ReplayFilter(data_types=["trade"], columns=["symbol", "price"])
        for b in engine.replay(f):
            assert "symbol" in b.schema.names
            assert "price" in b.schema.names

    def test_batch_size(self, simple_data: Path):
        engine = ReplayEngine(simple_data, ReplayConfig(batch_size=2))
        f = ReplayFilter(data_types=["trade"])
        for b in engine.replay(f):
            assert b.num_rows <= 2

    def test_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        engine = ReplayEngine(empty)
        assert list(engine.replay()) == []

    def test_data_dir_property(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        assert engine.data_dir == simple_data


# ---------------------------------------------------------------------------
# Time-range filtering
# ---------------------------------------------------------------------------

class TestTimeRangeReplay:
    def test_filter_single_day(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # Only Jan 16
        tr = TimeRange(
            start_ns=TS_BASE + TS_DAY,
            end_ns=TS_BASE + 2 * TS_DAY,
        )
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 5  # 5 rows on Jan 16

    def test_filter_partial_day(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # First 3 rows of Jan 15
        tr = TimeRange(
            start_ns=TS_BASE,
            end_ns=TS_BASE + 3 * 10**6,
        )
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 3

    def test_filter_empty_result(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # Gap between days
        tr = TimeRange(
            start_ns=TS_BASE + 10 * 10**6,  # After last row of Jan 15
            end_ns=TS_BASE + TS_DAY,          # Before first row of Jan 16
        )
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 0

    def test_open_ended_start(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # Everything up to end of Jan 15
        tr = TimeRange(end_ns=TS_BASE + 10 * 10**6)
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 5  # Only Jan 15

    def test_open_ended_end(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        # Everything from Jan 17 onward
        tr = TimeRange(start_ns=TS_BASE + 2 * TS_DAY)
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 5  # Only Jan 17


# ---------------------------------------------------------------------------
# K-way merge across files
# ---------------------------------------------------------------------------

class TestMergeReplay:
    def test_global_time_ordering(self, multi_file_merge_data: Path):
        # batch_size=1 so merge can interleave individual rows across files
        engine = ReplayEngine(multi_file_merge_data, ReplayConfig(batch_size=1))
        f = ReplayFilter(data_types=["trade"])
        all_ts = []
        for b in engine.replay(f):
            all_ts.extend(b.column("timestamp").cast(pa.int64()).to_pylist())
        # 8 rows total (4 AAPL + 4 MSFT), globally sorted
        assert len(all_ts) == 8
        assert all_ts == sorted(all_ts)

    def test_merge_preserves_all_rows(self, multi_file_merge_data: Path):
        engine = ReplayEngine(multi_file_merge_data)
        total = sum(b.num_rows for b in engine.replay())
        assert total == 8

    def test_merge_with_symbol_filter(self, multi_file_merge_data: Path):
        engine = ReplayEngine(multi_file_merge_data)
        f = ReplayFilter(symbols=["AAPL"])
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 4

    def test_multi_day_global_order(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data, ReplayConfig(batch_size=100))
        all_ts = []
        for b in engine.replay():
            all_ts.extend(b.column("timestamp").cast(pa.int64()).to_pylist())
        assert len(all_ts) == 15
        assert all_ts == sorted(all_ts)


# ---------------------------------------------------------------------------
# Row-group pruning
# ---------------------------------------------------------------------------

class TestRowGroupPruning:
    def test_prune_early_row_groups(self, row_group_data: Path):
        engine = ReplayEngine(row_group_data)
        # Request only the 3rd row group (200ms+ offsets)
        tr = TimeRange(start_ns=TS_BASE + 200 * 10**6)
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 10  # Only 3rd row group

    def test_prune_late_row_groups(self, row_group_data: Path):
        engine = ReplayEngine(row_group_data)
        # Request only first row group (before 100ms)
        tr = TimeRange(end_ns=TS_BASE + 10 * 10**6)
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 10  # Only 1st row group

    def test_no_pruning_without_time_range(self, row_group_data: Path):
        engine = ReplayEngine(row_group_data)
        total = sum(b.num_rows for b in engine.replay())
        assert total == 30  # All 3 row groups

    def test_all_groups_pruned(self, row_group_data: Path):
        engine = ReplayEngine(row_group_data)
        # Time range entirely outside data
        tr = TimeRange(start_ns=TS_BASE + 999 * 10**9)
        f = ReplayFilter(time_range=tr)
        total = sum(b.num_rows for b in engine.replay(f))
        assert total == 0


# ---------------------------------------------------------------------------
# Counting (metadata-only)
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_all(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        assert engine.count() == 6

    def test_count_by_type(self, simple_data: Path):
        engine = ReplayEngine(simple_data)
        assert engine.count(ReplayFilter(data_types=["trade"])) == 4

    def test_count_with_date_pruning(self, multi_day_data: Path):
        engine = ReplayEngine(multi_day_data)
        tr = TimeRange(
            start_ns=TS_BASE + TS_DAY,
            end_ns=TS_BASE + 2 * TS_DAY,
        )
        # With time range, uses row-group level count
        assert engine.count(ReplayFilter(time_range=tr)) == 5

    def test_count_empty(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        engine = ReplayEngine(empty)
        assert engine.count() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_row_file(self, tmp_path: Path):
        base = tmp_path / "data"
        _write_partition(
            base, "trade", "2024-01-15", 1,
            _trade_table(["AAPL"], [TS_BASE]),
        )
        engine = ReplayEngine(base)
        batches = list(engine.replay())
        assert sum(b.num_rows for b in batches) == 1

    def test_non_hive_file_included(self, tmp_path: Path):
        """Files outside Hive layout are included unconditionally."""
        base = tmp_path / "data"
        base.mkdir()
        table = _trade_table(["AAPL"], [TS_BASE])
        pq.write_table(table, base / "loose.parquet")
        engine = ReplayEngine(base)
        files = engine.discover_files(ReplayFilter(data_types=["trade"]))
        assert len(files) == 1  # Included since it's not in Hive layout

    def test_column_projection_adds_sort_col(self, simple_data: Path):
        """Column projection must include the sort column for ordering."""
        engine = ReplayEngine(simple_data)
        f = ReplayFilter(data_types=["trade"], columns=["symbol", "price"])
        batches = list(engine.replay(f))
        # Timestamp is included for sorting even though not requested
        for b in batches:
            assert "timestamp" in b.schema.names

    def test_descending_sort(self, simple_data: Path):
        engine = ReplayEngine(
            simple_data, ReplayConfig(sort_by="timestamp", ascending=False)
        )
        f = ReplayFilter(data_types=["trade"])
        all_ts = []
        for b in engine.replay(f):
            all_ts.extend(b.column("timestamp").cast(pa.int64()).to_pylist())
        assert all_ts == sorted(all_ts, reverse=True)

    def test_missing_sort_column(self, tmp_path: Path):
        """Files without the sort column should still work."""
        base = tmp_path / "data"
        _write_partition(
            base, "trade", "2024-01-15", 1,
            pa.table({"symbol": ["AAPL"], "price": [185.5]}),
        )
        engine = ReplayEngine(base, ReplayConfig(sort_by="timestamp"))
        batches = list(engine.replay())
        assert sum(b.num_rows for b in batches) == 1
