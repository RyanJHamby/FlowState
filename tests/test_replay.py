"""Tests for the historical replay engine."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.prism.replay import ReplayConfig, ReplayEngine, ReplayFilter, TimeRange


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a test data directory with partitioned Parquet files."""
    base = tmp_path / "data"

    # Create trade partition
    trade_dir = base / "type=trade" / "date=2024-01-15" / "bucket=0001"
    trade_dir.mkdir(parents=True)
    trade_table = pa.table({
        "symbol": ["AAPL", "MSFT", "AAPL", "GOOG"],
        "timestamp": pa.array(
            [1705320000 * 10**9 + i * 10**6 for i in range(4)],
            type=pa.timestamp("ns", tz="UTC"),
        ),
        "price": [185.50, 370.00, 185.60, 140.25],
        "size": [100.0, 200.0, 150.0, 300.0],
        "source": ["test"] * 4,
    })
    pq.write_table(trade_table, trade_dir / "part-00000.parquet")

    # Create quote partition
    quote_dir = base / "type=quote" / "date=2024-01-15" / "bucket=0001"
    quote_dir.mkdir(parents=True)
    quote_table = pa.table({
        "symbol": ["AAPL", "MSFT"],
        "timestamp": pa.array(
            [1705320000 * 10**9 + i * 10**6 for i in range(2)],
            type=pa.timestamp("ns", tz="UTC"),
        ),
        "bid_price": [185.40, 369.90],
        "ask_price": [185.60, 370.10],
        "source": ["test"] * 2,
    })
    pq.write_table(quote_table, quote_dir / "part-00000.parquet")

    return base


class TestReplayEngine:
    def test_discover_files(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        files = engine.discover_files()
        assert len(files) == 2

    def test_discover_with_type_filter(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        f = ReplayFilter(data_types=["trade"])
        files = engine.discover_files(f)
        assert len(files) == 1
        assert "type=trade" in str(files[0])

    def test_replay_all(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        batches = list(engine.replay())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 6  # 4 trades + 2 quotes

    def test_replay_with_symbol_filter(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        f = ReplayFilter(symbols=["AAPL"], data_types=["trade"])
        batches = list(engine.replay(f))
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 2

    def test_replay_time_sorted(self, data_dir: Path):
        engine = ReplayEngine(data_dir, ReplayConfig(sort_by="timestamp"))
        f = ReplayFilter(data_types=["trade"])
        batches = list(engine.replay(f))
        assert len(batches) >= 1
        # Verify timestamps are sorted
        all_ts = []
        for b in batches:
            all_ts.extend(b.column("timestamp").cast(pa.int64()).to_pylist())
        assert all_ts == sorted(all_ts)

    def test_replay_with_columns(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        f = ReplayFilter(data_types=["trade"], columns=["symbol", "price"])
        batches = list(engine.replay(f))
        for b in batches:
            assert "symbol" in b.schema.names
            assert "price" in b.schema.names
            assert "size" not in b.schema.names

    def test_count(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        assert engine.count() == 6
        assert engine.count(ReplayFilter(data_types=["trade"])) == 4

    def test_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        engine = ReplayEngine(empty)
        batches = list(engine.replay())
        assert len(batches) == 0

    def test_batch_size(self, data_dir: Path):
        engine = ReplayEngine(data_dir, ReplayConfig(batch_size=2))
        f = ReplayFilter(data_types=["trade"])
        batches = list(engine.replay(f))
        for b in batches:
            assert b.num_rows <= 2

    def test_data_dir_property(self, data_dir: Path):
        engine = ReplayEngine(data_dir)
        assert engine.data_dir == data_dir
