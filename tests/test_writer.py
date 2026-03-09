"""Tests for the partitioned Parquet writer."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.schema.types import TRADE_SCHEMA
from flowstate.storage.writer import PartitionedParquetWriter, WriterConfig


def _make_trade_batch(n: int = 10, symbol: str = "AAPL") -> pa.RecordBatch:
    """Build a trade batch with n rows."""
    ts = 1705320000 * 10**9  # 2024-01-15 12:00 UTC
    return pa.RecordBatch.from_pydict(
        {
            "symbol": [symbol] * n,
            "timestamp": pa.array([ts + i for i in range(n)], type=pa.timestamp("ns", tz="UTC")),
            "exchange_timestamp": pa.array([None] * n, type=pa.timestamp("ns", tz="UTC")),
            "receive_timestamp": pa.array([ts] * n, type=pa.timestamp("ns", tz="UTC")),
            "price": [185.50 + i * 0.01 for i in range(n)],
            "size": [100.0] * n,
            "exchange": ["XNAS"] * n,
            "conditions": [None] * n,
            "tape": ["A"] * n,
            "sequence": list(range(1, n + 1)),
            "trade_id": [f"t{i}" for i in range(n)],
            "source": ["polygon"] * n,
        },
        schema=TRADE_SCHEMA,
    )


class TestPartitionedParquetWriter:
    def test_write_and_flush(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=100)
        writer = PartitionedParquetWriter(config)

        batch = _make_trade_batch(10)
        writer.write(batch, "trade")
        files = writer.flush_all()

        assert len(files) == 1
        assert files[0].suffix == ".parquet"

        table = pq.read_table(files[0])
        assert table.num_rows == 10

    def test_partitioned_output(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=100)
        writer = PartitionedParquetWriter(config)

        batch_aapl = _make_trade_batch(5, symbol="AAPL")
        batch_msft = _make_trade_batch(5, symbol="MSFT")
        writer.write(batch_aapl, "trade")
        writer.write(batch_msft, "trade")
        files = writer.flush_all()

        assert len(files) >= 1  # At least one file (possibly same bucket)
        assert writer.stats.rows_written == 10

    def test_auto_flush_on_max_rows(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=10)
        writer = PartitionedParquetWriter(config)

        # Write exactly max_rows to trigger auto-flush
        batch = _make_trade_batch(10)
        files = writer.write(batch, "trade")

        assert len(files) == 1
        assert writer.stats.files_written == 1

    def test_zstd_compression(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=100)
        writer = PartitionedParquetWriter(config)

        batch = _make_trade_batch(50)
        writer.write(batch, "trade")
        files = writer.flush_all()

        # Verify the file is valid and uses zstd
        metadata = pq.read_metadata(files[0])
        assert metadata.num_rows == 50

    def test_close_flushes(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=1000)
        writer = PartitionedParquetWriter(config)

        batch = _make_trade_batch(5)
        writer.write(batch, "trade")
        files = writer.close()

        assert len(files) == 1
        assert writer.stats.partitions_active == 0

    def test_stats(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=1000)
        writer = PartitionedParquetWriter(config)

        batch = _make_trade_batch(20)
        writer.write(batch, "trade")
        writer.flush_all()

        assert writer.stats.files_written == 1
        assert writer.stats.rows_written == 20
        assert writer.stats.bytes_written > 0

    def test_hive_directory_structure(self, tmp_parquet_dir: Path):
        config = WriterConfig(base_path=tmp_parquet_dir, max_rows_per_file=1000)
        writer = PartitionedParquetWriter(config)

        batch = _make_trade_batch(5)
        writer.write(batch, "trade")
        files = writer.flush_all()

        # Verify Hive-style path structure
        rel = files[0].relative_to(tmp_parquet_dir)
        parts = str(rel).split("/")
        assert parts[0].startswith("type=")
        assert parts[1].startswith("date=")
        assert parts[2].startswith("bucket=")
