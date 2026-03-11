"""Extended tests for GPUDirect Storage — config, multi-batch, edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.prism.gpu_direct import (
    GPUBatch,
    GPUDirectConfig,
    GPUDirectReader,
    TransferStats,
)


@pytest.fixture
def multi_col_parquet(tmp_path: Path) -> Path:
    table = pa.table({
        "timestamp": pa.array(range(0, 100_000, 1000), type=pa.int64()),
        "price": [100.0 + i * 0.01 for i in range(100)],
        "volume": list(range(100)),
        "bid": [99.0 + i * 0.01 for i in range(100)],
        "ask": [101.0 + i * 0.01 for i in range(100)],
        "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
    })
    path = tmp_path / "multi.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def large_binary(tmp_path: Path) -> Path:
    data = np.random.randn(10000).astype(np.float64)
    path = tmp_path / "large.bin"
    data.tofile(str(path))
    return path


class TestGPUDirectConfig:
    def test_defaults(self):
        config = GPUDirectConfig()
        assert config.batch_size == 65536
        assert config.num_streams == 2

    def test_custom_config(self):
        config = GPUDirectConfig(
            device_id=1,
            num_streams=4,
            batch_size=1024,
            gds_task_size=8 * 1024 * 1024,
            gpu_columns=["price", "volume"],
        )
        assert config.device_id == 1
        assert config.num_streams == 4
        assert config.gpu_columns == ["price", "volume"]


class TestTransferStats:
    def test_defaults(self):
        stats = TransferStats()
        assert stats.parquet_reads == 0
        assert stats.h2d_transfers == 0
        assert stats.bytes_transferred == 0
        assert stats.fallback_count == 0


class TestMultiBatch:
    def test_read_batches_multiple(self, multi_col_parquet: Path):
        reader = GPUDirectReader(
            GPUDirectConfig(enable_gpu=False, batch_size=30),
        )
        batches = reader.read_batches(multi_col_parquet)
        assert len(batches) >= 2
        total = sum(b.num_rows for b in batches)
        assert total == 100

    def test_to_device_multiple_batches(self, multi_col_parquet: Path):
        reader = GPUDirectReader(
            GPUDirectConfig(enable_gpu=False, batch_size=50),
        )
        batches = reader.read_batches(multi_col_parquet)
        gpu_batches = [reader.to_device(b) for b in batches]
        total = sum(gb.num_rows for gb in gpu_batches)
        assert total == 100

    def test_stats_accumulate(self, multi_col_parquet: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        reader.read_parquet(multi_col_parquet)
        reader.read_parquet(multi_col_parquet)
        assert reader.stats.parquet_reads == 2


class TestGPUBatchExtended:
    def test_empty_batch(self):
        batch = GPUBatch(schema=pa.schema([("x", pa.float64())]), num_rows=0)
        assert batch.num_rows == 0
        assert batch.column_names == []

    def test_columns_property(self):
        batch = GPUBatch(
            schema=pa.schema([("a", pa.float64()), ("b", pa.int64())]),
            num_rows=2,
            columns={"a": np.array([1.0, 2.0]), "b": np.array([10, 20])},
            cpu_columns={"sym": ["X", "Y"]},
        )
        assert set(batch.gpu_column_names) == {"a", "b"}
        assert batch.cpu_column("sym") == ["X", "Y"]

    def test_missing_cpu_column_raises(self):
        batch = GPUBatch(
            schema=pa.schema([("x", pa.float64())]),
            num_rows=1,
            columns={"x": np.array([1.0])},
        )
        with pytest.raises(KeyError):
            batch.cpu_column("nonexistent")


class TestBinaryReadEdgeCases:
    def test_read_float64(self, large_binary: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        result = reader.read_binary_to_gpu(large_binary, dtype=np.float64)
        assert len(result) == 10000
        assert result.dtype == np.float64

    def test_read_binary_count_zero(self, large_binary: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        result = reader.read_binary_to_gpu(large_binary, dtype=np.float64, count=0)
        assert len(result) == 0

    def test_read_binary_nonexistent_raises(self):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        with pytest.raises((FileNotFoundError, OSError)):
            reader.read_binary_to_gpu("/nonexistent.bin", dtype=np.float32)


class TestColumnSelection:
    def test_gpu_columns_only_selected(self, multi_col_parquet: Path):
        reader = GPUDirectReader(
            GPUDirectConfig(enable_gpu=False, gpu_columns=["price", "bid"]),
        )
        table = reader.read_parquet(multi_col_parquet)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)

        assert "price" in result.gpu_column_names
        assert "bid" in result.gpu_column_names
        # Others go to CPU
        assert "volume" in result.cpu_columns
        assert "symbol" in result.cpu_columns

    def test_all_numeric_on_gpu_by_default(self, multi_col_parquet: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(multi_col_parquet)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)

        # All numeric columns should be GPU columns
        assert "price" in result.gpu_column_names
        assert "volume" in result.gpu_column_names
        assert "timestamp" in result.gpu_column_names
