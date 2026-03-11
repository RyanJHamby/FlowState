"""Tests for GPUDirect Storage integration with CPU fallback."""

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
    gds_available,
    gpu_available,
)


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    table = pa.table({
        "timestamp": pa.array([1_000_000, 2_000_000, 3_000_000], type=pa.int64()),
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "price": [185.50, 370.00, 140.25],
        "volume": [1000, 2000, 1500],
    })
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def binary_file(tmp_path: Path) -> Path:
    """Create a raw binary file with known float32 data."""
    data = np.arange(1000, dtype=np.float32)
    path = tmp_path / "test.bin"
    data.tofile(str(path))
    return path


class TestGPUAvailable:
    def test_returns_bool(self):
        assert isinstance(gpu_available(), bool)

    def test_gds_returns_bool(self):
        assert isinstance(gds_available(), bool)


class TestGPUDirectReader:
    def test_cpu_fallback(self, parquet_file: Path):
        config = GPUDirectConfig(enable_gpu=False)
        reader = GPUDirectReader(config)
        assert not reader.is_gpu_enabled
        assert not reader.is_gds_enabled

    def test_read_parquet(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file)
        assert table.num_rows == 3
        assert "symbol" in table.schema.names

    def test_read_with_columns(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file, columns=["symbol", "price"])
        assert table.num_columns == 2
        assert "volume" not in table.schema.names

    def test_read_nonexistent_raises(self):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        with pytest.raises(FileNotFoundError):
            reader.read_parquet("/nonexistent/file.parquet")

    def test_read_batches(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False, batch_size=2))
        batches = reader.read_batches(parquet_file)
        assert len(batches) >= 1
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 3

    def test_default_config(self):
        reader = GPUDirectReader()
        if not gpu_available():
            assert not reader.is_gpu_enabled


class TestToDeviceCPUFallback:
    """Test the CPU fallback path of to_device()."""

    def test_cpu_fallback_returns_gpu_batch(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)
        assert isinstance(result, GPUBatch)
        assert result.num_rows == 3

    def test_cpu_fallback_numeric_columns_are_numpy(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)
        # Numeric columns should be numpy arrays
        assert isinstance(result.columns["price"], np.ndarray)
        assert isinstance(result.columns["volume"], np.ndarray)
        assert isinstance(result.columns["timestamp"], np.ndarray)

    def test_cpu_fallback_string_columns_on_cpu(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)
        # String columns stay on CPU
        assert "symbol" in result.cpu_columns
        assert result.cpu_columns["symbol"] == ["AAPL", "MSFT", "GOOG"]

    def test_gpu_column_selection(self, parquet_file: Path):
        reader = GPUDirectReader(
            GPUDirectConfig(enable_gpu=False, gpu_columns=["price"])
        )
        table = reader.read_parquet(parquet_file)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)
        assert "price" in result.gpu_column_names
        # volume not selected — should be on CPU
        assert "volume" in result.cpu_columns

    def test_stats_tracking(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file)
        batch = table.to_batches()[0]
        reader.to_device(batch)
        assert reader.stats.parquet_reads == 1
        assert reader.stats.h2d_transfers == 1


class TestBinaryReadCPUFallback:
    """Test the binary read CPU fallback path."""

    def test_read_binary_fallback(self, binary_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        result = reader.read_binary_to_gpu(binary_file, dtype=np.float32)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1000
        np.testing.assert_array_equal(result, np.arange(1000, dtype=np.float32))

    def test_read_binary_partial(self, binary_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        result = reader.read_binary_to_gpu(
            binary_file, dtype=np.float32, count=100, file_offset=400
        )
        assert len(result) == 100
        # 400 bytes = 100 float32s, so starts at value 100
        np.testing.assert_array_equal(result, np.arange(100, 200, dtype=np.float32))

    def test_read_binary_stats(self, binary_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        reader.read_binary_to_gpu(binary_file, dtype=np.float32)
        assert reader.stats.fallback_count == 1

    def test_async_read_cpu_fallback(self, binary_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        future, data = reader.read_binary_async(binary_file, dtype=np.float32)
        assert future is None  # No async in CPU mode
        assert isinstance(data, np.ndarray)
        assert len(data) == 1000


class TestGPUBatch:
    def test_column_access(self):
        batch = GPUBatch(
            schema=pa.schema([("x", pa.float64())]),
            num_rows=3,
            columns={"x": np.array([1.0, 2.0, 3.0])},
            cpu_columns={"name": ["a", "b", "c"]},
        )
        np.testing.assert_array_equal(batch.gpu_column("x"), [1.0, 2.0, 3.0])
        assert batch.cpu_column("name") == ["a", "b", "c"]
        assert batch.column_names == ["x", "name"]
        assert batch.gpu_column_names == ["x"]

    def test_missing_column_raises(self):
        batch = GPUBatch(schema=pa.schema([]), num_rows=0)
        with pytest.raises(KeyError):
            batch.gpu_column("nonexistent")


class TestSynchronize:
    def test_sync_noop_without_gpu(self):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        # Should not raise
        reader.synchronize()
        reader.synchronize(stream_idx=0)
