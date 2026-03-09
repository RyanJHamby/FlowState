"""Tests for GPUDirect Storage integration with CPU fallback."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.prism.gpu_direct import GPUDirectConfig, GPUDirectReader, gpu_available


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    table = pa.table({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "price": [185.50, 370.00, 140.25],
        "volume": [1000.0, 2000.0, 1500.0],
    })
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


class TestGPUAvailable:
    def test_returns_bool(self):
        result = gpu_available()
        assert isinstance(result, bool)


class TestGPUDirectReader:
    def test_cpu_fallback(self, parquet_file: Path):
        config = GPUDirectConfig(enable_gpu=False)
        reader = GPUDirectReader(config)
        assert not reader.is_gpu_enabled

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

    def test_to_device_cpu_fallback(self, parquet_file: Path):
        reader = GPUDirectReader(GPUDirectConfig(enable_gpu=False))
        table = reader.read_parquet(parquet_file)
        batch = table.to_batches()[0]
        result = reader.to_device(batch)
        # In CPU mode, returns the batch unchanged
        assert result is batch

    def test_default_config(self):
        reader = GPUDirectReader()
        # Without GPU libs, should fall back to CPU
        if not gpu_available():
            assert not reader.is_gpu_enabled
