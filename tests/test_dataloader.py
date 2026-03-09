"""Tests for PyTorch/JAX DataLoader adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.prism.dataloader import (
    ArrowBatchConverter,
    FlowStateIterableDataset,
    JAXDataIterator,
)
from flowstate.prism.replay import ReplayFilter


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    base = tmp_path / "data" / "type=trade" / "date=2024-01-15" / "bucket=0001"
    base.mkdir(parents=True)
    table = pa.table({
        "symbol": ["AAPL", "MSFT", "AAPL"],
        "timestamp": pa.array(
            [1705320000 * 10**9 + i for i in range(3)],
            type=pa.timestamp("ns", tz="UTC"),
        ),
        "price": [185.50, 370.00, 185.60],
        "size": [100.0, 200.0, 150.0],
    })
    pq.write_table(table, base / "part-00000.parquet")
    return tmp_path / "data"


class TestArrowBatchConverter:
    def test_to_numpy(self):
        batch = pa.RecordBatch.from_pydict({
            "price": [1.0, 2.0, 3.0],
            "size": [10.0, 20.0, 30.0],
            "symbol": ["A", "B", "C"],
        })
        result = ArrowBatchConverter.to_numpy(batch)
        assert isinstance(result["price"], np.ndarray)
        assert isinstance(result["symbol"], list)

    def test_to_numpy_selected_columns(self):
        batch = pa.RecordBatch.from_pydict({
            "price": [1.0, 2.0],
            "size": [10.0, 20.0],
        })
        result = ArrowBatchConverter.to_numpy(batch, columns=["price"])
        assert "price" in result
        assert "size" not in result

    def test_timestamp_to_int64(self):
        batch = pa.RecordBatch.from_pydict({
            "timestamp": pa.array([1705320000 * 10**9], type=pa.timestamp("ns", tz="UTC")),
        })
        result = ArrowBatchConverter.to_numpy(batch)
        assert result["timestamp"].dtype == np.int64


class TestFlowStateIterableDataset:
    def test_iteration(self, data_dir: Path):
        ds = FlowStateIterableDataset(str(data_dir), numeric_columns=["price", "size"])
        batches = list(ds)
        assert len(batches) >= 1
        assert "price" in batches[0]

    def test_len(self, data_dir: Path):
        ds = FlowStateIterableDataset(str(data_dir))
        assert len(ds) == 3

    def test_with_filter(self, data_dir: Path):
        f = ReplayFilter(symbols=["AAPL"])
        ds = FlowStateIterableDataset(str(data_dir), replay_filter=f)
        total = sum(batch["price"].shape[0] for batch in ds if isinstance(batch.get("price"), np.ndarray))
        assert total == 2


class TestJAXDataIterator:
    def test_fallback_to_numpy(self, data_dir: Path):
        it = JAXDataIterator(str(data_dir), numeric_columns=["price"])
        batches = list(it)
        assert len(batches) >= 1
        # Without JAX, should return numpy arrays
        if "price" in batches[0]:
            assert isinstance(batches[0]["price"], np.ndarray)
