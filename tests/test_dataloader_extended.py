"""Extended tests for PyTorch/JAX DataLoader adapters."""

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
def multi_day_data(tmp_path: Path) -> Path:
    """Create test data spanning multiple days and types."""
    base = tmp_path / "data"
    for date in ["2024-01-15", "2024-01-16"]:
        for bucket in range(2):
            part_dir = base / "type=trade" / f"date={date}" / f"bucket={bucket:04d}"
            part_dir.mkdir(parents=True)
            table = pa.table({
                "symbol": ["AAPL", "MSFT"] * 5,
                "timestamp": pa.array(
                    [1705320000 * 10**9 + i * 1000 for i in range(10)],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "price": [185.0 + i * 0.1 for i in range(10)],
                "size": [100.0] * 10,
            })
            pq.write_table(table, part_dir / "data.parquet")
    return base


class TestArrowBatchConverterExtended:
    def test_missing_column_skipped(self):
        batch = pa.RecordBatch.from_pydict({"price": [1.0, 2.0]})
        result = ArrowBatchConverter.to_numpy(batch, columns=["price", "nonexistent"])
        assert "price" in result
        assert "nonexistent" not in result

    def test_integer_columns(self):
        batch = pa.RecordBatch.from_pydict({
            "count": pa.array([1, 2, 3], type=pa.int64()),
        })
        result = ArrowBatchConverter.to_numpy(batch)
        assert isinstance(result["count"], np.ndarray)
        assert result["count"].dtype in (np.int64, np.int32)

    def test_empty_batch(self):
        batch = pa.RecordBatch.from_pydict({
            "price": pa.array([], type=pa.float64()),
            "symbol": pa.array([], type=pa.string()),
        })
        result = ArrowBatchConverter.to_numpy(batch)
        assert len(result["price"]) == 0
        assert len(result["symbol"]) == 0

    def test_all_columns_by_default(self):
        batch = pa.RecordBatch.from_pydict({
            "a": [1.0],
            "b": [2.0],
            "c": ["x"],
        })
        result = ArrowBatchConverter.to_numpy(batch)
        assert set(result.keys()) == {"a", "b", "c"}


class TestFlowStateIterableDatasetExtended:
    def test_multi_file_iteration(self, multi_day_data: Path):
        ds = FlowStateIterableDataset(str(multi_day_data))
        batches = list(ds)
        total_rows = sum(
            b["price"].shape[0] if isinstance(b.get("price"), np.ndarray) else 0
            for b in batches
        )
        # 4 files × 10 rows = 40
        assert total_rows == 40

    def test_with_numeric_columns_only(self, multi_day_data: Path):
        ds = FlowStateIterableDataset(
            str(multi_day_data),
            numeric_columns=["price"],
        )
        for batch in ds:
            assert "price" in batch
            assert "size" not in batch

    def test_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = FlowStateIterableDataset(str(empty))
        batches = list(ds)
        assert batches == []

    def test_filter_by_symbol(self, multi_day_data: Path):
        f = ReplayFilter(symbols=["AAPL"])
        ds = FlowStateIterableDataset(str(multi_day_data), replay_filter=f)
        total = 0
        for batch in ds:
            if isinstance(batch.get("price"), np.ndarray):
                total += batch["price"].shape[0]
        # Each file has 5 AAPL rows, 4 files = 20
        assert total == 20


class TestJAXIteratorExtended:
    def test_multi_batch(self, multi_day_data: Path):
        it = JAXDataIterator(
            str(multi_day_data),
            numeric_columns=["price", "size"],
        )
        batches = list(it)
        assert len(batches) == 4  # 4 parquet files
        for b in batches:
            assert "price" in b

    def test_empty_data(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        it = JAXDataIterator(str(empty))
        assert list(it) == []
