"""Tests for the end-to-end GPU data feeding pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from flowstate import Pipeline
from flowstate.prism.gpu_pipeline import GPUDataPipeline, GPUPipelineConfig
from flowstate.prism.replay import ReplayConfig, ReplayFilter, TimeRange
from flowstate.schema.types import QUOTE_SCHEMA, TRADE_SCHEMA


def _ts(seconds: int) -> int:
    return seconds * 10**9


def _write_trades(data_dir: Path, symbols: list[str], n_per_symbol: int = 20) -> None:
    pipeline = Pipeline(data_dir=data_dir).build()
    base_ts = 1705320000 * 10**9
    for sym in symbols:
        batch = pa.RecordBatch.from_pydict(
            {
                "symbol": [sym] * n_per_symbol,
                "timestamp": pa.array(
                    [base_ts + i * 10**9 for i in range(n_per_symbol)],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "exchange_timestamp": pa.array(
                    [None] * n_per_symbol, type=pa.timestamp("ns", tz="UTC")
                ),
                "receive_timestamp": pa.array(
                    [base_ts] * n_per_symbol, type=pa.timestamp("ns", tz="UTC")
                ),
                "price": [185.50 + i * 0.01 for i in range(n_per_symbol)],
                "size": [100.0] * n_per_symbol,
                "exchange": ["XNAS"] * n_per_symbol,
                "conditions": [None] * n_per_symbol,
                "tape": ["A"] * n_per_symbol,
                "sequence": list(range(1, n_per_symbol + 1)),
                "trade_id": [f"t{i}" for i in range(n_per_symbol)],
                "source": ["polygon"] * n_per_symbol,
            },
            schema=TRADE_SCHEMA,
        )
        pipeline.write(batch, "trade")
    pipeline.close()


class TestGPUPipelineReplayOnly:
    """Test pipeline without alignment (replay → pin → prefetch)."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "data"
        _write_trades(d, ["AAPL", "MSFT"])
        return d

    def test_basic_run(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            numeric_columns=["price", "size"],
        )
        pipeline = GPUDataPipeline(config)

        batches = []
        for pb in pipeline.run():
            batches.append(pb)
            pb.release_to(pipeline.pool)

        total = sum(b.num_rows for b in batches)
        assert total == 40

    def test_with_filter(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            numeric_columns=["price"],
        )
        pipeline = GPUDataPipeline(config)

        rf = ReplayFilter(symbols=["AAPL"], data_types=["trade"])
        batches = []
        for pb in pipeline.run(rf):
            batches.append(pb)
            pb.release_to(pipeline.pool)

        total = sum(b.num_rows for b in batches)
        assert total == 20

    def test_pinned_columns(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            numeric_columns=["price", "size"],
        )
        pipeline = GPUDataPipeline(config)

        for pb in pipeline.run():
            assert "price" in pb.pinned_columns
            assert "size" in pb.pinned_columns
            prices = pb.column_numpy("price")
            assert isinstance(prices, np.ndarray)
            pb.release_to(pipeline.pool)

    def test_run_numpy(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            numeric_columns=["price", "size"],
        )
        pipeline = GPUDataPipeline(config)

        batches = list(pipeline.run_numpy())
        assert len(batches) >= 1
        for batch_dict in batches:
            assert "price" in batch_dict
            assert isinstance(batch_dict["price"], np.ndarray)

    def test_stats(self, data_dir: Path):
        config = GPUPipelineConfig(data_dir=data_dir)
        pipeline = GPUDataPipeline(config)

        for pb in pipeline.run():
            pb.release_to(pipeline.pool)

        stats = pipeline.stats
        assert stats.replay_batches > 0
        assert stats.output_batches > 0

    def test_empty_result(self, data_dir: Path):
        config = GPUPipelineConfig(data_dir=data_dir)
        pipeline = GPUDataPipeline(config)

        rf = ReplayFilter(symbols=["NONEXISTENT"])
        batches = list(pipeline.run(rf))
        assert len(batches) == 0


class TestGPUPipelineWithAlignment:
    """Test pipeline with temporal alignment."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "data"
        pipeline = Pipeline(data_dir=d).build()

        base_ts = 1705320000 * 10**9

        # Write trades
        trade_batch = pa.RecordBatch.from_pydict(
            {
                "symbol": ["AAPL"] * 10,
                "timestamp": pa.array(
                    [base_ts + i * 10**9 for i in range(10)],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "exchange_timestamp": pa.array(
                    [None] * 10, type=pa.timestamp("ns", tz="UTC")
                ),
                "receive_timestamp": pa.array(
                    [base_ts] * 10, type=pa.timestamp("ns", tz="UTC")
                ),
                "price": [185.50 + i * 0.01 for i in range(10)],
                "size": [100.0] * 10,
                "exchange": ["XNAS"] * 10,
                "conditions": [None] * 10,
                "tape": ["A"] * 10,
                "sequence": list(range(1, 11)),
                "trade_id": [f"t{i}" for i in range(10)],
                "source": ["polygon"] * 10,
            },
            schema=TRADE_SCHEMA,
        )
        pipeline.write(trade_batch, "trade")

        # Write quotes (every 2 seconds, offset by 500ms)
        n_quotes = 5
        quote_batch = pa.RecordBatch.from_pydict(
            {
                "symbol": ["AAPL"] * n_quotes,
                "timestamp": pa.array(
                    [base_ts + i * 2 * 10**9 + 500_000_000 for i in range(n_quotes)],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "exchange_timestamp": pa.array(
                    [None] * n_quotes, type=pa.timestamp("ns", tz="UTC")
                ),
                "receive_timestamp": pa.array(
                    [base_ts] * n_quotes, type=pa.timestamp("ns", tz="UTC")
                ),
                "bid_price": [185.40 + i * 0.01 for i in range(n_quotes)],
                "bid_size": [500.0] * n_quotes,
                "ask_price": [185.60 + i * 0.01 for i in range(n_quotes)],
                "ask_size": [500.0] * n_quotes,
                "bid_exchange": ["XNAS"] * n_quotes,
                "ask_exchange": ["XNAS"] * n_quotes,
                "conditions": [None] * n_quotes,
                "tape": ["A"] * n_quotes,
                "sequence": list(range(1, n_quotes + 1)),
                "source": ["polygon"] * n_quotes,
            },
            schema=QUOTE_SCHEMA,
        )
        pipeline.write(quote_batch, "quote")
        pipeline.close()

        return d

    def test_aligned_output_has_quote_columns(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            primary_type="trade",
            secondary_specs={"quote": ["bid_price", "ask_price"]},
        )
        pipeline = GPUDataPipeline(config)

        batches = []
        for pb in pipeline.run():
            batches.append(pb)
            pb.release_to(pipeline.pool)

        assert len(batches) >= 1
        # Check that quote columns were joined
        all_columns = set()
        for pb in batches:
            all_columns.update(pb.schema.names)
        assert "quote_bid_price" in all_columns
        assert "quote_ask_price" in all_columns

    def test_alignment_stats_populated(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            primary_type="trade",
            secondary_specs={"quote": ["bid_price"]},
        )
        pipeline = GPUDataPipeline(config)

        for pb in pipeline.run():
            pb.release_to(pipeline.pool)

        stats = pipeline.stats
        assert stats.aligned_rows > 0

    def test_aligned_with_tolerance(self, data_dir: Path):
        config = GPUPipelineConfig(
            data_dir=data_dir,
            primary_type="trade",
            secondary_specs={"quote": ["bid_price"]},
            tolerance_ns=1,  # Extremely tight tolerance → most won't match
        )
        pipeline = GPUDataPipeline(config)

        batches = []
        for pb in pipeline.run():
            batches.append(pb)
            pb.release_to(pipeline.pool)

        # Should still produce output (trades exist), just with null quote columns
        total = sum(b.num_rows for b in batches)
        assert total > 0
