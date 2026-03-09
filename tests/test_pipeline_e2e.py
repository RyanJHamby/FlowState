"""End-to-end tests for the top-level Pipeline and ReplaySession APIs."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from flowstate import Pipeline, ReplaySession, Schema
from flowstate.schema.types import TRADE_SCHEMA


def _make_trade_batch(n: int = 10, symbol: str = "AAPL") -> pa.RecordBatch:
    ts = 1705320000 * 10**9
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


class TestPipeline:
    def test_build(self, tmp_path: Path):
        pipeline = Pipeline(data_dir=tmp_path / "data").build()
        assert pipeline.data_dir == tmp_path / "data"

    def test_fluent_api(self, tmp_path: Path):
        pipeline = (
            Pipeline(data_dir=tmp_path / "data")
            .add_source("polygon", api_key="test")
            .subscribe(["AAPL", "MSFT"])
            .build()
        )
        assert pipeline.get_writer() is not None

    def test_write_and_flush(self, tmp_path: Path):
        pipeline = Pipeline(data_dir=tmp_path / "data").build()
        batch = _make_trade_batch(10)
        pipeline.write(batch, "trade")
        files = pipeline.flush()
        assert len(files) >= 1

    def test_write_before_build_raises(self, tmp_path: Path):
        pipeline = Pipeline(data_dir=tmp_path / "data")
        batch = _make_trade_batch(5)
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.write(batch, "trade")

    def test_close(self, tmp_path: Path):
        pipeline = Pipeline(data_dir=tmp_path / "data").build()
        batch = _make_trade_batch(10)
        pipeline.write(batch, "trade")
        files = pipeline.close()
        assert len(files) >= 1

    def test_metrics(self, tmp_path: Path):
        pipeline = Pipeline(data_dir=tmp_path / "data").build()
        batch = _make_trade_batch(10)
        pipeline.write(batch, "trade")
        snap = pipeline.metrics.snapshot()
        assert len(snap["throughput"]) >= 1


class TestReplaySession:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        pipeline = Pipeline(data_dir=tmp_path / "data").build()
        for sym in ["AAPL", "MSFT"]:
            batch = _make_trade_batch(20, symbol=sym)
            pipeline.write(batch, "trade")
        pipeline.close()
        return tmp_path / "data"

    def test_replay_all(self, data_dir: Path):
        session = ReplaySession(data_dir)
        batches = list(session)
        total = sum(b.num_rows for b in batches)
        assert total == 40

    def test_fluent_filter(self, data_dir: Path):
        session = (
            ReplaySession(data_dir)
            .symbols(["AAPL"])
            .data_types(["trade"])
            .batch_size(10)
        )
        batches = list(session)
        total = sum(b.num_rows for b in batches)
        assert total == 20

    def test_count(self, data_dir: Path):
        session = ReplaySession(data_dir)
        assert session.count() == 40

    def test_to_dataset(self, data_dir: Path):
        session = ReplaySession(data_dir).data_types(["trade"])
        ds = session.to_dataset(numeric_columns=["price", "size"])
        batches = list(ds)
        assert len(batches) >= 1


class TestSchema:
    def test_trade_schema(self):
        schema = Schema.trade()
        assert isinstance(schema, pa.Schema)
        assert "symbol" in schema.names

    def test_quote_schema(self):
        schema = Schema.quote()
        assert "bid_price" in schema.names

    def test_bar_schema(self):
        schema = Schema.bar()
        assert "open" in schema.names
