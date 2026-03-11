"""Tests for the feature materializer."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from flowstate.store.catalog import (
    FeatureCatalog,
    FeatureDefinition,
    JoinDirection,
)
from flowstate.store.materializer import FeatureMaterializer


def _make_trade_table(n: int = 100) -> pa.Table:
    return pa.table({
        "timestamp": pa.array(range(0, n * 1_000_000, 1_000_000), type=pa.int64()),
        "price": [100.0 + i * 0.01 for i in range(n)],
        "volume": [10 * (i + 1) for i in range(n)],
        "symbol": ["AAPL"] * (n // 2) + ["MSFT"] * (n - n // 2),
    })


def _make_quote_table(n: int = 100) -> pa.Table:
    return pa.table({
        "timestamp": pa.array(range(500_000, n * 1_000_000 + 500_000, 1_000_000), type=pa.int64()),
        "bid_price": [99.9 + i * 0.01 for i in range(n)],
        "ask_price": [100.1 + i * 0.01 for i in range(n)],
        "symbol": ["AAPL"] * (n // 2) + ["MSFT"] * (n - n // 2),
    })


@pytest.fixture
def catalog() -> FeatureCatalog:
    cat = FeatureCatalog()
    cat.register(FeatureDefinition(
        name="trade_with_quote",
        primary_stream="trade",
        secondary_stream="quote",
        columns=["bid_price", "ask_price"],
        direction=JoinDirection.BACKWARD,
        tolerance_ns=2_000_000,
    ))
    cat.register(FeatureDefinition(
        name="trade_only",
        primary_stream="trade",
        columns=["price", "volume"],
    ))
    return cat


class TestMaterializeSingle:
    def test_primary_only_feature(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        mat.add_stream("quote", _make_quote_table())

        feature = catalog.get("trade_only")
        result = mat.materialize(feature)
        assert result.success
        assert result.num_rows == 100
        assert result.output_path.exists()

    def test_aligned_feature(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        mat.add_stream("quote", _make_quote_table())

        feature = catalog.get("trade_with_quote")
        result = mat.materialize(feature)
        assert result.success
        assert result.num_rows > 0
        assert result.output_path.exists()

    def test_missing_primary_stream(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        # No streams added
        feature = catalog.get("trade_with_quote")
        result = mat.materialize(feature)
        assert not result.success
        assert "No data" in result.error

    def test_missing_secondary_stream(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        # No quote stream
        feature = catalog.get("trade_with_quote")
        result = mat.materialize(feature)
        assert not result.success
        assert "No data" in result.error

    def test_elapsed_ms_populated(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        result = mat.materialize(catalog.get("trade_only"))
        assert result.elapsed_ms > 0


class TestMaterializeAll:
    def test_materialize_all_active(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        mat.add_stream("quote", _make_quote_table())

        stats = mat.materialize_all()
        assert stats.total_features == 2
        assert stats.successful == 2
        assert stats.failed == 0
        assert stats.total_rows > 0

    def test_materialize_specific_features(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())

        features = [catalog.get("trade_only")]
        stats = mat.materialize_all(features=features)
        assert stats.total_features == 1
        assert stats.successful == 1

    def test_partial_failure(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        # No quote → trade_with_quote will fail, trade_only succeeds

        stats = mat.materialize_all()
        assert stats.successful == 1
        assert stats.failed == 1
        assert len(stats.results) == 2


class TestReadback:
    def test_read_materialized(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        mat.materialize(catalog.get("trade_only"))

        table = mat.read_materialized("trade_only")
        assert table is not None
        assert table.num_rows == 100

    def test_read_not_materialized(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        table = mat.read_materialized("trade_only")
        assert table is None


class TestStreamManagement:
    def test_add_stream_concat(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table(50))
        mat.add_stream("trade", _make_trade_table(50))
        mat.materialize(catalog.get("trade_only"))

        table = mat.read_materialized("trade_only")
        assert table is not None
        assert table.num_rows == 100

    def test_clear_streams(self, catalog: FeatureCatalog, tmp_path: Path):
        mat = FeatureMaterializer(catalog=catalog, output_dir=tmp_path)
        mat.add_stream("trade", _make_trade_table())
        mat.clear_streams()

        result = mat.materialize(catalog.get("trade_only"))
        assert not result.success


class TestOutputDir:
    def test_creates_output_dir(self, tmp_path: Path):
        output_dir = tmp_path / "nested" / "features"
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(
            name="f1", primary_stream="s", columns=["x"],
        ))
        mat = FeatureMaterializer(catalog=catalog, output_dir=output_dir)
        mat.add_stream("s", pa.table({
            "timestamp": pa.array([1, 2, 3], type=pa.int64()),
            "x": [1.0, 2.0, 3.0],
        }))
        result = mat.materialize(catalog.get("f1"))
        assert result.success
        assert output_dir.exists()
