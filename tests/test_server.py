"""Tests for the feature serving layer."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from flowstate.store.catalog import (
    FeatureCatalog,
    FeatureDefinition,
    FeatureStatus,
)
from flowstate.store.materializer import FeatureMaterializer
from flowstate.store.server import FeatureServer


def _make_trade_table(n: int = 100) -> pa.Table:
    return pa.table({
        "timestamp": pa.array(range(0, n * 1_000_000, 1_000_000), type=pa.int64()),
        "price": [100.0 + i * 0.01 for i in range(n)],
        "volume": [10 * (i + 1) for i in range(n)],
        "symbol": ["AAPL"] * (n // 2) + ["MSFT"] * (n - n // 2),
    })


@pytest.fixture
def populated_store(tmp_path: Path):
    """Create a catalog with materialized features."""
    catalog = FeatureCatalog()
    catalog.register(FeatureDefinition(
        name="trade_prices",
        primary_stream="trade",
        columns=["price", "volume", "symbol"],
        tags=["equities"],
    ))
    catalog.register(FeatureDefinition(
        name="deprecated_feat",
        primary_stream="trade",
        columns=["price"],
        status=FeatureStatus.DEPRECATED,
    ))

    mat_dir = tmp_path / "materialized"
    mat = FeatureMaterializer(catalog=catalog, output_dir=mat_dir)
    mat.add_stream("trade", _make_trade_table())
    mat.materialize_all(features=catalog.list_features())  # All, not just active

    return catalog, mat_dir


class TestListFeatures:
    def test_list_all(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        features = server.list_features()
        assert len(features) == 2
        assert features[0].feature_name == "deprecated_feat"
        assert features[1].feature_name == "trade_prices"

    def test_list_by_status(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        active = server.list_features(status=FeatureStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].feature_name == "trade_prices"

    def test_descriptor_metadata(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        features = server.list_features()
        desc = next(f for f in features if f.feature_name == "trade_prices")
        assert desc.num_rows == 100
        assert desc.size_bytes > 0
        assert desc.version == 1


class TestGetFeature:
    def test_get_full_table(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        table = server.get_feature("trade_prices")
        assert table.num_rows == 100
        assert "price" in table.schema.names

    def test_column_projection(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        table = server.get_feature("trade_prices", columns=["price"])
        assert table.schema.names == ["price"]
        assert table.num_rows == 100

    def test_symbol_filter(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        table = server.get_feature("trade_prices", symbols=["AAPL"])
        assert table.num_rows == 50

    def test_symbol_filter_multiple(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        table = server.get_feature("trade_prices", symbols=["AAPL", "MSFT"])
        assert table.num_rows == 100

    def test_not_materialized_raises(self, tmp_path: Path):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="ghost", primary_stream="s", columns=["x"]))
        server = FeatureServer(catalog=catalog, data_dir=tmp_path)
        with pytest.raises(KeyError, match="not materialized"):
            server.get_feature("ghost")

    def test_not_in_catalog_raises(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        with pytest.raises(KeyError, match="not found"):
            server.get_feature("nonexistent")


class TestGetMetadata:
    def test_metadata_for_materialized(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        meta = server.get_metadata("trade_prices")
        assert meta["name"] == "trade_prices"
        assert meta["materialized"] is True
        assert meta["num_rows"] == 100
        assert meta["size_bytes"] > 0
        assert meta["status"] == "active"

    def test_metadata_for_unmaterialized(self, tmp_path: Path):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(
            name="pending",
            primary_stream="trade",
            description="Not yet materialized",
        ))
        server = FeatureServer(catalog=catalog, data_dir=tmp_path)
        meta = server.get_metadata("pending")
        assert meta["materialized"] is False
        assert meta["description"] == "Not yet materialized"


class TestFeatureExists:
    def test_exists(self, populated_store):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        assert server.feature_exists("trade_prices")
        assert not server.feature_exists("nonexistent")


class TestRefresh:
    def test_refresh_picks_up_new_files(self, populated_store, tmp_path: Path):
        catalog, data_dir = populated_store
        server = FeatureServer(catalog=catalog, data_dir=data_dir)
        assert len(server.list_features()) == 2

        # Add a new feature and materialize it
        catalog.register(FeatureDefinition(
            name="new_feat",
            primary_stream="trade",
            columns=["price"],
        ))
        mat = FeatureMaterializer(catalog=catalog, output_dir=data_dir)
        mat.add_stream("trade", _make_trade_table(50))
        mat.materialize(catalog.get("new_feat"))

        # Server doesn't know about it yet
        assert not server.feature_exists("new_feat")

        # Refresh picks it up
        server.refresh()
        assert server.feature_exists("new_feat")
        assert len(server.list_features()) == 3


class TestEmptyStore:
    def test_empty_data_dir(self, tmp_path: Path):
        catalog = FeatureCatalog()
        server = FeatureServer(catalog=catalog, data_dir=tmp_path / "nonexistent")
        assert server.list_features() == []

    def test_empty_catalog(self, tmp_path: Path):
        catalog = FeatureCatalog()
        server = FeatureServer(catalog=catalog, data_dir=tmp_path)
        assert server.list_features() == []
