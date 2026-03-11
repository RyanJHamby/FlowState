"""Tests for the temporal feature catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowstate.store.catalog import (
    CatalogStats,
    FeatureCatalog,
    FeatureDefinition,
    FeatureStatus,
    JoinDirection,
)


class TestFeatureDefinition:
    def test_defaults(self):
        fd = FeatureDefinition(name="test_feature")
        assert fd.version == 1
        assert fd.direction == JoinDirection.BACKWARD
        assert fd.status == FeatureStatus.ACTIVE
        assert fd.qualified_name == "test_feature:v1"

    def test_qualified_name(self):
        fd = FeatureDefinition(name="spread", version=3)
        assert fd.qualified_name == "spread:v3"

    def test_all_fields(self):
        fd = FeatureDefinition(
            name="trade_with_quote",
            version=2,
            description="Trade enriched with nearest quote",
            primary_stream="trade",
            secondary_stream="quote",
            columns=["bid_price", "ask_price"],
            timestamp_column="ts",
            group_column="symbol",
            direction=JoinDirection.NEAREST,
            tolerance_ns=5_000_000_000,
            status=FeatureStatus.DRAFT,
            owner="quant-team",
            tags=["production", "equities"],
            depends_on=["raw_trade"],
        )
        assert fd.columns == ["bid_price", "ask_price"]
        assert fd.tolerance_ns == 5_000_000_000
        assert fd.direction == JoinDirection.NEAREST


class TestFeatureCatalog:
    def test_register_and_get(self):
        catalog = FeatureCatalog()
        fd = FeatureDefinition(name="feat_a", primary_stream="trade", columns=["price"])
        catalog.register(fd)
        assert catalog.get("feat_a") is fd
        assert len(catalog) == 1
        assert "feat_a" in catalog

    def test_register_duplicate_version_raises(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="feat", version=1))
        with pytest.raises(ValueError, match="already registered"):
            catalog.register(FeatureDefinition(name="feat", version=1))

    def test_register_older_version_raises(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="feat", version=3))
        with pytest.raises(ValueError, match="older than"):
            catalog.register(FeatureDefinition(name="feat", version=2))

    def test_version_upgrade(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="feat", version=1))
        catalog.register(FeatureDefinition(name="feat", version=2))
        assert catalog.get("feat").version == 2
        assert len(catalog) == 1

    def test_get_missing_raises(self):
        catalog = FeatureCatalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.get("nonexistent")

    def test_list_features(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="b_feat", status=FeatureStatus.ACTIVE))
        catalog.register(FeatureDefinition(name="a_feat", status=FeatureStatus.DRAFT))
        catalog.register(FeatureDefinition(name="c_feat", status=FeatureStatus.DEPRECATED))

        all_features = catalog.list_features()
        assert [f.name for f in all_features] == ["a_feat", "b_feat", "c_feat"]

        active = catalog.list_features(status=FeatureStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "b_feat"

    def test_list_features_by_tag(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="f1", tags=["equities", "prod"]))
        catalog.register(FeatureDefinition(name="f2", tags=["crypto"]))
        catalog.register(FeatureDefinition(name="f3", tags=["equities"]))

        equities = catalog.list_features(tag="equities")
        assert [f.name for f in equities] == ["f1", "f3"]

    def test_deprecate(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="old_feat"))
        catalog.deprecate("old_feat")
        assert catalog.get("old_feat").status == FeatureStatus.DEPRECATED

    def test_remove(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="temp"))
        catalog.remove("temp")
        assert len(catalog) == 0
        assert "temp" not in catalog

    def test_remove_missing_raises(self):
        catalog = FeatureCatalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.remove("ghost")

    def test_contains(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="x"))
        assert "x" in catalog
        assert "y" not in catalog


class TestDependencies:
    def test_single_dependency(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="base", columns=["a"], primary_stream="s"))
        catalog.register(FeatureDefinition(
            name="derived", depends_on=["base"], columns=["b"], primary_stream="s",
        ))

        deps = catalog.dependencies("derived")
        names = [d.name for d in deps]
        assert names == ["base", "derived"]

    def test_transitive_dependencies(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="raw"))
        catalog.register(FeatureDefinition(name="clean", depends_on=["raw"]))
        catalog.register(FeatureDefinition(name="feature", depends_on=["clean"]))

        deps = catalog.dependencies("feature")
        names = [d.name for d in deps]
        assert names == ["raw", "clean", "feature"]

    def test_diamond_dependency(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="a"))
        catalog.register(FeatureDefinition(name="b", depends_on=["a"]))
        catalog.register(FeatureDefinition(name="c", depends_on=["a"]))
        catalog.register(FeatureDefinition(name="d", depends_on=["b", "c"]))

        deps = catalog.dependencies("d")
        names = [d.name for d in deps]
        # a appears once, before both b and c
        assert "a" in names
        assert names.count("a") == 1
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")

    def test_missing_dependency_raises(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="broken", depends_on=["missing"]))
        with pytest.raises(KeyError, match="not found"):
            catalog.dependencies("broken")


class TestStats:
    def test_stats(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="a", status=FeatureStatus.ACTIVE))
        catalog.register(FeatureDefinition(name="b", status=FeatureStatus.ACTIVE))
        catalog.register(FeatureDefinition(name="c", status=FeatureStatus.DEPRECATED))
        catalog.register(FeatureDefinition(name="d", status=FeatureStatus.DRAFT))

        s = catalog.stats()
        assert s.total_features == 4
        assert s.active_features == 2
        assert s.deprecated_features == 1
        assert s.draft_features == 1

    def test_empty_stats(self):
        s = FeatureCatalog().stats()
        assert s == CatalogStats()


class TestValidation:
    def test_valid_catalog(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="a", primary_stream="s", columns=["x"]))
        assert catalog.validate() == []

    def test_missing_dependency(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(
            name="feat", depends_on=["ghost"], primary_stream="s", columns=["x"],
        ))
        issues = catalog.validate()
        assert any("ghost" in i for i in issues)

    def test_missing_primary_stream(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="feat", columns=["x"]))
        issues = catalog.validate()
        assert any("primary_stream" in i for i in issues)

    def test_missing_columns(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="feat", primary_stream="s"))
        issues = catalog.validate()
        assert any("columns" in i for i in issues)

    def test_multiple_issues(self):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="bad", depends_on=["missing"]))
        issues = catalog.validate()
        # Missing dep + missing stream + missing columns
        assert len(issues) >= 3


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "catalog.json"
        catalog = FeatureCatalog(path)
        catalog.register(FeatureDefinition(
            name="trade_quote",
            version=2,
            primary_stream="trade",
            secondary_stream="quote",
            columns=["bid", "ask"],
            direction=JoinDirection.NEAREST,
            tolerance_ns=1_000_000,
            tags=["prod"],
        ))
        catalog.save()

        # Load into new catalog
        loaded = FeatureCatalog(path)
        assert len(loaded) == 1
        feat = loaded.get("trade_quote")
        assert feat.version == 2
        assert feat.direction == JoinDirection.NEAREST
        assert feat.tolerance_ns == 1_000_000
        assert feat.columns == ["bid", "ask"]
        assert feat.tags == ["prod"]

    def test_save_to_explicit_path(self, tmp_path: Path):
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(name="f1", primary_stream="s", columns=["a"]))
        path = tmp_path / "sub" / "catalog.json"
        catalog.save(path)
        assert path.exists()

        raw = json.loads(path.read_text())
        assert "f1" in raw

    def test_save_no_path_raises(self):
        catalog = FeatureCatalog()
        with pytest.raises(ValueError, match="No path"):
            catalog.save()

    def test_roundtrip_all_statuses(self, tmp_path: Path):
        path = tmp_path / "catalog.json"
        catalog = FeatureCatalog(path)
        catalog.register(FeatureDefinition(name="a", status=FeatureStatus.ACTIVE))
        catalog.register(FeatureDefinition(name="d", status=FeatureStatus.DRAFT))
        catalog.register(FeatureDefinition(name="x", status=FeatureStatus.DEPRECATED))
        catalog.save()

        loaded = FeatureCatalog(path)
        assert loaded.get("a").status == FeatureStatus.ACTIVE
        assert loaded.get("d").status == FeatureStatus.DRAFT
        assert loaded.get("x").status == FeatureStatus.DEPRECATED

    def test_load_nonexistent_path(self, tmp_path: Path):
        catalog = FeatureCatalog(tmp_path / "nope.json")
        assert len(catalog) == 0
