"""Tests for versioned schema registry."""

from __future__ import annotations

import threading

import pyarrow as pa
import pytest

from flowstate.schema.registry import (
    CompatibilityError,
    CompatibilityMode,
    SchemaRegistry,
)


@pytest.fixture
def registry() -> SchemaRegistry:
    return SchemaRegistry()


@pytest.fixture
def simple_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ])


class TestSchemaRegistry:
    def test_register_and_get(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        entry = registry.register("test", simple_schema)
        assert entry.name == "test"
        assert entry.version == 1
        assert entry.schema == simple_schema

    def test_get_latest(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        extended = simple_schema.append(pa.field("extra", pa.utf8(), nullable=True))
        registry.register("test", extended)

        latest = registry.get("test")
        assert latest.version == 2
        assert latest.schema == extended

    def test_get_specific_version(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        extended = simple_schema.append(pa.field("extra", pa.utf8(), nullable=True))
        registry.register("test", extended)

        v1 = registry.get("test", version=1)
        assert v1.schema == simple_schema

    def test_get_missing_raises(self, registry: SchemaRegistry):
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_get_bad_version_raises(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        with pytest.raises(KeyError, match="Version"):
            registry.get("test", version=99)

    def test_list_schemas(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("a", simple_schema)
        registry.register("b", simple_schema)
        assert set(registry.list_schemas()) == {"a", "b"}

    def test_list_versions(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        extended = simple_schema.append(pa.field("extra", pa.utf8(), nullable=True))
        registry.register("test", extended)
        assert registry.list_versions("test") == [1, 2]

    def test_metadata(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        entry = registry.register("test", simple_schema, metadata={"owner": "team-a"})
        assert entry.metadata["owner"] == "team-a"


class TestBackwardCompatibility:
    def test_add_nullable_field_ok(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        extended = simple_schema.append(pa.field("extra", pa.utf8(), nullable=True))
        entry = registry.register("test", extended)
        assert entry.version == 2

    def test_remove_field_fails(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        reduced = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        with pytest.raises(CompatibilityError, match="removed"):
            registry.register("test", reduced)

    def test_change_type_fails(self, registry: SchemaRegistry, simple_schema: pa.Schema):
        registry.register("test", simple_schema)
        changed = pa.schema([
            pa.field("id", pa.utf8(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ])
        with pytest.raises(CompatibilityError, match="type changed"):
            registry.register("test", changed)


class TestForwardCompatibility:
    def test_new_non_nullable_field_fails(self, simple_schema: pa.Schema):
        registry = SchemaRegistry(compatibility=CompatibilityMode.FORWARD)
        registry.register("test", simple_schema)
        extended = simple_schema.append(pa.field("required", pa.utf8(), nullable=False))
        with pytest.raises(CompatibilityError, match="Forward incompatible"):
            registry.register("test", extended)

    def test_new_nullable_field_ok(self, simple_schema: pa.Schema):
        registry = SchemaRegistry(compatibility=CompatibilityMode.FORWARD)
        registry.register("test", simple_schema)
        extended = simple_schema.append(pa.field("optional", pa.utf8(), nullable=True))
        entry = registry.register("test", extended)
        assert entry.version == 2


class TestNoCompatibility:
    def test_any_change_allowed(self, simple_schema: pa.Schema):
        registry = SchemaRegistry(compatibility=CompatibilityMode.NONE)
        registry.register("test", simple_schema)
        totally_different = pa.schema([pa.field("x", pa.utf8())])
        entry = registry.register("test", totally_different)
        assert entry.version == 2


class TestThreadSafety:
    def test_concurrent_registration(self, simple_schema: pa.Schema):
        registry = SchemaRegistry(compatibility=CompatibilityMode.NONE)
        errors: list[Exception] = []

        def register_many(prefix: str) -> None:
            try:
                for i in range(50):
                    schema = simple_schema.append(
                        pa.field(f"{prefix}_{i}", pa.utf8(), nullable=True)
                    )
                    registry.register(f"{prefix}", schema)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(4):
            assert len(registry.list_versions(f"t{i}")) == 50
