"""Versioned schema registry with compatibility checks."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

import pyarrow as pa


class CompatibilityMode(str, Enum):
    """Schema evolution compatibility modes."""

    NONE = "none"
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"


@dataclass(frozen=True)
class SchemaEntry:
    """A versioned schema entry in the registry."""

    name: str
    version: int
    schema: pa.Schema
    metadata: dict[str, str] = field(default_factory=dict)


class CompatibilityError(Exception):
    """Raised when a schema change violates compatibility constraints."""


class SchemaRegistry:
    """Thread-safe, versioned schema registry with compatibility enforcement.

    Stores multiple versions of named schemas and enforces compatibility
    rules when new versions are registered.
    """

    def __init__(self, compatibility: CompatibilityMode = CompatibilityMode.BACKWARD) -> None:
        self._lock = threading.Lock()
        self._schemas: dict[str, list[SchemaEntry]] = {}
        self._compatibility = compatibility

    @property
    def compatibility(self) -> CompatibilityMode:
        return self._compatibility

    def register(
        self,
        name: str,
        schema: pa.Schema,
        metadata: dict[str, str] | None = None,
    ) -> SchemaEntry:
        """Register a new schema version.

        Args:
            name: Schema name (e.g. "trade", "quote").
            schema: The PyArrow schema.
            metadata: Optional metadata dict.

        Returns:
            The created SchemaEntry.

        Raises:
            CompatibilityError: If the new schema violates compatibility constraints.
        """
        with self._lock:
            versions = self._schemas.setdefault(name, [])
            next_version = len(versions) + 1

            if versions and self._compatibility != CompatibilityMode.NONE:
                latest = versions[-1]
                self._check_compatibility(latest.schema, schema)

            entry = SchemaEntry(
                name=name,
                version=next_version,
                schema=schema,
                metadata=metadata or {},
            )
            versions.append(entry)
            return entry

    def get(self, name: str, version: int | None = None) -> SchemaEntry:
        """Get a schema by name and optional version.

        Args:
            name: Schema name.
            version: Specific version number. If None, returns latest.

        Returns:
            The SchemaEntry.

        Raises:
            KeyError: If the schema name or version is not found.
        """
        with self._lock:
            if name not in self._schemas or not self._schemas[name]:
                raise KeyError(f"Schema not found: {name}")
            versions = self._schemas[name]
            if version is None:
                return versions[-1]
            if version < 1 or version > len(versions):
                raise KeyError(f"Version {version} not found for schema '{name}'")
            return versions[version - 1]

    def list_schemas(self) -> list[str]:
        """Return all registered schema names."""
        with self._lock:
            return list(self._schemas.keys())

    def list_versions(self, name: str) -> list[int]:
        """Return all version numbers for a schema name."""
        with self._lock:
            if name not in self._schemas:
                raise KeyError(f"Schema not found: {name}")
            return [e.version for e in self._schemas[name]]

    def _check_compatibility(self, old: pa.Schema, new: pa.Schema) -> None:
        """Verify schema evolution compatibility.

        Backward compatible: new schema can read old data (new may add nullable fields,
        must not remove or change existing fields).
        Forward compatible: old schema can read new data (old fields still present).
        Full: both backward and forward compatible.
        """
        if self._compatibility in (CompatibilityMode.BACKWARD, CompatibilityMode.FULL):
            self._check_backward(old, new)
        if self._compatibility in (CompatibilityMode.FORWARD, CompatibilityMode.FULL):
            self._check_forward(old, new)

    @staticmethod
    def _check_backward(old: pa.Schema, new: pa.Schema) -> None:
        """New schema must be able to read old data."""
        for i in range(len(old)):
            old_field = old.field(i)
            try:
                new_field = new.field(old_field.name)
            except KeyError:
                raise CompatibilityError(
                    f"Backward incompatible: field '{old_field.name}' removed"
                )
            if new_field.type != old_field.type:
                raise CompatibilityError(
                    f"Backward incompatible: field '{old_field.name}' type changed "
                    f"from {old_field.type} to {new_field.type}"
                )

    @staticmethod
    def _check_forward(old: pa.Schema, new: pa.Schema) -> None:
        """Old schema must be able to read new data — new fields must be nullable."""
        old_names = set(old.names)
        for i in range(len(new)):
            new_field = new.field(i)
            if new_field.name not in old_names and not new_field.nullable:
                raise CompatibilityError(
                    f"Forward incompatible: new non-nullable field '{new_field.name}'"
                )
