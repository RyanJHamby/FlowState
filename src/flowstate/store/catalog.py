"""Feature catalog: definitions, versioning, and dependency tracking.

The catalog is the source of truth for what features exist, how they are
computed, and which upstream data streams they depend on. It enforces:

- **Unique feature names** within a namespace.
- **Semantic versioning** for schema evolution.
- **Dependency DAG** tracking (feature B depends on feature A).
- **Point-in-time metadata**: each feature records its temporal semantics
  (join direction, tolerance, timestamp column).

The catalog is persisted as a JSON file alongside materialized data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class JoinDirection(StrEnum):
    """Temporal join direction for feature computation."""

    BACKWARD = "backward"
    FORWARD = "forward"
    NEAREST = "nearest"


class FeatureStatus(StrEnum):
    """Lifecycle status of a feature definition."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class FeatureDefinition:
    """A single feature in the catalog."""

    name: str
    version: int = 1
    description: str = ""
    # Temporal semantics
    primary_stream: str = ""
    secondary_stream: str = ""
    columns: list[str] = field(default_factory=list)
    timestamp_column: str = "timestamp"
    group_column: str | None = None
    direction: JoinDirection = JoinDirection.BACKWARD
    tolerance_ns: int | None = None
    # Metadata
    status: FeatureStatus = FeatureStatus.ACTIVE
    owner: str = ""
    tags: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.name}:v{self.version}"


@dataclass
class CatalogStats:
    """Summary statistics for the catalog."""

    total_features: int = 0
    active_features: int = 0
    deprecated_features: int = 0
    draft_features: int = 0


class FeatureCatalog:
    """In-memory feature catalog with JSON persistence.

    Example::

        catalog = FeatureCatalog("/data/features/catalog.json")

        catalog.register(FeatureDefinition(
            name="trade_with_quote",
            primary_stream="trade",
            secondary_stream="quote",
            columns=["bid_price", "ask_price"],
            tolerance_ns=5_000_000_000,
        ))

        defn = catalog.get("trade_with_quote")
        catalog.save()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._features: dict[str, FeatureDefinition] = {}

        if self._path and self._path.exists():
            self._load()

    @property
    def path(self) -> Path | None:
        return self._path

    def register(self, feature: FeatureDefinition) -> None:
        """Register a new feature or update an existing one.

        If a feature with the same name exists at a lower version, the
        new version supersedes it. Same version raises ValueError.
        """
        existing = self._features.get(feature.name)
        if existing and existing.version == feature.version:
            raise ValueError(
                f"Feature '{feature.name}' v{feature.version} already registered. "
                "Bump the version to update."
            )
        if existing and feature.version <= existing.version:
            raise ValueError(
                f"Feature '{feature.name}' v{feature.version} is older than "
                f"existing v{existing.version}."
            )

        self._features[feature.name] = feature
        logger.info(f"Registered feature: {feature.qualified_name}")

    def get(self, name: str) -> FeatureDefinition:
        """Get a feature definition by name. Raises KeyError if not found."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in catalog")
        return self._features[name]

    def list_features(
        self, status: FeatureStatus | None = None, tag: str | None = None,
    ) -> list[FeatureDefinition]:
        """List features, optionally filtered by status or tag."""
        features = list(self._features.values())
        if status is not None:
            features = [f for f in features if f.status == status]
        if tag is not None:
            features = [f for f in features if tag in f.tags]
        return sorted(features, key=lambda f: f.name)

    def deprecate(self, name: str) -> None:
        """Mark a feature as deprecated."""
        feature = self.get(name)
        feature.status = FeatureStatus.DEPRECATED
        logger.info(f"Deprecated feature: {feature.qualified_name}")

    def remove(self, name: str) -> None:
        """Remove a feature from the catalog."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found")
        del self._features[name]

    def dependencies(self, name: str) -> list[FeatureDefinition]:
        """Get the transitive dependency chain for a feature."""
        visited: set[str] = set()
        result: list[FeatureDefinition] = []
        self._resolve_deps(name, visited, result)
        return result

    def _resolve_deps(
        self, name: str, visited: set[str], result: list[FeatureDefinition],
    ) -> None:
        if name in visited:
            return
        visited.add(name)
        feature = self.get(name)
        for dep_name in feature.depends_on:
            self._resolve_deps(dep_name, visited, result)
        result.append(feature)

    def stats(self) -> CatalogStats:
        """Summary statistics."""
        features = list(self._features.values())
        return CatalogStats(
            total_features=len(features),
            active_features=sum(1 for f in features if f.status == FeatureStatus.ACTIVE),
            deprecated_features=sum(1 for f in features if f.status == FeatureStatus.DEPRECATED),
            draft_features=sum(1 for f in features if f.status == FeatureStatus.DRAFT),
        )

    def validate(self) -> list[str]:
        """Validate catalog consistency. Returns list of issues."""
        issues: list[str] = []
        for name, feature in self._features.items():
            for dep in feature.depends_on:
                if dep not in self._features:
                    issues.append(f"{name}: depends on unknown feature '{dep}'")
            if not feature.primary_stream:
                issues.append(f"{name}: missing primary_stream")
            if not feature.columns:
                issues.append(f"{name}: no columns defined")
        return issues

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Save the catalog to JSON."""
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No path specified for save")
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {name: asdict(f) for name, f in self._features.items()}
        target.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Saved catalog ({len(self._features)} features) to {target}")

    def _load(self) -> None:
        """Load catalog from JSON."""
        assert self._path is not None
        data = json.loads(self._path.read_text())
        for name, raw in data.items():
            raw["direction"] = JoinDirection(raw.get("direction", "backward"))
            raw["status"] = FeatureStatus(raw.get("status", "active"))
            self._features[name] = FeatureDefinition(**raw)
        logger.info(f"Loaded catalog ({len(self._features)} features) from {self._path}")

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features
