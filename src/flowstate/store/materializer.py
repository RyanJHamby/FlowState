"""Feature materializer: background alignment and caching to Arrow IPC.

The materializer reads feature definitions from the catalog, runs the
temporal alignment engine, and writes results as Arrow IPC files. This
enables fast point-in-time feature retrieval without re-computing joins.

Features are materialized per-symbol to preserve temporal ordering and
enable parallel materialization across symbols.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from flowstate.prism.alignment import (
    AlignmentSpec,
    AsOfConfig,
    align_streams,
)
from flowstate.store.catalog import FeatureCatalog, FeatureDefinition, JoinDirection

logger = logging.getLogger(__name__)


# Map catalog direction to alignment config direction string
_DIRECTION_MAP: dict[JoinDirection, str] = {
    JoinDirection.BACKWARD: "backward",
    JoinDirection.FORWARD: "forward",
    JoinDirection.NEAREST: "nearest",
}


@dataclass
class MaterializeResult:
    """Result of materializing a single feature."""

    feature_name: str
    output_path: Path
    num_rows: int
    elapsed_ms: float
    success: bool
    error: str = ""


@dataclass
class MaterializeStats:
    """Aggregate stats for a materialization run."""

    total_features: int = 0
    successful: int = 0
    failed: int = 0
    total_rows: int = 0
    total_elapsed_ms: float = 0.0
    results: list[MaterializeResult] = field(default_factory=list)


class FeatureMaterializer:
    """Materializes catalog features to Arrow IPC files.

    Given a catalog and raw stream data, the materializer:
    1. Reads feature definitions (primary/secondary streams, columns, tolerance)
    2. Runs temporal alignment via the alignment engine
    3. Writes aligned results as Arrow IPC files for fast serving

    Example::

        materializer = FeatureMaterializer(
            catalog=catalog,
            output_dir="/data/features/materialized",
        )

        # Provide raw stream data
        materializer.add_stream("trade", trade_table)
        materializer.add_stream("quote", quote_table)

        # Materialize all active features
        stats = materializer.materialize_all()
    """

    def __init__(
        self,
        catalog: FeatureCatalog,
        output_dir: str | Path,
        timestamp_col: str = "timestamp",
    ) -> None:
        self._catalog = catalog
        self._output_dir = Path(output_dir)
        self._timestamp_col = timestamp_col
        self._streams: dict[str, pa.Table] = {}

    @property
    def catalog(self) -> FeatureCatalog:
        return self._catalog

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def add_stream(self, name: str, table: pa.Table) -> None:
        """Register raw stream data for materialization."""
        if name in self._streams:
            self._streams[name] = pa.concat_tables([self._streams[name], table])
        else:
            self._streams[name] = table

    def clear_streams(self) -> None:
        """Clear all registered stream data."""
        self._streams.clear()

    def materialize(self, feature: FeatureDefinition) -> MaterializeResult:
        """Materialize a single feature definition.

        Runs temporal alignment using the feature's configured streams,
        columns, direction, and tolerance, then writes the result to IPC.
        """
        t0 = time.monotonic()
        try:
            aligned = self._align_feature(feature)
            if aligned is None or aligned.num_rows == 0:
                return MaterializeResult(
                    feature_name=feature.name,
                    output_path=Path(""),
                    num_rows=0,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    success=True,
                )

            output_path = self._write_ipc(feature, aligned)
            elapsed_ms = (time.monotonic() - t0) * 1000

            logger.info(
                f"Materialized {feature.qualified_name}: "
                f"{aligned.num_rows} rows in {elapsed_ms:.1f}ms"
            )
            return MaterializeResult(
                feature_name=feature.name,
                output_path=output_path,
                num_rows=aligned.num_rows,
                elapsed_ms=elapsed_ms,
                success=True,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(f"Failed to materialize {feature.name}: {e}")
            return MaterializeResult(
                feature_name=feature.name,
                output_path=Path(""),
                num_rows=0,
                elapsed_ms=elapsed_ms,
                success=False,
                error=str(e),
            )

    def materialize_all(
        self,
        features: list[FeatureDefinition] | None = None,
    ) -> MaterializeStats:
        """Materialize multiple features.

        If no feature list is provided, materializes all active features
        from the catalog.
        """
        if features is None:
            from flowstate.store.catalog import FeatureStatus
            features = self._catalog.list_features(status=FeatureStatus.ACTIVE)

        stats = MaterializeStats(total_features=len(features))
        for feature in features:
            result = self.materialize(feature)
            stats.results.append(result)
            if result.success:
                stats.successful += 1
                stats.total_rows += result.num_rows
            else:
                stats.failed += 1
            stats.total_elapsed_ms += result.elapsed_ms

        logger.info(
            f"Materialization complete: {stats.successful}/{stats.total_features} "
            f"features, {stats.total_rows} total rows"
        )
        return stats

    def _align_feature(self, feature: FeatureDefinition) -> pa.Table | None:
        """Run temporal alignment for a feature definition."""
        primary = self._streams.get(feature.primary_stream)
        if primary is None:
            raise ValueError(
                f"No data for primary stream '{feature.primary_stream}'"
            )

        if not feature.secondary_stream:
            # No secondary — return primary filtered to requested columns
            cols = [c for c in feature.columns if c in primary.schema.names]
            if self._timestamp_col in primary.schema.names:
                cols = [self._timestamp_col] + [c for c in cols if c != self._timestamp_col]
            return primary.select(cols) if cols else primary

        secondary = self._streams.get(feature.secondary_stream)
        if secondary is None:
            raise ValueError(
                f"No data for secondary stream '{feature.secondary_stream}'"
            )

        direction = _DIRECTION_MAP.get(feature.direction, "backward")

        config = AsOfConfig(
            direction=direction,
            tolerance_ns=feature.tolerance_ns,
        )

        spec = AlignmentSpec(
            name=feature.secondary_stream,
            table=secondary,
            value_columns=feature.columns if feature.columns else None,
            config=config,
        )

        result, _stats = align_streams(
            primary=primary,
            secondaries=[spec],
            primary_timestamp_col=self._timestamp_col,
            primary_symbol_col=feature.group_column or "symbol",
        )
        return result

    def _write_ipc(self, feature: FeatureDefinition, table: pa.Table) -> Path:
        """Write aligned result to Arrow IPC file."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / f"{feature.name}_v{feature.version}.arrow"

        with ipc.new_file(str(output_path), table.schema) as writer:
            writer.write_table(table)

        return output_path

    def read_materialized(self, feature_name: str) -> pa.Table | None:
        """Read a previously materialized feature from IPC."""
        feature = self._catalog.get(feature_name)
        path = self._output_dir / f"{feature.name}_v{feature.version}.arrow"
        if not path.exists():
            return None

        with ipc.open_file(str(path)) as reader:
            return reader.read_all()
