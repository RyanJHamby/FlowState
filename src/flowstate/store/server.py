"""Arrow Flight serving layer for materialized features.

Provides a gRPC-based Arrow Flight server that serves materialized
features from the temporal feature store. Clients can:

- List available features via ``list_flights()``
- Retrieve feature data via ``do_get()`` with feature name as ticket
- Query feature metadata via ``get_flight_info()``

This is a lightweight serving layer on top of the materializer. In
production, this would be deployed behind a load balancer with
authentication and rate limiting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from flowstate.store.catalog import FeatureCatalog, FeatureDefinition, FeatureStatus

logger = logging.getLogger(__name__)


@dataclass
class FlightDescriptor:
    """Describes a servable feature."""

    feature_name: str
    version: int
    schema: pa.Schema
    num_rows: int
    size_bytes: int


class FeatureServer:
    """Serves materialized features via Arrow IPC reads.

    This is the serving component of the feature store. It reads
    materialized Arrow IPC files and provides a typed API for
    feature retrieval.

    In a production deployment, this would be wrapped in a
    ``pyarrow.flight.FlightServerBase`` for gRPC transport. The
    core logic (discovery, reading, metadata) is kept separate
    from the transport layer for testability.

    Example::

        server = FeatureServer(
            catalog=catalog,
            data_dir="/data/features/materialized",
        )

        # List available features
        features = server.list_features()

        # Get feature data
        table = server.get_feature("trade_with_quote")

        # Get feature data for specific symbols
        table = server.get_feature("trade_with_quote", symbols=["AAPL", "MSFT"])
    """

    def __init__(
        self,
        catalog: FeatureCatalog,
        data_dir: str | Path,
    ) -> None:
        self._catalog = catalog
        self._data_dir = Path(data_dir)
        self._descriptors: dict[str, FlightDescriptor] = {}
        self._refresh_descriptors()

    @property
    def catalog(self) -> FeatureCatalog:
        return self._catalog

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def refresh(self) -> None:
        """Refresh the list of available features from disk."""
        self._refresh_descriptors()

    def list_features(
        self,
        status: FeatureStatus | None = None,
    ) -> list[FlightDescriptor]:
        """List available materialized features."""
        descriptors = list(self._descriptors.values())
        if status is not None:
            active_names = {
                f.name for f in self._catalog.list_features(status=status)
            }
            descriptors = [d for d in descriptors if d.feature_name in active_names]
        return sorted(descriptors, key=lambda d: d.feature_name)

    def get_feature(
        self,
        name: str,
        columns: list[str] | None = None,
        symbols: list[str] | None = None,
        symbol_col: str = "symbol",
    ) -> pa.Table:
        """Retrieve a materialized feature as an Arrow Table.

        Args:
            name: Feature name from the catalog.
            columns: Optional column projection.
            symbols: Optional symbol filter.
            symbol_col: Column name for symbol filtering.

        Returns:
            Arrow Table with the feature data.

        Raises:
            KeyError: If the feature is not found or not materialized.
        """
        feature = self._catalog.get(name)
        path = self._feature_path(feature)
        if not path.exists():
            raise KeyError(f"Feature '{name}' is not materialized at {path}")

        table = self._read_ipc(path)

        if symbols and symbol_col in table.schema.names:
            import pyarrow.compute as pc
            mask = pc.is_in(table.column(symbol_col), pa.array(symbols))
            table = table.filter(mask)

        if columns:
            available = [c for c in columns if c in table.schema.names]
            if available:
                table = table.select(available)

        return table

    def get_metadata(self, name: str) -> dict:
        """Get metadata for a materialized feature."""
        feature = self._catalog.get(name)
        descriptor = self._descriptors.get(name)

        meta: dict = {
            "name": feature.name,
            "version": feature.version,
            "description": feature.description,
            "primary_stream": feature.primary_stream,
            "secondary_stream": feature.secondary_stream,
            "direction": str(feature.direction),
            "tolerance_ns": feature.tolerance_ns,
            "status": str(feature.status),
            "tags": feature.tags,
        }

        if descriptor:
            meta["num_rows"] = descriptor.num_rows
            meta["size_bytes"] = descriptor.size_bytes
            meta["schema"] = str(descriptor.schema)
            meta["materialized"] = True
        else:
            meta["materialized"] = False

        return meta

    def feature_exists(self, name: str) -> bool:
        """Check if a feature is materialized and available."""
        return name in self._descriptors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_descriptors(self) -> None:
        """Scan data_dir for materialized feature files."""
        self._descriptors.clear()
        if not self._data_dir.exists():
            return

        for feature in self._catalog.list_features():
            path = self._feature_path(feature)
            if path.exists():
                try:
                    schema, num_rows = self._read_ipc_metadata(path)
                    self._descriptors[feature.name] = FlightDescriptor(
                        feature_name=feature.name,
                        version=feature.version,
                        schema=schema,
                        num_rows=num_rows,
                        size_bytes=path.stat().st_size,
                    )
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {feature.name}: {e}")

    def _feature_path(self, feature: FeatureDefinition) -> Path:
        return self._data_dir / f"{feature.name}_v{feature.version}.arrow"

    def _read_ipc(self, path: Path) -> pa.Table:
        with ipc.open_file(str(path)) as reader:
            return reader.read_all()

    def _read_ipc_metadata(self, path: Path) -> tuple[pa.Schema, int]:
        with ipc.open_file(str(path)) as reader:
            return reader.schema, sum(
                reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
            )
