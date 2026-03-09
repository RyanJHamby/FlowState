"""Historical replay engine for time-ordered batch iteration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from flowstate.prism.gpu_direct import GPUDirectConfig, GPUDirectReader

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for the replay engine."""

    batch_size: int = 65536
    prefetch_count: int = 4
    sort_by: str = "timestamp"
    ascending: bool = True


@dataclass
class TimeRange:
    """A time range filter for replay."""

    start_ns: int | None = None
    end_ns: int | None = None


@dataclass
class ReplayFilter:
    """Predicate pushdown filters for replay."""

    symbols: list[str] | None = None
    data_types: list[str] | None = None
    time_range: TimeRange | None = None
    columns: list[str] | None = None


class ReplayEngine:
    """Historical data replay engine with predicate pushdown.

    Reads partitioned Parquet data in time order, supporting filtering
    by symbol, time range, and data type. Produces Arrow RecordBatches
    suitable for ML training loops.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: ReplayConfig | None = None,
        gpu_config: GPUDirectConfig | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._config = config or ReplayConfig()
        self._reader = GPUDirectReader(gpu_config or GPUDirectConfig(enable_gpu=False))

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def config(self) -> ReplayConfig:
        return self._config

    def discover_files(self, replay_filter: ReplayFilter | None = None) -> list[Path]:
        """Discover Parquet files matching the filter criteria.

        Uses Hive partition structure for predicate pushdown on
        data_type and date.
        """
        pattern = "**/*.parquet"
        files = sorted(self._data_dir.glob(pattern))

        if replay_filter is not None and replay_filter.data_types:
            filtered = []
            for f in files:
                for dt in replay_filter.data_types:
                    if f"type={dt}" in str(f):
                        filtered.append(f)
                        break
            files = filtered

        return files

    def replay(self, replay_filter: ReplayFilter | None = None) -> Iterator[pa.RecordBatch]:
        """Iterate over historical data in time order.

        Args:
            replay_filter: Optional filter for symbols, time range, columns.

        Yields:
            RecordBatches in time order.
        """
        files = self.discover_files(replay_filter)
        if not files:
            return

        for file_path in files:
            yield from self._replay_file(file_path, replay_filter)

    def _replay_file(
        self, path: Path, replay_filter: ReplayFilter | None
    ) -> Iterator[pa.RecordBatch]:
        """Read and filter a single Parquet file."""
        columns = replay_filter.columns if replay_filter else None
        table = self._reader.read_parquet(path, columns=columns)

        # Apply symbol filter
        if replay_filter and replay_filter.symbols and "symbol" in table.schema.names:
            mask = pc.is_in(table.column("symbol"), pa.array(replay_filter.symbols))
            table = table.filter(mask)

        # Apply time range filter
        if replay_filter and replay_filter.time_range and "timestamp" in table.schema.names:
            tr = replay_filter.time_range
            ts_col = table.column("timestamp").cast(pa.int64())
            if tr.start_ns is not None:
                mask = pc.greater_equal(ts_col, tr.start_ns)
                table = table.filter(mask)
            if tr.end_ns is not None:
                ts_col = table.column("timestamp").cast(pa.int64())
                mask = pc.less(ts_col, tr.end_ns)
                table = table.filter(mask)

        # Sort by timestamp
        if self._config.sort_by in table.schema.names:
            indices = pc.sort_indices(table, sort_keys=[(self._config.sort_by, "ascending" if self._config.ascending else "descending")])
            table = table.take(indices)

        # Yield in batches
        yield from table.to_batches(max_chunksize=self._config.batch_size)

    def count(self, replay_filter: ReplayFilter | None = None) -> int:
        """Count total rows matching the filter without loading all data."""
        files = self.discover_files(replay_filter)
        total = 0
        for f in files:
            metadata = pq.read_metadata(f)
            total += metadata.num_rows
        return total
