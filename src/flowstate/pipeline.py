"""Top-level Pipeline orchestrator and ReplaySession fluent API."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from flowstate.ops.health import HealthChecker
from flowstate.ops.metrics import MetricsRegistry
from flowstate.prism.dataloader import FlowStateIterableDataset
from flowstate.prism.replay import ReplayConfig, ReplayEngine, ReplayFilter, TimeRange
from flowstate.schema.types import MarketDataType, get_schema
from flowstate.storage.writer import PartitionedParquetWriter, WriterConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full ingestion pipeline."""

    data_dir: str | Path
    num_buckets: int = 256
    ring_buffer_capacity: int = 65536
    ring_buffer_slot_size: int = 4096
    compression: str = "zstd"
    max_rows_per_file: int = 1_000_000


class Pipeline:
    """Top-level pipeline builder for market data ingestion.

    Provides a fluent API for configuring and running the complete
    ingestion pipeline: connect -> normalize -> validate -> write.

    Example::

        pipeline = (
            Pipeline(data_dir="/data/market")
            .add_source("polygon", api_key="...")
            .subscribe(["AAPL", "MSFT", "GOOG"])
            .build()
        )
        await pipeline.start()
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: PipelineConfig | None = None,
    ) -> None:
        self._config = config or PipelineConfig(data_dir=data_dir)
        self._data_dir = Path(data_dir)
        self._sources: list[dict[str, Any]] = []
        self._symbols: list[str] = []
        self._metrics = MetricsRegistry()
        self._health = HealthChecker()
        self._writer: PartitionedParquetWriter | None = None

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def health(self) -> HealthChecker:
        return self._health

    def add_source(self, provider: str, **kwargs: Any) -> Pipeline:
        """Add a data source to the pipeline.

        Args:
            provider: Source provider name ("polygon", "alpaca").
            **kwargs: Provider-specific configuration.

        Returns:
            Self for chaining.
        """
        self._sources.append({"provider": provider, **kwargs})
        return self

    def subscribe(self, symbols: list[str]) -> Pipeline:
        """Subscribe to market data for the given symbols.

        Args:
            symbols: List of symbol identifiers.

        Returns:
            Self for chaining.
        """
        self._symbols.extend(symbols)
        return self

    def build(self) -> Pipeline:
        """Build the pipeline components.

        Returns:
            Self for chaining.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._writer = PartitionedParquetWriter(
            WriterConfig(
                base_path=self._data_dir,
                compression=self._config.compression,
                num_buckets=self._config.num_buckets,
                max_rows_per_file=self._config.max_rows_per_file,
            )
        )
        self._health.register_metrics(self._metrics)
        return self

    def get_writer(self) -> PartitionedParquetWriter | None:
        return self._writer

    def write(self, batch: pa.RecordBatch, data_type: str) -> list[Path]:
        """Write a batch of data through the pipeline.

        Args:
            batch: Arrow RecordBatch to write.
            data_type: Market data type.

        Returns:
            List of written file paths.
        """
        if self._writer is None:
            raise RuntimeError("Pipeline not built. Call .build() first.")
        throughput = self._metrics.throughput("pipeline_write")
        throughput.increment(batch.num_rows)
        return self._writer.write(batch, data_type)

    def flush(self) -> list[Path]:
        """Flush all buffered data to disk."""
        if self._writer is None:
            return []
        return self._writer.flush_all()

    def close(self) -> list[Path]:
        """Close the pipeline and flush remaining data."""
        if self._writer is not None:
            return self._writer.close()
        return []


class ReplaySession:
    """Fluent API for historical data replay.

    Example::

        session = (
            ReplaySession("/data/market")
            .symbols(["AAPL", "MSFT"])
            .time_range(start_ns=..., end_ns=...)
            .data_types(["trade"])
            .batch_size(65536)
        )
        for batch in session:
            process(batch)
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._filter = ReplayFilter()
        self._config = ReplayConfig()

    def symbols(self, syms: list[str]) -> ReplaySession:
        """Filter by symbols."""
        self._filter.symbols = syms
        return self

    def data_types(self, types: list[str]) -> ReplaySession:
        """Filter by data types."""
        self._filter.data_types = types
        return self

    def time_range(
        self, start_ns: int | None = None, end_ns: int | None = None
    ) -> ReplaySession:
        """Filter by time range."""
        self._filter.time_range = TimeRange(start_ns=start_ns, end_ns=end_ns)
        return self

    def columns(self, cols: list[str]) -> ReplaySession:
        """Select specific columns."""
        self._filter.columns = cols
        return self

    def batch_size(self, size: int) -> ReplaySession:
        """Set the batch size."""
        self._config.batch_size = size
        return self

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        engine = ReplayEngine(self._data_dir, config=self._config)
        yield from engine.replay(self._filter)

    def count(self) -> int:
        """Count total matching rows."""
        engine = ReplayEngine(self._data_dir, config=self._config)
        return engine.count(self._filter)

    def to_dataset(
        self, numeric_columns: list[str] | None = None
    ) -> FlowStateIterableDataset:
        """Convert to an iterable dataset for ML frameworks."""
        return FlowStateIterableDataset(
            str(self._data_dir),
            replay_filter=self._filter,
            replay_config=self._config,
            numeric_columns=numeric_columns,
        )


class Schema:
    """Schema utility class for accessing canonical market data schemas."""

    @staticmethod
    def trade(version: int = 1) -> pa.Schema:
        return get_schema(MarketDataType.TRADE, version=version)

    @staticmethod
    def quote(version: int = 1) -> pa.Schema:
        return get_schema(MarketDataType.QUOTE, version=version)

    @staticmethod
    def bar(version: int = 1) -> pa.Schema:
        return get_schema(MarketDataType.BAR, version=version)
