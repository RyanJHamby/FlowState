"""Partitioned Parquet writer with zstd compression."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from flowstate.storage.partitioning import PartitionScheme

logger = logging.getLogger(__name__)


@dataclass
class WriterConfig:
    """Configuration for the partitioned Parquet writer."""

    base_path: str | Path
    compression: str = "zstd"
    compression_level: int = 3
    row_group_size: int = 128 * 1024  # 128K rows per row group
    max_rows_per_file: int = 1_000_000
    num_buckets: int = 256
    version: str = "2.6"


@dataclass
class WriteStats:
    """Statistics for the writer."""

    files_written: int = 0
    rows_written: int = 0
    bytes_written: int = 0
    partitions_active: int = 0


class PartitionBuffer:
    """Accumulates rows for a single partition until flush."""

    def __init__(self, schema: pa.Schema, max_rows: int) -> None:
        self._schema = schema
        self._max_rows = max_rows
        self._batches: list[pa.RecordBatch] = []
        self._row_count = 0

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def is_full(self) -> bool:
        return self._row_count >= self._max_rows

    def append(self, batch: pa.RecordBatch) -> None:
        self._batches.append(batch)
        self._row_count += batch.num_rows

    def flush(self) -> pa.Table:
        """Flush accumulated batches into a single Table and reset."""
        table = pa.Table.from_batches(self._batches, schema=self._schema)
        self._batches.clear()
        self._row_count = 0
        return table


class PartitionedParquetWriter:
    """Writes Arrow data to partitioned Parquet files with zstd compression.

    Data is partitioned by type, date, and symbol bucket using the PartitionScheme.
    Files are flushed when they reach max_rows_per_file.
    """

    def __init__(self, config: WriterConfig) -> None:
        self._config = config
        self._base_path = Path(config.base_path)
        self._scheme = PartitionScheme(num_buckets=config.num_buckets)
        self._lock = threading.Lock()
        self._buffers: dict[str, PartitionBuffer] = {}
        self._stats = WriteStats()
        self._file_counter: dict[str, int] = {}

    @property
    def config(self) -> WriterConfig:
        return self._config

    @property
    def stats(self) -> WriteStats:
        return self._stats

    def write(self, batch: pa.RecordBatch, data_type: str) -> list[Path]:
        """Write a RecordBatch, partitioning by symbol and timestamp.

        Args:
            batch: The RecordBatch to write.
            data_type: Market data type (e.g. "trade").

        Returns:
            List of file paths written (only files flushed during this call).
        """
        written_files: list[Path] = []

        # Group rows by partition key
        if "symbol" not in batch.schema.names or "timestamp" not in batch.schema.names:
            # Write all rows to a single partition
            path = self._write_partition(batch, f"type={data_type}/unpartitioned")
            if path:
                written_files.append(path)
            return written_files

        symbols = batch.column("symbol").to_pylist()
        # Use .cast to int64 to avoid nanosecond datetime conversion issues
        ts_array = batch.column("timestamp").cast(pa.int64())
        timestamps = ts_array.to_pylist()

        # Group indices by partition path
        partition_indices: dict[str, list[int]] = {}
        for i, (sym, ts_val) in enumerate(zip(symbols, timestamps, strict=False)):
            if ts_val is None:
                continue
            key = self._scheme.partition_key(sym, ts_val, data_type)
            path_str = key.path
            partition_indices.setdefault(path_str, []).append(i)

        for part_path, indices in partition_indices.items():
            sub_batch = batch.take(indices)
            path = self._write_partition(sub_batch, part_path)
            if path:
                written_files.append(path)

        return written_files

    def _write_partition(self, batch: pa.RecordBatch, partition_path: str) -> Path | None:
        """Buffer data for a partition and flush if full."""
        with self._lock:
            buf = self._buffers.get(partition_path)
            if buf is None:
                buf = PartitionBuffer(batch.schema, self._config.max_rows_per_file)
                self._buffers[partition_path] = buf
                self._stats.partitions_active = len(self._buffers)

            buf.append(batch)

            if buf.is_full:
                return self._flush_partition(partition_path, buf)
        return None

    def _flush_partition(self, partition_path: str, buf: PartitionBuffer) -> Path:
        """Flush a partition buffer to a Parquet file."""
        table = buf.flush()
        dir_path = self._base_path / partition_path
        dir_path.mkdir(parents=True, exist_ok=True)

        file_num = self._file_counter.get(partition_path, 0)
        self._file_counter[partition_path] = file_num + 1
        file_path = dir_path / f"part-{file_num:05d}.parquet"

        pq.write_table(
            table,
            file_path,
            compression=self._config.compression,
            compression_level=self._config.compression_level,
            row_group_size=self._config.row_group_size,
            version=self._config.version,
        )

        self._stats.files_written += 1
        self._stats.rows_written += table.num_rows
        self._stats.bytes_written += file_path.stat().st_size

        logger.debug(f"Wrote {table.num_rows} rows to {file_path}")
        return file_path

    def flush_all(self) -> list[Path]:
        """Flush all partition buffers to disk."""
        written: list[Path] = []
        with self._lock:
            for part_path, buf in list(self._buffers.items()):
                if buf.row_count > 0:
                    path = self._flush_partition(part_path, buf)
                    written.append(path)
        return written

    def close(self) -> list[Path]:
        """Flush remaining data and clean up."""
        written = self.flush_all()
        with self._lock:
            self._buffers.clear()
            self._stats.partitions_active = 0
        return written
