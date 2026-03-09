"""Deterministic Hive partitioning with xxhash-based bucket distribution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import xxhash


@dataclass(frozen=True)
class PartitionKey:
    """A deterministic partition key for Hive-style partitioned storage."""

    data_type: str
    date: str  # YYYY-MM-DD
    bucket: int

    @property
    def path(self) -> str:
        """Generate a Hive-style partition path."""
        return f"type={self.data_type}/date={self.date}/bucket={self.bucket:04d}"

    def full_path(self, base: str | Path) -> Path:
        return Path(base) / self.path


class PartitionScheme:
    """Deterministic Hive partitioning with xxhash-based bucket distribution.

    Distributes symbols across buckets using xxhash to prevent hot partitions
    from high-volume symbols. The number of buckets is fixed at creation time
    for query locality.

    Args:
        num_buckets: Number of hash buckets (default: 256).
    """

    def __init__(self, num_buckets: int = 256) -> None:
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1")
        self._num_buckets = num_buckets

    @property
    def num_buckets(self) -> int:
        return self._num_buckets

    def bucket_for(self, symbol: str) -> int:
        """Compute the deterministic bucket for a symbol.

        Uses xxhash64 for fast, well-distributed hashing.

        Args:
            symbol: The symbol/instrument identifier.

        Returns:
            Bucket index in [0, num_buckets).
        """
        h = xxhash.xxh64(symbol.encode()).intdigest()
        return h % self._num_buckets

    def partition_key(
        self,
        symbol: str,
        timestamp_ns: int,
        data_type: str,
    ) -> PartitionKey:
        """Compute the full partition key for a record.

        Args:
            symbol: The symbol identifier.
            timestamp_ns: Timestamp in nanoseconds since epoch.
            data_type: Market data type (e.g. "trade", "quote", "bar").

        Returns:
            A PartitionKey with type, date, and bucket.
        """
        dt = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        bucket = self.bucket_for(symbol)
        return PartitionKey(data_type=data_type, date=date_str, bucket=bucket)

    def partition_path(
        self,
        symbol: str,
        timestamp_ns: int,
        data_type: str,
        base: str | Path = "",
    ) -> Path:
        """Compute the full output path for a record.

        Args:
            symbol: The symbol identifier.
            timestamp_ns: Timestamp in nanoseconds since epoch.
            data_type: Market data type.
            base: Base directory path.

        Returns:
            Full path including Hive partitioning.
        """
        key = self.partition_key(symbol, timestamp_ns, data_type)
        return key.full_path(base)
