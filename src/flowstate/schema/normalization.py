"""Zero-copy normalization with A/B line arbitration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from flowstate.schema.types import TIMESTAMP_NS, MarketDataType, get_schema


@dataclass
class FieldMapping:
    """Maps a source field name to a target field name with optional transform."""

    source: str
    target: str
    transform: Any = None  # Callable[[Any], Any] | None


@dataclass
class NormalizationProfile:
    """Defines how to normalize a specific source's data to canonical schema."""

    source_name: str
    data_type: MarketDataType
    field_mappings: list[FieldMapping]
    timestamp_unit: str = "ns"
    timestamp_field: str = "timestamp"
    defaults: dict[str, Any] = field(default_factory=dict)


def _convert_timestamp(value: Any, source_unit: str) -> int:
    """Convert a timestamp value to nanoseconds since epoch."""
    if isinstance(value, int):
        multipliers = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}
        return value * multipliers.get(source_unit, 1)
    return int(value)


class Normalizer:
    """Zero-copy normalizer that maps vendor-specific data to canonical Arrow schemas.

    Supports field remapping, timestamp unit conversion, and default value injection.
    """

    def __init__(self, profile: NormalizationProfile) -> None:
        self._profile = profile
        self._target_schema = get_schema(profile.data_type)
        self._mapping_index = {m.source: m for m in profile.field_mappings}

    @property
    def profile(self) -> NormalizationProfile:
        return self._profile

    @property
    def target_schema(self) -> pa.Schema:
        return self._target_schema

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single raw record to the canonical schema.

        Args:
            raw: Raw vendor-specific record as a dict.

        Returns:
            Dict with keys matching the canonical schema fields.
        """
        result: dict[str, Any] = {}
        receive_ts = time.time_ns()

        for target_field in self._target_schema.names:
            # Check explicit mappings
            mapped = False
            for mapping in self._profile.field_mappings:
                if mapping.target == target_field:
                    if mapping.source in raw:
                        value = raw[mapping.source]
                        if mapping.transform is not None:
                            value = mapping.transform(value)
                        result[target_field] = value
                        mapped = True
                        break

            if not mapped:
                # Try direct name match
                if target_field in raw:
                    result[target_field] = raw[target_field]
                elif target_field in self._profile.defaults:
                    result[target_field] = self._profile.defaults[target_field]
                elif target_field == "receive_timestamp":
                    result[target_field] = receive_ts
                else:
                    result[target_field] = None

        # Convert timestamp fields (receive_timestamp is already in ns if auto-generated)
        auto_receive = "receive_timestamp" not in raw and not any(
            m.target == "receive_timestamp" for m in self._profile.field_mappings
        )
        convert_fields = ["timestamp", "exchange_timestamp"]
        if not auto_receive:
            convert_fields.append("receive_timestamp")
        for field_name in convert_fields:
            if field_name in result and result[field_name] is not None:
                result[field_name] = _convert_timestamp(
                    result[field_name], self._profile.timestamp_unit
                )

        return result

    def normalize_batch(self, records: list[dict[str, Any]]) -> pa.RecordBatch:
        """Normalize a batch of raw records into an Arrow RecordBatch.

        Args:
            records: List of raw vendor-specific records.

        Returns:
            A PyArrow RecordBatch conforming to the canonical schema.
        """
        normalized = [self.normalize(r) for r in records]
        columns: dict[str, list[Any]] = {name: [] for name in self._target_schema.names}
        for record in normalized:
            for name in self._target_schema.names:
                columns[name].append(record.get(name))

        arrays = []
        for i, name in enumerate(self._target_schema.names):
            field_type = self._target_schema.field(i).type
            arrays.append(pa.array(columns[name], type=field_type))

        return pa.RecordBatch.from_arrays(arrays, schema=self._target_schema)


class ABLineArbiter:
    """Arbitrates between dual A/B feed lines for redundancy.

    Selects the record with the best (earliest) exchange timestamp to ensure
    lowest latency and deduplicates across lines.
    """

    def __init__(self, dedup_window_ns: int = 1_000_000) -> None:
        self._dedup_window_ns = dedup_window_ns
        self._seen: dict[str, int] = {}

    def arbitrate(
        self,
        record_a: dict[str, Any] | None,
        record_b: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Select the best record from A/B lines.

        Args:
            record_a: Record from line A (or None).
            record_b: Record from line B (or None).

        Returns:
            The selected record, or None if both are duplicates.
        """
        if record_a is None and record_b is None:
            return None
        if record_a is None:
            return self._check_dedup(record_b)
        if record_b is None:
            return self._check_dedup(record_a)

        # Pick the one with earlier exchange timestamp
        ts_a = record_a.get("exchange_timestamp") or record_a.get("timestamp", 0)
        ts_b = record_b.get("exchange_timestamp") or record_b.get("timestamp", 0)

        winner = record_a if ts_a <= ts_b else record_b
        return self._check_dedup(winner)

    def _check_dedup(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None

        key = self._dedup_key(record)
        ts = record.get("exchange_timestamp") or record.get("timestamp", 0)

        if key in self._seen:
            if abs(ts - self._seen[key]) <= self._dedup_window_ns:
                return None

        self._seen[key] = ts
        return record

    @staticmethod
    def _dedup_key(record: dict[str, Any]) -> str:
        symbol = record.get("symbol", "")
        ts = record.get("exchange_timestamp") or record.get("timestamp", "")
        price = record.get("price", record.get("bid_price", ""))
        return f"{symbol}:{ts}:{price}"

    def clear(self) -> None:
        """Clear the deduplication cache."""
        self._seen.clear()
