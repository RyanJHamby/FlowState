"""Temporal alignment engine for heterogeneous market data streams.

Performs point-in-time correct as-of joins across trades, quotes, and bars
to produce aligned feature tensors for ML training. Guarantees no look-ahead
bias: each output row contains only data that was observable at the timestamp.

Key operations:
- as_of_join: Forward-fill join of a secondary stream onto a primary timeline
- align_streams: Multi-stream alignment producing a unified wide table
- TemporalAligner: Stateful engine for batch-level alignment with partition support

All operations work on PyArrow tables and preserve nanosecond precision.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

_NS_DTYPE = pa.int64()


@dataclass(frozen=True)
class AsOfConfig:
    """Configuration for a single as-of join."""

    right_prefix: str = ""
    tolerance_ns: int | None = None
    allow_exact_match: bool = True
    direction: Literal["backward", "forward", "nearest"] = "backward"


@dataclass
class AlignmentSpec:
    """Specification for a named data stream in multi-stream alignment."""

    name: str
    table: pa.Table
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    value_columns: list[str] | None = None
    config: AsOfConfig = field(default_factory=AsOfConfig)


@dataclass
class AlignmentStats:
    """Statistics from an alignment operation."""

    left_rows: int = 0
    right_rows: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    tolerance_exceeded: int = 0


def _to_ns_int64(col: pa.ChunkedArray | pa.Array) -> pa.Array:
    """Convert a timestamp or int64 column to a flat int64 array of nanoseconds."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_timestamp(col.type):
        return col.cast(_NS_DTYPE)
    return col


def as_of_join(
    left: pa.Table,
    right: pa.Table,
    on: str = "timestamp",
    by: str | None = "symbol",
    config: AsOfConfig | None = None,
) -> tuple[pa.Table, AlignmentStats]:
    """Point-in-time correct as-of join.

    For each row in `left`, finds the most recent row in `right` whose
    timestamp is <= the left timestamp (backward join). Guarantees no
    look-ahead bias.

    Args:
        left: Primary timeline table (defines output rows).
        right: Secondary data to join (forward-filled onto left timeline).
        on: Timestamp column name (must exist in both tables).
        by: Group-by column for per-symbol joins. None for global join.
        config: Join configuration (tolerance, direction, prefix).

    Returns:
        Tuple of (joined table, alignment statistics).
    """
    cfg = config or AsOfConfig()
    stats = AlignmentStats(left_rows=left.num_rows, right_rows=right.num_rows)

    if left.num_rows == 0:
        return left, stats

    # Determine right-side value columns
    right_value_cols = [
        c for c in right.schema.names
        if c != on and c != by
    ]

    if not right_value_cols:
        return left, stats

    # Prefix right columns to avoid collisions
    prefixed_names = {
        c: f"{cfg.right_prefix}{c}" if cfg.right_prefix else c
        for c in right_value_cols
    }

    if right.num_rows == 0:
        # No right data — append null columns
        return _append_null_columns(left, right, right_value_cols, prefixed_names, stats)

    if by is not None and by in left.schema.names and by in right.schema.names:
        result = _as_of_join_grouped(left, right, on, by, right_value_cols, prefixed_names, cfg, stats)
    else:
        result = _as_of_join_ungrouped(left, right, on, right_value_cols, prefixed_names, cfg, stats)

    stats.unmatched_rows = stats.left_rows - stats.matched_rows
    return result, stats


def _append_null_columns(
    left: pa.Table,
    right: pa.Table,
    right_value_cols: list[str],
    prefixed_names: dict[str, str],
    stats: AlignmentStats,
) -> tuple[pa.Table, AlignmentStats]:
    """Append all-null columns from right schema to left table."""
    result = left
    for col_name in right_value_cols:
        col_type = right.schema.field(col_name).type
        null_col = pa.nulls(left.num_rows, type=col_type)
        output_name = prefixed_names.get(col_name, col_name)
        result = result.append_column(output_name, null_col)
    return result, stats


def _as_of_join_ungrouped(
    left: pa.Table,
    right: pa.Table,
    on: str,
    right_value_cols: list[str],
    prefixed_names: dict[str, str],
    cfg: AsOfConfig,
    stats: AlignmentStats,
) -> pa.Table:
    """As-of join without grouping."""
    left_ts = _to_ns_int64(left.column(on))
    right_ts = _to_ns_int64(right.column(on))

    # Sort right by timestamp
    right_order = pc.sort_indices(right_ts)
    right_ts_sorted = right_ts.take(right_order)
    right_sorted = right.take(right_order)

    indices = _build_asof_indices(left_ts, right_ts_sorted, cfg)
    return _gather_and_append(left, right_sorted, right_value_cols, prefixed_names, indices, stats)


def _as_of_join_grouped(
    left: pa.Table,
    right: pa.Table,
    on: str,
    by: str,
    right_value_cols: list[str],
    prefixed_names: dict[str, str],
    cfg: AsOfConfig,
    stats: AlignmentStats,
) -> pa.Table:
    """As-of join with per-group (per-symbol) processing."""
    left_syms = left.column(by).to_pylist()
    right_syms = right.column(by).to_pylist()

    unique_symbols = list(set(left_syms))

    # Pre-index right side by symbol
    right_ts_all = _to_ns_int64(right.column(on)).to_pylist()
    right_by_sym: dict[str, list[tuple[int, int]]] = {}
    for i, sym in enumerate(right_syms):
        right_by_sym.setdefault(sym, []).append((right_ts_all[i], i))

    # Sort each symbol's right entries by timestamp
    for sym in right_by_sym:
        right_by_sym[sym].sort(key=lambda x: x[0])

    left_ts_all = _to_ns_int64(left.column(on)).to_pylist()

    # Build global index mapping
    global_indices: list[int | None] = [None] * left.num_rows

    for sym in unique_symbols:
        right_entries = right_by_sym.get(sym)
        if not right_entries:
            continue

        right_ts_sorted = [e[0] for e in right_entries]
        right_global_idx = [e[1] for e in right_entries]

        # Process all left rows for this symbol
        for i, (ls, lt) in enumerate(zip(left_syms, left_ts_all)):
            if ls != sym:
                continue
            local_idx = _find_match(lt, right_ts_sorted, len(right_ts_sorted), cfg)
            if local_idx is not None:
                global_indices[i] = right_global_idx[local_idx]

    return _gather_and_append(left, right, right_value_cols, prefixed_names, global_indices, stats)


def _build_asof_indices(
    left_ts: pa.Array,
    right_ts_sorted: pa.Array,
    cfg: AsOfConfig,
) -> list[int | None]:
    """Build index mapping from left rows to right rows."""
    n_right = len(right_ts_sorted)
    if n_right == 0:
        return [None] * len(left_ts)

    left_values = left_ts.to_pylist()
    right_values = right_ts_sorted.to_pylist()

    return [
        _find_match(lt, right_values, n_right, cfg) if lt is not None else None
        for lt in left_values
    ]


def _find_match(
    left_val: int,
    right_values: list[int],
    n_right: int,
    cfg: AsOfConfig,
) -> int | None:
    """Find the matching right index for a single left timestamp."""
    if cfg.direction == "backward":
        if cfg.allow_exact_match:
            pos = bisect.bisect_right(right_values, left_val) - 1
        else:
            pos = bisect.bisect_left(right_values, left_val) - 1
        if pos < 0:
            return None
        if cfg.tolerance_ns is not None and (left_val - right_values[pos]) > cfg.tolerance_ns:
            return None
        return pos

    elif cfg.direction == "forward":
        if cfg.allow_exact_match:
            pos = bisect.bisect_left(right_values, left_val)
        else:
            pos = bisect.bisect_right(right_values, left_val)
        if pos >= n_right:
            return None
        if cfg.tolerance_ns is not None and (right_values[pos] - left_val) > cfg.tolerance_ns:
            return None
        return pos

    else:  # nearest
        back_pos = bisect.bisect_right(right_values, left_val) - 1
        fwd_pos = bisect.bisect_left(right_values, left_val)

        back_dist = abs(left_val - right_values[back_pos]) if back_pos >= 0 else None
        fwd_dist = abs(right_values[fwd_pos] - left_val) if fwd_pos < n_right else None

        if back_dist is None and fwd_dist is None:
            return None

        if back_dist is not None and fwd_dist is not None:
            best_pos = back_pos if back_dist <= fwd_dist else fwd_pos
            best_dist = min(back_dist, fwd_dist)
        elif back_dist is not None:
            best_pos = back_pos
            best_dist = back_dist
        else:
            best_pos = fwd_pos
            best_dist = fwd_dist

        if not cfg.allow_exact_match and best_dist == 0:
            return None
        if cfg.tolerance_ns is not None and best_dist > cfg.tolerance_ns:
            return None
        return best_pos


def _gather_and_append(
    left: pa.Table,
    right: pa.Table,
    right_value_cols: list[str],
    prefixed_names: dict[str, str],
    indices: list[int | None],
    stats: AlignmentStats,
) -> pa.Table:
    """Gather right-side columns by index and append to left table."""
    stats.matched_rows = sum(1 for idx in indices if idx is not None)
    n = len(indices)
    has_nulls = stats.matched_rows < n

    result = left
    for col_name in right_value_cols:
        col_type = right.schema.field(col_name).type
        right_col = right.column(col_name)
        if isinstance(right_col, pa.ChunkedArray):
            right_col = right_col.combine_chunks()

        if has_nulls:
            # Build column value-by-value to handle nulls safely across all types
            values = []
            for idx in indices:
                if idx is not None:
                    values.append(right_col[idx].as_py())
                else:
                    values.append(None)
            gathered = pa.array(values, type=col_type)
        else:
            take_indices = pa.array([idx for idx in indices], type=pa.int32())
            gathered = right_col.take(take_indices)

        output_name = prefixed_names.get(col_name, col_name)
        result = result.append_column(output_name, gathered)

    return result


def align_streams(
    primary: pa.Table,
    secondaries: list[AlignmentSpec],
    primary_timestamp_col: str = "timestamp",
    primary_symbol_col: str = "symbol",
) -> tuple[pa.Table, dict[str, AlignmentStats]]:
    """Align multiple data streams onto a primary timeline.

    Performs sequential as-of joins of each secondary stream onto the
    primary table. Each secondary stream's columns are prefixed with
    the stream name to avoid collisions.

    Args:
        primary: The primary timeline (defines output rows).
        secondaries: List of secondary streams to join.
        primary_timestamp_col: Timestamp column in primary table.
        primary_symbol_col: Symbol column in primary table.

    Returns:
        Tuple of (aligned table, per-stream statistics).
    """
    result = primary
    all_stats: dict[str, AlignmentStats] = {}

    for spec in secondaries:
        # Determine value columns
        if spec.value_columns is not None:
            keep = list(set(spec.value_columns + [spec.timestamp_col]))
            if spec.symbol_col in spec.table.schema.names:
                keep.append(spec.symbol_col)
            keep = [c for c in keep if c in spec.table.schema.names]
            right = spec.table.select(keep)
        else:
            right = spec.table

        prefix = spec.config.right_prefix or f"{spec.name}_"
        cfg = AsOfConfig(
            right_prefix=prefix,
            tolerance_ns=spec.config.tolerance_ns,
            allow_exact_match=spec.config.allow_exact_match,
            direction=spec.config.direction,
        )

        by_col = primary_symbol_col
        if by_col not in result.schema.names or by_col not in right.schema.names:
            by_col = None

        result, join_stats = as_of_join(
            result, right,
            on=primary_timestamp_col,
            by=by_col,
            config=cfg,
        )
        all_stats[spec.name] = join_stats

    return result, all_stats


class TemporalAligner:
    """Stateful temporal alignment engine for batch-level processing.

    Maintains state across batches to handle late-arriving data and
    produces aligned output incrementally. Designed for integration
    with ReplayEngine for historical replay alignment.

    Example::

        aligner = TemporalAligner(
            primary_type="trade",
            secondary_specs={"quote": ["bid_price", "ask_price", "bid_size", "ask_size"]},
        )
        aligner.add_data("trade", trade_table)
        aligner.add_data("quote", quote_table)
        aligned, stats = aligner.flush()
    """

    def __init__(
        self,
        primary_type: str,
        secondary_specs: dict[str, list[str] | None] | None = None,
        tolerance_ns: int | None = None,
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
    ) -> None:
        self._primary_type = primary_type
        self._secondary_specs = secondary_specs or {}
        self._tolerance_ns = tolerance_ns
        self._timestamp_col = timestamp_col
        self._symbol_col = symbol_col

        self._buffers: dict[str, list[pa.Table]] = {primary_type: []}
        for name in self._secondary_specs:
            self._buffers[name] = []

        self._total_aligned: int = 0

    @property
    def primary_type(self) -> str:
        return self._primary_type

    @property
    def total_aligned(self) -> int:
        return self._total_aligned

    def add_data(self, stream_name: str, table: pa.Table) -> None:
        """Add a batch of data for a named stream."""
        if stream_name not in self._buffers:
            raise ValueError(
                f"Unknown stream '{stream_name}'. "
                f"Expected one of: {list(self._buffers.keys())}"
            )
        if table.num_rows > 0:
            self._buffers[stream_name].append(table)

    def flush(self) -> tuple[pa.Table | None, dict[str, AlignmentStats]]:
        """Align buffered data and return the result."""
        primary_tables = self._buffers[self._primary_type]
        if not primary_tables:
            return None, {}

        primary = pa.concat_tables(primary_tables) if len(primary_tables) > 1 else primary_tables[0]

        if self._timestamp_col in primary.schema.names:
            order = pc.sort_indices(primary, sort_keys=[(self._timestamp_col, "ascending")])
            primary = primary.take(order)

        secondaries: list[AlignmentSpec] = []
        for name, value_cols in self._secondary_specs.items():
            tables = self._buffers.get(name, [])
            if not tables:
                continue
            secondary = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

            if self._timestamp_col in secondary.schema.names:
                order = pc.sort_indices(secondary, sort_keys=[(self._timestamp_col, "ascending")])
                secondary = secondary.take(order)

            secondaries.append(AlignmentSpec(
                name=name,
                table=secondary,
                timestamp_col=self._timestamp_col,
                symbol_col=self._symbol_col,
                value_columns=value_cols,
                config=AsOfConfig(tolerance_ns=self._tolerance_ns),
            ))

        if secondaries:
            result, stats = align_streams(
                primary, secondaries,
                primary_timestamp_col=self._timestamp_col,
                primary_symbol_col=self._symbol_col,
            )
        else:
            result = primary
            stats = {}

        self._total_aligned += result.num_rows

        for name in self._buffers:
            self._buffers[name] = []

        return result, stats

    def reset(self) -> None:
        """Clear all buffered data without flushing."""
        for name in self._buffers:
            self._buffers[name] = []
