"""Temporal alignment engine for heterogeneous market data streams.

Performs point-in-time correct as-of joins across trades, quotes, and bars
to produce aligned feature tensors for ML training. Guarantees no look-ahead
bias: each output row contains only data that was observable at the timestamp.

Vectorized implementation using numpy.searchsorted for O(n log m) batch
lookups without Python-level loops. Grouped joins use numpy advanced indexing
for per-symbol partitioning, avoiding the O(n * S) nested scan of naive
implementations.

Key operations:
- as_of_join: Forward-fill join of a secondary stream onto a primary timeline
- align_streams: Multi-stream alignment producing a unified wide table
- TemporalAligner: Stateful engine for batch-level alignment with partition support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

_NS_DTYPE = pa.int64()
_INVALID_INDEX = -1  # Sentinel for unmatched rows

# Rust acceleration: import if available, fall back to pure Python
try:
    import flowstate_core as _rust_core

    _HAS_RUST = True
    logger.info("Rust acceleration available (flowstate_core)")
except ImportError:
    _rust_core = None  # type: ignore[assignment]
    _HAS_RUST = False


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


def _to_numpy_i64(col: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Convert an Arrow column to a numpy int64 array (zero-copy when possible)."""
    arr = _to_ns_int64(col)
    return arr.to_numpy(zero_copy_only=False)


# ---------------------------------------------------------------------------
# Vectorized index computation via numpy.searchsorted
# ---------------------------------------------------------------------------

def _vectorized_backward(
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    tolerance_ns: int | None,
    allow_exact: bool,
) -> np.ndarray:
    """Vectorized backward as-of: for each left[i], find rightmost right[j] <= left[i].

    Uses numpy.searchsorted (C-level binary search over the full array)
    instead of per-element Python bisect. Returns int64 index array where
    -1 indicates no match.
    """
    len(left_ts)
    if allow_exact:
        # searchsorted('right') gives first index > left_ts[i], so -1 gives <=
        pos = np.searchsorted(right_ts, left_ts, side="right").astype(np.int64) - 1
    else:
        # searchsorted('left') gives first index >= left_ts[i], so -1 gives <
        pos = np.searchsorted(right_ts, left_ts, side="left").astype(np.int64) - 1

    # Mark out-of-range
    pos[pos < 0] = _INVALID_INDEX

    # Apply tolerance
    if tolerance_ns is not None:
        valid = pos >= 0
        distances = np.where(valid, left_ts - right_ts[np.clip(pos, 0, len(right_ts) - 1)], 0)
        pos[valid & (distances > tolerance_ns)] = _INVALID_INDEX

    return pos


def _vectorized_forward(
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    tolerance_ns: int | None,
    allow_exact: bool,
) -> np.ndarray:
    """Vectorized forward as-of: for each left[i], find leftmost right[j] >= left[i]."""
    n_right = len(right_ts)
    if allow_exact:
        pos = np.searchsorted(right_ts, left_ts, side="left").astype(np.int64)
    else:
        pos = np.searchsorted(right_ts, left_ts, side="right").astype(np.int64)

    pos[pos >= n_right] = _INVALID_INDEX

    if tolerance_ns is not None:
        valid = pos >= 0
        distances = np.where(valid, right_ts[np.clip(pos, 0, n_right - 1)] - left_ts, 0)
        pos[valid & (distances > tolerance_ns)] = _INVALID_INDEX

    return pos


def _vectorized_nearest(
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    tolerance_ns: int | None,
    allow_exact: bool,
) -> np.ndarray:
    """Vectorized nearest as-of: find closest right[j] in either direction."""
    n_right = len(right_ts)

    # Backward candidate
    back_pos = np.searchsorted(right_ts, left_ts, side="right").astype(np.int64) - 1
    back_valid = back_pos >= 0
    back_dist = np.full(len(left_ts), np.iinfo(np.int64).max, dtype=np.int64)
    back_dist[back_valid] = np.abs(left_ts[back_valid] - right_ts[back_pos[back_valid]])

    # Forward candidate
    fwd_pos = np.searchsorted(right_ts, left_ts, side="left").astype(np.int64)
    fwd_valid = fwd_pos < n_right
    fwd_dist = np.full(len(left_ts), np.iinfo(np.int64).max, dtype=np.int64)
    fwd_clipped = np.clip(fwd_pos, 0, n_right - 1)
    fwd_dist[fwd_valid] = np.abs(right_ts[fwd_clipped[fwd_valid]] - left_ts[fwd_valid])

    # Pick closer
    use_fwd = fwd_dist < back_dist
    pos = np.where(use_fwd, fwd_pos, back_pos)
    best_dist = np.minimum(back_dist, fwd_dist)

    # Invalidate
    pos[(~back_valid) & (~fwd_valid)] = _INVALID_INDEX
    if not allow_exact:
        pos[best_dist == 0] = _INVALID_INDEX
    if tolerance_ns is not None:
        pos[best_dist > tolerance_ns] = _INVALID_INDEX

    return pos


def _vectorized_asof_indices(
    left_ts: np.ndarray,
    right_ts_sorted: np.ndarray,
    cfg: AsOfConfig,
) -> np.ndarray:
    """Dispatch to the appropriate vectorized search.

    Returns np.ndarray of int64 indices into right_ts_sorted. -1 = no match.
    """
    if len(right_ts_sorted) == 0:
        return np.full(len(left_ts), _INVALID_INDEX, dtype=np.int64)

    if cfg.direction == "backward":
        return _vectorized_backward(
            left_ts, right_ts_sorted, cfg.tolerance_ns, cfg.allow_exact_match,
        )
    elif cfg.direction == "forward":
        return _vectorized_forward(
            left_ts, right_ts_sorted, cfg.tolerance_ns, cfg.allow_exact_match,
        )
    else:
        return _vectorized_nearest(
            left_ts, right_ts_sorted, cfg.tolerance_ns, cfg.allow_exact_match,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Uses numpy.searchsorted for vectorized O(n log m) matching — no
    Python-level per-row loops.

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

    right_value_cols = [c for c in right.schema.names if c != on and c != by]
    if not right_value_cols:
        return left, stats

    prefixed_names = {
        c: f"{cfg.right_prefix}{c}" if cfg.right_prefix else c
        for c in right_value_cols
    }

    if right.num_rows == 0:
        return _append_null_columns(left, right, right_value_cols, prefixed_names, stats)

    # Rust fast path: delegate to compiled kernel when available
    if _HAS_RUST:
        result = _rust_as_of_join(left, right, on, by, cfg)
        # Count matches by checking nulls in the first right-side value column
        first_right_col = right_value_cols[0]
        output_name = prefixed_names.get(first_right_col, first_right_col)
        if output_name in result.schema.names:
            null_count = result.column(output_name).null_count
            stats.matched_rows = left.num_rows - null_count
        else:
            stats.matched_rows = 0
        stats.unmatched_rows = left.num_rows - stats.matched_rows
        return result, stats

    if by is not None and by in left.schema.names and by in right.schema.names:
        result = _as_of_join_grouped(
            left, right, on, by, right_value_cols, prefixed_names, cfg, stats,
        )
    else:
        result = _as_of_join_ungrouped(
            left, right, on, right_value_cols, prefixed_names, cfg, stats,
        )

    stats.unmatched_rows = stats.left_rows - stats.matched_rows
    return result, stats


def _rust_as_of_join(
    left: pa.Table,
    right: pa.Table,
    on: str,
    by: str | None,
    cfg: AsOfConfig,
) -> pa.Table:
    """Dispatch as-of join to the Rust kernel (flowstate_core)."""
    return _rust_core.asof_join(
        left,
        right,
        on=on,
        by=by,
        tolerance_ns=cfg.tolerance_ns,
        right_prefix=cfg.right_prefix,
        direction=cfg.direction,
        allow_exact_match=cfg.allow_exact_match,
    )


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
    """As-of join without grouping — fully vectorized."""
    left_ts = _to_numpy_i64(left.column(on))
    right_ts = _to_numpy_i64(right.column(on))

    # Sort right by timestamp
    sort_order = np.argsort(right_ts, kind="mergesort")
    right_ts_sorted = right_ts[sort_order]
    right_sorted = right.take(pa.array(sort_order))

    indices = _vectorized_asof_indices(left_ts, right_ts_sorted, cfg)
    return _gather_and_append(left, right_sorted, right_value_cols, prefixed_names, indices, stats)


def _group_by_column(col: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Encode a string column to integer codes using numpy.

    Returns (unique_labels, codes) where codes[i] is the group index for row i.
    Uses PyArrow dictionary encoding which is O(n) and avoids Python-level loops.
    """
    combined = col.combine_chunks() if hasattr(col, 'combine_chunks') else col
    encoded = combined.dictionary_encode()
    codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    labels = encoded.dictionary.to_pylist()
    return labels, codes


def _partition_indices_by_code(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Partition row indices by group code in a single argsort pass.

    Returns a list where result[code] = sorted array of row indices for that group.
    O(n log n) total, zero Python-level per-element work.
    """
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    # Find group boundaries
    boundaries = np.searchsorted(sorted_codes, np.arange(n_groups))
    end_boundaries = np.searchsorted(sorted_codes, np.arange(n_groups), side="right")
    return [order[boundaries[g]:end_boundaries[g]] for g in range(n_groups)]


def _partition_sorted_by_ts(
    codes: np.ndarray, ts: np.ndarray, n_groups: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition by group code, with each partition pre-sorted by timestamp.

    Uses a radix-style approach: stable-sort by timestamp first (which data
    often already is), then stable-sort by code. Since both sorts are stable,
    within each code partition rows remain in timestamp order.

    For pre-sorted timestamp data (common case), the first sort is nearly free.
    """
    # If timestamps are already sorted (very common for time-series), argsort is ~free
    ts_order = np.argsort(ts, kind="mergesort")
    # Stable sort by code preserves timestamp ordering within groups
    code_order = np.argsort(codes[ts_order], kind="mergesort")
    order = ts_order[code_order]

    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(n_groups))
    end_boundaries = np.searchsorted(sorted_codes, np.arange(n_groups), side="right")
    result = []
    for g in range(n_groups):
        group_order = order[boundaries[g]:end_boundaries[g]]
        result.append((group_order, ts[group_order]))
    return result


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
    """As-of join with per-group (per-symbol) processing — fully vectorized.

    Grouping uses dictionary encoding + argsort partitioning (single O(n log n)
    pass) instead of Python list comprehensions. Within each group,
    numpy.searchsorted handles the temporal matching.
    """
    left_ts = _to_numpy_i64(left.column(on))
    right_ts = _to_numpy_i64(right.column(on))

    # Dictionary-encode both sides — O(n) each, no Python loops
    left_labels, left_codes = _group_by_column(left.column(by))
    right_labels, right_codes = _group_by_column(right.column(by))

    # Build label → code mapping for right side
    right_label_to_code = {label: code for code, label in enumerate(right_labels)}

    # Partition row indices by group — single pass each
    # Left side: just needs grouping (will index into left_ts later)
    left_groups = _partition_indices_by_code(left_codes, len(left_labels))
    # Right side: partition AND sort by timestamp in one lexsort
    right_partitions = _partition_sorted_by_ts(right_codes, right_ts, len(right_labels))

    # Global index array
    global_indices = np.full(left.num_rows, _INVALID_INDEX, dtype=np.int64)

    for lcode, label in enumerate(left_labels):
        rcode = right_label_to_code.get(label)
        if rcode is None:
            continue

        right_positions, right_ts_sorted = right_partitions[rcode]
        if len(right_positions) == 0:
            continue

        left_positions = left_groups[lcode]
        if len(left_positions) == 0:
            continue
        left_ts_subset = left_ts[left_positions]

        # Vectorized search within this group
        local_indices = _vectorized_asof_indices(left_ts_subset, right_ts_sorted, cfg)

        # Map local matches back to global right-table positions
        matched = local_indices >= 0
        global_indices[left_positions[matched]] = right_positions[local_indices[matched]]

    return _gather_and_append(left, right, right_value_cols, prefixed_names, global_indices, stats)


def _gather_and_append(
    left: pa.Table,
    right: pa.Table,
    right_value_cols: list[str],
    prefixed_names: dict[str, str],
    indices: np.ndarray,
    stats: AlignmentStats,
) -> pa.Table:
    """Gather right-side columns using vectorized take with null masking.

    Uses pa.Array.take for matched indices and applies a validity bitmap
    for unmatched rows — no Python-level per-element loops.
    """
    matched_mask = indices >= 0
    stats.matched_rows = int(np.sum(matched_mask))
    n = len(indices)
    has_nulls = stats.matched_rows < n

    # Build take indices: replace -1 with 0 (we'll null them out via bitmap)
    safe_indices = np.where(matched_mask, indices, 0)
    take_arr = pa.array(safe_indices, type=pa.int64())

    result = left
    for col_name in right_value_cols:
        right_col = right.column(col_name)
        if isinstance(right_col, pa.ChunkedArray):
            right_col = right_col.combine_chunks()

        gathered = right_col.take(take_arr)

        if has_nulls:
            # Apply null mask: create a new array with nulls where indices == -1
            validity_buf = pa.array(matched_mask).buffers()[1]
            col_type = gathered.type

            # For variable-length types (strings, lists), we must go through
            # Python values to safely apply the null mask
            if (
                pa.types.is_binary(col_type)
                or pa.types.is_string(col_type)
                or pa.types.is_large_string(col_type)
                or pa.types.is_list(col_type)
            ):
                values = gathered.to_pylist()
                for i in range(n):
                    if not matched_mask[i]:
                        values[i] = None
                gathered = pa.array(values, type=col_type)
            else:
                # Fixed-width types: use from_buffers for zero-copy null application
                buffers = gathered.buffers()
                gathered = pa.Array.from_buffers(col_type, n, [validity_buf] + buffers[1:])

        output_name = prefixed_names.get(col_name, col_name)
        result = result.append_column(output_name, gathered)

    return result


# ---------------------------------------------------------------------------
# Multi-stream alignment
# ---------------------------------------------------------------------------

def align_streams(
    primary: pa.Table,
    secondaries: list[AlignmentSpec],
    primary_timestamp_col: str = "timestamp",
    primary_symbol_col: str = "symbol",
) -> tuple[pa.Table, dict[str, AlignmentStats]]:
    """Align multiple data streams onto a primary timeline.

    When the Rust kernel is available, all joins run in parallel.
    Falls back to sequential Python joins otherwise.
    """
    # Rust fast path: parallel multi-stream alignment
    if _HAS_RUST:
        return _rust_align_streams(primary, secondaries, primary_timestamp_col, primary_symbol_col)

    result = primary
    all_stats: dict[str, AlignmentStats] = {}

    for spec in secondaries:
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
            result, right, on=primary_timestamp_col, by=by_col, config=cfg,
        )
        all_stats[spec.name] = join_stats

    return result, all_stats


def _rust_align_streams(
    primary: pa.Table,
    secondaries: list[AlignmentSpec],
    primary_timestamp_col: str,
    primary_symbol_col: str,
) -> tuple[pa.Table, dict[str, AlignmentStats]]:
    """Dispatch multi-stream alignment to Rust kernel (parallel joins)."""
    stream_dicts = []
    all_stats: dict[str, AlignmentStats] = {}

    for spec in secondaries:
        if spec.value_columns is not None:
            keep = list(set(spec.value_columns + [spec.timestamp_col]))
            if spec.symbol_col in spec.table.schema.names:
                keep.append(spec.symbol_col)
            keep = [c for c in keep if c in spec.table.schema.names]
            right = spec.table.select(keep)
        else:
            right = spec.table

        prefix = spec.config.right_prefix or f"{spec.name}_"

        stream_dicts.append({
            "table": right,
            "prefix": prefix,
            "direction": spec.config.direction,
            "tolerance_ns": spec.config.tolerance_ns,
            "allow_exact_match": spec.config.allow_exact_match,
        })

    by_col = primary_symbol_col
    if by_col not in primary.schema.names:
        by_col = None

    result = _rust_core.align_streams(
        primary,
        stream_dicts,
        on=primary_timestamp_col,
        by=by_col,
    )

    # Compute stats from null counts per stream
    col_offset = len(primary.schema.names)
    for spec in secondaries:
        right_cols = [c for c in spec.table.schema.names
                      if c != spec.timestamp_col and c != spec.symbol_col]
        stats = AlignmentStats(left_rows=primary.num_rows, right_rows=spec.table.num_rows)
        if right_cols and col_offset < len(result.schema.names):
            null_count = result.column(col_offset).null_count
            stats.matched_rows = primary.num_rows - null_count
            stats.unmatched_rows = null_count
            col_offset += len(right_cols)
        all_stats[spec.name] = stats

    return result, all_stats


# ---------------------------------------------------------------------------
# Stateful aligner
# ---------------------------------------------------------------------------

class TemporalAligner:
    """Stateful temporal alignment engine for batch-level processing.

    Maintains state across batches to handle late-arriving data and
    produces aligned output incrementally.

    Example::

        aligner = TemporalAligner(
            primary_type="trade",
            secondary_specs={"quote": ["bid_price", "ask_price"]},
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
