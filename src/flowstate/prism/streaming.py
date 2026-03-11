"""Streaming temporal alignment with watermark-based emission.

Unlike batch alignment that requires all data upfront, the streaming
aligner accepts data incrementally and emits joined rows when the
watermark guarantees no future data can change the result.

Architecture:
  1. Left/right rows arrive via ``push_left()`` / ``push_right()``.
  2. Rows are buffered in sorted order internally.
  3. When the watermark advances past a left row's timestamp + lateness,
     that row is "sealed" — its as-of match is final.
  4. ``emit()`` returns all sealed rows as an Arrow Table.

This enables real-time feature construction on live market data without
waiting for the full dataset — critical for online ML inference.

When the Rust kernel (``flowstate_core.StreamingJoin``) is available,
all operations run through it. Otherwise, a pure-Python fallback is used.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from enum import StrEnum

import pyarrow as pa

logger = logging.getLogger(__name__)

try:
    import flowstate_core as _rust_core

    _HAS_RUST = True
except ImportError:
    _rust_core = None  # type: ignore[assignment]
    _HAS_RUST = False


class LatePolicy(StrEnum):
    """Policy for handling data that arrives after the watermark."""

    DROP = "drop"
    RECOMPUTE = "recompute"


@dataclass
class StreamingAlignStats:
    """Statistics for a streaming alignment session."""

    left_rows_pushed: int = 0
    right_rows_pushed: int = 0
    rows_emitted: int = 0
    batches_emitted: int = 0
    late_rows_dropped: int = 0
    late_rows_recomputed: int = 0
    watermark_ns: int = 0


@dataclass
class StreamingAlignConfig:
    """Configuration for the streaming aligner."""

    timestamp_col: str = "timestamp"
    group_col: str | None = None
    direction: str = "backward"
    tolerance_ns: int | None = None
    lateness_ns: int = 0
    late_policy: LatePolicy = LatePolicy.DROP
    allow_exact_match: bool = True


class StreamingAligner:
    """Incremental streaming as-of join with watermark semantics.

    Wraps the Rust ``StreamingJoin`` kernel when available, with a
    pure-Python fallback for environments without the native extension.

    Example::

        aligner = StreamingAligner(
            config=StreamingAlignConfig(
                timestamp_col="timestamp",
                group_col="symbol",
                tolerance_ns=5_000_000_000,
                lateness_ns=1_000_000_000,
            )
        )

        # Push incremental data
        aligner.push_left(trade_batch_1)
        aligner.push_right(quote_batch_1)

        # Advance watermark (e.g., to event time of latest data)
        aligner.advance_watermark(current_event_time_ns)

        # Emit sealed rows
        result = aligner.emit()
        if result is not None:
            process(result)

        # At end of stream, flush remaining
        final = aligner.flush()
    """

    def __init__(self, config: StreamingAlignConfig | None = None) -> None:
        self._config = config or StreamingAlignConfig()
        self._stats = StreamingAlignStats()

        if _HAS_RUST:
            self._rust_join = _rust_core.StreamingJoin(
                on=self._config.timestamp_col,
                by=self._config.group_col,
                direction=self._config.direction,
                tolerance_ns=self._config.tolerance_ns,
                allow_exact_match=self._config.allow_exact_match,
                lateness_ns=self._config.lateness_ns,
            )
            self._impl = "rust"
            logger.debug("StreamingAligner using Rust kernel")
        else:
            self._rust_join = None
            self._impl = "python"
            self._py_state = _PythonStreamingState(self._config)
            logger.debug("StreamingAligner using Python fallback")

    @property
    def config(self) -> StreamingAlignConfig:
        return self._config

    @property
    def stats(self) -> StreamingAlignStats:
        return self._stats

    @property
    def implementation(self) -> str:
        """Return 'rust' or 'python' indicating which backend is active."""
        return self._impl

    def push_left(self, table: pa.Table) -> None:
        """Push left-side (primary) data into the aligner."""
        if table.num_rows == 0:
            return

        self._stats.left_rows_pushed += table.num_rows

        if self._rust_join is not None:
            self._rust_join.push_left(table)
        else:
            late_count = self._py_state.push_left(table)
            self._stats.late_rows_dropped += late_count

    def push_right(self, table: pa.Table) -> None:
        """Push right-side (secondary) data into the aligner."""
        if table.num_rows == 0:
            return

        self._stats.right_rows_pushed += table.num_rows

        if self._rust_join is not None:
            self._rust_join.push_right(table)
        else:
            self._py_state.push_right(table)

    def advance_watermark(self, watermark_ns: int) -> None:
        """Advance the watermark to the given nanosecond timestamp.

        Rows with timestamp <= watermark - lateness_ns become eligible
        for emission. The watermark can only advance forward.
        """
        if watermark_ns <= self._stats.watermark_ns:
            return

        self._stats.watermark_ns = watermark_ns

        if self._rust_join is not None:
            self._rust_join.advance_watermark(watermark_ns)
        else:
            self._py_state.advance_watermark(watermark_ns)

    def emit(self) -> pa.Table | None:
        """Emit all sealed rows whose matches are finalized.

        Returns an Arrow Table of joined rows, or None if no rows
        are ready for emission.
        """
        if self._rust_join is not None:
            result = self._rust_join.emit()
            if result is not None:
                n = result.num_rows
                self._stats.rows_emitted += n
                self._stats.batches_emitted += 1
                return result
            return None
        else:
            result = self._py_state.emit()
            if result is not None and result.num_rows > 0:
                self._stats.rows_emitted += result.num_rows
                self._stats.batches_emitted += 1
                return result
            return None

    def flush(self) -> pa.Table | None:
        """Flush all remaining buffered rows, regardless of watermark.

        Call this at end-of-stream to emit any remaining data.
        """
        if self._rust_join is not None:
            try:
                result = self._rust_join.flush()
            except RuntimeError:
                # Rust kernel raises if no data was pushed yet
                return None
            if result is not None:
                self._stats.rows_emitted += result.num_rows
                self._stats.batches_emitted += 1
                return result
            return None
        else:
            result = self._py_state.flush()
            if result is not None and result.num_rows > 0:
                self._stats.rows_emitted += result.num_rows
                self._stats.batches_emitted += 1
                return result
            return None

    def reset(self) -> None:
        """Reset the aligner state for a new stream."""
        self._stats = StreamingAlignStats()
        if self._rust_join is not None:
            # Re-create the Rust join
            self._rust_join = _rust_core.StreamingJoin(
                on=self._config.timestamp_col,
                by=self._config.group_col,
                direction=self._config.direction,
                tolerance_ns=self._config.tolerance_ns,
                allow_exact_match=self._config.allow_exact_match,
                lateness_ns=self._config.lateness_ns,
            )
        else:
            self._py_state = _PythonStreamingState(self._config)


# ------------------------------------------------------------------
# Pure-Python fallback implementation
# ------------------------------------------------------------------


@dataclass
class _LeftRow:
    """A buffered left-side row awaiting match."""

    timestamp: int
    row_data: dict
    matched: bool = False
    match_data: dict = field(default_factory=dict)


class _PythonStreamingState:
    """Pure-Python streaming as-of join state.

    Maintains sorted buffers per group and emits rows once the
    watermark seals them.
    """

    def __init__(self, config: StreamingAlignConfig) -> None:
        self._config = config
        self._ts_col = config.timestamp_col
        self._group_col = config.group_col
        self._watermark: int = 0

        # Per-group state
        self._left_buffers: dict[str, list[_LeftRow]] = {}
        self._right_buffers: dict[str, list[tuple[int, dict]]] = {}
        self._right_timestamps: dict[str, list[int]] = {}

        # Track schemas
        self._left_schema: pa.Schema | None = None
        self._right_schema: pa.Schema | None = None

    def push_left(self, table: pa.Table) -> int:
        """Push left data, returns count of late rows dropped."""
        if self._left_schema is None:
            self._left_schema = table.schema

        late_count = 0
        seal_threshold = self._seal_threshold()

        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.schema.names}
            ts = row[self._ts_col]
            group = str(row.get(self._group_col, "")) if self._group_col else "__global__"

            is_late = ts < seal_threshold and self._watermark > 0
            if is_late and self._config.late_policy == LatePolicy.DROP:
                late_count += 1
                continue

            if group not in self._left_buffers:
                self._left_buffers[group] = []

            left_row = _LeftRow(timestamp=ts, row_data=row)
            buf = self._left_buffers[group]
            idx = bisect.bisect_right([r.timestamp for r in buf], ts)
            buf.insert(idx, left_row)

        return late_count

    def push_right(self, table: pa.Table) -> None:
        """Push right data."""
        if self._right_schema is None:
            self._right_schema = table.schema

        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.schema.names}
            ts = row[self._ts_col]
            group = str(row.get(self._group_col, "")) if self._group_col else "__global__"

            if group not in self._right_buffers:
                self._right_buffers[group] = []
                self._right_timestamps[group] = []

            buf = self._right_buffers[group]
            ts_buf = self._right_timestamps[group]
            idx = bisect.bisect_right(ts_buf, ts)
            buf.insert(idx, (ts, row))
            ts_buf.insert(idx, ts)

    def advance_watermark(self, watermark_ns: int) -> None:
        self._watermark = max(self._watermark, watermark_ns)

    def emit(self) -> pa.Table | None:
        """Emit rows sealed by the watermark."""
        if self._left_schema is None:
            return None

        seal_threshold = self._seal_threshold()
        return self._emit_sealed(seal_threshold)

    def flush(self) -> pa.Table | None:
        """Flush all remaining rows."""
        if self._left_schema is None:
            return None

        # Seal everything by using max possible threshold
        return self._emit_sealed(float("inf"))

    def _seal_threshold(self) -> float:
        if self._watermark == 0:
            return 0
        return self._watermark - self._config.lateness_ns

    def _emit_sealed(self, seal_threshold: float) -> pa.Table | None:
        """Emit all left rows with timestamp < seal_threshold."""
        emitted_rows: list[dict] = []

        for group, left_buf in self._left_buffers.items():
            right_buf = self._right_buffers.get(group, [])
            right_ts = self._right_timestamps.get(group, [])

            sealed_count = 0
            for left_row in left_buf:
                if left_row.timestamp >= seal_threshold:
                    break
                sealed_count += 1

                match = self._find_match(left_row.timestamp, right_ts, right_buf)
                combined = dict(left_row.row_data)
                if match is not None:
                    for k, v in match.items():
                        if k != self._ts_col and k != self._group_col:
                            combined[f"right_{k}"] = v
                else:
                    if self._right_schema is not None:
                        for col in self._right_schema.names:
                            if col != self._ts_col and col != self._group_col:
                                combined[f"right_{col}"] = None

                emitted_rows.append(combined)

            # Remove emitted rows
            if sealed_count > 0:
                del left_buf[:sealed_count]

        if not emitted_rows:
            return None

        return pa.Table.from_pylist(emitted_rows)

    def _find_match(
        self,
        left_ts: int,
        right_ts: list[int],
        right_buf: list[tuple[int, dict]],
    ) -> dict | None:
        """Find the best as-of match for a left timestamp."""
        if not right_buf:
            return None

        direction = self._config.direction
        tolerance = self._config.tolerance_ns

        if direction == "backward":
            idx = bisect.bisect_right(right_ts, left_ts) - 1
            if idx < 0:
                return None
            candidate_ts, candidate_data = right_buf[idx]
            if not self._config.allow_exact_match and candidate_ts == left_ts:
                idx -= 1
                if idx < 0:
                    return None
                candidate_ts, candidate_data = right_buf[idx]
            if tolerance is not None and left_ts - candidate_ts > tolerance:
                return None
            return candidate_data

        elif direction == "forward":
            idx = bisect.bisect_left(right_ts, left_ts)
            is_exact = idx < len(right_ts) and right_ts[idx] == left_ts
            if not self._config.allow_exact_match and is_exact:
                idx += 1
            if idx >= len(right_ts):
                return None
            candidate_ts, candidate_data = right_buf[idx]
            if tolerance is not None and candidate_ts - left_ts > tolerance:
                return None
            return candidate_data

        else:  # nearest
            idx_back = bisect.bisect_right(right_ts, left_ts) - 1
            idx_fwd = bisect.bisect_left(right_ts, left_ts)
            fwd_exact = idx_fwd < len(right_ts) and right_ts[idx_fwd] == left_ts
            if not self._config.allow_exact_match and fwd_exact:
                idx_fwd += 1
                if idx_back >= 0 and right_ts[idx_back] == left_ts:
                    idx_back -= 1

            candidates = []
            if idx_back >= 0:
                dist = left_ts - right_ts[idx_back]
                if tolerance is None or dist <= tolerance:
                    candidates.append((dist, idx_back))
            if idx_fwd < len(right_ts):
                dist = right_ts[idx_fwd] - left_ts
                if tolerance is None or dist <= tolerance:
                    candidates.append((dist, idx_fwd))

            if not candidates:
                return None
            candidates.sort()
            return right_buf[candidates[0][1]][1]
