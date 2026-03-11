//! Streaming incremental as-of join with watermark-based emission.
//!
//! Unlike batch joins that require all data upfront, the streaming aligner
//! accepts data incrementally and emits joined rows when the watermark
//! guarantees no future data can affect the result.
//!
//! Architecture:
//!   1. Left/right rows arrive in arbitrary order via `push_left`/`push_right`.
//!   2. Rows are buffered in sorted order (insertion via binary search).
//!   3. When the watermark advances past a left row's timestamp + tolerance,
//!      that row is "sealed" — its match is final and it can be emitted.
//!   4. `emit()` returns all sealed, unmatched rows as a RecordBatch.
//!
//! This enables real-time feature construction on live market data without
//! waiting for the full dataset — critical for online ML inference.

use ahash::AHashMap;
use arrow_array::{
    builder::{Float64Builder, Int64Builder, StringBuilder, UInt64Builder},
    Array, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use std::sync::Arc;

use super::config::AsOfDirection;

/// Per-group state for streaming as-of join.
struct GroupState {
    /// Right-side rows in sorted order.
    right_rows: Vec<(i64, usize, usize)>, // (timestamp, row_idx, batch_idx)
}

impl GroupState {
    fn new() -> Self {
        Self {
            right_rows: Vec::new(),
        }
    }

    /// Insert a right-side row maintaining sorted order.
    fn insert_right(&mut self, ts: i64, row_idx: usize, batch_idx: usize) {
        let pos = self.right_rows.partition_point(|r| r.0 <= ts);
        self.right_rows.insert(pos, (ts, row_idx, batch_idx));
    }

    /// Find the best match for a left timestamp using backward scan.
    fn match_backward(&self, left_ts: i64, tolerance_ns: Option<i64>, allow_exact: bool) -> Option<(usize, usize)> {
        if self.right_rows.is_empty() {
            return None;
        }
        // Find rightmost right_ts <= left_ts
        let pos = self.right_rows.partition_point(|r| r.0 <= left_ts);
        if pos == 0 {
            return None;
        }
        let candidate = &self.right_rows[pos - 1];
        if !allow_exact && candidate.0 == left_ts {
            if pos >= 2 {
                let prev = &self.right_rows[pos - 2];
                if let Some(tol) = tolerance_ns {
                    if left_ts - prev.0 <= tol {
                        return Some((prev.1, prev.2));
                    }
                } else {
                    return Some((prev.1, prev.2));
                }
            }
            return None;
        }
        if let Some(tol) = tolerance_ns {
            if left_ts - candidate.0 <= tol {
                return Some((candidate.1, candidate.2));
            }
            return None;
        }
        Some((candidate.1, candidate.2))
    }

    /// Find the best match using forward scan.
    fn match_forward(&self, left_ts: i64, tolerance_ns: Option<i64>, allow_exact: bool) -> Option<(usize, usize)> {
        if self.right_rows.is_empty() {
            return None;
        }
        let pos = if allow_exact {
            self.right_rows.partition_point(|r| r.0 < left_ts)
        } else {
            self.right_rows.partition_point(|r| r.0 <= left_ts)
        };
        if pos >= self.right_rows.len() {
            return None;
        }
        let candidate = &self.right_rows[pos];
        if let Some(tol) = tolerance_ns {
            if candidate.0 - left_ts <= tol {
                return Some((candidate.1, candidate.2));
            }
            return None;
        }
        Some((candidate.1, candidate.2))
    }

    /// Find the best match using nearest scan.
    fn match_nearest(&self, left_ts: i64, tolerance_ns: Option<i64>, allow_exact: bool) -> Option<(usize, usize)> {
        if self.right_rows.is_empty() {
            return None;
        }
        let pos = self.right_rows.partition_point(|r| r.0 < left_ts);

        // Backward candidate
        let back = if pos > 0 {
            let c = &self.right_rows[pos - 1];
            let dist = left_ts - c.0;
            if !allow_exact && dist == 0 {
                if pos >= 2 {
                    let p = &self.right_rows[pos - 2];
                    Some((p.1, p.2, left_ts - p.0))
                } else {
                    None
                }
            } else {
                Some((c.1, c.2, dist))
            }
        } else {
            None
        };

        // Forward candidate
        let fwd = if pos < self.right_rows.len() {
            let c = &self.right_rows[pos];
            let dist = c.0 - left_ts;
            if !allow_exact && dist == 0 {
                if pos + 1 < self.right_rows.len() {
                    let n = &self.right_rows[pos + 1];
                    Some((n.1, n.2, n.0 - left_ts))
                } else {
                    None
                }
            } else {
                Some((c.1, c.2, dist))
            }
        } else {
            None
        };

        let best = match (back, fwd) {
            (Some((bi, bb, bd)), Some((fi, fb, fd))) => {
                if bd <= fd { Some((bi, bb, bd)) } else { Some((fi, fb, fd)) }
            }
            (Some(b), None) => Some(b),
            (None, Some(f)) => Some(f),
            (None, None) => None,
        };

        if let Some((idx, batch, dist)) = best {
            if let Some(tol) = tolerance_ns {
                if dist <= tol { Some((idx, batch)) } else { None }
            } else {
                Some((idx, batch))
            }
        } else {
            None
        }
    }

    fn find_match(
        &self,
        left_ts: i64,
        direction: &AsOfDirection,
        tolerance_ns: Option<i64>,
        allow_exact: bool,
    ) -> Option<(usize, usize)> {
        match direction {
            AsOfDirection::Backward => self.match_backward(left_ts, tolerance_ns, allow_exact),
            AsOfDirection::Forward => self.match_forward(left_ts, tolerance_ns, allow_exact),
            AsOfDirection::Nearest => self.match_nearest(left_ts, tolerance_ns, allow_exact),
        }
    }
}

/// A pending left row waiting to be matched and emitted.
struct PendingLeft {
    timestamp: i64,
    row_idx: usize,
    batch_idx: usize,
    group: Option<String>,
    /// The current best match from the right side.
    matched_right: Option<(usize, usize)>, // (row_idx, batch_idx)
    /// Whether this row has been emitted.
    emitted: bool,
}

/// Streaming incremental as-of join engine.
///
/// Maintains state across `push_left`/`push_right` calls and emits
/// joined rows when the watermark guarantees match finality.
pub struct StreamingAsOfJoin {
    /// Configuration.
    direction: AsOfDirection,
    tolerance_ns: Option<i64>,
    allow_exact_match: bool,
    /// Watermark: all data with timestamp < watermark has arrived.
    watermark: i64,
    /// Lateness tolerance: how long to wait for out-of-order data.
    lateness_ns: i64,
    /// Per-group right-side state.
    groups: AHashMap<String, GroupState>,
    /// Global (ungrouped) right-side state.
    global_state: GroupState,
    /// Whether to use grouping.
    use_groups: bool,
    /// Pending left rows awaiting emission.
    pending: Vec<PendingLeft>,
    /// Right-side batches stored for value retrieval during emission.
    right_batches: Vec<RecordBatch>,
    /// Left-side batches stored for value retrieval during emission.
    left_batches: Vec<RecordBatch>,
    /// Schema of the timestamp column name.
    on_col: String,
    /// Group-by column name.
    by_col: Option<String>,
    /// Total rows emitted.
    pub total_emitted: usize,
    /// Total left rows received.
    pub total_left_received: usize,
    /// Total right rows received.
    pub total_right_received: usize,
}

impl StreamingAsOfJoin {
    /// Create a new streaming as-of join engine.
    pub fn new(
        direction: AsOfDirection,
        tolerance_ns: Option<i64>,
        allow_exact_match: bool,
        lateness_ns: i64,
        on_col: String,
        by_col: Option<String>,
    ) -> Self {
        let use_groups = by_col.is_some();
        Self {
            direction,
            tolerance_ns,
            allow_exact_match,
            watermark: i64::MIN,
            lateness_ns,
            groups: AHashMap::new(),
            global_state: GroupState::new(),
            use_groups,
            pending: Vec::new(),
            right_batches: Vec::new(),
            left_batches: Vec::new(),
            on_col,
            by_col,
            total_emitted: 0,
            total_left_received: 0,
            total_right_received: 0,
        }
    }

    /// Push a batch of right-side (secondary) data.
    pub fn push_right(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        let schema = batch.schema();
        let ts_idx = schema.index_of(&self.on_col)
            .map_err(|_| ArrowError::InvalidArgumentError(
                format!("Column '{}' not found in right batch", self.on_col)
            ))?;

        let ts_array = batch.column(ts_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| ArrowError::InvalidArgumentError(
                "Timestamp column must be Int64".into()
            ))?;

        let batch_idx = self.right_batches.len();
        self.right_batches.push(batch.clone());

        if self.use_groups {
            let by_col = self.by_col.as_ref().unwrap();
            let by_idx = schema.index_of(by_col)
                .map_err(|_| ArrowError::InvalidArgumentError(
                    format!("Column '{}' not found in right batch", by_col)
                ))?;
            let syms = batch.column(by_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| ArrowError::InvalidArgumentError(
                    "Group column must be Utf8".into()
                ))?;

            for i in 0..batch.num_rows() {
                let ts = ts_array.value(i);
                let sym = syms.value(i).to_string();
                let state = self.groups.entry(sym).or_insert_with(GroupState::new);
                state.insert_right(ts, i, batch_idx);
                self.total_right_received += 1;
            }
        } else {
            for i in 0..batch.num_rows() {
                let ts = ts_array.value(i);
                self.global_state.insert_right(ts, i, batch_idx);
                self.total_right_received += 1;
            }
        }

        // Re-evaluate pending left rows that might now have matches
        self.reevaluate_pending();

        Ok(())
    }

    /// Push a batch of left-side (primary) data.
    pub fn push_left(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        let schema = batch.schema();
        let ts_idx = schema.index_of(&self.on_col)
            .map_err(|_| ArrowError::InvalidArgumentError(
                format!("Column '{}' not found in left batch", self.on_col)
            ))?;

        let ts_array = batch.column(ts_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| ArrowError::InvalidArgumentError(
                "Timestamp column must be Int64".into()
            ))?;

        let batch_idx = self.left_batches.len();
        self.left_batches.push(batch.clone());

        if self.use_groups {
            let by_col = self.by_col.as_ref().unwrap();
            let by_idx = schema.index_of(by_col)
                .map_err(|_| ArrowError::InvalidArgumentError(
                    format!("Column '{}' not found in left batch", by_col)
                ))?;
            let syms = batch.column(by_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| ArrowError::InvalidArgumentError(
                    "Group column must be Utf8".into()
                ))?;

            for i in 0..batch.num_rows() {
                let ts = ts_array.value(i);
                let sym = syms.value(i).to_string();

                // Find current best match
                let matched = self.groups
                    .get(&sym)
                    .and_then(|state| state.find_match(
                        ts, &self.direction, self.tolerance_ns, self.allow_exact_match
                    ));

                self.pending.push(PendingLeft {
                    timestamp: ts,
                    row_idx: i,
                    batch_idx,
                    group: Some(sym),
                    matched_right: matched,
                    emitted: false,
                });
                self.total_left_received += 1;
            }
        } else {
            for i in 0..batch.num_rows() {
                let ts = ts_array.value(i);
                let matched = self.global_state.find_match(
                    ts, &self.direction, self.tolerance_ns, self.allow_exact_match
                );

                self.pending.push(PendingLeft {
                    timestamp: ts,
                    row_idx: i,
                    batch_idx,
                    group: None,
                    matched_right: matched,
                    emitted: false,
                });
                self.total_left_received += 1;
            }
        }

        Ok(())
    }

    /// Re-evaluate pending left rows after new right data arrives.
    fn reevaluate_pending(&mut self) {
        for pending in &mut self.pending {
            if pending.emitted {
                continue;
            }
            let new_match = if self.use_groups {
                if let Some(ref group) = pending.group {
                    self.groups.get(group).and_then(|state| {
                        state.find_match(
                            pending.timestamp,
                            &self.direction,
                            self.tolerance_ns,
                            self.allow_exact_match,
                        )
                    })
                } else {
                    None
                }
            } else {
                self.global_state.find_match(
                    pending.timestamp,
                    &self.direction,
                    self.tolerance_ns,
                    self.allow_exact_match,
                )
            };
            if new_match.is_some() {
                pending.matched_right = new_match;
            }
        }
    }

    /// Advance the watermark. Rows with timestamp <= watermark - lateness are sealed.
    pub fn advance_watermark(&mut self, new_watermark: i64) {
        self.watermark = self.watermark.max(new_watermark);
    }

    /// Emit all sealed rows as a RecordBatch.
    ///
    /// A row is sealed when: watermark - lateness >= timestamp (for backward/nearest)
    /// or watermark - lateness >= timestamp + tolerance (for forward).
    ///
    /// Returns None if no rows are ready for emission.
    pub fn emit(
        &mut self,
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        let seal_threshold = match self.direction {
            AsOfDirection::Forward => {
                // For forward joins, we need to wait until no future right data
                // could match — i.e., watermark past the left timestamp + tolerance
                self.watermark - self.lateness_ns - self.tolerance_ns.unwrap_or(0)
            }
            _ => {
                // For backward/nearest, sealed once watermark guarantees
                // no new right data <= left timestamp can arrive
                self.watermark - self.lateness_ns
            }
        };

        // Find rows ready for emission
        let mut emit_indices: Vec<usize> = Vec::new();
        for (i, pending) in self.pending.iter().enumerate() {
            if !pending.emitted && pending.timestamp <= seal_threshold {
                emit_indices.push(i);
            }
        }

        if emit_indices.is_empty() {
            return Ok(None);
        }

        // Build output RecordBatch from emitted rows
        let n_emit = emit_indices.len();

        // Determine right value columns (exclude on and by)
        let right_value_cols: Vec<(usize, &Field)> = right_schema.fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                let name = f.name();
                name != &self.on_col && self.by_col.as_ref().map_or(true, |b| name != b)
            })
            .map(|(i, f)| (i, f.as_ref()))
            .collect();

        // Build left-side columns
        let mut out_fields: Vec<Arc<Field>> = Vec::new();
        let mut out_columns: Vec<Arc<dyn Array>> = Vec::new();

        // For each left column, gather values from stored batches
        for (field_idx, field) in left_schema.fields().iter().enumerate() {
            out_fields.push(field.clone());

            match field.data_type() {
                DataType::Int64 => {
                    let mut builder = Int64Builder::with_capacity(n_emit);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        let batch = &self.left_batches[p.batch_idx];
                        let arr = batch.column(field_idx).as_any()
                            .downcast_ref::<Int64Array>().unwrap();
                        builder.append_value(arr.value(p.row_idx));
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                DataType::Float64 => {
                    let mut builder = Float64Builder::with_capacity(n_emit);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        let batch = &self.left_batches[p.batch_idx];
                        let arr = batch.column(field_idx).as_any()
                            .downcast_ref::<Float64Array>().unwrap();
                        builder.append_value(arr.value(p.row_idx));
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                DataType::Utf8 => {
                    let mut builder = StringBuilder::with_capacity(n_emit, n_emit * 8);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        let batch = &self.left_batches[p.batch_idx];
                        let arr = batch.column(field_idx).as_any()
                            .downcast_ref::<StringArray>().unwrap();
                        builder.append_value(arr.value(p.row_idx));
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                _ => {
                    // Generic path using UInt64 take indices
                    // For unsupported types, emit nulls
                    let mut builder = Int64Builder::with_capacity(n_emit);
                    for _ in &emit_indices {
                        builder.append_null();
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
            }
        }

        // Build right-side value columns
        for &(col_idx, field) in &right_value_cols {
            let nullable_field = Arc::new(Field::new(field.name(), field.data_type().clone(), true));
            out_fields.push(nullable_field);

            match field.data_type() {
                DataType::Int64 => {
                    let mut builder = Int64Builder::with_capacity(n_emit);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        if let Some((row_idx, batch_idx)) = p.matched_right {
                            let batch = &self.right_batches[batch_idx];
                            let arr = batch.column(col_idx).as_any()
                                .downcast_ref::<Int64Array>().unwrap();
                            builder.append_value(arr.value(row_idx));
                        } else {
                            builder.append_null();
                        }
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                DataType::Float64 => {
                    let mut builder = Float64Builder::with_capacity(n_emit);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        if let Some((row_idx, batch_idx)) = p.matched_right {
                            let batch = &self.right_batches[batch_idx];
                            let arr = batch.column(col_idx).as_any()
                                .downcast_ref::<Float64Array>().unwrap();
                            builder.append_value(arr.value(row_idx));
                        } else {
                            builder.append_null();
                        }
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                DataType::Utf8 => {
                    let mut builder = StringBuilder::with_capacity(n_emit, n_emit * 8);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        if let Some((row_idx, batch_idx)) = p.matched_right {
                            let batch = &self.right_batches[batch_idx];
                            let arr = batch.column(col_idx).as_any()
                                .downcast_ref::<StringArray>().unwrap();
                            builder.append_value(arr.value(row_idx));
                        } else {
                            builder.append_null();
                        }
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                DataType::UInt64 => {
                    let mut builder = UInt64Builder::with_capacity(n_emit);
                    for &ei in &emit_indices {
                        let p = &self.pending[ei];
                        if let Some((row_idx, batch_idx)) = p.matched_right {
                            let batch = &self.right_batches[batch_idx];
                            let arr = batch.column(col_idx).as_any()
                                .downcast_ref::<arrow_array::UInt64Array>().unwrap();
                            builder.append_value(arr.value(row_idx));
                        } else {
                            builder.append_null();
                        }
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
                _ => {
                    let mut builder = Float64Builder::with_capacity(n_emit);
                    for _ in &emit_indices {
                        builder.append_null();
                    }
                    out_columns.push(Arc::new(builder.finish()));
                }
            }
        }

        // Mark rows as emitted
        for &ei in &emit_indices {
            self.pending[ei].emitted = true;
        }
        self.total_emitted += n_emit;

        // Garbage-collect emitted rows from the front
        while self.pending.first().map_or(false, |p| p.emitted) {
            self.pending.remove(0);
        }

        let schema = Arc::new(Schema::new(out_fields));
        let batch = RecordBatch::try_new(schema, out_columns)?;
        Ok(Some(batch))
    }

    /// Flush all remaining pending rows regardless of watermark.
    pub fn flush(
        &mut self,
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        // Temporarily override lateness so seal_threshold = i64::MAX
        let saved_lateness = self.lateness_ns;
        self.lateness_ns = 0;
        self.watermark = i64::MAX;
        let result = self.emit(left_schema, right_schema);
        self.lateness_ns = saved_lateness;
        result
    }

    /// Number of pending (unsealed) left rows.
    pub fn pending_count(&self) -> usize {
        self.pending.iter().filter(|p| !p.emitted).count()
    }

    /// Current watermark value.
    pub fn watermark(&self) -> i64 {
        self.watermark
    }

    /// Prune right-side data older than the given timestamp.
    /// Frees memory for data that can no longer match any future left rows.
    pub fn prune_right_before(&mut self, timestamp: i64) {
        if self.use_groups {
            for state in self.groups.values_mut() {
                state.right_rows.retain(|r| r.0 >= timestamp);
            }
        } else {
            self.global_state.right_rows.retain(|r| r.0 >= timestamp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_right_batch(timestamps: &[i64], values: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("bid", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        ).unwrap()
    }

    fn make_left_batch(timestamps: &[i64], prices: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("price", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(Float64Array::from(prices.to_vec())),
            ],
        ).unwrap()
    }

    fn make_grouped_right(timestamps: &[i64], syms: &[&str], values: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("symbol", DataType::Utf8, false),
                Field::new("bid", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(StringArray::from(syms.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        ).unwrap()
    }

    fn make_grouped_left(timestamps: &[i64], syms: &[&str], prices: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("symbol", DataType::Utf8, false),
                Field::new("price", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(StringArray::from(syms.to_vec())),
                Arc::new(Float64Array::from(prices.to_vec())),
            ],
        ).unwrap()
    }

    #[test]
    fn test_basic_streaming_backward() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, None, true, 0,
            "timestamp".into(), None,
        );

        let right = make_right_batch(&[5, 15, 25], &[0.5, 1.5, 2.5]);
        aligner.push_right(&right).unwrap();

        let left = make_left_batch(&[10, 20, 30], &[1.0, 2.0, 3.0]);
        aligner.push_left(&left).unwrap();

        // Advance watermark past all data
        aligner.advance_watermark(100);

        let left_schema = left.schema();
        let right_schema = right.schema();
        let result = aligner.emit(&left_schema, &right_schema).unwrap().unwrap();

        assert_eq!(result.num_rows(), 3);
        let bids = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(bids.value(0), 0.5);  // ts=10, match ts=5
        assert_eq!(bids.value(1), 1.5);  // ts=20, match ts=15
        assert_eq!(bids.value(2), 2.5);  // ts=30, match ts=25
    }

    #[test]
    fn test_incremental_right_arrival() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, None, true, 0,
            "timestamp".into(), None,
        );

        // Left arrives first
        let left = make_left_batch(&[10, 20], &[1.0, 2.0]);
        aligner.push_left(&left).unwrap();

        // No right data yet — check pending
        assert_eq!(aligner.pending_count(), 2);

        // Right data arrives
        let right = make_right_batch(&[5, 15], &[0.5, 1.5]);
        aligner.push_right(&right).unwrap();

        aligner.advance_watermark(100);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap().unwrap();

        assert_eq!(result.num_rows(), 2);
        let bids = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(bids.value(0), 0.5);
        assert_eq!(bids.value(1), 1.5);
    }

    #[test]
    fn test_watermark_controls_emission() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, None, true, 5, // 5ns lateness
            "timestamp".into(), None,
        );

        let right = make_right_batch(&[5], &[0.5]);
        aligner.push_right(&right).unwrap();

        let left = make_left_batch(&[10, 20], &[1.0, 2.0]);
        aligner.push_left(&left).unwrap();

        // Watermark at 16: seal threshold = 16 - 5 = 11. Only ts=10 is sealed.
        aligner.advance_watermark(16);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap().unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(aligner.pending_count(), 1);

        // Watermark at 30: ts=20 now sealed
        aligner.advance_watermark(30);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap().unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(aligner.pending_count(), 0);
    }

    #[test]
    fn test_grouped_streaming() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, None, true, 0,
            "timestamp".into(), Some("symbol".into()),
        );

        let right = make_grouped_right(
            &[5, 5, 15, 15],
            &["A", "B", "A", "B"],
            &[0.5, 0.6, 1.5, 1.6],
        );
        aligner.push_right(&right).unwrap();

        let left = make_grouped_left(
            &[10, 10],
            &["A", "B"],
            &[1.0, 2.0],
        );
        aligner.push_left(&left).unwrap();

        aligner.advance_watermark(100);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap().unwrap();

        assert_eq!(result.num_rows(), 2);
        let bids = result.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        // A matches A's quote at ts=5 (bid=0.5)
        assert_eq!(bids.value(0), 0.5);
        // B matches B's quote at ts=5 (bid=0.6)
        assert_eq!(bids.value(1), 0.6);
    }

    #[test]
    fn test_flush_emits_all() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, None, true, 1_000_000, // huge lateness
            "timestamp".into(), None,
        );

        let right = make_right_batch(&[5], &[0.5]);
        aligner.push_right(&right).unwrap();

        let left = make_left_batch(&[10], &[1.0]);
        aligner.push_left(&left).unwrap();

        // Normal emit returns nothing (watermark too low)
        aligner.advance_watermark(20);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap();
        assert!(result.is_none());

        // Flush forces emission
        let result = aligner.flush(&left.schema(), &right.schema()).unwrap().unwrap();
        assert_eq!(result.num_rows(), 1);
    }

    #[test]
    fn test_tolerance_in_streaming() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, Some(10), true, 0,
            "timestamp".into(), None,
        );

        let right = make_right_batch(&[5], &[0.5]);
        aligner.push_right(&right).unwrap();

        let left = make_left_batch(&[10, 100], &[1.0, 2.0]);
        aligner.push_left(&left).unwrap();

        aligner.advance_watermark(200);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap().unwrap();

        assert_eq!(result.num_rows(), 2);
        let bids = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(bids.value(0), 0.5);  // ts=10, dist=5 <= 10
        assert!(bids.is_null(1));         // ts=100, dist=95 > 10
    }

    #[test]
    fn test_forward_streaming() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Forward, None, true, 0,
            "timestamp".into(), None,
        );

        let right = make_right_batch(&[15, 25], &[1.5, 2.5]);
        aligner.push_right(&right).unwrap();

        let left = make_left_batch(&[10, 20, 30], &[1.0, 2.0, 3.0]);
        aligner.push_left(&left).unwrap();

        aligner.advance_watermark(200);
        let result = aligner.emit(&left.schema(), &right.schema()).unwrap().unwrap();

        assert_eq!(result.num_rows(), 3);
        let bids = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(bids.value(0), 1.5);   // ts=10 → forward to ts=15
        assert_eq!(bids.value(1), 2.5);   // ts=20 → forward to ts=25
        assert!(bids.is_null(2));           // ts=30 → no future match
    }

    #[test]
    fn test_prune_frees_memory() {
        let mut aligner = StreamingAsOfJoin::new(
            AsOfDirection::Backward, None, true, 0,
            "timestamp".into(), None,
        );

        let right = make_right_batch(&[5, 15, 25, 35], &[0.5, 1.5, 2.5, 3.5]);
        aligner.push_right(&right).unwrap();
        assert_eq!(aligner.global_state.right_rows.len(), 4);

        aligner.prune_right_before(20);
        assert_eq!(aligner.global_state.right_rows.len(), 2); // only 25, 35 remain
    }
}
