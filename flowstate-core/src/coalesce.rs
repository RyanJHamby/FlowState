//! Adaptive RecordBatch coalescer for streaming pipelines.
//!
//! Streaming joins emit many small batches (often 1-10 rows). Passing these
//! directly to downstream consumers wastes CPU on per-batch overhead and
//! destroys vectorised processing throughput. The coalescer accumulates
//! small batches and flushes when a configurable row target is reached.
//!
//! ```text
//!  Streaming Join             Coalescer              Consumer
//! ┌──────────────┐  1-row   ┌──────────────┐  8192  ┌──────────────┐
//! │ emit(batch)  │────────→│ accumulate() │───────→│ vectorised   │
//! │              │  batches │ flush()      │  rows  │ feature eng  │
//! └──────────────┘          └──────────────┘        └──────────────┘
//! ```
//!
//! # Design
//!
//! - **Schema validation**: All pushed batches must share the same schema.
//! - **Zero-copy accumulation**: Batches are stored as-is until flush, then
//!   concatenated via `arrow::compute::concat_batches` (one allocation).
//! - **Adaptive flush**: Flushes when accumulated rows >= target, or on
//!   explicit flush/timeout.
//! - **Overflow handling**: If a single push exceeds the target, it triggers
//!   an immediate flush (never buffers more than target + one batch).

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use arrow_select::concat::concat_batches;

/// Configuration for the batch coalescer.
#[derive(Debug, Clone)]
pub struct CoalesceConfig {
    /// Target number of rows per output batch.
    pub target_rows: usize,
    /// Maximum number of batches to buffer before forcing a flush,
    /// regardless of row count. Prevents unbounded memory growth from
    /// many zero-row batches.
    pub max_buffered_batches: usize,
}

impl Default for CoalesceConfig {
    fn default() -> Self {
        Self {
            target_rows: 8192,
            max_buffered_batches: 256,
        }
    }
}

/// Accumulates small RecordBatches and flushes them as larger coalesced batches.
pub struct BatchCoalescer {
    config: CoalesceConfig,
    schema: Option<SchemaRef>,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    /// Total rows that have passed through the coalescer.
    pub total_rows_in: u64,
    /// Total coalesced batches emitted.
    pub total_batches_out: u64,
}

impl BatchCoalescer {
    /// Create a new coalescer with the given configuration.
    pub fn new(config: CoalesceConfig) -> Self {
        Self {
            config,
            schema: None,
            buffer: Vec::with_capacity(64),
            buffered_rows: 0,
            total_rows_in: 0,
            total_batches_out: 0,
        }
    }

    /// Create a coalescer with default configuration.
    pub fn with_target_rows(target_rows: usize) -> Self {
        Self::new(CoalesceConfig {
            target_rows,
            ..Default::default()
        })
    }

    /// Push a batch into the coalescer. Returns a coalesced batch if the
    /// target row count has been reached.
    pub fn push(&mut self, batch: RecordBatch) -> Result<Option<RecordBatch>, arrow_schema::ArrowError> {
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        // Capture or validate schema
        match &self.schema {
            None => {
                self.schema = Some(batch.schema());
            }
            Some(existing) => {
                if existing != &batch.schema() {
                    return Err(arrow_schema::ArrowError::SchemaError(format!(
                        "Schema mismatch in coalescer: expected {:?}, got {:?}",
                        existing, batch.schema()
                    )));
                }
            }
        }

        let incoming_rows = batch.num_rows();
        self.total_rows_in += incoming_rows as u64;
        self.buffered_rows += incoming_rows;
        self.buffer.push(batch);

        if self.should_flush() {
            self.flush()
        } else {
            Ok(None)
        }
    }

    /// Force flush all buffered batches into a single coalesced batch.
    /// Returns `None` if the buffer is empty.
    pub fn flush(&mut self) -> Result<Option<RecordBatch>, arrow_schema::ArrowError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let schema = self.schema.as_ref().expect("schema set when buffer non-empty").clone();
        let batches = std::mem::take(&mut self.buffer);
        self.buffered_rows = 0;

        let coalesced = concat_batches(&schema, &batches)?;
        self.total_batches_out += 1;

        Ok(Some(coalesced))
    }

    /// Number of rows currently buffered.
    #[inline]
    pub fn buffered_rows(&self) -> usize {
        self.buffered_rows
    }

    /// Number of batches currently buffered.
    #[inline]
    pub fn buffered_batches(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the coalescer has any buffered data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The schema of batches being coalesced (set after first push).
    pub fn schema(&self) -> Option<&SchemaRef> {
        self.schema.as_ref()
    }

    /// Reset the coalescer, clearing all buffered data and schema.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.buffered_rows = 0;
        self.schema = None;
    }

    /// Check if we should flush based on config thresholds.
    #[inline]
    fn should_flush(&self) -> bool {
        self.buffered_rows >= self.config.target_rows
            || self.buffer.len() >= self.config.max_buffered_batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let ts: Vec<i64> = (0..n as i64).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ts)),
                Arc::new(Float64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    fn make_batch_schema2(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int64, false),
            Field::new("other", DataType::Int64, false),
        ]));
        let ts: Vec<i64> = (0..n as i64).collect();
        let vals: Vec<i64> = (0..n as i64).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ts)),
                Arc::new(Int64Array::from(vals)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn empty_coalescer() {
        let mut c = BatchCoalescer::with_target_rows(100);
        assert!(c.is_empty());
        assert_eq!(c.buffered_rows(), 0);
        assert_eq!(c.flush().unwrap(), None);
    }

    #[test]
    fn single_batch_below_target() {
        let mut c = BatchCoalescer::with_target_rows(100);
        let result = c.push(make_batch(10)).unwrap();
        // Below target, no flush
        assert!(result.is_none());
        assert_eq!(c.buffered_rows(), 10);
        assert_eq!(c.buffered_batches(), 1);
    }

    #[test]
    fn flush_at_target() {
        let mut c = BatchCoalescer::with_target_rows(100);
        // Push 10 batches of 10 rows each = 100 rows = target
        for _ in 0..9 {
            assert!(c.push(make_batch(10)).unwrap().is_none());
        }
        // 10th push should trigger flush
        let result = c.push(make_batch(10)).unwrap();
        assert!(result.is_some());
        let batch = result.unwrap();
        assert_eq!(batch.num_rows(), 100);
        assert!(c.is_empty());
    }

    #[test]
    fn flush_over_target() {
        let mut c = BatchCoalescer::with_target_rows(50);
        // Push 90 rows, then 20 more = 110 > 50
        c.push(make_batch(45)).unwrap();
        let result = c.push(make_batch(20)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().num_rows(), 65);
    }

    #[test]
    fn single_large_batch_flushes_immediately() {
        let mut c = BatchCoalescer::with_target_rows(50);
        let result = c.push(make_batch(200)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().num_rows(), 200);
        assert!(c.is_empty());
    }

    #[test]
    fn explicit_flush_partial() {
        let mut c = BatchCoalescer::with_target_rows(1000);
        c.push(make_batch(10)).unwrap();
        c.push(make_batch(20)).unwrap();
        assert_eq!(c.buffered_rows(), 30);

        let result = c.flush().unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().num_rows(), 30);
        assert!(c.is_empty());
    }

    #[test]
    fn empty_batch_ignored() {
        let mut c = BatchCoalescer::with_target_rows(100);
        let result = c.push(make_batch(0)).unwrap();
        assert!(result.is_none());
        assert!(c.is_empty());
        assert_eq!(c.total_rows_in, 0);
    }

    #[test]
    fn schema_mismatch_rejected() {
        let mut c = BatchCoalescer::with_target_rows(100);
        c.push(make_batch(10)).unwrap();
        let result = c.push(make_batch_schema2(10));
        assert!(result.is_err());
    }

    #[test]
    fn metrics_tracked() {
        let mut c = BatchCoalescer::with_target_rows(20);
        c.push(make_batch(10)).unwrap();
        c.push(make_batch(15)).unwrap(); // triggers flush at 25 rows
        assert_eq!(c.total_rows_in, 25);
        assert_eq!(c.total_batches_out, 1);

        c.push(make_batch(5)).unwrap();
        c.flush().unwrap();
        assert_eq!(c.total_rows_in, 30);
        assert_eq!(c.total_batches_out, 2);
    }

    #[test]
    fn max_buffered_batches_triggers_flush() {
        let mut c = BatchCoalescer::new(CoalesceConfig {
            target_rows: 1_000_000, // Very high, won't trigger on rows
            max_buffered_batches: 4,
        });
        for _ in 0..3 {
            assert!(c.push(make_batch(1)).unwrap().is_none());
        }
        // 4th push should trigger flush via max_buffered_batches
        let result = c.push(make_batch(1)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().num_rows(), 4);
    }

    #[test]
    fn reset_clears_state() {
        let mut c = BatchCoalescer::with_target_rows(100);
        c.push(make_batch(50)).unwrap();
        assert!(!c.is_empty());

        c.reset();
        assert!(c.is_empty());
        assert_eq!(c.buffered_rows(), 0);
        assert!(c.schema().is_none());
    }

    #[test]
    fn coalesced_data_integrity() {
        let mut c = BatchCoalescer::with_target_rows(1000);
        // Push 3 batches with known data
        for offset in [0, 100, 200] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("ts", DataType::Int64, false),
                Field::new("value", DataType::Float64, false),
            ]));
            let ts: Vec<i64> = (offset..offset + 50).collect();
            let vals: Vec<f64> = ts.iter().map(|&t| t as f64 * 0.5).collect();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(ts)),
                    Arc::new(Float64Array::from(vals)),
                ],
            )
            .unwrap();
            c.push(batch).unwrap();
        }

        let result = c.flush().unwrap().unwrap();
        assert_eq!(result.num_rows(), 150);

        // Verify data ordering preserved
        let ts_col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // First batch: 0..50, second: 100..150, third: 200..250
        assert_eq!(ts_col.value(0), 0);
        assert_eq!(ts_col.value(49), 49);
        assert_eq!(ts_col.value(50), 100);
        assert_eq!(ts_col.value(99), 149);
        assert_eq!(ts_col.value(100), 200);
        assert_eq!(ts_col.value(149), 249);
    }
}
