//! Streaming pipeline: SPSC ring → streaming join → output ring.
//!
//! Wires the SPSC ring buffer and streaming join engine into a complete
//! data pipeline for real-time temporal alignment. The producer thread
//! pushes raw Arrow batches into an input ring, the pipeline thread
//! runs the streaming join with watermark advancement, and emitted
//! results are placed into an output ring for the consumer.
//!
//! ```text
//! Producer Thread          Pipeline Thread           Consumer Thread
//! ┌──────────────┐   SPSC  ┌──────────────┐   SPSC  ┌──────────────┐
//! │ Market Data  │───────→│ Streaming    │───────→│ Feature      │
//! │ Ingestion    │  ring   │ As-Of Join   │  ring   │ Pipeline     │
//! └──────────────┘         └──────────────┘         └──────────────┘
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;

use crate::asof::config::AsOfDirection;
use crate::asof::streaming::StreamingAsOfJoin;
use crate::coalesce::BatchCoalescer;
use crate::hdr::HdrHistogram;
use crate::spsc::SpscRing;

/// A message that flows through the pipeline.
#[derive(Debug)]
pub enum PipelineMessage {
    /// A left-side batch (e.g., trades).
    Left(RecordBatch),
    /// A right-side batch (e.g., quotes).
    Right(RecordBatch),
    /// Advance the watermark to this timestamp.
    Watermark(i64),
    /// Flush all pending state and emit.
    Flush,
}

/// Configuration for the streaming pipeline.
pub struct PipelineConfig {
    /// Ring buffer capacity for input messages.
    pub input_ring_capacity: usize,
    /// Ring buffer capacity for output batches.
    pub output_ring_capacity: usize,
    /// Join direction.
    pub direction: AsOfDirection,
    /// Tolerance in nanoseconds.
    pub tolerance_ns: Option<i64>,
    /// Allow exact timestamp matches.
    pub allow_exact_match: bool,
    /// Lateness tolerance for watermark-based emission.
    pub lateness_ns: i64,
    /// Timestamp column name.
    pub on: String,
    /// Optional group-by column name.
    pub by: Option<String>,
    /// Target rows per coalesced output batch (0 = no coalescing).
    pub coalesce_rows: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input_ring_capacity: 4096,
            output_ring_capacity: 1024,
            direction: AsOfDirection::Backward,
            tolerance_ns: None,
            allow_exact_match: true,
            lateness_ns: 0,
            on: "timestamp".to_string(),
            by: None,
            coalesce_rows: 0,
        }
    }
}

/// Latency metrics for the pipeline.
pub struct PipelineMetrics {
    /// End-to-end latency from push to emit (nanoseconds).
    pub join_latency: HdrHistogram,
    /// Number of batches processed.
    pub batches_processed: std::sync::atomic::AtomicU64,
    /// Number of rows emitted.
    pub rows_emitted: std::sync::atomic::AtomicU64,
}

impl PipelineMetrics {
    pub fn new() -> Self {
        Self {
            join_latency: HdrHistogram::new(),
            batches_processed: std::sync::atomic::AtomicU64::new(0),
            rows_emitted: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A streaming pipeline handle.
///
/// Owns the input/output rings and spawns the pipeline thread.
/// Drop the handle to signal shutdown.
pub struct StreamingPipeline {
    input: Arc<SpscRing<PipelineMessage>>,
    output: Arc<SpscRing<RecordBatch>>,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<PipelineMetrics>,
    handle: Option<thread::JoinHandle<()>>,
}

impl StreamingPipeline {
    /// Create and start a streaming pipeline.
    pub fn start(config: PipelineConfig) -> Self {
        let input = Arc::new(SpscRing::new(config.input_ring_capacity));
        let output = Arc::new(SpscRing::new(config.output_ring_capacity));
        let shutdown = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(PipelineMetrics::new());

        let pipeline_input = input.clone();
        let pipeline_output = output.clone();
        let pipeline_shutdown = shutdown.clone();
        let pipeline_metrics = metrics.clone();

        let handle = thread::spawn(move || {
            pipeline_loop(
                pipeline_input,
                pipeline_output,
                pipeline_shutdown,
                pipeline_metrics,
                config,
            );
        });

        Self {
            input,
            output,
            shutdown,
            metrics,
            handle: Some(handle),
        }
    }

    /// Push a message into the pipeline. Returns Err if the input ring is full.
    #[inline]
    pub fn try_send(&self, msg: PipelineMessage) -> Result<(), PipelineMessage> {
        self.input.try_push(msg)
    }

    /// Push a message, spinning until space is available.
    #[inline]
    pub fn send(&self, msg: PipelineMessage) {
        self.input.push_spin(msg);
    }

    /// Try to receive an aligned output batch.
    #[inline]
    pub fn try_recv(&self) -> Option<RecordBatch> {
        self.output.try_pop()
    }

    /// Drain all available output batches.
    pub fn drain_output(&self) -> Vec<RecordBatch> {
        self.output.drain()
    }

    /// Access pipeline metrics.
    pub fn metrics(&self) -> &Arc<PipelineMetrics> {
        &self.metrics
    }

    /// Signal the pipeline to shut down and wait for completion.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
    }

    /// Check if the pipeline is still running.
    pub fn is_running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }
}

impl Drop for StreamingPipeline {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
    }
}

/// Main pipeline loop running on the pipeline thread.
fn pipeline_loop(
    input: Arc<SpscRing<PipelineMessage>>,
    output: Arc<SpscRing<RecordBatch>>,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<PipelineMetrics>,
    config: PipelineConfig,
) {
    let mut join = StreamingAsOfJoin::new(
        config.direction,
        config.tolerance_ns,
        config.allow_exact_match,
        config.lateness_ns,
        config.on.clone(),
        config.by.clone(),
    );

    let mut coalescer = if config.coalesce_rows > 0 {
        Some(BatchCoalescer::with_target_rows(config.coalesce_rows))
    } else {
        None
    };

    let mut left_schema: Option<SchemaRef> = None;
    let mut right_schema: Option<SchemaRef> = None;

    loop {
        if shutdown.load(Ordering::Acquire) {
            // Drain remaining input before exiting
            while let Some(msg) = input.try_pop() {
                process_message(
                    msg,
                    &mut join,
                    &mut left_schema,
                    &mut right_schema,
                    &output,
                    &metrics,
                    &mut coalescer,
                );
            }
            // Final flush from join
            if let (Some(ls), Some(rs)) = (&left_schema, &right_schema) {
                if let Ok(Some(batch)) = join.flush(ls, rs) {
                    emit_batch(batch, &output, &metrics, &mut coalescer);
                }
            }
            // Flush remaining coalesced data
            if let Some(ref mut coal) = coalescer {
                if let Ok(Some(batch)) = coal.flush() {
                    metrics
                        .rows_emitted
                        .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
                    output.push_spin(batch);
                }
            }
            break;
        }

        match input.try_pop() {
            Some(msg) => {
                process_message(
                    msg,
                    &mut join,
                    &mut left_schema,
                    &mut right_schema,
                    &output,
                    &metrics,
                    &mut coalescer,
                );
            }
            None => {
                std::hint::spin_loop();
            }
        }
    }
}

/// Route a batch through the optional coalescer to the output ring.
#[inline]
fn emit_batch(
    batch: RecordBatch,
    output: &SpscRing<RecordBatch>,
    metrics: &PipelineMetrics,
    coalescer: &mut Option<BatchCoalescer>,
) {
    let rows = batch.num_rows() as u64;
    match coalescer {
        Some(ref mut coal) => {
            if let Ok(Some(coalesced)) = coal.push(batch) {
                metrics
                    .rows_emitted
                    .fetch_add(coalesced.num_rows() as u64, Ordering::Relaxed);
                output.push_spin(coalesced);
            }
            // Rows not yet emitted are buffered in the coalescer
        }
        None => {
            metrics.rows_emitted.fetch_add(rows, Ordering::Relaxed);
            output.push_spin(batch);
        }
    }
}

/// Process a single pipeline message.
fn process_message(
    msg: PipelineMessage,
    join: &mut StreamingAsOfJoin,
    left_schema: &mut Option<SchemaRef>,
    right_schema: &mut Option<SchemaRef>,
    output: &SpscRing<RecordBatch>,
    metrics: &PipelineMetrics,
    coalescer: &mut Option<BatchCoalescer>,
) {
    match msg {
        PipelineMessage::Left(batch) => {
            if left_schema.is_none() {
                *left_schema = Some(batch.schema());
            }
            metrics
                .batches_processed
                .fetch_add(1, Ordering::Relaxed);
            let _ = join.push_left(&batch);
        }
        PipelineMessage::Right(batch) => {
            if right_schema.is_none() {
                *right_schema = Some(batch.schema());
            }
            let _ = join.push_right(&batch);
        }
        PipelineMessage::Watermark(ts) => {
            let start = Instant::now();
            join.advance_watermark(ts);

            if let (Some(ls), Some(rs)) = (left_schema.as_ref(), right_schema.as_ref()) {
                if let Ok(Some(batch)) = join.emit(ls, rs) {
                    let elapsed_ns = start.elapsed().as_nanos() as u64;
                    metrics.join_latency.record(elapsed_ns);
                    emit_batch(batch, output, metrics, coalescer);
                }
            }
        }
        PipelineMessage::Flush => {
            if let (Some(ls), Some(rs)) = (left_schema.as_ref(), right_schema.as_ref()) {
                if let Ok(Some(batch)) = join.flush(ls, rs) {
                    emit_batch(batch, output, metrics, coalescer);
                }
            }
            // Also flush the coalescer on explicit Flush
            if let Some(ref mut coal) = coalescer {
                if let Ok(Some(batch)) = coal.flush() {
                    metrics
                        .rows_emitted
                        .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
                    output.push_spin(batch);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};

    fn make_batch(timestamps: Vec<i64>, values: Vec<f64>, col_name: &str) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new(col_name, DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(timestamps)),
                Arc::new(Float64Array::from(values)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn basic_pipeline_flow() {
        let pipeline = StreamingPipeline::start(PipelineConfig::default());

        let right = make_batch(vec![5, 15, 25], vec![0.5, 1.5, 2.5], "bid");
        let left = make_batch(vec![10, 20, 30], vec![1.0, 2.0, 3.0], "price");

        pipeline.send(PipelineMessage::Right(right));
        pipeline.send(PipelineMessage::Left(left));
        pipeline.send(PipelineMessage::Watermark(100));

        // Give pipeline thread time to process
        std::thread::sleep(std::time::Duration::from_millis(50));

        let results = pipeline.drain_output();
        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);

        assert!(pipeline.metrics().batches_processed.load(Ordering::Relaxed) >= 1);
        assert_eq!(
            pipeline.metrics().rows_emitted.load(Ordering::Relaxed),
            3
        );

        pipeline.shutdown();
    }

    #[test]
    fn pipeline_flush_on_shutdown() {
        let config = PipelineConfig {
            lateness_ns: i64::MAX, // Never emit via watermark
            ..Default::default()
        };
        let pipeline = StreamingPipeline::start(config);

        let right = make_batch(vec![5], vec![0.5], "bid");
        let left = make_batch(vec![10], vec![1.0], "price");

        pipeline.send(PipelineMessage::Right(right));
        pipeline.send(PipelineMessage::Left(left));
        pipeline.send(PipelineMessage::Watermark(20));

        std::thread::sleep(std::time::Duration::from_millis(20));

        // Shutdown should flush pending rows
        pipeline.shutdown();
    }

    #[test]
    fn pipeline_metrics_tracking() {
        let pipeline = StreamingPipeline::start(PipelineConfig::default());

        let right = make_batch(vec![5], vec![0.5], "bid");
        let left = make_batch(vec![10], vec![1.0], "price");

        pipeline.send(PipelineMessage::Right(right));
        pipeline.send(PipelineMessage::Left(left));
        pipeline.send(PipelineMessage::Watermark(100));

        std::thread::sleep(std::time::Duration::from_millis(50));

        let metrics = pipeline.metrics();
        assert_eq!(metrics.rows_emitted.load(Ordering::Relaxed), 1);
        assert!(metrics.join_latency.count() > 0);

        pipeline.shutdown();
    }

    #[test]
    fn pipeline_explicit_flush() {
        let config = PipelineConfig {
            lateness_ns: i64::MAX,
            ..Default::default()
        };
        let pipeline = StreamingPipeline::start(config);

        let right = make_batch(vec![5], vec![0.5], "bid");
        let left = make_batch(vec![10], vec![1.0], "price");

        pipeline.send(PipelineMessage::Right(right));
        pipeline.send(PipelineMessage::Left(left));
        pipeline.send(PipelineMessage::Flush);

        // Poll for output with timeout
        let mut total_rows = 0;
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let results = pipeline.drain_output();
            total_rows += results.iter().map(|b| b.num_rows()).sum::<usize>();
            if total_rows >= 1 {
                break;
            }
        }
        assert_eq!(total_rows, 1);

        pipeline.shutdown();
    }

    #[test]
    fn pipeline_backpressure() {
        let config = PipelineConfig {
            input_ring_capacity: 4, // Very small
            ..Default::default()
        };
        let pipeline = StreamingPipeline::start(config);

        // Fill the ring
        for i in 0..4 {
            let batch = make_batch(vec![i], vec![i as f64], "bid");
            let result = pipeline.try_send(PipelineMessage::Right(batch));
            // Some may succeed, some may fail due to backpressure
            if result.is_err() {
                // Backpressure working correctly
                break;
            }
        }

        pipeline.shutdown();
    }

    #[test]
    fn pipeline_with_coalescing() {
        let config = PipelineConfig {
            coalesce_rows: 100, // Coalesce into batches of ~100 rows
            ..Default::default()
        };
        let pipeline = StreamingPipeline::start(config);

        // Send 3 small watermark-triggered emissions
        let right = make_batch(vec![1, 5, 10, 15, 20], vec![0.1, 0.5, 1.0, 1.5, 2.0], "bid");
        pipeline.send(PipelineMessage::Right(right));

        for ts in [2, 6, 12, 16, 22] {
            let left = make_batch(vec![ts], vec![ts as f64], "price");
            pipeline.send(PipelineMessage::Left(left));
        }
        pipeline.send(PipelineMessage::Watermark(100));

        std::thread::sleep(std::time::Duration::from_millis(50));

        // With coalescing, small emissions are buffered. Shutdown flushes them.
        pipeline.shutdown();
    }
}
