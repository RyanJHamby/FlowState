//! Lock-free HDR (High Dynamic Range) histogram for nanosecond latency tracking.
//!
//! Tracks latency distributions with configurable precision across a wide range
//! (1ns to ~4.3s at default settings). Used for measuring p50/p99/p999 latencies
//! on join operations, streaming emission, and IPC reads.
//!
//! # Design
//!
//! - **Log-linear bucketing**: Exponentially spaced sub-ranges with linear buckets
//!   within each range. This gives constant relative error across the full range
//!   (like HDR Histogram by Gil Tene).
//! - **Lock-free recording**: `AtomicU64` counters allow concurrent recording from
//!   multiple threads without locking.
//! - **Zero-allocation on the hot path**: Fixed bucket array, no heap allocation
//!   when recording a value.
//!
//! # Precision
//!
//! With the default 3 significant digits of precision:
//! - Values from 1-1023 are tracked exactly (sub-range 0)
//! - Values from 1024-2047 are tracked to ±1
//! - Values from 1M-2M are tracked to ±1024
//! - Relative error is always < 0.1%

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of significant digits of precision. 3 gives <0.1% error.
const SIGNIFICANT_BITS: u32 = 10; // 2^10 = 1024 buckets per sub-range

/// Number of sub-ranges (exponential doublings).
const SUB_RANGE_COUNT: u32 = 22; // Covers up to 2^(22+10) = 2^32 ≈ 4.3 billion ns ≈ 4.3s

/// Total number of buckets.
const BUCKET_COUNT: usize = (SIGNIFICANT_BITS as usize + 1) * SUB_RANGE_COUNT as usize
    + SIGNIFICANT_BITS as usize;

/// A high-dynamic-range histogram for nanosecond latency measurement.
///
/// Thread-safe: multiple threads can call `record()` concurrently.
pub struct HdrHistogram {
    counts: Box<[AtomicU64]>,
    total_count: AtomicU64,
    total_sum: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl HdrHistogram {
    /// Create a new empty histogram.
    pub fn new() -> Self {
        let mut counts = Vec::with_capacity(BUCKET_COUNT);
        for _ in 0..BUCKET_COUNT {
            counts.push(AtomicU64::new(0));
        }
        Self {
            counts: counts.into_boxed_slice(),
            total_count: AtomicU64::new(0),
            total_sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    /// Record a latency value (in nanoseconds).
    ///
    /// Lock-free: uses `Relaxed` ordering since we only need eventual consistency
    /// for statistics, not strict ordering.
    #[inline]
    pub fn record(&self, value_ns: u64) {
        let idx = self.value_to_index(value_ns);
        if idx < self.counts.len() {
            self.counts[idx].fetch_add(1, Ordering::Relaxed);
        }
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum.fetch_add(value_ns, Ordering::Relaxed);

        // Update min (CAS loop)
        let mut current_min = self.min.load(Ordering::Relaxed);
        while value_ns < current_min {
            match self.min.compare_exchange_weak(
                current_min,
                value_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max (CAS loop)
        let mut current_max = self.max.load(Ordering::Relaxed);
        while value_ns > current_max {
            match self.max.compare_exchange_weak(
                current_max,
                value_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Total number of recorded values.
    #[inline]
    pub fn count(&self) -> u64 {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Minimum recorded value.
    #[inline]
    pub fn min(&self) -> u64 {
        let v = self.min.load(Ordering::Relaxed);
        if v == u64::MAX { 0 } else { v }
    }

    /// Maximum recorded value.
    #[inline]
    pub fn max(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }

    /// Mean of all recorded values.
    #[inline]
    pub fn mean(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            return 0.0;
        }
        self.total_sum.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Value at the given percentile (0.0 to 100.0).
    ///
    /// Returns the lowest value that is >= the given percentile of all recorded values.
    pub fn percentile(&self, pct: f64) -> u64 {
        let count = self.count();
        if count == 0 {
            return 0;
        }

        let target = ((pct / 100.0) * count as f64).ceil() as u64;
        let mut cumulative: u64 = 0;

        for (idx, bucket) in self.counts.iter().enumerate() {
            let bucket_count = bucket.load(Ordering::Relaxed);
            cumulative += bucket_count;
            if cumulative >= target {
                return self.index_to_value(idx);
            }
        }

        self.max()
    }

    /// Shorthand for common percentiles.
    #[inline]
    pub fn p50(&self) -> u64 {
        self.percentile(50.0)
    }
    #[inline]
    pub fn p90(&self) -> u64 {
        self.percentile(90.0)
    }
    #[inline]
    pub fn p99(&self) -> u64 {
        self.percentile(99.0)
    }
    #[inline]
    pub fn p999(&self) -> u64 {
        self.percentile(99.9)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        for bucket in self.counts.iter() {
            bucket.store(0, Ordering::Relaxed);
        }
        self.total_count.store(0, Ordering::Relaxed);
        self.total_sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }

    /// Summary statistics as a struct.
    pub fn summary(&self) -> HistogramSummary {
        HistogramSummary {
            count: self.count(),
            min_ns: self.min(),
            max_ns: self.max(),
            mean_ns: self.mean(),
            p50_ns: self.p50(),
            p90_ns: self.p90(),
            p99_ns: self.p99(),
            p999_ns: self.p999(),
        }
    }

    // -----------------------------------------------------------------------
    // Index mapping: value ↔ bucket index
    // -----------------------------------------------------------------------

    /// Map a value to its bucket index.
    ///
    /// Sub-range 0: values 0..2^SIGNIFICANT_BITS → 1:1 mapping (exact)
    /// Sub-range k (k>0): values in [2^(k+SIGNIFICANT_BITS-1), 2^(k+SIGNIFICANT_BITS))
    ///   → 2^SIGNIFICANT_BITS buckets, each covering a range of 2^(k-1)
    #[inline]
    fn value_to_index(&self, value: u64) -> usize {
        if value == 0 {
            return 0;
        }

        let bits = 64 - value.leading_zeros(); // Number of significant bits

        if bits <= SIGNIFICANT_BITS {
            // Sub-range 0: exact mapping
            value as usize
        } else {
            // Higher sub-ranges: log-linear
            let sub_range = bits - SIGNIFICANT_BITS;
            let shift = sub_range - 1;
            let bucket_in_range = (value >> shift) as usize & ((1 << SIGNIFICANT_BITS) - 1);
            let base = (sub_range as usize) * (1 << SIGNIFICANT_BITS);
            base + bucket_in_range
        }
    }

    /// Map a bucket index back to the representative value (lower bound of the bucket).
    #[inline]
    fn index_to_value(&self, index: usize) -> u64 {
        let sub_range = index >> SIGNIFICANT_BITS;
        let bucket_in_range = index & ((1 << SIGNIFICANT_BITS) - 1);

        if sub_range == 0 {
            bucket_in_range as u64
        } else {
            let shift = sub_range - 1;
            (bucket_in_range as u64 | (1 << SIGNIFICANT_BITS)) << shift
        }
    }
}

impl Default for HdrHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics snapshot.
#[derive(Debug, Clone)]
pub struct HistogramSummary {
    pub count: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram() {
        let h = HdrHistogram::new();
        assert_eq!(h.count(), 0);
        assert_eq!(h.min(), 0);
        assert_eq!(h.max(), 0);
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.p50(), 0);
        assert_eq!(h.p99(), 0);
    }

    #[test]
    fn single_value() {
        let h = HdrHistogram::new();
        h.record(1000);
        assert_eq!(h.count(), 1);
        assert_eq!(h.min(), 1000);
        assert_eq!(h.max(), 1000);
        assert_eq!(h.mean(), 1000.0);
        assert_eq!(h.p50(), 1000);
        assert_eq!(h.p99(), 1000);
    }

    #[test]
    fn exact_values_in_low_range() {
        let h = HdrHistogram::new();
        // Values below 1024 should be tracked exactly
        for v in [1, 10, 100, 500, 1023] {
            h.record(v);
        }
        assert_eq!(h.count(), 5);
        assert_eq!(h.min(), 1);
        assert_eq!(h.max(), 1023);
    }

    #[test]
    fn percentile_ordering() {
        let h = HdrHistogram::new();
        for i in 1..=1000 {
            h.record(i);
        }
        assert_eq!(h.count(), 1000);
        assert!(h.p50() <= h.p90());
        assert!(h.p90() <= h.p99());
        assert!(h.p99() <= h.p999());
    }

    #[test]
    fn percentile_accuracy() {
        let h = HdrHistogram::new();
        // Record values 1..=100
        for i in 1..=100 {
            h.record(i);
        }
        // p50 should be near 50
        let p50 = h.p50();
        assert!(p50 >= 49 && p50 <= 51, "p50={} expected ~50", p50);
        // p99 should be near 99
        let p99 = h.p99();
        assert!(p99 >= 98 && p99 <= 100, "p99={} expected ~99", p99);
    }

    #[test]
    fn high_values() {
        let h = HdrHistogram::new();
        // Microsecond-scale latencies
        h.record(1_000); // 1µs
        h.record(10_000); // 10µs
        h.record(100_000); // 100µs
        h.record(1_000_000); // 1ms
        assert_eq!(h.count(), 4);
        assert_eq!(h.min(), 1_000);
        assert_eq!(h.max(), 1_000_000);
    }

    #[test]
    fn reset_clears_all() {
        let h = HdrHistogram::new();
        for i in 1..=100 {
            h.record(i);
        }
        assert_eq!(h.count(), 100);

        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.min(), 0);
        assert_eq!(h.max(), 0);
        assert_eq!(h.p50(), 0);
    }

    #[test]
    fn index_roundtrip_low_range() {
        let h = HdrHistogram::new();
        // Values in the exact range should roundtrip perfectly
        for v in 0..1024u64 {
            let idx = h.value_to_index(v);
            let back = h.index_to_value(idx);
            assert_eq!(back, v, "roundtrip failed for value {}", v);
        }
    }

    #[test]
    fn index_roundtrip_high_range() {
        let h = HdrHistogram::new();
        // Higher values should roundtrip to the bucket lower bound
        for v in [1024, 2048, 4096, 10_000, 100_000, 1_000_000] {
            let idx = h.value_to_index(v);
            let back = h.index_to_value(idx);
            // back should be <= v and close to v (within bucket width)
            assert!(back <= v, "back={} > v={}", back, v);
            let error = (v - back) as f64 / v as f64;
            assert!(
                error < 0.002,
                "roundtrip error {:.4} too high for v={}",
                error,
                v
            );
        }
    }

    #[test]
    fn concurrent_recording() {
        use std::sync::Arc;
        use std::thread;

        let h = Arc::new(HdrHistogram::new());
        let n_threads = 4;
        let n_per_thread = 10_000;

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let h = h.clone();
                thread::spawn(move || {
                    for i in 0..n_per_thread {
                        h.record((t * n_per_thread + i) as u64);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(h.count(), (n_threads * n_per_thread) as u64);
    }

    #[test]
    fn summary_snapshot() {
        let h = HdrHistogram::new();
        for i in 1..=1000 {
            h.record(i);
        }
        let s = h.summary();
        assert_eq!(s.count, 1000);
        assert_eq!(s.min_ns, 1);
        assert_eq!(s.max_ns, 1000);
        assert!(s.p50_ns > 0);
        assert!(s.p99_ns > 0);
    }
}
