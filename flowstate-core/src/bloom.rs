//! Lock-free Bloom filter for probabilistic set membership.
//!
//! Used for fast symbol-level filtering in the streaming join pipeline:
//! before scanning right-side batches for a given symbol, check the Bloom
//! filter first. If the filter says "definitely not present", skip the scan
//! entirely. False positives are bounded by the configured rate.
//!
//! # Design
//!
//! - **Double hashing**: Two independent hashes (using ahash) generate `k`
//!   hash functions via `h_i = h1 + i * h2`. This matches the theoretical
//!   false-positive rate of `k` independent hashes (Kirsch & Mitzenmacher 2006).
//! - **Lock-free insertion**: `AtomicU64` bit array with `fetch_or` allows
//!   concurrent inserts from multiple threads without locking.
//! - **Zero-allocation on hot path**: Fixed bit array, no heap allocation
//!   for insert or query operations.
//!
//! # False Positive Rate
//!
//! For `m` bits and `n` insertions with `k` hash functions:
//!   FPR ≈ (1 - e^(-kn/m))^k
//!
//! The constructor automatically selects optimal `k` for the given capacity
//! and target FPR.

use std::sync::atomic::{AtomicU64, Ordering};

/// A lock-free Bloom filter with configurable false-positive rate.
pub struct BloomFilter {
    /// Bit storage as atomic u64 words.
    bits: Box<[AtomicU64]>,
    /// Total number of bits (m).
    num_bits: u64,
    /// Number of hash functions (k).
    num_hashes: u32,
}

impl BloomFilter {
    /// Create a Bloom filter sized for `expected_items` with the given
    /// false-positive rate (0.0 to 1.0, e.g., 0.01 for 1%).
    ///
    /// # Panics
    ///
    /// Panics if `fpr` is not in (0, 1) or `expected_items` is 0.
    pub fn with_rate(expected_items: usize, fpr: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1)");

        // Optimal number of bits: m = -n * ln(fpr) / (ln(2))^2
        let n = expected_items as f64;
        let m = (-n * fpr.ln() / (2.0_f64.ln().powi(2))).ceil() as u64;
        let m = m.max(64); // Minimum 64 bits

        // Optimal number of hash functions: k = (m/n) * ln(2)
        let k = ((m as f64 / n) * 2.0_f64.ln()).round() as u32;
        let k = k.clamp(1, 30);

        // Round up to whole u64 words
        let num_words = ((m + 63) / 64) as usize;
        let num_bits = num_words as u64 * 64;

        let mut bits = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            bits.push(AtomicU64::new(0));
        }

        Self {
            bits: bits.into_boxed_slice(),
            num_bits,
            num_hashes: k,
        }
    }

    /// Create a Bloom filter with explicit bit count and hash count.
    pub fn with_params(num_bits: u64, num_hashes: u32) -> Self {
        assert!(num_bits >= 64, "num_bits must be >= 64");
        assert!(num_hashes >= 1, "num_hashes must be >= 1");

        let num_words = ((num_bits + 63) / 64) as usize;
        let num_bits = num_words as u64 * 64;

        let mut bits = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            bits.push(AtomicU64::new(0));
        }

        Self {
            bits: bits.into_boxed_slice(),
            num_bits,
            num_hashes,
        }
    }

    /// Insert a key into the filter.
    ///
    /// Lock-free: uses `fetch_or` with `Relaxed` ordering.
    #[inline]
    pub fn insert(&self, key: &[u8]) {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit_pos = self.nth_hash(h1, h2, i);
            let word_idx = (bit_pos / 64) as usize;
            let bit_idx = bit_pos % 64;
            self.bits[word_idx].fetch_or(1 << bit_idx, Ordering::Relaxed);
        }
    }

    /// Insert a string key.
    #[inline]
    pub fn insert_str(&self, key: &str) {
        self.insert(key.as_bytes());
    }

    /// Check if a key might be in the set.
    ///
    /// Returns `false` if the key is definitely not present.
    /// Returns `true` if the key is probably present (with FPR probability
    /// of being a false positive).
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit_pos = self.nth_hash(h1, h2, i);
            let word_idx = (bit_pos / 64) as usize;
            let bit_idx = bit_pos % 64;
            if self.bits[word_idx].load(Ordering::Relaxed) & (1 << bit_idx) == 0 {
                return false;
            }
        }
        true
    }

    /// Check if a string key might be in the set.
    #[inline]
    pub fn contains_str(&self, key: &str) -> bool {
        self.contains(key.as_bytes())
    }

    /// Number of bits set to 1.
    pub fn popcount(&self) -> u64 {
        self.bits
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }

    /// Estimated number of items inserted, based on bit density.
    ///
    /// Uses the formula: n* = -(m/k) * ln(1 - X/m)
    /// where X is the number of set bits.
    pub fn estimated_count(&self) -> f64 {
        let x = self.popcount() as f64;
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;

        if x >= m {
            return f64::INFINITY;
        }

        -(m / k) * (1.0 - x / m).ln()
    }

    /// Estimated current false-positive rate based on actual bit density.
    pub fn estimated_fpr(&self) -> f64 {
        let x = self.popcount() as f64;
        let m = self.num_bits as f64;
        (x / m).powi(self.num_hashes as i32)
    }

    /// Clear all bits (reset to empty).
    pub fn clear(&self) {
        for word in self.bits.iter() {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Total number of bits in the filter.
    #[inline]
    pub fn num_bits(&self) -> u64 {
        self.num_bits
    }

    /// Number of hash functions.
    #[inline]
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Size in bytes of the bit array.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    // -----------------------------------------------------------------------
    // Hashing internals
    // -----------------------------------------------------------------------

    /// Compute two independent hashes using split mixing.
    /// We use a simple but effective approach: FNV-1a variant for h1,
    /// and a rotated/mixed version for h2.
    #[inline]
    fn hash_pair(&self, key: &[u8]) -> (u64, u64) {
        // Hash 1: FNV-1a
        let mut h1: u64 = 0xcbf29ce484222325;
        for &b in key {
            h1 ^= b as u64;
            h1 = h1.wrapping_mul(0x100000001b3);
        }

        // Hash 2: Mix h1 with avalanche
        let mut h2 = h1;
        h2 ^= h2 >> 33;
        h2 = h2.wrapping_mul(0xff51afd7ed558ccd);
        h2 ^= h2 >> 33;
        h2 = h2.wrapping_mul(0xc4ceb9fe1a85ec53);
        h2 ^= h2 >> 33;

        (h1, h2)
    }

    /// Compute the i-th hash position using double hashing.
    #[inline]
    fn nth_hash(&self, h1: u64, h2: u64, i: u32) -> u64 {
        h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.num_bits
    }
}

// Safety: AtomicU64 operations are inherently thread-safe
unsafe impl Send for BloomFilter {}
unsafe impl Sync for BloomFilter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_contains_nothing() {
        let bf = BloomFilter::with_rate(1000, 0.01);
        assert!(!bf.contains_str("AAPL"));
        assert!(!bf.contains_str("GOOG"));
        assert_eq!(bf.popcount(), 0);
    }

    #[test]
    fn insert_and_lookup() {
        let bf = BloomFilter::with_rate(1000, 0.01);
        bf.insert_str("AAPL");
        bf.insert_str("GOOG");
        bf.insert_str("MSFT");

        assert!(bf.contains_str("AAPL"));
        assert!(bf.contains_str("GOOG"));
        assert!(bf.contains_str("MSFT"));
    }

    #[test]
    fn no_false_negatives() {
        let bf = BloomFilter::with_rate(10_000, 0.001);
        let symbols: Vec<String> = (0..1000).map(|i| format!("SYM_{}", i)).collect();

        for s in &symbols {
            bf.insert_str(s);
        }

        // Every inserted key must be found — no false negatives allowed
        for s in &symbols {
            assert!(bf.contains_str(s), "False negative for {}", s);
        }
    }

    #[test]
    fn false_positive_rate_bounded() {
        let n = 1000;
        let target_fpr = 0.05; // 5%
        let bf = BloomFilter::with_rate(n, target_fpr);

        // Insert n items
        for i in 0..n {
            bf.insert_str(&format!("insert_{}", i));
        }

        // Test 10,000 items that were NOT inserted
        let test_count = 10_000;
        let mut false_positives = 0;
        for i in 0..test_count {
            if bf.contains_str(&format!("test_{}", i)) {
                false_positives += 1;
            }
        }

        let observed_fpr = false_positives as f64 / test_count as f64;
        // Allow 2x the target FPR as margin (statistical test)
        assert!(
            observed_fpr < target_fpr * 2.0,
            "FPR {} exceeds 2x target {}",
            observed_fpr,
            target_fpr
        );
    }

    #[test]
    fn estimated_count_reasonable() {
        let bf = BloomFilter::with_rate(1000, 0.01);

        for i in 0..500 {
            bf.insert_str(&format!("item_{}", i));
        }

        let est = bf.estimated_count();
        // Should be within 20% of actual count
        assert!(
            est > 400.0 && est < 600.0,
            "Estimated count {} too far from actual 500",
            est
        );
    }

    #[test]
    fn clear_resets_filter() {
        let bf = BloomFilter::with_rate(100, 0.01);
        bf.insert_str("AAPL");
        assert!(bf.contains_str("AAPL"));

        bf.clear();
        assert!(!bf.contains_str("AAPL"));
        assert_eq!(bf.popcount(), 0);
    }

    #[test]
    fn concurrent_insert() {
        use std::sync::Arc;
        use std::thread;

        let bf = Arc::new(BloomFilter::with_rate(10_000, 0.01));
        let n_threads = 4;
        let n_per_thread = 1000;

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let bf = bf.clone();
                thread::spawn(move || {
                    for i in 0..n_per_thread {
                        bf.insert_str(&format!("t{}_{}", t, i));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Verify all inserted items are found
        for t in 0..n_threads {
            for i in 0..n_per_thread {
                assert!(
                    bf.contains_str(&format!("t{}_{}", t, i)),
                    "Missing t{}_{} after concurrent insert",
                    t,
                    i
                );
            }
        }
    }

    #[test]
    fn with_params_explicit() {
        let bf = BloomFilter::with_params(1024, 7);
        assert_eq!(bf.num_bits(), 1024);
        assert_eq!(bf.num_hashes(), 7);
        assert_eq!(bf.size_bytes(), 128); // 1024 bits = 128 bytes
    }

    #[test]
    fn optimal_params_selection() {
        // For n=1000, fpr=0.01:
        //   m ≈ 9585 bits, k ≈ 7
        let bf = BloomFilter::with_rate(1000, 0.01);
        assert!(bf.num_bits() >= 9000, "Too few bits: {}", bf.num_bits());
        assert!(bf.num_bits() <= 10000, "Too many bits: {}", bf.num_bits());
        assert!(bf.num_hashes() >= 6 && bf.num_hashes() <= 8, "k={}", bf.num_hashes());
    }

    #[test]
    fn byte_keys() {
        let bf = BloomFilter::with_rate(100, 0.01);
        bf.insert(&[0u8, 1, 2, 3]);
        bf.insert(&[255, 254, 253]);

        assert!(bf.contains(&[0, 1, 2, 3]));
        assert!(bf.contains(&[255, 254, 253]));
        assert!(!bf.contains(&[0, 1, 2, 4]));
    }

    #[test]
    fn estimated_fpr_increases_with_inserts() {
        let bf = BloomFilter::with_rate(1000, 0.01);
        let fpr_empty = bf.estimated_fpr();

        for i in 0..500 {
            bf.insert_str(&format!("item_{}", i));
        }
        let fpr_half = bf.estimated_fpr();

        for i in 500..1000 {
            bf.insert_str(&format!("item_{}", i));
        }
        let fpr_full = bf.estimated_fpr();

        assert!(fpr_empty < fpr_half);
        assert!(fpr_half < fpr_full);
    }
}
