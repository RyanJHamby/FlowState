//! Lock-free single-producer single-consumer (SPSC) ring buffer.
//!
//! Designed for low-latency inter-thread communication where one thread produces
//! Arrow RecordBatches and another consumes them. This is the backbone of a
//! streaming market data pipeline: the ingestion thread pushes raw batches into
//! the ring, and the alignment thread drains them.
//!
//! # Design
//!
//! - **Lock-free**: Uses `AtomicU64` with `Ordering::Acquire/Release` — no mutexes,
//!   no syscalls on the hot path.
//! - **Cache-line padded**: Head and tail counters live on separate cache lines
//!   (128-byte alignment for Apple M-series / Intel) to prevent false sharing.
//! - **Power-of-two capacity**: Enables bitwise masking instead of modulo division.
//! - **Bounded**: Fixed capacity with explicit backpressure (try_push returns Err).
//!
//! # Memory Ordering
//!
//! ```text
//! Producer:                     Consumer:
//!   1. Write data to slot         1. Load head (Acquire)
//!   2. Store head (Release)       2. Read data from slot
//!                                 3. Store tail (Release)
//! ```
//!
//! The Release store on the producer side ensures the data write is visible
//! before the head pointer update. The Acquire load on the consumer side
//! ensures it sees the data written by the producer.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache line size for padding. 128 bytes covers both Intel (64B) and
/// Apple Silicon (128B) cache lines.
const CACHE_LINE: usize = 128;

/// A bounded, lock-free SPSC ring buffer.
///
/// `T` is the element type (e.g., `RecordBatch`, `Vec<u8>`, or any `Send` type).
/// Capacity must be a power of two.
pub struct SpscRing<T> {
    /// Ring buffer storage. UnsafeCell because producer writes and consumer reads
    /// different slots without synchronization (correctness guaranteed by head/tail).
    slots: Box<[UnsafeCell<Option<T>>]>,
    /// Bit mask for power-of-two wrap-around: index = counter & mask.
    mask: u64,
    /// Number of elements produced (monotonically increasing).
    /// Padded to its own cache line to prevent false sharing with `tail`.
    head: CacheLinePadded<AtomicU64>,
    /// Number of elements consumed (monotonically increasing).
    /// Padded to its own cache line to prevent false sharing with `head`.
    tail: CacheLinePadded<AtomicU64>,
}

// Safety: Only one thread may call push (producer) and only one thread may
// call pop (consumer). The atomic head/tail with Acquire/Release ordering
// ensures proper synchronization between the two threads.
unsafe impl<T: Send> Send for SpscRing<T> {}
unsafe impl<T: Send> Sync for SpscRing<T> {}

/// Cache-line padded wrapper to prevent false sharing between atomics.
#[repr(C)]
struct CacheLinePadded<T> {
    value: T,
    _pad: [u8; CACHE_LINE - std::mem::size_of::<AtomicU64>()],
}

impl<T> CacheLinePadded<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            _pad: [0u8; CACHE_LINE - std::mem::size_of::<AtomicU64>()],
        }
    }
}

impl<T> SpscRing<T> {
    /// Create a new SPSC ring buffer with the given capacity.
    ///
    /// Capacity is rounded up to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SPSC ring buffer capacity must be > 0");

        // Round up to power of two
        let capacity = capacity.next_power_of_two();

        let mut slots: Vec<UnsafeCell<Option<T>>> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(UnsafeCell::new(None));
        }

        Self {
            slots: slots.into_boxed_slice(),
            mask: (capacity - 1) as u64,
            head: CacheLinePadded::new(AtomicU64::new(0)),
            tail: CacheLinePadded::new(AtomicU64::new(0)),
        }
    }

    /// Capacity of the ring buffer (always a power of two).
    #[inline]
    pub fn capacity(&self) -> usize {
        (self.mask + 1) as usize
    }

    /// Number of elements currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Relaxed);
        (head - tail) as usize
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to push an element into the ring buffer.
    ///
    /// Returns `Err(value)` if the buffer is full (backpressure).
    /// This is the **producer** method — only one thread may call this.
    #[inline]
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);

        // Full check: head has lapped tail by exactly `capacity` elements
        if head - tail >= (self.mask + 1) {
            return Err(value);
        }

        let slot = (head & self.mask) as usize;
        // Safety: producer is the only writer to this slot, and the slot is
        // guaranteed to be empty (consumed) because head - tail < capacity.
        unsafe {
            *self.slots[slot].get() = Some(value);
        }

        // Release: ensure the data write above is visible before head update
        self.head.value.store(head + 1, Ordering::Release);
        Ok(())
    }

    /// Try to pop an element from the ring buffer.
    ///
    /// Returns `None` if the buffer is empty.
    /// This is the **consumer** method — only one thread may call this.
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);

        // Empty check
        if tail >= head {
            return None;
        }

        let slot = (tail & self.mask) as usize;
        // Safety: consumer is the only reader of this slot, and the slot is
        // guaranteed to contain data because tail < head.
        let value = unsafe { (*self.slots[slot].get()).take() };

        // Release: ensure the read above completes before tail update
        self.tail.value.store(tail + 1, Ordering::Release);
        value
    }

    /// Spin-push: busy-wait until the element is pushed.
    ///
    /// Use this when the consumer is expected to drain fast enough that
    /// spinning is cheaper than parking the thread.
    #[inline]
    pub fn push_spin(&self, mut value: T) {
        loop {
            match self.try_push(value) {
                Ok(()) => return,
                Err(v) => {
                    value = v;
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Spin-pop: busy-wait until an element is available.
    #[inline]
    pub fn pop_spin(&self) -> T {
        loop {
            if let Some(v) = self.try_pop() {
                return v;
            }
            std::hint::spin_loop();
        }
    }

    /// Drain all available elements into a Vec.
    ///
    /// Non-blocking: returns whatever is currently available.
    pub fn drain(&self) -> Vec<T> {
        let mut out = Vec::new();
        while let Some(v) = self.try_pop() {
            out.push(v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn basic_push_pop() {
        let ring = SpscRing::new(4);
        assert!(ring.is_empty());
        assert_eq!(ring.capacity(), 4);

        ring.try_push(1).unwrap();
        ring.try_push(2).unwrap();
        assert_eq!(ring.len(), 2);

        assert_eq!(ring.try_pop(), Some(1));
        assert_eq!(ring.try_pop(), Some(2));
        assert_eq!(ring.try_pop(), None);
        assert!(ring.is_empty());
    }

    #[test]
    fn full_backpressure() {
        let ring = SpscRing::new(2);
        ring.try_push(1).unwrap();
        ring.try_push(2).unwrap();

        // Buffer is full — push should fail
        assert_eq!(ring.try_push(3), Err(3));

        // Pop one, then push should succeed
        assert_eq!(ring.try_pop(), Some(1));
        ring.try_push(3).unwrap();
        assert_eq!(ring.try_pop(), Some(2));
        assert_eq!(ring.try_pop(), Some(3));
    }

    #[test]
    fn wraparound() {
        let ring = SpscRing::new(4);

        // Fill and drain multiple times to test wrap-around
        for cycle in 0..10 {
            let base = cycle * 4;
            for i in 0..4 {
                ring.try_push(base + i).unwrap();
            }
            for i in 0..4 {
                assert_eq!(ring.try_pop(), Some(base + i));
            }
            assert!(ring.is_empty());
        }
    }

    #[test]
    fn power_of_two_rounding() {
        let ring: SpscRing<i32> = SpscRing::new(3);
        assert_eq!(ring.capacity(), 4); // Rounded up

        let ring: SpscRing<i32> = SpscRing::new(5);
        assert_eq!(ring.capacity(), 8); // Rounded up

        let ring: SpscRing<i32> = SpscRing::new(8);
        assert_eq!(ring.capacity(), 8); // Already power of two
    }

    #[test]
    fn drain_batch() {
        let ring = SpscRing::new(8);
        ring.try_push(1).unwrap();
        ring.try_push(2).unwrap();
        ring.try_push(3).unwrap();

        let drained = ring.drain();
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(ring.is_empty());
    }

    #[test]
    fn cross_thread_correctness() {
        let ring = Arc::new(SpscRing::new(1024));
        let n = 100_000;

        let producer_ring = ring.clone();
        let producer = thread::spawn(move || {
            for i in 0..n {
                producer_ring.push_spin(i);
            }
        });

        let consumer_ring = ring.clone();
        let consumer = thread::spawn(move || {
            let mut received = Vec::with_capacity(n);
            for _ in 0..n {
                received.push(consumer_ring.pop_spin());
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();

        // Verify ordering is preserved (FIFO)
        assert_eq!(received.len(), n);
        for (i, &v) in received.iter().enumerate() {
            assert_eq!(v, i, "Out of order at index {}", i);
        }
    }

    #[test]
    fn cross_thread_large_elements() {
        // Test with heap-allocated elements to verify no double-free or leak
        let ring = Arc::new(SpscRing::new(256));
        let n = 10_000;

        let producer_ring = ring.clone();
        let producer = thread::spawn(move || {
            for i in 0..n {
                let data = vec![i as u8; 64]; // 64-byte payload
                producer_ring.push_spin(data);
            }
        });

        let consumer_ring = ring.clone();
        let consumer = thread::spawn(move || {
            let mut count = 0;
            for i in 0..n {
                let data = consumer_ring.pop_spin();
                assert_eq!(data.len(), 64);
                assert_eq!(data[0], i as u8);
                count += 1;
            }
            count
        });

        producer.join().unwrap();
        let count = consumer.join().unwrap();
        assert_eq!(count, n);
    }

    #[test]
    fn try_push_pop_interleaved() {
        // Simulate bursty producer / slow consumer
        let ring = SpscRing::new(4);

        // Burst of 4
        for i in 0..4 {
            ring.try_push(i).unwrap();
        }
        assert_eq!(ring.try_push(99), Err(99)); // Full

        // Consumer drains 2
        assert_eq!(ring.try_pop(), Some(0));
        assert_eq!(ring.try_pop(), Some(1));

        // Producer fills 2 more
        ring.try_push(4).unwrap();
        ring.try_push(5).unwrap();
        assert_eq!(ring.try_push(6), Err(6)); // Full again

        // Drain all
        assert_eq!(ring.try_pop(), Some(2));
        assert_eq!(ring.try_pop(), Some(3));
        assert_eq!(ring.try_pop(), Some(4));
        assert_eq!(ring.try_pop(), Some(5));
        assert_eq!(ring.try_pop(), None);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ring: SpscRing<i32> = SpscRing::new(0);
    }
}
