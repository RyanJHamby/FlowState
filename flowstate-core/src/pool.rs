//! Thread-safe object pool with pre-allocated slab storage.
//!
//! Eliminates allocation jitter on the hot path by recycling fixed-size
//! byte buffers. The streaming pipeline allocates temporary buffers for
//! batch construction, column gathering, and IPC serialization — all of
//! which benefit from avoiding `malloc`/`free` churn.
//!
//! # Design
//!
//! - **Slab allocation**: A contiguous `Vec<u8>` backing store divided into
//!   fixed-size slots. Each checkout returns a `PoolSlot` that auto-returns
//!   to the pool on drop.
//! - **Lock-free free-list**: Available slot indices stored in a `Mutex<Vec<usize>>`
//!   (fast for low contention, predictable). For higher contention, swap for
//!   a lock-free stack.
//! - **Bounded capacity**: Pool never grows beyond initial allocation. If
//!   exhausted, callers get `None` and can fall back to heap allocation.
//! - **Zero-initialization on return**: Slots are zeroed when returned to
//!   prevent data leakage between uses.
//!
//! ```text
//! Pool (contiguous memory)
//! ┌─────────┬─────────┬─────────┬─────────┐
//! │ Slot 0  │ Slot 1  │ Slot 2  │ Slot 3  │
//! │ (in use)│ (free)  │ (in use)│ (free)  │
//! └─────────┴─────────┴─────────┴─────────┘
//!   ↑                    ↑
//!   PoolSlot             PoolSlot
//!   (auto-return)        (auto-return)
//! ```

use std::sync::{Arc, Mutex};

/// A pool of fixed-size byte buffers.
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    /// Contiguous backing store.
    storage: Box<[u8]>,
    /// Size of each slot in bytes.
    slot_size: usize,
    /// Total number of slots.
    num_slots: usize,
    /// Stack of available slot indices.
    free_list: Mutex<Vec<usize>>,
}

/// A checked-out buffer slot. Returns to the pool on drop.
pub struct PoolSlot {
    pool: Arc<PoolInner>,
    slot_index: usize,
    /// Pointer to the start of this slot's memory.
    ptr: *mut u8,
    /// Size of the slot.
    len: usize,
}

// Safety: PoolSlot holds exclusive access to its slot. The backing memory
// is owned by the pool (Arc<PoolInner>) and the slot is not shared.
unsafe impl Send for PoolSlot {}

impl BufferPool {
    /// Create a pool with `num_slots` buffers, each `slot_size` bytes.
    ///
    /// Total memory = `num_slots * slot_size` bytes, allocated up front.
    ///
    /// # Panics
    ///
    /// Panics if `num_slots` or `slot_size` is 0.
    pub fn new(num_slots: usize, slot_size: usize) -> Self {
        assert!(num_slots > 0, "num_slots must be > 0");
        assert!(slot_size > 0, "slot_size must be > 0");

        let total_bytes = num_slots * slot_size;
        let storage = vec![0u8; total_bytes].into_boxed_slice();

        // All slots start as free
        let free_list = (0..num_slots).rev().collect::<Vec<_>>();

        Self {
            inner: Arc::new(PoolInner {
                storage,
                slot_size,
                num_slots,
                free_list: Mutex::new(free_list),
            }),
        }
    }

    /// Try to check out a buffer slot. Returns `None` if the pool is exhausted.
    pub fn checkout(&self) -> Option<PoolSlot> {
        let slot_index = {
            let mut free = self.inner.free_list.lock().unwrap();
            free.pop()?
        };

        let offset = slot_index * self.inner.slot_size;
        // Safety: We have exclusive access to this slot (removed from free list).
        // The storage lives as long as the Arc<PoolInner>.
        let ptr = unsafe {
            (self.inner.storage.as_ptr() as *mut u8).add(offset)
        };

        Some(PoolSlot {
            pool: self.inner.clone(),
            slot_index,
            ptr,
            len: self.inner.slot_size,
        })
    }

    /// Number of currently available (free) slots.
    pub fn available(&self) -> usize {
        self.inner.free_list.lock().unwrap().len()
    }

    /// Total number of slots in the pool.
    pub fn capacity(&self) -> usize {
        self.inner.num_slots
    }

    /// Number of currently checked-out slots.
    pub fn in_use(&self) -> usize {
        self.capacity() - self.available()
    }

    /// Size of each slot in bytes.
    pub fn slot_size(&self) -> usize {
        self.inner.slot_size
    }

    /// Total memory footprint of the pool in bytes.
    pub fn total_bytes(&self) -> usize {
        self.inner.num_slots * self.inner.slot_size
    }
}

impl PoolSlot {
    /// Get a mutable slice of the buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: We have exclusive access to this slot's memory range.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get an immutable slice of the buffer.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // Safety: We have exclusive access to this slot's memory range.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The size of this buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer has zero length (always false for pool slots).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The slot index within the pool.
    #[inline]
    pub fn slot_index(&self) -> usize {
        self.slot_index
    }
}

impl Drop for PoolSlot {
    fn drop(&mut self) {
        // Zero the slot to prevent data leakage
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) };
        slice.fill(0);

        // Return slot to the free list
        let mut free = self.pool.free_list.lock().unwrap();
        free.push(self.slot_index);
    }
}

impl std::fmt::Debug for PoolSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolSlot")
            .field("slot_index", &self.slot_index)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_checkout_return() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.in_use(), 0);

        let slot = pool.checkout().unwrap();
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.in_use(), 1);
        assert_eq!(slot.len(), 1024);

        drop(slot);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn write_and_read() {
        let pool = BufferPool::new(2, 64);
        let mut slot = pool.checkout().unwrap();

        let data = b"hello, pool!";
        slot.as_mut_slice()[..data.len()].copy_from_slice(data);

        assert_eq!(&slot.as_slice()[..data.len()], data);
    }

    #[test]
    fn exhaustion_returns_none() {
        let pool = BufferPool::new(2, 32);

        let _s1 = pool.checkout().unwrap();
        let _s2 = pool.checkout().unwrap();
        assert!(pool.checkout().is_none());
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn slot_zeroed_on_return() {
        let pool = BufferPool::new(1, 64);

        // Write data to slot
        {
            let mut slot = pool.checkout().unwrap();
            slot.as_mut_slice().fill(0xFF);
        }
        // Slot returned — should be zeroed

        let slot = pool.checkout().unwrap();
        assert!(slot.as_slice().iter().all(|&b| b == 0), "Slot not zeroed on return");
    }

    #[test]
    fn multiple_checkout_return_cycles() {
        let pool = BufferPool::new(2, 128);

        for cycle in 0..10 {
            let mut s1 = pool.checkout().unwrap();
            let mut s2 = pool.checkout().unwrap();
            assert!(pool.checkout().is_none());

            s1.as_mut_slice()[0] = cycle;
            s2.as_mut_slice()[0] = cycle + 100;

            drop(s1);
            drop(s2);
            assert_eq!(pool.available(), 2);
        }
    }

    #[test]
    fn concurrent_checkout() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(BufferPool::new(8, 256));
        let n_threads = 4;
        let iterations = 100;

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || {
                    for _ in 0..iterations {
                        if let Some(mut slot) = pool.checkout() {
                            // Write thread-local data
                            slot.as_mut_slice()[0] = 42;
                            assert_eq!(slot.as_slice()[0], 42);
                            // Slot auto-returned on drop
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All slots should be returned
        assert_eq!(pool.available(), 8);
    }

    #[test]
    fn disjoint_slot_memory() {
        let pool = BufferPool::new(3, 64);
        let mut s0 = pool.checkout().unwrap();
        let mut s1 = pool.checkout().unwrap();
        let mut s2 = pool.checkout().unwrap();

        // Write different patterns to each slot
        s0.as_mut_slice().fill(0xAA);
        s1.as_mut_slice().fill(0xBB);
        s2.as_mut_slice().fill(0xCC);

        // Verify no cross-contamination
        assert!(s0.as_slice().iter().all(|&b| b == 0xAA));
        assert!(s1.as_slice().iter().all(|&b| b == 0xBB));
        assert!(s2.as_slice().iter().all(|&b| b == 0xCC));
    }

    #[test]
    fn pool_metrics() {
        let pool = BufferPool::new(10, 4096);
        assert_eq!(pool.capacity(), 10);
        assert_eq!(pool.slot_size(), 4096);
        assert_eq!(pool.total_bytes(), 40960);
    }

    #[test]
    fn slot_debug_format() {
        let pool = BufferPool::new(1, 32);
        let slot = pool.checkout().unwrap();
        let debug = format!("{:?}", slot);
        assert!(debug.contains("PoolSlot"));
        assert!(debug.contains("slot_index"));
    }

    #[test]
    #[should_panic(expected = "num_slots must be > 0")]
    fn zero_slots_panics() {
        BufferPool::new(0, 64);
    }

    #[test]
    #[should_panic(expected = "slot_size must be > 0")]
    fn zero_size_panics() {
        BufferPool::new(4, 0);
    }
}
