//! CUDA pinned (page-locked) memory allocator with CPU fallback.
//!
//! Pinned memory is critical for high-throughput GPU data feeding: DMA
//! transfers from host to device can only run at full PCIe bandwidth when
//! the source buffer is page-locked (pinned). Unpinned memory requires the
//! driver to first copy into a staging buffer, halving effective bandwidth.
//!
//! # Design
//!
//! - **Pool-based allocation**: Pre-allocates a pool of pinned buffers at
//!   startup. Checkout/return avoids repeated `cudaMallocHost`/`cudaFreeHost`
//!   calls (each ~100µs due to TLB shootdown).
//! - **CPU fallback**: When CUDA is unavailable, allocates page-aligned
//!   buffers via `std::alloc` with page-size alignment. The downstream
//!   pipeline works identically — only DMA transfer speed differs.
//! - **Double-buffered prefetch**: The pool is sized for N+1 buffering so
//!   the CPU can fill buffer N+1 while the GPU processes buffer N.
//! - **Zero-copy to Arrow**: Pinned buffers can back Arrow `Buffer` objects
//!   for zero-copy handoff to the GPU kernel.
//!
//! ```text
//!   ┌─────────────────────────────────────────────────┐
//!   │ PinnedPool (N slots)                            │
//!   │ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
//!   │ │ Pinned 0 │ │ Pinned 1 │ │ Pinned 2 │  ...    │
//!   │ │ (GPU DMA)│ │ (CPU fill)│ │ (free)  │         │
//!   │ └──────────┘ └──────────┘ └──────────┘         │
//!   └─────────────────────────────────────────────────┘
//!         ↓ PCIe DMA       ↑ memcpy from Arrow
//!   ┌──────────┐     ┌──────────┐
//!   │ GPU VRAM │     │ RecordBatch│
//!   └──────────┘     └──────────┘
//! ```

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::Mutex;

/// Whether the allocator is using real CUDA pinned memory or CPU fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnedBackend {
    /// Real CUDA pinned memory via `cudaMallocHost`.
    Cuda,
    /// Page-aligned CPU memory (fallback when CUDA unavailable).
    CpuAligned,
}

/// A single pinned memory buffer.
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
    backend: PinnedBackend,
}

// Safety: PinnedBuffer owns its memory exclusively.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    /// Allocate a pinned buffer of `size` bytes.
    ///
    /// Tries CUDA pinned allocation first; falls back to page-aligned CPU
    /// memory if CUDA is unavailable or the feature is disabled.
    pub fn allocate(size: usize) -> Result<Self, PinnedAllocError> {
        if size == 0 {
            return Err(PinnedAllocError::ZeroSize);
        }

        // Try CUDA pinned allocation if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(buf) = Self::allocate_cuda(size) {
                return Ok(buf);
            }
        }

        // CPU fallback: page-aligned allocation
        Self::allocate_aligned(size)
    }

    /// Allocate page-aligned CPU memory (always available).
    fn allocate_aligned(size: usize) -> Result<Self, PinnedAllocError> {
        let page_size = Self::page_size();
        // Round up to page boundary for optimal DMA alignment
        let aligned_size = (size + page_size - 1) & !(page_size - 1);
        let layout = Layout::from_size_align(aligned_size, page_size)
            .map_err(|_| PinnedAllocError::LayoutError)?;

        // Safety: layout is valid (non-zero size, power-of-two alignment)
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(PinnedAllocError::AllocationFailed)?;

        Ok(Self {
            ptr,
            layout,
            len: size,
            backend: PinnedBackend::CpuAligned,
        })
    }

    /// CUDA pinned allocation (only compiled with `cuda` feature).
    #[cfg(feature = "cuda")]
    fn allocate_cuda(size: usize) -> Result<Self, PinnedAllocError> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // cudaMallocHost pins the memory and registers it with the CUDA driver
        let result = unsafe { cuda_sys::cudaMallocHost(&mut ptr, size) };
        if result != 0 || ptr.is_null() {
            return Err(PinnedAllocError::CudaError(result));
        }
        // Zero-initialize for safety
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size) };
        let layout = Layout::from_size_align(size, Self::page_size())
            .map_err(|_| PinnedAllocError::LayoutError)?;
        Ok(Self {
            ptr: NonNull::new(ptr as *mut u8).unwrap(),
            layout,
            len: size,
            backend: PinnedBackend::Cuda,
        })
    }

    /// Get a mutable slice of the buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get an immutable slice of the buffer.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Raw pointer to the buffer (for FFI / DMA registration).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable raw pointer to the buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Size of the buffer in bytes (requested size, not aligned size).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer has zero length (always false for valid buffers).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The actual allocated size (page-aligned, >= len).
    #[inline]
    pub fn allocated_size(&self) -> usize {
        self.layout.size()
    }

    /// Which backend is providing the memory.
    #[inline]
    pub fn backend(&self) -> PinnedBackend {
        self.backend
    }

    /// Whether the buffer is backed by real CUDA pinned memory.
    #[inline]
    pub fn is_cuda_pinned(&self) -> bool {
        self.backend == PinnedBackend::Cuda
    }

    /// System page size.
    fn page_size() -> usize {
        // POSIX: typically 4096, Apple Silicon: 16384
        #[cfg(target_os = "macos")]
        { 16384 }
        #[cfg(not(target_os = "macos"))]
        { 4096 }
    }

    /// Copy data from a byte slice into this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() > self.len`.
    #[inline]
    pub fn copy_from_slice(&mut self, data: &[u8]) {
        assert!(data.len() <= self.len, "data exceeds buffer capacity");
        self.as_mut_slice()[..data.len()].copy_from_slice(data);
    }

    /// Zero-fill the buffer.
    #[inline]
    pub fn zero(&mut self) {
        self.as_mut_slice().fill(0);
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        match self.backend {
            PinnedBackend::Cuda => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_sys::cudaFreeHost(self.ptr.as_ptr() as *mut std::ffi::c_void);
                }
            }
            PinnedBackend::CpuAligned => {
                // Safety: ptr was allocated with this layout via alloc::alloc_zeroed
                unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
            }
        }
    }
}

/// Errors from pinned memory allocation.
#[derive(Debug)]
pub enum PinnedAllocError {
    /// Requested zero-size allocation.
    ZeroSize,
    /// Layout computation failed.
    LayoutError,
    /// System allocator returned null.
    AllocationFailed,
    /// CUDA allocation failed with error code.
    #[allow(dead_code)]
    CudaError(i32),
}

impl std::fmt::Display for PinnedAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSize => write!(f, "cannot allocate zero-size pinned buffer"),
            Self::LayoutError => write!(f, "invalid allocation layout"),
            Self::AllocationFailed => write!(f, "system allocator returned null"),
            Self::CudaError(code) => write!(f, "cudaMallocHost failed with error {}", code),
        }
    }
}

impl std::error::Error for PinnedAllocError {}

// ---------------------------------------------------------------------------
// PinnedPool: pooled pinned buffers for double-buffered prefetch
// ---------------------------------------------------------------------------

/// A pool of pre-allocated pinned buffers for zero-allocation prefetch.
///
/// Sized for double-buffered (or N-buffered) GPU data feeding:
/// - Buffer N is being DMA'd to GPU
/// - Buffer N+1 is being filled by CPU
/// - Buffer N+2 is available for the next fill
pub struct PinnedPool {
    /// Available buffers ready for checkout.
    available: Mutex<Vec<PinnedBuffer>>,
    /// Buffer size for each slot.
    buffer_size: usize,
    /// Total slots allocated (including checked-out ones).
    total_slots: usize,
    /// Backend used for allocation.
    backend: PinnedBackend,
}

impl PinnedPool {
    /// Create a pool of `num_buffers` pinned buffers, each `buffer_size` bytes.
    ///
    /// All buffers are allocated upfront. Returns an error if any allocation fails.
    pub fn new(num_buffers: usize, buffer_size: usize) -> Result<Self, PinnedAllocError> {
        if num_buffers == 0 {
            return Err(PinnedAllocError::ZeroSize);
        }

        let mut buffers = Vec::with_capacity(num_buffers);
        for _ in 0..num_buffers {
            buffers.push(PinnedBuffer::allocate(buffer_size)?);
        }

        let backend = buffers[0].backend();

        Ok(Self {
            available: Mutex::new(buffers),
            buffer_size,
            total_slots: num_buffers,
            backend,
        })
    }

    /// Check out a pinned buffer from the pool. Returns `None` if exhausted.
    pub fn checkout(&self) -> Option<PinnedBuffer> {
        self.available.lock().unwrap().pop()
    }

    /// Return a buffer to the pool. The buffer is zeroed before reuse.
    pub fn checkin(&self, mut buffer: PinnedBuffer) {
        buffer.zero();
        self.available.lock().unwrap().push(buffer);
    }

    /// Number of available (free) buffers.
    pub fn available(&self) -> usize {
        self.available.lock().unwrap().len()
    }

    /// Number of currently checked-out buffers.
    pub fn in_use(&self) -> usize {
        self.total_slots - self.available()
    }

    /// Total number of buffer slots.
    pub fn capacity(&self) -> usize {
        self.total_slots
    }

    /// Size of each buffer in bytes.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Which backend the pool is using.
    pub fn backend(&self) -> PinnedBackend {
        self.backend
    }

    /// Total pinned memory footprint in bytes.
    pub fn total_bytes(&self) -> usize {
        self.total_slots * self.buffer_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_use() {
        let mut buf = PinnedBuffer::allocate(4096).unwrap();
        assert_eq!(buf.len(), 4096);
        assert!(!buf.is_empty());
        assert_eq!(buf.backend(), PinnedBackend::CpuAligned);

        // Write and read
        let data = b"temporal alignment";
        buf.copy_from_slice(data);
        assert_eq!(&buf.as_slice()[..data.len()], data);
    }

    #[test]
    fn page_aligned() {
        let buf = PinnedBuffer::allocate(100).unwrap();
        let ptr_val = buf.as_ptr() as usize;
        let page_size = PinnedBuffer::page_size();
        assert_eq!(ptr_val % page_size, 0, "Buffer not page-aligned");
        assert!(buf.allocated_size() >= buf.len());
        assert_eq!(buf.allocated_size() % page_size, 0);
    }

    #[test]
    fn zero_fill() {
        let mut buf = PinnedBuffer::allocate(256).unwrap();
        buf.as_mut_slice().fill(0xFF);
        buf.zero();
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn zero_size_rejected() {
        assert!(PinnedBuffer::allocate(0).is_err());
    }

    #[test]
    fn large_allocation() {
        // 16MB — typical batch buffer size
        let mut buf = PinnedBuffer::allocate(16 * 1024 * 1024).unwrap();
        assert_eq!(buf.len(), 16 * 1024 * 1024);
        // Write to last byte to verify full range is accessible
        let last = buf.len() - 1;
        buf.as_mut_slice()[last] = 0xAB;
        assert_eq!(buf.as_slice()[last], 0xAB);
    }

    #[test]
    fn pool_basic() {
        let pool = PinnedPool::new(4, 8192).unwrap();
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.buffer_size(), 8192);
        assert_eq!(pool.backend(), PinnedBackend::CpuAligned);
    }

    #[test]
    fn pool_checkout_checkin() {
        let pool = PinnedPool::new(2, 1024).unwrap();
        assert_eq!(pool.available(), 2);

        let buf1 = pool.checkout().unwrap();
        assert_eq!(pool.available(), 1);
        assert_eq!(pool.in_use(), 1);

        let buf2 = pool.checkout().unwrap();
        assert_eq!(pool.available(), 0);
        assert!(pool.checkout().is_none());

        pool.checkin(buf1);
        assert_eq!(pool.available(), 1);

        pool.checkin(buf2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn pool_zeroes_on_checkin() {
        let pool = PinnedPool::new(1, 64).unwrap();

        let mut buf = pool.checkout().unwrap();
        buf.as_mut_slice().fill(0xFF);
        pool.checkin(buf);

        let buf = pool.checkout().unwrap();
        assert!(buf.as_slice().iter().all(|&b| b == 0), "Buffer not zeroed on checkin");
    }

    #[test]
    fn pool_double_buffer_pattern() {
        // Simulate double-buffered prefetch: fill one while "DMA" the other
        let pool = PinnedPool::new(3, 4096).unwrap();

        // Checkout buffers 0 and 1 for double buffering
        let mut fill_buf = pool.checkout().unwrap();
        let dma_buf = pool.checkout().unwrap();
        assert_eq!(pool.available(), 1); // One spare

        // Fill buffer while "DMA is in progress"
        fill_buf.copy_from_slice(&[1u8; 100]);

        // Swap: return DMA buffer, promote fill to DMA
        pool.checkin(dma_buf);
        let mut new_fill = pool.checkout().unwrap();
        new_fill.copy_from_slice(&[2u8; 200]);

        // Return both
        pool.checkin(fill_buf);
        pool.checkin(new_fill);
        assert_eq!(pool.available(), 3);
    }

    #[test]
    fn pool_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(PinnedPool::new(8, 1024).unwrap());
        let n_threads = 4;
        let iterations = 50;

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || {
                    for _ in 0..iterations {
                        if let Some(mut buf) = pool.checkout() {
                            buf.copy_from_slice(&[42u8; 10]);
                            assert_eq!(buf.as_slice()[0], 42);
                            pool.checkin(buf);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(pool.available(), 8);
    }

    #[test]
    fn pool_total_bytes() {
        let pool = PinnedPool::new(4, 16384).unwrap();
        assert_eq!(pool.total_bytes(), 4 * 16384);
    }

    #[test]
    fn buffer_initial_zeroed() {
        let buf = PinnedBuffer::allocate(1024).unwrap();
        assert!(buf.as_slice().iter().all(|&b| b == 0), "Fresh buffer not zeroed");
    }

    #[test]
    #[should_panic(expected = "data exceeds buffer capacity")]
    fn copy_overflow_panics() {
        let mut buf = PinnedBuffer::allocate(16).unwrap();
        buf.copy_from_slice(&[0u8; 32]); // 32 > 16
    }
}
