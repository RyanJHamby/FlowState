//! Parallel chunked merge-scan kernels.
//!
//! Partitions the left timestamp array into chunks, binary-searches the right
//! array for each chunk's starting cursor, then runs independent merge-scans
//! per chunk via rayon. Achieves near-linear speedup on multi-core for large
//! inputs while falling back to the single-threaded path for small inputs.
//!
//! Key insight: because both arrays are sorted, we can binary-search for the
//! right-side cursor position at each chunk boundary, giving each chunk an
//! independent starting point without any cross-chunk coordination.

use rayon::prelude::*;
use std::cell::UnsafeCell;

/// Minimum elements per chunk before we bother parallelizing.
/// Below this threshold, thread overhead dominates.
const MIN_CHUNK_SIZE: usize = 8_192;

/// Thread-safe wrapper for writing to disjoint slices of a Vec.
/// Safety: callers must guarantee non-overlapping write regions.
struct UnsafeSlice<'a>(&'a [UnsafeCell<Option<usize>>]);
unsafe impl<'a> Send for UnsafeSlice<'a> {}
unsafe impl<'a> Sync for UnsafeSlice<'a> {}

impl<'a> UnsafeSlice<'a> {
    /// Write a value at index. Safety: no two threads may write to the same index.
    #[inline]
    unsafe fn write(&self, idx: usize, val: Option<usize>) {
        *self.0[idx].get() = val;
    }
}

/// Create an UnsafeSlice view over a Vec<Option<usize>>.
fn make_shared_slice(result: &[Option<usize>]) -> UnsafeSlice<'_> {
    // Safety: UnsafeCell<T> has the same layout as T
    let ptr = result.as_ptr() as *const UnsafeCell<Option<usize>>;
    let slice = unsafe { std::slice::from_raw_parts(ptr, result.len()) };
    UnsafeSlice(slice)
}

/// Parallel backward scan: partition left into chunks, binary-search right for
/// each chunk's cursor start, scan each chunk independently.
pub fn par_merge_scan_backward(
    left_ts: &[i64],
    right_ts: &[i64],
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_left = left_ts.len();
    let n_right = right_ts.len();

    if n_left == 0 {
        return Vec::new();
    }
    if n_right == 0 {
        return vec![None; n_left];
    }

    // For small inputs, single-threaded is faster
    let num_threads = rayon::current_num_threads();
    if n_left < MIN_CHUNK_SIZE * 2 || num_threads <= 1 {
        return super::scan::merge_scan_backward(left_ts, right_ts, tolerance_ns, allow_exact);
    }

    let chunk_size = (n_left + num_threads - 1) / num_threads;
    let result: Vec<Option<usize>> = vec![None; n_left];

    // Split into chunk descriptors: (start_idx, end_idx, cursor_start)
    let chunks: Vec<(usize, usize, usize)> = (0..n_left)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(n_left);
            // Binary search: find rightmost right[j] <= left[start]
            // This gives us the starting cursor for this chunk
            let cursor = if start == 0 {
                0
            } else {
                backward_cursor_start(left_ts[start], right_ts, allow_exact)
            };
            (start, end, cursor)
        })
        .collect();

    // Process chunks in parallel via UnsafeSlice for thread-safe disjoint writes
    let shared = make_shared_slice(&result);

    chunks.par_iter().for_each(|&(start, end, cursor_start)| {
        let chunk_left = &left_ts[start..end];
        let chunk_result = scan_backward_chunk(
            chunk_left,
            right_ts,
            cursor_start,
            tolerance_ns,
            allow_exact,
        );

        // Safety: chunks write to non-overlapping index ranges
        for (k, val) in chunk_result.into_iter().enumerate() {
            unsafe { shared.write(start + k, val); }
        }
    });

    result
}

/// Parallel forward scan with the same chunking strategy.
pub fn par_merge_scan_forward(
    left_ts: &[i64],
    right_ts: &[i64],
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_left = left_ts.len();
    let n_right = right_ts.len();

    if n_left == 0 {
        return Vec::new();
    }
    if n_right == 0 {
        return vec![None; n_left];
    }

    let num_threads = rayon::current_num_threads();
    if n_left < MIN_CHUNK_SIZE * 2 || num_threads <= 1 {
        return super::scan::merge_scan_forward(left_ts, right_ts, tolerance_ns, allow_exact);
    }

    let chunk_size = (n_left + num_threads - 1) / num_threads;
    let result: Vec<Option<usize>> = vec![None; n_left];

    let chunks: Vec<(usize, usize, usize)> = (0..n_left)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(n_left);
            let cursor = if start == 0 {
                0
            } else {
                forward_cursor_start(left_ts[start], right_ts, allow_exact)
            };
            (start, end, cursor)
        })
        .collect();

    let shared = make_shared_slice(&result);

    chunks.par_iter().for_each(|&(start, end, cursor_start)| {
        let chunk_left = &left_ts[start..end];
        let chunk_result = scan_forward_chunk(
            chunk_left,
            right_ts,
            cursor_start,
            tolerance_ns,
            allow_exact,
        );

        for (k, val) in chunk_result.into_iter().enumerate() {
            unsafe { shared.write(start + k, val); }
        }
    });

    result
}

/// Parallel nearest scan.
pub fn par_merge_scan_nearest(
    left_ts: &[i64],
    right_ts: &[i64],
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_left = left_ts.len();
    let n_right = right_ts.len();

    if n_left == 0 {
        return Vec::new();
    }
    if n_right == 0 {
        return vec![None; n_left];
    }

    let num_threads = rayon::current_num_threads();
    if n_left < MIN_CHUNK_SIZE * 2 || num_threads <= 1 {
        return super::scan::merge_scan_nearest(left_ts, right_ts, tolerance_ns, allow_exact);
    }

    let chunk_size = (n_left + num_threads - 1) / num_threads;
    let result: Vec<Option<usize>> = vec![None; n_left];

    let chunks: Vec<(usize, usize, usize)> = (0..n_left)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(n_left);
            // For nearest, start cursor at the forward position
            let cursor = if start == 0 {
                0
            } else {
                forward_cursor_start(left_ts[start], right_ts, true)
            };
            (start, end, cursor)
        })
        .collect();

    let shared = make_shared_slice(&result);

    chunks.par_iter().for_each(|&(start, end, cursor_start)| {
        let chunk_left = &left_ts[start..end];
        let chunk_result = scan_nearest_chunk(
            chunk_left,
            right_ts,
            cursor_start,
            tolerance_ns,
            allow_exact,
        );

        for (k, val) in chunk_result.into_iter().enumerate() {
            unsafe { shared.write(start + k, val); }
        }
    });

    result
}

// ---------------------------------------------------------------------------
// Cursor start computation via binary search
// ---------------------------------------------------------------------------

/// Find the starting cursor for a backward scan chunk.
/// Returns the index of the rightmost right[j] <= target, or 0 if none.
#[inline]
fn backward_cursor_start(target: i64, right_ts: &[i64], allow_exact: bool) -> usize {
    if right_ts.is_empty() {
        return 0;
    }

    // Binary search for rightmost right[j] <= target
    let mut lo: usize = 0;
    let mut hi: usize = right_ts.len();

    if allow_exact {
        // Find rightmost j where right[j] <= target
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if right_ts[mid] <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is now the first index > target, so lo-1 is our cursor
        lo.saturating_sub(1)
    } else {
        // Find rightmost j where right[j] < target
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if right_ts[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.saturating_sub(1)
    }
}

/// Find the starting cursor for a forward scan chunk.
/// Returns the index of the leftmost right[j] >= target (or > if !allow_exact).
#[inline]
fn forward_cursor_start(target: i64, right_ts: &[i64], allow_exact: bool) -> usize {
    if right_ts.is_empty() {
        return 0;
    }

    if allow_exact {
        // Find leftmost j where right[j] >= target
        right_ts.partition_point(|&v| v < target)
    } else {
        // Find leftmost j where right[j] > target
        right_ts.partition_point(|&v| v <= target)
    }
}

// ---------------------------------------------------------------------------
// Per-chunk scan kernels (no allocation of output Vec — just the chunk)
// ---------------------------------------------------------------------------

/// Scan a chunk of left timestamps backward against the full right array.
fn scan_backward_chunk(
    chunk_left: &[i64],
    right_ts: &[i64],
    mut cursor: usize,
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_right = right_ts.len();
    let mut out = Vec::with_capacity(chunk_left.len());

    for &target in chunk_left {
        // Advance cursor while next right value is still <= target
        while cursor + 1 < n_right && right_ts[cursor + 1] <= target {
            cursor += 1;
        }

        if right_ts[cursor] > target {
            out.push(None);
            continue;
        }

        if !allow_exact && right_ts[cursor] == target {
            if cursor > 0 && right_ts[cursor - 1] < target {
                if let Some(tol) = tolerance_ns {
                    if target - right_ts[cursor - 1] <= tol {
                        out.push(Some(cursor - 1));
                    } else {
                        out.push(None);
                    }
                } else {
                    out.push(Some(cursor - 1));
                }
            } else {
                out.push(None);
            }
            continue;
        }

        if let Some(tol) = tolerance_ns {
            if target - right_ts[cursor] <= tol {
                out.push(Some(cursor));
            } else {
                out.push(None);
            }
        } else {
            out.push(Some(cursor));
        }
    }

    out
}

/// Scan a chunk of left timestamps forward against the full right array.
fn scan_forward_chunk(
    chunk_left: &[i64],
    right_ts: &[i64],
    mut cursor: usize,
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_right = right_ts.len();
    let mut out = Vec::with_capacity(chunk_left.len());

    for &target in chunk_left {
        if allow_exact {
            while cursor < n_right && right_ts[cursor] < target {
                cursor += 1;
            }
        } else {
            while cursor < n_right && right_ts[cursor] <= target {
                cursor += 1;
            }
        }

        if cursor >= n_right {
            out.push(None);
            continue;
        }

        if let Some(tol) = tolerance_ns {
            if right_ts[cursor] - target <= tol {
                out.push(Some(cursor));
            } else {
                out.push(None);
            }
        } else {
            out.push(Some(cursor));
        }
    }

    out
}

/// Scan a chunk of left timestamps for nearest match against the full right array.
fn scan_nearest_chunk(
    chunk_left: &[i64],
    right_ts: &[i64],
    mut cursor: usize,
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_right = right_ts.len();
    let mut out = Vec::with_capacity(chunk_left.len());

    for &target in chunk_left {
        // Advance cursor so right[cursor] is the first value >= target
        while cursor < n_right && right_ts[cursor] < target {
            cursor += 1;
        }

        // Backward candidate
        let back = if cursor > 0 {
            let dist = target - right_ts[cursor - 1];
            if !allow_exact && dist == 0 {
                if cursor >= 2 {
                    Some((cursor - 2, target - right_ts[cursor - 2]))
                } else {
                    None
                }
            } else {
                Some((cursor - 1, dist))
            }
        } else {
            None
        };

        // Forward candidate
        let fwd = if cursor < n_right {
            let dist = right_ts[cursor] - target;
            if !allow_exact && dist == 0 {
                if cursor + 1 < n_right {
                    Some((cursor + 1, right_ts[cursor + 1] - target))
                } else {
                    None
                }
            } else {
                Some((cursor, dist))
            }
        } else {
            None
        };

        let best = match (back, fwd) {
            (Some((bi, bd)), Some((_fi, fd))) if bd <= fd => Some((bi, bd)),
            (Some(_), Some((fi, fd))) => Some((fi, fd)),
            (Some(b), None) => Some(b),
            (None, Some(f)) => Some(f),
            (None, None) => None,
        };

        if let Some((idx, dist)) = best {
            if let Some(tol) = tolerance_ns {
                if dist <= tol {
                    out.push(Some(idx));
                } else {
                    out.push(None);
                }
            } else {
                out.push(Some(idx));
            }
        } else {
            out.push(None);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Correctness: parallel results must match sequential results exactly
    // -----------------------------------------------------------------------

    fn gen_sorted_ts(n: usize, seed: u64) -> Vec<i64> {
        let mut ts = Vec::with_capacity(n);
        let mut val: i64 = seed as i64;
        for i in 0..n {
            val += 1 + ((i as i64 * 7 + seed as i64) % 10);
            ts.push(val);
        }
        ts
    }

    #[test]
    fn par_backward_matches_sequential() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_backward(&left, &right, None, true);
        let par = par_merge_scan_backward(&left, &right, None, true);
        assert_eq!(seq, par);
    }

    #[test]
    fn par_forward_matches_sequential() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_forward(&left, &right, None, true);
        let par = par_merge_scan_forward(&left, &right, None, true);
        assert_eq!(seq, par);
    }

    #[test]
    fn par_nearest_matches_sequential() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_nearest(&left, &right, None, true);
        let par = par_merge_scan_nearest(&left, &right, None, true);
        assert_eq!(seq, par);
    }

    #[test]
    fn par_backward_with_tolerance() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_backward(&left, &right, Some(50), true);
        let par = par_merge_scan_backward(&left, &right, Some(50), true);
        assert_eq!(seq, par);
    }

    #[test]
    fn par_backward_no_exact() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_backward(&left, &right, None, false);
        let par = par_merge_scan_backward(&left, &right, None, false);
        assert_eq!(seq, par);
    }

    #[test]
    fn par_forward_no_exact_with_tolerance() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_forward(&left, &right, Some(100), false);
        let par = par_merge_scan_forward(&left, &right, Some(100), false);
        assert_eq!(seq, par);
    }

    #[test]
    fn par_nearest_with_tolerance_no_exact() {
        let left = gen_sorted_ts(50_000, 42);
        let right = gen_sorted_ts(25_000, 17);
        let seq = super::super::scan::merge_scan_nearest(&left, &right, Some(20), false);
        let par = par_merge_scan_nearest(&left, &right, Some(20), false);
        assert_eq!(seq, par);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn par_empty_left() {
        let result = par_merge_scan_backward(&[], &[10, 20], None, true);
        assert!(result.is_empty());
    }

    #[test]
    fn par_empty_right() {
        let result = par_merge_scan_backward(&[10, 20], &[], None, true);
        assert_eq!(result, vec![None, None]);
    }

    #[test]
    fn par_small_input_uses_sequential() {
        // Below MIN_CHUNK_SIZE * 2, should fall through to sequential
        let left = gen_sorted_ts(100, 42);
        let right = gen_sorted_ts(50, 17);
        let seq = super::super::scan::merge_scan_backward(&left, &right, None, true);
        let par = par_merge_scan_backward(&left, &right, None, true);
        assert_eq!(seq, par);
    }

    // -----------------------------------------------------------------------
    // Binary search helpers
    // -----------------------------------------------------------------------

    #[test]
    fn backward_cursor_start_basic() {
        let right = vec![10, 20, 30, 40, 50];
        // Target 25: rightmost <= 25 is index 1 (value 20)
        assert_eq!(backward_cursor_start(25, &right, true), 1);
        // Target 30: rightmost <= 30 is index 2 (value 30)
        assert_eq!(backward_cursor_start(30, &right, true), 2);
        // Target 5: nothing <= 5, returns 0 (saturating)
        assert_eq!(backward_cursor_start(5, &right, true), 0);
    }

    #[test]
    fn forward_cursor_start_basic() {
        let right = vec![10, 20, 30, 40, 50];
        // Target 25: leftmost >= 25 is index 2 (value 30)
        assert_eq!(forward_cursor_start(25, &right, true), 2);
        // Target 30: leftmost >= 30 is index 2 (value 30)
        assert_eq!(forward_cursor_start(30, &right, true), 2);
        // Target 30 no exact: leftmost > 30 is index 3 (value 40)
        assert_eq!(forward_cursor_start(30, &right, false), 3);
    }
}
