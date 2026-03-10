//! Core merge-scan kernels for as-of join matching.
//!
//! All kernels operate on sorted i64 timestamp slices and return index arrays.
//! O(n+m) for backward/forward, O(n+m) for nearest (two-pointer).

use super::config::{AsOfDirection, RustAsOfConfig};

/// Dispatch to the appropriate scan kernel based on config.
pub fn merge_scan(
    left_ts: &[i64],
    right_ts: &[i64],
    config: &RustAsOfConfig,
) -> Vec<Option<usize>> {
    match config.direction {
        AsOfDirection::Backward => {
            merge_scan_backward(left_ts, right_ts, config.tolerance_ns, config.allow_exact_match)
        }
        AsOfDirection::Forward => {
            merge_scan_forward(left_ts, right_ts, config.tolerance_ns, config.allow_exact_match)
        }
        AsOfDirection::Nearest => {
            merge_scan_nearest(left_ts, right_ts, config.tolerance_ns, config.allow_exact_match)
        }
    }
}

/// Backward scan: for each left[i], find rightmost right[j] where right[j] <= left[i].
///
/// Single-pass O(n+m) with a monotonically advancing cursor.
pub fn merge_scan_backward(
    left_ts: &[i64],
    right_ts: &[i64],
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_left = left_ts.len();
    let n_right = right_ts.len();
    let mut indices: Vec<Option<usize>> = vec![None; n_left];

    if n_right == 0 {
        return indices;
    }

    let mut j: usize = 0;

    for i in 0..n_left {
        let target = left_ts[i];

        // Advance cursor while next right value is still <= target
        while j + 1 < n_right && right_ts[j + 1] <= target {
            j += 1;
        }

        if right_ts[j] > target {
            continue;
        }

        if !allow_exact && right_ts[j] == target {
            // Need strictly less than — walk back to find one
            if j > 0 && right_ts[j - 1] < target {
                // Check tolerance on the previous element
                if let Some(tol) = tolerance_ns {
                    if target - right_ts[j - 1] <= tol {
                        indices[i] = Some(j - 1);
                    }
                } else {
                    indices[i] = Some(j - 1);
                }
            }
            continue;
        }

        // Valid match: right[j] <= target (or < if !allow_exact)
        if let Some(tol) = tolerance_ns {
            if target - right_ts[j] <= tol {
                indices[i] = Some(j);
            }
        } else {
            indices[i] = Some(j);
        }
    }

    indices
}

/// Forward scan: for each left[i], find leftmost right[j] where right[j] >= left[i].
///
/// O(n+m) with a monotonically advancing cursor.
pub fn merge_scan_forward(
    left_ts: &[i64],
    right_ts: &[i64],
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_left = left_ts.len();
    let n_right = right_ts.len();
    let mut indices: Vec<Option<usize>> = vec![None; n_left];

    if n_right == 0 {
        return indices;
    }

    let mut j: usize = 0;

    for i in 0..n_left {
        let target = left_ts[i];

        if allow_exact {
            // Advance past values < target
            while j < n_right && right_ts[j] < target {
                j += 1;
            }
        } else {
            // Advance past values <= target (strict)
            while j < n_right && right_ts[j] <= target {
                j += 1;
            }
        }

        if j >= n_right {
            continue;
        }

        // right[j] >= target (or > if !allow_exact)
        if let Some(tol) = tolerance_ns {
            if right_ts[j] - target <= tol {
                indices[i] = Some(j);
            }
        } else {
            indices[i] = Some(j);
        }
    }

    indices
}

/// Nearest scan: for each left[i], find closest right[j] in either direction.
///
/// Two-pointer approach: maintains a backward candidate and a forward candidate,
/// picks the one with minimum absolute distance.
pub fn merge_scan_nearest(
    left_ts: &[i64],
    right_ts: &[i64],
    tolerance_ns: Option<i64>,
    allow_exact: bool,
) -> Vec<Option<usize>> {
    let n_left = left_ts.len();
    let n_right = right_ts.len();
    let mut indices: Vec<Option<usize>> = vec![None; n_left];

    if n_right == 0 {
        return indices;
    }

    let mut j: usize = 0;

    for i in 0..n_left {
        let target = left_ts[i];

        // Advance j so right[j] is the first value >= target
        while j < n_right && right_ts[j] < target {
            j += 1;
        }

        // Backward candidate: right[j-1] (if exists, < target)
        let back = if j > 0 {
            let dist = target - right_ts[j - 1];
            if !allow_exact && dist == 0 {
                // Check j-2
                if j >= 2 {
                    Some((j - 2, target - right_ts[j - 2]))
                } else {
                    None
                }
            } else {
                Some((j - 1, dist))
            }
        } else {
            None
        };

        // Forward candidate: right[j] (if exists, >= target)
        let fwd = if j < n_right {
            let dist = right_ts[j] - target;
            if !allow_exact && dist == 0 {
                if j + 1 < n_right {
                    Some((j + 1, right_ts[j + 1] - target))
                } else {
                    None
                }
            } else {
                Some((j, dist))
            }
        } else {
            None
        };

        // Pick minimum distance
        let best = match (back, fwd) {
            (Some((bi, bd)), Some((fi, fd))) => {
                if bd <= fd {
                    Some((bi, bd))
                } else {
                    Some((fi, fd))
                }
            }
            (Some(b), None) => Some(b),
            (None, Some(f)) => Some(f),
            (None, None) => None,
        };

        if let Some((idx, dist)) = best {
            if let Some(tol) = tolerance_ns {
                if dist <= tol {
                    indices[i] = Some(idx);
                }
            } else {
                indices[i] = Some(idx);
            }
        }
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Backward
    // -----------------------------------------------------------------------

    #[test]
    fn backward_basic() {
        let left = vec![10, 20, 30, 40, 50];
        let right = vec![5, 15, 25, 35, 45];
        let result = merge_scan_backward(&left, &right, None, true);
        assert_eq!(result, vec![Some(0), Some(1), Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn backward_exact_matches() {
        let left = vec![10, 20, 30];
        let right = vec![10, 20, 30];
        let result = merge_scan_backward(&left, &right, None, true);
        assert_eq!(result, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn backward_no_matches() {
        let left = vec![1, 2, 3];
        let right = vec![10, 20, 30];
        let result = merge_scan_backward(&left, &right, None, true);
        assert_eq!(result, vec![None, None, None]);
    }

    #[test]
    fn backward_empty_right() {
        let result = merge_scan_backward(&[10, 20], &[], None, true);
        assert_eq!(result, vec![None, None]);
    }

    #[test]
    fn backward_empty_left() {
        let result = merge_scan_backward(&[], &[10, 20], None, true);
        assert!(result.is_empty());
    }

    #[test]
    fn backward_tolerance() {
        let left = vec![10, 20, 100];
        let right = vec![5, 15];
        let result = merge_scan_backward(&left, &right, Some(10), true);
        assert_eq!(result, vec![Some(0), Some(1), None]);
    }

    #[test]
    fn backward_duplicates() {
        let left = vec![10, 10, 10];
        let right = vec![5, 10, 10, 15];
        let result = merge_scan_backward(&left, &right, None, true);
        assert_eq!(result, vec![Some(2), Some(2), Some(2)]);
    }

    #[test]
    fn backward_rightmost_match() {
        let left = vec![50];
        let right = vec![10, 20, 30, 40, 45];
        let result = merge_scan_backward(&left, &right, None, true);
        assert_eq!(result, vec![Some(4)]);
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    #[test]
    fn forward_basic() {
        let left = vec![10, 20, 30];
        let right = vec![15, 25, 35];
        let result = merge_scan_forward(&left, &right, None, true);
        assert_eq!(result, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn forward_exact_matches() {
        let left = vec![10, 20, 30];
        let right = vec![10, 20, 30];
        let result = merge_scan_forward(&left, &right, None, true);
        assert_eq!(result, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn forward_no_matches() {
        let left = vec![40, 50, 60];
        let right = vec![10, 20, 30];
        let result = merge_scan_forward(&left, &right, None, true);
        assert_eq!(result, vec![None, None, None]);
    }

    #[test]
    fn forward_tolerance() {
        let left = vec![10, 20, 100];
        let right = vec![15, 25];
        let result = merge_scan_forward(&left, &right, Some(10), true);
        assert_eq!(result, vec![Some(0), Some(1), None]);
    }

    #[test]
    fn forward_no_exact() {
        let left = vec![10, 20];
        let right = vec![10, 25];
        let result = merge_scan_forward(&left, &right, None, false);
        // left[0]=10: right[0]=10 is exact → skip, right[1]=25 is next
        assert_eq!(result, vec![Some(1), Some(1)]);
    }

    // -----------------------------------------------------------------------
    // Nearest
    // -----------------------------------------------------------------------

    #[test]
    fn nearest_basic() {
        let left = vec![12, 23, 37];
        let right = vec![10, 20, 30, 40];
        let result = merge_scan_nearest(&left, &right, None, true);
        // 12→10 (dist 2), 23→20 (dist 3, vs 30 dist 7), 37→40 (dist 3, vs 30 dist 7)
        assert_eq!(result, vec![Some(0), Some(1), Some(3)]);
    }

    #[test]
    fn nearest_exact() {
        let left = vec![10, 20, 30];
        let right = vec![10, 20, 30];
        let result = merge_scan_nearest(&left, &right, None, true);
        assert_eq!(result, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn nearest_tolerance() {
        let left = vec![10, 50, 100];
        let right = vec![12, 48];
        let result = merge_scan_nearest(&left, &right, Some(5), true);
        assert_eq!(result, vec![Some(0), Some(1), None]);
    }

    #[test]
    fn nearest_prefers_backward_on_tie() {
        let left = vec![15];
        let right = vec![10, 20];
        let result = merge_scan_nearest(&left, &right, None, true);
        // dist to 10 = 5, dist to 20 = 5 → tie, prefer backward
        assert_eq!(result, vec![Some(0)]);
    }
}
