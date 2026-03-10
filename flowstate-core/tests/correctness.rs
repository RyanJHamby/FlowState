//! Property-based correctness tests using proptest.
//!
//! Validates invariants that must hold for ANY input, not just hand-picked cases:
//! - Output length always equals left input length
//! - Backward matches are always <= left timestamp
//! - Forward matches are always >= left timestamp
//! - Nearest matches are the true minimum distance
//! - Tolerance is strictly enforced
//! - No-exact-match correctly excludes equal timestamps

use flowstate_core::asof::scan::{merge_scan_backward, merge_scan_forward, merge_scan_nearest};
use proptest::prelude::*;

/// Generate sorted timestamp vectors for property testing.
fn sorted_timestamps(max_len: usize) -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(0i64..1_000_000, 0..max_len).prop_map(|mut v| {
        v.sort_unstable();
        v
    })
}

proptest! {
    /// Output length always equals left input length.
    #[test]
    fn backward_output_length(
        left in sorted_timestamps(500),
        right in sorted_timestamps(500),
    ) {
        let result = merge_scan_backward(&left, &right, None, true);
        prop_assert_eq!(result.len(), left.len());
    }

    /// Every backward match has right[j] <= left[i].
    #[test]
    fn backward_match_is_le(
        left in sorted_timestamps(500),
        right in sorted_timestamps(500),
    ) {
        let result = merge_scan_backward(&left, &right, None, true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert!(
                    right[*j] <= left[i],
                    "backward: right[{}]={} > left[{}]={}",
                    j, right[*j], i, left[i]
                );
            }
        }
    }

    /// Backward match is the rightmost right[j] <= left[i].
    #[test]
    fn backward_match_is_rightmost(
        left in sorted_timestamps(200),
        right in sorted_timestamps(200),
    ) {
        let result = merge_scan_backward(&left, &right, None, true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                // No right element after j should also be <= left[i]
                for k in (*j + 1)..right.len() {
                    prop_assert!(
                        right[k] > left[i],
                        "backward: right[{}]={} <= left[{}]={} but matched j={}",
                        k, right[k], i, left[i], j
                    );
                }
            }
        }
    }

    /// Every forward match has right[j] >= left[i].
    #[test]
    fn forward_match_is_ge(
        left in sorted_timestamps(500),
        right in sorted_timestamps(500),
    ) {
        let result = merge_scan_forward(&left, &right, None, true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert!(
                    right[*j] >= left[i],
                    "forward: right[{}]={} < left[{}]={}",
                    j, right[*j], i, left[i]
                );
            }
        }
    }

    /// Forward match is the leftmost right[j] >= left[i].
    #[test]
    fn forward_match_is_leftmost(
        left in sorted_timestamps(200),
        right in sorted_timestamps(200),
    ) {
        let result = merge_scan_forward(&left, &right, None, true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                // No right element before j should also be >= left[i]
                if *j > 0 {
                    for k in 0..*j {
                        prop_assert!(
                            right[k] < left[i],
                            "forward: right[{}]={} >= left[{}]={} but matched j={}",
                            k, right[k], i, left[i], j
                        );
                    }
                }
            }
        }
    }

    /// Nearest match minimizes |right[j] - left[i]|.
    #[test]
    fn nearest_minimizes_distance(
        left in sorted_timestamps(200),
        right in sorted_timestamps(200),
    ) {
        let result = merge_scan_nearest(&left, &right, None, true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                let matched_dist = (right[*j] - left[i]).unsigned_abs();
                // No other right element should be strictly closer
                for (k, &rt) in right.iter().enumerate() {
                    let other_dist = (rt - left[i]).unsigned_abs();
                    prop_assert!(
                        matched_dist <= other_dist,
                        "nearest: right[{}] dist={} < matched right[{}] dist={} for left[{}]={}",
                        k, other_dist, j, matched_dist, i, left[i]
                    );
                }
            } else if !right.is_empty() {
                // If no match with no tolerance, something is wrong
                prop_assert!(false, "nearest returned None with non-empty right for left[{}]={}", i, left[i]);
            }
        }
    }

    /// Tolerance is strictly enforced for backward.
    #[test]
    fn backward_tolerance_enforced(
        left in sorted_timestamps(300),
        right in sorted_timestamps(300),
        tol in 1i64..10_000,
    ) {
        let result = merge_scan_backward(&left, &right, Some(tol), true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                let dist = left[i] - right[*j];
                prop_assert!(
                    dist <= tol,
                    "backward tolerance: dist {} > tol {} at i={}, j={}",
                    dist, tol, i, j
                );
            }
        }
    }

    /// Tolerance is strictly enforced for forward.
    #[test]
    fn forward_tolerance_enforced(
        left in sorted_timestamps(300),
        right in sorted_timestamps(300),
        tol in 1i64..10_000,
    ) {
        let result = merge_scan_forward(&left, &right, Some(tol), true);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                let dist = right[*j] - left[i];
                prop_assert!(
                    dist <= tol,
                    "forward tolerance: dist {} > tol {} at i={}, j={}",
                    dist, tol, i, j
                );
            }
        }
    }

    /// No-exact-match excludes equal timestamps in backward.
    #[test]
    fn backward_no_exact_excludes_equal(
        left in sorted_timestamps(300),
        right in sorted_timestamps(300),
    ) {
        let result = merge_scan_backward(&left, &right, None, false);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert!(
                    right[*j] < left[i],
                    "backward no-exact: right[{}]={} == left[{}]={}",
                    j, right[*j], i, left[i]
                );
            }
        }
    }

    /// No-exact-match excludes equal timestamps in forward.
    #[test]
    fn forward_no_exact_excludes_equal(
        left in sorted_timestamps(300),
        right in sorted_timestamps(300),
    ) {
        let result = merge_scan_forward(&left, &right, None, false);
        for (i, opt_j) in result.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert!(
                    right[*j] > left[i],
                    "forward no-exact: right[{}]={} == left[{}]={}",
                    j, right[*j], i, left[i]
                );
            }
        }
    }

    /// Backward + forward together cover at least as many matches as nearest.
    #[test]
    fn nearest_matches_at_least_backward_or_forward(
        left in sorted_timestamps(200),
        right in sorted_timestamps(200),
    ) {
        let backward = merge_scan_backward(&left, &right, None, true);
        let forward = merge_scan_forward(&left, &right, None, true);
        let nearest = merge_scan_nearest(&left, &right, None, true);

        let bw_count = backward.iter().filter(|o| o.is_some()).count();
        let fw_count = forward.iter().filter(|o| o.is_some()).count();
        let nr_count = nearest.iter().filter(|o| o.is_some()).count();

        // Nearest should match at least as many as the better of backward/forward
        prop_assert!(
            nr_count >= bw_count.max(fw_count),
            "nearest {} < max(backward {}, forward {})",
            nr_count, bw_count, fw_count
        );
    }
}
