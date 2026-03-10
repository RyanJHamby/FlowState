//! As-of join orchestration: concat, sort, group, dispatch to scan kernels.

use ahash::AHashMap;
use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_cast::cast;
use arrow_ord::sort::{sort_to_indices, SortOptions};
use arrow_schema::{ArrowError, DataType, SchemaRef};
use arrow_select::take::take;
use rayon::prelude::*;
use std::sync::Arc;

use super::config::{AsOfDirection, RustAsOfConfig};
use super::gather::gather_and_append;
use super::parallel_scan::{par_merge_scan_backward, par_merge_scan_forward, par_merge_scan_nearest};
use super::scan::merge_scan;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate multiple RecordBatches into one.
fn concat_batches(
    batches: &[RecordBatch],
    schema: &SchemaRef,
) -> Result<RecordBatch, ArrowError> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    if batches.len() == 1 {
        return Ok(batches[0].clone());
    }
    arrow_select::concat::concat_batches(schema, batches)
}

/// Check if an i64 slice is sorted ascending (O(n) but branchless-friendly).
#[inline]
fn is_sorted_ascending(ts: &[i64]) -> bool {
    ts.windows(2).all(|w| w[0] <= w[1])
}

/// Extract timestamp column as i64 slice reference or owned Vec.
/// Returns the batch (possibly sorted) and a reference-or-owned timestamp slice.
///
/// Avoids copying timestamps when the data is already Int64 and sorted —
/// the common case for time-series data.
fn prepare_sorted(
    batch: &RecordBatch,
    on: &str,
) -> Result<(RecordBatch, TimestampSlice), ArrowError> {
    let col_idx = batch
        .schema()
        .index_of(on)
        .map_err(|_| ArrowError::InvalidArgumentError(format!("Column '{}' not found", on)))?;

    let ts_col = batch.column(col_idx);
    let needs_cast = matches!(ts_col.data_type(), DataType::Timestamp(_, _));

    // Cast timestamp → i64 only if needed
    let ts_i64 = if needs_cast {
        cast(ts_col, &DataType::Int64)?
    } else {
        ts_col.clone()
    };

    let ts_array = ts_i64
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ArrowError::InvalidArgumentError("Timestamp column is not Int64".into()))?;

    // Zero-copy: borrow the underlying buffer directly
    let ts_values = ts_array.values();

    // Fast path: skip sort if already ordered
    if is_sorted_ascending(ts_values) {
        return Ok((batch.clone(), TimestampSlice::Borrowed(ts_array.clone())));
    }

    // Sort needed
    let ts_vec: Vec<i64> = ts_values.to_vec();
    let sort_indices = sort_to_indices(&ts_i64, Some(SortOptions::default()), None)?;
    let sorted_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &sort_indices, None))
        .collect::<Result<_, _>>()?;

    let sorted_batch = RecordBatch::try_new(batch.schema(), sorted_columns)?;
    let sorted_ts: Vec<i64> = sort_indices
        .values()
        .iter()
        .map(|&idx| ts_vec[idx as usize])
        .collect();

    Ok((sorted_batch, TimestampSlice::Owned(sorted_ts)))
}

/// Zero-copy timestamp access: borrows Arrow buffer when possible, owns Vec when sorted.
enum TimestampSlice {
    Borrowed(Int64Array),
    Owned(Vec<i64>),
}

impl TimestampSlice {
    fn as_slice(&self) -> &[i64] {
        match self {
            TimestampSlice::Borrowed(arr) => arr.values(),
            TimestampSlice::Owned(vec) => vec,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Execute an as-of join (backward, forward, or nearest) with optional grouping.
pub fn asof_join_impl(
    left_batches: &[RecordBatch],
    left_schema: &SchemaRef,
    right_batches: &[RecordBatch],
    right_schema: &SchemaRef,
    on: &str,
    by: Option<&str>,
    config: &RustAsOfConfig,
) -> Result<Vec<RecordBatch>, ArrowError> {
    let left = concat_batches(left_batches, left_schema)?;
    let right = concat_batches(right_batches, right_schema)?;

    if left.num_rows() == 0 {
        return Ok(vec![left]);
    }

    let (left_sorted, left_ts_slice) = prepare_sorted(&left, on)?;

    if right.num_rows() == 0 {
        let result = gather_and_append(
            &left_sorted,
            &right,
            &vec![None; left.num_rows()],
            on,
            by,
            &config.right_prefix,
        )?;
        return Ok(vec![result]);
    }

    let (right_sorted, right_ts_slice) = prepare_sorted(&right, on)?;

    match by {
        Some(by_col) => grouped_join(
            &left_sorted,
            left_ts_slice.as_slice(),
            &right_sorted,
            right_ts_slice.as_slice(),
            on,
            by_col,
            config,
        ),
        None => ungrouped_join(
            &left_sorted,
            left_ts_slice.as_slice(),
            &right_sorted,
            right_ts_slice.as_slice(),
            on,
            config,
        ),
    }
}

// ---------------------------------------------------------------------------
// Ungrouped
// ---------------------------------------------------------------------------

fn ungrouped_join(
    left: &RecordBatch,
    left_ts: &[i64],
    right: &RecordBatch,
    right_ts: &[i64],
    on: &str,
    config: &RustAsOfConfig,
) -> Result<Vec<RecordBatch>, ArrowError> {
    // Use parallel chunked scan for large ungrouped joins
    let indices = parallel_merge_scan(left_ts, right_ts, config);
    let result = gather_and_append(left, right, &indices, on, None, &config.right_prefix)?;
    Ok(vec![result])
}

/// Dispatch to parallel scan kernels for ungrouped joins.
/// The parallel versions internally fall back to sequential for small inputs.
fn parallel_merge_scan(
    left_ts: &[i64],
    right_ts: &[i64],
    config: &RustAsOfConfig,
) -> Vec<Option<usize>> {
    match config.direction {
        AsOfDirection::Backward => {
            par_merge_scan_backward(left_ts, right_ts, config.tolerance_ns, config.allow_exact_match)
        }
        AsOfDirection::Forward => {
            par_merge_scan_forward(left_ts, right_ts, config.tolerance_ns, config.allow_exact_match)
        }
        AsOfDirection::Nearest => {
            par_merge_scan_nearest(left_ts, right_ts, config.tolerance_ns, config.allow_exact_match)
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped with rayon parallelism
// ---------------------------------------------------------------------------

/// Group metadata: row indices and their corresponding timestamps.
struct GroupData {
    row_indices: Vec<usize>,
    timestamps: Vec<i64>,
}

/// Build groups using borrowed &str keys (zero-copy from StringArray) and ahash.
fn build_groups<'a>(
    syms: &'a StringArray,
    all_ts: &[i64],
) -> AHashMap<&'a str, GroupData> {
    let n = syms.len();
    // Estimate ~sqrt(n) unique groups, capped at n
    let est_groups = (n as f64).sqrt() as usize;
    let est_per_group = n / est_groups.max(1);
    let mut groups: AHashMap<&'a str, GroupData> = AHashMap::with_capacity(est_groups);

    for i in 0..n {
        if syms.is_null(i) {
            continue;
        }
        let sym = syms.value(i);
        let entry = groups.entry(sym).or_insert_with(|| GroupData {
            row_indices: Vec::with_capacity(est_per_group),
            timestamps: Vec::with_capacity(est_per_group),
        });
        entry.row_indices.push(i);
        entry.timestamps.push(all_ts[i]);
    }
    groups
}

fn grouped_join(
    left: &RecordBatch,
    left_ts: &[i64],
    right: &RecordBatch,
    right_ts: &[i64],
    on: &str,
    by: &str,
    config: &RustAsOfConfig,
) -> Result<Vec<RecordBatch>, ArrowError> {
    let left_schema = left.schema();
    let right_schema = right.schema();

    let left_by_idx = left_schema
        .index_of(by)
        .map_err(|_| ArrowError::InvalidArgumentError(format!("Column '{}' not found in left", by)))?;
    let right_by_idx = right_schema
        .index_of(by)
        .map_err(|_| ArrowError::InvalidArgumentError(format!("Column '{}' not found in right", by)))?;

    let left_syms = left
        .column(left_by_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::InvalidArgumentError("Left group column must be Utf8".into()))?;
    let right_syms = right
        .column(right_by_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::InvalidArgumentError("Right group column must be Utf8".into()))?;

    // Build group indices — zero-copy &str keys, ahash for fast hashing
    let left_groups = build_groups(left_syms, left_ts);
    let right_groups = build_groups(right_syms, right_ts);

    // Collect matching groups as (key, left_data, right_data)
    let matching: Vec<(&&str, &GroupData, &GroupData)> = left_groups
        .iter()
        .filter_map(|(sym, lg)| right_groups.get(sym).map(|rg| (sym, lg, rg)))
        .collect();

    // Pre-allocate global indices
    let global_indices: Vec<Option<usize>> = vec![None; left.num_rows()];

    // Parallel per-group merge scan — write directly to global_indices.
    // Safety: each group writes to disjoint left row indices, so no data races.
    // Use UnsafeCell to allow interior mutability across threads.
    use std::cell::UnsafeCell;
    struct UnsafeSlice<'a>(&'a [UnsafeCell<Option<usize>>]);
    unsafe impl<'a> Send for UnsafeSlice<'a> {}
    unsafe impl<'a> Sync for UnsafeSlice<'a> {}

    impl<'a> UnsafeSlice<'a> {
        unsafe fn write(&self, idx: usize, val: Option<usize>) {
            *self.0[idx].get() = val;
        }
    }

    // Transmute the slice to UnsafeCell slice (same layout guaranteed)
    let global_slice = unsafe {
        std::slice::from_raw_parts(
            global_indices.as_ptr() as *const UnsafeCell<Option<usize>>,
            global_indices.len(),
        )
    };
    let shared = UnsafeSlice(global_slice);

    matching.par_iter().for_each(|(_, left_group, right_group)| {
        let local_indices = merge_scan(
            &left_group.timestamps,
            &right_group.timestamps,
            config,
        );

        for (local_left, opt_local_right) in local_indices.into_iter().enumerate() {
            if let Some(local_right) = opt_local_right {
                let global_left = left_group.row_indices[local_left];
                let global_right = right_group.row_indices[local_right];
                unsafe { shared.write(global_left, Some(global_right)); }
            }
        }
    });

    let result = gather_and_append(left, right, &global_indices, on, Some(by), &config.right_prefix)?;
    Ok(vec![result])
}
