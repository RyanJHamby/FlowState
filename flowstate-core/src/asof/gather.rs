//! Column gathering: take right-side values by index and append to left batch.

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{ArrowError, Field, Schema};
use arrow_select::take::take;
use rayon::prelude::*;
use std::sync::Arc;

/// Gather columns from the right table using match indices and append to left.
///
/// For each right-side column (excluding `on` and `by`), takes values at the
/// matched indices and appends as a new column to the left batch. Unmatched
/// rows (None in indices) become nulls.
///
/// Uses rayon to parallelize take() across multiple right columns.
pub fn gather_and_append(
    left: &RecordBatch,
    right: &RecordBatch,
    indices: &[Option<usize>],
    on: &str,
    by: Option<&str>,
    right_prefix: &str,
) -> Result<RecordBatch, ArrowError> {
    let take_indices: UInt64Array = indices
        .iter()
        .map(|opt| opt.map(|v| v as u64))
        .collect();

    let left_schema = left.schema();
    let right_schema = right.schema();

    // Pre-compute which right columns to include
    let right_col_indices: Vec<usize> = (0..right.num_columns())
        .filter(|&i| {
            let name = right_schema.field(i).name();
            name != on && by.map_or(true, |b| name != b)
        })
        .collect();

    // Parallel take across right columns (significant for wide tables)
    let gathered: Vec<Result<(Arc<Field>, Arc<dyn Array>), ArrowError>> = right_col_indices
        .par_iter()
        .map(|&i| {
            let field = right_schema.field(i);
            let gathered = take(right.column(i).as_ref(), &take_indices, None)?;

            let output_name = if right_prefix.is_empty() {
                field.name().clone()
            } else {
                format!("{}{}", right_prefix, field.name())
            };

            let new_field = Arc::new(Field::new(
                &output_name,
                field.data_type().clone(),
                true,
            ));
            Ok((new_field, gathered))
        })
        .collect();

    // Pre-allocate with exact capacity
    let total_cols = left.num_columns() + right_col_indices.len();
    let mut fields: Vec<Arc<Field>> = Vec::with_capacity(total_cols);
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(total_cols);

    // Copy left columns
    for f in left_schema.fields().iter() {
        fields.push(f.clone());
    }
    for i in 0..left.num_columns() {
        columns.push(left.column(i).clone());
    }

    // Append gathered right columns
    for result in gathered {
        let (field, column) = result?;
        fields.push(field);
        columns.push(column);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array, StringArray};
    use arrow_schema::DataType;

    #[test]
    fn test_gather_basic() {
        let left = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("price", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![10, 20, 30])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();

        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("quote", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![5, 15, 25])),
                Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5])),
            ],
        )
        .unwrap();

        let indices = vec![Some(0), Some(1), Some(2)];
        let result = gather_and_append(&left, &right, &indices, "timestamp", None, "q_").unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.schema().field(2).name(), "q_quote");
    }

    #[test]
    fn test_gather_with_nulls() {
        let left = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
            ])),
            vec![Arc::new(Int64Array::from(vec![10, 20, 30]))],
        )
        .unwrap();

        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("val", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![5, 15])),
                Arc::new(Float64Array::from(vec![0.5, 1.5])),
            ],
        )
        .unwrap();

        let indices = vec![Some(0), Some(1), None];
        let result = gather_and_append(&left, &right, &indices, "timestamp", None, "").unwrap();

        assert_eq!(result.num_rows(), 3);
        let val_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(val_col.is_valid(0));
        assert!(val_col.is_valid(1));
        assert!(val_col.is_null(2));
    }

    #[test]
    fn test_gather_skips_by_column() {
        let left = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("symbol", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![10])),
                Arc::new(StringArray::from(vec!["AAPL"])),
            ],
        )
        .unwrap();

        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("symbol", DataType::Utf8, false),
                Field::new("bid", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![5])),
                Arc::new(StringArray::from(vec!["AAPL"])),
                Arc::new(Float64Array::from(vec![150.0])),
            ],
        )
        .unwrap();

        let indices = vec![Some(0)];
        let result = gather_and_append(&left, &right, &indices, "timestamp", Some("symbol"), "q_").unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.schema().field(2).name(), "q_bid");
    }

    #[test]
    fn test_no_prefix() {
        let left = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
            ])),
            vec![Arc::new(Int64Array::from(vec![10]))],
        )
        .unwrap();

        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("bid", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![5])),
                Arc::new(Float64Array::from(vec![100.0])),
            ],
        )
        .unwrap();

        let indices = vec![Some(0)];
        let result = gather_and_append(&left, &right, &indices, "timestamp", None, "").unwrap();
        assert_eq!(result.schema().field(1).name(), "bid");
    }
}
