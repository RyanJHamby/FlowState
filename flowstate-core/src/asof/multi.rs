//! Multi-stream temporal alignment: join N secondary streams onto a primary timeline.
//!
//! Unlike sequential Python alignment, this performs all independent joins in
//! parallel using rayon, then merges columns in a single pass.

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Field, Schema, SchemaRef};
use rayon::prelude::*;
use std::sync::Arc;

use super::config::RustAsOfConfig;
use super::join::asof_join_impl;

/// Specification for a single secondary stream in multi-stream alignment.
pub struct StreamSpec {
    pub batches: Vec<RecordBatch>,
    pub schema: SchemaRef,
    pub config: RustAsOfConfig,
    pub on: String,
    pub by: Option<String>,
}

/// Align multiple secondary streams onto a primary timeline in parallel.
///
/// Each secondary stream is independently as-of joined against the primary.
/// All joins run concurrently via rayon, then results are merged column-wise.
pub fn align_streams_impl(
    primary_batches: &[RecordBatch],
    primary_schema: &SchemaRef,
    streams: Vec<StreamSpec>,
    primary_on: &str,
    primary_by: Option<&str>,
) -> Result<Vec<RecordBatch>, ArrowError> {
    if streams.is_empty() {
        return Ok(primary_batches.to_vec());
    }

    // Run all joins in parallel — each produces a Vec<RecordBatch>
    let join_results: Vec<Result<Vec<RecordBatch>, ArrowError>> = streams
        .par_iter()
        .map(|spec| {
            let on = if spec.on.is_empty() { primary_on } else { &spec.on };
            let by = spec.by.as_deref().or(primary_by);

            asof_join_impl(
                primary_batches,
                primary_schema,
                &spec.batches,
                &spec.schema,
                on,
                by,
                &spec.config,
            )
        })
        .collect();

    // Check for errors
    let mut joined_tables: Vec<RecordBatch> = Vec::with_capacity(streams.len());
    for result in join_results {
        let batches = result?;
        if !batches.is_empty() {
            joined_tables.push(batches.into_iter().next().unwrap());
        }
    }

    if joined_tables.is_empty() {
        return Ok(primary_batches.to_vec());
    }

    // Merge: start with the primary columns, then append new columns from each join.
    // Each join result contains primary cols + that stream's right cols.
    // We extract only the new (right-side) columns from each join result.
    let primary_col_count = primary_schema.fields().len();
    let first = &joined_tables[0];

    // Build merged schema and columns
    let mut fields: Vec<Arc<Field>> = Vec::new();
    let mut columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();

    // Primary columns from first join result (they're all identical)
    for i in 0..primary_col_count {
        fields.push(first.schema().field(i).clone().into());
        columns.push(first.column(i).clone());
    }

    // Right-side columns from each join
    for table in &joined_tables {
        let schema = table.schema();
        for i in primary_col_count..table.num_columns() {
            fields.push(schema.field(i).clone().into());
            columns.push(table.column(i).clone());
        }
    }

    let merged_schema = Arc::new(Schema::new(fields));
    let merged = RecordBatch::try_new(merged_schema, columns)?;
    Ok(vec![merged])
}
