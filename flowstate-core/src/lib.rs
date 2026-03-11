//! FlowState Core: high-performance temporal join engine.
//!
//! Provides Rust-accelerated as-of joins exposed to Python via PyO3.
//! Accepts and returns PyArrow tables through the Arrow C Data Interface
//! for zero-copy data exchange.

pub mod asof;
pub mod bloom;
pub mod coalesce;
pub mod hdr;
pub mod ipc;
pub mod pinned;
pub mod pipeline;
pub mod pool;
pub mod spsc;

// ---------------------------------------------------------------------------
// Python bindings (only compiled with the "python" feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use pyo3_arrow::error::PyArrowResult;
    use pyo3_arrow::PyTable;

    use crate::asof::config::{AsOfDirection, RustAsOfConfig};
    use crate::asof::multi::StreamSpec;
    use crate::asof::streaming::StreamingAsOfJoin;

    /// Echo a PyArrow table through Rust and back — validates zero-copy round-trip.
    #[pyfunction]
    pub fn echo_table<'py>(py: Python<'py>, table: PyTable) -> PyArrowResult<Bound<'py, PyAny>> {
        let (batches, schema) = table.into_inner();
        let output = PyTable::try_new(batches, schema)?;
        Ok(output.into_pyarrow(py)?)
    }

    /// Return the number of rows in a PyArrow table (processed in Rust).
    #[pyfunction]
    pub fn row_count(table: PyTable) -> PyArrowResult<usize> {
        let (batches, _schema) = table.into_inner();
        Ok(batches.iter().map(|b| b.num_rows()).sum())
    }

    /// As-of join with configurable direction (backward, forward, nearest).
    #[pyfunction]
    #[pyo3(signature = (left, right, on="timestamp", by=None, tolerance_ns=None, right_prefix="", direction="backward", allow_exact_match=true))]
    pub fn asof_join<'py>(
        py: Python<'py>,
        left: PyTable,
        right: PyTable,
        on: &str,
        by: Option<&str>,
        tolerance_ns: Option<i64>,
        right_prefix: &str,
        direction: &str,
        allow_exact_match: bool,
    ) -> PyArrowResult<Bound<'py, PyAny>> {
        let dir = match direction {
            "backward" => AsOfDirection::Backward,
            "forward" => AsOfDirection::Forward,
            "nearest" => AsOfDirection::Nearest,
            _ => {
                return Err(pyo3_arrow::error::PyArrowError::PyErr(
                    pyo3::exceptions::PyValueError::new_err(
                        format!("Invalid direction '{}'. Must be 'backward', 'forward', or 'nearest'.", direction),
                    ),
                ));
            }
        };

        let (left_batches, left_schema) = left.into_inner();
        let (right_batches, right_schema) = right.into_inner();

        let config = RustAsOfConfig {
            tolerance_ns,
            right_prefix: right_prefix.to_string(),
            direction: dir,
            allow_exact_match,
        };

        let result = crate::asof::join::asof_join_impl(
            &left_batches,
            &left_schema,
            &right_batches,
            &right_schema,
            on,
            by,
            &config,
        )?;

        if result.is_empty() || result[0].num_rows() == 0 {
            let output = PyTable::try_new(result, left_schema)?;
            return Ok(output.into_pyarrow(py)?);
        }

        let result_schema = result[0].schema();
        let output = PyTable::try_new(result, result_schema)?;
        Ok(output.into_pyarrow(py)?)
    }

    // Keep backward-only alias for API compatibility
    #[pyfunction]
    #[pyo3(signature = (left, right, on="timestamp", by=None, tolerance_ns=None, right_prefix=""))]
    pub fn asof_join_backward<'py>(
        py: Python<'py>,
        left: PyTable,
        right: PyTable,
        on: &str,
        by: Option<&str>,
        tolerance_ns: Option<i64>,
        right_prefix: &str,
    ) -> PyArrowResult<Bound<'py, PyAny>> {
        asof_join(py, left, right, on, by, tolerance_ns, right_prefix, "backward", true)
    }

    /// Align multiple secondary streams onto a primary timeline in parallel.
    #[pyfunction]
    #[pyo3(signature = (primary, streams, on="timestamp", by=None))]
    pub fn align_streams<'py>(
        py: Python<'py>,
        primary: PyTable,
        streams: Vec<Bound<'py, pyo3::types::PyDict>>,
        on: &str,
        by: Option<&str>,
    ) -> PyArrowResult<Bound<'py, PyAny>> {
        let (primary_batches, primary_schema) = primary.into_inner();

        let mut specs: Vec<StreamSpec> = Vec::with_capacity(streams.len());

        for stream_dict in &streams {
            let table_obj = stream_dict
                .get_item("table")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'table' key"))?;
            let py_table: PyTable = table_obj.extract()?;
            let (batches, schema) = py_table.into_inner();

            let prefix: String = stream_dict
                .get_item("prefix")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_default();

            let direction_str: String = stream_dict
                .get_item("direction")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "backward".to_string());

            let tolerance_ns: Option<i64> = stream_dict
                .get_item("tolerance_ns")?
                .and_then(|v| if v.is_none() { None } else { Some(v) })
                .map(|v| v.extract::<i64>())
                .transpose()?;

            let allow_exact: bool = stream_dict
                .get_item("allow_exact_match")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(true);

            let dir = match direction_str.as_str() {
                "backward" => AsOfDirection::Backward,
                "forward" => AsOfDirection::Forward,
                "nearest" => AsOfDirection::Nearest,
                _ => {
                    return Err(pyo3_arrow::error::PyArrowError::PyErr(
                        pyo3::exceptions::PyValueError::new_err(
                            format!("Invalid direction '{}'.", direction_str),
                        ),
                    ));
                }
            };

            specs.push(StreamSpec {
                batches,
                schema,
                config: RustAsOfConfig {
                    tolerance_ns,
                    right_prefix: prefix,
                    direction: dir,
                    allow_exact_match: allow_exact,
                },
                on: String::new(),
                by: by.map(|s| s.to_string()),
            });
        }

        let result = crate::asof::multi::align_streams_impl(
            &primary_batches,
            &primary_schema,
            specs,
            on,
            by,
        )?;

        if result.is_empty() || result[0].num_rows() == 0 {
            let output = PyTable::try_new(result, primary_schema)?;
            return Ok(output.into_pyarrow(py)?);
        }

        let result_schema = result[0].schema();
        let output = PyTable::try_new(result, result_schema)?;
        Ok(output.into_pyarrow(py)?)
    }

    fn parse_direction(direction: &str) -> PyResult<AsOfDirection> {
        match direction {
            "backward" => Ok(AsOfDirection::Backward),
            "forward" => Ok(AsOfDirection::Forward),
            "nearest" => Ok(AsOfDirection::Nearest),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid direction '{}'.", direction),
            )),
        }
    }

    /// Streaming incremental as-of join engine.
    #[pyclass(name = "StreamingJoin")]
    pub struct PyStreamingJoin {
        inner: StreamingAsOfJoin,
        left_schema: Option<arrow_schema::SchemaRef>,
        right_schema: Option<arrow_schema::SchemaRef>,
    }

    #[pymethods]
    impl PyStreamingJoin {
        #[new]
        #[pyo3(signature = (on="timestamp", by=None, direction="backward", tolerance_ns=None, allow_exact_match=true, lateness_ns=0))]
        fn new(
            on: &str,
            by: Option<&str>,
            direction: &str,
            tolerance_ns: Option<i64>,
            allow_exact_match: bool,
            lateness_ns: i64,
        ) -> PyResult<Self> {
            let dir = parse_direction(direction)?;
            Ok(Self {
                inner: StreamingAsOfJoin::new(
                    dir, tolerance_ns, allow_exact_match, lateness_ns,
                    on.to_string(), by.map(|s| s.to_string()),
                ),
                left_schema: None,
                right_schema: None,
            })
        }

        fn push_right(&mut self, table: PyTable) -> PyResult<()> {
            let (batches, schema) = table.into_inner();
            if self.right_schema.is_none() { self.right_schema = Some(schema); }
            for batch in &batches {
                self.inner.push_right(batch)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            Ok(())
        }

        fn push_left(&mut self, table: PyTable) -> PyResult<()> {
            let (batches, schema) = table.into_inner();
            if self.left_schema.is_none() { self.left_schema = Some(schema); }
            for batch in &batches {
                self.inner.push_left(batch)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            Ok(())
        }

        fn advance_watermark(&mut self, watermark_ns: i64) {
            self.inner.advance_watermark(watermark_ns);
        }

        fn emit<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
            let left_schema = self.left_schema.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No left data pushed yet"))?;
            let right_schema = self.right_schema.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No right data pushed yet"))?;
            let result = self.inner.emit(left_schema, right_schema)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            match result {
                Some(batch) => {
                    let schema = batch.schema();
                    let output = PyTable::try_new(vec![batch], schema)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    Ok(Some(output.into_pyarrow(py)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?))
                }
                None => Ok(None),
            }
        }

        fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
            let left_schema = self.left_schema.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No left data pushed yet"))?;
            let right_schema = self.right_schema.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No right data pushed yet"))?;
            let result = self.inner.flush(left_schema, right_schema)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            match result {
                Some(batch) => {
                    let schema = batch.schema();
                    let output = PyTable::try_new(vec![batch], schema)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    Ok(Some(output.into_pyarrow(py)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?))
                }
                None => Ok(None),
            }
        }

        #[getter]
        fn pending_count(&self) -> usize { self.inner.pending_count() }
        #[getter]
        fn watermark(&self) -> i64 { self.inner.watermark() }
        #[getter]
        fn total_emitted(&self) -> usize { self.inner.total_emitted }
        #[getter]
        fn total_left_received(&self) -> usize { self.inner.total_left_received }
        #[getter]
        fn total_right_received(&self) -> usize { self.inner.total_right_received }

        fn prune_right_before(&mut self, timestamp_ns: i64) {
            self.inner.prune_right_before(timestamp_ns);
        }
    }

    // -----------------------------------------------------------------------
    // IPC file I/O
    // -----------------------------------------------------------------------

    /// Read an Arrow IPC file and return as a PyArrow table.
    #[pyfunction]
    #[pyo3(signature = (path, projection=None, batch_limit=None))]
    pub fn read_ipc<'py>(
        py: Python<'py>,
        path: &str,
        projection: Option<Vec<usize>>,
        batch_limit: Option<usize>,
    ) -> PyArrowResult<Bound<'py, PyAny>> {
        let config = crate::ipc::ScanConfig { projection, batch_limit };
        let (batches, schema) = crate::ipc::scan_ipc_file(std::path::Path::new(path), &config)
            .map_err(|e| pyo3_arrow::error::PyArrowError::PyErr(
                pyo3::exceptions::PyIOError::new_err(e.to_string()),
            ))?;
        let output = PyTable::try_new(batches, schema)?;
        Ok(output.into_pyarrow(py)?)
    }

    /// Write a PyArrow table to an Arrow IPC file.
    #[pyfunction]
    pub fn write_ipc(table: PyTable, path: &str) -> PyResult<()> {
        let (batches, schema) = table.into_inner();
        crate::ipc::write_ipc_file(std::path::Path::new(path), &batches, &schema)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Read an Arrow IPC file, filtering batches by timestamp range.
    #[pyfunction]
    #[pyo3(signature = (path, on="timestamp", min_ts=i64::MIN, max_ts=i64::MAX))]
    pub fn read_ipc_time_range<'py>(
        py: Python<'py>,
        path: &str,
        on: &str,
        min_ts: i64,
        max_ts: i64,
    ) -> PyArrowResult<Bound<'py, PyAny>> {
        let (batches, schema) = crate::ipc::scan_ipc_time_range(
            std::path::Path::new(path),
            on,
            min_ts,
            max_ts,
            &crate::ipc::ScanConfig::default(),
        )
        .map_err(|e| pyo3_arrow::error::PyArrowError::PyErr(
            pyo3::exceptions::PyIOError::new_err(e.to_string()),
        ))?;
        let output = PyTable::try_new(batches, schema)?;
        Ok(output.into_pyarrow(py)?)
    }

    #[pymodule]
    pub fn flowstate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(echo_table, m)?)?;
        m.add_function(wrap_pyfunction!(row_count, m)?)?;
        m.add_function(wrap_pyfunction!(asof_join, m)?)?;
        m.add_function(wrap_pyfunction!(asof_join_backward, m)?)?;
        m.add_function(wrap_pyfunction!(align_streams, m)?)?;
        m.add_function(wrap_pyfunction!(read_ipc, m)?)?;
        m.add_function(wrap_pyfunction!(write_ipc, m)?)?;
        m.add_function(wrap_pyfunction!(read_ipc_time_range, m)?)?;
        m.add_class::<PyStreamingJoin>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::flowstate_core;
