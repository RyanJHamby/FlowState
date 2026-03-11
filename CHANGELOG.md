# Changelog

All notable changes to FlowState are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-03-11

### Added
- Temporal feature store: versioned catalog, alignment-based materializer, Arrow IPC serving with symbol filtering
- Streaming temporal alignment with watermark-based emission and configurable lateness tolerance
- Full-stack benchmark suite (`benchmarks/bench_full_suite.py`) covering all subsystems
- 14 end-to-end integration tests validating the replay → align → materialize → serve pipeline
- GitHub Actions CI: Python test matrix (3.11–3.13), Rust tests, Criterion benchmarks, integration tests
- PEP 561 `py.typed` marker for downstream type checking
- Public API exports in all subpackage `__init__.py` files

### Changed
- Replaced production `unwrap()` calls in Rust with proper error propagation (ipc.rs, multi.rs, pinned.rs)
- Added doc comments to all `StreamingJoin` PyO3 methods
- Expanded crate-level documentation in `lib.rs`
- Updated Cargo.toml with repository, keywords, and categories metadata

## [0.2.0] - 2025-03-10

### Added
- Rust as-of join kernel: O(n+m) merge-scan with parallel chunked scan, ahash grouping, Rayon dispatch
- Multi-stream parallel alignment beating Polars at 4+ streams
- Streaming join engine with watermark-based emission (~900 lines Rust)
- Arrow IPC I/O: read, write, scan, column projection, temporal range filtering
- Lock-free infrastructure: SPSC ring buffer, HDR histogram, Bloom filter, slab buffer pool
- Streaming pipeline: SPSC → join → coalesce → output ring with latency tracking
- CUDA pinned memory allocator with pool and CPU fallback
- Double-buffered async prefetch pipeline
- GPU data feeding: kvikio GDS reads, CUDA stream async H2D transfers
- Distributed replay with file-level sharding (round-robin, symbol-affinity, time-range)
- PyTorch `IterableDataset` and JAX iterator adapters
- Criterion benchmarks for Rust kernels

## [0.1.0] - 2025-03-09

### Added
- Arrow-native schemas (trade/quote/bar) with nanosecond timestamps
- Schema registry with versioned compatibility checks
- Zero-copy normalization with A/B line arbitration
- Deterministic Hive partitioning (xxhash bucketing)
- Partitioned Parquet writer (zstd compression)
- NVMe LRU cache tier with fsspec backends (S3/GCS/Azure)
- Replay engine with 3-level pruning (partition → row-group → column)
- Temporal alignment engine: as-of joins (backward/forward/nearest), multi-stream, tolerance, per-symbol grouping
- WebSocket clients (Polygon, Alpaca)
- Ingestion pipeline orchestrator
- Microstructure feature library (EWMA, VWAP, Kyle's Lambda, etc.)
