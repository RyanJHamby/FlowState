# Changelog

All notable changes to FlowState are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-04

First public release. Versions 0.1–0.3 in earlier drafts of this file were internal development
milestones and were never published; their contents are folded into this entry below.

### Packaging
- Published to PyPI as **`flowstate-asof`** (import name `flowstate`) and **`flowstate-asof-core`**
  (import name `flowstate_core`, the Rust kernel). The bare name `flowstate` is taken on PyPI by an
  unrelated project.
- Rust kernel ships as abi3 wheels (CPython 3.11+) for Linux x86_64/aarch64, macOS x86_64/arm64
  and Windows x64, built with maturin in CI (`.github/workflows/release.yml`).
- `flowstate-asof` depends on `flowstate-asof-core`; the pure-Python fallback is still used if the
  kernel cannot be imported.

### Added
- `examples/try_it.py` (shown at the top of the README) and a test that keeps it working.
- `benchmarks/reproduce.py`: one-command benchmark that prints hardware, library versions and
  dataset parameters, and checks every engine's output against Polars.

### Changed
- README performance section rewritten to measured numbers. **On the reproducible benchmark
  (Apple M4, Polars 1.38.1) Polars is ~1.4x faster than FlowState on a 1M × 500K grouped as-of
  join**; the previously published 1.8x speedup could not be reproduced and was removed.
- Removed the kdb+/pandas/DuckDB/Polars capability matrix in favour of a status table.
- Documented that the GPU path (kvikio GDS, CUDA streams, pinned memory) has not been tested on a
  real GPU.

### Included in this release
- Temporal alignment: Rust O(n+m) as-of join kernel (backward/forward/nearest, tolerance,
  per-symbol grouping), parallel chunked scan, multi-stream alignment
- Streaming join engine with watermark-based emission and configurable lateness tolerance
- Temporal feature store: versioned catalog, alignment-based materializer, Arrow IPC serving
- Arrow IPC I/O with column projection and temporal range filtering
- Lock-free infrastructure: SPSC ring buffer, HDR histogram, Bloom filter, slab buffer pool
- Arrow-native trade/quote/bar schemas with nanosecond timestamps, schema registry
- Hive partitioning (xxhash bucketing), Parquet writer (zstd), NVMe LRU cache, fsspec backends
- Replay engine with 3-level pruning; distributed replay with file-level sharding
- PyTorch `IterableDataset` and JAX iterator adapters; CUDA pinned memory pool and kvikio GDS
  reader (CPU fallback only tested, see above)
- WebSocket clients (Polygon, Alpaca), microstructure feature library
- C++ limit order book (header-only, Catch2 tests)
- CI: Python 3.11–3.13, Rust tests, Criterion benchmarks, C++ tests, integration tests

## Development history (pre-release, unpublished)

### [0.3.0] - 2025-03-11

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

### [0.2.0] - 2025-03-10

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

### [0.1.0] - 2025-03-09

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

[Unreleased]: https://github.com/RyanJHamby/flowstate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RyanJHamby/flowstate/releases/tag/v0.1.0
