# FlowState: System Design

This document describes the architecture, algorithms, and design decisions behind FlowState. It is intended for engineers evaluating the system's internals.

---

## 1. Problem Statement

Quantitative trading systems require joining heterogeneous time-series streams into feature matrices for model training. The core operation is the **as-of join**: for each row in a primary stream (e.g., trades), find the most recent corresponding row in a secondary stream (e.g., quotes) as of that timestamp. This must be:

- **Point-in-time correct.** A backward join at timestamp `T` may only match rows at `T' <= T`. Violating this constraint introduces look-ahead bias, which invalidates any downstream backtest or model evaluation.
- **Performant at scale.** Production datasets contain billions of rows across thousands of symbols. The join must complete in seconds, not minutes.
- **Streaming-capable.** Research uses batch replay over historical data. Production uses incremental alignment on live feeds, where data arrives out-of-order and completeness is determined by watermarks.
- **Zero-copy from storage to GPU.** The aligned output feeds into PyTorch or JAX training loops. Every intermediate copy between disk and GPU VRAM is wasted latency and memory bandwidth.

General-purpose DataFrame libraries (Polars, pandas, DuckDB) treat as-of joins as one operation among hundreds. FlowState treats it as the only operation and exploits every invariant that implies.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Python API Layer                              │
│  TemporalAligner · StreamingAligner · FeatureStore · ReplayEngine       │
│  7,800 lines · Full type annotations · PEP 561 typed                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ Arrow PyCapsule Interface (zero-copy)
┌───────────────────────────────▼─────────────────────────────────────────┐
│                          Rust Core (PyO3)                               │
│  As-of join kernels · Streaming join · Arrow IPC I/O                    │
│  6,400 lines · 132 tests (121 unit + 11 proptest)                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
    ┌───────────────────┬───────┴───────┬───────────────────┐
    ▼                   ▼               ▼                   ▼
 Storage            Alignment       Streaming          Data Feeding
 Hive partitions    Merge-scan      Watermarks         Pinned memory
 NVMe LRU cache     Rayon parallel  Late data policy   kvikio GDS
 S3/GCS/Azure       Multi-stream    Batch coalescing   CUDA streams
```

### Data path

```
PyArrow Table
  → Arrow PyCapsule Interface (pyo3-arrow 0.17)
    → arrow-rs RecordBatch (zero-copy, shared memory)
      → Rust computation (Rayon parallel)
    → Arrow PyCapsule Interface
  → PyArrow Table
```

No serialization. No copying. The Rust kernel operates directly on the same memory buffers that Python allocated. This is the critical design decision that makes the system practical — a serialization boundary would dominate the runtime at these data volumes.

---

## 3. As-Of Join Engine

### 3.1 Core algorithm: sorted linear merge-scan

The as-of join kernel uses an O(n+m) sorted merge-scan, identical in complexity class to Polars. For two sorted timestamp arrays `left[0..n]` and `right[0..m]`:

```
backward_scan(left, right) → indices:
    j = 0
    for i in 0..n:
        while j+1 < m and right[j+1] <= left[i]:
            j += 1
        if j < m and right[j] <= left[i]:
            indices[i] = j
        else:
            indices[i] = NULL
```

Three scan variants exist: `backward` (match `<=`), `forward` (match `>=`), and `nearest` (minimum absolute distance, ties broken backward). All three are implemented in `flowstate-core/src/asof/scan.rs` as branchless inner loops for pipeline-friendly execution.

### 3.2 Why we outperform Polars

FlowState is faster on grouped temporal joins because it exploits domain-specific invariants that a general-purpose engine cannot assume:

1. **Skip sort verification.** Market data arrives pre-sorted by timestamp from exchanges. FlowState detects sortedness with a single branchless pass and skips the O(n log n) sort that Polars must defensively perform.

2. **ahash grouping with borrowed keys.** Symbol grouping uses `ahash` (the same hash Polars uses internally) but borrows `&str` keys directly from Arrow string arrays — zero `String` allocation during the group-build phase. The hash map maps `&str → Vec<u32>` indices.

3. **Rayon parallel per-group joins.** Each symbol group is independent. After grouping, Rayon's work-stealing scheduler distributes groups across cores with no coordination. At 1,000 symbols, this parallelizes cleanly across 8–16 cores.

4. **Parallel chunked scan.** For ungrouped joins on large inputs (5M+ rows), the left array is split into chunks. Binary search finds the starting right-cursor position for each chunk. Chunks execute in parallel with disjoint writes to the output index array via `UnsafeCell` (sound because write ranges are provably non-overlapping).

5. **Tolerance early termination.** When a tolerance window is configured (common in production — e.g., "stale quotes older than 5 seconds are invalid"), the scan breaks early once the distance exceeds the tolerance, avoiding unnecessary comparisons.

### 3.3 Multi-stream alignment

Joining N secondary streams against one primary is a common operation (e.g., align trades with quotes, bars, signals, and reference data simultaneously). Polars requires N sequential `join_asof` calls. FlowState does this in one call via `align_streams`, which dispatches N independent joins to Rayon's `par_iter`. At 8 streams this is 15% faster than sequential Polars — the only library that offers this as a single operation.

### 3.4 Parallel column gather

After computing join indices, the output table must be assembled by gathering columns from the right table at the computed positions. FlowState parallelizes this via Arrow's `take()` kernel dispatched per-column through Rayon. For wide right tables (20+ columns), this provides measurable speedup over sequential gather.

---

## 4. Streaming Join Engine

The streaming join engine (`src/asof/streaming.rs`, 900 lines) handles incremental alignment on live data with watermark semantics. This is the capability that no general-purpose engine provides.

### 4.1 Design

```
push_left(batch)  ──→  ┌──────────────┐
                       │  Left Buffer  │ (sorted by timestamp, grouped by symbol)
push_right(batch) ──→  │  Right Buffer │
                       └──────┬───────┘
                              │
advance_watermark(ts) ──→     ▼
                       ┌──────────────┐
                       │  Scan & Emit │ All left rows with ts < watermark are sealed
                       └──────┬───────┘
                              │
                    emit() ◄──┘  Returns joined pa.Table or None
```

- **Watermark semantics.** A watermark at time `W` declares: "no more data will arrive with timestamp < W." Left rows below the watermark are sealed — their join result is final and can be emitted. This decouples output latency from input completeness.
- **Configurable lateness.** `lateness_ns` extends the watermark by a tolerance window, allowing late-arriving data to still participate in joins before being dropped.
- **Right-side pruning.** `prune_right_before(ts)` garbage-collects right-side rows that can no longer match any future left row, bounding memory usage for long-running streams.
- **Per-symbol state.** Each symbol group maintains independent left/right buffers and watermark positions. Cross-symbol isolation prevents one slow symbol from blocking emission for others.

### 4.2 Python wrapper

`StreamingAligner` (Python) wraps the Rust `StreamingJoin` kernel. If the Rust extension is not installed, it falls back to a pure-Python implementation using `bisect`-based matching on sorted per-symbol buffers. The Python fallback produces identical results at lower throughput.

---

## 5. Storage Layer

### 5.1 Hive partitioning

Data is stored in Hive-partitioned Parquet with deterministic bucket assignment:

```
data/
└── type=trade/
    └── date=2024-01-15/
        ├── bucket=0000/data.parquet
        ├── bucket=0001/data.parquet
        └── ...
```

Bucket assignment uses `xxhash` on the symbol string, which provides uniform distribution and symbol affinity — the same symbol always maps to the same bucket, enabling per-bucket parallelism without cross-bucket coordination.

### 5.2 Three-level pruning

The replay engine eliminates unnecessary I/O at three levels:

1. **Partition pruning.** Hive partition keys (type, date) are matched against the replay filter before any file is opened.
2. **Row-group pushdown.** Parquet row-group statistics (min/max timestamp) are compared against the requested time range. Row groups outside the range are skipped entirely.
3. **Column projection.** Only requested columns are read from disk. For a 20-column Parquet file where the query needs 3 columns, this reduces I/O by 85%.

### 5.3 Caching

An NVMe LRU cache tier sits between cloud object storage (S3/GCS/Azure via fsspec) and the replay engine. Files are cached locally on first access. Cache eviction is LRU by total byte size, with configurable max capacity.

---

## 6. GPU Data Feeding

### 6.1 Pipeline

```
Parquet on disk
  → Replay engine (partition-pruned, projected)
    → Temporal alignment (Rust kernel)
      → Batch coalescer (target row count)
        → Pinned memory pool (cudaMallocHost or page-aligned mmap)
          → CUDA stream async H2D transfer
            → PyTorch / JAX tensor
```

### 6.2 Pinned memory pool

`PinnedBufferPool` pre-allocates CUDA pinned memory (`cudaMallocHost`) at startup. Pinned memory enables async DMA transfers — the GPU can read from host memory while the CPU prepares the next batch. If CUDA is not available, the pool falls back to page-aligned CPU memory via `mmap`, maintaining the same API.

### 6.3 GPUDirect Storage

On systems with NVMe SSDs and NVIDIA GPUDirect Storage support, `GPUDirectReader` uses kvikio `CuFile` to DMA data directly from NVMe to GPU VRAM, bypassing the CPU entirely. This eliminates two memory copies (NVMe→CPU, CPU→GPU) from the data path.

### 6.4 Double-buffered prefetch

`PrefetchPipeline` runs a background thread that fills buffer N+1 while the GPU processes buffer N. Configurable prefetch depth (default 2) keeps the GPU pipeline saturated. Backpressure prevents the prefetcher from running ahead of a slow consumer.

---

## 7. Lock-Free Infrastructure

### 7.1 SPSC ring buffer

`spsc.rs` implements a single-producer single-consumer ring buffer using `AtomicU64` with Acquire/Release ordering. The ring is cache-line padded (128 bytes) to prevent false sharing between producer and consumer threads. Throughput: 82M elements/second single-threaded, 20M elements/second cross-thread.

### 7.2 Streaming pipeline

`pipeline.rs` composes SPSC rings into a streaming pipeline:

```
Input SPSC → StreamingJoin → BatchCoalescer → Output SPSC
```

Each stage runs on its own thread. The `BatchCoalescer` accumulates small output batches into larger ones (configurable target row count) to amortize downstream overhead. Pipeline latency is tracked via an HDR histogram with log-linear bucketing and CAS-based concurrent recording.

### 7.3 Bloom filter

`bloom.rs` implements a lock-free Bloom filter with double-hashing and auto-tuned false positive rate. Used for fast negative lookups in partition pruning — if a symbol is definitely not in a partition's Bloom filter, the partition can be skipped without reading its Parquet metadata.

### 7.4 Slab buffer pool

`pool.rs` provides a pre-allocated slab pool with automatic return-on-drop and zero-on-return semantics. Used by the streaming pipeline to avoid allocation in the hot path.

---

## 8. Temporal Feature Store

The feature store provides a catalog → materialize → serve workflow for managing aligned feature sets:

- **Catalog** (`catalog.py`): Versioned feature definitions with dependency DAG, status lifecycle (active/deprecated/archived), tag-based filtering, and JSON persistence.
- **Materializer** (`materializer.py`): Reads catalog definitions, runs `align_streams()`, writes results to Arrow IPC files. Handles dependency ordering — features are materialized in topological order.
- **Server** (`server.py`): Reads materialized IPC files with column projection and symbol filtering. Provides feature metadata (schema, row count, byte size) for operational monitoring.

---

## 9. Correctness Guarantees

- **No look-ahead bias by default.** Backward joins are the default direction. Every matched right-side timestamp is `<=` the left-side timestamp. This is verified by dedicated integration tests.
- **Property-based testing.** 11 proptest tests in Rust generate random sorted timestamp arrays and verify kernel output against a brute-force O(n*m) reference implementation. Properties tested include: backward match is rightmost, forward match is leftmost, nearest minimizes distance, tolerance is enforced, output length equals left length.
- **Streaming-batch parity.** Integration tests verify that streaming alignment on complete data produces identical results to batch alignment.
- **End-to-end pipeline tests.** 14 integration tests validate that data flows correctly from raw generation through partitioned storage, replay, alignment, materialization, and serving with no row loss or corruption.

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| **O(n+m) merge-scan, not O(n log m) binary search** | Market data is pre-sorted. Linear scan has better cache locality and constant factors than binary search for sorted inputs. Same complexity class as Polars. |
| **ahash with borrowed `&str` keys** | Zero `String` allocation during grouping. `ahash` is the fastest non-cryptographic hash for short keys. Borrowed keys avoid copying Arrow string buffers. |
| **`UnsafeCell` for parallel index writes** | Chunked parallel scan writes to disjoint ranges of the output index array. `UnsafeCell` avoids the overhead of per-element atomic operations. Soundness relies on provably non-overlapping chunk boundaries. |
| **`TimestampSlice` enum for zero-copy ts access** | Borrows the Arrow buffer directly when data is already sorted, avoiding a copy into a separate `Vec<i64>`. Falls back to a sorted copy only when input is unsorted. |
| **Feature-gated PyO3** | `#[cfg(feature = "python")]` allows all Rust code to compile and test without Python. CI runs `cargo test --no-default-features` for pure-Rust validation. |
| **Pure Python fallback** | Every Rust-accelerated operation has a Python fallback using NumPy and bisect. The Rust kernel is a transparent accelerator imported at runtime — `pip install flowstate` works without a Rust toolchain. |
| **Arrow PyCapsule Interface (pyo3-arrow 0.17)** | Zero-copy data exchange between Python and Rust. No serialization, no intermediate buffers. The Rust kernel operates on the same physical memory that PyArrow allocated. |
| **Watermark-based streaming emission** | Borrowed from Apache Flink's event-time processing model. Watermarks decouple output completeness from input ordering, which is essential for real-time feeds where data arrives out-of-order. |
| **Deterministic xxhash bucketing** | Same symbol always maps to the same partition bucket regardless of timestamp or data type. Enables symbol-affinity sharding for distributed replay without runtime coordination. |

---

## 11. Test Matrix

| Layer | Count | Tool | What it covers |
|---|---|---|---|
| Rust unit tests | 121 | `cargo test` | Scan kernels, streaming join, SPSC, HDR, Bloom, coalescer, pool, IPC, pinned memory |
| Rust property tests | 11 | proptest | Random-input correctness against brute-force reference for all three join directions |
| Python unit tests | 611 | pytest | All Python modules: alignment, replay, storage, streaming, GPU, feature store, schemas, microstructure |
| Integration tests | 14 | pytest | End-to-end pipeline, point-in-time correctness, streaming-batch parity, cache, partitioning, distributed replay |
| Criterion benchmarks | 8 | `cargo bench` | Rust kernel throughput: sequential vs parallel, tolerance, all directions, SPSC ring |
| Python benchmarks | 10 | bench_full_suite.py | Full-stack: as-of join, multi-stream, streaming, IPC, replay, alignment, feature store, cache, partitioning, prefetcher |
