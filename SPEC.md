# FlowState v2: Production System Specification

## Thesis

FlowState becomes the **fastest temporal join engine for financial data**, implemented as a Rust core with a Python API. The differentiator over Polars/DuckDB: purpose-built for market data alignment with features no general-purpose engine offers — streaming incremental joins, multi-stream alignment in a single pass, pre-partitioned symbol storage that eliminates runtime grouping, and tolerance-aware early termination.

**Target:** Beat Polars on grouped as-of joins by 2-5x on market data workloads. Ship as `pip install flowstate`.

---

## Architecture

```
User Python Code
       │
       ▼
┌─────────────────────────────────────────┐
│  flowstate (Python)                     │
│  ├── as_of_join(left, right, ...)       │  ← Public API (unchanged)
│  ├── align_streams(primary, [...])      │
│  ├── TemporalAligner                    │
│  └── MicrostructureEngine               │
└──────────────┬──────────────────────────┘
               │ Arrow C Data Interface (zero-copy)
               ▼
┌─────────────────────────────────────────┐
│  flowstate_core (Rust, PyO3)            │
│  ├── asof_join_ungrouped()              │  ← O(n+m) linear merge scan
│  ├── asof_join_grouped()                │  ← Hash-partitioned + parallel merge
│  ├── asof_join_streaming()              │  ← Incremental with watermarks
│  ├── multi_align()                      │  ← N-stream single-pass alignment
│  └── microstructure kernels             │  ← SIMD feature computation
└──────────────┬──────────────────────────┘
               │
               ▼
         arrow-rs + rayon
```

### Zero-Copy Data Path

```
PyArrow Table
  → Arrow PyCapsule Interface (pyo3-arrow)
    → arrow-rs RecordBatch (zero-copy, shared memory)
      → Rust computation (rayon parallel)
    → Arrow PyCapsule Interface
  → PyArrow Table
```

No serialization. No copying. The Rust kernel operates directly on the same memory buffers Python allocated.

---

## Rust Crate: `flowstate_core`

### File Layout

```
flowstate-core/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # PyO3 module, Python bindings
│   ├── asof/
│   │   ├── mod.rs              # AsOfConfig, dispatch logic
│   │   ├── merge_scan.rs       # O(n+m) sorted linear scan (core kernel)
│   │   ├── grouped.rs          # Hash-partitioned grouped join
│   │   ├── nearest.rs          # Bidirectional nearest match
│   │   └── tolerance.rs        # Tolerance filtering + early termination
│   ├── multi_align.rs          # N-stream single-pass alignment
│   ├── partition.rs            # Symbol hash partitioning for parallelism
│   ├── streaming.rs            # Watermark-based incremental alignment
│   └── simd.rs                 # SIMD timestamp comparison (optional, x86/ARM)
├── benches/
│   ├── asof_bench.rs           # Criterion benchmarks
│   └── vs_polars.rs            # Head-to-head comparison
└── tests/
    ├── correctness.rs          # Property-based tests (proptest)
    └── edge_cases.rs           # Nulls, empty tables, single-row, etc.
```

### Core Algorithm: Sorted Linear Merge Scan

Polars uses this. It's O(n+m) instead of O(n log m) binary search. For sorted market data this is strictly better.

```
merge_scan_backward(left_ts[], right_ts[]) -> indices[]:
    j = 0  // right cursor
    for i in 0..left.len():
        // Advance right cursor while right[j] <= left[i]
        while j < right.len() && right[j+1] <= left[i]:
            j += 1
        if right[j] <= left[i]:
            indices[i] = j
        else:
            indices[i] = NULL
```

**Why we beat Polars:** Polars must handle arbitrary DataFrame schemas and join types. We handle exactly one thing: temporal joins on sorted timestamp data. This means:

1. **No sort verification overhead** — we require sorted input (market data always is) and skip the check, or verify with a single branchless pass
2. **Pre-partitioned groups** — if data was written by FlowState's partitioned writer, symbol groups are already separated on disk. We skip hash table construction entirely
3. **Tolerance early termination** — when tolerance is set (common in production), we break out of the scan early once distance exceeds the window
4. **Batch-oriented SIMD** — for the ungrouped case, we can compare 8 timestamps simultaneously with AVX2/NEON
5. **Rayon parallel groups** — each symbol group is independent; parallelize with zero coordination

### Advantage Matrix: What No General-Purpose Engine Offers

| Feature | Polars | DuckDB | FlowState v2 |
|---------|--------|--------|---------------|
| As-of join | Yes (one of many ops) | Yes (SQL) | Yes (the only op) |
| Streaming/incremental | No (full materialization) | No | Yes (watermark-based) |
| Multi-stream single pass | No (N sequential joins) | No | Yes |
| Pre-partitioned skip | No | No | Yes (Hive-aware) |
| Tolerance early-exit | Partial | No | Full |
| SIMD timestamp scan | No (generic types) | Yes (internal) | Yes (specialized) |
| Point-in-time audit trail | No | No | Yes (provenance tracking) |
| Nanosecond-native | Yes | Microsecond | Yes |

---

## Python Package: `flowstate`

### API (Unchanged for Users)

```python
from flowstate import as_of_join, align_streams, AsOfConfig

# Single join — dispatches to Rust kernel
result, stats = as_of_join(
    left=trades_table,       # pa.Table
    right=quotes_table,      # pa.Table
    on="timestamp",
    by="symbol",
    config=AsOfConfig(
        direction="backward",
        tolerance_ns=5_000_000_000,
    ),
)

# Multi-stream — single Rust call, not N sequential Python calls
aligned, stats = align_streams(
    primary=trades,
    secondaries=[quotes_spec, bars_spec],
)
```

### Fallback Strategy

```python
# In alignment.py
try:
    from flowstate_core import _asof_join_rust
    _USE_RUST = True
except ImportError:
    _USE_RUST = False

def as_of_join(left, right, on="timestamp", by=None, config=None):
    if _USE_RUST:
        return _asof_join_rust(left, right, on, by, config)
    return _asof_join_python(left, right, on, by, config)  # Current numpy impl
```

Pure Python install still works. Rust kernel is a transparent accelerator.

---

## Project Structure (Final)

```
FlowState/
├── flowstate-core/                    # Rust crate (NEW)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── asof/
│   │   │   ├── mod.rs
│   │   │   ├── merge_scan.rs
│   │   │   ├── grouped.rs
│   │   │   ├── nearest.rs
│   │   │   └── tolerance.rs
│   │   ├── multi_align.rs
│   │   ├── partition.rs
│   │   ├── streaming.rs
│   │   └── simd.rs
│   ├── benches/
│   │   └── asof_bench.rs
│   └── tests/
│       └── correctness.rs
├── src/flowstate/                     # Python package (EXISTING)
│   ├── __init__.py
│   ├── prism/
│   │   ├── alignment.py               # Dispatches to Rust or numpy
│   │   ├── replay.py
│   │   └── ...
│   ├── features/
│   │   └── microstructure.py
│   └── ...
├── tests/                             # Python tests (EXISTING + NEW)
│   ├── test_temporal_alignment.py     # Existing — now tests both backends
│   ├── test_rust_parity.py            # NEW — verifies Rust == Python results
│   └── ...
├── benchmarks/
│   ├── bench_vs_polars.py             # Head-to-head Rust kernel vs Polars
│   ├── bench_scaling.py               # Scaling from 100K to 100M rows
│   └── ...
├── pyproject.toml                     # Updated: maturin build backend
└── README.md
```

---

## Implementation Steps

### Phase 0: Scaffolding (1 session)

**Goal:** Rust crate compiles, Python can import it, one trivial function round-trips a PyArrow table.

1. Create `flowstate-core/` directory with `Cargo.toml`
   - Dependencies: `pyo3`, `pyo3-arrow`, `arrow` (arrow-rs), `rayon`
   - `crate-type = ["cdylib"]`

2. Implement `lib.rs` with PyO3 module exposing one function:
   ```rust
   #[pyfunction]
   fn echo_table(table: PyTable) -> PyArrowResult<PyTable> {
       // Accept a PyArrow table, return it unchanged
       // Proves zero-copy round-trip works
   }
   ```

3. Update `pyproject.toml` for maturin:
   ```toml
   [build-system]
   requires = ["maturin>=1.0,<2.0"]
   build-backend = "maturin"
   ```

4. `maturin develop` — verify `from flowstate_core import echo_table` works

5. Add one Python test that sends a PyArrow table through Rust and back, verifying data integrity.

**Exit criteria:** `pytest tests/test_rust_roundtrip.py` passes. CI builds on Linux/macOS.

---

### Phase 1: Ungrouped Backward Join (1-2 sessions)

**Goal:** Rust kernel for the simplest case — two sorted arrays, backward join, no grouping.

1. Implement `asof/merge_scan.rs`:
   - Accept two `Int64Array` (timestamps), return `Int64Array` (indices, -1 for no match)
   - Stateful cursor scan: O(n+m)
   - Handle: empty arrays, all-null, single element

2. Implement `asof/tolerance.rs`:
   - Post-filter indices where `left_ts - right_ts[idx] > tolerance`
   - Early termination variant: break scan when distance exceeds tolerance

3. Wire into `lib.rs`:
   ```rust
   #[pyfunction]
   fn asof_join_backward(
       left: PyTable, right: PyTable,
       on: &str, tolerance_ns: Option<i64>,
   ) -> PyArrowResult<PyTable>
   ```

4. Python integration in `alignment.py` — dispatch to Rust when available

5. **Benchmark:** Compare against current numpy implementation AND Polars
   - Target: match or beat Polars on ungrouped backward join

6. **Tests:** Run all existing `test_temporal_alignment.py` tests against Rust backend

**Exit criteria:** All existing ungrouped tests pass with Rust backend. Benchmark shows >= Polars speed.

---

### Phase 2: Grouped Join with Hash Partitioning (1-2 sessions)

**Goal:** Per-symbol grouped joins using hash table + parallel merge scans.

1. Implement `partition.rs`:
   - Build hash map: `symbol_bytes → Vec<row_index>` from a StringArray
   - Return partition mapping for both left and right sides

2. Implement `asof/grouped.rs`:
   - For each symbol present in both sides:
     - Extract sorted timestamp slices
     - Run merge_scan within the group
     - Map local indices back to global positions
   - Use `rayon::par_iter` over symbol groups

3. Wire into `lib.rs` with `by` parameter

4. **Benchmark at scale:**
   - 1M rows, 200 symbols — target: 2x faster than Polars
   - 10M rows, 500 symbols — target: match Polars (memory-bound)
   - 100M rows, 1000 symbols — target: within 1.5x of Polars

5. **Property-based tests:** Use `proptest` in Rust to generate random sorted arrays and verify against a naive O(n*m) implementation

**Exit criteria:** All existing grouped tests pass. Faster than Polars at 1M+ rows with many symbols.

---

### Phase 3: Forward + Nearest + Full Config (1 session)

**Goal:** Complete feature parity with the Python implementation.

1. Implement `asof/nearest.rs`:
   - Two-cursor scan: maintain backward and forward candidates
   - Pick minimum distance, break ties by backward preference

2. Add `allow_exact_match` support (skip equal timestamps)

3. Add `right_prefix` handling in the Rust gather step

4. Full `AsOfConfig` struct in Rust mirroring Python dataclass

5. **Tests:** Run entire `test_temporal_alignment.py` suite against Rust backend for all directions

**Exit criteria:** 100% feature parity. Every test passes on both backends. `bench_vs_polars.py` shows results.

---

### Phase 4: Multi-Stream Single-Pass Alignment (1 session)

**Goal:** Align N streams in one pass instead of N sequential joins.

1. Implement `multi_align.rs`:
   - Accept primary table + N secondary tables
   - Merge-scan all N secondaries simultaneously against primary timeline
   - One pass through primary, N cursors on secondaries
   - Output: wide table with all secondary columns appended

2. This is a unique capability — Polars requires N separate `join_asof` calls

3. **Benchmark:** 1 primary + 4 secondaries (trades + quotes + bars + signals + reference)
   - Compare: 4x sequential Polars joins vs 1x FlowState multi_align

**Exit criteria:** `align_streams()` dispatches to Rust. 3-4x faster than sequential joins.

---

### Phase 5: Streaming Incremental Joins (1-2 sessions)

**Goal:** As-of joins on live/streaming data with watermark-based emission.

1. Implement `streaming.rs`:
   - `StreamingJoiner` struct that accepts incremental batches
   - Maintains sorted buffers for left and right with configurable max size
   - Watermark: emit joined rows when left timestamp is below watermark
   - Late data: configurable (drop, recompute, buffer)

2. Python `StreamingAligner` class wrapping the Rust engine

3. **This is the killer feature.** No existing engine does streaming as-of joins well.
   - Polars: batch only
   - DuckDB: batch only
   - Flink: has temporal joins but not as-of semantics with tolerance
   - Kafka Streams: no native as-of join

**Exit criteria:** Streaming aligner handles out-of-order data, emits correct results, benchmarks show sub-millisecond latency per batch.

---

### Phase 6: SIMD + Performance Polish (1 session)

**Goal:** Extract maximum performance from the Rust kernel.

1. `simd.rs`: SIMD-accelerated timestamp comparison
   - x86_64: AVX2 `_mm256_cmpgt_epi64` for 4-wide i64 comparison
   - aarch64: NEON `vcgtq_s64` for 2-wide
   - Fallback: scalar (auto-vectorization by LLVM usually sufficient)

2. Cache-line alignment for timestamp arrays during processing

3. Branch-free merge scan inner loop (conditional moves instead of branches)

4. Profile with `perf stat` and `cargo flamegraph`

5. **Benchmark gauntlet:**
   - 100K, 1M, 10M, 100M rows
   - 1, 10, 100, 1000, 10000 symbols
   - With and without tolerance
   - All three directions
   - Compare against Polars, DuckDB, and kdb+ (if available)

**Exit criteria:** Published benchmark table in README showing FlowState vs Polars vs DuckDB.

---

### Phase 7: CI, Packaging, Release (1 session)

**Goal:** `pip install flowstate` just works on Linux/macOS/Windows with pre-built Rust binaries.

1. GitHub Actions matrix:
   - `maturin build` for: linux-x86_64, linux-aarch64, macos-x86_64, macos-aarch64, windows-x86_64
   - Python 3.11, 3.12, 3.13
   - Wheel upload to PyPI

2. `pyproject.toml` with proper metadata, classifiers, URLs

3. Pure-Python fallback: if `flowstate_core` import fails, fall back to numpy implementation with a warning

4. README with:
   - Benchmark results table
   - Installation instructions
   - Migration guide from Polars `join_asof`
   - API reference

5. Criterion benchmarks in CI (catch regressions)

**Exit criteria:** `pip install flowstate` on a fresh machine installs pre-built wheel. All tests pass. Benchmarks published.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Ungrouped backward join | >= Polars throughput |
| Grouped backward join (200+ symbols) | 2-5x faster than Polars |
| Multi-stream alignment (4 streams) | 3-4x faster than sequential Polars |
| Streaming incremental join latency | < 1ms per 10K-row batch |
| Memory overhead vs Polars | <= 1.2x |
| `pip install` time | < 30 seconds (pre-built wheel) |
| Test count | 400+ (Python) + 100+ (Rust proptest) |
| Supported platforms | Linux x86/ARM, macOS x86/ARM, Windows x86 |

---

## What Makes This Valuable Enough to Adopt

1. **Faster than Polars at the one thing quant researchers do constantly** — as-of joins. Not 10% faster. 2-5x on real workloads (many symbols, tolerance windows).

2. **Streaming joins that don't exist anywhere else** — live data alignment without full re-materialization. This is the feature that makes firms bring it in-house.

3. **Multi-stream alignment in one call** — researchers currently write error-prone chains of sequential joins. FlowState does it correctly in one pass.

4. **Drop-in replacement** — accepts PyArrow tables (the standard interchange format). No new data format to learn. Works alongside Polars, Pandas, DuckDB.

5. **Auditable point-in-time correctness** — the only library that makes no-lookahead-bias a first-class guarantee with provenance tracking.

6. **Pure Python fallback** — `pip install flowstate` works even without Rust toolchain. The Rust kernel is a transparent accelerator, not a hard dependency.
