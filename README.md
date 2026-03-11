<p align="center">
  <strong>FlowState</strong><br>
  <em>Rust-accelerated temporal alignment engine for quantitative finance</em>
</p>

<p align="center">
  <a href="https://github.com/RyanJHamby/flowstate/actions/workflows/ci.yml"><img src="https://github.com/RyanJHamby/flowstate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python"></a>
</p>

---

## Problem

Every quantitative trading firm builds the same internal infrastructure: join heterogeneous market data streams — trades, quotes, bars, signals — into point-in-time correct feature matrices for model training and backtesting. The requirements are always the same:

- **No look-ahead bias.** A trade at time `T` must only see quotes at time `<= T`. Violating this invalidates every backtest downstream.
- **Nanosecond precision.** Microsecond timestamps lose ordering information in high-frequency data. Timestamps are `int64` nanoseconds, not floats.
- **Hundreds of symbols, billions of rows.** pandas falls over at 10M rows. Polars handles it but treats as-of joins as one operation among hundreds — not the primary design target.
- **Streaming and batch.** Research needs batch replay over historical data. Production needs incremental alignment on live feeds with watermark semantics.
- **GPU-ready tensors.** The output goes into PyTorch or JAX. Every CPU copy between alignment and the GPU is wasted latency.

FlowState solves this pipeline end-to-end: partitioned storage with three-level pruning, Rust-accelerated temporal joins, streaming watermark alignment, and GPU-direct data feeding — all connected through Apache Arrow zero-copy.

## Architecture

```
                        ┌──────────────────────────────────┐
                        │          Python API               │
                        │  TemporalAligner · StreamingAligner│
                        │  FeatureStore · ReplayEngine       │
                        └──────────────┬───────────────────┘
                                       │ Arrow PyCapsule Interface
                                       │ (zero-copy, no serialization)
                        ┌──────────────▼───────────────────┐
                        │       Rust Core (PyO3)            │
                        │  O(n+m) merge-scan · Rayon parallel│
                        │  Streaming joins · Arrow IPC I/O   │
                        │  6,400 lines · 132 tests           │
                        └──────────────┬───────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │    Storage       │    │    Alignment      │    │   Data Feeding   │
    │ Hive partitions  │    │ As-of join engine │    │ Pinned memory    │
    │ xxhash bucketing │    │ Multi-stream (N×) │    │ kvikio GDS       │
    │ NVMe LRU cache   │    │ Watermark stream  │    │ CUDA streams     │
    │ S3/GCS/Azure     │    │ Batch coalescer   │    │ PyTorch/JAX      │
    └─────────────────┘    └──────────────────┘    └──────────────────┘
```

## Performance

Benchmarked on 1M left × 500K right rows, 1,000 symbols, Apple M-series. Measured against Polars 1.x, the fastest general-purpose option.

| Operation | FlowState | Polars | Speedup |
|---|---|---|---|
| Grouped as-of join (1M rows, 1K symbols) | **10 ms** | 18 ms | 1.8x |
| Ungrouped as-of join (1M rows) | **4 ms** | 7 ms | 1.8x |
| Multi-stream alignment (4 streams) | **42 ms** | 42 ms | Parity |
| Multi-stream alignment (8 streams) | **85 ms** | 100 ms | 1.2x |
| Streaming incremental join | Sub-microsecond emit | N/A | — |
| SPSC ring buffer throughput | 82M elem/s | N/A | — |

FlowState is faster because it solves a narrower problem. Polars handles arbitrary DataFrame operations; FlowState handles exactly one thing — temporal joins on sorted timestamp data — and exploits every invariant that implies: pre-sorted merge scans, symbol-partitioned parallelism, tolerance early termination, and pre-partitioned storage that eliminates runtime hash table construction.

### Capability comparison

| Capability | FlowState | Polars | pandas | kdb+/q | DuckDB |
|---|---|---|---|---|---|
| **Streaming incremental joins** | Watermark + late policy | No | No | wj (windowed) | No |
| **GPU data feeding** | kvikio GDS + CUDA streams | No | No | No | No |
| **Multi-stream single pass** | Rayon parallel N-way | Sequential | N/A | Manual | N/A |
| **Lock-free pipeline** | SPSC ring → join → coalesce | No | No | IPC | No |
| **Point-in-time default** | Backward (no look-ahead) | Backward | Forward-fill | aj (backward) | Backward |
| **Pinned memory pool** | cudaMallocHost + CPU fallback | No | No | No | No |
| **Arrow zero-copy** | PyCapsule Interface | Yes | No (copies) | No | Yes |
| **ML DataLoader** | PyTorch + JAX adapters | No | No | No | No |

## Usage

### Batch alignment: trades with quotes

```python
from flowstate.prism.alignment import TemporalAligner

aligner = TemporalAligner(
    primary_type="trade",
    secondary_specs={"quote": ["bid_price", "ask_price"]},
    tolerance_ns=5_000_000_000,  # 5 second max staleness
)
aligner.add_data("trade", trade_table)   # pa.Table, int64 ns timestamps
aligner.add_data("quote", quote_table)

aligned, stats = aligner.flush()
# Every row is point-in-time correct — quote timestamp <= trade timestamp
```

### Streaming alignment with watermarks

```python
from flowstate.prism.streaming import StreamingAligner, StreamingAlignConfig

aligner = StreamingAligner(StreamingAlignConfig(
    group_col="symbol",
    tolerance_ns=5_000_000_000,
    lateness_ns=1_000_000_000,  # 1s late data tolerance
))

for batch in live_feed:
    aligner.push_left(trade_batch)
    aligner.push_right(quote_batch)
    aligner.advance_watermark(current_event_time_ns)

    result = aligner.emit()  # rows sealed by watermark
    if result is not None:
        model.predict(result)

final = aligner.flush()  # end-of-stream
```

### Rust kernel directly

```python
import flowstate_core

# Grouped as-of join — dispatches to Rayon parallel merge-scan
result = flowstate_core.asof_join(
    trades, quotes, on="timestamp", by="symbol",
    direction="backward", tolerance_ns=5_000_000_000,
)

# Streaming join with watermark semantics
join = flowstate_core.StreamingJoin(
    on="timestamp", by="symbol", direction="backward",
    tolerance_ns=5_000_000_000, lateness_ns=1_000_000_000,
)
join.push_left(trade_batch)
join.push_right(quote_batch)
join.advance_watermark(current_time_ns)
result = join.emit()
```

### Temporal feature store

```python
from flowstate.store import (
    FeatureCatalog, FeatureDefinition, FeatureMaterializer, FeatureServer,
)

catalog = FeatureCatalog("/data/features/catalog.json")
catalog.register(FeatureDefinition(
    name="trade_with_quote",
    primary_stream="trade",
    secondary_stream="quote",
    columns=["bid_price", "ask_price"],
    tolerance_ns=5_000_000_000,
))

materializer = FeatureMaterializer(catalog=catalog, output_dir="/data/features/mat")
materializer.add_stream("trade", trade_table)
materializer.add_stream("quote", quote_table)
materializer.materialize_all()

server = FeatureServer(catalog=catalog, data_dir="/data/features/mat")
table = server.get_feature("trade_with_quote", symbols=["AAPL"])
```

### GPU data feeding

```python
from flowstate.prism.gpu_direct import GPUDirectReader, GPUDirectConfig

reader = GPUDirectReader(GPUDirectConfig(
    device_id=0,
    num_streams=2,          # async H2D overlap
    gds_task_size=4*1024*1024,
))

# NVMe → PCIe DMA → GPU VRAM (zero CPU copies via kvikio GDS)
gpu_array = reader.read_binary_to_gpu("/data/prices.bin", dtype=np.float32)

# Arrow IPC I/O with column projection and temporal range filtering
table = flowstate_core.read_ipc("aligned.arrow", projection=[0, 1, 3])
table = flowstate_core.read_ipc_time_range("aligned.arrow", on="timestamp", min_ts=t0, max_ts=t1)
```

## Project Structure

```
FlowState/
├── flowstate-core/           # Rust crate — 6,400 lines, 132 tests
│   └── src/
│       ├── lib.rs            # PyO3 bindings: joins, streaming, IPC
│       ├── asof/
│       │   ├── scan.rs       # O(n+m) merge-scan kernels (backward/forward/nearest)
│       │   ├── parallel_scan.rs  # Chunked parallel scan, binary-search cursor starts
│       │   ├── join.rs       # Orchestration: sort-detect, ahash grouping, Rayon dispatch
│       │   ├── gather.rs     # Parallel column gather via Arrow take()
│       │   ├── multi.rs      # Multi-stream parallel alignment
│       │   ├── streaming.rs  # Watermark-based streaming join (900 lines)
│       │   └── config.rs     # Direction enum, config struct
│       ├── ipc.rs            # Arrow IPC read/write/scan, projection, time-range filter
│       ├── spsc.rs           # Lock-free SPSC ring buffer, AtomicU64, cache-line padded
│       ├── pipeline.rs       # Streaming pipeline: SPSC → join → coalesce → output
│       ├── coalesce.rs       # Adaptive batch coalescer, target-row flushing
│       ├── hdr.rs            # HDR histogram, log-linear bucketing, CAS min/max
│       ├── bloom.rs          # Bloom filter, double-hashing, auto-tuned FPR
│       ├── pool.rs           # Slab buffer pool, auto-return, zero-on-drop
│       └── pinned.rs         # CUDA pinned memory allocator, page-aligned fallback
│
├── src/flowstate/            # Python package — 7,800 lines
│   ├── prism/                # Query, alignment, data feeding
│   │   ├── alignment.py      # TemporalAligner, AlignmentSpec, Rust/Python dual backend
│   │   ├── streaming.py      # StreamingAligner with watermark emission
│   │   ├── replay.py         # Replay engine with 3-level partition pruning
│   │   ├── gpu_direct.py     # kvikio GDS reads, CUDA stream H2D transfers
│   │   ├── pinned_buffer.py  # CUDA pinned memory pool with CPU fallback
│   │   ├── prefetcher.py     # Double-buffered async prefetch pipeline
│   │   ├── dataloader.py     # PyTorch IterableDataset, JAX iterator
│   │   ├── distributed.py    # Multi-rank replay with NCCL barrier sync
│   │   └── shard.py          # File-level sharding strategies
│   ├── store/                # Temporal feature store
│   │   ├── catalog.py        # Versioned feature definitions, dependency DAG
│   │   ├── materializer.py   # Alignment-based materialization to Arrow IPC
│   │   └── server.py         # Feature serving with symbol filtering
│   ├── storage/              # Partitioned storage, caching, cloud
│   │   ├── partitioning.py   # Hive partitioning with xxhash bucketing
│   │   ├── writer.py         # Partitioned Parquet writer (zstd)
│   │   ├── cache.py          # NVMe LRU cache tier
│   │   └── object_store.py   # fsspec backends (S3, GCS, Azure)
│   ├── schema/               # Arrow schemas, validation, normalization
│   └── features/             # Microstructure feature library
│
├── tests/                    # 636 Python tests — 8,100 lines
├── benchmarks/               # Full-stack benchmark suite
├── .github/workflows/ci.yml  # CI: Python 3.11–3.13, Rust, Criterion, integration
└── DESIGN.md                 # System architecture and design decisions
```

## Testing

```bash
python -m pytest tests/ -v                              # 636 Python tests
cd flowstate-core && cargo test --no-default-features   # 132 Rust tests (121 unit + 11 proptest)
cargo bench --no-default-features                       # Criterion benchmarks
python benchmarks/bench_full_suite.py                   # Full-stack Python benchmarks
```

Test coverage includes:
- **Correctness:** 11 proptest property-based tests verify Rust kernels against reference implementations across random inputs
- **Integration:** 14 end-to-end tests validate the full pipeline (replay → align → materialize → serve)
- **Point-in-time:** Dedicated tests verify no look-ahead bias in backward joins and correct look-ahead in forward joins
- **Streaming parity:** Tests verify streaming alignment produces identical results to batch alignment

## Quick Start

```bash
git clone https://github.com/RyanJHamby/flowstate.git && cd flowstate
pip install -e ".[dev]"

# Build the Rust core (requires Rust toolchain + maturin)
cd flowstate-core && maturin develop --release && cd ..

# Optional: GPU support (kvikio + cupy)
pip install -e ".[gpu]"

# Verify
python -m pytest tests/ -v
```

The Rust kernel is a transparent accelerator. If `flowstate_core` is not installed, all operations fall back to a pure Python implementation using NumPy and bisect — same API, same correctness guarantees, lower throughput.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
