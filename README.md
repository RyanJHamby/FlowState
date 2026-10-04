<p align="center">
  <strong>FlowState</strong><br>
  <em>Rust-accelerated temporal alignment engine for quantitative finance</em>
</p>

<p align="center">
  <a href="https://github.com/RyanJHamby/flowstate/actions/workflows/ci.yml"><img src="https://github.com/RyanJHamby/flowstate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/flowstate-asof/"><img src="https://img.shields.io/pypi/v/flowstate-asof.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python"></a>
</p>

---

## Try it

```bash
pip install flowstate-asof        # imports as `flowstate`; includes the Rust kernel wheel
```

```python
import numpy as np, pyarrow as pa
from flowstate.prism.alignment import AsOfConfig, as_of_join

rng, n, ns = np.random.default_rng(0), 100_000, pa.timestamp("ns", tz="UTC")
def side(m, col):  # m rows, sorted int64-ns timestamps over ~1000 s, 3 symbols
    return pa.table({"timestamp": pa.array(np.sort(rng.integers(0, 10**12, m)), ns),
                     "symbol": rng.choice(["AAPL", "MSFT", "NVDA"], m), col: rng.normal(100, 1, m)})
trades, quotes = side(n, "price"), side(4 * n, "bid")
joined, stats = as_of_join(trades, quotes, config=AsOfConfig(tolerance_ns=5 * 10**9))
print(joined.slice(0, 3).to_pylist()); print(stats)  # each trade sees only quotes at or before it
```

Also in [`examples/try_it.py`](examples/try_it.py). (`flowstate-asof` is the PyPI name; the import
is `flowstate`. The unrelated `flowstate` project on PyPI is a different package.)

## Problem

Every quantitative trading firm builds the same internal infrastructure: join heterogeneous market data streams — trades, quotes, bars, signals — into point-in-time correct feature matrices for model training and backtesting. The requirements are always the same:

- **No look-ahead bias.** A trade at time `T` must only see quotes at time `<= T`. Violating this invalidates every backtest downstream.
- **Nanosecond precision.** Microsecond timestamps lose ordering information in high-frequency data. Timestamps are `int64` nanoseconds, not floats.
- **Hundreds of symbols, billions of rows.** A single in-memory as-of join is fine in pandas or Polars at 10M rows (see [Performance](#performance)). The target here is pipelines that align many streams over partitioned data too large for one in-memory frame.
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

Reproduce with one command (prints hardware, versions and dataset alongside the timings):

```bash
pip install flowstate-asof polars pandas && python benchmarks/reproduce.py --pandas-scale --runs 5
```

Backward grouped as-of join, 1,000 symbols, synthetic random-walk quotes (seeds 42/99), inputs
globally sorted by timestamp, median of 5 runs after 2 warmups (`--runs 5`), data generation excluded. Every
engine's output is checked against Polars before timing.
Measured 2026-10-04 on **Apple M4 (10 cores, 32 GB)**, macOS 26.6, Python 3.14.3,
Polars 1.38.1, pandas 3.0.6, pyarrow 23.0.1, FlowState 0.1.0 (release build).

| Left × right rows | Polars | **FlowState** | pandas `merge_asof` |
|---|---|---|---|
| 1M × 500K | **9.0 ms** | 13.1 ms | 74.5 ms |
| 10M × 5M | **122.5 ms** | 183.1 ms | 759.5 ms |

**On this workload Polars is about 1.4x faster than FlowState** and both are 6–8x faster than
pandas. FlowState's value is not raw join speed against Polars: it is the pieces around the join
(watermark streaming alignment, the SPSC streaming pipeline, partitioned replay and the temporal
feature store). Earlier README numbers claiming a 1.8x lead over Polars could not be reproduced
and were removed. Ungrouped joins, multi-stream alignment and DuckDB (`--duckdb`; its `ASOF JOIN`
did not finish at 1M × 1K symbols in our run) are not part of the headline table. Rust
micro-benchmarks live in `flowstate-core/benches` (`cargo bench --no-default-features`).

### What is implemented, and how well it is tested

| Capability | Status |
|---|---|
| Backward / forward / nearest as-of join, tolerance, per-symbol grouping | Rust kernel, property-tested against a reference implementation |
| Multi-stream alignment (N secondary streams, one pass) | Rust + Rayon, tested; no published speed comparison |
| Streaming incremental join with watermark and late-data policy | Rust, tested for parity with batch alignment |
| Lock-free SPSC pipeline (ring → join → coalesce) | Rust, unit-tested |
| Partitioned Parquet storage, NVMe cache, S3/GCS/Azure via fsspec | Tested locally; cloud backends not exercised against real buckets in CI |
| PyTorch / JAX data adapters | Tested on CPU |
| **GPU feeding: kvikio GDS, CUDA streams, pinned memory** | **Not yet tested on a real GPU.** The test suite exercises the CPU fallback paths only. Treat the GPUDirect/CUDA code as untested. |

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

> **Status: untested on real hardware.** This path (kvikio GDS, CUDA streams, pinned memory) has
> only been exercised through its CPU fallbacks. No GPU benchmark numbers are claimed.

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
├── orderbook/                # C++ limit order book — header-only, 25 Catch2 tests
│   └── include/orderbook/
│       ├── types.h           # Integer-tick prices, order/fill structs
│       ├── price_level.h     # FIFO queue per price (std::deque, not std::list)
│       └── order_book.h      # Array-indexed levels, O(1) BBO, FIFO matching
│
├── tests/                    # 637 Python tests — 8,100 lines
├── benchmarks/               # Full-stack benchmark suite
├── .github/workflows/ci.yml  # CI: Python 3.11–3.13, Rust, C++, Criterion, integration
└── DESIGN.md                 # System architecture and design decisions
```

## Testing

```bash
python -m pytest tests/ -v                              # 637 Python tests
cd flowstate-core && cargo test --no-default-features   # 132 Rust tests (121 unit + 11 proptest)
cargo bench --no-default-features                       # Criterion benchmarks
python benchmarks/reproduce.py                          # Reproducible join benchmark
python benchmarks/bench_full_suite.py                   # Full-stack Python benchmarks
```

Test coverage includes:
- **Correctness:** 11 proptest property-based tests verify Rust kernels against reference implementations across random inputs
- **Integration:** 14 end-to-end tests validate the full pipeline (replay → align → materialize → serve)
- **Point-in-time:** Dedicated tests verify no look-ahead bias in backward joins and correct look-ahead in forward joins
- **Streaming parity:** Tests verify streaming alignment produces identical results to batch alignment

## Quick Start

```bash
pip install flowstate-asof            # end users: prebuilt wheels, no Rust toolchain needed

# Contributors:
git clone https://github.com/RyanJHamby/flowstate.git && cd flowstate
python -m venv .venv && source .venv/bin/activate
pip install maturin && (cd flowstate-core && maturin develop --release)   # needs a Rust toolchain
pip install -e ".[dev]"

# Optional: GPU support (kvikio + cupy)
pip install -e ".[gpu]"

# Verify
python -m pytest tests/ -v
```

The Rust kernel is a transparent accelerator. If `flowstate_core` is not importable, all operations fall back to a pure Python implementation using NumPy and bisect — same API, same correctness guarantees, lower throughput.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
