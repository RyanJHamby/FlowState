"""Full-stack benchmark suite for FlowState.

Measures performance across all major subsystems and produces a
formatted results table. Designed to run in CI or locally.

Usage:
    python benchmarks/bench_full_suite.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def bench(fn, warmup: int = 2, runs: int = 5) -> float:
    """Return median execution time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[runs // 2]


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def fmt_rate(rows: int, seconds: float) -> str:
    rate = rows / seconds
    if rate > 1_000_000:
        return f"{rate / 1_000_000:.1f}M rows/s"
    if rate > 1_000:
        return f"{rate / 1_000:.0f}K rows/s"
    return f"{rate:.0f} rows/s"


# ─── Data generation ──────────────────────────────────────────────────

def make_market_data(n: int, n_symbols: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]
    ts = np.sort(rng.integers(0, 10**12, n))
    syms = rng.choice(symbols, n)

    trades = pa.table({
        "timestamp": pa.array(ts, type=pa.int64()),
        "symbol": pa.array(syms),
        "price": pa.array(rng.uniform(50, 500, n)),
        "volume": pa.array(rng.integers(1, 10000, n)),
    })

    quote_ts = np.sort(rng.integers(0, 10**12, n))
    quote_syms = rng.choice(symbols, n)
    quotes = pa.table({
        "timestamp": pa.array(quote_ts, type=pa.int64()),
        "symbol": pa.array(quote_syms),
        "bid": pa.array(rng.uniform(50, 500, n)),
        "ask": pa.array(rng.uniform(50, 500, n)),
    })
    return trades, quotes


def write_hive_data(base_dir: Path, trades: pa.Table, n_buckets: int = 4):
    """Write trade data in Hive partition layout."""
    for bucket in range(n_buckets):
        part_dir = base_dir / "type=trade" / "date=2024-01-15" / f"bucket={bucket:04d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        chunk = trades.slice(
            bucket * (trades.num_rows // n_buckets),
            trades.num_rows // n_buckets,
        )
        pq.write_table(chunk, part_dir / "data.parquet")


# ─── Benchmarks ───────────────────────────────────────────────────────

results: list[dict] = []


def record(component: str, operation: str, rows: int, elapsed: float, detail: str = ""):
    results.append({
        "component": component,
        "operation": operation,
        "rows": rows,
        "elapsed_s": elapsed,
        "time": fmt_time(elapsed),
        "throughput": fmt_rate(rows, elapsed),
        "detail": detail,
    })


def bench_asof_join():
    """As-of join kernel benchmarks."""
    try:
        import flowstate_core
    except ImportError:
        print("  [skip] flowstate_core not available")
        return

    for n in [100_000, 1_000_000]:
        trades, quotes = make_market_data(n, n_symbols=500)

        # Grouped join — bind loop vars via default args
        _trades, _quotes = trades, quotes
        t = bench(lambda _t=_trades, _q=_quotes: flowstate_core.asof_join(
            _t, _q, on="timestamp", by="symbol", direction="backward",
        ))
        record("Rust Kernel", f"as-of join grouped {n // 1000}K", n, t, "backward, 500 symbols")

        # Ungrouped
        _t_ug = trades.drop("symbol")
        _q_ug = quotes.drop("symbol")
        t = bench(lambda _l=_t_ug, _r=_q_ug: flowstate_core.asof_join(
            _l, _r, on="timestamp", direction="backward",
        ))
        record("Rust Kernel", f"as-of join ungrouped {n // 1000}K", n, t)


def bench_multi_stream():
    """Multi-stream parallel alignment."""
    try:
        import flowstate_core
    except ImportError:
        return

    n = 1_000_000
    trades, _ = make_market_data(n, n_symbols=500)
    rng = np.random.default_rng(99)
    symbols = [f"SYM{i:04d}" for i in range(500)]

    for n_streams in [4, 8]:
        stream_dicts = []
        for s in range(n_streams):
            ts = np.sort(rng.integers(0, 10**12, 200_000))
            syms = rng.choice(symbols, 200_000)
            t = pa.table({
                "timestamp": pa.array(ts, type=pa.int64()),
                "symbol": pa.array(syms),
                f"val_{s}": pa.array(rng.uniform(0, 1, 200_000)),
            })
            stream_dicts.append({"table": t, "prefix": f"s{s}_", "direction": "backward"})

        _sd = stream_dicts
        elapsed = bench(lambda _s=_sd: flowstate_core.align_streams(
            trades, _s, on="timestamp", by="symbol",
        ))
        detail = f"{n_streams} parallel joins"
        record("Rust Kernel", f"multi-stream {n_streams}x", n, elapsed, detail)


def bench_streaming_join():
    """Streaming incremental join."""
    try:
        import flowstate_core
    except ImportError:
        return

    n_batches = 100
    batch_size = 10_000
    total = n_batches * batch_size
    rng = np.random.default_rng(42)

    batches_left = []
    batches_right = []
    for i in range(n_batches):
        base = i * 10_000_000
        ts = np.sort(rng.integers(base, base + 10_000_000, batch_size))
        syms = rng.choice([f"S{j}" for j in range(50)], batch_size)
        batches_left.append(pa.table({
            "timestamp": pa.array(ts, type=pa.int64()),
            "symbol": pa.array(syms),
            "price": pa.array(rng.uniform(100, 200, batch_size)),
        }))
        ts_r = np.sort(rng.integers(base, base + 10_000_000, batch_size))
        syms_r = rng.choice([f"S{j}" for j in range(50)], batch_size)
        batches_right.append(pa.table({
            "timestamp": pa.array(ts_r, type=pa.int64()),
            "symbol": pa.array(syms_r),
            "bid": pa.array(rng.uniform(100, 200, batch_size)),
        }))

    def run_streaming():
        j = flowstate_core.StreamingJoin(
            on="timestamp", by="symbol", direction="backward", lateness_ns=1_000_000,
        )
        rows = 0
        for i in range(n_batches):
            j.push_left(batches_left[i])
            j.push_right(batches_right[i])
            j.advance_watermark((i + 1) * 10_000_000 - 1_000_000)
            result = j.emit()
            if result is not None:
                rows += result.num_rows
        final = j.flush()
        if final is not None:
            rows += final.num_rows
        return rows

    elapsed = bench(run_streaming, warmup=1, runs=3)
    record("Rust Kernel", "streaming join 1M", total, elapsed, "100 batches × 10K rows")


def bench_ipc_io():
    """Arrow IPC read/write."""
    try:
        import flowstate_core
    except ImportError:
        return

    import tempfile
    n = 1_000_000
    trades, _ = make_market_data(n)

    with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
        path = f.name
        # Write
        t_write = bench(lambda: flowstate_core.write_ipc(trades, path))
        record("IPC I/O", f"write {n // 1000}K rows", n, t_write)

        # Read
        t_read = bench(lambda: flowstate_core.read_ipc(path))
        record("IPC I/O", f"read {n // 1000}K rows", n, t_read)

        # Read with projection
        t_proj = bench(lambda: flowstate_core.read_ipc(path, projection=[0, 1]))
        record("IPC I/O", f"read projected {n // 1000}K", n, t_proj, "2 of 4 columns")


def bench_replay():
    """Replay engine with partition pruning."""
    import tempfile

    from flowstate.prism.replay import ReplayEngine, ReplayFilter

    n = 100_000
    trades, _ = make_market_data(n, n_symbols=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        write_hive_data(base, trades, n_buckets=8)

        engine = ReplayEngine(str(base))
        t = bench(lambda: list(engine.replay()), warmup=1, runs=3)
        record("Replay", f"full scan {n // 1000}K rows", n, t, "8 partitions, Hive layout")

        filt = ReplayFilter(data_types=["trade"])
        t_f = bench(lambda: list(engine.replay(filt)), warmup=1, runs=3)
        record("Replay", f"filtered scan {n // 1000}K", n, t_f, "type=trade filter")


def bench_alignment_python():
    """Python temporal alignment (with Rust acceleration)."""
    from flowstate.prism.alignment import AlignmentSpec, AsOfConfig, align_streams

    n = 100_000
    trades, quotes = make_market_data(n, n_symbols=100)

    spec = AlignmentSpec(
        name="quotes",
        table=quotes,
        value_columns=["bid", "ask"],
        config=AsOfConfig(direction="backward"),
    )

    t = bench(lambda: align_streams(trades, [spec]))
    record("Alignment", f"align_streams {n // 1000}K", n, t, "single secondary, 100 symbols")


def bench_feature_store():
    """Feature store materialize + serve cycle."""
    import tempfile

    from flowstate.store.catalog import FeatureCatalog, FeatureDefinition
    from flowstate.store.materializer import FeatureMaterializer
    from flowstate.store.server import FeatureServer

    n = 100_000
    trades, quotes = make_market_data(n, n_symbols=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = FeatureCatalog()
        catalog.register(FeatureDefinition(
            name="trade_with_quote",
            primary_stream="trade",
            secondary_stream="quote",
            columns=["bid", "ask"],
        ))

        mat = FeatureMaterializer(catalog=catalog, output_dir=tmpdir)
        mat.add_stream("trade", trades)
        mat.add_stream("quote", quotes)

        t_mat = bench(lambda: mat.materialize(catalog.get("trade_with_quote")), warmup=1, runs=3)
        record("Feature Store", f"materialize {n // 1000}K", n, t_mat, "align + write IPC")

        # Ensure materialized for serve bench
        mat.materialize(catalog.get("trade_with_quote"))
        server = FeatureServer(catalog=catalog, data_dir=tmpdir)

        t_serve = bench(lambda: server.get_feature("trade_with_quote"), warmup=1, runs=5)
        record("Feature Store", f"serve {n // 1000}K", n, t_serve, "IPC read")

        t_filtered = bench(
            lambda: server.get_feature("trade_with_quote", symbols=["SYM0042"]),
            warmup=1, runs=5,
        )
        record("Feature Store", f"serve filtered {n // 1000}K", n, t_filtered, "symbol filter")


def bench_cache():
    """LRU cache operations."""
    import tempfile

    from flowstate.storage.cache import CacheConfig, LRUCache

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CacheConfig(cache_dir=Path(tmpdir) / "cache", max_size_bytes=100 * 1024**2)
        cache = LRUCache(config)

        # Create source files
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()
        files = []
        for i in range(100):
            p = src_dir / f"file_{i:04d}.parquet"
            p.write_bytes(b"x" * 10_000)
            files.append(p)

        # Put
        t_put = bench(lambda: [cache.put(f"f{i}.parquet", f) for i, f in enumerate(files)],
                       warmup=1, runs=3)
        record("Cache", "put 100 files", 100, t_put, "10KB each")

        # Get (all hits)
        t_get = bench(lambda: [cache.get(f"f{i}.parquet") for i in range(100)])
        record("Cache", "get 100 files (hits)", 100, t_get)


def bench_partitioning():
    """Hive partitioning hash throughput."""
    from flowstate.storage.partitioning import PartitionScheme

    scheme = PartitionScheme(num_buckets=256)
    symbols = [f"SYM{i:04d}" for i in range(10_000)]
    ts_ns = 1705320000 * 10**9

    t = bench(lambda: [scheme.partition_key(s, ts_ns, "trade") for s in symbols])
    record("Partitioning", "hash 10K symbols", 10_000, t, "xxhash, 256 buckets")


def bench_prefetcher():
    """Prefetch pipeline throughput."""
    from flowstate.prism.pinned_buffer import PinnedBufferPool
    from flowstate.prism.prefetcher import PrefetchPipeline

    n_batches = 50
    batch_size = 10_000
    total = n_batches * batch_size

    def make_batches():
        for i in range(n_batches):
            yield pa.RecordBatch.from_pydict({
                "timestamp": pa.array(range(i * batch_size, (i + 1) * batch_size), type=pa.int64()),
                "price": [100.0 + j * 0.001 for j in range(batch_size)],
            })

    pool = PinnedBufferPool()
    pipeline = PrefetchPipeline(pool=pool, numeric_columns=["price"])

    def run():
        for pb in pipeline.iter(make_batches()):
            pb.release_to(pool)

    t = bench(run, warmup=1, runs=3)
    detail = f"{n_batches} batches, pinned memory"
    record("Prefetcher", f"prefetch {total // 1000}K rows", total, t, detail)


# ─── Main ─────────────────────────────────────────────────────────────

def print_results():
    print()
    print("=" * 90)
    print("FlowState Benchmark Results")
    print("=" * 90)
    print(
        f"{'Component':<16s} {'Operation':<30s} "
        f"{'Time':>10s} {'Throughput':>15s} {'Detail':<s}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['component']:<16s} {r['operation']:<30s} "
            f"{r['time']:>10s} {r['throughput']:>15s} {r['detail']:<s}"
        )


def save_json(path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    print("FlowState Full Benchmark Suite")
    print(f"Python {sys.version.split()[0]}")
    print()

    benchmarks = [
        ("As-Of Join Kernel", bench_asof_join),
        ("Multi-Stream Alignment", bench_multi_stream),
        ("Streaming Join", bench_streaming_join),
        ("IPC I/O", bench_ipc_io),
        ("Replay Engine", bench_replay),
        ("Python Alignment", bench_alignment_python),
        ("Feature Store", bench_feature_store),
        ("LRU Cache", bench_cache),
        ("Partitioning", bench_partitioning),
        ("Prefetcher", bench_prefetcher),
    ]

    for name, fn in benchmarks:
        print(f"Running: {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  [error] {e}")

    print_results()

    if "--json" in sys.argv:
        idx = sys.argv.index("--json")
        if idx + 1 < len(sys.argv):
            save_json(sys.argv[idx + 1])
        else:
            save_json("benchmark_results.json")
