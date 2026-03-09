"""Benchmark: FlowState as-of join vs Polars and DuckDB.

Honest comparison against production-grade engines to identify real
bottlenecks and measure where we stand.
"""

from __future__ import annotations

import time

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from flowstate.prism.alignment import AsOfConfig, as_of_join

TS = pa.timestamp("ns", tz="UTC")


def _make_data(n: int, n_symbols: int, seed: int = 42):
    """Generate test data in all three formats."""
    rng = np.random.default_rng(seed)
    n_per_sym = n // n_symbols
    actual_n = n_per_sym * n_symbols

    symbols = []
    timestamps = []
    prices = []
    sizes = []

    for i in range(n_symbols):
        sym = f"SYM{i:04d}"
        ts = np.cumsum(rng.integers(50, 150, size=n_per_sym)).astype(np.int64)
        px = 100.0 + np.cumsum(rng.normal(0, 0.01, n_per_sym))
        sz = rng.uniform(10, 1000, n_per_sym)
        symbols.extend([sym] * n_per_sym)
        timestamps.append(ts)
        prices.append(px)
        sizes.append(sz)

    ts_arr = np.concatenate(timestamps)
    px_arr = np.concatenate(prices)
    sz_arr = np.concatenate(sizes)

    # Sort by timestamp globally
    order = np.argsort(ts_arr, kind="mergesort")
    ts_arr = ts_arr[order]
    px_arr = px_arr[order]
    sz_arr = sz_arr[order]
    sym_arr = [symbols[i] for i in order]

    return ts_arr, px_arr, sz_arr, sym_arr


def _time_fn(fn, warmup: int = 2, runs: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def bench(n_rows: int, n_symbols: int, quote_multiplier: int = 2):
    """Run a single benchmark comparison."""
    # Generate data
    ts_l, px_l, sz_l, sym_l = _make_data(n_rows, n_symbols, seed=42)
    ts_r, px_r, sz_r, sym_r = _make_data(n_rows * quote_multiplier, n_symbols, seed=99)

    # --- Polars ---
    left_pl = pl.DataFrame({
        "timestamp": ts_l,
        "trade_price": px_l,
        "trade_size": sz_l,
        "symbol": sym_l,
    }).sort("timestamp")

    right_pl = pl.DataFrame({
        "timestamp": ts_r,
        "quote_price": px_r,
        "quote_size": sz_r,
        "symbol": sym_r,
    }).sort("timestamp")

    def polars_join():
        return left_pl.join_asof(
            right_pl,
            on="timestamp",
            by="symbol",
            strategy="backward",
        )

    # --- DuckDB ---
    con = duckdb.connect()
    con.execute("CREATE TABLE left_t AS SELECT * FROM left_pl")
    con.execute("CREATE TABLE right_t AS SELECT * FROM right_pl")

    def duckdb_join():
        return con.execute("""
            SELECT l.*, r.quote_price, r.quote_size
            FROM left_t l
            ASOF LEFT JOIN right_t r
            ON l.symbol = r.symbol AND l.timestamp >= r.timestamp
        """).fetchall()

    # --- FlowState ---
    left_pa = pa.table({
        "timestamp": pa.array(ts_l, type=TS),
        "trade_price": pa.array(px_l, type=pa.float64()),
        "trade_size": pa.array(sz_l, type=pa.float64()),
        "symbol": pa.array(sym_l, type=pa.string()),
    })
    right_pa = pa.table({
        "timestamp": pa.array(ts_r, type=TS),
        "quote_price": pa.array(px_r, type=pa.float64()),
        "quote_size": pa.array(sz_r, type=pa.float64()),
        "symbol": pa.array(sym_r, type=pa.string()),
    })
    cfg = AsOfConfig(right_prefix="quote_", direction="backward")

    def flowstate_join():
        return as_of_join(left_pa, right_pa, config=cfg)

    # Verify all produce same number of output rows
    pl_result = polars_join()
    dk_result = duckdb_join()
    fs_result = flowstate_join()
    assert len(pl_result) == len(ts_l), f"Polars: {len(pl_result)} != {len(ts_l)}"
    assert len(dk_result) == len(ts_l), f"DuckDB: {len(dk_result)} != {len(ts_l)}"
    assert fs_result[0].num_rows == len(ts_l), f"FlowState: {fs_result[0].num_rows} != {len(ts_l)}"

    pl_ms = _time_fn(polars_join)
    dk_ms = _time_fn(duckdb_join)
    fs_ms = _time_fn(flowstate_join)

    con.close()
    return pl_ms, dk_ms, fs_ms


def main():
    print("Benchmark: FlowState vs Polars vs DuckDB — Grouped As-Of Join")
    print("(backward join, grouped by symbol, median of 5 runs after warmup)\n")

    configs = [
        (10_000, 10),
        (50_000, 50),
        (100_000, 50),
        (100_000, 200),
        (500_000, 50),
    ]

    print(f"{'ROWS':>10} {'SYMS':>6} {'POLARS (ms)':>14} {'DUCKDB (ms)':>14} {'FLOWST (ms)':>14} {'vs POLARS':>10} {'vs DUCKDB':>10}")
    print("=" * 88)

    for n_rows, n_syms in configs:
        pl_ms, dk_ms, fs_ms = bench(n_rows, n_syms)
        vs_pl = f"{fs_ms / pl_ms:.1f}x slow" if fs_ms > pl_ms else f"{pl_ms / fs_ms:.1f}x fast"
        vs_dk = f"{fs_ms / dk_ms:.1f}x slow" if fs_ms > dk_ms else f"{dk_ms / fs_ms:.1f}x fast"
        print(f"{n_rows:>10,} {n_syms:>6} {pl_ms:>13.1f} {dk_ms:>13.1f} {fs_ms:>13.1f} {vs_pl:>10} {vs_dk:>10}")


if __name__ == "__main__":
    main()
