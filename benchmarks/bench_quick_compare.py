"""Quick comparison: FlowState vs Polars vs DuckDB on as-of join.

Single run each, small enough to complete fast, big enough to be meaningful.
"""

from __future__ import annotations

import time

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from flowstate.prism.alignment import AsOfConfig, as_of_join

TS = pa.timestamp("ns", tz="UTC")


def make_data(n: int, n_symbols: int, seed: int):
    rng = np.random.default_rng(seed)
    n_per = n // n_symbols
    actual = n_per * n_symbols

    syms = np.repeat([f"S{i:03d}" for i in range(n_symbols)], n_per)
    ts = np.empty(actual, dtype=np.int64)
    px = np.empty(actual, dtype=np.float64)
    for i in range(n_symbols):
        s = i * n_per
        e = s + n_per
        ts[s:e] = np.cumsum(rng.integers(50, 150, size=n_per))
        px[s:e] = 100.0 + np.cumsum(rng.normal(0, 0.01, n_per))

    # Sort by timestamp
    order = np.argsort(ts, kind="mergesort")
    return ts[order], px[order], syms[order]


def run(n_rows: int, n_symbols: int):
    ts_l, px_l, sym_l = make_data(n_rows, n_symbols, seed=42)
    ts_r, px_r, sym_r = make_data(n_rows * 2, n_symbols, seed=99)

    # Polars
    lp = pl.DataFrame({"timestamp": ts_l, "price": px_l, "symbol": sym_l}).sort("timestamp")
    rp = pl.DataFrame({"timestamp": ts_r, "qprice": px_r, "symbol": sym_r}).sort("timestamp")

    t0 = time.perf_counter()
    _ = lp.join_asof(rp, on="timestamp", by="symbol", strategy="backward")
    pl_ms = (time.perf_counter() - t0) * 1000

    # DuckDB
    con = duckdb.connect()
    con.execute("CREATE TABLE l AS SELECT * FROM lp")
    con.execute("CREATE TABLE r AS SELECT * FROM rp")
    t0 = time.perf_counter()
    _ = con.execute("SELECT l.*, r.qprice FROM l ASOF LEFT JOIN r ON l.symbol = r.symbol AND l.timestamp >= r.timestamp").fetchall()
    dk_ms = (time.perf_counter() - t0) * 1000
    con.close()

    # FlowState
    la = pa.table({
        "timestamp": pa.array(ts_l, type=TS),
        "price": pa.array(px_l, type=pa.float64()),
        "symbol": pa.array(sym_l.tolist(), type=pa.string()),
    })
    ra = pa.table({
        "timestamp": pa.array(ts_r, type=TS),
        "qprice": pa.array(px_r, type=pa.float64()),
        "symbol": pa.array(sym_r.tolist(), type=pa.string()),
    })
    cfg = AsOfConfig(right_prefix="q_", direction="backward")

    t0 = time.perf_counter()
    _ = as_of_join(la, ra, config=cfg)
    fs_ms = (time.perf_counter() - t0) * 1000

    return pl_ms, dk_ms, fs_ms


if __name__ == "__main__":
    print("FlowState vs Polars vs DuckDB — Grouped As-Of Join (single run)")
    print(f"{'ROWS':>10} {'SYMS':>6} {'POLARS':>10} {'DUCKDB':>10} {'FLOWST':>10} {'FS/PL':>8} {'FS/DK':>8}")
    print("=" * 74)

    for n, s in [(10_000, 10), (50_000, 50), (100_000, 50), (100_000, 200)]:
        pl_ms, dk_ms, fs_ms = run(n, s)
        print(f"{n:>10,} {s:>6} {pl_ms:>9.1f} {dk_ms:>9.1f} {fs_ms:>9.1f} {fs_ms/pl_ms:>7.0f}x {fs_ms/dk_ms:>7.0f}x")
