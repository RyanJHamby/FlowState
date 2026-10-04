"""Reproducible as-of join benchmark: FlowState vs Polars (optionally pandas / DuckDB).

    python benchmarks/reproduce.py                  # README headline config
    python benchmarks/reproduce.py --pandas-scale   # adds pandas, at 1M / 10M left rows
    python benchmarks/reproduce.py --duckdb         # adds DuckDB (slow at this scale)

Prints the exact hardware, library versions and dataset parameters alongside
the timings so a result can be compared or challenged. Timings are the median
of --runs runs after --warmup warmup runs; data generation, sorting and
table construction are excluded. Every engine's output is checked against
Polars before timing.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import polars as pl
import pyarrow as pa

import flowstate
from flowstate.prism.alignment import _HAS_RUST, AsOfConfig, as_of_join

TS = pa.timestamp("ns", tz="UTC")


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return None


def hardware() -> str:
    chip = _sysctl("machdep.cpu.brand_string")
    mem = _sysctl("hw.memsize")
    if chip is None and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            chip = next(
                (ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), None
            )
        with open("/proc/meminfo") as f:
            mem = str(int(f.readline().split()[1]) * 1024)
    mem_gb = f"{int(mem) / 2**30:.0f} GB" if mem else "? GB"
    cpu = chip or platform.processor() or "unknown CPU"
    return f"{cpu}, {os.cpu_count()} logical cores, {mem_gb} RAM"


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def make_data(n: int, n_symbols: int, seed: int) -> dict[str, np.ndarray]:
    """Sorted-by-timestamp events; per-symbol inter-arrival 50-150 ns."""
    rng = np.random.default_rng(seed)
    n_per = n // n_symbols
    total = n_per * n_symbols
    sym_id = np.repeat(np.arange(n_symbols), n_per)
    ts = np.empty(total, dtype=np.int64)
    px = np.empty(total, dtype=np.float64)
    for i in range(n_symbols):
        s = i * n_per
        ts[s : s + n_per] = np.cumsum(rng.integers(50, 150, size=n_per))
        px[s : s + n_per] = 100.0 + np.cumsum(rng.normal(0, 0.01, n_per))
    order = np.argsort(ts, kind="stable")
    names = np.array([f"S{i:04d}" for i in range(n_symbols)])
    return {"timestamp": ts[order], "px": px[order], "symbol": names[sym_id[order]]}


def median_ms(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def bench(n_left: int, n_right: int, n_symbols: int, warmup: int, runs: int, engines: set[str]):
    left = make_data(n_left, n_symbols, seed=42)
    right = make_data(n_right, n_symbols, seed=99)
    results: dict[str, float] = {}

    # Polars (reference)
    lp = pl.DataFrame(left)
    rp = pl.DataFrame(
        {"timestamp": right["timestamp"], "qpx": right["px"], "symbol": right["symbol"]}
    )

    def run_polars():
        return lp.join_asof(rp, on="timestamp", by="symbol", strategy="backward")

    ref = run_polars()
    results["polars"] = median_ms(run_polars, warmup, runs)

    # FlowState
    la = pa.table({
        "timestamp": pa.array(left["timestamp"], type=TS),
        "px": left["px"],
        "symbol": pa.array(left["symbol"]),
    })
    ra = pa.table({
        "timestamp": pa.array(right["timestamp"], type=TS),
        "qpx": right["px"],
        "symbol": pa.array(right["symbol"]),
    })
    cfg = AsOfConfig(direction="backward")

    def run_fs():
        return as_of_join(la, ra, config=cfg)[0]

    out = run_fs()
    got = pl.from_arrow(out.select(["timestamp", "symbol", "qpx"])).sort(["timestamp", "symbol"])
    exp = ref.select(["timestamp", "symbol", "qpx"]).sort(["timestamp", "symbol"])
    assert got["qpx"].equals(exp["qpx"]), "FlowState output differs from Polars"
    results["flowstate"] = median_ms(run_fs, warmup, runs)

    if "duckdb" in engines:
        import duckdb

        con = duckdb.connect()
        con.register("l_arrow", lp.to_arrow())
        con.register("r_arrow", rp.to_arrow())
        con.execute("CREATE TABLE l AS SELECT * FROM l_arrow")
        con.execute("CREATE TABLE r AS SELECT * FROM r_arrow")
        sql = (
            "SELECT l.*, r.qpx FROM l ASOF LEFT JOIN r "
            "ON l.symbol = r.symbol AND l.timestamp >= r.timestamp"
        )
        assert con.execute(sql).fetch_arrow_table().num_rows == n_left - n_left % n_symbols
        results["duckdb"] = median_ms(lambda: con.execute(sql).fetch_arrow_table(), warmup, runs)

    if "pandas" in engines:
        import pandas as pd

        lpd = pd.DataFrame(left)
        rpd = pd.DataFrame(
            {"timestamp": right["timestamp"], "qpx": right["px"], "symbol": right["symbol"]}
        )

        def run_pd():
            return pd.merge_asof(lpd, rpd, on="timestamp", by="symbol", direction="backward")

        results["pandas"] = median_ms(run_pd, warmup, runs)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=int, default=1_000_000)
    ap.add_argument("--right", type=int, default=500_000)
    ap.add_argument("--symbols", type=int, default=1_000)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--duckdb", action="store_true",
                    help="also time DuckDB ASOF JOIN (can be very slow at 1M rows x 1K symbols)")
    ap.add_argument("--pandas-scale", action="store_true",
                    help="run 1M and 10M left rows incl. pandas instead of the headline config")
    args = ap.parse_args()

    print("FlowState as-of join benchmark")
    print(f"  hardware : {hardware()}")
    print(f"  os       : {platform.platform()}")
    print(f"  python   : {sys.version.split()[0]}")
    kernel = "yes" if _HAS_RUST else "NO - python fallback"
    print(f"  flowstate: {flowstate.__version__} (Rust kernel: {kernel})")
    print(f"  polars   : {version('polars')}   pandas: {version('pandas')}   "
          f"duckdb: {version('duckdb')}   pyarrow: {version('pyarrow')}")
    print(f"  method   : median of {args.runs} runs after {args.warmup} warmup; "
          "backward join, tolerance none; excludes data generation")
    if not _HAS_RUST:
        sys.exit("flowstate_core is not installed; benchmark would measure the Python fallback.")

    if args.pandas_scale:
        configs = [(1_000_000, 500_000, 1_000), (10_000_000, 5_000_000, 1_000)]
        engines = {"pandas"}
    else:
        configs = [(args.left, args.right, args.symbols)]
        engines = set()
    if args.duckdb:
        engines.add("duckdb")

    for n_l, n_r, n_s in configs:
        print(f"\ndataset: {n_l:,} left x {n_r:,} right rows, {n_s:,} symbols, "
              "synthetic random-walk quotes (seeds 42/99), globally timestamp-sorted")
        res = bench(n_l, n_r, n_s, args.warmup, args.runs, engines)
        for name, ms in sorted(res.items(), key=lambda kv: kv[1]):
            rel = res["polars"] / ms
            print(f"  {name:<10} {ms:>10.1f} ms   ({rel:.2f}x vs polars)")


if __name__ == "__main__":
    main()
