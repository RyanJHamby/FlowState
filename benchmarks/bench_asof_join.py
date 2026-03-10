"""FlowState vs Polars as-of join benchmarks.

Measures single-join and multi-stream alignment performance across
varying dataset sizes and symbol counts. Produces a summary table
suitable for README or presentation.

Usage:
    python benchmarks/bench_asof_join.py
"""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa

import flowstate_core
import polars as pl


def make_data(n_left: int, n_right: int, n_symbols: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

    left_ts = np.sort(rng.integers(0, 10**12, n_left))
    left_syms = rng.choice(symbols, n_left)
    left = pa.table({
        "timestamp": pa.array(left_ts, type=pa.int64()),
        "symbol": pa.array(left_syms),
        "price": pa.array(rng.uniform(50, 500, n_left)),
    })

    right_ts = np.sort(rng.integers(0, 10**12, n_right))
    right_syms = rng.choice(symbols, n_right)
    right = pa.table({
        "timestamp": pa.array(right_ts, type=pa.int64()),
        "symbol": pa.array(right_syms),
        "bid": pa.array(rng.uniform(50, 500, n_right)),
    })

    return left, right


def bench(fn, warmup: int = 2, runs: int = 7) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[runs // 2]  # median


def bench_single_join():
    print("=" * 72)
    print("SINGLE AS-OF JOIN (backward, grouped by symbol)")
    print("=" * 72)
    print(f"{'Left':>10s} {'Right':>10s} {'Symbols':>8s} {'Rust ms':>9s} {'Polars ms':>10s} {'Ratio':>7s}")
    print("-" * 72)

    configs = [
        (100_000, 50_000, 100),
        (500_000, 250_000, 500),
        (1_000_000, 500_000, 1_000),
        (5_000_000, 2_500_000, 2_000),
    ]

    for n_left, n_right, n_sym in configs:
        left, right = make_data(n_left, n_right, n_sym)
        left_pl = pl.from_arrow(left)
        right_pl = pl.from_arrow(right)

        rust_ms = bench(lambda: flowstate_core.asof_join(
            left, right, on="timestamp", by="symbol", direction="backward"
        )) * 1000

        polars_ms = bench(lambda: left_pl.join_asof(
            right_pl, on="timestamp", by="symbol", strategy="backward"
        )) * 1000

        ratio = rust_ms / polars_ms
        print(f"{n_left:>10,d} {n_right:>10,d} {n_sym:>8,d} {rust_ms:>9.1f} {polars_ms:>10.1f} {ratio:>6.2f}x")


def bench_ungrouped():
    print()
    print("=" * 72)
    print("SINGLE AS-OF JOIN (backward, ungrouped)")
    print("=" * 72)
    print(f"{'Left':>10s} {'Right':>10s} {'Rust ms':>9s} {'Polars ms':>10s} {'Ratio':>7s}")
    print("-" * 72)

    configs = [
        (100_000, 50_000),
        (500_000, 250_000),
        (1_000_000, 500_000),
        (5_000_000, 2_500_000),
    ]

    for n_left, n_right in configs:
        left, right = make_data(n_left, n_right, 1)
        # Strip symbol column for ungrouped
        left_ug = left.drop("symbol")
        right_ug = right.drop("symbol")
        left_pl = pl.from_arrow(left_ug)
        right_pl = pl.from_arrow(right_ug)

        rust_ms = bench(lambda: flowstate_core.asof_join(
            left_ug, right_ug, on="timestamp", direction="backward"
        )) * 1000

        polars_ms = bench(lambda: left_pl.join_asof(
            right_pl, on="timestamp", strategy="backward"
        )) * 1000

        ratio = rust_ms / polars_ms
        print(f"{n_left:>10,d} {n_right:>10,d} {rust_ms:>9.1f} {polars_ms:>10.1f} {ratio:>6.2f}x")


def bench_multi_stream():
    print()
    print("=" * 72)
    print("MULTI-STREAM ALIGNMENT (parallel Rust vs sequential Polars)")
    print("=" * 72)
    print(f"{'Streams':>8s} {'Left':>10s} {'Per-stream':>11s} {'Rust ms':>9s} {'Polars ms':>10s} {'Ratio':>7s}")
    print("-" * 72)

    n_left = 1_000_000
    n_sym = 1_000
    rng = np.random.default_rng(42)
    symbols = [f"SYM{i:04d}" for i in range(n_sym)]

    left_ts = np.sort(rng.integers(0, 10**12, n_left))
    left_syms = rng.choice(symbols, n_left)
    primary = pa.table({
        "timestamp": pa.array(left_ts, type=pa.int64()),
        "symbol": pa.array(left_syms),
        "price": pa.array(rng.uniform(50, 500, n_left)),
    })
    primary_pl = pl.from_arrow(primary)

    for n_streams in [2, 4, 8]:
        n_per = 200_000
        stream_dicts = []
        stream_tables_pl = []

        for s in range(n_streams):
            ts = np.sort(rng.integers(0, 10**12, n_per))
            syms = rng.choice(symbols, n_per)
            t = pa.table({
                "timestamp": pa.array(ts, type=pa.int64()),
                "symbol": pa.array(syms),
                f"val_{s}": pa.array(rng.uniform(0, 1, n_per)),
            })
            stream_dicts.append({
                "table": t,
                "prefix": f"s{s}_",
                "direction": "backward",
            })
            stream_tables_pl.append(pl.from_arrow(t))

        # Rust: parallel multi-stream
        rust_ms = bench(lambda: flowstate_core.align_streams(
            primary, stream_dicts, on="timestamp", by="symbol"
        )) * 1000

        # Polars: sequential joins (Polars has no multi-stream API)
        def polars_sequential():
            result = primary_pl
            for s, tpl in enumerate(stream_tables_pl):
                result = result.join_asof(
                    tpl, on="timestamp", by="symbol", strategy="backward", suffix=f"_s{s}"
                )
            return result

        polars_ms = bench(polars_sequential) * 1000

        ratio = rust_ms / polars_ms
        print(f"{n_streams:>8d} {n_left:>10,d} {n_per:>11,d} {rust_ms:>9.1f} {polars_ms:>10.1f} {ratio:>6.2f}x")


def bench_directions():
    print()
    print("=" * 72)
    print("JOIN DIRECTIONS (1M left, 500K right, 1000 symbols)")
    print("=" * 72)
    print(f"{'Direction':>10s} {'Rust ms':>9s} {'Polars ms':>10s} {'Ratio':>7s}")
    print("-" * 72)

    left, right = make_data(1_000_000, 500_000, 1_000)
    left_pl = pl.from_arrow(left)
    right_pl = pl.from_arrow(right)

    for direction, polars_strategy in [("backward", "backward"), ("forward", "forward"), ("nearest", "nearest")]:
        rust_ms = bench(lambda: flowstate_core.asof_join(
            left, right, on="timestamp", by="symbol", direction=direction
        )) * 1000

        polars_ms = bench(lambda: left_pl.join_asof(
            right_pl, on="timestamp", by="symbol", strategy=polars_strategy
        )) * 1000

        ratio = rust_ms / polars_ms
        print(f"{direction:>10s} {rust_ms:>9.1f} {polars_ms:>10.1f} {ratio:>6.2f}x")


if __name__ == "__main__":
    print("FlowState Temporal Join Engine — Benchmark Suite")
    print(f"Rust kernel: flowstate_core v{flowstate_core.__version__ if hasattr(flowstate_core, '__version__') else '0.1.0'}")
    print()

    bench_ungrouped()
    bench_single_join()
    bench_directions()
    bench_multi_stream()

    print()
    print("Note: Ratio < 1.0 means FlowState is faster than Polars.")
    print("      Ratio > 1.0 means Polars is faster.")
