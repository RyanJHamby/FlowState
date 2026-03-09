"""Benchmark: vectorized numpy.searchsorted vs Python bisect as-of join.

Measures wall-clock time for backward as-of joins at varying dataset sizes
to quantify the actual speedup of the vectorized implementation.

Three benchmark levels:
1. Raw matching (searchsorted vs bisect loop) — isolates the core operation
2. Full as-of join pipeline (Arrow tables, single symbol) — includes gather overhead
3. Multi-symbol grouped join — where per-symbol Python loops dominated the old impl

The old implementation used:
- bisect.bisect_right in a Python for-loop per row
- Python for-loop per symbol group
- Row-by-row gather into result columns

The new implementation uses:
- numpy.searchsorted (single vectorized call)
- Dictionary-based grouping with numpy advanced indexing
- pa.Array.take with validity bitmaps
"""

from __future__ import annotations

import bisect
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from flowstate.prism.alignment import AsOfConfig, as_of_join

TS = pa.timestamp("ns", tz="UTC")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_table(
    n: int,
    symbols: list[str],
    col_prefix: str = "",
    seed: int = 42,
) -> pa.Table:
    """Create a table with sorted timestamps, multiple symbols, realistic ticks."""
    rng = np.random.default_rng(seed)
    n_per_sym = n // len(symbols)

    all_ts, all_prices, all_syms = [], [], []
    for sym in symbols:
        ts = np.cumsum(rng.integers(50, 150, size=n_per_sym)).astype(np.int64)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n_per_sym))
        all_ts.append(ts)
        all_prices.append(prices)
        all_syms.extend([sym] * n_per_sym)

    # Interleave and sort by timestamp (simulates real multi-symbol feed)
    ts_arr = np.concatenate(all_ts)
    price_arr = np.concatenate(all_prices)
    order = np.argsort(ts_arr, kind="mergesort")
    ts_arr = ts_arr[order]
    price_arr = price_arr[order]
    sym_arr = [all_syms[i] for i in order]

    return pa.table(
        {
            "timestamp": pa.array(ts_arr, type=TS),
            f"{col_prefix}price": pa.array(price_arr, type=pa.float64()),
            f"{col_prefix}size": pa.array(rng.uniform(10, 1000, len(ts_arr)), type=pa.float64()),
            "symbol": pa.array(sym_arr, type=pa.string()),
        }
    )


# ---------------------------------------------------------------------------
# Baseline: Python bisect grouped join (what we replaced)
# ---------------------------------------------------------------------------


def _bisect_grouped_asof(
    left: pa.Table, right: pa.Table, by: str = "symbol"
) -> np.ndarray:
    """Old-style grouped as-of join: Python loops over symbols and rows."""
    left_ts = left.column("timestamp").cast(pa.int64()).to_numpy()
    right_ts = right.column("timestamp").cast(pa.int64()).to_numpy()
    left_syms = left.column(by).to_pylist()
    right_syms = right.column(by).to_pylist()

    # Group by symbol (Python dict)
    right_groups: dict[str, list[int]] = {}
    for i, s in enumerate(right_syms):
        right_groups.setdefault(s, []).append(i)

    indices = np.full(len(left_ts), -1, dtype=np.int64)

    # Per-symbol bisect loop
    for sym, right_idxs in right_groups.items():
        right_sub_ts = [right_ts[j] for j in right_idxs]  # Python list for bisect
        for i, s in enumerate(left_syms):
            if s != sym:
                continue
            t = left_ts[i]
            j = bisect.bisect_right(right_sub_ts, t) - 1
            if j >= 0:
                indices[i] = right_idxs[j]

    return indices


def _vectorized_grouped_asof(
    left: pa.Table, right: pa.Table, by: str = "symbol"
) -> np.ndarray:
    """Current vectorized grouped as-of join."""
    left_ts = left.column("timestamp").cast(pa.int64()).to_numpy()
    right_ts = right.column("timestamp").cast(pa.int64()).to_numpy()
    left_syms = left.column(by).to_pylist()
    right_syms = right.column(by).to_pylist()

    # Group by symbol using numpy
    left_sym_dict: dict[str, np.ndarray] = {}
    right_sym_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i, s in enumerate(left_syms):
        left_sym_dict.setdefault(s, []).append(i)
    for i, s in enumerate(right_syms):
        right_sym_dict.setdefault(s, []).append(i)

    indices = np.full(len(left_ts), -1, dtype=np.int64)

    for sym, left_idxs_list in left_sym_dict.items():
        if sym not in right_sym_dict:
            continue
        left_idxs = np.array(left_idxs_list, dtype=np.int64)
        right_idxs = np.array(right_sym_dict[sym], dtype=np.int64)

        left_sub = left_ts[left_idxs]
        right_sub = right_ts[right_idxs]

        # Vectorized matching for this symbol
        pos = np.searchsorted(right_sub, left_sub, side="right").astype(np.int64) - 1
        valid = pos >= 0
        indices[left_idxs[valid]] = right_idxs[pos[valid]]

    return indices


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _time_fn(fn, *args, warmup: int = 2, runs: int = 5) -> float:
    """Time a function, return median wall-clock ms."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # Median


def bench_raw_matching(sizes: list[tuple[int, int]]):
    """Benchmark 1: Raw index matching, no grouping."""
    print("=" * 76)
    print("Benchmark 1: Raw Matching (bisect loop vs searchsorted)")
    print(f"{'LEFT':>10} {'RIGHT':>10} {'BISECT (ms)':>14} {'VECTOR (ms)':>14} {'SPEEDUP':>10}")
    print("-" * 76)

    for n_left, n_right in sizes:
        rng = np.random.default_rng(42)
        left_ts = np.sort(rng.integers(0, n_left * 100, size=n_left))
        right_ts = np.sort(rng.integers(0, n_right * 100, size=n_right))

        # Verify correctness
        def bisect_fn():
            idx = np.full(n_left, -1, dtype=np.int64)
            for i, t in enumerate(left_ts):
                j = bisect.bisect_right(right_ts, t) - 1
                if j >= 0:
                    idx[i] = j
            return idx

        def vector_fn():
            pos = np.searchsorted(right_ts, left_ts, side="right").astype(np.int64) - 1
            pos[pos < 0] = -1
            return pos

        assert np.array_equal(bisect_fn(), vector_fn()), "Results differ!"

        bisect_ms = _time_fn(bisect_fn)
        vector_ms = _time_fn(vector_fn)
        speedup = bisect_ms / vector_ms if vector_ms > 0 else float("inf")
        print(f"{n_left:>10,} {n_right:>10,} {bisect_ms:>13.2f} {vector_ms:>13.2f} {speedup:>9.1f}x")


def bench_grouped_matching(n_rows: int, symbol_counts: list[int]):
    """Benchmark 2: Grouped join (per-symbol loops vs vectorized grouping)."""
    print()
    print("=" * 76)
    print(f"Benchmark 2: Grouped Join ({n_rows:,} rows/side, varying symbol count)")
    print(f"{'SYMBOLS':>10} {'BISECT (ms)':>14} {'VECTOR (ms)':>14} {'SPEEDUP':>10}")
    print("-" * 76)

    for n_sym in symbol_counts:
        symbols = [f"SYM{i:04d}" for i in range(n_sym)]
        left = _make_table(n_rows, symbols, col_prefix="trade_", seed=42)
        right = _make_table(n_rows * 2, symbols, col_prefix="quote_", seed=99)

        # Verify correctness (spot check — both should match on same rows)
        baseline = _bisect_grouped_asof(left, right)
        vectorized = _vectorized_grouped_asof(left, right)
        matched_base = np.sum(baseline >= 0)
        matched_vec = np.sum(vectorized >= 0)
        assert matched_base == matched_vec, f"Match count differs: {matched_base} vs {matched_vec}"

        bisect_ms = _time_fn(_bisect_grouped_asof, left, right, runs=3)
        vector_ms = _time_fn(_vectorized_grouped_asof, left, right, runs=3)
        speedup = bisect_ms / vector_ms if vector_ms > 0 else float("inf")
        print(f"{n_sym:>10,} {bisect_ms:>13.2f} {vector_ms:>13.2f} {speedup:>9.1f}x")


def bench_full_pipeline(sizes: list[int], n_symbols: int = 50):
    """Benchmark 3: End-to-end as_of_join (the public API)."""
    print()
    print("=" * 76)
    print(f"Benchmark 3: Full as_of_join Pipeline ({n_symbols} symbols)")
    print(f"{'ROWS':>10} {'TIME (ms)':>14} {'ROWS/SEC':>14}")
    print("-" * 76)

    cfg = AsOfConfig(right_prefix="quote_", direction="backward")
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

    for n in sizes:
        left = _make_table(n, symbols, col_prefix="trade_", seed=42)
        right = _make_table(n * 2, symbols, col_prefix="quote_", seed=99)
        left = left.rename_columns(["timestamp", "trade_price", "trade_size", "symbol"])
        right = right.rename_columns(["timestamp", "quote_price", "quote_size", "symbol"])

        def run_join():
            return as_of_join(left, right, config=cfg)

        elapsed_ms = _time_fn(run_join, runs=3)
        rows_per_sec = n / (elapsed_ms / 1000)
        print(f"{n:>10,} {elapsed_ms:>13.2f} {rows_per_sec:>13,.0f}")


if __name__ == "__main__":
    print("Benchmark: Vectorized vs Bisect As-Of Join")
    print(f"(median of runs, after warmup)\n")

    bench_raw_matching([
        (1_000, 1_000),
        (10_000, 10_000),
        (100_000, 100_000),
        (500_000, 500_000),
        (1_000_000, 1_000_000),
    ])

    bench_grouped_matching(
        n_rows=100_000,
        symbol_counts=[1, 10, 50, 200, 500],
    )

    bench_full_pipeline([10_000, 100_000, 500_000])
