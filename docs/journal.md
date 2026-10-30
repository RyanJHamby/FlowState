# FlowState: engineering journal

Dated notes on what I tried, what happened, and why I made each call, including
the dead ends. Entries are appended with `pow log` and `pow run`, so dates match
commit history. Raw outputs for any number quoted in the README live in
[`results/`](../results/).

## 2026-10-04 17:29 EDT (at 0df6f05)

Ran `asof-join-vs-polars-pandas` → [results/2026-10-04-1729-asof-join-vs-polars-pandas](../results/2026-10-04-1729-asof-join-vs-polars-pandas) (exit 0, 14s). Notes: first reproducible head-to-head. Polars is ~1.4x faster than FlowState on grouped backward as-of (1M x 500K and 10M x 5M, 1K symbols); pandas merge_asof does 10M in 0.76 s, so "pandas falls over at 10M" was wrong. The earlier README claim of 1.8x over Polars did not reproduce and was removed. Suspect the earlier number came from a stale (March) flowstate_core build or a different dataset; not verified.

## 2026-10-04 17:29 EDT (at 1b190c6)

Chose separate GitHub environments (pypi, pypi-core) per package: PyPI allows one pending publisher per (repo, workflow, environment). Published flowstate-asof and flowstate-asof-core 0.1.0 via trusted publishing. macos-13 runner is retired; use macos-15-intel.

## 2026-10-04 17:29 EDT (at 1b190c6)

Fixed Linux CI: price_level.h used std::vector without <vector> (worked on macOS libc++, broke GCC). Dead end avoided: DuckDB ASOF JOIN did not finish at 1M x 1K symbols, so it is opt-in in reproduce.py.

## 2026-10-30 14:12 EDT (at fe67b08)

Ran `asof-join-spread` → [results/2026-10-30-1412-asof-join-spread](../results/2026-10-30-1412-asof-join-spread) (exit 0, 21s). Notes: _TODO_

## 2026-10-30 14:12 EDT (at fe67b08)

Re-ran the as-of join benchmark with min/max (reproduce.py now prints them), 10 runs on Apple M4 laptop. FlowState is slower than Polars in both captures: 0.74x at 1M rows and 0.70x at 10M here, 0.71x at both on 2026-10-04. Ranges across the 10 runs do not overlap between the two engines. Absolute times drifted between sessions (Polars 8.7 ms on 10-04, 9.9 ms on 10-30 at 1M rows), so only same-run comparisons are meaningful. Single machine, synthetic data, no thermal or power-state control recorded.
