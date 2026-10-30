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

## 2026-10-30 14:23 EDT (at fdc202b)

Ran `asof-join-march-build-3c7486a` → [results/2026-10-30-1423-asof-join-march-build-3c7486a](../results/2026-10-30-1423-asof-join-march-build-3c7486a) (exit 0, 10s). Notes: _TODO_

## 2026-10-30 14:23 EDT (at fdc202b)

Tested the 'stale March build' explanation: built the kernel and Python package at 3c7486a (README that claimed 1.8x) in a separate worktree with the same pinned Polars 1.38.1/pandas/pyarrow and Python 3.14.3, and ran today's reproduce.py against it, 10 runs, 1M x 500K rows, 1K symbols. FlowState 12.9 ms vs Polars 8.9 ms: still slower, ranges do not overlap. So the March kernel does not explain the 1.8x claim. Not ruled out: a different dataset, or Polars being faster now than in March (the March README says only 'Polars 1.x', no version). The March kernel was built with today's Rust toolchain, and the benchmark script is today's, not March's. Note: pow records git sha of the main checkout; the code under test was the 3c7486a worktree.

## 2026-10-30 14:49 EDT (at b8f9fc3)

Planned the benchmark/foundations work as specs. Wrote five specs with strict-xfail failing tests (bench-result-equality, bench-interleaved-timing, bench-env-capture, bench-phase-breakdown, adr-foundation), a method v1 proposal and a ranked roadmap of the rest. Each test was shown failing for the right reason (missing module) and shown satisfiable with a throwaway reference implementation, which was then deleted. Caught three test bugs on the way: a pandas NaN-vs-null assumption, two ADR tests that passed vacuously on an empty directory, and over-long lines that would have failed CI lint. Found that Polars and FlowState outputs already agree on a small dataset, but with different timestamp dtypes (Int64 vs timestamp[ns, UTC]). FlowState is not onboarded to autopilot (no .autopilot/), so the handshake was run against drillpad's repo.yaml and is provisional.

## 2026-10-30 16:52 EDT (at 2d3e6df)

Filed the five spec issues (#4 bench-result-equality, #5 bench-interleaved-timing, #6 bench-env-capture, #7 bench-phase-breakdown, #8 adr-foundation). Ryan ran gh issue create himself because the call was blocked in my session. Specs marked ready with their issue numbers; ap specs shows no problems; handshake still ACCEPTED (provisional: FlowState has no .autopilot yet).
