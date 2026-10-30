---
id: bench-phase-breakdown
goal: A script reports, as JSON, how long the as-of join spends in input conversion, the Rust kernel alone, the full Python wrapper, and Polars on the same data.
acceptance:
  - python -m pytest tests/test_bench_phase_breakdown.py --runxfail -q
  - python -m pytest tests/test_bench_phase_breakdown.py -q
scope:
  - benchmarks/phases.py
  - tests/test_bench_phase_breakdown.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines (the plain pytest acceptance command fails while the strict xfail marker remains and the work is done)
  - no new runtime dependencies (numpy, polars, pyarrow, pandas are already installed)
  - do not modify src/flowstate or flowstate-core; this is benchmark tooling only
  - do not try to make FlowState faster; this spec only measures where the time goes
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
`flowstate.prism.alignment.as_of_join` does more than call the Rust kernel (it counts nulls and builds stats around it),
and that work sits inside the timed region, while Polars' timed call has no equivalent. Before changing any code we need to
know how much of the measured gap is wrapper overhead versus kernel time.

## What to build (`benchmarks/phases.py`)
- `measure_phases(n_left, n_right, n_symbols, runs, warmup) -> dict` using `benchmarks.reproduce.make_data` (seeds 42 and 99). Return
  `{"params": {...exactly the five arguments...}, "phases": {"convert_in": S, "kernel": S, "wrapper": S, "polars": S}, "wrapper_overhead_ms": float, "rows_out": int}`
  where each `S` is `{"samples_ms": [...], "min_ms", "median_ms", "max_ms"}`, `wrapper_overhead_ms = wrapper.median_ms - kernel.median_ms`, and `rows_out` is the row count of the wrapper's output.
- `convert_in`: building the two Arrow tables (timestamp as `timestamp[ns, tz=UTC]`) from the numpy arrays.
- `kernel`: `flowstate_core.asof_join(left_table, right_table, on="timestamp", by="symbol", direction="backward")` on the tables from `convert_in`. Read `src/flowstate/prism/alignment.py` around the `_HAS_RUST` branch for the exact call the wrapper makes, and make the same call.
- `wrapper`: `flowstate.prism.alignment.as_of_join(left, right, config=AsOfConfig(direction="backward"))`.
- `polars`: `join_asof(..., on="timestamp", by="symbol", strategy="backward")` on pre-built frames.
- `python -m benchmarks.phases --left N --right N --symbols N --runs N --warmup N` prints that dict as one JSON document on stdout and exits 0.

## Facts you cannot infer
- The Rust kernel must be installed (`pip install ./flowstate-core` or `maturin develop --release`); CI already does this.
- Time one call per sample with `time.perf_counter`; no need for the interleaved harness yet.

## Log
