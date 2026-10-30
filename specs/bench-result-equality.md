---
id: bench-result-equality
goal: A benchmark can assert that two engines returned the same result (columns, dtypes, row count, values), so only equal work is ever timed.
acceptance:
  - python -m pytest tests/test_bench_result_equality.py --runxfail -q
  - python -m pytest tests/test_bench_result_equality.py -q
scope:
  - benchmarks/harness.py
  - tests/test_bench_result_equality.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines (the plain pytest acceptance command fails while the strict xfail marker remains and the work is done)
  - no new runtime dependencies (numpy, polars, pyarrow, pandas are already installed)
  - do not modify src/flowstate or flowstate-core; this is benchmark tooling only
  - do not wire the check into benchmarks/reproduce.py (a later spec does that)
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
Today `benchmarks/reproduce.py` checks only the `qpx` column of FlowState's output against Polars before timing. Polars
returns a fully materialized frame; FlowState's wrapper returns a different table. Nobody has shown the timed work is equal.
A skeptic will say so, and the published post is currently open to that attack.

## What to build
`benchmarks/harness.py` with `assert_same_result(expected, actual, *, key=("timestamp", "symbol"))`. Accept `polars.DataFrame`,
`pyarrow.Table` or `pandas.DataFrame`. Rules (the tests pin every one):
- compare the same set of columns; raise `AssertionError` naming `missing` and `extra` columns
- equal row count, else the message contains `rows`
- row order is ignored: sort both sides canonically before comparing values
- timestamp columns compare by their integer nanosecond value; Int64 and `timestamp[ns]` (with or without tz) are the same thing
- every other column must have the same dtype, else the message contains `dtype`
- values compare exactly (no tolerance: a join copies values, it does not compute them)
- null equals null; NaN equals NaN; NaN does not equal null. pandas cannot tell NaN from null, so treat NaN as null for pandas input only
- the first mismatch message names the column and the row and shows both values (the test looks for `px` and `99.0`)

## Facts you cannot infer
- On a 2,000 x 1,000 row dataset with 5 symbols, Polars `join_asof` and `flowstate.prism.alignment.as_of_join` already return the same columns, values and row order (verified by hand). Polars returns `timestamp` as Int64, FlowState as `timestamp[ns, tz=UTC]`.
- `benchmarks/` is now a package (`__init__.py` exists). `benchmarks/reproduce.py` has `make_data(n, n_symbols, seed)`; the last test imports it.
- Style: type hints, short functions, `from __future__ import annotations`, ruff clean (`ruff check benchmarks tests`).

## Log
