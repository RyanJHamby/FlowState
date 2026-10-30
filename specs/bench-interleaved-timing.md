---
id: bench-interleaved-timing
goal: Engines are timed in interleaved, shuffled rounds with bootstrap confidence intervals, and the module offers an A/A check and a ratio CI so a canary can be detected.
acceptance:
  - python -m pytest tests/test_bench_interleaved_timing.py --runxfail -q
  - python -m pytest tests/test_bench_interleaved_timing.py -q
scope:
  - benchmarks/timing.py
  - tests/test_bench_interleaved_timing.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines (the plain pytest acceptance command fails while the strict xfail marker remains and the work is done)
  - no new runtime dependencies (numpy, polars, pyarrow, pandas are already installed)
  - do not modify src/flowstate or flowstate-core; this is benchmark tooling only
  - do not change benchmarks/reproduce.py
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
`reproduce.py` times each engine in its own block (all Polars runs, then all FlowState runs), so drift between sessions or
thermal changes land on one engine. Absolute times drifted about 14% between two sessions on identical code. Interleaving
within a run and reporting intervals is the minimum a skeptic expects.

## What to build (`benchmarks/timing.py`, no new dependencies; numpy is fine)
- `interleaved_samples(engines: dict[str, Callable[[], object]], warmup: int, runs: int, seed: int = 0, timer=time.perf_counter) -> dict[str, list[float]]`: each round calls every engine once in an order shuffled by `random.Random(seed)`. Discard `warmup` rounds. Return per-engine milliseconds, one value per run. The `timer` argument returns seconds and exists so tests can use a fake clock.
- `bootstrap_ci(samples, stat="median", confidence=0.95, n_resamples=2000, seed=0) -> (point, lo, hi)` using `numpy.random.default_rng(seed)`. Deterministic for a seed. Constant samples give `lo == hi == point`.
- `ratio_ci(a, b, confidence=0.95, n_resamples=2000, seed=0) -> (ratio, lo, hi)`: ratio of medians `a/b`, bootstrapped over paired indices.
- `aa_check(fn, warmup, runs, seed=0, timer=...) -> (ratio, lo, hi)`: runs the same callable under two names, interleaved, and returns `ratio_ci`. A healthy harness gives an interval containing 1.0.

## Facts you cannot infer
- The tests use a fake clock; do not read wall time anywhere except through `timer`.
- Median is the only `stat` required; reject others with `ValueError`.

## Log
