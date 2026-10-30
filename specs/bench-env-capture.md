---
id: bench-env-capture
goal: Every benchmark run can record the machine and settings it ran under in one JSON file, with 'unknown' (never omitted) for anything that cannot be read.
acceptance:
  - python -m pytest tests/test_bench_env_capture.py --runxfail -q
  - python -m pytest tests/test_bench_env_capture.py -q
scope:
  - benchmarks/envcapture.py
  - tests/test_bench_env_capture.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines (the plain pytest acceptance command fails while the strict xfail marker remains and the work is done)
  - no new runtime dependencies (numpy, polars, pyarrow, pandas are already installed)
  - do not modify src/flowstate or flowstate-core; this is benchmark tooling only
  - do not shell out to anything that needs root or the network
risk_tier: 2
max_runs: 3
autonomy: inherit
status: ready
issue: 6
---

## Why
The current results record CPU, RAM and library versions but not power state, governor, thread settings, SMT, NUMA, huge
pages or ASLR. A server run is only credible if those are captured automatically and not typed by hand.

## What to build (`benchmarks/envcapture.py`)
`capture_env() -> dict` and `write_env(path)`. Required keys are listed in `REQUIRED` in the test. Rules:
- all file reads from `/sys` and `/proc` go through ONE helper `_read_sys(path) -> str | None` (tests monkeypatch it)
- a value that cannot be read is the string `"unknown"`, never `None`, never omitted, never an empty string
- governor: `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
- transparent hugepages: `/sys/kernel/mm/transparent_hugepage/enabled`, return only the bracketed active value (`madvise` from `always [madvise] never`)
- SMT: `/sys/devices/system/cpu/smt/active`, `"0"` is `"off"`, `"1"` is `"on"`
- ASLR: `/proc/sys/kernel/randomize_va_space`, `"0"` is `"off"`, any other readable value is `"on"`
- `threads`: a dict of `RAYON_NUM_THREADS` and `POLARS_MAX_THREADS`, value `"unset"` when not in the environment
- `packages`: versions of polars, pandas, pyarrow, numpy and flowstate from `importlib.metadata` (`"unknown"` if absent)
- `git_sha` and `git_dirty` come from `git -C <repo root>`; `git_dirty` is a real `bool`
- `python_free_threaded`: whether the interpreter is a free-threaded build (`sysconfig.get_config_var("Py_GIL_DISABLED")`). A free-threaded build once broke our install, so this is worth recording
- on macOS most Linux-only fields are `"unknown"`; that is fine
- `memory_gb`, `numa_nodes`, `rustc`, `cpu_model` use whatever stdlib or `subprocess` call works on Linux and macOS, else `"unknown"`

## Log
- 2026-10-30 ready, issue #6
