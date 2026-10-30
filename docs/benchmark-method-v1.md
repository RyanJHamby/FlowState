# Benchmark method v1 (proposal)

Status: **Proposed.** The owner decides the workload matrix before any spec depends on it. Nothing here is a result.

A benchmark result is a dated snapshot. It is only comparable to another result if both used the same *method version*. This
file defines method v1 so that results taken months apart, after FlowState and its competitors change, stay comparable, and so
that a reader can see exactly what was and was not controlled.

## Question
How long does a backward, grouped as-of join take in FlowState, and how does it compare with other engines doing the same work
on the same inputs, on named hardware?

## Engines and what "same work" means
- FlowState: the Rust kernel called directly, and the full Python wrapper, reported separately.
- Polars `join_asof`, DuckDB `ASOF JOIN`, pandas `merge_asof`. Each is configured the way its own documentation recommends for sorted inputs; the configuration used is recorded. Baseline tuning is itself a reviewed change.
- Before any timing, every engine's full output is compared with `assert_same_result` (spec `bench-result-equality`). A run whose engines disagree is invalid and publishes no timings.

## Workload matrix (proposed; owner to confirm)
Synthetic first because it is reproducible from a seed; a public real-data set is added later.
| Axis | Values |
|---|---|
| left rows | 1M, 10M (100M only on hardware with the memory) |
| right : left ratio | 0.5, 1.0 |
| symbols | 1, 100, 1,000, 100,000 |
| symbol skew | uniform, Zipf |
| input order | globally sorted by timestamp; sorted by (symbol, timestamp) |
| direction / tolerance | backward, none; backward with a tolerance |
| grouping | grouped by symbol; ungrouped |

## Measurement
- **Interleaved rounds:** every engine runs once per round in shuffled order; warmup rounds are discarded (spec `bench-interleaved-timing`).
- **Statistic:** median per engine with a 95% bootstrap interval; ratios reported with their own interval.
- **Validity checks that must pass before results are kept:** an A/A check (an engine against itself gives an interval containing 1.0) and a canary (a deliberately slowed engine is detected).
- **What is timed:** the join call only. Input construction, sorting, and generation are excluded and said so.
- **Phases:** conversion, kernel, and wrapper time are reported separately (spec `bench-phase-breakdown`).

## Environment (recorded automatically; spec `bench-env-capture`)
Instance type and AMI id (cloud) or machine model (local); CPU model and microcode; kernel version; governor; turbo; SMT;
transparent hugepages; NUMA layout; ASLR; thread settings (`RAYON_NUM_THREADS`, `POLARS_MAX_THREADS`); pinning; allocator;
Python, Rust, and library versions with lockfile hashes; the build profile and flags of the Rust kernel; git sha and dirty flag.
Anything not recorded is listed as *not controlled* in the result.

## Reporting rules
- Every result names: method version, date, hardware, versions, git sha.
- Raw output is committed with `pow run` at a pinned commit; results are never edited, only superseded by a new dated result and a changelog line.
- A claim names its scope (workload, versions, hardware). "Slower than Polars" is never stated in general.
- Laptop results are labelled as such and are never the headline.

## Not in method v1
Energy and power, GPU feeding, streaming-join latency, memory footprint, cold-cache I/O, and multi-node. Each needs its own method version.
