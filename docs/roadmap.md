# Roadmap: foundations and benchmark harness

Each line becomes one spec and one small PR (spec-writer workflow). Order is by value and dependency. **Filed** specs have an
issue; the rest are drafts and are filed a few at a time, because the autopilot queue is capped. Items marked **owner** touch
`.github/` or require access an agent does not have, so they are written as specs but done by hand.

## Filed (p0)
1. `bench-result-equality` full-result equality check
2. `bench-interleaved-timing` interleaved timing, bootstrap CI, A/A and canary helpers
3. `bench-env-capture` environment capture
4. `bench-phase-breakdown` conversion / kernel / wrapper / Polars split
5. `adr-foundation` nine ADRs from DESIGN.md section 11

## Next: harness (p1)
6. `bench-harness-v1-entrypoint` one command, `python -m benchmarks.run --method v1`, tying 1 to 4 together and writing a JSON schema result plus env file
7. `bench-validity-gate` refuse to keep results when A/A or canary fails
8. `bench-workload-sweep` the matrix in `docs/benchmark-method-v1.md`
9. `bench-thread-scaling` 1, 2, 4, 8 threads for both engines
10. `bench-baseline-tuning` document and encode each baseline's recommended configuration with links
11. `bench-wire-equality-into-reproduce` make `reproduce.py` use 1 and 2

## Next: tests (p1)
12. `test-asof-edge-semantics` duplicate timestamps, ties, nulls, tolerance bounds, empty inputs, single symbol, unsorted input
13. `test-differential-vs-engines` property tests comparing FlowState with Polars, DuckDB, pandas
14. `test-kernel-fuzz` cargo-fuzz or extended proptest on the kernel
15. `test-streaming-batch-parity` randomized parity between streaming and batch joins

## Next: CI and release (p1/p2)
16. `ci-python-314-macos-arm` Python 3.14 and macOS/ARM in the matrix (**owner**: `.github/`)
17. `ci-bench-regression-gate` `benchmarks/compare.py` plus a scheduled workflow with stored baselines (script is an agent spec; workflow is **owner**)
18. `release-wheels` cibuildwheel for the kernel and a release checklist (**owner**)
19. `citation-and-snapshots` `CITATION.cff`, tag and archive process for result snapshots

## Diagnosis and experiments (p2, need server hardware)
20. `aws-bench-launcher` cloud-init and tuning script that applies governor, hugepages, pinning and records them (**owner** runs it)
21. `perf-hypotheses` `docs/perf/hypotheses.md`: one row per suspected cost (sort skipping, key hashing, scheduling, output building, wrapper, allocator, vector instructions) with how to test it
22. `exp-wrapper-fast-path` remove avoidable work from the wrapper, decided by spec 4
23. `exp-allocator`, `exp-target-cpu`, `exp-lto-pgo`, `exp-panic-abort`: one experiment per spec, each a change plus a before/after capture on named hardware
24. `design-3-2-reconcile` rewrite DESIGN.md section 3.2 to match the captures (needs the owner's decision)

## Project health and growth (p2)
25. `repo-scope-split` move the C++ order book out of the main package, or into an `experimental/` area
26. `gpu-mark-experimental` label the GPU path experimental until it has a GPU test
27. `readme-restructure` lead with what FlowState is for, who should use it, and when not to
28. `comparison-page` honest "when to use which" against Polars, DuckDB, pandas, using only captured results
29. `contributing-and-good-first-issues` contributor guide, labeled starter issues, issue templates
30. `docs-site` a small docs site with the ADRs, method, and results index
31. `results-index` a living `/benchmarks` page listing every result with method version, hardware, date and status
32. `security-and-deps` `cargo deny`, `pip-audit`, supply-chain checks

## Rules for every spec
Small signed commits; a failing test first; no number without a `pow run` capture; hardware named; a `pow log` entry for each
decision or failed experiment.
