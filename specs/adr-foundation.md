---
id: adr-foundation
goal: DESIGN.md section 11's nine key decisions exist as numbered ADRs in docs/adr with an index, a template, and a link from DESIGN.md.
acceptance:
  - python -m pytest tests/test_adr_foundation.py --runxfail -q
  - python -m pytest tests/test_adr_foundation.py -q
scope:
  - docs/adr/**
  - DESIGN.md
  - tests/test_adr_foundation.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines (the plain pytest acceptance command fails while the strict xfail marker remains and the work is done)
  - do not add performance numbers, speedups or claims that FlowState beats anything; section 3.2 of DESIGN.md is known to be unreproduced and is a separate spec
  - do not edit DESIGN.md except to add a pointer to docs/adr
risk_tier: 1
max_runs: 3
autonomy: inherit
status: ready
issue: 8
---

## Why
The design trade-offs live in one table in DESIGN.md section 11. Contributors need each decision, its context and its
consequences as a separate, reviewable record, and a place to add the next one.

## What to build
- `docs/adr/0000-template.md` (Status / Context / Decision / Consequences) and `docs/adr/README.md` (an index linking every ADR).
- Nine files `0001-<slug>.md` to `0009-<slug>.md`, one per row of the table in DESIGN.md section 11 (merge-scan vs binary search; ahash with borrowed keys; UnsafeCell parallel writes; TimestampSlice; feature-gated PyO3; pure Python fallback; Arrow PyCapsule interface; watermark emission; deterministic xxhash bucketing).
- Each ADR: title line `# ADR-000N: <title>`, a line `Status: Accepted` (or Proposed/Superseded/Deprecated), and `## Context`, `## Decision`, `## Consequences` sections, at least 150 words. Draw the content from DESIGN.md and the source code, and cite the file in the text. Under Consequences, list the downside honestly (for example the memory cost or the unsafe code).
- Add one line to DESIGN.md pointing to `docs/adr`.

## Facts you cannot infer
- Where a rationale in DESIGN.md says something is faster or better, either drop the claim or restate it as the author's stated reason ("DESIGN.md argues ..."). Section 3.2 of DESIGN.md claims FlowState outperforms Polars; the project's own README says that did not reproduce.
- The test rejects the word "outperform" and any "N x faster/slower" string.

## Log
- 2026-10-30 ready, issue #8
