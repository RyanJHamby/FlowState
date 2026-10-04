# FlowState: engineering journal

Dated notes on what I tried, what happened, and why I made each call, including
the dead ends. Entries are appended with `pow log` and `pow run`, so dates match
commit history. Raw outputs for any number quoted in the README live in
[`results/`](../results/).

## 2026-10-04 17:29 EDT (at 0df6f05)

Ran `asof-join-vs-polars-pandas` → [results/2026-10-04-1729-asof-join-vs-polars-pandas](../results/2026-10-04-1729-asof-join-vs-polars-pandas) (exit 0, 14s). Notes: first reproducible head-to-head. Polars is ~1.4x faster than FlowState on grouped backward as-of (1M x 500K and 10M x 5M, 1K symbols); pandas merge_asof does 10M in 0.76 s, so "pandas falls over at 10M" was wrong. The earlier README claim of 1.8x over Polars did not reproduce and was removed. Suspect the earlier number came from a stale (March) flowstate_core build or a different dataset; not verified.
