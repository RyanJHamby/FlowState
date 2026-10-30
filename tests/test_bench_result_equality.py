"""Spec bench-result-equality: a benchmark may only time engines that return the same result."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-result-equality (remove this marker when implemented)"
)


def _h():
    return importlib.import_module("benchmarks.harness")


def _base() -> dict:
    return {
        "timestamp": np.array([1, 2, 3, 4], dtype=np.int64),
        "symbol": ["A", "B", "A", "B"],
        "px": np.array([1.5, 2.5, 3.5, 4.5]),
        "qpx": np.array([0.5, 2.0, 3.0, 4.0]),
    }


@XFAIL
def test_same_content_passes_across_container_types():
    d = _base()
    h = _h()
    h.assert_same_result(pl.DataFrame(d), pa.table(d))
    h.assert_same_result(pa.table(d), pd.DataFrame(d))


@XFAIL
def test_row_order_does_not_matter():
    d = _base()
    shuffled = {
        k: (np.asarray(v)[[3, 1, 0, 2]] if k != "symbol" else [v[i] for i in [3, 1, 0, 2]])
        for k, v in d.items()
    }
    _h().assert_same_result(pl.DataFrame(d), pl.DataFrame(shuffled))


@XFAIL
def test_changed_value_is_reported_with_column_and_row():
    d = _base()
    bad = dict(d, px=np.array([1.5, 2.5, 99.0, 4.5]))
    with pytest.raises(AssertionError) as e:
        _h().assert_same_result(pl.DataFrame(d), pl.DataFrame(bad))
    msg = str(e.value)
    assert "px" in msg and "99.0" in msg


@XFAIL
def test_missing_and_extra_columns_are_reported():
    d = _base()
    h = _h()
    with pytest.raises(AssertionError, match="missing.*qpx"):
        h.assert_same_result(
            pl.DataFrame(d), pl.DataFrame({k: v for k, v in d.items() if k != "qpx"})
        )
    with pytest.raises(AssertionError, match="extra.*zzz"):
        h.assert_same_result(
            pl.DataFrame({k: v for k, v in d.items() if k != "qpx"}),
            pl.DataFrame(dict(d, zzz=[1, 2, 3, 4])).drop("qpx"),
        )


@XFAIL
def test_dtype_and_row_count_mismatch_fail():
    d = _base()
    h = _h()
    with pytest.raises(AssertionError, match="dtype"):
        h.assert_same_result(pl.DataFrame(d), pl.DataFrame(dict(d, px=d["px"].astype(np.float32))))
    with pytest.raises(AssertionError, match="rows"):
        h.assert_same_result(pl.DataFrame(d), pl.DataFrame(d).head(3))


@XFAIL
def test_null_equals_null_but_not_a_value():
    d = _base()
    with_null = pl.DataFrame(d).with_columns(
        pl.when(pl.col("timestamp") == 2).then(None).otherwise(pl.col("qpx")).alias("qpx")
    )
    h = _h()
    h.assert_same_result(with_null, with_null.clone())
    with pytest.raises(AssertionError):
        h.assert_same_result(with_null, pl.DataFrame(d))


@XFAIL
def test_nan_equals_nan_but_not_null():
    # pandas cannot tell NaN from null, so this rule is for Polars and Arrow inputs.
    nan_df = pl.DataFrame(dict(_base(), qpx=np.array([0.5, np.nan, 3.0, 4.0])))
    null_df = nan_df.with_columns(pl.col("qpx").fill_nan(None))
    h = _h()
    h.assert_same_result(nan_df, nan_df.clone())
    with pytest.raises(AssertionError):
        h.assert_same_result(nan_df, null_df)


@XFAIL
def test_timestamp_compares_by_integer_nanoseconds_not_dtype():
    d = _base()
    as_ts = pa.table(dict(d, timestamp=pa.array(d["timestamp"], type=pa.timestamp("ns", tz="UTC"))))
    _h().assert_same_result(pl.DataFrame(d), as_ts)


@XFAIL
def test_real_engines_agree_on_a_small_dataset():
    from benchmarks import reproduce as r
    from flowstate.prism.alignment import AsOfConfig, as_of_join

    ts = pa.timestamp("ns", tz="UTC")
    left, right = r.make_data(2000, 5, seed=42), r.make_data(1000, 5, seed=99)
    lp = pl.DataFrame(left)
    rp = pl.DataFrame(
        {"timestamp": right["timestamp"], "qpx": right["px"], "symbol": right["symbol"]}
    )
    expected = lp.join_asof(rp, on="timestamp", by="symbol", strategy="backward")
    la = pa.table(
        {
            "timestamp": pa.array(left["timestamp"], type=ts),
            "px": left["px"],
            "symbol": pa.array(left["symbol"]),
        }
    )
    ra = pa.table(
        {
            "timestamp": pa.array(right["timestamp"], type=ts),
            "qpx": right["px"],
            "symbol": pa.array(right["symbol"]),
        }
    )
    actual, _ = as_of_join(la, ra, config=AsOfConfig(direction="backward"))
    _h().assert_same_result(expected, actual)
