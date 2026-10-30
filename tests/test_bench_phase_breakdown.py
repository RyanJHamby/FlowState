"""Spec bench-phase-breakdown: split the as-of join into conversion, kernel and wrapper time."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-phase-breakdown (remove this marker when implemented)"
)
REPO = Path(__file__).resolve().parents[1]
PHASES = ("convert_in", "kernel", "wrapper", "polars")


def _p():
    return importlib.import_module("benchmarks.phases")


@XFAIL
def test_measure_phases_reports_every_phase_with_spread():
    out = _p().measure_phases(n_left=20_000, n_right=10_000, n_symbols=20, runs=3, warmup=1)
    for ph in PHASES:
        s = out["phases"][ph]
        assert len(s["samples_ms"]) == 3 and all(x > 0 for x in s["samples_ms"])
        assert s["min_ms"] <= s["median_ms"] <= s["max_ms"]
    assert out["params"] == {
        "n_left": 20_000,
        "n_right": 10_000,
        "n_symbols": 20,
        "runs": 3,
        "warmup": 1,
    }


@XFAIL
def test_wrapper_overhead_is_wrapper_minus_kernel_and_phases_are_not_double_counted():
    out = _p().measure_phases(n_left=20_000, n_right=10_000, n_symbols=20, runs=5, warmup=1)
    ph = out["phases"]
    assert out["wrapper_overhead_ms"] == pytest.approx(
        ph["wrapper"]["median_ms"] - ph["kernel"]["median_ms"]
    )
    # The kernel call alone cannot take longer than the whole wrapper call by more than noise.
    assert ph["kernel"]["median_ms"] <= ph["wrapper"]["median_ms"] * 1.5


@XFAIL
def test_cli_prints_valid_json():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.phases",
            "--left",
            "20000",
            "--right",
            "10000",
            "--symbols",
            "20",
            "--runs",
            "3",
            "--warmup",
            "1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert set(data["phases"]) == set(PHASES)


@XFAIL
def test_kernel_and_wrapper_return_the_same_row_count():
    out = _p().measure_phases(n_left=5_000, n_right=2_000, n_symbols=10, runs=2, warmup=0)
    assert out["rows_out"] == 5_000
