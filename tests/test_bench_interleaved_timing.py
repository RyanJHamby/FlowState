"""Spec bench-interleaved-timing: interleaved rounds, bootstrap intervals, A/A and canary checks."""

from __future__ import annotations

import importlib

import pytest

XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-interleaved-timing (remove this marker when implemented)"
)


def _t():
    return importlib.import_module("benchmarks.timing")


class FakeClock:
    """Deterministic clock: each engine call advances it by that engine's fixed cost (seconds)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def engine(self, cost_s: float, log: list | None = None, name: str = ""):
        def run():
            self.now += cost_s
            if log is not None:
                log.append(name)

        return run


@XFAIL
def test_samples_have_one_value_per_run_in_ms_and_warmup_is_discarded():
    clock, log = FakeClock(), []
    out = _t().interleaved_samples(
        {"a": clock.engine(0.002, log, "a"), "b": clock.engine(0.004, log, "b")},
        warmup=2,
        runs=5,
        seed=1,
        timer=clock,
    )
    assert out["a"] == pytest.approx([2.0] * 5) and out["b"] == pytest.approx([4.0] * 5)
    assert log.count("a") == 7 and log.count("b") == 7


@XFAIL
def test_order_is_shuffled_per_round_and_reproducible_for_a_seed():
    def order(seed):
        clock, log = FakeClock(), []
        _t().interleaved_samples(
            {"a": clock.engine(1e-3, log, "a"), "b": clock.engine(1e-3, log, "b")},
            warmup=0,
            runs=30,
            seed=seed,
            timer=clock,
        )
        return ["".join(log[i : i + 2]) for i in range(0, len(log), 2)]

    first, again = order(7), order(7)
    assert first == again
    assert {"ab", "ba"} <= set(first)


@XFAIL
def test_bootstrap_ci_contains_the_point_estimate_and_is_deterministic():
    t = _t()
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    point, lo, hi = t.bootstrap_ci(samples, seed=3)
    assert point == 5.0 and lo <= point <= hi
    assert t.bootstrap_ci(samples, seed=3) == (point, lo, hi)
    assert t.bootstrap_ci([4.0] * 20, seed=3) == (4.0, 4.0, 4.0)
    assert (hi - lo) > 0
    wider = t.bootstrap_ci([1.0, 9.0, 2.0, 20.0, 3.0, 40.0, 4.0, 60.0], seed=3)
    assert (wider[2] - wider[1]) > (hi - lo)


@XFAIL
def test_canary_slower_engine_is_detected():
    t = _t()
    clock = FakeClock()
    s = t.interleaved_samples(
        {"fast": clock.engine(0.002), "slow": clock.engine(0.004)},
        warmup=1,
        runs=20,
        seed=0,
        timer=clock,
    )
    ratio, lo, hi = t.ratio_ci(s["slow"], s["fast"], seed=0)
    assert ratio == pytest.approx(2.0) and lo > 1.5


@XFAIL
def test_aa_check_same_engine_gives_ratio_one():
    t = _t()
    clock = FakeClock()
    ratio, lo, hi = t.aa_check(clock.engine(0.003), warmup=1, runs=20, seed=0, timer=clock)
    assert ratio == pytest.approx(1.0) and lo <= 1.0 <= hi


@XFAIL
def test_jittered_aa_stays_close_to_one():
    import random

    t = _t()
    clock, rng = FakeClock(), random.Random(5)

    def jittery():
        clock.now += 0.003 * rng.uniform(0.9, 1.1)

    ratio, lo, hi = t.aa_check(jittery, warmup=2, runs=60, seed=0, timer=clock)
    assert 0.9 < ratio < 1.1 and (hi - lo) < 0.3
