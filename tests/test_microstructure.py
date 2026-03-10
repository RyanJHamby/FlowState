"""Tests for streaming microstructure feature computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flowstate.features.microstructure import (
    EWMA,
    AmihudIlliquidity,
    IncrementalTWAP,
    IncrementalVWAP,
    KyleLambda,
    MicrostructureEngine,
    MicrostructureFeatures,
    OrderFlowImbalance,
    SlidingWelford,
    TradeClassifier,
    WelfordVariance,
    YangZhangVolatility,
)

# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


class TestEWMA:
    def test_requires_alpha_or_span(self):
        with pytest.raises(ValueError, match="alpha or span"):
            EWMA()

    def test_first_value_equals_input(self):
        e = EWMA(alpha=0.5)
        assert e.update(10.0) == 10.0

    def test_alpha_from_span(self):
        e = EWMA(span=9)
        assert e._alpha == pytest.approx(0.2)

    def test_exponential_decay(self):
        e = EWMA(alpha=0.5)
        e.update(10.0)
        v = e.update(20.0)
        # 0.5 * 20 + 0.5 * 10 = 15
        assert v == pytest.approx(15.0)

    def test_converges_to_constant(self):
        e = EWMA(alpha=0.1)
        for _ in range(500):
            e.update(42.0)
        assert e.value == pytest.approx(42.0, abs=1e-10)

    def test_update_batch(self):
        e1 = EWMA(alpha=0.3)
        e2 = EWMA(alpha=0.3)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        batch_result = e1.update_batch(vals)
        for v in vals:
            e2.update(v)
        assert batch_result[-1] == pytest.approx(e2.value)
        assert len(batch_result) == 5

    def test_reset(self):
        e = EWMA(alpha=0.5)
        e.update(100.0)
        e.reset()
        assert e.value is None
        assert e.count == 0

    def test_count_tracks(self):
        e = EWMA(alpha=0.5)
        for i in range(10):
            e.update(float(i))
        assert e.count == 10


# ---------------------------------------------------------------------------
# Welford's Variance
# ---------------------------------------------------------------------------


class TestWelfordVariance:
    def test_single_value_zero_variance(self):
        w = WelfordVariance()
        w.update(5.0)
        assert w.variance == 0.0
        assert w.mean == 5.0

    def test_known_variance(self):
        w = WelfordVariance()
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for x in data:
            w.update(x)
        expected_var = np.var(data, ddof=1)
        assert w.variance == pytest.approx(expected_var, rel=1e-10)
        assert w.mean == pytest.approx(np.mean(data), rel=1e-10)

    def test_std(self):
        w = WelfordVariance()
        for x in [1.0, 2.0, 3.0]:
            w.update(x)
        assert w.std == pytest.approx(math.sqrt(w.variance))

    def test_batch_matches_sequential(self):
        w1 = WelfordVariance()
        w2 = WelfordVariance()
        data = np.random.default_rng(42).normal(0, 1, 100)
        w1.update_batch(data)
        for x in data:
            w2.update(x)
        assert w1.mean == pytest.approx(w2.mean, rel=1e-10)
        assert w1.variance == pytest.approx(w2.variance, rel=1e-10)

    def test_merge_parallel(self):
        """Chan's parallel merge produces same result as single pass."""
        rng = np.random.default_rng(99)
        data = rng.normal(10, 3, 200)
        w_full = WelfordVariance()
        w_full.update_batch(data)

        w_a = WelfordVariance()
        w_a.update_batch(data[:120])
        w_b = WelfordVariance()
        w_b.update_batch(data[120:])
        w_a.merge(w_b)

        assert w_a.mean == pytest.approx(w_full.mean, rel=1e-10)
        assert w_a.variance == pytest.approx(w_full.variance, rel=1e-8)

    def test_merge_with_empty(self):
        w = WelfordVariance()
        w.update_batch(np.array([1.0, 2.0, 3.0]))
        empty = WelfordVariance()
        w.merge(empty)
        assert w.count == 3

    def test_reset(self):
        w = WelfordVariance()
        w.update_batch(np.array([1.0, 2.0, 3.0]))
        w.reset()
        assert w.count == 0
        assert w.mean == 0.0


# ---------------------------------------------------------------------------
# SlidingWelford
# ---------------------------------------------------------------------------


class TestSlidingWelford:
    def test_filling_window(self):
        sw = SlidingWelford(5)
        for x in [1.0, 2.0, 3.0]:
            sw.update(x)
        assert sw.count == 3
        assert sw.mean == pytest.approx(2.0)

    def test_full_window_mean(self):
        sw = SlidingWelford(3)
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            sw.update(x)
        # Window contains [3, 4, 5]
        assert sw.mean == pytest.approx(4.0, rel=1e-10)

    def test_full_window_variance(self):
        sw = SlidingWelford(4)
        data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        for x in data:
            sw.update(x)
        # Window: [30, 40, 50, 60]
        expected = np.var([30.0, 40.0, 50.0, 60.0], ddof=1)
        assert sw.variance == pytest.approx(expected, rel=1e-6)

    def test_constant_input_zero_variance(self):
        sw = SlidingWelford(10)
        for _ in range(50):
            sw.update(7.0)
        assert sw.variance == pytest.approx(0.0, abs=1e-10)

    def test_reset(self):
        sw = SlidingWelford(5)
        for x in [1.0, 2.0, 3.0]:
            sw.update(x)
        sw.reset()
        assert sw.count == 0
        assert sw.mean == 0.0


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility
# ---------------------------------------------------------------------------


class TestYangZhangVolatility:
    def test_insufficient_data_returns_none(self):
        yz = YangZhangVolatility(window_size=5)
        yz.update(100, 105, 98, 102)
        yz.update(102, 106, 99, 104)
        assert yz.value is None  # Need >= 3

    def test_returns_positive_value(self):
        yz = YangZhangVolatility(window_size=5)
        # Simulate 5 bars of data
        bars = [
            (100, 105, 98, 102),
            (102, 107, 100, 105),
            (105, 110, 103, 108),
            (108, 112, 106, 110),
            (110, 115, 108, 113),
        ]
        for o, h, low, c in bars:
            yz.update(o, h, low, c)
        vol = yz.value
        assert vol is not None
        assert vol > 0

    def test_higher_range_higher_vol(self):
        """Wider high-low range should produce higher volatility."""
        yz_narrow = YangZhangVolatility(window_size=5)
        yz_wide = YangZhangVolatility(window_size=5)

        for i in range(6):
            base = 100 + i
            yz_narrow.update(base, base + 1, base - 1, base + 0.5)
            yz_wide.update(base, base + 10, base - 10, base + 0.5)

        assert yz_wide.value > yz_narrow.value

    def test_window_eviction(self):
        yz = YangZhangVolatility(window_size=3)
        for i in range(10):
            base = 100 + i
            yz.update(base, base + 2, base - 2, base + 1)
        # Should only retain last window_size + 1 bars
        assert yz.count == 4  # window_size + 1

    def test_reset(self):
        yz = YangZhangVolatility()
        yz.update(100, 105, 98, 102)
        yz.reset()
        assert yz.count == 0


# ---------------------------------------------------------------------------
# Order Flow Imbalance
# ---------------------------------------------------------------------------


class TestOrderFlowImbalance:
    def test_all_buys(self):
        ofi = OrderFlowImbalance(window_size=10)
        for _ in range(5):
            ofi.update(100.0, is_buy=True)
        assert ofi.windowed_ofi == pytest.approx(1.0)

    def test_all_sells(self):
        ofi = OrderFlowImbalance(window_size=10)
        for _ in range(5):
            ofi.update(100.0, is_buy=False)
        assert ofi.windowed_ofi == pytest.approx(-1.0)

    def test_balanced_flow(self):
        ofi = OrderFlowImbalance(window_size=10)
        for _ in range(5):
            ofi.update(100.0, is_buy=True)
            ofi.update(100.0, is_buy=False)
        assert ofi.windowed_ofi == pytest.approx(0.0)

    def test_window_eviction(self):
        ofi = OrderFlowImbalance(window_size=3)
        # Fill with buys
        for _ in range(3):
            ofi.update(100.0, is_buy=True)
        assert ofi.windowed_ofi == pytest.approx(1.0)
        # Now add 3 sells — buys should be evicted
        for _ in range(3):
            ofi.update(100.0, is_buy=False)
        assert ofi.windowed_ofi == pytest.approx(-1.0)

    def test_ewma_ofi_populated(self):
        ofi = OrderFlowImbalance(window_size=10, ewma_span=5)
        ofi.update(100.0, is_buy=True)
        assert ofi.ewma_ofi is not None

    def test_empty_returns_zero(self):
        ofi = OrderFlowImbalance()
        assert ofi.windowed_ofi == 0.0

    def test_reset(self):
        ofi = OrderFlowImbalance()
        ofi.update(100.0, is_buy=True)
        ofi.reset()
        assert ofi.windowed_ofi == 0.0
        assert ofi.ewma_ofi is None


# ---------------------------------------------------------------------------
# Kyle's Lambda
# ---------------------------------------------------------------------------


class TestKyleLambda:
    def test_initial_state(self):
        kl = KyleLambda()
        assert kl.lambda_value == 0.0
        assert kl.alpha == 0.0

    def test_positive_impact(self):
        """When buying pushes prices up, lambda should be positive."""
        kl = KyleLambda(forgetting_factor=0.99)
        rng = np.random.default_rng(42)
        true_lambda = 0.001
        for _ in range(200):
            signed_vol = rng.normal(0, 1000)
            dp = true_lambda * signed_vol + rng.normal(0, 0.01)
            kl.update(dp, signed_vol)
        assert kl.lambda_value > 0
        assert kl.lambda_value == pytest.approx(true_lambda, rel=0.3)

    def test_count(self):
        kl = KyleLambda()
        for _i in range(10):
            kl.update(0.01, 100.0)
        assert kl.count == 10

    def test_reset(self):
        kl = KyleLambda()
        kl.update(0.01, 100.0)
        kl.reset()
        assert kl.count == 0
        assert kl.lambda_value == 0.0


# ---------------------------------------------------------------------------
# Amihud Illiquidity
# ---------------------------------------------------------------------------


class TestAmihudIlliquidity:
    def test_single_observation(self):
        a = AmihudIlliquidity(window_size=10)
        a.update(abs_return=0.01, volume=1_000_000)
        expected = 0.01 / 1_000_000
        assert a.value == pytest.approx(expected)

    def test_zero_volume_ignored(self):
        a = AmihudIlliquidity()
        a.update(0.01, 0)
        assert a.count == 0
        assert a.value == 0.0

    def test_window_eviction(self):
        a = AmihudIlliquidity(window_size=2)
        a.update(0.01, 100)  # ratio = 0.0001
        a.update(0.02, 100)  # ratio = 0.0002
        a.update(0.03, 100)  # ratio = 0.0003, evicts first
        # Window: [0.0002, 0.0003], mean = 0.00025
        assert a.value == pytest.approx(0.00025)

    def test_higher_return_higher_illiquidity(self):
        a1 = AmihudIlliquidity()
        a2 = AmihudIlliquidity()
        for _ in range(10):
            a1.update(0.001, 1_000_000)
            a2.update(0.01, 1_000_000)
        assert a2.value > a1.value

    def test_reset(self):
        a = AmihudIlliquidity()
        a.update(0.01, 100)
        a.reset()
        assert a.count == 0
        assert a.value == 0.0


# ---------------------------------------------------------------------------
# Trade Classifier
# ---------------------------------------------------------------------------


class TestTradeClassifier:
    def test_above_mid_is_buy(self):
        tc = TradeClassifier()
        assert tc.classify(100.5, bid=100.0, ask=101.0) == 1  # Above mid of 100.5...
        # price == mid, falls to tick rule, first trade defaults to +1
        # Actually mid = 100.5, price = 100.5, so equals mid → tick rule → default +1
        # Let's use a clearer case:
        assert tc.classify(100.8, bid=100.0, ask=101.0) == 1

    def test_below_mid_is_sell(self):
        tc = TradeClassifier()
        assert tc.classify(100.2, bid=100.0, ask=101.0) == -1

    def test_at_mid_uses_tick_rule_uptick(self):
        tc = TradeClassifier()
        tc.classify(99.0, bid=98.0, ask=100.0)  # Set last_price = 99, below mid → sell
        # Now trade at mid with uptick
        result = tc.classify(100.0, bid=99.5, ask=100.5)  # mid = 100, price = 100
        # Tick rule: 100 > 99 → buy
        assert result == 1

    def test_at_mid_uses_tick_rule_downtick(self):
        tc = TradeClassifier()
        tc.classify(101.0, bid=100.0, ask=102.0)  # last_price = 101, above mid → buy
        result = tc.classify(100.5, bid=100.0, ask=101.0)  # mid = 100.5, price = 100.5
        # Tick rule: 100.5 < 101 → sell
        assert result == -1

    def test_at_mid_no_change_keeps_direction(self):
        tc = TradeClassifier()
        tc.classify(100.5, bid=100.0, ask=101.0)  # mid=100.5, price=100.5 → tick → default +1
        result = tc.classify(100.5, bid=100.0, ask=101.0)  # same price → keep direction
        assert result == 1  # kept from previous

    def test_classify_batch(self):
        tc = TradeClassifier()
        prices = np.array([100.8, 100.2, 100.8])
        bids = np.array([100.0, 100.0, 100.0])
        asks = np.array([101.0, 101.0, 101.0])
        result = tc.classify_batch(prices, bids, asks)
        assert list(result) == [1, -1, 1]

    def test_reset(self):
        tc = TradeClassifier()
        tc.classify(100.0, 99.0, 101.0)
        tc.reset()
        assert tc._last_price is None
        assert tc._last_direction == 1


# ---------------------------------------------------------------------------
# VWAP / TWAP
# ---------------------------------------------------------------------------


class TestIncrementalVWAP:
    def test_single_trade(self):
        v = IncrementalVWAP(window_size=10)
        result = v.update(100.0, 50.0)
        assert result == pytest.approx(100.0)

    def test_known_vwap(self):
        v = IncrementalVWAP(window_size=10)
        # Trades: (price=100, vol=10), (price=200, vol=30)
        v.update(100.0, 10.0)
        result = v.update(200.0, 30.0)
        # VWAP = (100*10 + 200*30) / (10+30) = 7000/40 = 175
        assert result == pytest.approx(175.0)

    def test_window_eviction(self):
        v = IncrementalVWAP(window_size=2)
        v.update(100.0, 10.0)
        v.update(200.0, 10.0)
        v.update(300.0, 10.0)  # Evicts (100, 10)
        # Window: (200, 10), (300, 10) → VWAP = 5000/20 = 250
        assert v.value == pytest.approx(250.0)

    def test_total_volume(self):
        v = IncrementalVWAP(window_size=100)
        v.update(100.0, 50.0)
        v.update(200.0, 30.0)
        assert v.total_volume == pytest.approx(80.0)

    def test_empty_returns_zero(self):
        v = IncrementalVWAP()
        assert v.value == 0.0

    def test_reset(self):
        v = IncrementalVWAP()
        v.update(100.0, 50.0)
        v.reset()
        assert v.value == 0.0


class TestIncrementalTWAP:
    def test_single_price(self):
        t = IncrementalTWAP(window_size=10)
        result = t.update(100.0)
        assert result == pytest.approx(100.0)

    def test_known_twap(self):
        t = IncrementalTWAP(window_size=10)
        t.update(100.0)
        t.update(200.0)
        assert t.value == pytest.approx(150.0)

    def test_window_eviction(self):
        t = IncrementalTWAP(window_size=2)
        t.update(100.0)
        t.update(200.0)
        t.update(300.0)
        # Window: [200, 300] → mean = 250
        assert t.value == pytest.approx(250.0)

    def test_reset(self):
        t = IncrementalTWAP()
        t.update(100.0)
        t.reset()
        assert t.count == 0


# ---------------------------------------------------------------------------
# MicrostructureEngine
# ---------------------------------------------------------------------------


class TestMicrostructureEngine:
    def test_auto_creates_symbol(self):
        engine = MicrostructureEngine()
        engine.update_trade("AAPL", 1_000_000_000, 150.0, 100.0, bid=149.9, ask=150.1)
        assert "AAPL" in engine.symbols

    def test_pre_registered_symbols(self):
        engine = MicrostructureEngine(symbols=["AAPL", "MSFT"])
        assert set(engine.symbols) == {"AAPL", "MSFT"}

    def test_snapshot_after_trades(self):
        engine = MicrostructureEngine()
        for i in range(10):
            price = 100.0 + i * 0.1
            engine.update_trade(
                "AAPL", i * 1_000_000_000, price, 100.0, bid=price - 0.05, ask=price + 0.05
            )
        snap = engine.snapshot("AAPL")
        assert isinstance(snap, MicrostructureFeatures)
        assert snap.symbol == "AAPL"
        assert snap.ewma_price > 0
        assert snap.vwap > 0
        assert snap.twap > 0
        assert snap.kyle_lambda != 0  # Should have some estimate after 10 trades

    def test_snapshot_has_timestamp(self):
        engine = MicrostructureEngine()
        engine.update_trade("AAPL", 42_000_000_000, 150.0, 100.0)
        snap = engine.snapshot("AAPL")
        assert snap.timestamp_ns == 42_000_000_000

    def test_multi_symbol_isolation(self):
        engine = MicrostructureEngine()
        engine.update_trade("AAPL", 1, 150.0, 100.0, bid=149.9, ask=150.1)
        engine.update_trade("MSFT", 1, 300.0, 50.0, bid=299.9, ask=300.1)

        aapl = engine.snapshot("AAPL")
        msft = engine.snapshot("MSFT")
        assert aapl.ewma_price == pytest.approx(150.0)
        assert msft.ewma_price == pytest.approx(300.0)

    def test_update_bar_feeds_yang_zhang(self):
        engine = MicrostructureEngine()
        bars = [
            (100, 105, 98, 102),
            (102, 107, 100, 105),
            (105, 110, 103, 108),
            (108, 112, 106, 110),
        ]
        for o, h, low, c in bars:
            engine.update_bar("AAPL", o, h, low, c)
        snap = engine.snapshot("AAPL")
        assert snap.yang_zhang_vol is not None
        assert snap.yang_zhang_vol > 0

    def test_ofi_populated(self):
        engine = MicrostructureEngine()
        for i in range(20):
            engine.update_trade("AAPL", i, 100.0 + i * 0.01, 100.0, bid=99.9, ask=100.1)
        snap = engine.snapshot("AAPL")
        # All trades are above midpoint → buy classified → OFI should be positive
        assert snap.ofi_windowed > 0

    def test_trade_without_quotes_still_works(self):
        """Trades without bid/ask skip classification but don't error."""
        engine = MicrostructureEngine()
        engine.update_trade("AAPL", 1, 150.0, 100.0)
        engine.update_trade("AAPL", 2, 151.0, 100.0)
        snap = engine.snapshot("AAPL")
        assert snap.ewma_price > 0
        assert snap.trade_direction == 1  # Default

    def test_reset_single_symbol(self):
        engine = MicrostructureEngine(symbols=["AAPL", "MSFT"])
        engine.update_trade("AAPL", 1, 150.0, 100.0)
        engine.update_trade("MSFT", 1, 300.0, 50.0)
        engine.reset("AAPL")
        aapl = engine.snapshot("AAPL")
        msft = engine.snapshot("MSFT")
        assert aapl.ewma_price == 0.0  # Reset
        assert msft.ewma_price == pytest.approx(300.0)  # Untouched

    def test_reset_all(self):
        engine = MicrostructureEngine(symbols=["AAPL", "MSFT"])
        engine.update_trade("AAPL", 1, 150.0, 100.0)
        engine.update_trade("MSFT", 1, 300.0, 50.0)
        engine.reset()
        assert engine.snapshot("AAPL").ewma_price == 0.0
        assert engine.snapshot("MSFT").ewma_price == 0.0

    def test_amihud_populated_after_multiple_trades(self):
        engine = MicrostructureEngine()
        for i in range(5):
            price = 100.0 * math.exp(0.001 * i)  # Small positive returns
            engine.update_trade("AAPL", i, price, 1000.0, bid=price - 0.05, ask=price + 0.05)
        snap = engine.snapshot("AAPL")
        assert snap.amihud_illiquidity > 0
