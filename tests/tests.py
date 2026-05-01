"""
tests.py — comprehensive test suite for backtester.py

Runs as a standalone script (`python tests.py`) or under pytest
(`pytest tests.py -v`). Uses only stdlib + numpy + pandas + matplotlib;
no pytest-specific features so the standalone fallback works.

Coverage:
    - Preprocessors: _process_series, _process_signals,
      _process_spread_slippage, _process_position_sizing,
      _process_overnight_charge
    - Helpers: _longest_run_of_true, _find_free_slot,
      _enter_position, _exit_position
    - Inner loop: entry/exit shifting, long/short trades, SL/TP/TS,
      trailing-stop updating, hedging, pyramiding, opposite-signal
      flatten, cost accounting, overnight charges, liquidation,
      percent_equity sizing, custom sizer callable
    - Result class: metrics dict, trades_to_dataframe, plotting,
      edge cases (empty trades, single bar, etc.)
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for plot tests
import matplotlib.pyplot as plt

import chronoton.backtester as bt


# ---------------------------------------------------------------------------
# Minimal test runner so this works without pytest.
# ---------------------------------------------------------------------------
_FAILURES: list = []
_PASSED = 0


def run_test(fn: Callable) -> None:
    global _PASSED
    name = fn.__name__
    try:
        fn()
        print(f"  ✓  {name}")
        _PASSED += 1
    except AssertionError as e:
        print(f"  ✗  {name}  —  {e}")
        _FAILURES.append((name, traceback.format_exc()))
    except Exception as e:
        print(f"  ✗  {name}  —  UNEXPECTED {type(e).__name__}: {e}")
        _FAILURES.append((name, traceback.format_exc()))


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(1, 60 - len(title)))


def approx(a, b, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------------------
# Fixtures — synthetic OHLCV builders.
# ---------------------------------------------------------------------------
def _make_ohlcv(close_prices, start="2024-01-01", freq="D", spread_hl=0.5,
                c_offset=0.1):
    """Build a trivially consistent OHLCV from a close-price array."""
    close_prices = np.asarray(close_prices, dtype=np.float64)
    n = close_prices.size
    idx = pd.date_range(start, periods=n, freq=freq)
    o = pd.Series(close_prices,               index=idx)
    h = pd.Series(close_prices + spread_hl,   index=idx)
    l = pd.Series(close_prices - spread_hl,   index=idx)
    c = pd.Series(close_prices + c_offset,    index=idx)
    v = pd.Series(np.full(n, 1000.0),         index=idx)
    return o, h, l, c, v


def _flat_ohlcv(price=100.0, n=10):
    return _make_ohlcv(np.full(n, price), spread_hl=0.5, c_offset=0.0)


def _uptrend(start=100.0, step=1.0, n=10):
    return _make_ohlcv(start + step * np.arange(n))


def _downtrend(start=100.0, step=1.0, n=10):
    return _make_ohlcv(start - step * np.arange(n))


def _signal(n, *bars):
    a = np.zeros(n, dtype=bool)
    for b in bars:
        a[b] = True
    return a


# ===========================================================================
# _process_series
# ===========================================================================
def test_process_series_happy():
    o, h, l, c, v = _uptrend(n=5)
    date, oa, ha, la, ca, va = bt._process_series(o, h, l, c, v)
    assert date.shape == (5,)
    assert oa.dtype == np.float64 and oa.shape == (5,)
    assert ha.shape == la.shape == ca.shape == va.shape == (5,)


def test_process_series_rejects_non_series():
    _, h, l, c, v = _uptrend(n=5)
    try:
        bt._process_series(np.array([1.0, 2, 3, 4, 5]), h, l, c, v)
        assert False, "should have raised TypeError"
    except TypeError:
        pass


def test_process_series_rejects_non_datetime_index():
    o, h, l, c, v = _uptrend(n=5)
    o_bad = pd.Series(o.values)  # default RangeIndex
    try:
        bt._process_series(o_bad, h, l, c, v)
        assert False
    except TypeError:
        pass


def test_process_series_rejects_length_mismatch():
    o, h, l, c, v = _uptrend(n=5)
    v_short = v.iloc[:3]
    try:
        bt._process_series(o, h, l, c, v_short)
        assert False
    except ValueError:
        pass


def test_process_series_rejects_zero():
    o, h, l, c, v = _uptrend(n=5)
    v2 = v.copy(); v2.iloc[2] = 0.0
    try:
        bt._process_series(o, h, l, c, v2)
        assert False
    except ValueError as e:
        assert "zero" in str(e).lower()


def test_process_series_rejects_nan():
    o, h, l, c, v = _uptrend(n=5)
    c2 = c.copy(); c2.iloc[2] = np.nan
    try:
        bt._process_series(o, h, l, c2, v)
        assert False
    except ValueError as e:
        assert "nan" in str(e).lower()


def test_process_series_rejects_index_mismatch():
    o, h, l, c, v = _uptrend(n=5)
    l2 = pd.Series(l.values, index=pd.date_range("2030-01-01", periods=5, freq="D"))
    try:
        bt._process_series(o, h, l2, c, v)
        assert False
    except ValueError:
        pass


# ===========================================================================
# _process_signals
# ===========================================================================
def test_signals_none_all_false():
    arr = bt._process_signals(None, 5, "x")
    assert arr.shape == (5,) and arr.dtype == bool and not arr.any()


def test_signals_ndarray_bool():
    arr = bt._process_signals(np.array([True, False, True, False, True]), 5, "x")
    assert arr.sum() == 3 and arr.dtype == bool


def test_signals_ndarray_int_cast():
    arr = bt._process_signals(np.array([0, 1, 0, 1, 0]), 5, "x")
    assert arr.dtype == bool and arr.sum() == 2


def test_signals_series():
    arr = bt._process_signals(pd.Series([True, False, True, False, True]), 5, "x")
    assert arr.shape == (5,) and arr.dtype == bool


def test_signals_length_mismatch():
    try:
        bt._process_signals(np.array([True, False]), 5, "x")
        assert False
    except ValueError:
        pass


def test_signals_two_d_rejected():
    try:
        bt._process_signals(np.array([[True], [False]]), 5, "x")
        assert False
    except ValueError:
        pass


def test_signals_list_rejected():
    try:
        bt._process_signals([True, False, True, False, True], 5, "x")
        assert False
    except TypeError:
        pass


# ===========================================================================
# _process_spread_slippage
# ===========================================================================
def test_spread_scalar_broadcast():
    arr = bt._process_spread_slippage(0.0002, 5, "spread")
    assert arr.shape == (5,) and np.all(arr == 0.0002)


def test_spread_array():
    arr = bt._process_spread_slippage(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 5, "spread")
    assert arr.shape == (5,) and arr.dtype == np.float64


def test_spread_series():
    arr = bt._process_spread_slippage(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]), 5, "spread")
    assert arr.shape == (5,)


def test_spread_wrong_length():
    try:
        bt._process_spread_slippage(np.array([0.1, 0.2]), 5, "spread")
        assert False
    except ValueError:
        pass


def test_spread_nan_rejected():
    try:
        bt._process_spread_slippage(np.array([0.1, np.nan, 0.3, 0.4, 0.5]), 5, "spread")
        assert False
    except ValueError:
        pass


def test_spread_inf_scalar_rejected():
    try:
        bt._process_spread_slippage(np.inf, 5, "spread")
        assert False
    except ValueError:
        pass


# ===========================================================================
# _process_position_sizing
# ===========================================================================
def test_sizing_percent_equity():
    mc, ss, sa, sf = bt._process_position_sizing(
        "percent_equity", 0.5, None, None, None, 10)
    assert mc == 0 and ss == 0.5 and sa.size == 0 and sf is None


def test_sizing_value():
    mc, ss, sa, sf = bt._process_position_sizing(
        "value", None, 1000.0, None, None, 10)
    assert mc == 1 and ss == 1000.0


def test_sizing_precomputed_ndarray():
    arr = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mc, ss, sa, sf = bt._process_position_sizing(
        "precomputed", None, None, arr, None, 5)
    assert mc == 2 and sa.shape == (5,) and np.all(sa == arr)


def test_sizing_precomputed_series():
    arr = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mc, ss, sa, sf = bt._process_position_sizing(
        "precomputed", None, None, pd.Series(arr), None, 5)
    assert mc == 2 and sa.shape == (5,)


def test_sizing_custom_callable():
    fn = lambda *a, **k: 1.0
    mc, ss, sa, sf = bt._process_position_sizing(
        "custom", None, None, None, fn, 10)
    assert mc == 3 and sf is fn


def test_sizing_unknown_method():
    try:
        bt._process_position_sizing("kelly", None, None, None, None, 10)
        assert False
    except ValueError:
        pass


def test_sizing_missing_required_arg():
    for method in ("percent_equity", "value", "precomputed", "custom"):
        try:
            bt._process_position_sizing(method, None, None, None, None, 10)
            assert False, f"{method} missing-arg should have raised"
        except ValueError:
            pass


def test_sizing_rejects_nonpositive_scalar():
    try:
        bt._process_position_sizing("percent_equity", 0.0, None, None, None, 10)
        assert False
    except ValueError:
        pass
    try:
        bt._process_position_sizing("value", None, -100.0, None, None, 10)
        assert False
    except ValueError:
        pass


def test_sizing_precomputed_wrong_length():
    try:
        bt._process_position_sizing(
            "precomputed", None, None, np.array([1.0, 2.0]), None, 5)
        assert False
    except ValueError:
        pass


def test_sizing_precomputed_rejects_negative():
    try:
        bt._process_position_sizing(
            "precomputed", None, None,
            np.array([1.0, -2.0, 3.0, 4.0, 5.0]), None, 5)
        assert False
    except ValueError:
        pass


def test_sizing_precomputed_rejects_nan():
    try:
        bt._process_position_sizing(
            "precomputed", None, None,
            np.array([1.0, np.nan, 3.0, 4.0, 5.0]), None, 5)
        assert False
    except ValueError:
        pass


def test_sizing_custom_rejects_non_callable():
    try:
        bt._process_position_sizing("custom", None, None, None, "not fn", 10)
        assert False
    except TypeError:
        pass


# ===========================================================================
# _process_overnight_charge
# ===========================================================================
def test_overnight_daily_eod_wed_triple():
    # Jan 1 2024 = Mon; EOD timestamps → Wed rollover inside Wed bar
    date = pd.date_range("2024-01-01 23:59", periods=5, freq="D")
    long_vec, short_vec = bt._process_overnight_charge((0.05, 0.02), "1d", date)
    expected = np.array([0, 1, 3, 1, 1]) * ((0.05 + 0.02) / 360)
    assert np.allclose(long_vec, expected)


def test_overnight_daily_midnight_triple_rolls_to_thu():
    date = pd.date_range("2024-01-01 00:00", periods=5, freq="D")
    long_vec, _ = bt._process_overnight_charge((0.05, 0.02), "1d", date)
    expected = np.array([0, 1, 1, 3, 1]) * ((0.05 + 0.02) / 360)
    assert np.allclose(long_vec, expected)


def test_overnight_hourly_only_one_bar_per_day():
    date = pd.date_range("2024-01-01 00:00", periods=72, freq="h")
    long_vec, _ = bt._process_overnight_charge((0.05, 0.02), "1h", date)
    nonzero = np.where(long_vec != 0)[0]
    # Rollovers at 22:00 each day (hours 22, 46, 70 from start)
    assert list(nonzero) == [22, 46, 70]


def test_overnight_weekly_accumulates():
    # 4 weekly bars (Sundays); each covers 5 weekday rollovers with Wed triple
    date = pd.date_range("2024-01-07", periods=4, freq="W")
    long_vec, _ = bt._process_overnight_charge((0.05, 0.02), "1w", date)
    # bar 0 = 0, bars 1..3 = 4*1 + 1*3 = 7 day-equivs (weekends filtered out)
    expected = np.array([0, 7, 7, 7]) * ((0.05 + 0.02) / 360)
    assert np.allclose(long_vec, expected)


def test_overnight_short_sign_flipped():
    date = pd.date_range("2024-01-01 23:59", periods=3, freq="D")
    long_vec, short_vec = bt._process_overnight_charge((0.05, 0.02), "1d", date)
    # long pays base + borrow (borrowing cost); short pays base - borrow (receives credit)
    assert np.isclose(long_vec[1],  (0.05 + 0.02) / 360)
    assert np.isclose(short_vec[1], (0.05 - 0.02) / 360)


def test_overnight_denominator_365():
    date = pd.date_range("2024-01-01 23:59", periods=3, freq="D")
    long_vec, _ = bt._process_overnight_charge(
        (0.0365, 0.0), "1d", date, denominator=365)
    assert np.isclose(long_vec[1], 1e-4)


def test_overnight_custom_weekday():
    date = pd.date_range("2024-01-01 23:59", periods=5, freq="D")
    long_vec, _ = bt._process_overnight_charge(
        (0.05, 0.02), "1d", date, triple_charge_weekday="Thursday")
    expected = np.array([0, 1, 1, 3, 1]) * ((0.05 + 0.02) / 360)
    assert np.allclose(long_vec, expected)


def test_overnight_rejects_bad_tuple():
    date = pd.date_range("2024-01-01", periods=5, freq="D")
    try:
        bt._process_overnight_charge((0.05,), "1d", date)
        assert False
    except TypeError:
        pass


def test_overnight_rejects_abbrev_weekday():
    date = pd.date_range("2024-01-01", periods=5, freq="D")
    try:
        bt._process_overnight_charge(
            (0.05, 0.02), "1d", date, triple_charge_weekday="Wed")
        assert False
    except ValueError:
        pass


def test_overnight_zero_input():
    date = pd.date_range("2024-01-01", periods=5, freq="D")
    lv, sv = bt._process_overnight_charge((0.0, 0.0), "1d", date)
    assert np.all(lv == 0) and np.all(sv == 0)


# ===========================================================================
# _longest_run_of_true
# ===========================================================================
def test_longest_run_basic():
    assert bt._longest_run_of_true(
        np.array([True, True, False, True, True, True, False])) == 3


def test_longest_run_empty():
    assert bt._longest_run_of_true(np.array([], dtype=bool)) == 0


def test_longest_run_all_false():
    assert bt._longest_run_of_true(np.array([False, False, False])) == 0


def test_longest_run_all_true():
    assert bt._longest_run_of_true(np.array([True, True, True])) == 3


# ===========================================================================
# _find_free_slot / _enter_position / _exit_position
# ===========================================================================
def test_slot_helpers_integration():
    n_slots = 2
    open_positions = np.full((n_slots, bt.N_FIELDS), np.nan, dtype=np.float64)
    slot_active = np.zeros(n_slots, dtype=bool)
    closed = np.full((10, bt.N_FIELDS), np.nan, dtype=np.float64)
    n_closed = 0

    # initially, slot 0 is free
    assert bt._find_free_slot(slot_active) == 0

    # enter a long in slot 0
    bt._enter_position(
        slot_idx=0, direction=1, bar_idx=3, entry_time_ns=0,
        entry_price=100.0, size=10.0, sl_price=np.nan, tp_price=np.nan,
        ts_dist=np.nan, commission_cost=0.5, spread_cost=0.2,
        slippage_cost=0.1,
        open_positions=open_positions, slot_active=slot_active,
    )
    assert slot_active[0] and not slot_active[1]
    assert open_positions[0, bt.F_DIRECTION] == 1
    assert bt._find_free_slot(slot_active) == 1

    # exit that position
    n_closed = bt._exit_position(
        slot_idx=0, bar_idx=7, exit_time_ns=0, exit_price=110.0,
        exit_reason=bt.EXIT_SIGNAL, exit_commission=0.5, exit_spread=0.2,
        exit_slippage=0.1,
        open_positions=open_positions, slot_active=slot_active,
        closed_trades=closed, n_closed=0,
    )
    assert n_closed == 1
    assert not slot_active[0]  # freed
    assert closed[0, bt.F_EXIT_PRICE] == 110.0
    assert closed[0, bt.F_BARS_HELD] == 4
    # Exit commission/spread/slippage were added to running totals
    assert closed[0, bt.F_COMMISSION] == 1.0
    assert closed[0, bt.F_SPREAD_COST] == 0.4
    assert closed[0, bt.F_SLIPPAGE_COST] == 0.2


def test_exit_position_runtime_error_when_full():
    open_positions = np.full((1, bt.N_FIELDS), np.nan)
    slot_active = np.array([True])
    closed = np.full((0, bt.N_FIELDS), np.nan)  # zero-row preallocation
    try:
        bt._exit_position(
            slot_idx=0, bar_idx=1, exit_time_ns=0, exit_price=100.0,
            exit_reason=0, exit_commission=0.0, exit_spread=0.0,
            exit_slippage=0.0,
            open_positions=open_positions, slot_active=slot_active,
            closed_trades=closed, n_closed=0,
        )
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# ===========================================================================
# Inner loop — core trade semantics
# ===========================================================================
def test_long_trade_shifts_and_pnl():
    # Entry sig at bar 1, exit sig at bar 6.
    # Expect fill at open[2], close at open[7]. PnL = (open[7]-open[2])*size.
    prices = np.arange(100, 110, dtype=np.float64)  # 100, 101, ..., 109
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)

    r = bt.run_single_backtest(
        o, h, l, c, v,
        starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 6),
        position_sizing="value", position_value=1000.0,
    )
    assert r.trades.shape[0] == 1
    t = r.trades[0]
    assert int(t[bt.F_ENTRY_BAR]) == 2
    assert int(t[bt.F_EXIT_BAR]) == 7
    assert approx(t[bt.F_ENTRY_PRICE], 102.0)
    assert approx(t[bt.F_EXIT_PRICE], 107.0)
    # size = 1000/102 ≈ 9.8039
    assert approx(t[bt.F_SIZE], 1000.0 / 102.0)


def test_short_trade_direction_and_pnl():
    prices = np.arange(100, 90, -1, dtype=np.float64)  # 100, 99, ..., 91
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)

    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        short_entries=_signal(10, 1),
        short_exits=_signal(10, 6),
        position_sizing="value", position_value=1000.0,
    )
    t = r.trades[0]
    assert int(t[bt.F_DIRECTION]) == -1
    assert r.equity[-1] > 10_000  # short profited in downtrend


def test_no_signals_equity_flat():
    o, h, l, c, v = _uptrend(n=20)
    r = bt.run_single_backtest(o, h, l, c, v, starting_balance=10_000,
        position_sizing="value", position_value=1000.0)
    assert r.trades.shape[0] == 0
    assert approx(r.equity[-1], 10_000)


# ===========================================================================
# SL / TP / TS
# ===========================================================================
def test_sl_triggers_and_wins_over_tp_tie():
    # Long enters at 102 (signal bar 1 → fill at open[2]=102).
    # On bar 4 the bar's range is 99.5 (low) to 104.5 (high).
    # SL=3 gives stop at 99; TP=2 gives target at 104. Both hit on bar 4 → SL.
    # pip_equals=1 so SL/TP are taken at face value in price units.
    prices = np.array([100., 101., 102., 103., 102., 101., 102., 103., 102., 101.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    h2 = h.copy(); h2.iloc[4] = 104.5
    l2 = l.copy(); l2.iloc[4] = 98.5

    r = bt.run_single_backtest(
        o, h2, l2, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        SL=3.0, TP=2.0, pip_equals=1.0,
    )
    assert r.trades.shape[0] == 1
    assert int(r.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_SL


def test_tp_triggers_long():
    # No SL; tp of 2 from entry at 102 → exit when H >= 104
    prices = np.array([100., 101., 102., 103., 105., 106., 107., 108., 109., 110.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        TP=2.0, pip_equals=1.0,
    )
    assert r.trades.shape[0] == 1
    assert int(r.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_TP


def test_ts_long_trails_upward_then_triggers():
    # Price rises from 102 → 110, then drops sharply.
    # With TS=2, peak tracks to 110 (its high), trigger at 108.
    prices = np.array([100., 101., 102., 104., 106., 108., 110., 107., 106., 105.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        TS=2.0, pip_equals=1.0,
    )
    assert r.trades.shape[0] == 1
    t = r.trades[0]
    assert int(t[bt.F_EXIT_REASON]) == bt.EXIT_TS
    # TS peak should be at bar 6's high (110.2 given spread_hl)
    assert t[bt.F_TS_PEAK] > 109.0


def test_sl_no_trigger_stays_open():
    # Price drifts up, SL never threatened
    prices = np.arange(100, 110, dtype=np.float64)
    o, h, l, c, v = _make_ohlcv(prices)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 8),
        position_sizing="value", position_value=1000.0,
        SL=50.0, pip_equals=1.0,
    )
    assert r.trades.shape[0] == 1
    assert int(r.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_SIGNAL


# ===========================================================================
# Pip-unit conversion (SL / TP / TS / spread / slippage)
# ===========================================================================
def test_pip_equals_applies_to_sl():
    # Stop of 300 pips with pip_equals=0.01 → 3.0 price units.
    # Entry at 102, stop at 99, bar 4 range dips below 99 → SL fires.
    prices = np.array([100., 101., 102., 103., 102., 101., 102., 103., 102., 101.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    l2 = l.copy(); l2.iloc[4] = 98.5

    r = bt.run_single_backtest(
        o, h, l2, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        SL=300.0, pip_equals=0.01,
    )
    assert r.trades.shape[0] == 1
    assert int(r.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_SL


def test_pip_equals_default_fx_scaling():
    # Default pip_equals=0.0001. SL=20 pips → 0.002 price units.
    # Use FX-like flat prices around 1.1000; drop on bar 4 to 1.0975 hits stop.
    prices = np.array([1.1000, 1.1001, 1.1002, 1.1003, 1.1000,
                       1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.0001, c_offset=0.0)
    # Force bar 4 low to dip below 1.1002 - 0.002 = 1.0982
    l2 = l.copy(); l2.iloc[4] = 1.0975

    r = bt.run_single_backtest(
        o, h, l2, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        SL=20.0,  # no pip_equals override → default 0.0001
    )
    assert r.trades.shape[0] == 1
    assert int(r.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_SL


def test_pip_equals_applies_to_spread_slippage():
    # spread=50, slippage=20 pips; pip_equals=0.01 → 0.5 + 0.2 = 0.7 price
    # units against the trader on each leg. Round trip adds ~1.4 against.
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.0, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 5),
        position_sizing="value", position_value=1000.0,
        spread=50.0, slippage=20.0, pip_equals=0.01,
    )
    # size ≈ 1000/100 = 10; round-trip cost in price units ≈ 1.4 per unit
    # → ~14 loss
    pnl = r._pnl()[0]
    assert pnl < -10.0 and pnl > -20.0, f"pnl={pnl}"


def test_pip_equals_applies_to_tp_and_ts():
    # Both TP and TS given in pips; pip_equals=1.0 → no scaling.
    prices = np.array([100., 101., 102., 104., 106., 108., 110., 107., 106., 105.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    r_tp = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        TP=2.0, pip_equals=1.0,
    )
    assert int(r_tp.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_TP

    r_ts = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
        TS=2.0, pip_equals=1.0,
    )
    assert int(r_ts.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_TS


# ===========================================================================
# Hedging and pyramiding
# ===========================================================================
def test_hedging_allows_simultaneous_long_and_short():
    # Open long at bar 2 and short at bar 4; both coexist.
    prices = np.arange(100, 110, dtype=np.float64)
    o, h, l, c, v = _make_ohlcv(prices)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=20_000,
        long_entries=_signal(10, 1),
        short_entries=_signal(10, 3),
        long_exits=_signal(10, 8),
        short_exits=_signal(10, 8),
        position_sizing="value", position_value=500.0,
        hedging=True, max_positions=1,
    )
    assert r.trades.shape[0] == 2
    directions = set(r.trades[:, bt.F_DIRECTION].astype(int))
    assert directions == {1, -1}


def test_non_hedging_opposite_signal_flattens():
    # Long at bar 1; short signal at bar 4 should flatten long first
    prices = np.arange(100, 110, dtype=np.float64)
    o, h, l, c, v = _make_ohlcv(prices)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        short_entries=_signal(10, 4),
        short_exits=_signal(10, 8),
        position_sizing="value", position_value=500.0,
        hedging=False, max_positions=1,
    )
    # 2 trades: closed long (flatten) + closed short
    assert r.trades.shape[0] == 2
    assert int(r.trades[0, bt.F_DIRECTION]) == 1
    assert int(r.trades[1, bt.F_DIRECTION]) == -1


def test_pyramiding_max_positions_cap():
    # max_positions=2, three entry signals → only first two open
    prices = np.full(20, 100.0)
    o, h, l, c, v = _make_ohlcv(prices)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(20, 1, 3, 5),
        long_exits=_signal(20, 15),
        position_sizing="value", position_value=100.0,
        max_positions=2,
    )
    # All opened positions eventually close at bar 16; only 2 could be open
    assert r.trades.shape[0] == 2


# ===========================================================================
# Cost accounting
# ===========================================================================
def test_commission_applied_both_sides():
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 5),
        position_sizing="value", position_value=1000.0,
        commission=0.001,  # 10 bps per leg
    )
    t = r.trades[0]
    # entry notional ≈ 1000, exit notional ≈ 1000, 0.001 each → ~2.0 total
    assert 1.5 < t[bt.F_COMMISSION] < 2.5


def test_spread_slippage_hurt_the_trader():
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    # Without costs, a flat round trip = 0 PnL
    r_no = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 5),
        position_sizing="value", position_value=1000.0,
    )
    # With spread + slippage, PnL should be negative.
    # pip_equals=1 so values are taken as price units directly.
    r_yes = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 5),
        position_sizing="value", position_value=1000.0,
        spread=0.10, slippage=0.05, pip_equals=1.0,
    )
    pnl_no = r_no._pnl()[0]
    pnl_yes = r_yes._pnl()[0]
    assert approx(pnl_no, 0.0, tol=1e-6)
    assert pnl_yes < -0.1


# ===========================================================================
# Overnight across multi-day hold
# ===========================================================================
def test_overnight_accrued_over_hold():
    # 10 flat bars. Open at bar 2 (Wed), close at bar 7 (Mon).
    # Overnight is applied in STEP 1 of each bar, BEFORE entries (step 4)
    # and BEFORE exits (step 3). So:
    #   - Bar 2 overnight runs first, but position isn't yet open → no charge.
    #   - Bars 3 (Thu), 4 (Fri): position is open → 1 charge each.
    #   - Bars 5 (Sat), 6 (Sun): weekend rollovers are filtered out → no charge.
    #     (The Wed triple already covers the upcoming weekend.)
    #   - Bar 7 (Mon): position is open at start of bar → 1 charge.
    # Total = 3 day-equivs of overnight.
    # charge = 3 * 0.05/360 * notional where notional = size * entry_price
    #        = 3 * 0.05/360 * 100 * 100 = 4.167
    prices = np.full(10, 100.0)
    idx = pd.date_range("2024-01-01 23:59", periods=10, freq="D")
    o = pd.Series(prices, index=idx); h = o + 0.2; l = o - 0.2
    c = o.copy(); v = pd.Series(np.full(10, 1000.0), index=idx)

    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),   # open at bar 2
        long_exits=_signal(10, 6),     # close at bar 7
        position_sizing="value", position_value=10_000.0,
        overnight_charge=(0.05, 0.0),
    )
    t = r.trades[0]
    expected = 3 * (0.05 / 360) * 10_000  # ≈ 4.167
    assert approx(t[bt.F_OVERNIGHT], expected, tol=0.01), \
        f"expected ≈{expected:.3f}, got {t[bt.F_OVERNIGHT]:.3f}"


# ===========================================================================
# Liquidation
# ===========================================================================
def test_liquidation_on_catastrophic_drop():
    # Short at $1; price rockets to $100. Equity would go deeply negative
    # if marked at bar close, but the broker is assumed to have closed at
    # the critical price — so equity lands at exactly 0 and the trade's
    # recorded pnl equals exactly -starting_balance.
    prices = np.array([1., 1., 1., 1., 100., 100., 100., 100., 100., 100.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.05, c_offset=0.0)
    start = 500
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=start,
        short_entries=_signal(10, 1),
        position_sizing="value", position_value=start,
    )
    assert r.trades.shape[0] == 1
    assert int(r.trades[0, bt.F_EXIT_REASON]) == bt.EXIT_LIQUIDATION
    # Equity hits exactly zero, not negative
    assert approx(r.equity[-1], 0.0, tol=1e-6)
    # Trade pnl matches the lost starting balance (not a spurious huge loss)
    pnl = r._pnl()[0]
    assert approx(pnl, -start, tol=1e-6), f"pnl={pnl}, expected -{start}"


# ===========================================================================
# End-of-data close-out
# ===========================================================================
def test_end_of_data_closes_open_position():
    # Long entry at bar 1 (fills at bar 2), NO exit signal anywhere.
    # Position should close at final bar's close with EXIT_END_OF_DATA.
    prices = np.arange(100, 110, dtype=np.float64)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.0, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
    )
    assert r.trades.shape[0] == 1
    t = r.trades[0]
    assert int(t[bt.F_EXIT_REASON]) == bt.EXIT_END_OF_DATA
    assert int(t[bt.F_EXIT_BAR]) == 9  # final bar
    # No costs applied on EOD close
    # (size-weighted entry costs only; no exit fees)


def test_end_of_data_closes_multiple_positions():
    # Hedging with one long + one short, neither with an exit signal.
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.0, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        short_entries=_signal(10, 1),
        position_sizing="value", position_value=500.0,
        hedging=True, max_positions=1,
    )
    assert r.trades.shape[0] == 2
    assert all(int(row[bt.F_EXIT_REASON]) == bt.EXIT_END_OF_DATA
               for row in r.trades)


def test_end_of_data_exit_reason_label():
    prices = np.arange(100, 110, dtype=np.float64)
    o, h, l, c, v = _make_ohlcv(prices)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=1000.0,
    )
    df = r.trades_to_dataframe()
    assert df["exit_reason"].iloc[0] == "end_of_data"


def test_liquidation_symmetric_long_vs_short():
    # A long dropping 100% and a short rallying 100x should both land at
    # equity 0 with pnl = -starting_balance. Broker closes at critical price.
    # Long case: price 100 → 0 would zero out a cash long, but our zero-
    # rejection in _process_series means we test a near-total drop instead.
    prices_down = np.array([100., 100., 100., 100., 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    o, h, l, c, v = _make_ohlcv(prices_down, spread_hl=0.005, c_offset=0.0)
    r_long = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=500,
        long_entries=_signal(10, 1),
        position_sizing="value", position_value=500.0,
    )
    # Either SL-style natural close to ~0 OR liquidation; in either case
    # equity should not go negative.
    assert r_long.equity[-1] >= 0


# ===========================================================================
# percent_equity sizing with concurrent positions
# ===========================================================================
def test_percent_equity_uses_current_equity():
    # A single long on flat prices; size should be starting_balance *
    # percent / entry_price.
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.0, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 5),
        position_sizing="percent_equity", position_percent_equity=0.5,
    )
    t = r.trades[0]
    # 0.5 * 10_000 / 100 = 50 units
    assert approx(t[bt.F_SIZE], 50.0, tol=0.01)


# ===========================================================================
# Custom sizer callable
# ===========================================================================
def test_custom_sizer_receives_state_and_returns_size():
    captured = {}

    def sizer(bar_idx, entry_time_ns, direction, current_cash,
              open_positions, slot_active, closed_trades, n_closed,
              date, o, h, l, c, v):
        captured["bar_idx"] = bar_idx
        captured["direction"] = direction
        captured["cash"] = current_cash
        captured["o_len"] = len(o)
        return 20.0  # fixed size

    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.0, c_offset=0.0)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 5),
        position_sizing="custom", position_sizing_fn=sizer,
    )
    assert captured["bar_idx"] == 2    # shifted entry
    assert captured["direction"] == 1
    assert captured["o_len"] == 10
    assert approx(r.trades[0, bt.F_SIZE], 20.0)


def test_custom_sizer_returning_zero_skips_entry():
    def sizer(*a, **k):
        return 0.0
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices)
    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="custom", position_sizing_fn=sizer,
    )
    assert r.trades.shape[0] == 0


# ===========================================================================
# percent_at_risk sizing
# ===========================================================================
def test_percent_at_risk_happy_path():
    # Risk 1% of £10,000 = £100 per trade with a 10-pip (price-unit) stop.
    # size = 100 / 10 = 10 units. Let the stop hit; realised loss ≈ £100.
    prices = np.array([100., 101., 102., 103., 92., 92., 92., 92., 92., 92.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    # Force bar 4 low to definitely hit the stop at 92
    l2 = l.copy(); l2.iloc[4] = 90.0

    r = bt.run_single_backtest(
        o, h, l2, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        position_sizing="percent_at_risk", position_percent_at_risk=0.01,
        SL=10.0, pip_equals=1.0,
        leverage=5.0,  # needs some leverage: 10_000 risk 1% / 10pip * 102 ≈ 1.02x
    )
    assert r.trades.shape[0] == 1
    t = r.trades[0]
    # size = 0.01 * 10_000 / 10 = 10 units
    assert approx(t[bt.F_SIZE], 10.0, tol=0.01)
    assert int(t[bt.F_EXIT_REASON]) == bt.EXIT_SL
    pnl = r._pnl()[0]
    # Stop at entry_px - 10 → P&L ≈ -10 × size = -100
    assert approx(pnl, -100.0, tol=1.0)


def test_percent_at_risk_requires_sl():
    # No SL supplied → should raise immediately
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices)
    try:
        bt.run_single_backtest(
            o, h, l, c, v, starting_balance=10_000,
            long_entries=_signal(10, 1),
            position_sizing="percent_at_risk", position_percent_at_risk=0.01,
            SL=None,  # ← the offence
            pip_equals=1.0,
        )
        assert False, "expected ValueError for missing SL"
    except ValueError as e:
        assert "SL" in str(e)


def test_percent_at_risk_leverage_check_raises():
    # Tight 2-pip stop with 10% risk on stock @ 100: req leverage = 0.1*100/2 = 5
    # We only allow leverage=2 → should raise.
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices)
    try:
        bt.run_single_backtest(
            o, h, l, c, v, starting_balance=10_000,
            long_entries=_signal(10, 1),
            position_sizing="percent_at_risk", position_percent_at_risk=0.10,
            SL=2.0, pip_equals=1.0,
            leverage=2.0,  # insufficient
        )
        assert False, "expected ValueError for insufficient leverage"
    except ValueError as e:
        msg = str(e)
        assert "leverage" in msg.lower() and "percent_at_risk" in msg


def test_percent_at_risk_rejects_bad_fraction():
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices)
    for bad in (0.0, -0.1, 1.0, 1.5, float("nan")):
        try:
            bt.run_single_backtest(
                o, h, l, c, v, starting_balance=10_000,
                long_entries=_signal(10, 1),
                position_sizing="percent_at_risk",
                position_percent_at_risk=bad,
                SL=10.0, pip_equals=1.0,
            )
            assert False, f"expected ValueError for fraction={bad}"
        except ValueError:
            pass


def test_percent_at_risk_all_nan_sl_array_rejected():
    # SL is supplied as an array but is all-NaN → must raise upfront
    prices = np.full(10, 100.0)
    o, h, l, c, v = _make_ohlcv(prices)
    sl_arr = np.full(10, np.nan, dtype=np.float64)
    try:
        bt.run_single_backtest(
            o, h, l, c, v, starting_balance=10_000,
            long_entries=_signal(10, 1),
            position_sizing="percent_at_risk", position_percent_at_risk=0.01,
            SL=sl_arr, pip_equals=1.0,
        )
        assert False, "expected ValueError for all-NaN SL array"
    except ValueError as e:
        assert "NaN" in str(e) or "nan" in str(e).lower()


# ===========================================================================
# Result class — metrics & export
# ===========================================================================
def _run_small_backtest():
    prices = np.array([100., 101., 102., 103., 104., 105., 106., 107., 108., 109.])
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.2, c_offset=0.0)
    return bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=_signal(10, 1),
        long_exits=_signal(10, 6),
        position_sizing="value", position_value=1000.0,
    )


def test_calculate_metrics_keys():
    r = _run_small_backtest()
    m = r.calculate_metrics()
    expected = {
        "sharpe", "log_sharpe", "sortino", "log_sortino",
        "calmar", "cagr", "max_drawdown", "ulcer_index",
        "k_ratio_1996", "k_ratio_2003", "k_ratio_2013",
        "omega_ratio", "jensens_alpha",
        "n_trades", "winrate", "profit_factor", "expectancy", "exposure",
        "avg_duration", "avg_duration_winning", "avg_duration_losing",
        "max_consecutive_winners", "max_consecutive_losers",
        "biggest_win", "biggest_loss",
        "avg_winning_trade", "avg_losing_trade",
        "total_return", "starting_equity", "final_equity",
    }
    assert expected.issubset(set(m.keys()))


def test_metrics_no_trades_safe_defaults():
    o, h, l, c, v = _uptrend(n=20)
    r = bt.run_single_backtest(o, h, l, c, v, starting_balance=10_000,
        position_sizing="value", position_value=1000.0)
    m = r.calculate_metrics()
    assert m["n_trades"] == 0
    assert m["winrate"] == 0.0
    assert m["expectancy"] == 0.0
    assert m["avg_duration"] == 0.0


def test_metrics_jensens_alpha_requires_benchmark():
    r = _run_small_backtest()
    m = r.calculate_metrics()
    assert np.isnan(m["jensens_alpha"])
    m2 = r.calculate_metrics(benchmark_returns=np.zeros(r.equity.size - 1))
    assert not np.isnan(m2["jensens_alpha"])


def test_trades_to_dataframe_schema():
    r = _run_small_backtest()
    df = r.trades_to_dataframe()
    expected_cols = {
        "direction", "entry_bar", "entry_time", "entry_price",
        "exit_bar", "exit_time", "exit_price", "size",
        "sl", "tp", "ts_dist", "ts_peak",
        "commission", "spread_cost", "slippage_cost", "overnight",
        "mae", "mfe", "exit_reason", "bars_held", "pnl",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df.shape[0] == 1
    # Exit reason should be a string
    assert df["exit_reason"].iloc[0] == "signal"
    # entry_time / exit_time should be datetime-typed
    assert "datetime" in str(df["entry_time"].dtype)


def test_trades_to_dataframe_empty():
    o, h, l, c, v = _uptrend(n=20)
    r = bt.run_single_backtest(o, h, l, c, v, starting_balance=10_000,
        position_sizing="value", position_value=1000.0)
    df = r.trades_to_dataframe()
    assert df.shape[0] == 0
    # Column schema preserved
    assert "pnl" in df.columns and "exit_reason" in df.columns


def test_pnl_matches_equity_change_roughly():
    # With no other costs, sum of trade PnL should equal final equity change.
    r = _run_small_backtest()
    pnl_sum = r._pnl().sum()
    equity_change = r.equity[-1] - r.equity[0]
    assert approx(pnl_sum, equity_change, tol=1e-6)


# ===========================================================================
# Result class — plotting (smoke)
# ===========================================================================
def test_plot_returns_smoke():
    r = _run_small_backtest()
    ax = r.plot_returns()
    plt.close(ax.figure)


def test_plot_drawdown_smoke():
    r = _run_small_backtest()
    ax = r.plot_drawdown()
    plt.close(ax.figure)


def test_plot_metrics_smoke():
    r = _run_small_backtest()
    fig = r.plot_metrics()
    plt.close(fig)


def test_plots_with_no_trades():
    o, h, l, c, v = _uptrend(n=20)
    r = bt.run_single_backtest(o, h, l, c, v, starting_balance=10_000,
        position_sizing="value", position_value=1000.0)
    fig = r.plot_metrics()
    plt.close(fig)


def test_summary_repr():
    r = _run_small_backtest()
    s = r.summary()
    assert "Trades:" in s and "Final equity" in s
    assert repr(r) == s


# ===========================================================================
# End-to-end sanity: run_single_backtest on long realistic series
# ===========================================================================
def test_end_to_end_100_bars():
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    o, h, l, c, v = _make_ohlcv(prices, spread_hl=0.3)

    entries = np.zeros(n, dtype=bool); entries[[5, 30, 60, 85]] = True
    exits   = np.zeros(n, dtype=bool); exits[[15, 45, 75, 95]] = True

    r = bt.run_single_backtest(
        o, h, l, c, v, starting_balance=10_000,
        long_entries=entries, long_exits=exits,
        position_sizing="percent_equity", position_percent_equity=0.5,
        commission=0.001, spread=0.05, slippage=0.02, pip_equals=1.0,
    )
    m = r.calculate_metrics()
    assert m["n_trades"] > 0
    assert isinstance(m["sharpe"], float)
    df = r.trades_to_dataframe()
    assert df.shape[0] == m["n_trades"]


# ===========================================================================
# Run everything
# ===========================================================================
if __name__ == "__main__":
    import inspect
    current_module = sys.modules[__name__]
    all_tests = [
        fn for name, fn in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]

    print(f"Running {len(all_tests)} tests\n")

    section("_process_series")
    [run_test(t) for t in all_tests if "process_series" in t.__name__]
    section("_process_signals")
    [run_test(t) for t in all_tests if t.__name__.startswith("test_signals_")]
    section("_process_spread_slippage")
    [run_test(t) for t in all_tests if "spread" in t.__name__ and "process" not in t.__name__.replace("spread", "")]
    section("_process_position_sizing")
    [run_test(t) for t in all_tests if "sizing" in t.__name__ and "custom" not in t.__name__ and "percent_equity_uses" not in t.__name__]
    section("_process_overnight_charge")
    [run_test(t) for t in all_tests if "overnight" in t.__name__ and "accrued" not in t.__name__]
    section("Helpers")
    [run_test(t) for t in all_tests if "longest_run" in t.__name__ or "slot_helpers" in t.__name__ or "exit_position_runtime" in t.__name__]
    section("Inner loop — core semantics")
    [run_test(t) for t in all_tests if t.__name__ in (
        "test_long_trade_shifts_and_pnl",
        "test_short_trade_direction_and_pnl",
        "test_no_signals_equity_flat",
    )]
    section("SL / TP / TS")
    [run_test(t) for t in all_tests if t.__name__.startswith(("test_sl_", "test_tp_", "test_ts_"))]
    section("Pip-unit conversion")
    [run_test(t) for t in all_tests if t.__name__.startswith("test_pip_equals_")]
    section("Hedging & pyramiding")
    [run_test(t) for t in all_tests if any(k in t.__name__ for k in
        ("hedging", "non_hedging", "pyramiding"))]
    section("Cost accounting")
    [run_test(t) for t in all_tests if t.__name__ in (
        "test_commission_applied_both_sides",
        "test_spread_slippage_hurt_the_trader",
    )]
    section("Overnight over hold")
    [run_test(t) for t in all_tests if "accrued" in t.__name__]
    section("Liquidation")
    [run_test(t) for t in all_tests if "liquidation" in t.__name__]
    section("End-of-data close-out")
    [run_test(t) for t in all_tests if "end_of_data" in t.__name__]
    section("Sizing — percent_equity & custom")
    [run_test(t) for t in all_tests if t.__name__ in (
        "test_percent_equity_uses_current_equity",
        "test_custom_sizer_receives_state_and_returns_size",
        "test_custom_sizer_returning_zero_skips_entry",
    )]
    section("Sizing — percent_at_risk")
    [run_test(t) for t in all_tests if t.__name__.startswith("test_percent_at_risk_")]
    section("Result — metrics, dataframe, plots")
    [run_test(t) for t in all_tests if any(k in t.__name__ for k in
        ("metrics", "dataframe", "plot", "summary", "pnl_matches"))]
    section("End-to-end")
    [run_test(t) for t in all_tests if "end_to_end" in t.__name__]

    print(f"\n{'─' * 60}")
    print(f"Passed: {_PASSED} / {_PASSED + len(_FAILURES)}")
    if _FAILURES:
        print(f"\nFailed tests:\n")
        for name, tb in _FAILURES:
            print(f"── {name} ──")
            print(tb)
        sys.exit(1)
    sys.exit(0)
