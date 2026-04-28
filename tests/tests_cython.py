"""
tests_cython.py — run the existing test suite against the Cython dispatcher
and add a few Cython-specific parity / routing checks.

How this works:
    - Imports `tests` (the pure-Python test module) as a library.
    - Rebinds `tests.bt` to point at `cython_backtester` (the dispatcher).
    - Runs every `test_*` function from `tests` through the dispatcher.
    - Adds Cython-specific tests at the end.

The 90 pure-Python tests cover every API surface, so the big value here
is CONFIRMING THE DISPATCHER IS A DROP-IN REPLACEMENT. If any of the
reused tests fail, the dispatcher has diverged from the pure-Python API.

Plus Cython-specific tests:
    - `cython_available()` / `cython_import_error()` diagnostics work.
    - Custom sizer routes to the Python fallback (can't cross nogil).
    - Fast path and Python path produce identical results for the same
      inputs (parity — the strongest guarantee we have that the Cython
      inner loop matches the reference).

Usage:
    python tests_cython.py

    # or under pytest
    pytest tests_cython.py -v

The script exits 0 on success, 1 on any failure.
"""

from __future__ import annotations

import inspect
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Import the existing test module and the Cython dispatcher
import tests as py_tests
import chronoton.cython_backtester as bt_cy
import chronoton.backtester as bt_py


# ---------------------------------------------------------------------------
# Rebind: every `bt.xxx` reference in tests.py now hits the Cython dispatcher.
# ---------------------------------------------------------------------------
py_tests.bt = bt_cy

# Also reset the test module's result counters so we can reuse its runner
py_tests._FAILURES = []
py_tests._PASSED = 0


# ---------------------------------------------------------------------------
# Cython-specific tests
# ---------------------------------------------------------------------------
def _small_inputs(n: int = 20):
    """Shared fixture: small deterministic OHLCV + a couple of signals."""
    np.random.seed(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.3)
    o = pd.Series(prices, index=idx)
    h = pd.Series(prices + 0.4, index=idx)
    l = pd.Series(prices - 0.4, index=idx)
    c = pd.Series(prices + 0.1, index=idx)
    v = pd.Series(np.full(n, 1000.0), index=idx)
    entries = np.zeros(n, dtype=bool); entries[2] = True; entries[12] = True
    exits   = np.zeros(n, dtype=bool); exits[7] = True; exits[17] = True
    return o, h, l, c, v, entries, exits


def test_cython_available_diagnostic():
    """`cython_available()` must return a bool; `cython_import_error()`
    must return None iff the extension loaded."""
    assert isinstance(bt_cy.cython_available(), bool)
    if bt_cy.cython_available():
        assert bt_cy.cython_import_error() is None
    else:
        assert bt_cy.cython_import_error() is not None


def test_custom_sizer_routes_to_python_fallback():
    """
    A custom sizer callable cannot cross `nogil`, so it MUST route to
    the pure-Python fallback regardless of whether the extension is
    available. The simplest check: the backtest runs and produces a
    correctly-sized trade.
    """
    o, h, l, c, v, entries, exits = _small_inputs()

    def sizer(bar_idx, entry_time_ns, direction, current_cash,
              open_positions, slot_active, closed_trades, n_closed,
              date, o_, h_, l_, c_, v_):
        return 25.0  # fixed

    r = bt_cy.run_single_backtest(
        o, h, l, c, v,
        starting_balance=10_000,
        long_entries=entries, long_exits=exits,
        position_sizing="custom", position_sizing_fn=sizer,
    )
    assert r.trades.shape[0] > 0
    assert abs(r.trades[0, bt_py.F_SIZE] - 25.0) < 1e-9


def test_parity_pure_python_vs_dispatcher_percent_equity():
    """
    Running the SAME backtest through `backtester.run_single_backtest`
    and `cython_backtester.run_single_backtest` must produce identical
    cash, equity, and trade log. This is the strongest parity test.
    If the Cython extension is loaded, it confirms the Cython loop
    matches the pure-Python reference; if not, it confirms the
    dispatcher is cleanly delegating to the Python fallback.
    """
    o, h, l, c, v, entries, exits = _small_inputs()

    kwargs = dict(
        starting_balance=10_000,
        long_entries=entries, long_exits=exits,
        position_sizing="percent_equity", position_percent_equity=0.5,
        commission=0.0005, spread=2.0, slippage=1.0, pip_equals=0.01,
        SL=30.0, TP=60.0, TS=40.0,
    )
    r_py = bt_py.run_single_backtest(o, h, l, c, v, **kwargs)
    r_cy = bt_cy.run_single_backtest(o, h, l, c, v, **kwargs)

    assert np.allclose(r_py.cash,   r_cy.cash,   atol=1e-9), \
        "cash series mismatch between pure-Python and Cython dispatcher"
    assert np.allclose(r_py.equity, r_cy.equity, atol=1e-9), \
        "equity series mismatch"
    assert r_py.trades.shape == r_cy.trades.shape, \
        f"trade-log shape mismatch: {r_py.trades.shape} vs {r_cy.trades.shape}"
    # nanclose because NaN fields (e.g. unused SL when none) must match NaN
    py_t = r_py.trades
    cy_t = r_cy.trades
    for i in range(py_t.shape[0]):
        for j in range(py_t.shape[1]):
            a, b = py_t[i, j], cy_t[i, j]
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-9, \
                f"trade[{i}] field {j} differs: py={a}, cy={b}"


def test_parity_percent_at_risk():
    """Parity on the new percent_at_risk sizing method."""
    o, h, l, c, v, entries, exits = _small_inputs()
    kwargs = dict(
        starting_balance=10_000,
        long_entries=entries, long_exits=exits,
        position_sizing="percent_at_risk", position_percent_at_risk=0.01,
        SL=30.0, pip_equals=0.01,
        leverage=5.0,
    )
    r_py = bt_py.run_single_backtest(o, h, l, c, v, **kwargs)
    r_cy = bt_cy.run_single_backtest(o, h, l, c, v, **kwargs)

    assert np.allclose(r_py.equity, r_cy.equity, atol=1e-9)
    assert r_py.trades.shape == r_cy.trades.shape


def test_parity_hedging_and_pyramiding():
    """Parity on a more complex run: hedging + multiple positions."""
    o, h, l, c, v, entries, exits = _small_inputs()
    shorts = np.zeros(len(entries), dtype=bool); shorts[5] = True
    short_exits = np.zeros(len(entries), dtype=bool); short_exits[10] = True

    kwargs = dict(
        starting_balance=20_000,
        long_entries=entries, long_exits=exits,
        short_entries=shorts, short_exits=short_exits,
        position_sizing="value", position_value=500.0,
        hedging=True, max_positions=2,
        commission=0.0005, spread=1.0, slippage=0.5, pip_equals=0.01,
    )
    r_py = bt_py.run_single_backtest(o, h, l, c, v, **kwargs)
    r_cy = bt_cy.run_single_backtest(o, h, l, c, v, **kwargs)

    assert np.allclose(r_py.equity, r_cy.equity, atol=1e-9)
    assert r_py.trades.shape == r_cy.trades.shape


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Reuse the section and run_test helpers from the Python test module.
    run_test = py_tests.run_test
    section = py_tests.section

    # Get all test_* functions from the Python test module (they now use bt_cy)
    py_all_tests = [
        fn for name, fn in inspect.getmembers(py_tests, inspect.isfunction)
        if name.startswith("test_")
    ]

    # Local tests
    local = sys.modules[__name__]
    cy_tests = [
        fn for name, fn in inspect.getmembers(local, inspect.isfunction)
        if name.startswith("test_")
    ]

    print(f"Running {len(py_all_tests)} pure-Python tests via Cython "
          f"dispatcher + {len(cy_tests)} Cython-specific tests\n")

    if bt_cy.cython_available():
        print("✓ Cython extension loaded — fast path in use.\n")
    else:
        print("⚠  Cython extension NOT loaded — dispatcher will fall back "
              "to pure-Python loop for every test.")
        print(f"    Import error: {bt_cy.cython_import_error()}\n")

    # Run the full Python test suite, rebound to the Cython dispatcher
    print("═" * 64)
    print("Pure-Python tests running through Cython dispatcher")
    print("═" * 64)
    for t in py_all_tests:
        run_test(t)

    # Then Cython-specific tests
    print()
    print("═" * 64)
    print("Cython-specific tests")
    print("═" * 64)
    for t in cy_tests:
        run_test(t)

    total = py_tests._PASSED + len(py_tests._FAILURES)
    print(f"\n{'─' * 64}")
    print(f"Passed: {py_tests._PASSED} / {total}")
    if py_tests._FAILURES:
        print(f"\nFailed tests:\n")
        for name, tb in py_tests._FAILURES:
            print(f"── {name} ──")
            print(tb)
        sys.exit(1)
    sys.exit(0)
