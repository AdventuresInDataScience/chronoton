"""
cython_backtester1.py — public dispatcher for the Cython-accelerated backtester.

Same public API as ``backtester.run_single_backtest``, but routes through the
compiled fast-path (``_cy_inner.so``/``pyd``) when possible and falls back to
the pure-Python ``_inner_loop`` in ``backtester.py`` for the custom-sizer
path. All validation, preprocessing, and Result-wrapping logic is reused
from ``backtester.py`` — this file is a thin shim.

Public API:
    run_single_backtest(...)   # drop-in replacement for backtester.run_single_backtest
    Result                      # re-exported for convenience

Private:
    _inner_loop_cy(...)         # wrapper that adapts dtypes and calls _cy_inner
    _should_use_fast_path(...)  # tiny heuristic

The import graph:

    cython_backtester1.py   ← public API
        │
        ├── backtester (all preprocessors, Result, constants, Python loop)
        └── _cy_inner           ← compiled .so; fast path only
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

# Reuse everything from the pure-Python module: preprocessors, constants,
# Result, and the pure-Python inner loop (used as the fallback for custom
# sizers, which can't cross the `nogil` boundary).
from .backtester import (
    # Public
    Result,
    # Preprocessors
    _process_series,
    _process_signals,
    _process_spread_slippage,
    _process_position_sizing,
    _process_overnight_charge,
    # Pure-Python fallback loop
    _inner_loop as _inner_loop_py,
    # Slot helpers (re-exported for testability; the Cython fast path
    # has its own compiled equivalents used internally)
    _find_free_slot,
    _enter_position,
    _exit_position,
    # Misc helpers used by tests
    _longest_run_of_true,
    # Field layout & exit-reason constants (re-exported for test suites)
    F_DIRECTION, F_ENTRY_BAR, F_ENTRY_TIME, F_ENTRY_PRICE,
    F_EXIT_BAR, F_EXIT_TIME, F_EXIT_PRICE, F_SIZE,
    F_SL, F_TP, F_TS_DIST, F_TS_PEAK,
    F_COMMISSION, F_SPREAD_COST, F_SLIPPAGE_COST, F_OVERNIGHT,
    F_MAE, F_MFE, F_EXIT_REASON, F_BARS_HELD, N_FIELDS,
    EXIT_SIGNAL, EXIT_SL, EXIT_TP, EXIT_TS,
    EXIT_LIQUIDATION, EXIT_END_OF_DATA,
    _EXIT_REASON_NAMES,
)

try:
    from ._cy_inner import inner_loop_fast as _inner_loop_fast_raw
    _CYTHON_AVAILABLE = True
except ImportError as exc:
    _CYTHON_AVAILABLE = False
    _CYTHON_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Fast-path wrapper
# ---------------------------------------------------------------------------
def _inner_loop_cy(
    date: np.ndarray,
    o: np.ndarray, h: np.ndarray, l: np.ndarray,
    c: np.ndarray, v: np.ndarray,
    long_entries_shifted: np.ndarray,
    long_exits_shifted: np.ndarray,
    short_entries_shifted: np.ndarray,
    short_exits_shifted: np.ndarray,
    starting_balance: float,
    sizing_method_code: int,
    sizing_static: float,
    sizing_array: np.ndarray,
    sl_arr: np.ndarray,
    tp: float,
    ts: float,
    leverage: float,
    commission: float,
    spread_arr: np.ndarray,
    slippage_arr: np.ndarray,
    long_fee_vec: np.ndarray,
    short_fee_vec: np.ndarray,
    max_positions: int,
    hedging: bool,
) -> tuple:
    """
    Wrap the compiled ``inner_loop_fast``: allocate output buffers, convert
    bool signal arrays to uint8 (required by the ``unsigned char[:]``
    memoryview contract on the Cython side), call the compiled loop,
    trim the trade-log to the returned row count, and return
    ``(cash, equity, closed_trades)``.

    Raises
    ------
    RuntimeError
        If the compiled loop reports the closed_trades pre-allocation was
        exhausted (pathological edge case: very short trades with high
        max_positions).
    """
    n = o.size
    n_slots = max_positions * 2 if hedging else max_positions

    # Pre-allocated output buffers
    cash_out = np.empty(n, dtype=np.float64)
    equity_out = np.empty(n, dtype=np.float64)
    open_positions = np.full((n_slots, N_FIELDS), np.nan, dtype=np.float64)
    slot_active = np.zeros(n_slots, dtype=np.uint8)
    # Matches pure-Python convention: n rows is a safe upper bound
    closed_trades = np.full((n, N_FIELDS), np.nan, dtype=np.float64)

    # Signal arrays: bool → uint8 (memoryview type contract)
    long_entries_u8 = long_entries_shifted.astype(np.uint8, copy=False)
    long_exits_u8   = long_exits_shifted.astype(np.uint8, copy=False)
    short_entries_u8 = short_entries_shifted.astype(np.uint8, copy=False)
    short_exits_u8   = short_exits_shifted.astype(np.uint8, copy=False)

    # date is datetime64[ns]; view as int64 then cast to float64 so it
    # fits into a double-typed memoryview. Precision loss is negligible
    # for the nanosecond range covered by practical backtest horizons
    # (~±292 years from 1970 before float64 starts losing ns precision).
    date_ns = date.astype("datetime64[ns]").astype(np.int64).astype(np.float64)

    # Empty sentinel array when sizing_array isn't used, so the memoryview
    # binding still succeeds with a zero-length view.
    if sizing_array is None or sizing_array.size == 0:
        sizing_array = np.empty(0, dtype=np.float64)
    else:
        sizing_array = np.ascontiguousarray(sizing_array, dtype=np.float64)

    n_closed = _inner_loop_fast_raw(
        np.ascontiguousarray(o, dtype=np.float64),
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(l, dtype=np.float64),
        np.ascontiguousarray(c, dtype=np.float64),
        np.ascontiguousarray(v, dtype=np.float64),
        np.ascontiguousarray(date_ns, dtype=np.float64),
        np.ascontiguousarray(long_entries_u8, dtype=np.uint8),
        np.ascontiguousarray(long_exits_u8, dtype=np.uint8),
        np.ascontiguousarray(short_entries_u8, dtype=np.uint8),
        np.ascontiguousarray(short_exits_u8, dtype=np.uint8),
        float(starting_balance),
        int(sizing_method_code),
        float(sizing_static),
        sizing_array,
        np.ascontiguousarray(sl_arr, dtype=np.float64),
        float(tp),
        float(ts),
        float(leverage),
        float(commission),
        np.ascontiguousarray(spread_arr, dtype=np.float64),
        np.ascontiguousarray(slippage_arr, dtype=np.float64),
        np.ascontiguousarray(long_fee_vec, dtype=np.float64),
        np.ascontiguousarray(short_fee_vec, dtype=np.float64),
        bool(hedging),
        cash_out,
        equity_out,
        open_positions,
        slot_active,
        closed_trades,
    )

    if n_closed < 0:
        raise RuntimeError(
            "closed_trades pre-allocation exhausted in Cython inner loop. "
            "Very short trades with high max_positions can exceed the default "
            "n_bars cap."
        )

    return cash_out, equity_out, closed_trades[:n_closed]


# ---------------------------------------------------------------------------
# Dispatch heuristic
# ---------------------------------------------------------------------------
def _should_use_fast_path(sizing_method_code: int) -> bool:
    """
    Fast path applies when:
        - The compiled extension is available.
        - Sizing is not 'custom' (method_code != 3); a Python callable can't
          cross the nogil boundary.
    """
    return _CYTHON_AVAILABLE and sizing_method_code != 3


# ---------------------------------------------------------------------------
# Public API — same signature as backtester.run_single_backtest.
# This is deliberately near-identical to the pure-Python version so it's a
# drop-in replacement. The only new behaviour is the routing decision.
# ---------------------------------------------------------------------------
def run_single_backtest(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    v: pd.Series,
    pip_equals: float = 0.0001,
    starting_balance: float = 10_000.0,
    long_entries: Optional[np.ndarray] = None,
    long_exits: Optional[np.ndarray] = None,
    short_entries: Optional[np.ndarray] = None,
    short_exits: Optional[np.ndarray] = None,
    position_sizing: str = "percent_equity",
    position_percent_equity: Optional[float] = 1.0,
    position_value: Optional[float] = None,
    position_sizes: Optional[np.ndarray] = None,
    position_sizing_fn: Optional[Callable] = None,
    position_percent_at_risk: Optional[float] = None,
    SL: Optional[Union[float, np.ndarray]] = None,
    TP: Optional[float] = None,
    TS: Optional[float] = None,
    leverage: float = 1.0,
    commission: float = 0.0,
    spread: Union[float, np.ndarray, pd.Series] = 0.0,
    slippage: Union[float, np.ndarray, pd.Series] = 0.0,
    overnight_charge: tuple = (0.0, 0.0),
    max_positions: int = 1,
    hedging: bool = False,
    timeframe: str = "1d",
    *args,
) -> Result:
    """
    Run a single backtest. Drop-in replacement for
    ``backtester.run_single_backtest``. Routes through the compiled Cython
    fast path when possible; falls back to the pure-Python loop for the
    custom-sizer case or when the extension is not built.

    See ``backtester.run_single_backtest`` for full parameter semantics.
    """
    # --- strip & validate OHLCV ---------------------------------------
    date, o_arr, h_arr, l_arr, c_arr, v_arr = _process_series(o, h, l, c, v)
    n = o_arr.size

    # --- signals (shift by +1) ---------------------------------------
    def _shift(arr):
        out = np.zeros(n, dtype=bool)
        if arr.size > 0:
            out[1:] = arr[:-1]
        return out

    long_entries_v  = _shift(_process_signals(long_entries,  n, "long_entries"))
    long_exits_v    = _shift(_process_signals(long_exits,    n, "long_exits"))
    short_entries_v = _shift(_process_signals(short_entries, n, "short_entries"))
    short_exits_v   = _shift(_process_signals(short_exits,   n, "short_exits"))

    # --- cost-distance inputs are taken from the user in PIPS -----------
    # SL, TP, TS, spread, and slippage are all supplied in pip units and
    # converted to price-unit distances here. See backtester.py for detail.

    # --- spread / slippage (pips → price units) ----------------------
    spread_arr   = _process_spread_slippage(spread,   n, "spread")   * pip_equals
    slippage_arr = _process_spread_slippage(slippage, n, "slippage") * pip_equals

    # --- SL / TP / TS (pips → price units) ---------------------------
    if SL is None:
        sl_arr = np.full(n, np.nan, dtype=np.float64)
    elif isinstance(SL, (int, float, np.integer, np.floating)):
        sl_arr = np.full(n, float(SL) * pip_equals, dtype=np.float64)
    else:
        sl_arr = _process_spread_slippage(SL, n, "SL") * pip_equals

    tp_val = float("nan") if TP is None else float(TP) * pip_equals
    ts_val = float("nan") if TS is None else float(TS) * pip_equals

    # --- position sizing ---------------------------------------------
    method_code, static_size, sizes_array, sizing_fn = _process_position_sizing(
        position_sizing, position_percent_equity, position_value,
        position_sizes, position_sizing_fn, n,
        position_percent_at_risk=position_percent_at_risk,
    )

    # --- overnight financing -----------------------------------------
    long_fee_vec, short_fee_vec = _process_overnight_charge(
        overnight_charge, timeframe, date,
    )

    # --- scalar validation -------------------------------------------
    if leverage <= 0 or not np.isfinite(leverage):
        raise ValueError(f"leverage must be positive finite, got {leverage!r}")
    if commission < 0 or not np.isfinite(commission):
        raise ValueError(f"commission must be non-negative finite, got {commission!r}")
    if starting_balance <= 0 or not np.isfinite(starting_balance):
        raise ValueError("starting_balance must be positive finite")
    if not isinstance(max_positions, (int, np.integer)) or max_positions < 1:
        raise ValueError("max_positions must be an int >= 1")

    # --- percent_at_risk sanity check (see backtester.py for detail) -
    if method_code == 4:  # percent_at_risk
        if SL is None:
            raise ValueError(
                "position_sizing='percent_at_risk' requires a non-None SL "
                "(the stop defines the risk denominator)."
            )
        valid_sl = sl_arr[~np.isnan(sl_arr)]
        if valid_sl.size == 0:
            raise ValueError(
                "position_sizing='percent_at_risk' requires at least one "
                "non-NaN SL value; the provided SL array is entirely NaN."
            )
        min_sl_dist = float(valid_sl.min())
        ref_price = float(np.median(c_arr))
        required_leverage = static_size * ref_price / min_sl_dist
        if required_leverage > leverage:
            min_sl_pips = min_sl_dist / pip_equals
            raise ValueError(
                f"percent_at_risk={static_size} with tightest SL={min_sl_dist:.6f} "
                f"({min_sl_pips:.2f} pips at pip_equals={pip_equals}) "
                f"at reference price {ref_price:.4f} requires leverage "
                f">= {required_leverage:.1f}, but leverage={leverage}. "
                f"Reduce risk %, widen SL, or increase leverage."
            )

    # --- dispatch -----------------------------------------------------
    if _should_use_fast_path(method_code):
        cash, equity, closed = _inner_loop_cy(
            date, o_arr, h_arr, l_arr, c_arr, v_arr,
            long_entries_v, long_exits_v, short_entries_v, short_exits_v,
            float(starting_balance),
            method_code, static_size, sizes_array,
            sl_arr, tp_val, ts_val,
            float(leverage), float(commission),
            spread_arr, slippage_arr,
            long_fee_vec, short_fee_vec,
            int(max_positions), bool(hedging),
        )
    else:
        # Pure-Python fallback: custom sizer, OR compiled extension unavailable
        cash, equity, closed = _inner_loop_py(
            date, o_arr, h_arr, l_arr, c_arr, v_arr,
            long_entries_v, long_exits_v, short_entries_v, short_exits_v,
            float(starting_balance),
            method_code, static_size, sizes_array, sizing_fn,
            sl_arr, tp_val, ts_val,
            float(leverage), float(commission),
            spread_arr, slippage_arr,
            long_fee_vec, short_fee_vec,
            int(max_positions), bool(hedging),
        )

    return Result(cash, equity, closed, timeframe, date)


# ---------------------------------------------------------------------------
# Utility: did we actually get the compiled path?
# ---------------------------------------------------------------------------
def cython_available() -> bool:
    """Return True iff the compiled ``_cy_inner`` extension was importable."""
    return _CYTHON_AVAILABLE


def cython_import_error() -> Optional[ImportError]:
    """Return the ImportError encountered at module load, or None."""
    return None if _CYTHON_AVAILABLE else _CYTHON_IMPORT_ERROR
