"""
Backtester skeleton.

Note on @njit: profiling showed a pure numpy Python loop running ~10x faster
than an @njit-compiled equivalent on ~10 years of daily data. @njit is
therefore left off by default. Apply it to the inner loop only if/when
moving to much larger datasets (e.g. minute data) or large parameter sweeps
where JIT compile cost amortizes.
"""

from typing import Callable, Optional, Union
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trade-record field layout.
# A "trade record" is a 20-float row used for both currently-open positions
# and closed trades. Columns are accessed via these constants; never by int
# literal in the loop. Layout is shared so closing a position is a row-copy.
# ---------------------------------------------------------------------------
F_DIRECTION     = 0    # +1 long, -1 short
F_ENTRY_BAR     = 1    # int (cast to float)
F_ENTRY_TIME    = 2    # datetime64[ns] as int64 (cast to float)
F_ENTRY_PRICE   = 3    # fill price after spread + slippage
F_EXIT_BAR      = 4
F_EXIT_TIME     = 5
F_EXIT_PRICE    = 6
F_SIZE          = 7    # units held (always positive; direction column gives sign)
F_SL            = 8    # absolute stop-loss price; nan = no SL
F_TP            = 9    # absolute take-profit price; nan = no TP
F_TS_DIST       = 10   # trailing-stop distance in price units; nan = no TS
F_TS_PEAK       = 11   # running peak (long) or trough (short); nan until first update
F_COMMISSION    = 12   # running total of commission paid (entry + exit)
F_SPREAD_COST   = 13   # running total of spread cost (entry + exit)
F_SLIPPAGE_COST = 14   # running total of slippage cost (entry + exit)
F_OVERNIGHT     = 15   # running total of overnight financing charges
F_MAE           = 16   # max adverse excursion (worst unrealized P&L, negative)
F_MFE           = 17   # max favorable excursion (best unrealized P&L, positive)
F_EXIT_REASON   = 18   # int code, see EXIT_* constants below
F_BARS_HELD     = 19
N_FIELDS        = 20

# Exit reason codes (stored in F_EXIT_REASON as float; cast back to int on read)
EXIT_SIGNAL       = 0
EXIT_SL           = 1
EXIT_TP           = 2
EXIT_TS           = 3
EXIT_LIQUIDATION  = 4
EXIT_END_OF_DATA  = 5

_EXIT_REASON_NAMES = {
    EXIT_SIGNAL:      "signal",
    EXIT_SL:          "sl",
    EXIT_TP:          "tp",
    EXIT_TS:          "ts",
    EXIT_LIQUIDATION: "liquidation",
    EXIT_END_OF_DATA: "end_of_data",
}


# ---------------------------------------------------------------------------
# INTERNAL HELPERS — position-array bookkeeping.
# Pulled out of the main loop so list-sync-style bugs can't happen: every
# enter/exit goes through these three functions.
# ---------------------------------------------------------------------------
def _find_free_slot(slot_active: np.ndarray) -> int:
    """Return the index of the first inactive slot, or -1 if all are in use."""
    for k in range(slot_active.size):
        if not slot_active[k]:
            return k
    return -1


def _enter_position(
    slot_idx: int,
    direction: int,
    bar_idx: int,
    entry_time_ns: int,
    entry_price: float,
    size: float,
    sl_price: float,
    tp_price: float,
    ts_dist: float,
    commission_cost: float,
    spread_cost: float,
    slippage_cost: float,
    open_positions: np.ndarray,
    slot_active: np.ndarray,
) -> None:
    """Write a new position into open_positions[slot_idx]; flip slot active."""
    row = open_positions[slot_idx]
    row[F_DIRECTION]     = direction
    row[F_ENTRY_BAR]     = bar_idx
    row[F_ENTRY_TIME]    = entry_time_ns
    row[F_ENTRY_PRICE]   = entry_price
    row[F_EXIT_BAR]      = np.nan
    row[F_EXIT_TIME]     = np.nan
    row[F_EXIT_PRICE]    = np.nan
    row[F_SIZE]          = size
    row[F_SL]            = sl_price
    row[F_TP]            = tp_price
    row[F_TS_DIST]       = ts_dist
    row[F_TS_PEAK]       = entry_price  # initialise to entry; updates on first bar
    row[F_COMMISSION]    = commission_cost
    row[F_SPREAD_COST]   = spread_cost
    row[F_SLIPPAGE_COST] = slippage_cost
    row[F_OVERNIGHT]     = 0.0
    row[F_MAE]           = 0.0
    row[F_MFE]           = 0.0
    row[F_EXIT_REASON]   = np.nan
    row[F_BARS_HELD]     = 0
    slot_active[slot_idx] = True


def _exit_position(
    slot_idx: int,
    bar_idx: int,
    exit_time_ns: int,
    exit_price: float,
    exit_reason: int,
    exit_commission: float,
    exit_spread: float,
    exit_slippage: float,
    open_positions: np.ndarray,
    slot_active: np.ndarray,
    closed_trades: np.ndarray,
    n_closed: int,
) -> int:
    """
    Close the position in slot_idx: finalise its exit fields, copy the row
    into closed_trades[n_closed], clear the slot, return the incremented
    closed-count.

    EDGE CASE — very short trades combined with high max_positions could
    theoretically produce more closed trades than bars (e.g. open and close
    multiple positions on the same bar across many slots). closed_trades is
    pre-allocated to n_bars as a memory-efficient upper bound for typical
    use; if that's exceeded we raise clearly rather than silently overflow.
    """
    if n_closed >= closed_trades.shape[0]:
        raise RuntimeError(
            f"closed_trades pre-allocation exhausted at n_closed={n_closed}. "
            f"Very short trades with high max_positions can exceed the "
            f"default n_bars cap. Increase pre-allocation or reduce "
            f"max_positions."
        )

    row = open_positions[slot_idx]
    row[F_EXIT_BAR]      = bar_idx
    row[F_EXIT_TIME]     = exit_time_ns
    row[F_EXIT_PRICE]    = exit_price
    row[F_EXIT_REASON]   = exit_reason
    row[F_COMMISSION]   += exit_commission
    row[F_SPREAD_COST]  += exit_spread
    row[F_SLIPPAGE_COST] += exit_slippage
    row[F_BARS_HELD]     = bar_idx - int(row[F_ENTRY_BAR])

    closed_trades[n_closed] = row
    slot_active[slot_idx] = False
    return n_closed + 1


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
) -> "Result":
    """
    Run a single backtest over OHLCV and signal arrays.

    overnight_charge is a tuple of (annual_base, annual_borrow). The base is
    applied to all open positions regardless of direction; the borrow is the
    direction-dependent component (typically +/- depending on long vs short).

    Returns a Result object holding timeseries of cash, equity, and all trades.
    """
    # --- strip & validate OHLCV ----------------------------------------
    date, o_arr, h_arr, l_arr, c_arr, v_arr = _process_series(o, h, l, c, v)
    n = o_arr.size

    # --- signals (shift by +1: close-of-bar signal → next-bar-open fill) ---
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
    # SL, TP, TS, spread, and slippage are all supplied in pip units for a
    # consistent mental model ("20-pip stop, 2-pip spread"). They are
    # converted to price-unit distances here — the rest of the pipeline and
    # the inner loop continue to operate in price units with no changes.
    #
    # pip_equals sets the conversion: 0.0001 for FX majors, 1.0 for equity
    # spread-betting (where 1p quote tick = 1 pip), 0.01 for JPY pairs, etc.

    # --- spread & slippage (pips → price units) -----------------------
    spread_arr   = _process_spread_slippage(spread,   n, "spread")   * pip_equals
    slippage_arr = _process_spread_slippage(slippage, n, "slippage") * pip_equals

    # --- SL / TP / TS (pips → price units) ----------------------------
    # SL may be scalar or per-bar array (for ATR-/vol-based stops).
    # TP and TS are scalars by current design.
    if SL is None:
        sl_arr = np.full(n, np.nan, dtype=np.float64)
    elif isinstance(SL, (int, float, np.integer, np.floating)):
        sl_arr = np.full(n, float(SL) * pip_equals, dtype=np.float64)
    else:
        sl_arr = _process_spread_slippage(SL, n, "SL") * pip_equals

    tp_val = float("nan") if TP is None else float(TP) * pip_equals
    ts_val = float("nan") if TS is None else float(TS) * pip_equals

    # --- position sizing ------------------------------------------------
    method_code, static_size, sizes_array, sizing_fn = _process_position_sizing(
        position_sizing, position_percent_equity, position_value,
        position_sizes, position_sizing_fn, n,
        position_percent_at_risk=position_percent_at_risk,
    )

    # --- overnight financing vectors ------------------------------------
    long_fee_vec, short_fee_vec = _process_overnight_charge(
        overnight_charge, timeframe, date,
    )

    # --- scalar validation ---------------------------------------------
    if leverage <= 0 or not np.isfinite(leverage):
        raise ValueError(f"leverage must be positive finite, got {leverage!r}")
    if commission < 0 or not np.isfinite(commission):
        raise ValueError(f"commission must be non-negative finite, got {commission!r}")
    if starting_balance <= 0 or not np.isfinite(starting_balance):
        raise ValueError(f"starting_balance must be positive finite")
    if not isinstance(max_positions, (int, np.integer)) or max_positions < 1:
        raise ValueError("max_positions must be an int >= 1")

    # --- percent_at_risk sanity check ----------------------------------
    # Must have an SL. Also check that the worst-case (tightest stop,
    # median price) doesn't require more leverage than the configured
    # `leverage`. This is a sanity check — it assumes first-bar-ish prices
    # are representative. Actual entries with extreme prices may still be
    # skipped silently at entry time if their required notional exceeds
    # available cash.
    #
    # FUTURE UPGRADE: expose a `when_max_leverage_breached` kwarg that
    # lets the user pick between: (a) raise upfront (current behaviour),
    # (b) cap size at max leverage and continue, (c) skip the trade and
    # log to a diagnostics stream. Added complexity not worth it for
    # most use cases — well-chosen sizing should stay clear of the
    # leverage ceiling anyway.
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
            # Convert the tightest stop back to pips for a user-friendly hint.
            min_sl_pips = min_sl_dist / pip_equals
            raise ValueError(
                f"percent_at_risk={static_size} with tightest SL={min_sl_dist:.6f} "
                f"({min_sl_pips:.2f} pips at pip_equals={pip_equals}) "
                f"at reference price {ref_price:.4f} requires leverage "
                f">= {required_leverage:.1f}, but leverage={leverage}. "
                f"Reduce risk %, widen SL, or increase leverage."
            )

    # --- run ------------------------------------------------------------
    cash, equity, closed = _inner_loop(
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

    return Result(cash, equity, closed, timeframe)


def _process_series(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    v: pd.Series,
) -> tuple:
    """
    Validate 5 OHLCV pandas Series and strip to aligned numpy arrays.

    Checks:
        - Each input is a pd.Series with a DatetimeIndex.
        - All 5 have the same length.
        - All 5 share identical indexes (required for correctness of the
          single ``date`` output returned).
        - No NaN, no inf, no zero values in any series.
        - Values cast to contiguous float64.

    Returns
    -------
    date : np.ndarray  (datetime64[ns])
    o, h, l, c, v : np.ndarray  (float64)
        All six arrays have identical length.

    Raises
    ------
    TypeError
        If any input is not a pd.Series or its index is not a DatetimeIndex.
    ValueError
        If lengths differ, indexes mismatch, or series contain NaN / inf / 0.
    """
    inputs = {"o": o, "h": h, "l": l, "c": c, "v": v}

    # Type checks
    for name, s in inputs.items():
        if not isinstance(s, pd.Series):
            raise TypeError(
                f"{name!r} must be a pandas Series, got {type(s).__name__}"
            )
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(
                f"{name!r} must have a DatetimeIndex, "
                f"got {type(s.index).__name__}"
            )

    # Length check
    lengths = {name: len(s) for name, s in inputs.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"OHLCV series lengths differ: {lengths}")

    # Index alignment check
    ref_index = o.index
    for name, s in inputs.items():
        if not s.index.equals(ref_index):
            raise ValueError(f"{name!r} index does not match o index")

    # Extract date array (shared across all series)
    date = np.asarray(ref_index.values, dtype="datetime64[ns]")

    # Extract + validate each value array
    arrays = {}
    for name, s in inputs.items():
        arr = np.ascontiguousarray(s.to_numpy(dtype=np.float64))
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name!r} contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name!r} contains inf values")
        if np.any(arr == 0):
            raise ValueError(f"{name!r} contains zero values")
        arrays[name] = arr

    return date, arrays["o"], arrays["h"], arrays["l"], arrays["c"], arrays["v"]


_SIZING_METHODS = ("percent_equity", "value", "precomputed", "custom", "percent_at_risk")
_SIZING_CODES = {name: i for i, name in enumerate(_SIZING_METHODS)}


def _process_signals(
    signals: Optional[Union[np.ndarray, pd.Series]],
    n: int,
    name: str = "signals",
) -> np.ndarray:
    """
    INTERNAL HELPER — not part of the public API.

    Strip a signal input (long_entries / long_exits / short_entries /
    short_exits) to a validated length-``n`` contiguous boolean ndarray.

    - ``None`` → all-False array of length n (signal never fires).
    - pd.Series or np.ndarray → validated for length, 1-D, and cast to bool.
    - Any other type raises.

    Parameters
    ----------
    signals : None | np.ndarray | pd.Series
        Per-bar boolean signal array, or None for "never fires".
    n : int
        Required length.
    name : str
        Label used in error messages (e.g. "long_entries").

    Returns
    -------
    np.ndarray
        Length-n, contiguous, dtype=bool.

    Raises
    ------
    TypeError
        On unsupported input type.
    ValueError
        On wrong length or non-1-D input.
    """
    if signals is None:
        return np.zeros(n, dtype=bool)

    if isinstance(signals, pd.Series):
        arr = signals.to_numpy()
    elif isinstance(signals, np.ndarray):
        arr = signals
    else:
        raise TypeError(
            f"{name} must be np.ndarray, pd.Series, or None, "
            f"got {type(signals).__name__}"
        )

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {arr.ndim}-D")
    if arr.size != n:
        raise ValueError(
            f"{name} length {arr.size} does not match required {n}"
        )

    return np.ascontiguousarray(arr.astype(bool))


def _process_position_sizing(
    position_sizing: str,
    position_percent_equity: Optional[float],
    position_value: Optional[float],
    position_sizes: Optional[Union[np.ndarray, pd.Series]],
    position_sizing_fn: Optional[Callable],
    n: int,
    position_percent_at_risk: Optional[float] = None,
) -> tuple:
    """
    Resolve the sizing method and its associated argument into a uniform
    internal form for the inner loop.

    Parameters
    ----------
    position_sizing : str
        One of:
            "percent_equity"   — uses ``position_percent_equity`` (fraction
                of equity per new position; leverage applies on top).
            "value"            — uses ``position_value`` (fixed notional
                per new position, in account currency).
            "precomputed"      — uses ``position_sizes`` (per-bar array
                of sizes, length n, indexed at signal time).
            "custom"           — uses ``position_sizing_fn`` (callable
                receiving full state and returning a size).
            "percent_at_risk"  — uses ``position_percent_at_risk`` together
                with the bar's SL distance. Size is chosen so that if the
                stop is hit, exactly ``fraction × equity`` is lost.
                Requires a non-NaN SL at entry time; ``run_single_backtest``
                validates this upfront.
    position_percent_equity, position_value, position_percent_at_risk : float, optional
        Scalars for the corresponding methods. Must be > 0.
    position_sizes : np.ndarray | pd.Series, optional
        Per-bar sizes for "precomputed". Length n, finite, non-negative.
    position_sizing_fn : Callable, optional
        User callable for "custom".
    n : int
        Number of bars; used to validate ``position_sizes`` length.

    Returns
    -------
    (method_code, static_size, sizes_array, sizing_fn) : tuple
        method_code  : int
            0=percent_equity, 1=value, 2=precomputed, 3=custom,
            4=percent_at_risk.
        static_size  : float
            Scalar for methods 0, 1, 4; 0.0 otherwise.
        sizes_array  : np.ndarray
            Per-bar array for method 2; empty otherwise.
        sizing_fn    : Callable | None
            Callable for method 3; None otherwise.

    Raises
    ------
    ValueError
        On unknown method, missing required arg, or invalid values.
    TypeError
        On wrong argument types.
    """
    if position_sizing not in _SIZING_CODES:
        raise ValueError(
            f"position_sizing must be one of {_SIZING_METHODS}, "
            f"got {position_sizing!r}"
        )
    method_code = _SIZING_CODES[position_sizing]

    # Default return slots
    static_size = 0.0
    sizes_array = np.empty(0, dtype=np.float64)
    sizing_fn = None

    if position_sizing == "percent_equity":
        if position_percent_equity is None:
            raise ValueError(
                "position_sizing='percent_equity' requires position_percent_equity"
            )
        if not isinstance(position_percent_equity, (int, float, np.integer, np.floating)):
            raise TypeError("position_percent_equity must be a number")
        if not np.isfinite(position_percent_equity) or position_percent_equity <= 0:
            raise ValueError(
                f"position_percent_equity must be a positive finite number, "
                f"got {position_percent_equity!r}"
            )
        static_size = float(position_percent_equity)

    elif position_sizing == "value":
        if position_value is None:
            raise ValueError(
                "position_sizing='value' requires position_value"
            )
        if not isinstance(position_value, (int, float, np.integer, np.floating)):
            raise TypeError("position_value must be a number")
        if not np.isfinite(position_value) or position_value <= 0:
            raise ValueError(
                f"position_value must be a positive finite number, "
                f"got {position_value!r}"
            )
        static_size = float(position_value)

    elif position_sizing == "precomputed":
        if position_sizes is None:
            raise ValueError(
                "position_sizing='precomputed' requires position_sizes"
            )
        if isinstance(position_sizes, pd.Series):
            arr = np.ascontiguousarray(position_sizes.to_numpy(dtype=np.float64))
        elif isinstance(position_sizes, np.ndarray):
            arr = np.ascontiguousarray(position_sizes, dtype=np.float64)
        else:
            raise TypeError(
                f"position_sizes must be np.ndarray or pd.Series, "
                f"got {type(position_sizes).__name__}"
            )
        if arr.ndim != 1:
            raise ValueError(f"position_sizes must be 1-D, got {arr.ndim}-D")
        if arr.size != n:
            raise ValueError(
                f"position_sizes length {arr.size} does not match n={n}"
            )
        if np.any(np.isnan(arr)):
            raise ValueError("position_sizes contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError("position_sizes contains inf values")
        if np.any(arr < 0):
            raise ValueError("position_sizes contains negative values")
        sizes_array = arr

    elif position_sizing == "custom":
        if position_sizing_fn is None:
            raise ValueError(
                "position_sizing='custom' requires position_sizing_fn"
            )
        if not callable(position_sizing_fn):
            raise TypeError("position_sizing_fn must be callable")
        sizing_fn = position_sizing_fn

    elif position_sizing == "percent_at_risk":
        if position_percent_at_risk is None:
            raise ValueError(
                "position_sizing='percent_at_risk' requires position_percent_at_risk"
            )
        if not isinstance(position_percent_at_risk,
                          (int, float, np.integer, np.floating)):
            raise TypeError("position_percent_at_risk must be a number")
        if (not np.isfinite(position_percent_at_risk)
                or position_percent_at_risk <= 0
                or position_percent_at_risk >= 1):
            raise ValueError(
                f"position_percent_at_risk must be in (0, 1), "
                f"got {position_percent_at_risk!r}"
            )
        static_size = float(position_percent_at_risk)

    return method_code, static_size, sizes_array, sizing_fn


def _process_spread_slippage(
    value: Union[float, np.ndarray, pd.Series],
    n: int,
    name: str = "value",
) -> np.ndarray:
    """
    Normalize a spread OR slippage input to a length-``n`` float64 ndarray.

    Scalars are broadcast; arrays/Series are validated and cast.
    Used identically for both spread and slippage inputs — pass ``name``
    for clearer error messages.

    Parameters
    ----------
    value : float | np.ndarray | pd.Series
        Scalar (applied uniformly) or per-bar values.
    n : int
        Required length for array inputs; broadcast target for scalars.
    name : str
        Label used in error messages (e.g. "spread", "slippage").

    Returns
    -------
    np.ndarray
        Length-``n``, contiguous, float64.

    Raises
    ------
    TypeError
        On unsupported input type.
    ValueError
        On wrong length, NaN, or inf.
    """
    # Scalar path
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{name} scalar must be finite, got {value!r}")
        return np.full(n, float(value), dtype=np.float64)

    # Array-like path
    if isinstance(value, pd.Series):
        arr = np.ascontiguousarray(value.to_numpy(dtype=np.float64))
    elif isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value, dtype=np.float64)
    else:
        raise TypeError(
            f"{name} must be a scalar, numpy array, or pandas Series, "
            f"got {type(value).__name__}"
        )

    if arr.ndim != 1:
        raise ValueError(f"{name} array must be 1-D, got {arr.ndim}-D")
    if arr.size != n:
        raise ValueError(
            f"{name} array length {arr.size} does not match required {n}"
        )
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains inf values")

    return arr


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday")


def _process_overnight_charge(
    overnight_charge: tuple,
    timeframe: str,
    date: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    denominator: int = 360,
    rollover_hour_utc: int = 22,
    triple_charge_weekday: str = "Wednesday",
) -> tuple:
    """
    Convert annual overnight financing rates into per-bar VECTORS of long
    and short charges, with zeros on non-rollover bars.

    Moves all rollover-timing logic out of the hot loop: for any bar i the
    main backtester just multiplies ``long_vec[i] * notional`` (or short),
    with zero being a no-op on intraday bars that don't cross a rollover.

    Parameters
    ----------
    overnight_charge : tuple[float, float]
        ``(annual_base, annual_borrow)``.

        - ``annual_base``  — fixed financing component applied regardless
          of direction (e.g. funding / risk-free rate).
        - ``annual_borrow`` — direction-dependent component. Conventionally
          debited from shorts and credited to / reducing cost on longs.
          Sign convention applied here:
              long_daily  = (annual_base - annual_borrow) / denominator
              short_daily = (annual_base + annual_borrow) / denominator
          Flip the sign of ``annual_borrow`` at the call site if your
          venue's convention is inverted.

    timeframe : str
        Bar granularity key (see ``_BARS_PER_YEAR``). Used for validation
        and documentation only — actual bar-duration logic is derived from
        ``date`` itself.

    date : np.ndarray | pd.Series | pd.DatetimeIndex
        Bar CLOSE timestamps, length n. Stripped to a contiguous
        ``datetime64[ns]`` ndarray internally.

    denominator : int, default 360
        Day-count convention for annual → daily conversion.
        - 360: ACT/360. Standard for most money-market and FX/CFD financing.
        - 365: ACT/365. GBP money markets, some crypto venues.

    rollover_hour_utc : int, default 22
        UTC hour at which the daily rollover occurs. 22:00 UTC (~17:00 NY)
        is the standard IB / FX / CFD rollover time.

    triple_charge_weekday : str, default "Wednesday"
        Full weekday name on which a 3× charge is applied to cover the
        weekend. FX/CFD brokers typically use Wednesday.

    Returns
    -------
    long_vec, short_vec : tuple[np.ndarray, np.ndarray]
        Per-bar financing charges, length n, float64. Fractions of notional.
        Positive values are DEBITS. Zero on bars without a rollover.

    Notes
    -----
    Attribution: a rollover at time R is charged to the first bar i with
    ``date[i] >= R``. Consequences for daily data:

        - Timestamps at END of day (e.g. 23:59): Wednesday's 22:00 rollover
          falls inside Wednesday's bar span → triple appears on the WED bar.
        - Timestamps at START of day / midnight: Wednesday's 22:00 rollover
          falls between Wed midnight and Thu midnight → triple appears on
          the THU bar. If you want the triple on the Wed bar with
          midnight-timestamped data, either shift timestamps by one bar or
          set ``triple_charge_weekday="Thursday"``.

    Bar 0 always carries zero charge (no prior bar to have held from).

    For weekly / monthly bars the vector naturally accumulates all rollovers
    within each bar's span (e.g. a weekly bar gets ~7 days of financing with
    one Wed rollover counted as 3, totalling 9 day-equivalents).
    """
    # ---- validate scalars ------------------------------------------------
    if not isinstance(overnight_charge, tuple) or len(overnight_charge) != 2:
        raise TypeError(
            "overnight_charge must be a tuple of (annual_base, annual_borrow)"
        )
    annual_base, annual_borrow = overnight_charge
    if not (np.isfinite(annual_base) and np.isfinite(annual_borrow)):
        raise ValueError("overnight_charge values must be finite")

    if timeframe not in _BARS_PER_YEAR:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; "
            f"expected one of {list(_BARS_PER_YEAR)}"
        )

    if not isinstance(rollover_hour_utc, (int, np.integer)) or not (
        0 <= rollover_hour_utc <= 23
    ):
        raise ValueError("rollover_hour_utc must be an int in [0, 23]")

    if triple_charge_weekday not in _WEEKDAYS:
        raise ValueError(
            f"triple_charge_weekday must be one of {_WEEKDAYS}, "
            f"got {triple_charge_weekday!r}"
        )
    triple_charge_weekday = _WEEKDAYS.index(triple_charge_weekday)

    # ---- strip date to ndarray ------------------------------------------
    if isinstance(date, pd.Series):
        date_arr = np.ascontiguousarray(date.to_numpy(dtype="datetime64[ns]"))
    elif isinstance(date, pd.DatetimeIndex):
        date_arr = np.ascontiguousarray(date.values.astype("datetime64[ns]"))
    elif isinstance(date, np.ndarray):
        date_arr = np.ascontiguousarray(date.astype("datetime64[ns]"))
    else:
        raise TypeError(
            f"date must be np.ndarray, pd.Series, or pd.DatetimeIndex, "
            f"got {type(date).__name__}"
        )

    n = date_arr.size
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # ---- daily rates -----------------------------------------------------
    daily_base = float(annual_base) / denominator
    daily_borrow = float(annual_borrow) / denominator
    long_daily = daily_base - daily_borrow
    short_daily = daily_base + daily_borrow

    # ---- enumerate rollover moments in (date[0], date[-1]] --------------
    t_start = pd.Timestamp(date_arr[0])
    t_end = pd.Timestamp(date_arr[-1])

    # First rollover strictly after t_start
    first_rollover = t_start.normalize() + pd.Timedelta(hours=rollover_hour_utc)
    if first_rollover <= t_start:
        first_rollover += pd.Timedelta(days=1)

    if first_rollover > t_end:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    rollovers = pd.date_range(start=first_rollover, end=t_end, freq="D")
    if len(rollovers) == 0:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    # ---- weights: 3× on triple day --------------------------------------
    weights = np.where(
        rollovers.weekday == triple_charge_weekday, 3.0, 1.0
    ).astype(np.float64)

    # ---- bucket into bars ------------------------------------------------
    rollover_ns = rollovers.values.astype("datetime64[ns]")
    bar_idx = np.searchsorted(date_arr, rollover_ns, side="left")

    # drop any that fall past the last bar (shouldn't happen given t_end
    # filtering, but be defensive against exact-equality edge cases)
    valid = bar_idx < n
    bar_idx = bar_idx[valid]
    weights = weights[valid]

    # ---- accumulate into per-bar vectors --------------------------------
    long_vec = np.zeros(n, dtype=np.float64)
    short_vec = np.zeros(n, dtype=np.float64)
    np.add.at(long_vec, bar_idx, weights * long_daily)
    np.add.at(short_vec, bar_idx, weights * short_daily)

    return long_vec, short_vec


# ---------------------------------------------------------------------------
# Timeframe → bars per year. Used for annualizing Sharpe/Sortino/CAGR/etc.
#
# Defaults below follow US EQUITY convention: 252 trading days/year,
# 6.5-hour regular session, no overnight bars. Change these for other venues:
#
#   Crypto (24/7):
#       "1d" = 365, "1h" = 365*24 = 8760, "1m" = 365*24*60 = 525_600
#
#   FX spot / FX spreadbetting (24/5, Sun 22:00 → Fri 22:00 UTC):
#       "1d" ≈ 260, "1h" ≈ 260*24 = 6240
#
#   Stocks / equity ETFs (~252 trading days, 6.5h RTH):
#       "1d" = 252, "1h" = 252*6.5 = 1638. European markets use 8.5h,
#       Asian markets vary (TSE ~5h, HKEX ~5.5h incl. lunch break).
#
#   Commodities / metals futures (CME Globex ~23h, Sun-Fri):
#       "1d" ≈ 252, "1h" ≈ 252*23 = 5796. Per-contract schedules vary.
#
# TODO: expose a bars_per_year override via kwargs on run_single_backtest
# so users don't have to monkey-patch this dict.
# ---------------------------------------------------------------------------
_BARS_PER_YEAR = {
    "1s":  252 * 6.5 * 3600,
    "1m":  252 * 6.5 * 60,
    "5m":  252 * 6.5 * 12,
    "15m": 252 * 6.5 * 4,
    "30m": 252 * 13,
    "1h":  252 * 6.5,
    "4h":  252 * 1.625,
    "1d":  252,
    "1w":  52,
    "1M":  12,
}


# ---------------------------------------------------------------------------
# INTERNAL HELPER — used by Result to compute max consecutive winners/losers.
# Pulled out of the class because it doesn't touch instance state, in line
# with the project's preference for flat private functions over staticmethods.
# Will migrate to utils.py when the file is eventually split.
# ---------------------------------------------------------------------------
def _longest_run_of_true(mask: np.ndarray) -> int:
    """Longest consecutive run of True in a boolean array (vectorized)."""
    if mask.size == 0:
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


# ---------------------------------------------------------------------------
# INNER LOOP — the event-driven simulation.
#
# Pure Python + numpy. Profiled ~10x faster than @njit on 10 years of daily
# data; keep this form for now and revisit numba only for minute data or
# large sweeps.
#
# Execution semantics (fixed):
#   - Entry and exit signals are SHIFTED by +1 upstream. A signal on bar i's
#     close fills at bar i+1's open.
#   - Within-bar order:
#        1. Apply overnight financing to open positions.
#        2. Update TS peaks, MAE/MFE, then check SL/TP/TS against H/L.
#           SL beats TP when both would hit in the same bar.
#        3. Process shifted exit signals at open[i].
#        4. Process shifted entry signals at open[i].
#        5. Mark-to-market at close[i], write cash[i] and equity[i].
#        6. Liquidation check: if equity[i] <= 0, flatten at synthetic
#           exit prices that bring equity to exactly zero (no clamp; the
#           broker is assumed to have closed at the critical price).
#   - SL/TP/TS values captured at entry and fixed for the position's life
#     (except TS peak/trough, which updates bar-by-bar).
#   - With max_positions > 1 and hedging=False, an opposite-direction signal
#     flattens ALL open positions, then opens the new one.
# ---------------------------------------------------------------------------
def _inner_loop(
    date: np.ndarray,
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray,
    long_entries_shifted: np.ndarray,
    long_exits_shifted: np.ndarray,
    short_entries_shifted: np.ndarray,
    short_exits_shifted: np.ndarray,
    starting_balance: float,
    sizing_method_code: int,
    sizing_static: float,
    sizing_array: np.ndarray,
    sizing_fn: Optional[Callable],
    sl_arr: np.ndarray,       # per-bar SL distance (price units); nan = no SL
    tp: float,                # scalar TP distance (price units); nan = no TP
    ts: float,                # scalar TS distance (price units); nan = no TS
    leverage: float,
    commission: float,        # fraction of notional
    spread_arr: np.ndarray,   # per-bar half-spread (price units)
    slippage_arr: np.ndarray, # per-bar slippage (price units)
    long_fee_vec: np.ndarray,
    short_fee_vec: np.ndarray,
    max_positions: int,
    hedging: bool,
) -> tuple:
    """
    Run the event-driven simulation. Returns (cash, equity, closed_trades)
    where closed_trades is trimmed to n_closed rows.
    """
    n = o.size

    # --- state ----------------------------------------------------------
    cash = np.empty(n, dtype=np.float64)
    equity = np.empty(n, dtype=np.float64)

    n_slots = max_positions * 2 if hedging else max_positions
    open_positions = np.full((n_slots, N_FIELDS), np.nan, dtype=np.float64)
    slot_active = np.zeros(n_slots, dtype=bool)

    # See _exit_position for the overflow caveat. n_bars is the memory-
    # efficient upper bound for typical use; runtime check guards against
    # pathological cases.
    closed_trades = np.full((n, N_FIELDS), np.nan, dtype=np.float64)
    n_closed = 0

    current_cash = starting_balance

    # Convenience: date as int64 ns for fast assignment into float rows
    date_ns = date.astype("datetime64[ns]").astype(np.int64)

    # --- main loop ------------------------------------------------------
    for i in range(n):
        price_o = o[i]
        price_h = h[i]
        price_l = l[i]
        price_c = c[i]
        t_ns = int(date_ns[i])

        # (1) Overnight financing on currently-open positions --------------
        if long_fee_vec[i] != 0.0 or short_fee_vec[i] != 0.0:
            for k in range(n_slots):
                if not slot_active[k]:
                    continue
                row = open_positions[k]
                notional = row[F_SIZE] * row[F_ENTRY_PRICE]
                if row[F_DIRECTION] > 0:
                    charge = long_fee_vec[i] * notional
                else:
                    charge = short_fee_vec[i] * notional
                row[F_OVERNIGHT] += charge
                current_cash -= charge

        # (2) Update TS peaks, MAE/MFE, then check SL/TP/TS ----------------
        for k in range(n_slots):
            if not slot_active[k]:
                continue
            row = open_positions[k]
            direction = row[F_DIRECTION]
            entry_px = row[F_ENTRY_PRICE]
            size = row[F_SIZE]

            # Update trailing-stop peak
            ts_dist = row[F_TS_DIST]
            if not np.isnan(ts_dist):
                if direction > 0 and price_h > row[F_TS_PEAK]:
                    row[F_TS_PEAK] = price_h
                elif direction < 0 and price_l < row[F_TS_PEAK]:
                    row[F_TS_PEAK] = price_l

            # MAE/MFE in P&L terms (use bar's adverse/favorable extreme)
            if direction > 0:
                worst_px, best_px = price_l, price_h
            else:
                worst_px, best_px = price_h, price_l
            worst_pnl = direction * (worst_px - entry_px) * size
            best_pnl = direction * (best_px - entry_px) * size
            if worst_pnl < row[F_MAE]:
                row[F_MAE] = worst_pnl
            if best_pnl > row[F_MFE]:
                row[F_MFE] = best_pnl

            # SL / TP / TS checks. SL wins if both SL and TP would hit.
            sl_px = row[F_SL]
            tp_px = row[F_TP]
            ts_trigger_px = np.nan
            if not np.isnan(ts_dist):
                if direction > 0:
                    ts_trigger_px = row[F_TS_PEAK] - ts_dist
                else:
                    ts_trigger_px = row[F_TS_PEAK] + ts_dist

            # Determine which (if any) fires. SL has priority over TP.
            exit_reason = -1
            exit_px = np.nan
            if direction > 0:
                if not np.isnan(sl_px) and price_l <= sl_px:
                    exit_reason = EXIT_SL
                    exit_px = sl_px
                elif not np.isnan(tp_px) and price_h >= tp_px:
                    exit_reason = EXIT_TP
                    exit_px = tp_px
                elif not np.isnan(ts_trigger_px) and price_l <= ts_trigger_px:
                    exit_reason = EXIT_TS
                    exit_px = ts_trigger_px
            else:  # short
                if not np.isnan(sl_px) and price_h >= sl_px:
                    exit_reason = EXIT_SL
                    exit_px = sl_px
                elif not np.isnan(tp_px) and price_l <= tp_px:
                    exit_reason = EXIT_TP
                    exit_px = tp_px
                elif not np.isnan(ts_trigger_px) and price_h >= ts_trigger_px:
                    exit_reason = EXIT_TS
                    exit_px = ts_trigger_px

            if exit_reason != -1:
                # Apply exit costs
                exit_spread = spread_arr[i] * size
                exit_slippage = slippage_arr[i] * size
                # Spread/slippage push fill price against us
                exit_px_net = exit_px - direction * (spread_arr[i] + slippage_arr[i])
                exit_commission = commission * abs(exit_px_net * size)
                # Proceeds back to cash
                proceeds = direction * (exit_px_net - entry_px) * size
                current_cash += (size * entry_px) + proceeds  # return margin + pnl
                current_cash -= exit_commission
                n_closed = _exit_position(
                    k, i, t_ns, exit_px_net, exit_reason,
                    exit_commission, exit_spread, exit_slippage,
                    open_positions, slot_active, closed_trades, n_closed,
                )

        # (3) Shifted exit signals at open[i] ------------------------------
        want_long_exit = long_exits_shifted[i]
        want_short_exit = short_exits_shifted[i]
        if want_long_exit or want_short_exit:
            for k in range(n_slots):
                if not slot_active[k]:
                    continue
                row = open_positions[k]
                direction = row[F_DIRECTION]
                if (direction > 0 and want_long_exit) or (direction < 0 and want_short_exit):
                    size = row[F_SIZE]
                    entry_px = row[F_ENTRY_PRICE]
                    exit_px_net = price_o - direction * (spread_arr[i] + slippage_arr[i])
                    exit_commission = commission * abs(exit_px_net * size)
                    proceeds = direction * (exit_px_net - entry_px) * size
                    current_cash += (size * entry_px) + proceeds
                    current_cash -= exit_commission
                    n_closed = _exit_position(
                        k, i, t_ns, exit_px_net, EXIT_SIGNAL,
                        exit_commission,
                        spread_arr[i] * size,
                        slippage_arr[i] * size,
                        open_positions, slot_active, closed_trades, n_closed,
                    )

        # (4) Shifted entry signals at open[i] -----------------------------
        want_long_entry = long_entries_shifted[i]
        want_short_entry = short_entries_shifted[i]

        # If max_positions>1 and hedging=False, an opposite-direction entry
        # flattens all before the new open.
        if not hedging and (want_long_entry or want_short_entry):
            desired_dir = 1 if want_long_entry else -1
            for k in range(n_slots):
                if not slot_active[k]:
                    continue
                row = open_positions[k]
                if row[F_DIRECTION] != desired_dir:
                    size = row[F_SIZE]
                    direction = row[F_DIRECTION]
                    entry_px = row[F_ENTRY_PRICE]
                    exit_px_net = price_o - direction * (spread_arr[i] + slippage_arr[i])
                    exit_commission = commission * abs(exit_px_net * size)
                    proceeds = direction * (exit_px_net - entry_px) * size
                    current_cash += (size * entry_px) + proceeds
                    current_cash -= exit_commission
                    n_closed = _exit_position(
                        k, i, t_ns, exit_px_net, EXIT_SIGNAL,
                        exit_commission,
                        spread_arr[i] * size,
                        slippage_arr[i] * size,
                        open_positions, slot_active, closed_trades, n_closed,
                    )

        # Now open new positions (if any room)
        for desired_dir, want in ((1, want_long_entry), (-1, want_short_entry)):
            if not want:
                continue

            slot_idx = _find_free_slot(slot_active)
            if slot_idx == -1:
                continue  # no free slot

            # Fill price: open + direction * (spread + slippage)
            entry_px_net = price_o + desired_dir * (spread_arr[i] + slippage_arr[i])

            # Compute size
            if sizing_method_code == 0:  # percent_equity
                # Current equity = cash + unrealized P&L of anything still open
                equity_now = current_cash
                for kk in range(n_slots):
                    if slot_active[kk]:
                        r2 = open_positions[kk]
                        d2 = r2[F_DIRECTION]
                        equity_now += d2 * (price_c - r2[F_ENTRY_PRICE]) * r2[F_SIZE]
                size = (sizing_static * equity_now * leverage) / entry_px_net
            elif sizing_method_code == 1:  # value
                size = (sizing_static * leverage) / entry_px_net
            elif sizing_method_code == 2:  # precomputed
                size = sizing_array[i]
            elif sizing_method_code == 4:  # percent_at_risk
                # Size so that if SL is hit, exactly fraction * equity is lost.
                # size = (fraction * equity) / sl_distance
                # Leverage is intentionally NOT applied here — risk is the
                # point of this method. The upfront leverage-sanity check in
                # run_single_backtest ensures the resulting notional is
                # feasible given the configured leverage.
                sl_dist_now = sl_arr[i]
                if np.isnan(sl_dist_now) or sl_dist_now <= 0:
                    # Skip: can't size without a valid stop.
                    continue
                equity_now = current_cash
                for kk in range(n_slots):
                    if slot_active[kk]:
                        r2 = open_positions[kk]
                        d2 = r2[F_DIRECTION]
                        equity_now += d2 * (price_c - r2[F_ENTRY_PRICE]) * r2[F_SIZE]
                size = (sizing_static * equity_now) / sl_dist_now
            else:  # custom callable
                size = float(sizing_fn(
                    i, t_ns, desired_dir, current_cash,
                    open_positions, slot_active, closed_trades, n_closed,
                    date, o, h, l, c, v,
                ))

            if size <= 0 or not np.isfinite(size):
                continue

            # Entry costs
            entry_spread = spread_arr[i] * size
            entry_slippage = slippage_arr[i] * size
            entry_commission = commission * abs(entry_px_net * size)
            margin = size * entry_px_net  # reserved; freed on exit

            if current_cash < margin + entry_commission:
                continue  # insufficient cash

            current_cash -= margin + entry_commission

            # SL / TP / TS absolute prices captured at entry
            sl_dist = sl_arr[i]
            if np.isnan(sl_dist):
                sl_price = np.nan
            else:
                sl_price = entry_px_net - desired_dir * sl_dist
            if np.isnan(tp):
                tp_price = np.nan
            else:
                tp_price = entry_px_net + desired_dir * tp
            ts_dist_val = np.nan if np.isnan(ts) else ts

            _enter_position(
                slot_idx, desired_dir, i, t_ns, entry_px_net, size,
                sl_price, tp_price, ts_dist_val,
                entry_commission, entry_spread, entry_slippage,
                open_positions, slot_active,
            )

        # (5) Mark-to-market at close[i] -----------------------------------
        unrealized = 0.0
        margin_held = 0.0
        for k in range(n_slots):
            if not slot_active[k]:
                continue
            row = open_positions[k]
            d = row[F_DIRECTION]
            unrealized += d * (price_c - row[F_ENTRY_PRICE]) * row[F_SIZE]
            margin_held += row[F_SIZE] * row[F_ENTRY_PRICE]
            row[F_BARS_HELD] = i - int(row[F_ENTRY_BAR])
        equity[i] = current_cash + margin_held + unrealized
        cash[i] = current_cash

        # (6) Liquidation --------------------------------------------------
        # If mark-to-market equity would cross zero on this bar, the broker
        # would have flattened at the critical price. We close each open
        # position at a synthetic exit price such that total realised P&L
        # equals exactly -(cash + total_margin), making equity land at 0.
        # Loss is attributed across positions in proportion to each
        # position's unrealised loss at close[i] (so winners — if any open
        # at liquidation — close at close[i] unchanged). No cash clamp,
        # no negative-equity intermediate state.
        if equity[i] <= 0:
            # Total loss to be taken across all positions
            total_loss_budget = current_cash + margin_held  # equity-before = this + unrealized
            # Collect each position's unrealized at close[i]
            unrealized_by_slot = np.zeros(n_slots, dtype=np.float64)
            total_bad = 0.0
            for k in range(n_slots):
                if not slot_active[k]:
                    continue
                row = open_positions[k]
                d = row[F_DIRECTION]
                u = d * (price_c - row[F_ENTRY_PRICE]) * row[F_SIZE]
                unrealized_by_slot[k] = u
                if u < 0:
                    total_bad += -u  # magnitude of bad unrealized

            for k in range(n_slots):
                if not slot_active[k]:
                    continue
                row = open_positions[k]
                direction = row[F_DIRECTION]
                size = row[F_SIZE]
                entry_px = row[F_ENTRY_PRICE]
                u = unrealized_by_slot[k]

                if u >= 0 or total_bad == 0:
                    # Non-losing leg (or degenerate): close at bar close
                    realised_pnl = u
                    exit_px_net = price_c
                else:
                    # Losing leg: take its share of the loss budget
                    share = (-u) / total_bad
                    realised_pnl = -share * total_loss_budget
                    # Back out the implied exit price
                    # realised = direction * (exit - entry) * size
                    exit_px_net = entry_px + realised_pnl / (direction * size)

                current_cash += (size * entry_px) + realised_pnl
                n_closed = _exit_position(
                    k, i, t_ns, exit_px_net, EXIT_LIQUIDATION,
                    0.0, 0.0, 0.0,
                    open_positions, slot_active, closed_trades, n_closed,
                )

            # Equity now exactly zero (all margin + cash consumed)
            equity[i] = current_cash
            cash[i] = current_cash
            # Fill remaining bars with final equity (blown account)
            if i + 1 < n:
                cash[i + 1:] = current_cash
                equity[i + 1:] = current_cash
            break

    # --- end-of-data close-out -----------------------------------------
    # Any position still open when we run out of bars is closed at the
    # final bar's close price, with exit_reason = EXIT_END_OF_DATA.
    # No spread / slippage / commission: this is a synthetic close, not
    # a real broker fill, and by the time you've run out of bars the
    # precision of these costs is immaterial.
    # Skipped if the loop already broke early due to liquidation.
    if np.any(slot_active):
        final_bar = n - 1
        final_price = c[final_bar]
        final_t_ns = int(date_ns[final_bar])
        for k in range(n_slots):
            if not slot_active[k]:
                continue
            row = open_positions[k]
            direction = row[F_DIRECTION]
            size = row[F_SIZE]
            entry_px = row[F_ENTRY_PRICE]
            realised_pnl = direction * (final_price - entry_px) * size
            current_cash += (size * entry_px) + realised_pnl
            n_closed = _exit_position(
                k, final_bar, final_t_ns, final_price, EXIT_END_OF_DATA,
                0.0, 0.0, 0.0,
                open_positions, slot_active, closed_trades, n_closed,
            )
        # Update final bar's cash/equity: unrealised is now realised
        equity[final_bar] = current_cash
        cash[final_bar] = current_cash

    return cash, equity, closed_trades[:n_closed]


class Result:
    """
    Backtest output: timeseries plus trade log plus computed performance metrics.

    ``self.trades`` is a 2D float64 ndarray of shape (n_trades, N_FIELDS).
    Column layout matches the F_* constants at module top. Empty trades
    (zero rows) are valid.

    Convention: ``risk_free`` and Omega ``threshold`` are ANNUAL rates,
    converted to per-bar internally using ``self.timeframe``.
    """

    def __init__(self, cash: np.ndarray, equity: np.ndarray,
                 trades: np.ndarray, timeframe: str = "1d"):
        self.cash = cash
        self.equity = equity
        self.trades = trades
        self.timeframe = timeframe

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def _bars_per_year(self) -> float:
        return _BARS_PER_YEAR.get(self.timeframe, 252)

    def _returns(self) -> np.ndarray:
        """Per-bar simple (arithmetic) returns from the equity curve."""
        if self.equity.size < 2:
            return np.array([], dtype=np.float64)
        return np.diff(self.equity) / self.equity[:-1]

    def _log_returns(self) -> np.ndarray:
        """Per-bar log returns from the equity curve."""
        if self.equity.size < 2 or np.any(self.equity <= 0):
            return np.array([], dtype=np.float64)
        return np.diff(np.log(self.equity))

    def _pnl(self) -> np.ndarray:
        """
        Net P&L per trade as 1-D float ndarray.

        Derivation: ``direction * (exit_px - entry_px) * size``, minus
        commission and overnight costs. Spread and slippage are NOT
        subtracted here because they're already baked into the stored
        entry and exit prices (the inner loop adjusts the fill price by
        ``direction * (spread + slippage)`` on both legs, so any exit-vs-
        entry arithmetic using those stored prices already reflects them).
        The per-trade ``spread_cost`` / ``slippage_cost`` fields are
        reported for diagnostics only.
        """
        if self.trades.shape[0] == 0:
            return np.array([], dtype=np.float64)
        t = self.trades
        direction = t[:, F_DIRECTION]
        entry = t[:, F_ENTRY_PRICE]
        exit_ = t[:, F_EXIT_PRICE]
        size = t[:, F_SIZE]
        gross = direction * (exit_ - entry) * size
        costs = t[:, F_COMMISSION] + t[:, F_OVERNIGHT]
        return gross - costs

    def _bars_held(self) -> np.ndarray:
        """Bars held by each trade as 1-D int ndarray."""
        if self.trades.shape[0] == 0:
            return np.array([], dtype=np.int64)
        return self.trades[:, F_BARS_HELD].astype(np.int64)

    # ------------------------------------------------------------------ #
    # Return/risk metrics                                                 #
    # ------------------------------------------------------------------ #
    def _calculate_sharpe(self, risk_free: float = 0.0) -> float:
        """
        Annualized Sharpe ratio using SIMPLE (arithmetic) returns.
        ``risk_free`` is annual; converted to per-bar by plain division
        (good approximation for small rates; revisit if using very high rf).
        """
        r = self._returns()
        if r.size == 0 or r.std() == 0:
            return 0.0
        bpy = self._bars_per_year()
        rf_per_bar = risk_free / bpy
        return np.sqrt(bpy) * (r.mean() - rf_per_bar) / r.std()

    def _calculate_log_sharpe(self, risk_free: float = 0.0) -> float:
        """
        Annualized Sharpe ratio using LOG returns.
        Per-bar rf here is log(1+risk_free)/bpy for internal consistency
        with log-return arithmetic.
        """
        r = self._log_returns()
        if r.size == 0 or r.std() == 0:
            return 0.0
        bpy = self._bars_per_year()
        rf_per_bar = np.log(1.0 + risk_free) / bpy
        return np.sqrt(bpy) * (r.mean() - rf_per_bar) / r.std()

    def _calculate_sortino(self, risk_free: float = 0.0) -> float:
        """
        Annualized Sortino ratio using SIMPLE returns.
        Downside deviation measured vs rf_per_bar.
        """
        r = self._returns()
        if r.size == 0:
            return 0.0
        bpy = self._bars_per_year()
        rf_per_bar = risk_free / bpy
        downside = r[r < rf_per_bar] - rf_per_bar
        if downside.size == 0:
            return np.inf
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std == 0:
            return 0.0
        return np.sqrt(bpy) * (r.mean() - rf_per_bar) / downside_std

    def _calculate_log_sortino(self, risk_free: float = 0.0) -> float:
        """Annualized Sortino ratio using LOG returns."""
        r = self._log_returns()
        if r.size == 0:
            return 0.0
        bpy = self._bars_per_year()
        rf_per_bar = np.log(1.0 + risk_free) / bpy
        downside = r[r < rf_per_bar] - rf_per_bar
        if downside.size == 0:
            return np.inf
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std == 0:
            return 0.0
        return np.sqrt(bpy) * (r.mean() - rf_per_bar) / downside_std

    def _calculate_max_drawdown(self) -> float:
        """Peak-to-trough drawdown as a fraction (negative number)."""
        if self.equity.size == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        dd = (self.equity - peak) / peak
        return float(dd.min())

    def _calculate_cagr(self) -> float:
        """Compound annual growth rate of equity."""
        if self.equity.size < 2 or self.equity[0] <= 0 or self.equity[-1] <= 0:
            return 0.0
        years = self.equity.size / self._bars_per_year()
        if years <= 0:
            return 0.0
        return (self.equity[-1] / self.equity[0]) ** (1.0 / years) - 1.0

    def _calculate_calmar(self) -> float:
        """CAGR / |max drawdown|."""
        mdd = self._calculate_max_drawdown()
        if mdd == 0:
            return np.inf
        return self._calculate_cagr() / abs(mdd)

    def _calculate_ulcer_index(self) -> float:
        """Ulcer Index: RMS of percent drawdowns from running peak."""
        if self.equity.size == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        dd_pct = 100.0 * (self.equity - peak) / peak
        return float(np.sqrt(np.mean(dd_pct ** 2)))

    def _k_ratio_components(self):
        """
        Shared OLS fit of log-equity on time index.
        Returns (slope, stderr_slope, n) or None if not computable.
        Different K-ratio variants combine these differently.

        NOTE: formulas in the literature differ between papers and web
        sources. The three versions below are the most commonly cited
        — revisit against a primary source if a specific value is needed.
        """
        if self.equity.size < 3 or np.any(self.equity <= 0):
            return None
        log_eq = np.log(self.equity)
        t = np.arange(log_eq.size, dtype=np.float64)
        t_mean = t.mean()
        y_mean = log_eq.mean()
        ss_xx = np.sum((t - t_mean) ** 2)
        if ss_xx == 0:
            return None
        slope = np.sum((t - t_mean) * (log_eq - y_mean)) / ss_xx
        residuals = log_eq - (y_mean + slope * (t - t_mean))
        n = log_eq.size
        stderr = np.sqrt(np.sum(residuals ** 2) / (n - 2)) / np.sqrt(ss_xx)
        if stderr == 0:
            return None
        return slope, stderr, n

    def _calculate_k_ratio_1996(self) -> float:
        """
        Kestner 1996 (original):  K = slope / (stderr * sqrt(n))
        Produces larger values for longer series.
        """
        comp = self._k_ratio_components()
        if comp is None:
            return 0.0
        slope, stderr, n = comp
        return float(slope / (stderr * np.sqrt(n)))

    def _calculate_k_ratio_2003(self) -> float:
        """
        Kestner 2003 (revised):  K = slope / (stderr * n)
        Reduces sensitivity to series length vs the 1996 version.
        """
        comp = self._k_ratio_components()
        if comp is None:
            return 0.0
        slope, stderr, n = comp
        return float(slope / (stderr * n))

    def _calculate_k_ratio_2013(self) -> float:
        """
        Kestner 2013 (timeframe-normalized):
            K = (slope / stderr) * sqrt(bars_per_year / n)
        Comparable across timeframes — a daily and hourly backtest of the
        same strategy should yield similar values.
        """
        comp = self._k_ratio_components()
        if comp is None:
            return 0.0
        slope, stderr, n = comp
        bpy = self._bars_per_year()
        return float((slope / stderr) * np.sqrt(bpy / n))

    def _calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Omega ratio relative to an ANNUAL threshold, converted to per-bar.
        Sum of excess gains over threshold divided by |sum of excess losses|.
        """
        r = self._returns()
        if r.size == 0:
            return 0.0
        thresh_per_bar = threshold / self._bars_per_year()
        excess = r - thresh_per_bar
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        if losses == 0:
            return np.inf
        return float(gains / losses)

    def _calculate_jensens_alpha(self, benchmark_returns: np.ndarray,
                                 risk_free: float = 0.0) -> float:
        """
        Annualized Jensen's alpha vs a benchmark per-bar return series.
        risk_free is annual.
        """
        r = self._returns()
        b = np.asarray(benchmark_returns, dtype=np.float64)
        if r.size == 0 or b.size == 0:
            return 0.0
        n = min(r.size, b.size)
        r = r[-n:]
        b = b[-n:]
        bpy = self._bars_per_year()
        rf_per_bar = risk_free / bpy
        var_b = np.var(b)
        if var_b == 0:
            return 0.0
        beta = np.mean((r - r.mean()) * (b - b.mean())) / var_b
        alpha_per_bar = r.mean() - (rf_per_bar + beta * (b.mean() - rf_per_bar))
        return float(alpha_per_bar * bpy)

    # ------------------------------------------------------------------ #
    # Trade-level metrics                                                 #
    # ------------------------------------------------------------------ #
    def _calculate_expectancy(self) -> float:
        """Mean pnl per trade."""
        pnl = self._pnl()
        return float(pnl.mean()) if pnl.size else 0.0

    def _calculate_exposure(self) -> float:
        """
        Fraction of bars spent in the market.

        Current implementation: sum(bars_held across all trades) / n_bars.

        KNOWN LIMITATION: overstates exposure when positions overlap
        (max_positions > 1, hedging, or pyramiding) because overlapping
        bars are double-counted. Proper fix requires the inner loop to
        emit a per-bar "is any position open" boolean — revisit once
        the loop is implemented.
        """
        if self.equity.size == 0:
            return 0.0
        bh = self._bars_held()
        if bh.size == 0:
            return 0.0
        return float(bh.sum() / self.equity.size)

    def _calculate_avg_duration(self) -> float:
        bh = self._bars_held()
        return float(bh.mean()) if bh.size else 0.0

    def _calculate_avg_duration_winning(self) -> float:
        pnl, bh = self._pnl(), self._bars_held()
        mask = pnl > 0
        return float(bh[mask].mean()) if mask.any() else 0.0

    def _calculate_avg_duration_losing(self) -> float:
        pnl, bh = self._pnl(), self._bars_held()
        mask = pnl < 0
        return float(bh[mask].mean()) if mask.any() else 0.0

    def _calculate_max_consecutive_winners(self) -> int:
        return _longest_run_of_true(self._pnl() > 0)

    def _calculate_max_consecutive_losers(self) -> int:
        return _longest_run_of_true(self._pnl() < 0)

    def _calculate_biggest_win(self) -> float:
        pnl = self._pnl()
        return float(pnl.max()) if pnl.size else 0.0

    def _calculate_biggest_loss(self) -> float:
        pnl = self._pnl()
        return float(pnl.min()) if pnl.size else 0.0

    def _calculate_avg_winning_trade(self) -> float:
        pnl = self._pnl()
        wins = pnl[pnl > 0]
        return float(wins.mean()) if wins.size else 0.0

    def _calculate_avg_losing_trade(self) -> float:
        pnl = self._pnl()
        losses = pnl[pnl < 0]
        return float(losses.mean()) if losses.size else 0.0

    def _calculate_winrate(self) -> float:
        pnl = self._pnl()
        return float((pnl > 0).mean()) if pnl.size else 0.0

    def _calculate_profitfactor(self) -> float:
        pnl = self._pnl()
        if pnl.size == 0:
            return 0.0
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        if losses == 0:
            return np.inf
        return float(wins / losses)

    # ------------------------------------------------------------------ #
    # Aggregate + plots + export                                          #
    # ------------------------------------------------------------------ #
    def calculate_metrics(
        self,
        risk_free: float = 0.0,
        omega_threshold: float = 0.0,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute all performance metrics and return as a dict.

        Parameters
        ----------
        risk_free : float
            Annual risk-free rate for Sharpe, Sortino, Jensen's alpha.
        omega_threshold : float
            Annual threshold for Omega ratio.
        benchmark_returns : np.ndarray, optional
            Per-bar benchmark return series for Jensen's alpha. If omitted,
            alpha is returned as nan.
        """
        metrics = {
            # Return / risk
            "sharpe":          self._calculate_sharpe(risk_free),
            "log_sharpe":      self._calculate_log_sharpe(risk_free),
            "sortino":         self._calculate_sortino(risk_free),
            "log_sortino":     self._calculate_log_sortino(risk_free),
            "calmar":          self._calculate_calmar(),
            "cagr":            self._calculate_cagr(),
            "max_drawdown":    self._calculate_max_drawdown(),
            "ulcer_index":     self._calculate_ulcer_index(),
            "k_ratio_1996":    self._calculate_k_ratio_1996(),
            "k_ratio_2003":    self._calculate_k_ratio_2003(),
            "k_ratio_2013":    self._calculate_k_ratio_2013(),
            "omega_ratio":     self._calculate_omega_ratio(omega_threshold),
            "jensens_alpha":   (
                self._calculate_jensens_alpha(benchmark_returns, risk_free)
                if benchmark_returns is not None else float("nan")
            ),
            # Trade-level
            "n_trades":                 int(self.trades.shape[0]),
            "winrate":                  self._calculate_winrate(),
            "profit_factor":            self._calculate_profitfactor(),
            "expectancy":               self._calculate_expectancy(),
            "exposure":                 self._calculate_exposure(),
            "avg_duration":             self._calculate_avg_duration(),
            "avg_duration_winning":     self._calculate_avg_duration_winning(),
            "avg_duration_losing":      self._calculate_avg_duration_losing(),
            "max_consecutive_winners":  self._calculate_max_consecutive_winners(),
            "max_consecutive_losers":   self._calculate_max_consecutive_losers(),
            "biggest_win":              self._calculate_biggest_win(),
            "biggest_loss":             self._calculate_biggest_loss(),
            "avg_winning_trade":        self._calculate_avg_winning_trade(),
            "avg_losing_trade":         self._calculate_avg_losing_trade(),
            # Overall
            "total_return": (
                float(self.equity[-1] / self.equity[0] - 1.0)
                if self.equity.size >= 2 and self.equity[0] > 0 else 0.0
            ),
            "starting_equity": float(self.equity[0]) if self.equity.size else 0.0,
            "final_equity":    float(self.equity[-1]) if self.equity.size else 0.0,
        }
        return metrics

    def trades_to_dataframe(self) -> pd.DataFrame:
        """
        Return the trade log as a pandas DataFrame with labelled columns,
        datetime-typed entry/exit times, and exit_reason as strings.
        Computed pnl is appended as a convenience column.
        """
        t = self.trades
        if t.shape[0] == 0:
            return pd.DataFrame(columns=[
                "direction", "entry_bar", "entry_time", "entry_price",
                "exit_bar", "exit_time", "exit_price", "size",
                "sl", "tp", "ts_dist", "ts_peak",
                "commission", "spread_cost", "slippage_cost", "overnight",
                "mae", "mfe", "exit_reason", "bars_held", "pnl",
            ])
        df = pd.DataFrame({
            "direction":       t[:, F_DIRECTION].astype(np.int8),
            "entry_bar":       t[:, F_ENTRY_BAR].astype(np.int64),
            "entry_time":      pd.to_datetime(t[:, F_ENTRY_TIME].astype(np.int64)),
            "entry_price":     t[:, F_ENTRY_PRICE],
            "exit_bar":        t[:, F_EXIT_BAR].astype(np.int64),
            "exit_time":       pd.to_datetime(t[:, F_EXIT_TIME].astype(np.int64)),
            "exit_price":      t[:, F_EXIT_PRICE],
            "size":            t[:, F_SIZE],
            "sl":              t[:, F_SL],
            "tp":              t[:, F_TP],
            "ts_dist":         t[:, F_TS_DIST],
            "ts_peak":         t[:, F_TS_PEAK],
            "commission":      t[:, F_COMMISSION],
            "spread_cost":     t[:, F_SPREAD_COST],
            "slippage_cost":   t[:, F_SLIPPAGE_COST],
            "overnight":       t[:, F_OVERNIGHT],
            "mae":             t[:, F_MAE],
            "mfe":             t[:, F_MFE],
            "exit_reason":     [
                _EXIT_REASON_NAMES.get(int(c), "unknown")
                for c in t[:, F_EXIT_REASON]
            ],
            "bars_held":       t[:, F_BARS_HELD].astype(np.int64),
            "pnl":             self._pnl(),
        })
        return df

    # ------------------------------------------------------------------ #
    # Plotting                                                            #
    # ------------------------------------------------------------------ #
    def plot_returns(self, ax=None, log: bool = False):
        """
        Plot the equity curve. Returns the matplotlib Axes.
        Pass an existing ``ax`` to compose into a larger figure.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.equity, linewidth=1.2)
        if log:
            ax.set_yscale("log")
        ax.set_title("Equity curve")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_drawdown(self, ax=None):
        """Underwater drawdown curve. Returns the matplotlib Axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        if self.equity.size:
            peak = np.maximum.accumulate(self.equity)
            dd = (self.equity - peak) / peak * 100.0
            ax.fill_between(np.arange(dd.size), dd, 0, color="tab:red", alpha=0.4)
            ax.plot(dd, color="tab:red", linewidth=0.8)
        ax.set_title("Drawdown")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_metrics(self, figsize: tuple = (12, 8)):
        """
        Dashboard grid: equity curve, drawdown, trade P&L distribution,
        cumulative trade P&L. Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        self.plot_returns(ax=axes[0, 0])
        self.plot_drawdown(ax=axes[0, 1])

        pnl = self._pnl()
        ax_hist = axes[1, 0]
        if pnl.size:
            ax_hist.hist(pnl, bins=30, color="tab:blue", alpha=0.7)
            ax_hist.axvline(0, color="k", linewidth=0.6)
        ax_hist.set_title("Trade P&L distribution")
        ax_hist.set_xlabel("P&L")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.3)

        ax_cum = axes[1, 1]
        if pnl.size:
            ax_cum.plot(np.cumsum(pnl), linewidth=1.2)
        ax_cum.set_title("Cumulative trade P&L")
        ax_cum.set_xlabel("Trade #")
        ax_cum.set_ylabel("Cumulative P&L")
        ax_cum.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def summary(self) -> str:
        """Human-readable metrics summary (for ``print(result)``)."""
        m = self.calculate_metrics()
        lines = [
            f"Backtest Result ({self.timeframe})",
            f"  Bars:            {self.equity.size}",
            f"  Trades:          {m['n_trades']}",
            f"  Starting equity: {m['starting_equity']:.2f}",
            f"  Final equity:    {m['final_equity']:.2f}",
            f"  Total return:    {m['total_return']*100:.2f}%",
            f"  CAGR:            {m['cagr']*100:.2f}%",
            f"  Max drawdown:    {m['max_drawdown']*100:.2f}%",
            f"  Sharpe:          {m['sharpe']:.2f}",
            f"  Sortino:         {m['sortino']:.2f}",
            f"  Calmar:          {m['calmar']:.2f}",
            f"  Profit factor:   {m['profit_factor']:.2f}",
            f"  Win rate:        {m['winrate']*100:.2f}%",
            f"  Expectancy:      {m['expectancy']:.2f}",
            f"  Exposure:        {m['exposure']*100:.2f}%",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
