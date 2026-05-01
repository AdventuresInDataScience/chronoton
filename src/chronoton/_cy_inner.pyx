# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""
_cy_inner.pyx — fast-path inner loop and slot helpers (private compiled helper).

Compile with:
    python setup.py build_ext --inplace

Produces `_cy_inner.so` (Linux/Mac) or `_cy_inner.pyd` (Windows). Users
never import this directly; they import the public dispatcher
`cython_backtester`, which routes to this module for the fast path and
to the pure-Python `_inner_loop` in `backtester` for the custom-sizer
fallback.

Semantics must match backtester.py::_inner_loop exactly — see
backtester_docs.md §2–§3. Any divergence is a bug.

IMPORTANT — what this file does NOT handle:
    - position_sizing='custom' (callable). The Python dispatcher falls
      back to the pure-Python `_inner_loop` in backtester.py for that
      case because a Python callable cannot be called from a nogil block
      without re-acquiring the GIL, which defeats the point.

Field layout (matches F_* in the Python module):
    0  DIRECTION        10 TS_DIST
    1  ENTRY_BAR        11 TS_PEAK
    2  ENTRY_TIME       12 COMMISSION
    3  ENTRY_PRICE      13 SPREAD_COST
    4  EXIT_BAR         14 SLIPPAGE_COST
    5  EXIT_TIME        15 OVERNIGHT
    6  EXIT_PRICE       16 MAE
    7  SIZE             17 MFE
    8  SL               18 EXIT_REASON
    9  TP               19 BARS_HELD

Exit reason codes (must match Python):
    0 SIGNAL, 1 SL, 2 TP, 3 TS, 4 LIQUIDATION, 5 END_OF_DATA

Sizing method codes (must match Python; note: custom=3 is NOT handled here):
    0 PERCENT_EQUITY, 1 VALUE, 2 PRECOMPUTED
"""

import numpy as np
cimport cython
from libc.math cimport isnan, NAN, fabs


# --- Field index constants --------------------------------------------------
# Using `cdef enum` so these become C ints at compile time. The values are
# duplicated from the Python module; tests verify they stay in sync.
cdef enum:
    F_DIRECTION     = 0
    F_ENTRY_BAR     = 1
    F_ENTRY_TIME    = 2
    F_ENTRY_PRICE   = 3
    F_EXIT_BAR      = 4
    F_EXIT_TIME     = 5
    F_EXIT_PRICE    = 6
    F_SIZE          = 7
    F_SL            = 8
    F_TP            = 9
    F_TS_DIST       = 10
    F_TS_PEAK       = 11
    F_COMMISSION    = 12
    F_SPREAD_COST   = 13
    F_SLIPPAGE_COST = 14
    F_OVERNIGHT     = 15
    F_MAE           = 16
    F_MFE           = 17
    F_EXIT_REASON   = 18
    F_BARS_HELD     = 19
    N_FIELDS        = 20

cdef enum:
    EXIT_SIGNAL       = 0
    EXIT_SL           = 1
    EXIT_TP           = 2
    EXIT_TS           = 3
    EXIT_LIQUIDATION  = 4
    EXIT_END_OF_DATA  = 5

cdef enum:
    SIZING_PERCENT_EQUITY  = 0
    SIZING_VALUE           = 1
    SIZING_PRECOMPUTED     = 2
    # SIZING_CUSTOM = 3 is handled in Python fallback
    SIZING_PERCENT_AT_RISK = 4


# ---------------------------------------------------------------------------
# Slot helpers — all `nogil`, pure C, no Python object interaction.
# ---------------------------------------------------------------------------
cdef inline double _idx(const double[:] arr, Py_ssize_t i) noexcept nogil:
    """Read arr[i], or arr[0] when arr is a size-1 scalar sentinel."""
    if arr.shape[0] == 1:
        return arr[0]
    return arr[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _find_free_slot(const unsigned char[:] slot_active,
                                       Py_ssize_t n_slots) nogil:
    """Index of first inactive slot, or -1 if full."""
    cdef Py_ssize_t k
    for k in range(n_slots):
        if slot_active[k] == 0:
            return k
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _enter_position(
    Py_ssize_t slot_idx,
    double direction,
    Py_ssize_t bar_idx,
    double entry_time_ns,
    double entry_price,
    double size,
    double sl_price,
    double tp_price,
    double ts_dist,
    double commission_cost,
    double spread_cost,
    double slippage_cost,
    double[:, :] open_positions,
    unsigned char[:] slot_active,
) noexcept nogil:
    """Populate open_positions[slot_idx] and flip slot_active on."""
    open_positions[slot_idx, F_DIRECTION]     = direction
    open_positions[slot_idx, F_ENTRY_BAR]     = <double>bar_idx
    open_positions[slot_idx, F_ENTRY_TIME]    = entry_time_ns
    open_positions[slot_idx, F_ENTRY_PRICE]   = entry_price
    open_positions[slot_idx, F_EXIT_BAR]      = NAN
    open_positions[slot_idx, F_EXIT_TIME]     = NAN
    open_positions[slot_idx, F_EXIT_PRICE]    = NAN
    open_positions[slot_idx, F_SIZE]          = size
    open_positions[slot_idx, F_SL]            = sl_price
    open_positions[slot_idx, F_TP]            = tp_price
    open_positions[slot_idx, F_TS_DIST]       = ts_dist
    open_positions[slot_idx, F_TS_PEAK]       = entry_price
    open_positions[slot_idx, F_COMMISSION]    = commission_cost
    open_positions[slot_idx, F_SPREAD_COST]   = spread_cost
    open_positions[slot_idx, F_SLIPPAGE_COST] = slippage_cost
    open_positions[slot_idx, F_OVERNIGHT]     = 0.0
    open_positions[slot_idx, F_MAE]           = 0.0
    open_positions[slot_idx, F_MFE]           = 0.0
    open_positions[slot_idx, F_EXIT_REASON]   = NAN
    open_positions[slot_idx, F_BARS_HELD]     = 0.0
    slot_active[slot_idx] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _exit_position(
    Py_ssize_t slot_idx,
    Py_ssize_t bar_idx,
    double exit_time_ns,
    double exit_price,
    double exit_reason,
    double exit_commission,
    double exit_spread,
    double exit_slippage,
    double[:, :] open_positions,
    unsigned char[:] slot_active,
    double[:, :] closed_trades,
    Py_ssize_t n_closed,
    Py_ssize_t closed_capacity,
    int[:] overflow_flag,
) nogil:
    """
    Finalise exit fields, copy row to closed_trades[n_closed], clear slot.
    Returns n_closed + 1 on success, or -1 on overflow (with overflow_flag
    set to 1). The overflow_flag is checked by the caller after returning
    to the GIL to raise a clean RuntimeError.
    """
    cdef Py_ssize_t j
    cdef double entry_bar_f

    if n_closed >= closed_capacity:
        overflow_flag[0] = 1
        return -1

    open_positions[slot_idx, F_EXIT_BAR]       = <double>bar_idx
    open_positions[slot_idx, F_EXIT_TIME]      = exit_time_ns
    open_positions[slot_idx, F_EXIT_PRICE]     = exit_price
    open_positions[slot_idx, F_EXIT_REASON]    = exit_reason
    open_positions[slot_idx, F_COMMISSION]    += exit_commission
    open_positions[slot_idx, F_SPREAD_COST]   += exit_spread
    open_positions[slot_idx, F_SLIPPAGE_COST] += exit_slippage
    entry_bar_f = open_positions[slot_idx, F_ENTRY_BAR]
    open_positions[slot_idx, F_BARS_HELD] = <double>(bar_idx - <Py_ssize_t>entry_bar_f)

    # Copy row → closed_trades[n_closed]
    for j in range(N_FIELDS):
        closed_trades[n_closed, j] = open_positions[slot_idx, j]

    slot_active[slot_idx] = 0
    return n_closed + 1


# ---------------------------------------------------------------------------
# Main fast-path loop.
# Returns the final n_closed count as a Python int. `cash`, `equity`,
# `closed_trades` are modified in place by the caller.
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def inner_loop_fast(
    const double[:] o,
    const double[:] h,
    const double[:] l,
    const double[:] c,
    const double[:] v,
    const double[:] date_ns,               # datetime64[ns] reinterpreted as float64
    const unsigned char[:] long_entries_shifted,
    const unsigned char[:] long_exits_shifted,
    const unsigned char[:] short_entries_shifted,
    const unsigned char[:] short_exits_shifted,
    double starting_balance,
    int sizing_method_code,
    double sizing_static,
    const double[:] sizing_array,          # empty (size 0) when unused
    const double[:] sl_arr,
    const double[:] tp_arr,
    const double[:] ts_arr,
    double leverage,
    double commission,
    const double[:] spread_arr,
    const double[:] slippage_arr,
    const double[:] long_fee_vec,
    const double[:] short_fee_vec,
    bint hedging,
    double[:] cash_out,              # length n, pre-allocated
    double[:] equity_out,            # length n, pre-allocated
    double[:, :] open_positions,     # (n_slots, N_FIELDS)
    unsigned char[:] slot_active,    # (n_slots,)
    double[:, :] closed_trades,      # (n, N_FIELDS)
):
    """
    Fast-path event-driven simulation. Semantics identical to the
    Python reference in backtester.py. Does NOT handle custom sizer.
    Overflow of closed_trades pre-allocation is reported via return
    (n_closed == -1) so caller can raise.
    """
    cdef Py_ssize_t n = o.shape[0]
    cdef Py_ssize_t n_slots = open_positions.shape[0]
    cdef Py_ssize_t closed_capacity = closed_trades.shape[0]
    cdef Py_ssize_t n_closed = 0
    cdef int[:] overflow_flag = np.zeros(1, dtype=np.intc)

    cdef double current_cash = starting_balance

    cdef Py_ssize_t i, k, kk, slot_idx
    cdef double price_o, price_h, price_l, price_c
    cdef double t_ns
    cdef double spread_i, slippage_i, sl_bar, tp_bar, ts_bar
    cdef double direction, entry_px, size
    cdef double ts_dist, sl_px, tp_px, ts_trigger_px
    cdef double worst_px, best_px, worst_pnl, best_pnl
    cdef int exit_reason
    cdef double exit_px, exit_px_net
    cdef double exit_spread, exit_slippage, exit_commission
    cdef double proceeds, notional, charge
    cdef double d, u
    cdef double sl_dist, sl_price, tp_price, ts_dist_val
    cdef double entry_px_net, entry_spread, entry_slippage, entry_commission, margin
    cdef double equity_now, unrealized, margin_held
    cdef double total_loss_budget, total_bad, share, realised_pnl
    cdef bint want_long_entry, want_short_entry
    cdef bint want_long_exit, want_short_exit
    cdef int desired_dir
    cdef int any_active

    # Per-liquidation scratch: unrealised per slot.
    # Allocated here so it can be reused without touching Python in the loop.
    cdef double[:] unrealized_by_slot = np.zeros(n_slots, dtype=np.float64)

    cdef bint liquidated = 0

    # Precompute scalar-sentinel flags and cached values once before the loop.
    # Inside the loop we do a register comparison (bint) rather than chasing
    # the memoryview shape pointer 5 × n times.
    cdef bint _spread_scalar   = spread_arr.shape[0]   == 1
    cdef bint _slippage_scalar = slippage_arr.shape[0] == 1
    cdef bint _sl_scalar       = sl_arr.shape[0]       == 1
    cdef bint _tp_scalar       = tp_arr.shape[0]       == 1
    cdef bint _ts_scalar       = ts_arr.shape[0]       == 1
    cdef double _spread_c   = spread_arr[0]   if _spread_scalar   else 0.0
    cdef double _slippage_c = slippage_arr[0] if _slippage_scalar else 0.0
    cdef double _sl_c       = sl_arr[0]       if _sl_scalar       else 0.0
    cdef double _tp_c       = tp_arr[0]       if _tp_scalar       else 0.0
    cdef double _ts_c       = ts_arr[0]       if _ts_scalar       else 0.0

    with nogil:
        for i in range(n):
            price_o = o[i]
            price_h = h[i]
            price_l = l[i]
            price_c = c[i]
            t_ns = date_ns[i]
            spread_i   = _spread_c   if _spread_scalar   else spread_arr[i]
            slippage_i = _slippage_c if _slippage_scalar else slippage_arr[i]
            sl_bar     = _sl_c       if _sl_scalar       else sl_arr[i]
            tp_bar     = _tp_c       if _tp_scalar       else tp_arr[i]
            ts_bar     = _ts_c       if _ts_scalar       else ts_arr[i]

            # (1) Overnight financing ------------------------------------
            if long_fee_vec[i] != 0.0 or short_fee_vec[i] != 0.0:
                for k in range(n_slots):
                    if slot_active[k] == 0:
                        continue
                    notional = (open_positions[k, F_SIZE]
                                * open_positions[k, F_ENTRY_PRICE])
                    if open_positions[k, F_DIRECTION] > 0:
                        charge = long_fee_vec[i] * notional
                    else:
                        charge = short_fee_vec[i] * notional
                    open_positions[k, F_OVERNIGHT] += charge
                    current_cash -= charge

            # (2) TS peaks, MAE/MFE, SL/TP/TS checks ---------------------
            for k in range(n_slots):
                if slot_active[k] == 0:
                    continue
                direction = open_positions[k, F_DIRECTION]
                entry_px = open_positions[k, F_ENTRY_PRICE]
                size = open_positions[k, F_SIZE]

                # Update trailing-stop peak
                ts_dist = open_positions[k, F_TS_DIST]
                if not isnan(ts_dist):
                    if direction > 0 and price_h > open_positions[k, F_TS_PEAK]:
                        open_positions[k, F_TS_PEAK] = price_h
                    elif direction < 0 and price_l < open_positions[k, F_TS_PEAK]:
                        open_positions[k, F_TS_PEAK] = price_l

                # MAE/MFE
                if direction > 0:
                    worst_px = price_l
                    best_px = price_h
                else:
                    worst_px = price_h
                    best_px = price_l
                worst_pnl = direction * (worst_px - entry_px) * size
                best_pnl = direction * (best_px - entry_px) * size
                if worst_pnl < open_positions[k, F_MAE]:
                    open_positions[k, F_MAE] = worst_pnl
                if best_pnl > open_positions[k, F_MFE]:
                    open_positions[k, F_MFE] = best_pnl

                # SL / TP / TS triggers — SL has priority on tie
                sl_px = open_positions[k, F_SL]
                tp_px = open_positions[k, F_TP]
                ts_trigger_px = NAN
                if not isnan(ts_dist):
                    if direction > 0:
                        ts_trigger_px = open_positions[k, F_TS_PEAK] - ts_dist
                    else:
                        ts_trigger_px = open_positions[k, F_TS_PEAK] + ts_dist

                exit_reason = -1
                exit_px = NAN
                if direction > 0:
                    if (not isnan(sl_px)) and price_l <= sl_px:
                        exit_reason = EXIT_SL
                        exit_px = sl_px
                    elif (not isnan(tp_px)) and price_h >= tp_px:
                        exit_reason = EXIT_TP
                        exit_px = tp_px
                    elif (not isnan(ts_trigger_px)) and price_l <= ts_trigger_px:
                        exit_reason = EXIT_TS
                        exit_px = ts_trigger_px
                else:
                    if (not isnan(sl_px)) and price_h >= sl_px:
                        exit_reason = EXIT_SL
                        exit_px = sl_px
                    elif (not isnan(tp_px)) and price_l <= tp_px:
                        exit_reason = EXIT_TP
                        exit_px = tp_px
                    elif (not isnan(ts_trigger_px)) and price_h >= ts_trigger_px:
                        exit_reason = EXIT_TS
                        exit_px = ts_trigger_px

                if exit_reason != -1:
                    exit_spread = spread_i * size
                    exit_slippage = slippage_i * size
                    exit_px_net = exit_px - direction * (spread_i + slippage_i)
                    exit_commission = commission * fabs(exit_px_net * size)
                    proceeds = direction * (exit_px_net - entry_px) * size
                    current_cash += (size * entry_px / leverage) + proceeds
                    current_cash -= exit_commission
                    n_closed = _exit_position(
                        k, i, t_ns, exit_px_net, <double>exit_reason,
                        exit_commission, exit_spread, exit_slippage,
                        open_positions, slot_active, closed_trades,
                        n_closed, closed_capacity, overflow_flag,
                    )
                    if n_closed < 0:
                        break  # overflow; return to caller to raise

            if n_closed < 0:
                break

            # (3) Shifted exit signals ----------------------------------
            want_long_exit = long_exits_shifted[i] != 0
            want_short_exit = short_exits_shifted[i] != 0
            if want_long_exit or want_short_exit:
                for k in range(n_slots):
                    if slot_active[k] == 0:
                        continue
                    direction = open_positions[k, F_DIRECTION]
                    if ((direction > 0 and want_long_exit) or
                        (direction < 0 and want_short_exit)):
                        size = open_positions[k, F_SIZE]
                        entry_px = open_positions[k, F_ENTRY_PRICE]
                        exit_px_net = price_o - direction * (spread_i + slippage_i)
                        exit_commission = commission * fabs(exit_px_net * size)
                        proceeds = direction * (exit_px_net - entry_px) * size
                        current_cash += (size * entry_px / leverage) + proceeds
                        current_cash -= exit_commission
                        n_closed = _exit_position(
                            k, i, t_ns, exit_px_net, <double>EXIT_SIGNAL,
                            exit_commission,
                            spread_i * size,
                            slippage_i * size,
                            open_positions, slot_active, closed_trades,
                            n_closed, closed_capacity, overflow_flag,
                        )
                        if n_closed < 0:
                            break
                if n_closed < 0:
                    break

            # (4) Shifted entry signals ---------------------------------
            want_long_entry = long_entries_shifted[i] != 0
            want_short_entry = short_entries_shifted[i] != 0

            # Non-hedging: flatten opposite-direction positions first
            if (not hedging) and (want_long_entry or want_short_entry):
                desired_dir = 1 if want_long_entry else -1
                for k in range(n_slots):
                    if slot_active[k] == 0:
                        continue
                    direction = open_positions[k, F_DIRECTION]
                    if <int>direction != desired_dir:
                        size = open_positions[k, F_SIZE]
                        entry_px = open_positions[k, F_ENTRY_PRICE]
                        exit_px_net = price_o - direction * (spread_i + slippage_i)
                        exit_commission = commission * fabs(exit_px_net * size)
                        proceeds = direction * (exit_px_net - entry_px) * size
                        current_cash += (size * entry_px / leverage) + proceeds
                        current_cash -= exit_commission
                        n_closed = _exit_position(
                            k, i, t_ns, exit_px_net, <double>EXIT_SIGNAL,
                            exit_commission,
                            spread_i * size,
                            slippage_i * size,
                            open_positions, slot_active, closed_trades,
                            n_closed, closed_capacity, overflow_flag,
                        )
                        if n_closed < 0:
                            break
                if n_closed < 0:
                    break

            # Open new positions. Unrolled for long and short separately
            # rather than iterating `for desired_dir in (1, -1):` because
            # tuple-iteration inside `nogil` is fragile across Cython versions.
            # The two blocks are identical except for `desired_dir` and which
            # `want_*_entry` flag is consulted — kept as a helper-free inline
            # to stay fully in C.
            if want_long_entry:
                desired_dir = 1
                slot_idx = _find_free_slot(slot_active, n_slots)
                if slot_idx != -1:
                    entry_px_net = price_o + desired_dir * (spread_i + slippage_i)

                    if sizing_method_code == SIZING_PERCENT_EQUITY:
                        equity_now = current_cash
                        for kk in range(n_slots):
                            if slot_active[kk] != 0:
                                d = open_positions[kk, F_DIRECTION]
                                equity_now += (open_positions[kk, F_SIZE]
                                               * open_positions[kk, F_ENTRY_PRICE] / leverage)
                                equity_now += (d * (price_c - open_positions[kk, F_ENTRY_PRICE])
                                                 * open_positions[kk, F_SIZE])
                        size = (sizing_static * equity_now * leverage) / entry_px_net
                    elif sizing_method_code == SIZING_VALUE:
                        size = (sizing_static * leverage) / entry_px_net
                    elif sizing_method_code == SIZING_PERCENT_AT_RISK:
                        sl_dist = sl_bar
                        if isnan(sl_dist) or sl_dist <= 0.0:
                            size = 0.0
                        else:
                            equity_now = current_cash
                            for kk in range(n_slots):
                                if slot_active[kk] != 0:
                                    d = open_positions[kk, F_DIRECTION]
                                    equity_now += (open_positions[kk, F_SIZE]
                                                   * open_positions[kk, F_ENTRY_PRICE] / leverage)
                                    equity_now += (d * (price_c - open_positions[kk, F_ENTRY_PRICE])
                                                     * open_positions[kk, F_SIZE])
                            size = (sizing_static * equity_now) / sl_dist
                    else:  # SIZING_PRECOMPUTED
                        size = sizing_array[i]

                    if size > 0.0 and not isnan(size):
                        entry_spread = spread_i * size
                        entry_slippage = slippage_i * size
                        entry_commission = commission * fabs(entry_px_net * size)
                        margin = size * entry_px_net / leverage

                        if current_cash >= margin + entry_commission:
                            current_cash -= margin + entry_commission

                            sl_dist = sl_bar
                            if isnan(sl_dist):
                                sl_price = NAN
                            else:
                                sl_price = entry_px_net - desired_dir * sl_dist
                            if isnan(tp_bar):
                                tp_price = NAN
                            else:
                                tp_price = entry_px_net + desired_dir * tp_bar
                            if isnan(ts_bar):
                                ts_dist_val = NAN
                            else:
                                ts_dist_val = ts_bar

                            _enter_position(
                                slot_idx, <double>desired_dir, i, t_ns,
                                entry_px_net, size,
                                sl_price, tp_price, ts_dist_val,
                                entry_commission, entry_spread, entry_slippage,
                                open_positions, slot_active,
                            )

            if want_short_entry:
                desired_dir = -1
                slot_idx = _find_free_slot(slot_active, n_slots)
                if slot_idx != -1:
                    entry_px_net = price_o + desired_dir * (spread_i + slippage_i)

                    if sizing_method_code == SIZING_PERCENT_EQUITY:
                        equity_now = current_cash
                        for kk in range(n_slots):
                            if slot_active[kk] != 0:
                                d = open_positions[kk, F_DIRECTION]
                                equity_now += (open_positions[kk, F_SIZE]
                                               * open_positions[kk, F_ENTRY_PRICE] / leverage)
                                equity_now += (d * (price_c - open_positions[kk, F_ENTRY_PRICE])
                                                 * open_positions[kk, F_SIZE])
                        size = (sizing_static * equity_now * leverage) / entry_px_net
                    elif sizing_method_code == SIZING_VALUE:
                        size = (sizing_static * leverage) / entry_px_net
                    elif sizing_method_code == SIZING_PERCENT_AT_RISK:
                        sl_dist = sl_bar
                        if isnan(sl_dist) or sl_dist <= 0.0:
                            size = 0.0
                        else:
                            equity_now = current_cash
                            for kk in range(n_slots):
                                if slot_active[kk] != 0:
                                    d = open_positions[kk, F_DIRECTION]
                                    equity_now += (open_positions[kk, F_SIZE]
                                                   * open_positions[kk, F_ENTRY_PRICE] / leverage)
                                    equity_now += (d * (price_c - open_positions[kk, F_ENTRY_PRICE])
                                                     * open_positions[kk, F_SIZE])
                            size = (sizing_static * equity_now) / sl_dist
                    else:
                        size = sizing_array[i]

                    if size > 0.0 and not isnan(size):
                        entry_spread = spread_i * size
                        entry_slippage = slippage_i * size
                        entry_commission = commission * fabs(entry_px_net * size)
                        margin = size * entry_px_net / leverage

                        if current_cash >= margin + entry_commission:
                            current_cash -= margin + entry_commission

                            sl_dist = sl_bar
                            if isnan(sl_dist):
                                sl_price = NAN
                            else:
                                sl_price = entry_px_net - desired_dir * sl_dist
                            if isnan(tp_bar):
                                tp_price = NAN
                            else:
                                tp_price = entry_px_net + desired_dir * tp_bar
                            if isnan(ts_bar):
                                ts_dist_val = NAN
                            else:
                                ts_dist_val = ts_bar

                            _enter_position(
                                slot_idx, <double>desired_dir, i, t_ns,
                                entry_px_net, size,
                                sl_price, tp_price, ts_dist_val,
                                entry_commission, entry_spread, entry_slippage,
                                open_positions, slot_active,
                            )

            # (5) Mark-to-market ---------------------------------------
            unrealized = 0.0
            margin_held = 0.0
            for k in range(n_slots):
                if slot_active[k] == 0:
                    continue
                d = open_positions[k, F_DIRECTION]
                unrealized += (d * (price_c - open_positions[k, F_ENTRY_PRICE])
                                 * open_positions[k, F_SIZE])
                margin_held += open_positions[k, F_SIZE] * open_positions[k, F_ENTRY_PRICE] / leverage
                open_positions[k, F_BARS_HELD] = <double>(
                    i - <Py_ssize_t>open_positions[k, F_ENTRY_BAR]
                )
            equity_out[i] = current_cash + margin_held + unrealized
            cash_out[i] = current_cash

            # (6) Liquidation ------------------------------------------
            if equity_out[i] <= 0.0:
                total_loss_budget = current_cash + margin_held
                total_bad = 0.0
                for k in range(n_slots):
                    if slot_active[k] == 0:
                        unrealized_by_slot[k] = 0.0
                        continue
                    d = open_positions[k, F_DIRECTION]
                    u = (d * (price_c - open_positions[k, F_ENTRY_PRICE])
                           * open_positions[k, F_SIZE])
                    unrealized_by_slot[k] = u
                    if u < 0.0:
                        total_bad += -u

                for k in range(n_slots):
                    if slot_active[k] == 0:
                        continue
                    direction = open_positions[k, F_DIRECTION]
                    size = open_positions[k, F_SIZE]
                    entry_px = open_positions[k, F_ENTRY_PRICE]
                    u = unrealized_by_slot[k]

                    if u >= 0.0 or total_bad == 0.0:
                        realised_pnl = u
                        exit_px_net = price_c
                    else:
                        share = (-u) / total_bad
                        realised_pnl = -share * total_loss_budget
                        exit_px_net = entry_px + realised_pnl / (direction * size)

                    current_cash += (size * entry_px / leverage) + realised_pnl
                    n_closed = _exit_position(
                        k, i, t_ns, exit_px_net, <double>EXIT_LIQUIDATION,
                        0.0, 0.0, 0.0,
                        open_positions, slot_active, closed_trades,
                        n_closed, closed_capacity, overflow_flag,
                    )
                    if n_closed < 0:
                        break

                if n_closed < 0:
                    break

                equity_out[i] = current_cash
                cash_out[i] = current_cash
                # Fill remaining bars with terminal equity
                if i + 1 < n:
                    for k in range(i + 1, n):
                        cash_out[k] = current_cash
                        equity_out[k] = current_cash
                liquidated = 1
                break

        # --- end-of-data close-out (only if not liquidated) -------------
        # Any slot still active after the main loop ends normally is closed
        # at the final bar's close, with no costs. Skipped if we broke out
        # early via liquidation.
        if not liquidated and n_closed >= 0:
            any_active = 0
            for k in range(n_slots):
                if slot_active[k] != 0:
                    any_active = 1
                    break

            if any_active == 1:
                for k in range(n_slots):
                    if slot_active[k] == 0:
                        continue
                    direction = open_positions[k, F_DIRECTION]
                    size = open_positions[k, F_SIZE]
                    entry_px = open_positions[k, F_ENTRY_PRICE]
                    realised_pnl = direction * (c[n - 1] - entry_px) * size
                    current_cash += (size * entry_px / leverage) + realised_pnl
                    n_closed = _exit_position(
                        k, n - 1, date_ns[n - 1], c[n - 1],
                        <double>EXIT_END_OF_DATA,
                        0.0, 0.0, 0.0,
                        open_positions, slot_active, closed_trades,
                        n_closed, closed_capacity, overflow_flag,
                    )
                    if n_closed < 0:
                        break
                if n_closed >= 0:
                    equity_out[n - 1] = current_cash
                    cash_out[n - 1] = current_cash

    # Return overflow status to Python
    if overflow_flag[0] == 1:
        return -1
    return n_closed
