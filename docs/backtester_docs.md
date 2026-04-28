# Backtester — Design & Reference Document

This document accompanies `backtester.py`. It is not a README for end users —
it is an internal design document capturing methodology, the assumptions
baked into the implementation, known limitations, planned future work, a
function-by-function reference, and a worked example. A proper README with
install instructions, dependency list, and a test matrix will be written
once the module is split up, packaged, and stabilised.

---

## Table of contents

1. Purpose & scope
2. Methodology
3. Execution semantics (the fixed rules)
4. Assumptions
5. Known limitations & gotchas
6. Planned upgrades
7. File layout
8. Architecture map (how functions connect)
9. Public API reference — `run_single_backtest`
10. `Result` class reference
11. Private helpers reference
12. The 20-column trade-record layout
13. Numeric conventions
14. Worked example — SPY daily with yfinance
15. Testing status

---

## 1. Purpose & scope

A single-file, event-driven backtester for vectorised strategies expressed
as boolean entry/exit signal arrays on OHLCV data. Focus is on:

- **Correctness first, speed second.** Pure-numpy loop. Empirically ~10×
  faster than the equivalent `@njit` version on 10 years of daily data
  because numba's compile overhead does not amortise at that scale.
- **Long, short, hedged, and pyramided positions.**
- **Realistic costs** — commission, spread, slippage, and per-bar overnight
  financing (with triple-charge weekend convention).
- **Four position-sizing modes** — percent of equity, fixed notional,
  precomputed per-bar sizes, or a user-supplied callable.
- **A rich trade log** — every closed trade stores entry/exit, costs, MAE,
  MFE, and exit reason, ready for export to pandas.
- **A metrics suite** — 22 performance metrics covering returns, risk,
  and trade-level statistics.
- **Ports cleanly to numba later.** All hot-path data structures are
  pre-allocated float64 arrays with integer column indices. The callable
  sizing path is the only piece that does not port as-is; a `@cfunc`
  contract is the intended escape hatch.

Out of scope for this iteration: multi-asset portfolios, options, futures
roll logic, order book microstructure, walk-forward harness, parameter
optimisation.

---

## 2. Methodology

### 2.1 Signal-driven, bar-by-bar simulation

The user supplies boolean arrays that are True on bars where a signal
fires. The backtester walks the OHLCV arrays in a single pass, maintaining
open-position state across iterations and emitting a trade log and per-bar
cash/equity series.

### 2.2 Signal shifting

Entry and exit signals fire at the **close** of the bar on which they are
True, and are shifted by +1 internally before entering the loop. Fills
occur at the **open** of the next bar. This prevents lookahead: at close
time on bar *i*, the signal must be computed from information available by
close[i], and the trade executes at the next observable price, open[i+1].

Both entry and exit arrays are shifted identically by the `_shift` helper
in `run_single_backtest`. Users pass unshifted arrays aligned to OHLCV.

### 2.3 Within-bar ordering

For each bar *i*, the loop runs steps in this fixed order. The order is
intentional — reordering changes backtest results.

1. **Overnight financing.** Apply the per-bar financing rate to every open
   position. Zero on non-rollover bars, so effectively a no-op except at
   the bar holding the daily rollover moment.
2. **Update trailing-stop peaks; update MAE / MFE; check SL / TP / TS.**
   Each open position has its peak/trough refreshed against the bar's
   H/L, then its stop/target levels tested against H/L. If a level is
   breached, the position closes at that level.
3. **Shifted exit signals.** Close any positions whose direction matches
   the firing signal, at open[i].
4. **Shifted entry signals.** Open new positions at open[i]. If
   `max_positions>1` and `hedging=False`, an opposite-direction entry
   first flattens all existing positions.
5. **Mark-to-market.** Compute equity = cash + margin held + unrealised
   P&L of all open positions, valued at close[i]. Write `cash[i]` and
   `equity[i]`.
6. **Liquidation check.** If `equity[i] <= 0`, flatten all positions at
   close[i], fill the remaining cash/equity bars with that value, and
   exit the loop.

### 2.4 SL / TP / TS priority

When both a stop-loss and take-profit would hit on the same bar, the
**stop-loss wins** (conservative). This reflects the reality that intraday
price-path information is not available from OHLC alone, so the worst
plausible fill is assumed. Trailing stops are checked after SL and TP, on
the assumption they only fire when neither hard level is hit.

### 2.5 Fill-price mechanics

On entry and exit, the fill price is:

```
fill = open_or_exit_price + direction * (spread_bar + slippage_bar)
```

Spread and slippage act **against** the trader — a long pays more on
entry, receives less on exit; a short is the mirror. Both are supplied
by the user in **pips** and converted to price units internally via
``pip_equals`` (see §2.11).

Commission is applied as a fraction of notional (`commission * |price * size|`)
on both entry and exit, debited from cash in addition to the fill.

### 2.6 Margin and leverage

For `position_sizing="percent_equity"` and `"value"`, the notional size
is scaled by the `leverage` multiplier:

```
size = percent * equity * leverage / fill_price
size = value * leverage / fill_price
```

On entry, `size * entry_price` is reserved as margin (debited from cash).
On exit, the margin returns to cash alongside the realised P&L and
commission is debited once more. No margin-call mechanic other than the
equity-goes-negative liquidation check in step 6.

### 2.7 Overnight financing

Two annual rates are provided as a tuple `(annual_base, annual_borrow)`.

- `annual_base` is the common financing rate (funding / risk-free),
  applied to long and short identically.
- `annual_borrow` is the direction-dependent component. It is conventionally
  a debit for shorts and a credit (or reduced debit) for longs. The sign
  convention applied in `_process_overnight_charge` is:

```
long_daily  = (annual_base - annual_borrow) / denominator
short_daily = (annual_base + annual_borrow) / denominator
```

Users whose venue uses the opposite sign convention can simply flip the
sign of `annual_borrow` at the call site.

The preprocessor returns two per-bar vectors (`long_fee_vec`,
`short_fee_vec`) that are zero on every bar *except* bars containing a
daily rollover. On those bars the value is `daily_rate` with a 3×
multiplier applied on the triple-charge weekday (default Wednesday, to
cover Saturday and Sunday's financing).

Rollover timing defaults to 22:00 UTC (standard IB / FX / CFD). A rollover
at time *R* is attributed to the first bar *i* with `date[i] >= R`. This
means:

- For **end-of-day** daily timestamps (23:59-ish), Wednesday's 22:00
  rollover falls inside Wednesday's bar → triple on **Wed bar**.
- For **midnight** daily timestamps, Wednesday's 22:00 rollover falls
  between Wed midnight and Thu midnight → triple on **Thu bar**. If the
  Wed bar is preferred, set `triple_charge_weekday="Thursday"` or shift
  timestamps by one bar upstream.

### 2.8 Position-sizing modes

Five explicit modes selected by a string, each with its own argument:

| `position_sizing`   | Required arg                | Semantics                                                                                                                                                                                                                |
| ------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `"percent_equity"`  | `position_percent_equity`   | `size = fraction * current_equity * leverage / fill_price`                                                                                                                                                               |
| `"value"`           | `position_value`            | `size = value * leverage / fill_price`                                                                                                                                                                                   |
| `"precomputed"`     | `position_sizes`            | Per-bar ndarray/Series of sizes. The value at the entry bar is used as-is.                                                                                                                                               |
| `"custom"`          | `position_sizing_fn`        | Callable receiving full state, returns a float size.                                                                                                                                                                     |
| `"percent_at_risk"` | `position_percent_at_risk`  | `size = fraction * current_equity / sl_distance`. Risk-based — if the stop is hit, exactly `fraction × equity` is lost. Requires a non-NaN SL. **Leverage is not applied** (risk is the point). See §2.8.1 for details.  |

Negative sizes are rejected (direction comes from the signal, not the
magnitude). Non-positive sizes returned by a custom callable cause the
entry to be skipped silently.

#### 2.8.1 percent_at_risk (risk-based sizing)

Common in spread-betting and retail FX. The trader picks a fixed fraction
of equity they're willing to lose per trade (typically 0.5–2%), and the
position size is back-calculated so that the SL, if hit, realises exactly
that loss:

```
size = (percent_at_risk × current_equity) / sl_distance
```

**Upfront leverage sanity check.** `run_single_backtest` rejects the
configuration immediately if the tightest non-NaN SL combined with the
risk fraction would require more leverage than is configured. The check
uses the median close as the reference price:

```
required_leverage = percent_at_risk × median(c) / min(sl_arr)
if required_leverage > leverage:   raise ValueError
```

This catches the common "1% risk on a 2-pip stop needs 250× leverage"
mistake at the API boundary rather than silently skipping trades or
emitting garbage. The error message includes the offending SL in pip
units for a user-friendly hint.

**Error conditions** (all raised before the loop starts):

- `SL=None` → "SL is required as the risk denominator."
- `SL=array` but all entries are NaN → "At least one non-NaN SL required."
- Leverage too small → descriptive error with suggested fixes.
- `position_percent_at_risk` not in (0, 1) → rejected.

**Behaviour during the loop.** If a specific bar's SL is NaN (partial
NaN in an array-valued SL), that entry is silently skipped — the
configuration is valid overall, just this bar has nothing to size
against.

**Future upgrade — `when_max_leverage_breached` kwarg.** The current
implementation raises once at configuration time. A richer version
would pick between: (a) raise (current), (b) cap size at max-leverage
and continue, (c) skip the trade and log to diagnostics. Most users
should stay well clear of their leverage ceiling regardless, so the
extra complexity has not been added.

### 2.9 The custom sizer callable

`position_sizing_fn` receives everything the loop has available:

```
sizing_fn(
    bar_idx,           # int
    entry_time_ns,     # int64 (datetime64[ns] as int)
    direction,         # +1 or -1
    current_cash,      # float
    open_positions,    # (n_slots, N_FIELDS) float64 array
    slot_active,       # (n_slots,) bool array
    closed_trades,     # (n_bars, N_FIELDS) float64, only first n_closed valid
    n_closed,          # int
    date, o, h, l, c, v,  # full OHLCV arrays
) -> float
```

The user indexes what they need. Typical uses: Kelly (needs trade history),
vol-targeting (needs a lookback slice of `c`), regime-conditional sizing.

### 2.10 Mark-to-market

Equity is not a derived quantity computed after the loop; it is written at
each bar alongside cash. The per-bar equity series drives Sharpe, Sortino,
drawdown, Ulcer, CAGR, Calmar, Omega, K-ratios, and Jensen's alpha.

### 2.11 Pip-unit inputs

Five cost-distance inputs are supplied by the user in **pips**, not price
units: ``SL``, ``TP``, ``TS``, ``spread``, ``slippage``. Each is
multiplied by ``pip_equals`` in the outer ``run_single_backtest`` before
any downstream processing. The inner loop, Cython fast path, and all
preprocessors continue to operate in price units.

Rationale: users think about stops, targets, and spreads natively in
pips ("20-pip stop, 2-pip spread"). Forcing them to convert to price
units is a footgun — a value of ``20`` meant to be pips but interpreted
as price units stops out a $100 stock instantly.

Typical values of ``pip_equals``:

| Market                                          | ``pip_equals`` |
| ----------------------------------------------- | -------------- |
| FX majors (EURUSD, GBPUSD, etc.)                | ``0.0001``     |
| FX JPY pairs                                    | ``0.01``       |
| Equity spread-betting (quote in 100ths of unit) | ``1.0``        |
| Equities (price-unit semantics preferred)       | ``1.0``        |
| Crypto majors (BTCUSD, ETHUSD)                  | ``1.0``        |

When ``pip_equals = 1.0``, pip and price unit coincide, so passing
numeric values in price units (e.g. ``SL=3.0`` for a $3 stock stop)
behaves exactly as if no conversion were applied.

---

## 3. Execution semantics (the fixed rules)

Pinned here in one place for reference:

- **Fill price:** open[i+1] for a signal on bar *i*, adjusted by
  direction × (spread + slippage).
- **SL / TP / TS checks:** against H[i] / L[i]. SL beats TP on ties.
- **SL / TP captured at entry** as absolute prices (converted from pip
  distances at entry time). TS captured as a distance; peak/trough
  updates bar-by-bar.
- **Commission:** fraction of notional, on entry and exit.
- **Spread / slippage:** added to entry price, subtracted from exit.
- **Overnight charges:** per-bar vectors, applied in step 1 of each bar,
  triple on Wed by default.
- **Liquidation:** `equity <= 0` flattens all at close[i], terminates loop.
- **Opposite-direction signal (non-hedging, max_positions>1):** flattens
  all, then opens new.
- **Capacity check:** entries skipped silently when all slots are full or
  cash is insufficient for margin + entry commission.

---

## 4. Assumptions

- **OHLCV are pandas Series with a DatetimeIndex**, equal-length, indexes
  aligned, no NaN / inf / zero values. Zeros are rejected because they
  typically indicate a data-feed gap rather than a real price.
- **Bar timestamps represent bar close.** Signals on a bar are assumed to
  be observable at that close, and the resulting fill happens at the next
  bar's open.
- **No intrabar price path.** When both SL and TP could hit on the same
  bar, we conservatively assume SL hit first. When a stop/target is hit,
  the fill happens exactly at the stop/target level.
- **Fills are always filled.** No partial fills, no rejection due to
  liquidity. Slippage is the mechanism for modelling fill-quality.
- **Prices are given in account currency.** No FX conversion layer.
- **Bars per year for annualisation** follow US equity convention (252
  trading days). Override via `_BARS_PER_YEAR` until a proper kwarg is
  added.
- **Single instrument only.** OHLCV is one series; no portfolio across
  tickers.
- **Commission is symmetric** on entry and exit. Some venues have
  asymmetric fee structures (maker/taker, per-contract vs. percent);
  not modelled.
- **Spread is supplied as half-spread in pips.** A `spread = 2` with
  `pip_equals = 0.0001` means 2 pips of half-spread (0.0002 in price
  units), i.e. 4 pips round-trip.

---

## 5. Known limitations & gotchas

1. **`_calculate_exposure` overstates when positions overlap.** It sums
   `bars_held` across trades and divides by total bars. If `max_positions>1`
   or `hedging=True`, overlapping bars are double-counted. Fixing requires
   a per-bar "any position open" flag emitted by the loop.
2. **`_BARS_PER_YEAR` is hardcoded to US equity convention.** Crypto, FX,
   and metals users must monkey-patch the dict. TODO: expose as a kwarg.
3. **`closed_trades` pre-allocated to `n_bars` rows.** Safe for typical
   use. If `max_positions` is high and trades are very short (1 bar
   each), it is theoretically possible to exceed this. A clear
   `RuntimeError` is raised in that case — it is not a silent overflow.
4. **Liquidation fills at close price with no costs.** Slippage on the
   liquidating fill would be more realistic.
5. **No margin-call mechanic other than equity<=0.** Real brokers margin
   you out well before that.
6. **Custom sizer callable blocks numba.** When `position_sizing="custom"`,
   any future numba port must fall back to a pure-Python path or require
   the user to supply a `@cfunc`-decorated callable.
7. **K-ratio formulas differ in the literature.** The three variants
   exposed match the most commonly cited versions (Kestner 1996, 2003,
   2013); a specific benchmark may need a different formula.
8. **Jensen's alpha requires a benchmark return series.** If not provided,
   `calculate_metrics` returns `nan` for that key.
9. **Signal types must be `None`, `np.ndarray`, or `pd.Series`.** Plain
   Python lists are rejected with a clear error. This is deliberate for
   performance and consistency.
10. **Daily midnight timestamps + Wednesday triple charge** places the
    triple on the Thursday bar, not Wednesday. Documented in
    `_process_overnight_charge`; workaround is to use EOD timestamps or
    change `triple_charge_weekday`.
11. **Position sizing for precomputed mode uses the raw value.** Unlike
    percent/value modes, the leverage multiplier is **not** applied in
    precomputed mode — the user is assumed to have already baked any
    scaling into the array. Worth confirming before using it.
12. **No commission floor / min ticket.** Commission is purely percent
    of notional; small trades pay small commission.

---

## 6. Planned upgrades

Grouped by effort.

**Small / mechanical**

- Expose `bars_per_year` as a kwarg on `run_single_backtest`.
- Add a `sl_tp_priority` kwarg (default "sl") for users who want the
  permissive interpretation.
- Add a `commission_floor` kwarg for minimum-ticket venues.
- Add TP and TS as array inputs (mirroring SL).
- Emit per-bar `any_position_open` flag for correct exposure.
- Add `when_max_leverage_breached` kwarg for `percent_at_risk` sizing:
  choose between raise (current), cap size at max leverage, or skip the
  trade and log. Plus a diagnostic stream capturing skipped trades.

**Medium**

- Proper margin-call logic (`margin_call_level` kwarg triggering
  flatten-all above 0 equity).
- Walk-forward / rolling-window harness wrapping `run_single_backtest`.
- Parameter-sweep helper returning a grid of Result objects.
- Parallel run helper (`run_batch_backtests`) for independent param sets.
- More exit-reason codes (margin_call, session_end, max_bars_held).
- Session-aware overnight charges for instruments with defined sessions
  (futures with gaps).

**Larger**

- Multi-asset portfolio layer: multiple OHLCV inputs, per-asset sizing,
  aggregate equity. Likely a new top-level function.
- Numba AOT path with dispatch on sizing-mode and hedging-flag
  specialisation. 4–8 compiled variants. Custom callable path stays in
  pure Python.
- Cython alternative for distribution without JIT overhead.
- A proper test suite covering TP, TS, hedging, pyramiding, overnight
  across multi-day holds, liquidation, custom sizer callable, and
  percent-equity sizing with concurrent positions.

---

## 7. File layout

Order of definitions in `backtester.py`, top to bottom:

```
imports  (numpy, pandas, typing)

# ── module constants ──────────────────────────────────────────
F_DIRECTION .. F_BARS_HELD        # 20 field-index constants
N_FIELDS = 20
EXIT_SIGNAL .. EXIT_LIQUIDATION   # 5 exit-reason codes
_EXIT_REASON_NAMES                # int → string map

# ── trade-record bookkeeping helpers ─────────────────────────
_find_free_slot
_enter_position
_exit_position

# ── public entrypoint ────────────────────────────────────────
run_single_backtest
    └ internal _shift() for signal shifting

# ── preprocessors ────────────────────────────────────────────
_process_series
_SIZING_METHODS / _SIZING_CODES
_process_signals
_process_position_sizing
_process_spread_slippage
_WEEKDAYS
_process_overnight_charge

# ── annualisation lookup ─────────────────────────────────────
_BARS_PER_YEAR

# ── misc helpers ─────────────────────────────────────────────
_longest_run_of_true

# ── inner simulation loop ────────────────────────────────────
_inner_loop

# ── results class ────────────────────────────────────────────
class Result
    __init__
    _bars_per_year, _returns, _log_returns, _pnl, _bars_held
    _calculate_sharpe, _calculate_log_sharpe
    _calculate_sortino, _calculate_log_sortino
    _calculate_max_drawdown, _calculate_cagr, _calculate_calmar
    _calculate_ulcer_index
    _k_ratio_components, _calculate_k_ratio_{1996,2003,2013}
    _calculate_omega_ratio, _calculate_jensens_alpha
    _calculate_expectancy, _calculate_exposure
    _calculate_avg_duration{,_winning,_losing}
    _calculate_max_consecutive_{winners,losers}
    _calculate_biggest_{win,loss}
    _calculate_avg_{winning,losing}_trade
    _calculate_winrate, _calculate_profitfactor
    calculate_metrics
    trades_to_dataframe
    plot_returns, plot_drawdown, plot_metrics
    summary, __repr__
```

---

## 8. Architecture map

```
                                  USER
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │      run_single_backtest         │
                    │  (public entrypoint; validates,  │
                    │   normalises, dispatches)        │
                    └──────────────┬───────────────────┘
                                   │
         ┌──────────────┬──────────┼──────────┬──────────────────┐
         ▼              ▼          ▼          ▼                  ▼
  ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌─────────────┐ ┌──────────────┐
  │_process_   │ │_process_ │ │_process_│ │_process_    │ │_process_     │
  │series      │ │signals   │ │spread_  │ │position_    │ │overnight_    │
  │            │ │(×4)      │ │slippage │ │sizing       │ │charge        │
  │OHLCV check │ │None→0-vec│ │(×2)     │ │4-mode norm  │ │annual→per-bar│
  │strip Series│ │shift +1  │ │scalar→  │ │method code  │ │vec w/triple  │
  │→ ndarrays  │ │→ bool    │ │array    │ │+ static +   │ │charge Wed    │
  │+ date      │ │arrays    │ │         │ │array + fn   │ │              │
  └─────┬──────┘ └────┬─────┘ └────┬────┘ └──────┬──────┘ └──────┬───────┘
        │             │            │             │               │
        └─────────────┴──────┬─────┴─────────────┴───────────────┘
                             │  normalised, validated inputs
                             ▼
                   ┌──────────────────────┐
                   │     _inner_loop      │    ←──  _find_free_slot
                   │  event-driven sim    │    ←──  _enter_position
                   │  1. overnight        │    ←──  _exit_position
                   │  2. SL/TP/TS         │
                   │  3. shifted exits    │
                   │  4. shifted entries  │
                   │  5. mark-to-market   │
                   │  6. liquidation chk  │
                   └──────────┬───────────┘
                              │
                              │ (cash, equity, closed_trades)
                              ▼
                   ┌──────────────────────┐
                   │      Result          │
                   │  cash, equity,       │
                   │  trades, timeframe   │
                   └──────────┬───────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   ┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐
   │calculate_       │ │plot_*       │ │trades_to_        │
   │metrics          │ │returns,     │ │dataframe         │
   │                 │ │drawdown,    │ │                  │
   │22 _calculate_*  │ │metrics grid │ │labelled DataFrame│
   │methods          │ │             │ │with datetime +   │
   │→ dict           │ │             │ │exit_reason names │
   └─────────────────┘ └─────────────┘ └──────────────────┘
```

**Data flow.** OHLCV and signals enter as pandas Series / numpy arrays.
The five preprocessors normalise everything into contiguous float64 arrays
with validated lengths, NaN/inf/zero checks, and mode codes. `_inner_loop`
consumes only normalised inputs and emits three raw arrays: `cash`, `equity`,
and a trimmed trade-log matrix. `Result` wraps those three and computes
everything else on demand.

**Invariant.** Nothing inside `_inner_loop` performs validation or type
coercion. All checks happen in `run_single_backtest` and the
`_process_*` helpers. This keeps the hot path numba-portable later.

---

## 9. Public API reference — `run_single_backtest`

```python
run_single_backtest(
    o, h, l, c, v,
    pip_equals=0.0001,
    starting_balance=10_000.0,
    long_entries=None, long_exits=None,
    short_entries=None, short_exits=None,
    position_sizing="percent_equity",
    position_percent_equity=1.0, position_value=None,
    position_sizes=None, position_sizing_fn=None,
    position_percent_at_risk=None,
    SL=None, TP=None, TS=None,
    leverage=1.0,
    commission=0.0, spread=0.0, slippage=0.0,
    overnight_charge=(0.0, 0.0),
    max_positions=1, hedging=False,
    timeframe="1d",
    *args,
) -> Result
```

**OHLCV (required).** `pd.Series` with a `DatetimeIndex`. Equal length,
aligned indexes, no NaN / inf / zero.

**pip_equals.** Price units per pip. ``0.0001`` for FX majors, ``0.01`` for
JPY pairs, ``1.0`` for equities / equity-spread-bet quotes / crypto.
Applied to SL, TP, TS, spread, and slippage (each supplied in pips) before
they enter the pipeline. See §2.11.

**starting_balance.** Initial cash, account currency. Must be > 0.

**Signals.** Four optional boolean arrays aligned to OHLCV. `None` is
interpreted as "never fires." Accepts `np.ndarray` or `pd.Series`. Values
must be castable to bool. Signals are shifted by +1 internally.

**Position sizing.** See section 2.8. Exactly one of the five
`position_*` args is consulted based on `position_sizing`.
``position_percent_at_risk`` additionally requires a non-NaN ``SL`` and
triggers an upfront leverage-feasibility check; see §2.8.1.

**SL / TP / TS.** Distance in **pips** (converted to price units via
``pip_equals``).
- `SL`: scalar or per-bar array/Series. `None` disables.
- `TP`: scalar. `None` disables.
- `TS`: scalar, the trail distance. `None` disables.

**Leverage.** Multiplier applied to percent-equity and value sizing.
Does not affect precomputed or custom sizing.

**Commission.** Fraction of notional, applied on entry and exit.

**Spread, slippage.** Scalar or per-bar array/Series, in **pips**
(converted via ``pip_equals``). Both act against the trader.

**overnight_charge.** `(annual_base, annual_borrow)` tuple. See §2.7.

**max_positions.** Maximum concurrent positions. With `hedging=True`, the
underlying slot array is doubled (long book + short book each capped at
`max_positions`).

**hedging.** If True, long and short positions may coexist. If False, an
opposite-direction entry flattens all existing positions first.

**timeframe.** One of `_BARS_PER_YEAR`'s keys. Drives overnight scaling
and metric annualisation.

**Returns.** A `Result` object.

**Raises.** `TypeError` / `ValueError` from any preprocessor for malformed
inputs; `RuntimeError` from the loop if the `closed_trades` cap is
exhausted.

---

## 10. `Result` class reference

### Construction

```python
Result(cash, equity, trades, timeframe="1d")
```

- `cash`: 1-D float64 ndarray, per-bar free cash.
- `equity`: 1-D float64 ndarray, per-bar total equity.
- `trades`: 2-D float64 ndarray of shape `(n_trades, 20)`.
- `timeframe`: string key into `_BARS_PER_YEAR`.

Users normally do not construct this directly; `run_single_backtest`
returns it.

### `calculate_metrics(risk_free=0.0, omega_threshold=0.0, benchmark_returns=None) -> dict`

Returns a flat dict with 28 keys:

**Return / risk (11):** `sharpe`, `log_sharpe`, `sortino`, `log_sortino`,
`calmar`, `cagr`, `max_drawdown`, `ulcer_index`, `k_ratio_1996`,
`k_ratio_2003`, `k_ratio_2013`, `omega_ratio`, `jensens_alpha`.

**Trade-level (13):** `n_trades`, `winrate`, `profit_factor`, `expectancy`,
`exposure`, `avg_duration`, `avg_duration_winning`, `avg_duration_losing`,
`max_consecutive_winners`, `max_consecutive_losers`, `biggest_win`,
`biggest_loss`, `avg_winning_trade`, `avg_losing_trade`.

**Overall (3):** `total_return`, `starting_equity`, `final_equity`.

Rates (`risk_free`, `omega_threshold`) are annual; converted to per-bar
internally. `benchmark_returns` must be a per-bar return series if
provided; otherwise `jensens_alpha` is `nan`.

### `trades_to_dataframe() -> pd.DataFrame`

21-column DataFrame. Datetime columns are real `datetime64`, `direction`
is int8, `exit_reason` is a string (`"signal"`, `"sl"`, `"tp"`, `"ts"`,
`"liquidation"`), and a computed `pnl` column is appended. Empty trade
log returns an empty DataFrame with the correct column schema.

### `plot_returns(ax=None, log=False) -> matplotlib.axes.Axes`

Equity curve. `log=True` for log y-axis. Pass an `ax` to compose into a
larger figure.

### `plot_drawdown(ax=None) -> matplotlib.axes.Axes`

Underwater curve (percent drawdown from running peak).

### `plot_metrics(figsize=(12, 8)) -> matplotlib.figure.Figure`

2×2 dashboard: equity (top-left), drawdown (top-right), trade-P&L
histogram (bottom-left), cumulative trade P&L (bottom-right).

### `summary() -> str` / `__repr__`

Human-readable one-screen summary. `print(result)` just works.

### Private helpers (not intended for direct use)

- `_bars_per_year`, `_returns`, `_log_returns`, `_pnl`, `_bars_held`:
  data-access primitives.
- `_calculate_*`: 22 metric methods, each taking only the args its formula
  needs. Aggregated by `calculate_metrics`.
- `_k_ratio_components`: shared OLS fit reused by the three K-ratio
  variants.

---

## 11. Private helpers reference

### `_find_free_slot(slot_active) -> int`

First-fit linear scan. Returns the index of the first inactive slot, or
`-1` if all slots are in use.

### `_enter_position(...)`

Writes a new position into `open_positions[slot_idx]`. Initialises exit
fields to NaN, running costs to entry values, MAE/MFE to 0.0, TS peak to
entry price, bars_held to 0. Flips `slot_active[slot_idx] = True`.

### `_exit_position(...)`

Finalises exit fields, adds exit costs to the running totals, copies the
row into `closed_trades[n_closed]`, flips `slot_active[slot_idx] = False`,
returns `n_closed + 1`. Raises `RuntimeError` if the pre-allocated
`closed_trades` capacity is exhausted (see §5 limitation 3).

### `_process_series(o, h, l, c, v) -> (date, o, h, l, c, v)`

Strips five pandas Series with a shared `DatetimeIndex` into six aligned
contiguous float64 ndarrays (`date` being the extracted `datetime64[ns]`
index). Validates type, length, index alignment, and rejects NaN / inf /
zero values.

### `_process_signals(signals, n, name) -> ndarray[bool]`

`None` → all-False length-n. Otherwise casts to bool, validates 1-D and
length n. `name` is used in error messages.

### `_process_position_sizing(position_sizing, percent, value, sizes, fn, n) -> (method_code, static_size, sizes_array, sizing_fn)`

Resolves the four-mode sizing selector into a uniform internal form.
Validates that the correct argument is populated and well-formed.
`method_code` is 0/1/2/3 for percent/value/precomputed/custom.

### `_process_spread_slippage(value, n, name) -> ndarray[float64]`

Normalises scalar, ndarray, or pandas Series into a length-n contiguous
float64 array. Scalar is broadcast; arrays are validated for length,
1-D, and finite values.

### `_process_overnight_charge(overnight_charge, timeframe, date, denominator=360, rollover_hour_utc=22, triple_charge_weekday="Wednesday") -> (long_vec, short_vec)`

Enumerates daily rollover moments over the date range, weights them
(3× on the triple-charge weekday, 1× otherwise), buckets each into the
first bar closing at-or-after the rollover, and accumulates into two
per-bar float64 vectors. See §2.7 for attribution details.

### `_longest_run_of_true(mask) -> int`

Longest consecutive run of True in a boolean array. Vectorised.

### `_inner_loop(...) -> (cash, equity, closed_trades)`

The simulation. See §2.3 and §3 for semantics. Returns raw arrays;
wrapping in `Result` is done by `run_single_backtest`.

---

## 12. The 20-column trade-record layout

Used for both `open_positions` (per-slot) and `closed_trades` (per-row).
Column indices exported as `F_*` module constants; `N_FIELDS = 20`.

| Idx | Constant          | Meaning                                                          |
| --- | ----------------- | ---------------------------------------------------------------- |
|   0 | `F_DIRECTION`     | +1 (long) or -1 (short)                                          |
|   1 | `F_ENTRY_BAR`     | Bar index of entry fill                                          |
|   2 | `F_ENTRY_TIME`    | Entry timestamp as int64 ns                                      |
|   3 | `F_ENTRY_PRICE`   | Fill price after spread + slippage                               |
|   4 | `F_EXIT_BAR`      | Bar index of exit fill                                           |
|   5 | `F_EXIT_TIME`     | Exit timestamp                                                   |
|   6 | `F_EXIT_PRICE`    | Fill price after spread + slippage                               |
|   7 | `F_SIZE`          | Units held (always positive; sign from direction)                |
|   8 | `F_SL`            | Absolute SL price; NaN if disabled                               |
|   9 | `F_TP`            | Absolute TP price; NaN if disabled                               |
|  10 | `F_TS_DIST`       | Trail distance; NaN if disabled                                  |
|  11 | `F_TS_PEAK`       | Running peak (long) / trough (short)                             |
|  12 | `F_COMMISSION`    | Running total of commission (entry + exit)                       |
|  13 | `F_SPREAD_COST`   | Running total of spread cost                                     |
|  14 | `F_SLIPPAGE_COST` | Running total of slippage cost                                   |
|  15 | `F_OVERNIGHT`     | Running total of overnight financing                             |
|  16 | `F_MAE`           | Max adverse excursion (worst unrealised P&L, negative)           |
|  17 | `F_MFE`           | Max favorable excursion (best unrealised P&L, positive)          |
|  18 | `F_EXIT_REASON`   | Int code: 0=signal, 1=SL, 2=TP, 3=TS, 4=liquidation              |
|  19 | `F_BARS_HELD`     | `exit_bar - entry_bar`                                           |

Net P&L is **not stored** as a column; it is computed on demand by
`Result._pnl`:

```
pnl = direction * (exit_price - entry_price) * size
      - (commission + overnight)
```

Note: ``spread_cost`` and ``slippage_cost`` are **not** subtracted here
because they are already baked into the stored ``entry_price`` and
``exit_price`` (the inner loop shifts the fill price by
``direction * (spread + slippage)`` on both legs). The per-trade
``F_SPREAD_COST`` / ``F_SLIPPAGE_COST`` fields are retained for
diagnostics only (e.g. to answer "what fraction of gross went to
spread?" after the fact).

---

## 13. Numeric conventions

- All prices, sizes, and costs in account currency.
- Returns are arithmetic (`diff / prev`) unless a method is explicitly
  named `_log_*`.
- Drawdown values are negative fractions (e.g. `-0.15` = 15% peak-to-trough).
- Ulcer index is percent-based (multiplied by 100 internally).
- `cagr`, `total_return`: decimal fractions (0.12 = 12%).
- `winrate`, `exposure`: decimal fractions (0.60 = 60%).
- `jensens_alpha`: annualised, decimal fraction.
- `profit_factor`: gross-win / |gross-loss|; `inf` if no losing trades.
- `expectancy`: mean P&L per trade, in account currency.
- `sharpe`, `sortino`: annualised, dimensionless.
- `calmar`: CAGR / |max_dd|, dimensionless.
- `k_ratio_*`: see docstrings; three variants differ in scaling.

All metrics return `0.0` (not NaN) when the input is empty or degenerate
unless explicitly noted (Jensen's alpha is NaN when no benchmark is given;
Omega, Calmar, profit_factor return `inf` when the denominator is zero).

---

## 14. Worked example — SPY daily with yfinance

End-to-end demonstration: a SMA-crossover strategy on SPY daily data,
with realistic costs.

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester import run_single_backtest

# ─── 1. Data ─────────────────────────────────────────────────
df = yf.download("SPY", start="2015-01-01", end="2024-12-31", auto_adjust=True)
df = df.dropna()
# yfinance returns a multi-index; flatten
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

o = df["Open"]
h = df["High"]
l = df["Low"]
c = df["Close"]
v = df["Volume"].replace(0, 1)  # guard against rare zero-volume days

# ─── 2. Signals: 50/200 SMA crossover, long-only ─────────────
sma_fast = c.rolling(50).mean()
sma_slow = c.rolling(200).mean()

# Golden cross (50 crosses above 200) → long entry
# Death cross (50 crosses below 200) → long exit
golden_cross = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
death_cross  = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

# Drop the warmup (first 200 bars have NaN indicators)
valid = sma_slow.notna()
o, h, l, c, v = [s[valid] for s in (o, h, l, c, v)]
golden_cross = golden_cross[valid].fillna(False).to_numpy()
death_cross  = death_cross[valid].fillna(False).to_numpy()

# ─── 3. Run ──────────────────────────────────────────────────
# SPY is an equity, so pip_equals=1.0 — pips and price units coincide.
# Costs and stops are expressed in dollars of price.
result = run_single_backtest(
    o, h, l, c, v,
    starting_balance = 10_000,
    long_entries     = golden_cross,
    long_exits       = death_cross,
    position_sizing  = "percent_equity",
    position_percent_equity = 0.95,   # 95% of equity per trade
    commission       = 0.0005,        # 5 bps per leg
    spread           = 0.01,          # 1c half-spread
    slippage         = 0.02,          # 2c slippage
    overnight_charge = (0.05, 0.0),   # 5% annual base, no borrow (long-only)
    SL               = 10.0,          # $10 stop-loss
    TS               = 15.0,          # $15 trailing stop
    pip_equals       = 1.0,           # equities: 1 unit of price = 1 pip
    timeframe        = "1d",
)

# ─── 4. Results ──────────────────────────────────────────────
print(result)                            # one-screen summary
metrics = result.calculate_metrics()     # full dict of 28 metrics
trades  = result.trades_to_dataframe()   # full trade log

print(f"\nFirst five trades:\n{trades.head().to_string()}")
print(f"\nExit reason breakdown:\n{trades['exit_reason'].value_counts()}")

fig = result.plot_metrics(figsize=(14, 9))
plt.show()
```

**What the output looks like.** Expect a golden/death-cross strategy on
SPY 2015–2024 to:

- Generate a small number of trades (~5–10 over 10 years).
- Spend most of the time in the market (high exposure).
- Track the index fairly closely with occasional large drawdowns during
  crosses that whipsaw.
- Show a mix of signal-closed trades (from death crosses) and possibly
  one or two trailing-stop or stop-loss closes during sharp dips (March
  2020, Oct 2022).

The exact numbers depend on when yfinance's data starts and exactly how
the crossover bars align, so running the snippet is the best way to see
real values.

**Using a custom sizer.** To swap the percent-equity sizing for a simple
volatility-targeting callable:

```python
def vol_target_sizer(bar_idx, entry_time_ns, direction, current_cash,
                    open_positions, slot_active, closed_trades, n_closed,
                    date, o, h, l, c, v):
    # 20-bar realised vol; target 1% daily vol of notional
    lookback = 20
    if bar_idx < lookback:
        return 0.0
    returns = np.diff(np.log(c[bar_idx - lookback : bar_idx + 1]))
    daily_vol = returns.std()
    if daily_vol <= 0:
        return 0.0
    target_notional = 0.01 * current_cash / daily_vol
    return target_notional / c[bar_idx]

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance    = 10_000,
    long_entries        = golden_cross,
    long_exits          = death_cross,
    position_sizing     = "custom",
    position_sizing_fn  = vol_target_sizer,
)
```

---

## 15. Testing status

End-to-end smoke-tested:

- Single long trade: entry-at-next-open and exit-at-next-open arithmetic
  verified.
- SL triggering: exit_reason=1 on a downside gap.
- Short trade: direction and P&L sign correct.
- No-signals case: equity stays flat.
- Metrics dict: all 28 keys populate without errors.
- `trades_to_dataframe`: schema and dtypes correct, empty case handled.
- Plotting: all three methods render without error, including with zero
  trades.
- Preprocessors: individually unit-tested for happy path and common
  error cases (wrong types, wrong lengths, NaN/inf/zero, string
  validation for weekday names, etc.).

**Not yet explicitly tested (deferred to the test suite):**

- TP triggering (only SL tested so far).
- Trailing-stop peak / trough updating across multi-bar holds.
- `hedging=True` — long and short open simultaneously.
- `max_positions>1` pyramiding.
- Opposite-direction signal flatten-all behaviour under non-hedging.
- Commission / spread / slippage cost accounting verified against a
  hand-computed round trip.
- Overnight charges across a multi-day hold, including a Wednesday
  triple.
- Liquidation: equity-goes-negative path, remaining cash/equity
  fill-forward, loop termination.
- `position_sizing="percent_equity"` with current-equity calculation
  while positions are concurrently open.
- `position_sizing="custom"` callable contract (argument order, types,
  return value handling).

This is the obvious next workstream before packaging.
