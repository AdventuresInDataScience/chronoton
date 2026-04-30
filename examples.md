# chronoton — Worked Examples

All examples use synthetic data so they run without any external data feed. Swap the OHLCV construction for real data from yfinance, pandas-datareader, or a CSV when you're ready.

Each example is self-contained: copy-paste it into a script or notebook and it will run as-is.

---

## Contents

1. [Minimal long-only strategy](#1-minimal-long-only-strategy)
2. [Long/short with stop-loss and take-profit](#2-longshort-with-stop-loss-and-take-profit)
3. [Trailing stop on a trending market](#3-trailing-stop-on-a-trending-market)
4. [FX spread-betting with overnight financing](#4-fx-spread-betting-with-overnight-financing)
5. [Risk-based sizing (percent-at-risk)](#5-risk-based-sizing-percent-at-risk)
6. [Pyramiding — multiple concurrent positions](#6-pyramiding--multiple-concurrent-positions)
7. [Hedging — simultaneous long and short book](#7-hedging--simultaneous-long-and-short-book)
8. [Custom position sizer callable](#8-custom-position-sizer-callable)
9. [Inspecting the trade log](#9-inspecting-the-trade-log)
10. [Reading the metrics dict](#10-reading-the-metrics-dict)
11. [Cython fast path and parity check](#11-cython-fast-path-and-parity-check)

---

## Setup used across all examples

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # remove if running in a notebook
import matplotlib.pyplot as plt

from chronoton import run_single_backtest
```

### Synthetic data helpers

```python
def make_ohlcv(close_prices, start="2022-01-03", freq="B"):
    """Build a trivially consistent OHLCV from a close-price sequence."""
    c = np.asarray(close_prices, dtype=np.float64)
    n = c.size
    idx = pd.date_range(start, periods=n, freq=freq)
    return (
        pd.Series(c,          index=idx),   # open  ≈ close
        pd.Series(c + 0.50,   index=idx),   # high
        pd.Series(c - 0.50,   index=idx),   # low
        pd.Series(c,          index=idx),   # close
        pd.Series(np.full(n, 1_000_000.0), index=idx),  # volume
    )

def signal_at(n, *bars):
    """Boolean array of length n that is True only at the given bar indices."""
    arr = np.zeros(n, dtype=bool)
    for b in bars:
        arr[b] = True
    return arr
```

---

## 1. Minimal long-only strategy

The simplest possible run: a single long trade opened at bar 5 and closed at bar 15. No costs, no stops. Good for sanity-checking fill prices and P&L arithmetic.

```python
prices = np.arange(100.0, 130.0)   # prices rise by 1 each bar
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=signal_at(n, 5),    # signal on bar 5 → fill at open[6]
    long_exits=signal_at(n, 15),     # signal on bar 15 → fill at open[16]
    position_sizing="value",
    position_value=1_000.0,          # buy £1000 notional
    pip_equals=1.0,
)

print(result)
# Expected: entry at open[6]=106, exit at open[16]=116, size=1000/106≈9.43
# P&L ≈ (116 - 106) × 9.43 ≈ £94.30
```

**What to look at:**

```python
t = result.trades_to_dataframe()
print(t[["entry_bar", "entry_price", "exit_bar", "exit_price", "pnl"]])
```

---

## 2. Long/short with stop-loss and take-profit

Demonstrates both a stop-loss and a take-profit firing. Note that when **both would hit on the same bar**, the stop-loss wins (conservative assumption — we don't know the intrabar order).

```python
# A price sequence that triggers SL on one trade and TP on another
prices_sl  = np.array([100., 101., 102., 103., 98.,  98.,  98.,  98.])
prices_tp  = np.array([100., 101., 102., 103., 107., 107., 107., 107.])

for label, prices in [("SL fires", prices_sl), ("TP fires", prices_tp)]:
    o, h, l, c, v = make_ohlcv(prices)
    n = len(prices)

    # Widen the high/low so the trigger levels are actually reached
    h_mod = h.copy(); h_mod.iloc[4] = prices[4] + 0.6
    l_mod = l.copy(); l_mod.iloc[4] = prices[4] - 0.6

    result = run_single_backtest(
        o, h_mod, l_mod, c, v,
        starting_balance=10_000,
        long_entries=signal_at(n, 1),   # enter at open[2]
        position_sizing="value", position_value=1_000.0,
        SL=3.0,    # 3-pip (£3) stop below entry
        TP=4.0,    # 4-pip (£4) target above entry
        pip_equals=1.0,
    )

    t = result.trades_to_dataframe()
    print(f"{label}: exit_reason={t['exit_reason'].iloc[0]}, "
          f"pnl={t['pnl'].iloc[0]:.2f}")
```

**Short trades** work identically — swap `long_entries` for `short_entries`:

```python
prices = np.array([100., 101., 102., 103., 104., 98., 97., 96., 95., 94.])
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    short_entries=signal_at(n, 1),
    short_exits=signal_at(n, 8),
    position_sizing="value", position_value=1_000.0,
    pip_equals=1.0,
)

t = result.trades_to_dataframe()
print(t[["direction", "entry_price", "exit_price", "pnl"]])
# direction=-1, entry ≈ 101, exit ≈ 95 → short profits from fall
```

---

## 3. Trailing stop on a trending market

A trailing stop locks in profits as the market moves in your favour. The stop trails the running peak (long) or trough (short) by a fixed pip distance. It only ever moves in the profitable direction.

```python
# Price climbs to 115 then reverses
prices = np.array([100., 101., 103., 106., 110., 115., 112., 109., 107., 105.])
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=signal_at(n, 0),   # enter immediately
    position_sizing="value", position_value=5_000.0,
    TS=3.0,        # trail 3 price units behind running peak
    pip_equals=1.0,
)

print(result)
t = result.trades_to_dataframe()
print(f"Exit reason : {t['exit_reason'].iloc[0]}")   # ts
print(f"TS peak     : {t['ts_peak'].iloc[0]:.2f}")   # ≈ 115.5 (high of peak bar)
print(f"Exit price  : {t['exit_price'].iloc[0]:.2f}")# ≈ 112.5 (peak - 3)
print(f"P&L         : {t['pnl'].iloc[0]:.2f}")
```

---

## 4. FX spread-betting with overnight financing

A realistic FX / CFD setup: pip-unit distances, a half-spread, slippage, commission, and overnight financing that triples on Wednesdays. The key parameter is `pip_equals=0.0001` for major FX pairs (e.g. EURUSD).

```python
# Simulate 20 daily EURUSD bars around 1.0800
np.random.seed(7)
base = 1.0800
moves = np.cumsum(np.random.randn(20) * 0.0003)
prices = base + moves
o, h, l, c, v = make_ohlcv(prices, start="2024-01-08")   # Mon

n = len(prices)

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=signal_at(n, 1),
    long_exits=signal_at(n, 14),
    position_sizing="value",
    position_value=10_000.0,      # £10k notional per trade
    leverage=30.0,                # 30:1 leverage typical for FX retail

    # Costs in pips; pip_equals converts to price units
    pip_equals=0.0001,
    spread=1.5,                   # 1.5 pip half-spread
    slippage=0.5,                 # 0.5 pip slippage
    commission=0.0,               # spread-bet: no separate commission

    # Overnight: 3% annual base funding, 1% borrow spread
    # → long pays (base + borrow) = 4% / 360 per day
    # → short pays (base - borrow) = 2% / 360 per day
    overnight_charge=(0.03, 0.01),
    timeframe="1d",
)

print(result)
t = result.trades_to_dataframe()
print(f"\nOvernight cost : £{t['overnight'].iloc[0]:.4f}")
print(f"Spread cost    : £{t['spread_cost'].iloc[0]:.4f}")
print(f"Slippage cost  : £{t['slippage_cost'].iloc[0]:.4f}")
```

**Interpreting the overnight cost:** the position is held for ~13 bars (bars 2–15). The daily long rate is `(0.03 + 0.01) / 360 = 0.0001111` per unit of notional. With a Wednesday in the hold window charged at 3×, the total is slightly more than `13 × rate × notional`.

---

## 5. Risk-based sizing (percent-at-risk)

Size a position so that if the stop-loss is hit, you lose exactly a fixed fraction of your equity — regardless of where the stop is. This is common in spread-betting and discretionary FX trading.

```python
# A stock moving around £50. We risk 1% of equity per trade on a £2 stop.
# size = (0.01 × equity) / sl_distance = (0.01 × 10_000) / 2 = 50 shares.

np.random.seed(99)
prices = 50 + np.cumsum(np.random.randn(60) * 0.3)
prices = np.clip(prices, 0.01, None)
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

# Force bar 20 low to hit the stop
l_mod = l.copy()
l_mod.iloc[20] = 44.0   # well below entry − 2

result = run_single_backtest(
    o, h, l_mod, c, v,
    starting_balance=10_000,
    long_entries=signal_at(n, 5),
    position_sizing="percent_at_risk",
    position_percent_at_risk=0.01,   # 1% of equity at risk per trade
    SL=2.0,                          # £2 stop (pip_equals=1.0 → 2 price units)
    pip_equals=1.0,
    leverage=5.0,                    # required: check size doesn't exceed leverage
)

t = result.trades_to_dataframe()
print(f"Size         : {t['size'].iloc[0]:.1f} shares")   # ≈ 50
print(f"Exit reason  : {t['exit_reason'].iloc[0]}")        # sl
print(f"P&L          : £{t['pnl'].iloc[0]:.2f}")           # ≈ -£100
```

If the required leverage to honour the sizing would exceed the configured `leverage`, `run_single_backtest` raises a descriptive `ValueError` at the call site rather than silently emitting wrong-sized trades.

---

## 6. Pyramiding — multiple concurrent positions

Set `max_positions > 1` to allow more than one open position at a time in the same direction. Each entry signal opens an additional slot (up to the cap); each exit signal closes all matching-direction slots.

```python
prices = np.full(30, 100.0)   # flat prices: no P&L noise
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

# Three entry signals, two exit signals — cap at 2 slots
long_ent = signal_at(n, 2, 6, 10)   # signals at bars 2, 6, 10
long_ext = signal_at(n, 20)          # one exit closes all open longs

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=long_ent,
    long_exits=long_ext,
    position_sizing="value", position_value=500.0,
    max_positions=2,     # only 2 slots → bar 10 entry is rejected (slots full)
    pip_equals=1.0,
)

print(f"Trades executed: {result.trades.shape[0]}")  # 2 (third rejected)
t = result.trades_to_dataframe()
print(t[["entry_bar", "exit_bar", "size"]])
```

---

## 7. Hedging — simultaneous long and short book

With `hedging=True`, long and short positions may coexist. The slot count is doubled internally (one long book + one short book, each capped at `max_positions`). An opposite-direction entry no longer flattens existing positions.

```python
prices = np.arange(100.0, 120.0)
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=20_000,
    long_entries=signal_at(n, 0),    # open long immediately
    short_entries=signal_at(n, 3),   # open short at bar 3 — coexists with long
    long_exits=signal_at(n, 15),
    short_exits=signal_at(n, 15),
    position_sizing="value", position_value=1_000.0,
    hedging=True,
    max_positions=1,    # 1 long + 1 short simultaneously
    pip_equals=1.0,
)

t = result.trades_to_dataframe()
print(t[["direction", "entry_bar", "exit_bar", "pnl"]])
# Two rows: one long (profits from rising prices), one short (loses from same)
```

Compare to the non-hedging behaviour (default `hedging=False`): a short entry at bar 3 would first flatten the long before opening the short.

---

## 8. Custom position sizer callable

Use `position_sizing="custom"` when your sizing logic depends on trade history, a volatility estimate, or any other state that the built-in modes don't cover. The callable receives the full loop state on every entry.

### 8a. Fixed-dollar Kelly fraction (simplified)

```python
def kelly_sizer(bar_idx, entry_time_ns, direction, current_cash,
                open_positions, slot_active, closed_trades, n_closed,
                date, o, h, l, c, v):
    """
    Simplified half-Kelly using historical win rate and average win/loss.
    Falls back to a fixed small size until there are at least 10 closed trades.
    """
    if n_closed < 10:
        return 2.0   # fixed warm-up size

    recent = closed_trades[:n_closed]
    # Net P&L per trade (gross − commission − overnight)
    pnl = (recent[:, 0] *                    # direction
           (recent[:, 6] - recent[:, 3]) *   # exit_price - entry_price
           recent[:, 7]                       # size
           - recent[:, 12] - recent[:, 15])  # commission + overnight

    wins  = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    if losses.size == 0 or wins.size == 0:
        return 2.0

    win_rate  = wins.size / n_closed
    avg_win   = wins.mean()
    avg_loss  = losses.mean()
    edge      = win_rate / (1 - win_rate)     # b * p / q in Kelly
    kelly_f   = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    half_kelly = max(0.0, kelly_f * 0.5)

    # Translate fraction to units at current price
    notional = half_kelly * current_cash
    return notional / c[bar_idx]


np.random.seed(11)
prices = 100 + np.cumsum(np.random.randn(200) * 0.4)
prices = np.clip(prices, 0.01, None)
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

entries = np.zeros(n, dtype=bool); entries[::15] = True
exits   = np.zeros(n, dtype=bool); exits[7::15]  = True

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=entries,
    long_exits=exits,
    position_sizing="custom",
    position_sizing_fn=kelly_sizer,
    pip_equals=1.0,
)
print(result)
```

### 8b. Volatility-targeting sizer

```python
def vol_target_sizer(bar_idx, entry_time_ns, direction, current_cash,
                     open_positions, slot_active, closed_trades, n_closed,
                     date, o, h, l, c, v):
    """
    Target 1% daily volatility of notional.
    size = (target_vol × cash) / (realised_vol × price)
    """
    lookback = 20
    if bar_idx < lookback + 1:
        return 0.0
    log_returns = np.diff(np.log(c[bar_idx - lookback : bar_idx + 1]))
    daily_vol = log_returns.std()
    if daily_vol <= 0:
        return 0.0
    target_vol = 0.01
    notional = (target_vol * current_cash) / daily_vol
    return notional / c[bar_idx]
```

Note: `position_sizing="custom"` always uses the **pure-Python inner loop**, even when `cython_backtester` is the importer, because a Python callable cannot cross the Cython `nogil` boundary.

---

## 9. Inspecting the trade log

`result.trades_to_dataframe()` returns a 21-column DataFrame with one row per closed trade.

```python
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(100) * 0.6)
prices = np.clip(prices, 0.01, None)
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

entries = np.zeros(n, dtype=bool); entries[[5, 25, 50, 70, 85]] = True
exits   = np.zeros(n, dtype=bool); exits[[15, 40, 60, 80, 95]]  = True

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=entries, long_exits=exits,
    position_sizing="percent_equity", position_percent_equity=0.5,
    SL=5.0, TP=8.0,
    commission=0.001, spread=0.10, slippage=0.05, pip_equals=1.0,
)

df = result.trades_to_dataframe()

# Key columns
print(df[["entry_time", "entry_price", "exit_price",
          "exit_reason", "bars_held", "mae", "mfe", "pnl"]].to_string())

# Cost breakdown
df["total_cost"] = df["commission"] + df["spread_cost"] + df["slippage_cost"] + df["overnight"]
print("\nCost breakdown per trade:")
print(df[["commission", "spread_cost", "slippage_cost", "overnight", "total_cost"]].to_string())

# Exit reason distribution
print("\nExit reasons:")
print(df["exit_reason"].value_counts())
```

**Column reference** (selected):

| Column | Meaning |
|---|---|
| `direction` | +1 long, -1 short |
| `entry_price` / `exit_price` | Fill prices including spread + slippage |
| `mae` | Max adverse excursion — worst unrealised P&L during the trade (negative) |
| `mfe` | Max favourable excursion — best unrealised P&L during the trade (positive) |
| `exit_reason` | `"signal"`, `"sl"`, `"tp"`, `"ts"`, `"liquidation"`, `"end_of_data"` |
| `bars_held` | Number of bars between entry and exit |
| `pnl` | Net P&L (gross − commission − overnight) |
| `spread_cost` / `slippage_cost` | Diagnostics; already baked into stored fill prices |

---

## 10. Reading the metrics dict

`result.calculate_metrics()` returns a flat dict of 28 values. Most are self-explanatory; a few need care.

```python
result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=entries, long_exits=exits,
    position_sizing="percent_equity", position_percent_equity=0.6,
    commission=0.001, pip_equals=1.0,
)

m = result.calculate_metrics(
    risk_free=0.05,          # 5% annual risk-free rate for Sharpe/Sortino/alpha
    omega_threshold=0.02,    # 2% annual threshold for Omega ratio
    # benchmark_returns=spy_returns,  # per-bar returns for Jensen's alpha
)

# Return / risk
print(f"CAGR          : {m['cagr']*100:.1f}%")
print(f"Max drawdown  : {m['max_drawdown']*100:.1f}%")
print(f"Sharpe        : {m['sharpe']:.2f}")
print(f"Sortino       : {m['sortino']:.2f}")
print(f"Calmar        : {m['calmar']:.2f}")
print(f"Ulcer index   : {m['ulcer_index']:.2f}")
print(f"Omega ratio   : {m['omega_ratio']:.2f}")
print(f"K-ratio 2013  : {m['k_ratio_2013']:.4f}")

# Trade-level
print(f"\nTrades        : {m['n_trades']}")
print(f"Win rate      : {m['winrate']*100:.1f}%")
print(f"Profit factor : {m['profit_factor']:.2f}")
print(f"Expectancy    : £{m['expectancy']:.2f} per trade")
print(f"Exposure      : {m['exposure']*100:.1f}% of bars in market")
print(f"Avg duration  : {m['avg_duration']:.1f} bars")
print(f"Biggest win   : £{m['biggest_win']:.2f}")
print(f"Biggest loss  : £{m['biggest_loss']:.2f}")
```

### Numeric conventions to remember

- `cagr`, `total_return`, `winrate`, `exposure` — decimal fractions (`0.12` = 12%)
- `max_drawdown` — negative fraction (`-0.15` = 15% peak-to-trough)
- `jensens_alpha` — annualised decimal; `nan` if no benchmark supplied
- `profit_factor` — `inf` if there are no losing trades
- `calmar` — `inf` if max drawdown is zero

### Text tearsheet

```python
print(result.tearsheet())
# Prints a formatted block with sections: EQUITY, RISK, TRADES, DURATION,
# COSTS — and a long/short breakdown when both directions are present.
# "Expectancy" is shown as "Avg PnL per trade" to avoid ambiguity.
```

### Visual tearsheet

```python
# Full 9-panel dashboard
fig = result.plot_tearsheet(figsize=(18, 26))
plt.show()

# Compact 4-panel dashboard (equity, drawdown, P&L hist, cumulative P&L)
fig = result.plot_metrics(figsize=(14, 9))
plt.show()
```

### Individual panels

```python
ax = result.plot_returns(log=False)       # equity curve (date x-axis)
ax = result.plot_drawdown()               # underwater drawdown curve
ax = result.plot_monthly_returns()        # CAGR heatmap by month/year
ax = result.plot_annual_returns()         # bar chart by calendar year
ax = result.plot_return_by_month()        # seasonality — by calendar month
ax = result.plot_return_by_dow()          # seasonality — by day of week
ax = result.plot_rolling_sharpe(window=252)  # rolling Sharpe
ax = result.plot_mae_mfe()               # MAE vs MFE scatter by trade
ax = result.plot_duration_hist()         # trade duration histogram
plt.show()
```

---

## 11. Cython fast path and parity check

`from chronoton import run_single_backtest` **already dispatches to the Cython fast path** when the compiled extension is available — no import change is needed. You can confirm this at runtime:

```python
from chronoton import cython_available, cython_import_error
print(cython_available())      # True → compiled fast path active
print(cython_import_error())   # None if compiled, else the ImportError
```

The dispatcher automatically falls back to the pure-Python loop for `position_sizing="custom"` (a Python callable can't cross the `nogil` boundary) or when the extension is not built.

For parity testing — or to force the pure-Python path explicitly — you can import both modules directly:

```python
import chronoton.backtester as bt_py
import chronoton.cython_backtester as bt_cy

print(f"Cython available: {bt_cy.cython_available()}")

# Build identical inputs
np.random.seed(0)
prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
prices = np.clip(prices, 0.01, None)
o, h, l, c, v = make_ohlcv(prices)
n = len(prices)

entries = np.zeros(n, dtype=bool); entries[::20] = True
exits   = np.zeros(n, dtype=bool); exits[10::20] = True

kwargs = dict(
    starting_balance=10_000,
    long_entries=entries, long_exits=exits,
    position_sizing="percent_equity", position_percent_equity=0.5,
    SL=3.0, commission=0.001, spread=0.05, pip_equals=1.0,
)

r_py = bt_py.run_single_backtest(o, h, l, c, v, **kwargs)
r_cy = bt_cy.run_single_backtest(o, h, l, c, v, **kwargs)

# Results must be numerically identical
assert np.allclose(r_py.equity, r_cy.equity), "Equity curves diverged!"
assert np.allclose(r_py.trades, r_cy.trades), "Trade logs diverged!"

print("Pure-Python and Cython paths produce identical results.")
print(f"\nPure-Python  final equity: £{r_py.equity[-1]:.2f}")
print(f"Cython       final equity: £{r_cy.equity[-1]:.2f}")
```

### Timing comparison

```python
import time

def time_run(module, n_runs=10):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        module.run_single_backtest(o, h, l, c, v, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)   # best-of to reduce noise

if bt_cy.cython_available():
    t_py = time_run(bt_py)
    t_cy = time_run(bt_cy)
    print(f"Pure-Python : {t_py*1000:.2f} ms")
    print(f"Cython      : {t_cy*1000:.2f} ms")
    print(f"Speedup     : {t_py/t_cy:.1f}×")
```

The Cython path shows its biggest advantage on longer series (e.g. minute data over several years) where the per-bar overhead compounds.

---

## Complete end-to-end example

A fuller realistic simulation — SMA crossover on synthetic daily equity data, with all costs enabled and a trailing stop to capture trends.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronoton import run_single_backtest

# ── 1. Synthetic data: a trending-then-choppy equity ────────────────────────
np.random.seed(123)
n = 504   # ~2 trading years
trend  = np.linspace(0, 20, n)
noise  = np.cumsum(np.random.randn(n) * 0.4)
prices = 100 + trend + noise
prices = np.clip(prices, 0.01, None)

idx = pd.date_range("2022-01-03", periods=n, freq="B")
o = pd.Series(prices,          index=idx)
h = pd.Series(prices + 0.60,   index=idx)
l = pd.Series(prices - 0.60,   index=idx)
c = pd.Series(prices,          index=idx)
v = pd.Series(np.full(n, 2e6), index=idx)

# ── 2. Signals: 20/50 SMA crossover ─────────────────────────────────────────
fast_ma = c.rolling(20).mean()
slow_ma = c.rolling(50).mean()

long_entries = (
    (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
).fillna(False).to_numpy()

long_exits = (
    (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
).fillna(False).to_numpy()

# ── 3. Run ───────────────────────────────────────────────────────────────────
result = run_single_backtest(
    o, h, l, c, v,
    starting_balance   = 10_000,
    long_entries       = long_entries,
    long_exits         = long_exits,
    position_sizing    = "percent_equity",
    position_percent_equity = 0.95,
    SL                 = 4.0,     # £4 hard stop
    TS                 = 6.0,     # £6 trailing stop
    commission         = 0.001,   # 10 bps per leg
    spread             = 0.05,    # 5p half-spread
    slippage           = 0.02,    # 2p slippage
    pip_equals         = 1.0,
    timeframe          = "1d",
)

# ── 4. Summary ───────────────────────────────────────────────────────────────
print(result.tearsheet())       # full text stats block

m = result.calculate_metrics()
print(f"\nCAGR         : {m['cagr']*100:.1f}%")
print(f"Max drawdown : {m['max_drawdown']*100:.1f}%")
print(f"Sharpe       : {m['sharpe']:.2f}")
print(f"Profit factor: {m['profit_factor']:.2f}")

df = result.trades_to_dataframe()
print(f"\nTrades by exit reason:\n{df['exit_reason'].value_counts()}")
print(f"\nFirst 5 trades:\n{df.head().to_string()}")

# ── 5. Plot ──────────────────────────────────────────────────────────────────
fig = result.plot_tearsheet(figsize=(18, 26))   # 9-panel visual tearsheet
plt.suptitle("SMA 20/50 Crossover — Synthetic Equity", y=1.01)
plt.tight_layout()
plt.show()
```
