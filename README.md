# chronoton

An event-driven backtester for vectorised strategies expressed as boolean signal arrays on OHLCV data. Install it as a package, import from it, and run backtests.

---

## Features

- **Long, short, hedged, and pyramided positions**
- **Five position-sizing modes** — percent of equity, fixed notional, precomputed per-bar sizes, risk-based (percent-at-risk), or a user-supplied callable
- **Realistic costs** — commission, bid-ask spread, slippage, and per-bar overnight financing with triple-charge weekend convention
- **Stop-loss, take-profit, and trailing stop** — scalar or per-bar array SL; SL beats TP when both fire on the same bar
- **28 performance metrics** — Sharpe, Sortino, Calmar, CAGR, max drawdown, Ulcer index, three K-ratio variants, Omega ratio, Jensen's alpha, win rate, profit factor, expectancy, and more
- **Rich trade log** — every closed trade records entry/exit, MAE, MFE, costs, and exit reason, exportable to pandas DataFrame
- **Equity-curve plots** — equity, drawdown, P&L histogram, cumulative P&L
- **Optional Cython fast path** — compiled automatically on install; pure-Python fallback always available

---

## Installation

### From GitHub

```bash
pip install "chronoton @ git+https://github.com/AdventuresInDataScience/chronoton.git"
```

Or with uv:

```bash
uv pip install "chronoton @ git+https://github.com/AdventuresInDataScience/chronoton.git"
```

The Cython extension is compiled during install. You need a C compiler:

| Platform | Requirement |
|---|---|
| Windows | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| macOS | `xcode-select --install` |
| Linux | `sudo apt install python3-dev gcc` (or equivalent) |

If compilation fails the package still installs and works — `chronoton.cython_backtester` silently falls back to the pure-Python loop.

### For development

```bash
git clone https://github.com/AdventuresInDataScience/chronoton.git
cd chronoton
uv venv && uv pip install -e ".[build,test]"
```

---

## Quick start

```python
import numpy as np
import pandas as pd
from chronoton import run_single_backtest

# OHLCV must be pandas Series with a DatetimeIndex
n = 252
idx = pd.date_range("2023-01-01", periods=n, freq="B")
prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

o = pd.Series(prices,          index=idx)
h = pd.Series(prices + 0.50,   index=idx)
l = pd.Series(prices - 0.50,   index=idx)
c = pd.Series(prices,          index=idx)
v = pd.Series(np.full(n, 1e6), index=idx)

# Boolean signal arrays aligned to OHLCV (shifted internally — no look-ahead)
fast = c.rolling(10).mean()
slow = c.rolling(30).mean()
long_entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False).to_numpy()
long_exits   = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False).to_numpy()

result = run_single_backtest(
    o, h, l, c, v,
    starting_balance=10_000,
    long_entries=long_entries,
    long_exits=long_exits,
    position_sizing="percent_equity",
    position_percent_equity=0.95,
    SL=2.0,            # 2-unit stop (pip_equals=1.0 → price units)
    commission=0.001,  # 10 bps per leg
    spread=0.05,
    pip_equals=1.0,    # equities: 1 pip = 1 price unit
    timeframe="1d",
)

print(result)                         # one-screen summary
metrics = result.calculate_metrics() # dict of 28 metrics
trades  = result.trades_to_dataframe()
fig     = result.plot_metrics()
```

### Using the Cython fast path

```python
from chronoton.cython_backtester import run_single_backtest

# Drop-in replacement — identical API and results, faster on large datasets
result = run_single_backtest(o, h, l, c, v, ...)
```

Check whether the compiled extension loaded:

```python
import chronoton.cython_backtester as cy
print(cy.cython_available())      # True → compiled fast path active
print(cy.cython_import_error())   # None, or the ImportError if build failed
```

---

## Package layout

```
src/
└── chronoton/
    ├── __init__.py          # re-exports run_single_backtest, Result, constants
    ├── backtester.py        # pure-Python backtester (public API + inner loop)
    ├── cython_backtester.py # Cython dispatcher — drop-in replacement
    └── _cy_inner.pyx        # Cython source for the compiled inner loop
tests/
    tests.py                 # 90 standalone tests (python tests.py or pytest)
    tests_cython.py          # 95 tests via the Cython dispatcher
setup.py                     # builds chronoton._cy_inner
pyproject.toml               # package metadata and build config
README.md
examples.md                  # worked examples
```

---

## Running the tests

After a development install:

```bash
python tests.py           # 90/90 pure-Python
python tests_cython.py    # 95/95 via Cython dispatcher

# or under pytest
pytest tests.py tests_cython.py -v
```

---

## Key concepts

### Signal arrays

Pass **unshifted** boolean arrays. The backtester shifts by +1 bar internally: a signal on bar *i* (visible at close *i*) fills at the **open of bar *i+1***. This prevents look-ahead.

### pip_equals

All distance inputs (SL, TP, TS, spread, slippage) are in **pips**, converted to price units via `pip_equals`:

| Market | `pip_equals` |
|---|---|
| Equities, crypto | `1.0` |
| FX majors (EURUSD, GBPUSD) | `0.0001` |
| FX JPY pairs | `0.01` |

### Position sizing

| `position_sizing` | Key argument | Behaviour |
|---|---|---|
| `"percent_equity"` | `position_percent_equity` | Fraction of current equity per trade |
| `"value"` | `position_value` | Fixed notional per trade |
| `"precomputed"` | `position_sizes` | Per-bar size array |
| `"percent_at_risk"` | `position_percent_at_risk` | Size so that SL hit = exactly that fraction of equity lost |
| `"custom"` | `position_sizing_fn` | Callable with full loop state |

See [examples.md](examples.md) for worked examples of each mode.
