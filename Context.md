# Context

Three issues found after running chronoton on 8.3M EUR/USD 1-minute bars. Two are real bugs in the installed package; one is a combination of a formula bug + genuinely low expected charges for short-duration trades.

---

## Issue 2 — Cython never called (CRITICAL, `__init__.py` bug)

`__init__.py:2` imports `run_single_backtest` from `backtester` (pure Python). The Cython-dispatching version is in `cython_backtester.py` and is never exposed in the public API.

```python
# __init__.py — current (wrong)
from chronoton.backtester import (
    run_single_backtest,   # pure Python, Cython never dispatched
    ...
)
```

The dispatch logic at `cython_backtester.py:327` is completely bypassed. 30–60 s on 8.3M bars is consistent with the pure-Python `_inner_loop`.

**Fix:** Change `__init__.py` to import `run_single_backtest` (and `cython_available`/`cython_import_error`) from `cython_backtester`. The field/exit-reason constants and `Result` remain from `backtester`.

---

## Issue 1a — Overnight formula inverted (`backtester.py` bug)

`backtester.py:810-811` currently:
```python
long_daily  = daily_base - daily_borrow   # wrong: longs pay LESS than base
short_daily = daily_base + daily_borrow   # wrong: shorts pay MORE than base
```

Correct convention: longs borrow capital, paying base + spread; shorts lend the security/currency, earning the borrow and paying only base:
```python
long_daily  = daily_base + daily_borrow   # correct: long pays full cost
short_daily = daily_base - daily_borrow   # correct: short earns borrow credit
```

The docstring on lines 709-713 must also be updated to match.

Both `backtester.run_single_backtest` and `cython_backtester.run_single_backtest` call `_process_overnight_charge` from the same place, so this single change fixes both paths.

---

## Issue 1b — Why overnight totals appear small (expected behaviour)

With 120-bar (2-hour) trades on 1-minute data, the chance of spanning the 22:00 UTC rollover is ~8% per trade. For the EMA(180)/EMA(1440) strategy specifically, crossovers on a 24-hour moving average are rare (~40-70 per year), so with 15+ years of data there may be only 500–700 trades total. Expected total overnight:

```
~600 trades × 8% × (0.04/360 × 30,000 notional) = ~£200
```

At current (wrong) rate (2% not 4%): ~£100. The £66–£16 figures the user sees are plausible for strategies with fewer total trades. The formula fix doubles all charges.

**To verify** once the formula is fixed, add these diagnostics to the test file:
```python
ov = t['overnight']
print(f"Total overnight  : £{ov.sum():.2f}")
print(f"Trades with charge: {(ov > 0).sum()} / {len(ov)}")
print(f"Avg charge (non-zero): £{ov[ov > 0].mean():.4f}")
```

---

## Files to change

| File | Change |
|------|--------|
| `.venv/Lib/site-packages/chronoton/__init__.py` | Import `run_single_backtest`, `cython_available`, `cython_import_error` from `cython_backtester`; keep all constants/`Result` from `backtester` |
| `.venv/Lib/site-packages/chronoton/backtester.py:810-811` | Swap `long_daily`/`short_daily` formula; update docstring lines ~709-713 |
| `chronoton_testing/chronoton_test.py` | Add `cython_available()` check print; add overnight diagnostics |

> These edits are to the installed `.venv` copy. The same changes should be pushed upstream to the git source so reinstalls don't regress.

---

## Issue 3 — Downward equity curves

Cython fix will not change equity curve shape (Python fallback is semantically identical, just slower). Awaiting user's results with the inverted crossover before diagnosing further. If both directions produce smooth declines, the next step is to run with zero spread/slippage to isolate whether the engine has a sign error vs. the strategy being genuinely unprofitable.

---

## Verification

1. After `__init__.py` fix: `from chronoton.cython_backtester import cython_available; print(cython_available())` → `True`, and runtime should drop to < 5 s.
2. After formula fix: re-run the test and print `t['overnight'].sum()` — should be ~2× the previous figure.
3. To isolate engine correctness: run with `spread=0, slippage=0, overnight_charge=(0,0)` on a trivial all-long strategy (buy at open, sell at close each day) and verify that equity tracks the sum of `close - open` moves exactly.
