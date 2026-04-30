#uv pip install "chronoton @ git+https://github.com/AdventuresInDataScience/chronoton.git"
#%%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Agg")   # remove if running in a notebook
import matplotlib.pyplot as plt
from chronoton import run_single_backtest, cython_available
import talib as ta
import time
print(f"Cython fast path active: {cython_available()}")
#%% Load Data
df = pd.read_csv("C:\\Users\\malha\\Documents\\Data\\Forex\\eurusd\\eurusd_1m.csv",
compression='gzip', parse_dates=['Datetime'])
df['Volume'] = 10_000  # dummy volume column, not used in this test
df.set_index('Datetime', inplace=True)
df = df[df['Close'] < 2] 
df = df[df['Close'] >0.8]# filter out bad data

#%% Quick Example Logic
long_entry = (ta.EMA(df['Close'], timeperiod=180) > ta.EMA(df['Close'], timeperiod=1440)) & (ta.EMA(df['Close'].shift(1), timeperiod=180) < ta.EMA(df['Close'].shift(1), timeperiod=1440))
#short_entry = ta.EMA(df['Close'], timeperiod=50) < ta.EMA(df['Close'], timeperiod=200)
long_exit = long_entry.shift(720).fillna(False)  # exit long after 30 minutes
#short_exit = short_entry.shift(30).fillna(False)  # exit short after 30 minutes


#%%
start_time = time.time()
result = run_single_backtest(
    df['Open'], df['High'], df['Low'], df['Close'], df['Volume'],
    starting_balance=1_000_000,
    long_entries=long_entry,
    short_entries=None, # short_entry,
    long_exits=long_exit,
    short_exits=None, # short_exit,
    position_sizing="value",
    position_value=1_000.0,      # £10k notional per trade
    leverage=30.0,                # 30:1 leverage typical for FX retail

    # Costs in pips; pip_equals converts to price units
    pip_equals=0.0001,
    spread=1.2,                   # 1.5 pip half-spread
    slippage=0.1,                 # 0.1 pip slippage
    commission=0.0,               # spread-bet: no separate commission

    # Overnight: 3% annual base funding, 1% borrow spread
    # → long pays 4% / 360 per day, short pays 2% / 360 per day
    overnight_charge=(0.03, 0.01),
    timeframe="1m",
)
end_time = time.time()
print(f"Backtest completed in {end_time - start_time:.2f} seconds")
print(result.tearsheet())
t = result.trades_to_dataframe()

ov = t['overnight']
print(f"\nTotal overnight             : £{ov.sum():.2f}")
print(f"Trades with overnight charge: {(ov > 0).sum()} / {len(ov)}")
if (ov > 0).any():
    print(f"Avg overnight (non-zero)    : £{ov[ov > 0].mean():.4f}")
print(f"Total spread cost           : £{t['spread_cost'].sum():.2f}")
print(f"Total slippage cost         : £{t['slippage_cost'].sum():.2f}")

wins   = t[t['pnl'] > 0]
losses = t[t['pnl'] < 0]
print(f"\nAvg PnL    (all trades)     : £{t['pnl'].mean():.2f}")
print(f"Avg profit (winning trades) : £{wins['pnl'].mean():.2f}" if len(wins) else "No winning trades")
print(f"Avg loss   (losing trades)  : £{losses['pnl'].mean():.2f}" if len(losses) else "No losing trades")
(t['pnl'].cumsum() + 1_000_000).plot()
result.plot_tearsheet()
plt.show()

# Issues
#1. Overnight often not appearing even when thre are thousands of trades
#2. 30-60s backtest on 8.3 million bars seems long for a simple strategy - need to profile and optimize
#3. Logic incorrect. The same exits/entries reversed with no fees, yield different curves, not mirrored as expected. Need to investigate and fix.
#4. Above test, i 'reverse' mode, yields a huge drop at the end of the backtest, which is not expected. Need to investigate and fix.
# %%
