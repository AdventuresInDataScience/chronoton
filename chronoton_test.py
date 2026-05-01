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
long_entry = (df['Close'] > ta.EMA(df['Close'], timeperiod=72000)) & (df['Close'].shift(1) < ta.EMA(df['Close'].shift(1), timeperiod=72000))
SLs = 50
TPs = 50

#%%
start_time = time.time()
result = run_single_backtest(
    df['Open'], df['High'], df['Low'], df['Close'], df['Volume'],
    starting_balance=1_000_000,
    long_entries=long_entry,
    short_entries=None, # short_entry,
    long_exits=None,
    short_exits=None, # short_exit,
    position_sizing="percent_at_risk",
    position_percent_at_risk=0.01,
    leverage=30.0,                # 30:1 leverage typical for FX retail
    SL=SLs,
    TP=TPs,

    # Costs in pips; pip_equals converts to price units
    pip_equals=0.0001,
    spread=0,                   
    slippage=0.0,                 
    commission=0.0,               

    # Overnight: 3% annual base funding, 1% borrow spread
    # → long pays 4% / 360 per day, short pays 2% / 360 per day
    overnight_charge=(0.03, 0.01),
    timeframe="1m",
    bars_per_year="forex",
)
end_time = time.time()
print(f"Backtest completed in {end_time - start_time:.2f} seconds")
print(result.tearsheet())
#result.plot_tearsheet()
#plt.show()
# Example of single chart
result.plot_returns(log=False) 
plt.show()
result.trades_to_dataframe()
#%% Issues
#1. Overnight often not appearing even when thre are thousands of trades
#2. 30-60s backtest on 8.3 million bars seems long for a simple strategy - need to profile and optimize
#3. Logic incorrect. The same exits/entries reversed with no fees, yield different curves, not mirrored as expected. Need to investigate and fix.
#4. Above test, i 'reverse' mode, yields a huge drop at the end of the backtest, which is not expected. Need to investigate and fix.
# %%
