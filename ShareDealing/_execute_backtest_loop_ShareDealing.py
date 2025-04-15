import numpy as np
cimport numpy as np
from libc.math cimport isnan

cpdef tuple execute_backtest_loop(np.ndarray[double, ndim=2] data, 
                                 np.ndarray[np.uint8_t, ndim=1] buy_signals,
                                 np.ndarray[np.uint8_t, ndim=1] sell_signals,
                                 np.ndarray[np.uint8_t, ndim=1] short_signals,
                                 np.ndarray[np.uint8_t, ndim=1] cover_signals,
                                 double commission=0.0, 
                                 double commission_pct=0.0):
    """
    Execute the backtest loop using the provided signals.
    
    Parameters:
    -----------
    data : numpy.ndarray (2D)
        Market data to run the backtest against, assumes column 3 is Close price
    buy_signals, sell_signals, short_signals, cover_signals : numpy.ndarray (1D, boolean)
        Boolean arrays indicating trading signals
    commission : float
        Fixed commission per trade
    commission_pct : float
        Percentage commission per trade
    
    Returns:
    --------
    tuple
        Tuple of (equity_curve, trades)
    """
    cdef int data_length = data.shape[0]
    cdef np.ndarray[double, ndim=1] equity_curve = np.zeros(data_length, dtype=np.float64)
    
    # Trade tracking variables
    cdef list trades = []
    cdef int position = 0  # 0 = no position, 1 = long, -1 = short
    cdef double entry_price = 0.0
    cdef int entry_idx = 0
    cdef double cash = 10000.0  # Starting cash
    cdef double shares = 0.0
    cdef double price, trade_commission, trade_profit
    cdef int i
    
    # Initialize equity curve with starting cash
    equity_curve[0] = cash
    
    # Main backtest loop
    for i in range(1, data_length):
        price = data[i, 3]  # Assume close price is at index 3
        
        # Skip if price is NaN
        if isnan(price):
            equity_curve[i] = equity_curve[i-1]
            continue
        
        # Process signals
        if position == 0:  # No position
            # Check for buy signal
            if buy_signals[i]:
                position = 1
                entry_price = price
                entry_idx = i
                shares = cash / (price * (1.0 + commission_pct) + commission)
                cash = 0.0
                
                # Record trade entry
                trades.append({
                    'entry_idx': i,
                    'entry_price': price,
                    'direction': 'long',
                    'shares': shares,
                })
                
            # Check for short signal
            elif short_signals[i]:
                position = -1
                entry_price = price
                entry_idx = i
                shares = cash / (price * (1.0 + commission_pct) + commission)
                cash = 0.0
                
                # Record trade entry
                trades.append({
                    'entry_idx': i,
                    'entry_price': price,
                    'direction': 'short',
                    'shares': shares,
                })
                
        elif position == 1:  # Long position
            # Check for sell signal
            if sell_signals[i]:
                # Calculate trade value and commission
                trade_commission = (price * shares * commission_pct) + commission
                cash = (price * shares) - trade_commission
                
                # Calculate profit
                trade_profit = (price - entry_price) * shares - trade_commission
                
                # Update trade record
                trades[-1].update({
                    'exit_idx': i,
                    'exit_price': price,
                    'profit': trade_profit,
                    'duration': i - entry_idx,
                })
                
                # Reset position
                position = 0
                shares = 0.0
                
        elif position == -1:  # Short position
            # Check for cover signal
            if cover_signals[i]:
                # Calculate trade value and commission
                trade_commission = (price * shares * commission_pct) + commission
                cash = (2 * entry_price - price) * shares - trade_commission
                
                # Calculate profit
                trade_profit = (entry_price - price) * shares - trade_commission
                
                # Update trade record
                trades[-1].update({
                    'exit_idx': i,
                    'exit_price': price,
                    'profit': trade_profit,
                    'duration': i - entry_idx,
                })
                
                # Reset position
                position = 0
                shares = 0.0
        
        # Update equity curve
        if position == 0:
            equity_curve[i] = cash
        elif position == 1:
            equity_curve[i] = shares * price
        elif position == -1:
            equity_curve[i] = (2 * entry_price - price) * shares
    
    return equity_curve, trades