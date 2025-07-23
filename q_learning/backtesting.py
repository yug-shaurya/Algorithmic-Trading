import yfinance as yf
import pandas as pd
import numpy as np

# Download data
stock = 'INFY.NS'
data = yf.download(stock, start='2020-01-01', end='2024-12-31', auto_adjust=True)

# Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# RSI Calculation
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Entry/Exit Signals
data['Signal'] = 0
data.loc[(data['RSI'] < 30) & (data['MA50'] > data['MA200']), 'Signal'] = 1
data.loc[(data['RSI'] > 70) | (data['MA50'] < data['MA200']), 'Signal'] = -1

# Backtest Logic
capital = 10_00_000
position = 0.0  # force it to be float
buy_price = 0.0
results = []

for i in range(1, len(data)):
    signal = data['Signal'].iloc[i]
    close_price = data['Close'].iloc[i]

    # Use float() to prevent it from becoming Series
    if signal == 1 and float(position) == 0.0:
        buy_price = float(close_price)
        position = float(capital / close_price)
        buy_date = data.index[i]

    elif signal == -1 and float(position) > 0.0:
        sell_price = float(close_price)
        sell_date = data.index[i]
        capital = float(position * sell_price)
        profit = capital - 10_00_000
        results.append({
            'Buy_Date': buy_date,
            'Sell_Date': sell_date,
            'Buy_Price': buy_price,
            'Sell_Price': sell_price,
            'Return': profit
        })
        position = 0.0

# Convert to DataFrame
results_df = pd.DataFrame(results)
returns = results_df['Return'] / 10_00_000

# Metrics
total_trades = len(results_df)
avg_return = results_df['Return'].mean()
avg_win = results_df[results_df['Return'] > 0]['Return'].mean()
avg_loss = results_df[results_df['Return'] <= 0]['Return'].mean()
win_rate = (results_df['Return'] > 0).mean() * 100
win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

# Output
print("\n--- Backtest Summary ---")
print(f"Total Trades: {total_trades}")
print(f"Average Return per Trade: ₹{avg_return:.2f}")
print(f"Avg Winning Trade: ₹{avg_win:.2f}")
print(f"Avg Losing Trade: ₹{avg_loss:.2f}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
