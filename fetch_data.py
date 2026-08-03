import yfinance as yf
import pandas as pd
import numpy as np

TICKER = "NVDA"

def get_stock_data(ticker, period="2y"):
    data = yf.download(ticker, period=period)
    return data

def calculate_volatility(data, window=21):
    # Daily returns
    data["Return"] = data["Close"].pct_change()

    # Rolling volatility (standard deviation of returns over the window)
    data["Volatility"] = data["Return"].rolling(window=window).std() * np.sqrt(252)

    return data

df = get_stock_data(TICKER)
df = calculate_volatility(df)

print(df[["Close", "Return", "Volatility"]].tail(10))

df.to_csv("stock_data.csv")
print(f"\nSaved {len(df)} rows to stock_data.csv")