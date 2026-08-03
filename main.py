from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import yfinance as yf

app = FastAPI(title="Volatility Forecaster API")

model = joblib.load("volatility_model.pkl")

@app.get("/")
def home():
    return {"message": "Volatility Forecaster API is running. Use /predict/{ticker} to get a forecast."}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    data = yf.download(ticker, period="3mo")

    if data.empty:
        return {"error": f"No data found for ticker '{ticker}'"}

    data.columns = data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else data.columns

    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(21).std() * np.sqrt(252)
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Volatility_lag1"] = data["Volatility"].shift(1)
    data["Volatility_lag5"] = data["Volatility"].shift(5)
    data["Return_lag1"] = data["Return"].shift(1)
    data["Volume_change"] = data["Volume"].pct_change()

    latest = data.dropna().iloc[-1:]

    if latest.empty:
        return {"error": "Not enough data to make a prediction"}

    features = ["MA_5", "MA_20", "Volatility_lag1", "Volatility_lag5", "Return_lag1", "Volume_change"]
    X = latest[features]

    prediction = model.predict(X)[0]

    return {
        "ticker": ticker.upper(),
        "predicted_next_day_volatility": round(float(prediction), 4),
        "as_of_date": str(latest.index[0].date())
    }