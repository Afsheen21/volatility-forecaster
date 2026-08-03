import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("stock_data.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
df.columns = ["Close", "High", "Low", "Open", "Volume", "Return", "Volatility"]

# --- Feature engineering ---
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility_lag1"] = df["Volatility"].shift(1)
df["Volatility_lag5"] = df["Volatility"].shift(5)
df["Return_lag1"] = df["Return"].shift(1)
df["Volume_change"] = df["Volume"].pct_change()

# Target: predict tomorrow's volatility
df["Target"] = df["Volatility"].shift(-1)

df = df.dropna()

features = ["MA_5", "MA_20", "Volatility_lag1", "Volatility_lag5", "Return_lag1", "Volume_change"]
X = df[features]
y = df["Target"]

# --- Time-based train/test split (NOT random!) ---
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# --- Train two models to compare ---
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Linear Regression — MAE:", mean_absolute_error(y_test, lr_preds), "| R²:", r2_score(y_test, lr_preds))
print("Random Forest     — MAE:", mean_absolute_error(y_test, rf_preds), "| R²:", r2_score(y_test, rf_preds))

print("\nFeature importance (Random Forest):")
for feat, importance in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {importance:.3f}")

import joblib

joblib.dump(rf, "volatility_model.pkl")
print("\nModel saved to volatility_model.pkl")