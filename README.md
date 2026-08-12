# Volatility Forecaster 📈
![API Demo](<Image demo.png>))

![CI](https://github.com/Afsheen21/volatility-forecaster/actions/workflows/ci.yml/badge.svg)

A machine learning system that predicts next-day stock volatility, served as a live API with automated testing, CI, and containerized deployment.

## Overview

This project pulls historical stock data, engineers time-series features, and trains models to forecast next-day volatility. The trained model is served through a FastAPI REST API that works for any publicly traded stock ticker, backed by automated tests, a CI pipeline, and a Docker container ready for deployment.

## Features

- Historical stock data collection via yfinance
- Feature engineering: moving averages, lagged volatility, return lags, volume change
- Model comparison: Linear Regression vs Random Forest, evaluated with time-respecting train/test splits (no data leakage)
- Live prediction API — request any stock ticker, get a real-time volatility forecast
- Automated tests covering the home endpoint, valid predictions, and error handling
- CI pipeline (GitHub Actions) that runs tests on every push
- Fully containerized with Docker, built and verified working end-to-end

## Tech Stack

Python, yfinance, pandas, scikit-learn, FastAPI, Uvicorn, pytest, Docker, GitHub Actions

## How It Works

1. **`fetch_data.py`** — downloads historical price data and calculates rolling annualized volatility
2. **`train_model.py`** — engineers features (moving averages, lagged volatility, volume change), trains and compares two models using a chronological (not random) train/test split, and saves the best model
3. **`main.py`** — FastAPI service that loads the trained model and exposes a `/predict/{ticker}` endpoint, recalculating live features for any requested stock
4. **`test_main.py`** — automated tests verifying the API's core behavior
5. **`.github/workflows/ci.yml`** — runs the test suite automatically on every push
6. **`Dockerfile`** — packages the entire application so it runs identically anywhere

## Model Performance

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 0.025 | 0.43 |
| Random Forest | 0.019 | 0.67 |

The Random Forest model was selected for deployment. Feature importance analysis showed `Volatility_lag1` (yesterday's volatility) as the dominant predictor — consistent with the well-documented "volatility clustering" phenomenon in financial markets.

## Example Request

```
GET /predict/AAPL
```

```json
{
  "ticker": "AAPL",
  "predicted_next_day_volatility": 0.3819,
  "as_of_date": "2026-08-06"
}
```

## Running It Locally

**1. Clone and set up**
```bash
git clone https://github.com/Afsheen21/volatility-forecaster.git
cd volatility-forecaster
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Collect data and train the model**
```bash
python fetch_data.py
python train_model.py
```

**3. Run the API**
```bash
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

**4. Run tests**
```bash
pytest test_main.py -v
```

**5. Run with Docker**
```bash
docker build -t volatility-forecaster .
docker run -p 8000:8000 volatility-forecaster
```
Then visit `http://localhost:8000/docs`

## Future Improvements

- Add ARIMA/classical time-series methods for comparison against the ML approach
- Add more granular volatility windows (e.g. 5-day, 60-day)
- Expand test coverage to include model performance regression checks
- Add a simple frontend for non-technical users to query predictions

## Author

Built by [Afsheen Fatima](https://github.com/Afsheen21) as a portfolio project demonstrating end-to-end ML engineering — from data collection through model deployment, automated testing, and containerization.

