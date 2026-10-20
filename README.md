# Volatility Forecaster 📈
<img src="Image%20demo.png" alt="API demo" width="600">

![CI](https://github.com/Afsheen21/volatility-forecaster/actions/workflows/ci.yml/badge.svg)

An end-to-end ML engineering project: it pulls stock data, engineers time-series features, trains models to forecast volatility, and serves the result as a live API with automated tests, CI, and a Docker container.

> **Result in one line:** the Random Forest reaches R² = 0.67, but a naive "tomorrow's volatility = today's" rule scores 0.86 on the same test set, so the model does not beat a trivial baseline. The engineering pipeline (API, tests, CI, Docker) is the contribution of this project, not the forecast. Details and the cause are in [Baseline check](#baseline-check-added).

## Overview

This project pulls historical stock data, engineers time-series features, and trains models to forecast next-day volatility, defined as the annualised 21-day rolling standard deviation of daily returns. The trained model is served through a FastAPI REST API that accepts any publicly traded stock ticker, backed by automated tests, a CI pipeline, and a Docker container ready for deployment. The model itself was trained on NVDA only, and its accuracy on other tickers has not been tested (see Limitations).

## Features

- Historical stock data collection via yfinance
- Feature engineering: moving averages, lagged volatility, return lags, volume change
- Model comparison: Linear Regression vs Random Forest, evaluated with a chronological (not random) 80/20 train/test split
- Baseline check showing how the models compare with a naive "tomorrow = today" rule
- Live prediction API: request any stock ticker, get a volatility forecast
- Automated tests covering the home endpoint, valid predictions, and error handling
- CI pipeline (GitHub Actions) that runs tests on every push
- Fully containerized with Docker, built and verified working end-to-end

## Tech Stack

Python, yfinance, pandas, scikit-learn, FastAPI, Uvicorn, pytest, Docker, GitHub Actions

## How It Works

1. **`fetch_data.py`**: downloads historical price data and calculates rolling annualized volatility
2. **`train_model.py`**: engineers features (moving averages, lagged volatility, volume change), trains and compares two models using a chronological train/test split, and saves the model
3. **`baseline_check.py`**: re-runs the same split and adds baselines (see below). Running it retrains and re-saves `volatility_model.pkl`
4. **`main.py`**: FastAPI service that loads the trained model and exposes a `/predict/{ticker}` endpoint, recalculating live features for any requested stock
5. **`test_main.py`**: automated tests verifying the API's core behavior
6. **`.github/workflows/ci.yml`**: runs the test suite automatically on every push
7. **`Dockerfile`**: packages the entire application so it runs identically anywhere

## Model Performance (original evaluation)

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 0.025 | 0.43 |
| Random Forest | 0.019 | 0.67 |

These numbers use a chronological 80/20 split. They look strong, but they are **not enough on their own**, because they were reported without a baseline. The next section adds one.

## Baseline check (added)

The target is tomorrow's 21-day rolling volatility. Tomorrow's window and today's window share 20 of their 21 daily returns, so the target is almost a copy of the model's own inputs, and a model can score a high R² just by carrying today's value forward. `baseline_check.py` tests this on the same split:

| Model (same test set) | R² |
|---|---|
| **Persistence: tomorrow = today (no model)** | **0.859** |
| Random Forest | 0.665 |
| Linear Regression | 0.426 |
| Random Forest, on the *change* in volatility | -1.38 |
| Linear Regression, on the *change* in volatility | -3.08 |

**Reading this honestly:**
- The Random Forest's R² of 0.67 is real, but a rule with no model at all scores higher (0.86). On this target, the model does not beat the simplest baseline.
- On the *change* in volatility, which is the hard part, both models are worse than predicting no change (negative R²).
- "Negative R²" means worse than predicting that volatility will not change.
- The dominant feature, `Volatility_lag1` (yesterday's volatility), is consistent with volatility clustering, but here it also reflects the window overlap described above.

**A separate test with a stricter setup.** In my earnings-call study repository (`study/04_volatility_baselines.py` in [Earnings-call-sentiment-analysis](https://github.com/Afsheen21/Earnings-call-sentiment-analysis)), I used a non-overlapping target (realised volatility over the *next 5 days*), a time-ordered 70/30 split with a gap, and a HAR regression as the standard finance benchmark. NVDA, 2015-2025:

| Model | R² |
|---|---|
| Persistence (last week's volatility) | -0.553 |
| HAR regression | 0.049 |
| Random Forest | 0.027 |

With a random shuffle split, which leaks information between neighbouring days, the same Random Forest reaches 0.20. Even with a clean target, the Random Forest does not beat HAR.

**What this project does demonstrate:** the engineering pipeline (data collection, API, tests, CI, Docker). **What it does not demonstrate:** a forecasting model that beats a simple baseline.

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

The value is the predicted annualised 21-day rolling volatility for the next day.

## Limitations

- **The forecast does not beat a naive baseline** (see Baseline check), so the live endpoint should be treated as a demonstration of the serving pipeline, not as a usable forecast.
- **Trained on NVDA only.** The API accepts any ticker, but accuracy on other stocks has not been tested.
- **The target overlaps its inputs** (21-day rolling windows), which inflates R².
- **No uncertainty estimate**: the API returns a single point value.

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
python baseline_check.py
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

- Predict the *change* in volatility, or a non-overlapping horizon (for example the next 5 days), so the target is not a copy of the inputs
- Add HAR and persistence baselines to `train_model.py` and report them next to every model
- Add ARIMA/GARCH-type methods for comparison against the ML approach
- Add a model performance regression test (fail CI if a model does not beat the baseline)
- Add a simple frontend for non-technical users to query predictions

## Author

Built by [Afsheen Fatima](https://github.com/Afsheen21) as a portfolio project demonstrating end-to-end ML engineering: data collection, model deployment, automated testing, and containerization, plus an honest evaluation of where the model falls short.
