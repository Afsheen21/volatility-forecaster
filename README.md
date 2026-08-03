# Volatility Forecaster 📈

![CI](https://github.com/Afsheen21/volatility-forecaster/actions/workflows/ci.yml/badge.svg)

A machine learning system that predicts next-day stock volatility, served as a live API with automated testing and containerization.

## Overview

This project pulls historical stock data, engineers time-series features, and trains models to forecast next-day volatility. The trained model is served through a FastAPI REST API that works for any publicly traded stock ticker, backed by automated tests and a CI pipeline.

## Features

- Historical stock data collection via yfinance
- Feature engineering: moving averages, lagged volatility, return lags, volume change
- Model comparison: Linear Regression vs Random Forest, evaluated with time-respecting train/test splits (no data leakage)
- Live prediction API — request any stock ticker, get a real-time volatility forecast
- Automated tests covering the home endpoint, valid predictions, and error handling
- CI pipeline (GitHub Actions) that runs tests on every push
- Dockerized for portable deployment

## Tech Stack

Python, yfinance, pandas, scikit-learn, FastAPI, Uvicorn, pytest, Docker, GitHub Actions

## How It Works

1. **`fetch_data.py`** — downloads historical price data and calculates rolling annualized volatility
2. **`train_model.py`** — engineers features (moving averages, lagged volatility, volume change), trains and compares two models using a chronological (not random) train/test split, and saves the best model
3. **`main.py`** — FastAPI service that loads the trained model and exposes a `/predict/{ticker}` endpoint, recalculating live features for any requested stock
4. **`test_main.py`** — automated tests verifying the API's core behavior
5. **`.github/workflows/ci.yml`** — runs the test suite automatically on every push

## Model Performance

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 0.025 | 0.43 |
| Random Forest | 0.019 | 0.67 |

The Random Forest model was selected for deployment. Feature importance analysis showed `Volatility_lag1` (yesterday's volatility) as the dominant predictor — consistent with the well-documented "volatility clustering" phenomenon in financial markets.

## Running It Locally

**1. Clone and set up**
```bash
git clone https://github.com/Afsheen21/volatility-forecaster.git
cd volatility-forecaster
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

