# Stock Market Analytics

A full-stack stock market analysis platform — combining historical market data ingestion, technical indicators, multi-model price prediction, Black-Scholes option pricing, and trade risk analysis behind a JWT-authenticated REST API, with a frontend for exploring the data and insights.

## Overview

Stock Market Analytics lets you track stocks, compute technical indicators, train and compare machine learning models to forecast prices, price options using Black-Scholes, and analyze trade risk — all through a secured API with a dedicated frontend client.

## Features

- **Market data ingestion** — historical OHLCV data from Yahoo Finance or CSV import
- **Technical indicators** — computed and stored per stock from price history
- **Price prediction** — multiple regression models (Linear Regression, Random Forest, HistGradientBoosting, LightGBM, Extra Trees) with backtesting and model comparison
- **Options pricing** — Black-Scholes call/put pricing with Greeks, plus prediction-driven forecasts
- **Trade risk analysis** — position sizing and risk/target calculations
- **Authentication** — JWT access/refresh tokens, Argon2 password hashing, rate limiting

## Project Structure

```
Stock-Market-Analysis/
├── backend/     # FastAPI backend — API, database, ML models
└── frontend/    # Frontend client
```

## Getting Started

Each part of the project has its own setup instructions and documentation:

### Backend

FastAPI service powering data ingestion, predictions, options pricing, and auth.

- You can read this file [LINK](backend/README.md)

### Frontend

Client application for browsing stocks, indicators, and predictions.

- You can read this file [LINK](frontend/README.md)

### Report

For the reprot and docx for this project.

- You can read this file [LINK](report/final.docx)

## Contributors

- [goutam-tech](https://github.com/goutam-tech)
- [itzmeraghav](https://github.com/itzmeraghav)
- [chetankumar72](https://github.com/chetankumar72)