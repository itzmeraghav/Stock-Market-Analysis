# Stock Market Analytics API

A FastAPI backend for stock market analysis — combining historical market data ingestion, technical indicators, multi-model price prediction, Black-Scholes option pricing, and trade risk analysis behind a JWT-authenticated REST API.

## Features

- **Market data ingestion** — pulls historical OHLCV data from Yahoo Finance (`yfinance`) or CSV import, with validation and deduplication before persisting.
- **Technical indicators** — computes indicators from price history and stores them per stock.
- **Price prediction** — trains and compares multiple regression models (Linear Regression, Random Forest, HistGradientBoosting, LightGBM, Extra Trees) on a stock's price/indicator history, with backtesting and actuals reconciliation.
- **Options pricing** — Black-Scholes call/put pricing with Greeks (delta, gamma, vega, theta, rho), plus a forecast endpoint that chains prediction + historical volatility + option pricing.
- **Trade risk analysis** — position sizing and risk/target calculations based on investment amount and risk tolerance.
- **Authentication** — JWT access/refresh token pairs, Argon2 password hashing, refresh-token rotation, and account lockout after repeated failed logins.
- **Rate limiting** — per-route limits on auth endpoints plus a global authenticated/unauthenticated rate-limiting middleware.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Database | SQLAlchemy ORM + Alembic migrations |
| ML / Prediction | scikit-learn, LightGBM, NumPy, pandas |
| Options math | SciPy (Black-Scholes) |
| Auth | PyJWT, Argon2 (`argon2-cffi`) |
| Rate limiting | slowapi + `limits` |
| Market data | yfinance |
| Testing | pytest, pytest-cov |
| Tooling | black, ruff, mypy, pip-audit |
| Containerization | Docker / Docker Compose |

## Project Structure

```
backend/
├── src/stockmarketanalytics/
│   ├── main.py                  # FastAPI app, routers, startup hook
│   ├── settings.py              # Environment-driven configuration
│   ├── constants.py             # Supported prediction models
│   ├── rate_limiter.py          # Global + per-route rate limiting
│   ├── auth_dependencies.py     # Bearer-token auth dependencies
│   ├── data/                    # DB session/context and initializer
│   ├── models/                  # SQLAlchemy ORM models
│   ├── schemas/                 # Pydantic request/response schemas
│   ├── services/                # Business logic (auth, market data,
│   │                             #  indicators, prediction, options,
│   │                             #  volatility, risk management)
│   └── endpoints/                # API routers
├── alembic/                     # DB migrations
├── tests/                       # pytest suite
├── DockerFile
├── docker-compose.yaml
├── requirements.txt
└── pyproject.toml
```

## Getting Started

### Prerequisites

- Python 3.11+
- A database reachable via `DATABASE_URL` (SQLAlchemy-compatible)

### Local Setup

```bash
# Clone and enter the backend directory
git clone https://github.com/itzmeraghav/Stock-Market-Analysis.git
cd Stock-Market-Analysis/backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env           # then fill in the values

# Apply database migrations
alembic upgrade head

# Run the API
python -m uvicorn stockmarketanalytics.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Running with Docker

```bash
cd backend
cp .env.example .env           # then fill in the values
docker-compose up --build
```

This builds the image from `DockerFile` and starts the API on port `8000`.

## Environment Variables

Configured via a `.env` file (see `.env.example`):

| Variable | Description |
|---|---|
| `DATABASE_URL` | SQLAlchemy database connection string |
| `APP_NAME` | Application name |
| `DEBUG` | Debug mode flag |
| `JWT_SECRET_KEY` | Secret key for signing JWTs |
| `JWT_ALGORITHM` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime |
| `LOGIN_RATE_LIMIT_PER_IP` | Rate limit for the login endpoint, per IP |
| `LOGIN_MAX_FAILED_ATTEMPTS` | Failed logins before account lockout |
| `LOGIN_LOCKOUT_MINUTES` | Lockout duration after max failed attempts |
| `REFRESH_MIN_INTERVAL_SECONDS` | Minimum interval between token refreshes |
| `TOKEN_GLOBAL_CAP_PER_HOUR` | Global hourly cap on refresh calls |
| `RATE_LIMIT_AUTHENTICATED` | Global rate limit for authenticated requests |
| `RATE_LIMIT_UNAUTHENTICATED` | Global rate limit for unauthenticated requests |

## API Overview

All routes are prefixed with `/api`. Every router except `/api/auth` and `/api/health` requires a valid `Authorization: Bearer <access_token>` header.

| Router | Prefix | Purpose |
|---|---|---|
| Health | `/api/health` | Liveness check |
| Auth | `/api/auth` | Register, login, refresh, logout |
| Stocks | `/api/stocks` | List/fetch stocks, price history, market data updates, CSV import |
| Indicators | `/api/indicators` | Fetch and compute technical indicators |
| Predictions | `/api/predictions` | Train models, generate/persist predictions, reconcile actuals, compare models |
| Backtest | `/api/backtest` | Backtest a prediction model against historical data |
| Options | `/api/options` | Black-Scholes calculation, prediction-driven option forecast, calculation history |
| Trading | `/api/trading` | Position sizing and trade risk analysis |

Full request/response schemas are available via the auto-generated OpenAPI docs at `/docs` once the server is running.

## Testing

```bash
pytest --cov=stockmarketanalytics
```

Test suite covers endpoints, services, schemas, and models under `tests/`.

## Code Quality

```bash
black .          # formatting
ruff check .     # linting
mypy .           # type checking
pip-audit        # dependency vulnerability scan
```

## Contributors

- [goutam-tech](https://github.com/goutam-tech)
- [itzmeraghav](https://github.com/itzmeraghav)
- [chetankumar72](https://github.com/chetankumar72)