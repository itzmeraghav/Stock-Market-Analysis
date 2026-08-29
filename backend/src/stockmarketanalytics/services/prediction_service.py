from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy.orm import Session
from stockmarketanalytics.constants import SUPPORTED_MODELS
from stockmarketanalytics.models.prediction import Prediction
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.models.technical_indicator import TechnicalIndicator
from stockmarketanalytics.schemas.prediction_schemas import (
    BacktestResult,
    ModelComparisonEntry,
    MultiModelPredictionResponse,
    PredictedDirection,
    PredictionResponse,
    PredictionTrainResult,
)

logger = logging.getLogger("prediction_service")

FEATURE_COLUMNS = [
    "close_to_sma20",
    "close_to_sma50",
    "close_to_ema20",
    "rsi14",
    "macd_norm",
    "volatility",
    "volume_change",
    "daily_return",
]

models_train = SUPPORTED_MODELS


def _build_sklearn_model(model_name: str):
    if model_name == "LinearRegression":
        return LinearRegression()
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
    elif model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=6, random_state=42
        )
    elif model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        )
    elif model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=-1,
        )
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Supported: {models_train}"
        )


class PredictionService:
    def __init__(self, db: Session, model_dir: Path | str = "models"):
        self.db = db
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _model_path(self, symbol: str, model_name: str) -> Path:
        return self.model_dir / f"{symbol}_{model_name}.joblib"

    def load_feature_frame(self, stock_id: int) -> pd.DataFrame:
        rows = (
            self.db.query(StockPrice, TechnicalIndicator)
            .join(
                TechnicalIndicator, TechnicalIndicator.stock_price_id == StockPrice.id
            )
            .filter(StockPrice.stock_id == stock_id)
            .order_by(StockPrice.trading_date.asc())
            .all()
        )
        if not rows:
            raise ValueError(
                f"No indicator data found for stock_id={stock_id}. Run indicator computation first."
            )

        records = []
        for price, indicator in rows:
            records.append(
                {
                    "trading_date": price.trading_date,
                    "close": price.close,
                    "volume": price.volume,
                    "sma20": indicator.sma20,
                    "sma50": indicator.sma50,
                    "ema20": indicator.ema20,
                    "rsi14": indicator.rsi14,
                    "macd": indicator.macd,
                    "volatility": indicator.volatility,
                }
            )

        df = pd.DataFrame(records)
        df["daily_return"] = df["close"].pct_change()
        df["volume_change"] = (
            df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
        df["close_to_sma20"] = df["close"] / df["sma20"] - 1
        df["close_to_sma50"] = df["close"] / df["sma50"] - 1
        df["close_to_ema20"] = df["close"] / df["ema20"] - 1
        df["macd_norm"] = df["macd"] / df["close"]
        df = df.dropna(subset=["daily_return"]).reset_index(drop=True)
        return df

    def build_training_set(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        df = df.copy()
        df["target_return"] = df["close"].shift(-1) / df["close"] - 1
        df = df.dropna(subset=FEATURE_COLUMNS + ["target_return"]).reset_index(
            drop=True
        )
        X = df[FEATURE_COLUMNS]
        y = df["target_return"]
        return X, y, df

    def _directional_accuracy(self, y_test: np.ndarray, preds: np.ndarray) -> float:
        return float(np.mean(np.sign(y_test) == np.sign(preds)) * 100)

    def train(
        self, stock: Stock, model_name: str = "LinearRegression"
    ) -> PredictionTrainResult:
        if model_name == "LSTM":
            return self._train_lstm(stock)
        return self._train_sklearn(stock, model_name)

    def train_all(self, stock: Stock) -> list[PredictionTrainResult]:
        results: list[PredictionTrainResult] = []
        for name in models_train:
            try:
                results.append(self.train(stock, model_name=name))
            except ValueError as exc:
                logger.warning(
                    "Skipping training %s for %s: %s", name, stock.symbol, exc
                )

        if not results:
            raise ValueError(
                f"Could not train any model for {stock.symbol}. Check that price/indicator data exists."
            )

        return results

    def _train_sklearn(self, stock: Stock, model_name: str) -> PredictionTrainResult:
        df = self.load_feature_frame(stock.id)
        X, y, df = self.build_training_set(df)

        if len(X) < 60:
            raise ValueError(
                "Not enough historical data to train a model (need at least 60 rows)"
            )

        splitter = TimeSeriesSplit(n_splits=5)
        maes, rmses, mapes, dir_accs = [], [], [], []

        close_series = df["close"]

        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            close_test = close_series.iloc[test_idx]

            model = _build_sklearn_model(model_name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            actual_price = close_test.values * (1 + y_test.values)
            predicted_price = close_test.values * (1 + preds)

            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            mapes.append(
                float(
                    np.mean(np.abs((actual_price - predicted_price) / actual_price))
                    * 100
                )
            )
            dir_accs.append(self._directional_accuracy(y_test.values, preds))

        final_model = _build_sklearn_model(model_name)
        final_model.fit(X, y)
        joblib.dump(final_model, self._model_path(stock.symbol, model_name))

        logger.info(
            "Trained %s for %s: MAE=%.6f RMSE=%.6f DirAcc=%.2f%%",
            model_name,
            stock.symbol,
            np.mean(maes),
            np.mean(rmses),
            np.mean(dir_accs),
        )

        return PredictionTrainResult(
            symbol=stock.symbol,
            model_name=model_name,
            mae=float(np.mean(maes)),
            rmse=float(np.mean(rmses)),
            mape=float(np.mean(mapes)),
            directional_accuracy=float(np.mean(dir_accs)),
            trained_at=datetime.now(UTC),
        )

    def predict(
        self, stock: Stock, model_name: str = "LinearRegression", persist: bool = False
    ) -> PredictionResponse:
        if model_name == "LSTM":
            return self._predict_lstm(stock, persist=persist)
        return self._predict_sklearn(stock, model_name, persist=persist)

    def predict_all(
        self, stock: Stock, persist: bool = False
    ) -> MultiModelPredictionResponse:
        predictions: dict[str, PredictionResponse] = {}
        for name in models_train:
            try:
                predictions[name] = self.predict(
                    stock, model_name=name, persist=persist
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping prediction for %s (%s): %s", stock.symbol, name, exc
                )

        if not predictions:
            raise ValueError(
                f"No trained models available for {stock.symbol}. Train models first via POST /predictions/train/{stock.symbol}."
            )

        comparison = self.compare_models(stock, model_names=list(predictions.keys()))
        best_model_name = (
            comparison[0].model_name if comparison else next(iter(predictions))
        )
        best_model_rmse = comparison[0].rmse if comparison else None

        return MultiModelPredictionResponse(
            symbol=stock.symbol,
            predictions=list(predictions.values()),
            best_model=best_model_name,
            best_model_rmse=best_model_rmse,
        )

    def _predict_sklearn(
        self, stock: Stock, model_name: str, persist: bool
    ) -> PredictionResponse:
        model_path = self._model_path(stock.symbol, model_name)
        if not model_path.exists():
            raise ValueError(
                f"No trained {model_name} model found for {stock.symbol}. Train it first."
            )

        model = joblib.load(model_path)
        df = self.load_feature_frame(stock.id)
        if df.empty:
            raise ValueError(f"No feature data available for {stock.symbol}")

        latest = df.iloc[-1]
        current_price = float(latest["close"])
        X_latest = latest[FEATURE_COLUMNS].to_frame().T
        predicted_return = float(model.predict(X_latest)[0])
        predicted_close = current_price * (1 + predicted_return)

        return self._finalize_prediction(
            stock,
            current_price,
            predicted_close,
            model_name,
            latest["trading_date"],
            persist,
        )

    def _finalize_prediction(
        self,
        stock: Stock,
        current_price: float,
        predicted_close: float,
        model_name: str,
        prediction_date,
        persist: bool,
    ) -> PredictionResponse:
        expected_change_percent = (
            (predicted_close - current_price) / current_price
        ) * 100

        if expected_change_percent > 0.15:
            direction = PredictedDirection.BULLISH
        elif expected_change_percent < -0.15:
            direction = PredictedDirection.BEARISH
        else:
            direction = PredictedDirection.FLAT

        confidence = min(0.99, max(0.5, 1 - abs(expected_change_percent) / 10))

        if persist:
            target_date = prediction_date + timedelta(days=1)
            record = Prediction(
                stock_id=stock.id,
                prediction_date=prediction_date,
                target_date=target_date,
                actual_close=None,
                predicted_close=predicted_close,
                predicted_direction=direction.value,
                confidence=confidence,
                model_name=model_name,
                created_at=datetime.now(UTC),
            )
            self.db.add(record)
            self.db.commit()

        return PredictionResponse(
            symbol=stock.symbol,
            current_price=current_price,
            predicted_price=predicted_close,
            expected_change_percent=expected_change_percent,
            direction=direction,
            model=model_name,
        )

    def backtest(
        self,
        stock: Stock,
        model_name: str = "LinearRegression",
        test_ratio: float = 0.2,
    ) -> BacktestResult:
        return self._backtest_sklearn(stock, model_name, test_ratio)

    def _backtest_sklearn(
        self, stock: Stock, model_name: str, test_ratio: float
    ) -> BacktestResult:
        df = self.load_feature_frame(stock.id)
        X, y, df = self.build_training_set(df)

        if len(X) < 60:
            raise ValueError(
                "Not enough historical data to backtest (need at least 60 rows)"
            )

        split_index = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        test_dates = df["trading_date"].iloc[split_index:]
        test_close = df["close"].iloc[split_index:]

        model = _build_sklearn_model(model_name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        actual_price = test_close.values * (1 + y_test.values)
        predicted_price = test_close.values * (1 + preds)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mape = float(
            np.mean(np.abs((actual_price - predicted_price) / actual_price)) * 100
        )
        directional_accuracy = self._directional_accuracy(y_test.values, preds)

        actual_close_series = actual_price.tolist()
        predicted_close_series = predicted_price.tolist()

        return BacktestResult(
            symbol=stock.symbol,
            model_name=model_name,
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=directional_accuracy,
            actual_series=[float(v) for v in actual_close_series],
            predicted_series=[float(v) for v in predicted_close_series],
            dates=list(test_dates),
        )

    def compare_models(
        self, stock: Stock, model_names: list[str] | None = None
    ) -> list[ModelComparisonEntry]:
        if model_names is None:
            model_names = models_train

        results: list[ModelComparisonEntry] = []
        for name in model_names:
            if name not in models_train:
                logger.warning("Skipping unsupported model in comparison: %s", name)
                continue

            start = time.perf_counter()
            try:
                bt = self.backtest(stock, model_name=name)
            except ValueError as exc:
                logger.warning(
                    "Skipping %s in comparison for %s: %s", name, stock.symbol, exc
                )
                continue
            elapsed_seconds = time.perf_counter() - start

            results.append(
                ModelComparisonEntry(
                    model_name=name,
                    mae=bt.mae,
                    rmse=bt.rmse,
                    mape=bt.mape,
                    directional_accuracy=bt.directional_accuracy,
                    training_time_seconds=round(elapsed_seconds, 3),
                )
            )

        results.sort(key=lambda r: r.rmse)
        return results

    def reconcile_actuals(self, stock: Stock) -> int:
        pending = (
            self.db.query(Prediction)
            .filter(Prediction.stock_id == stock.id, Prediction.actual_close.is_(None))
            .all()
        )
        if not pending:
            return 0

        updated = 0
        for prediction in pending:
            price_row = (
                self.db.query(StockPrice)
                .filter(
                    StockPrice.stock_id == stock.id,
                    StockPrice.trading_date == prediction.target_date,
                )
                .first()
            )
            if price_row is not None:
                prediction.actual_close = price_row.close
                updated += 1

        if updated:
            self.db.commit()
            logger.info(
                "Reconciled %d actual_close values for %s", updated, stock.symbol
            )

        return updated
