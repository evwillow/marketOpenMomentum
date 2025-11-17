"""
Shared utilities for the UPRO/SPXU momentum research workflow.

The functions here are intentionally lightweight so the notebooks load fast.
All expensive work (data downloads, feature generation, model training, backtests)
should be executed from the notebooks that import this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from yahooquery import Ticker

from dotenv import load_dotenv


DEFAULT_DATA_DIR = Path("data")
DEFAULT_PRICES_FILE = DEFAULT_DATA_DIR / "prices.parquet"
DEFAULT_FEATURE_FILE = DEFAULT_DATA_DIR / "features.parquet"


def bootstrap_env(dotenv_path: Path = Path(".env")) -> None:
    """Load environment variables once at startup."""
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


def load_prices(
    symbols: Tuple[str, str] = ("UPRO", "SPXU"),
    period: str = "30d",
    interval: str = "5m",
    cache_file: Path = DEFAULT_PRICES_FILE,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download recent prices via yfinance and cache them.
    """
    DEFAULT_DATA_DIR.mkdir(exist_ok=True)
    if cache_file.exists() and not force:
        return pd.read_parquet(cache_file)

    tk = Ticker(" ".join(symbols))
    hist = tk.history(period=period, interval=interval)
    if hist.empty:
        raise RuntimeError("No historical data returned from Yahoo Finance.")
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index()
    hist = hist.rename(columns={"date": "ts", "symbol": "Symbol", "close": "close"})
    pivot = hist.pivot(index="ts", columns="Symbol", values="close")
    pivot.columns = [c.upper() for c in pivot.columns]
    prices = pivot.dropna().sort_index()
    if prices.index.tzinfo is None:
        prices.index = prices.index.tz_localize("UTC")
    prices.index = prices.index.tz_convert("America/New_York")
    prices.to_parquet(cache_file)
    return prices


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices[~prices.index.duplicated()].sort_index()
    prices = prices.dropna(how="any")
    return prices


def compute_features(
    price_df: pd.DataFrame, window_short: int = 5, window_long: int = 20
) -> pd.DataFrame:
    feats = {}
    for sym in price_df.columns:
        close = price_df[sym]
        feats[f"{sym}_ret_1"] = close.pct_change()
        feats[f"{sym}_ema_short"] = close.ewm(span=window_short, adjust=False).mean()
        feats[f"{sym}_ema_long"] = close.ewm(span=window_long, adjust=False).mean()
        feats[f"{sym}_ema_ratio"] = (
            feats[f"{sym}_ema_short"] / feats[f"{sym}_ema_long"]
        )
        feats[f"{sym}_vol_10"] = close.pct_change().rolling(10).std(ddof=0)
        feats[f"{sym}_rsi_14"] = _rsi(close, period=14)
    frame = pd.DataFrame(feats).dropna()
    return frame


def label_future_returns(price_df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Create a classification target: +1 if UPRO outperforms SPXU over the horizon, else 0.
    """
    upro = price_df["UPRO"].pct_change(periods=horizon)
    spxu = price_df["SPXU"].pct_change(periods=horizon)
    spread = upro - spxu
    labels = (spread > 0).astype(int)
    return labels.loc[labels.index.intersection(price_df.index)]


def train_model(X: pd.DataFrame, y: pd.Series):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_features)]
    )

    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                XGBClassifier(
                    max_depth=4,
                    learning_rate=0.05,
                    n_estimators=300,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.2,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="auc",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    importances = None
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.DataFrame(
            {"feature": numeric_features, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)

    return model, report, importances


def generate_signals(model, X: pd.DataFrame, threshold: float = 0.55) -> pd.Series:
    proba = model.predict_proba(X)[:, 1]
    signals = pd.Series(0, index=X.index, dtype=int)
    signals[proba >= threshold] = 1
    signals[proba <= 1 - threshold] = -1
    return signals


def backtest_signals(
    price_df: pd.DataFrame,
    signals: pd.Series,
    cash: float = 100_000.0,
    trading_cost_bps: float = 1.0,
) -> pd.DataFrame:
    aligned_prices = price_df.loc[signals.index]
    base_ret = aligned_prices["UPRO"].pct_change().fillna(0) - aligned_prices[
        "SPXU"
    ].pct_change().fillna(0)
    strat_returns = base_ret * signals.shift().fillna(0)
    costs = np.abs(signals.diff().fillna(0)) * (trading_cost_bps / 10_000)
    strat_returns = strat_returns - costs
    equity = (1 + strat_returns).cumprod() * cash
    stats = _performance_stats(strat_returns)
    out = pd.DataFrame({"equity": equity, "strategy_return": strat_returns})
    out.attrs["stats"] = stats
    return out


def _performance_stats(returns: pd.Series) -> Dict[str, float]:
    periods = len(returns)
    if periods == 0:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
    freq_minutes = _infer_minutes(returns.index)
    ann_periods = int((252 * 6.5 * 60) / freq_minutes)
    mean = returns.mean()
    std = returns.std(ddof=0)
    sharpe = (mean * ann_periods) / (std * np.sqrt(ann_periods)) if std > 0 else 0.0
    cagr = (1 + returns).prod() ** (ann_periods / periods) - 1
    curve = (1 + returns).cumprod()
    max_drawdown = ((curve.cummax() - curve) / curve.cummax()).max()
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": -float(max_drawdown)}


def _infer_minutes(index: pd.Index) -> int:
    if not isinstance(index, pd.DatetimeIndex) or index.size < 2:
        return 1
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 1
    median_delta = deltas.median().total_seconds() / 60
    return max(1, int(round(median_delta)))


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


__all__ = [
    "bootstrap_env",
    "load_prices",
    "validate_prices",
    "compute_features",
    "label_future_returns",
    "train_model",
    "generate_signals",
    "backtest_signals",
]


