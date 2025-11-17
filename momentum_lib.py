"""
Shared utilities for the UPRO/SPXU momentum research workflow.

The functions here are intentionally lightweight so the notebooks load fast.
All expensive work (data downloads, feature generation, model training, backtests)
should be executed from the notebooks that import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

try:
    from dotenv import load_dotenv
except ImportError as e:  # pragma: no cover - handled in notebooks
    raise RuntimeError(
        "python-dotenv is required. Install dependencies via `pip install -r requirements.txt`."
    ) from e


DEFAULT_DATA_DIR = Path("data")
DEFAULT_FEATURE_FILE = DEFAULT_DATA_DIR / "uprx_dataset.parquet"


def bootstrap_env(dotenv_path: Path = Path(".env")) -> None:
    """Load environment variables once at startup."""
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


@dataclass
class AlpacaDataClient:
    api: "tradeapi.REST"
    symbols: Tuple[str, str] = ("UPRO", "SPXU")

    def fetch_minutes(
        self,
        days: int = 60,
        tz: str = "America/New_York",
        cache_file: Path = DEFAULT_DATA_DIR / "minute_bars.parquet",
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download minute bars for the configured symbols.
        Data is cached so notebooks can reuse the same frame instantly.
        """
        if cache_file.exists() and not force:
            return pd.read_parquet(cache_file)

        import alpaca_trade_api as tradeapi  # local import for speed

        if not isinstance(self.api, tradeapi.REST):
            raise TypeError("api must be an instance of tradeapi.REST")

        end = pd.Timestamp.now(tz=tz)
        start = end - pd.Timedelta(days=days)
        data_frames = []
        for sym in self.symbols:
            bars = self.api.get_bars(
                sym,
                timeframe="1Min",
                start=start.isoformat(),
                end=end.isoformat(),
                limit=10_000,
            ).df
            bars = bars.tz_convert(tz).reset_index()
            bars["symbol"] = sym
            data_frames.append(bars)

        df = pd.concat(data_frames, ignore_index=True)
        df = df.rename(columns={"timestamp": "ts"})
        df.to_parquet(cache_file, index=False)
        return df


def pivot_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a stacked Alpaca frame to a multi-column pivoted frame indexed by timestamp.
    """
    wide = (
        raw.pivot_table(index="ts", columns="symbol", values="close")
        .dropna()
        .sort_index()
    )
    wide.columns = pd.Index([str(c).upper() for c in wide.columns], name="symbol")
    return wide


def compute_features(
    price_df: pd.DataFrame, window_short: int = 5, window_long: int = 20
) -> pd.DataFrame:
    """
    Derive a feature panel suitable for ML modeling.
    """
    feats = price_df.copy()
    for sym in feats.columns:
        close = feats[sym]
        feats[(sym, "ret_1")] = close.pct_change()
        feats[(sym, "ema_short")] = close.ewm(span=window_short, adjust=False).mean()
        feats[(sym, "ema_long")] = close.ewm(span=window_long, adjust=False).mean()
        feats[(sym, "ema_ratio")] = feats[(sym, "ema_short")] / feats[(sym, "ema_long")]
        feats[(sym, "vol_10")] = close.pct_change().rolling(10).std(ddof=0)
        feats[(sym, "rsi_14")] = _rsi(close, period=14)
    feats = feats.drop(columns=list(feats.columns[: len(price_df.columns)]))
    feats = feats.dropna()
    feats.columns = pd.Index([f"{sym}_{feat}" for sym, feat in feats.columns])
    return feats


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
    """
    Train a simple but strong gradient boosting model.
    Returns a tuple of (sklearn Pipeline, feature_importances DataFrame).
    """
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
    """
    Vectorized long-short backtest with friction.
    """
    aligned_prices = price_df.loc[signals.index]
    returns = aligned_prices["UPRO"].pct_change().fillna(0) - aligned_prices["SPXU"].pct_change().fillna(0)
    strat_returns = returns * signals.shift().fillna(0)
    costs = np.abs(signals.diff().fillna(0)) * (trading_cost_bps / 10_000)
    strat_returns -= costs
    equity = (1 + strat_returns).cumprod() * cash
    stats = _performance_stats(strat_returns)
    out = pd.DataFrame({"equity": equity, "strategy_return": strat_returns})
    out.attrs["stats"] = stats
    return out


def _performance_stats(returns: pd.Series) -> Dict[str, float]:
    ann_factor = 252 * 6.5 * 60  # assume minute data
    mean = returns.mean()
    std = returns.std(ddof=0)
    sharpe = (mean * ann_factor) / (std * np.sqrt(ann_factor)) if std > 0 else 0.0
    cagr = (1 + returns).prod() ** (ann_factor / len(returns)) - 1
    max_drawdown = ((1 + returns).cumprod().cummax() - (1 + returns).cumprod()).max()
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": -max_drawdown}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


__all__ = [
    "bootstrap_env",
    "AlpacaDataClient",
    "pivot_ohlc",
    "compute_features",
    "label_future_returns",
    "train_model",
    "generate_signals",
    "backtest_signals",
]


