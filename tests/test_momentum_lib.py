import pandas as pd
from momentum_lib import compute_features, backtest_signals, generate_signals


def synthetic_data():
    idx = pd.date_range("2024-01-01", periods=120, freq="T")
    prices = pd.DataFrame(
        {
            "UPRO": 100 + (idx.hour * 0.1),
            "SPXU": 20 + (idx.hour * 0.05),
        },
        index=idx,
    )
    return prices


def test_compute_features_shape():
    feats = compute_features(synthetic_data())
    assert not feats.empty
    assert "UPRO_ret_1" in feats.columns


def test_backtest_runs():
    prices = synthetic_data()
    feats = compute_features(prices)
    # Simple deterministic signals
    signals = pd.Series(0, index=feats.index)
    signals.iloc[::10] = 1
    backtest = backtest_signals(prices.loc[signals.index], signals)
    stats = backtest.attrs["stats"]
    assert "sharpe" in stats

