import pandas as pd

from src.ml.data_utils import build_features


def test_build_features_lags_order():
    df = pd.DataFrame({"close": range(10), "target": range(10)})
    X, y, cols = build_features(df, feature_set="lags", window=3)
    assert cols == ["lag_1", "lag_2", "lag_3"]
    assert list(X.columns) == cols
    assert len(X) == len(y)


def test_build_features_tech_order():
    # Create enough rows so moving averages and RSI are defined
    n = 30
    df = pd.DataFrame({"close": range(n), "target": [0] * n})
    X, y, cols = build_features(df, feature_set="tech")
    assert cols == ["return", "sma_5", "sma_10", "rsi_14"]
    assert list(X.columns) == cols
    assert len(X) == len(y)
