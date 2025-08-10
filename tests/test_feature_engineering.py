import numpy as np
import pandas as pd

from feature_engineering import add_tech_indicators, make_supervised, scale_features


def sample_df(n=120):
    rng = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame(index=rng)
    df["open"] = prices + np.random.randn(n)
    df["high"] = df["open"] + np.abs(np.random.randn(n))
    df["low"] = df["open"] - np.abs(np.random.randn(n))
    df["close"] = prices + np.random.randn(n)
    df["volume"] = np.random.randint(100, 1000, size=n)
    return df


def test_make_supervised_shapes():
    np.random.seed(42)
    df = sample_df()
    X, y, features = make_supervised(df, target_horizon=1, window=30)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(features)


def test_scale_features():
    np.random.seed(42)
    df = sample_df()
    X, y, features = make_supervised(df, target_horizon=1, window=30)
    X_train, X_test = X[:50], X[50:]
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    # mean of scaled training data is ~0
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-6)
