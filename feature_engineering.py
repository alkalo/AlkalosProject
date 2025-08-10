import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns 'open', 'high', 'low', 'close', 'volume'.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for each indicator.
    """
    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log1p(df["returns"])

    # Simple Moving Averages
    df["SMA_7"] = df["close"].rolling(window=7).mean()
    df["SMA_21"] = df["close"].rolling(window=21).mean()

    # Exponential Moving Averages
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()

    # Bollinger Bands
    middle = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["BB_m"] = middle
    df["BB_u"] = middle + 2 * std
    df["BB_l"] = middle - 2 * std

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = signal
    df["MACD_hist"] = macd - signal

    return df


def make_supervised(
    df: pd.DataFrame,
    target_horizon: int = 1,
    window: int = 30,
    features: list | None = None,
):
    """Convert price data into supervised learning datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame.
    target_horizon : int, default 1
        Number of steps ahead to predict.
    window : int, default 30
        Number of initial rows to drop to avoid look-ahead bias.
    features : list, optional
        Specific feature columns to use.

    Returns
    -------
    tuple
        (X, y, feature_names)
    """
    df = add_tech_indicators(df.copy())

    df["target"] = (df["close"].shift(-target_horizon) > df["close"]).astype(int)

    if features is None:
        exclude = {"open", "high", "low", "close", "volume", "target"}
        features = [c for c in df.columns if c not in exclude]

    # Purge initial window and final target_horizon rows
    df = df.iloc[window:-target_horizon].dropna()

    X = df[features].to_numpy()
    y = df["target"].to_numpy()

    return X, y, features


def scale_features(X_train, X_test):
    """Scale features using StandardScaler fitted on X_train."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
