# src/ml/data_utils.py
from __future__ import annotations

import numpy as np
import pandas as pd

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["timestamp", "date", "Datetime", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df = df.set_index(col).sort_index()
            return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    return df.sort_index()

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _build_lag_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    for k in range(1, window + 1):
        out[f"close_lag_{k}"] = close.shift(k)
        out[f"ret_{k}"] = close.pct_change(k)
    out["ret_1_abs"] = out["ret_1"].abs()
    return out

def _build_tech_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    out["rsi_14"] = _rsi(close, 14)
    out["ema_12"] = _ema(close, 12)
    out["ema_26"] = _ema(close, 26)
    out["sma_10"] = close.rolling(10, min_periods=10).mean()
    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["bb_mid_20"] = out["sma_20"]
    std20 = close.rolling(20, min_periods=20).std()
    out["bb_up_20"] = out["bb_mid_20"] + 2 * std20
    out["bb_dn_20"] = out["bb_mid_20"] - 2 * std20
    tr = (
        pd.concat([high, close.shift()], axis=1).max(axis=1)
        - pd.concat([low, close.shift()], axis=1).min(axis=1)
    )
    out["true_range"] = tr
    out["atr_14"] = tr.rolling(14, min_periods=14).mean()
    for k in (1, 2, 3, 5):
        out[f"roc_{k}"] = close.pct_change(k)
    return out


def make_lagged_features(series: pd.Series, window: int) -> tuple[pd.DataFrame, pd.Series]:
    """Create a simple lagged feature representation.

    Parameters
    ----------
    series:
        Univariate time series indexed by a ``DatetimeIndex`` or any sortable
        index.  The index is preserved in the returned frames.
    window:
        Number of past observations to include as features.

    Returns
    -------
    X, y:
        ``X`` contains ``window`` lagged values labelled ``lag_1`` … ``lag_n``.
        ``y`` is the contemporaneous value of ``series``.
    """

    if not isinstance(series, pd.Series):  # pragma: no cover - defensive
        raise TypeError("series must be a pandas Series")

    X = pd.concat(
        {f"lag_{i}": series.shift(i) for i in range(1, window + 1)}, axis=1
    ).dropna()
    y = series.loc[X.index]
    return X, y


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None = None,
    *,
    test_size: float = 0.2,
):
    """Split data chronologically without shuffling.

    The final ``test_size`` proportion (or absolute number) of samples forms the
    test set with the remainder used for training.  ``dates`` is split in the
    same manner if provided.
    """

    n_samples = len(X)
    if 0 < test_size < 1:
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)
    split = n_samples - n_test

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if dates is None:
        return X_train, X_test, y_train, y_test

    dates_train, dates_test = dates.iloc[:split], dates.iloc[split:]
    return X_train, X_test, y_train, y_test, dates_train, dates_test

def build_features(
    df: pd.DataFrame,
    *,
    feature_set: str = "lags",
    window: int = 3,
):
    """Return feature matrix, target vector and column names.

    Parameters
    ----------
    df:
        DataFrame containing at least ``close`` and ``target`` columns.
    feature_set:
        ``"lags"`` to generate lagged ``close`` prices or ``"tech"`` for a small
        set of technical indicators.
    window:
        Number of lags to generate when ``feature_set="lags"``.
    """

    if feature_set == "lags":
        cols = [f"lag_{i}" for i in range(1, window + 1)]
        X = pd.concat({c: df["close"].shift(i) for i, c in enumerate(cols, 1)}, axis=1)
    elif feature_set == "tech":
        X = pd.DataFrame(index=df.index)
        X["return"] = df["close"].pct_change()
        X["sma_5"] = df["close"].rolling(5).mean()
        X["sma_10"] = df["close"].rolling(10).mean()
        X["rsi_14"] = _rsi(df["close"], 14)
        cols = ["return", "sma_5", "sma_10", "rsi_14"]
    else:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unknown feature_set: {feature_set}")

    X = X.dropna()
    y = df.loc[X.index, "target"]
    return X, y, cols
