# src/ml/data_utils.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

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
    tr = (pd.concat([high, close.shift()], axis=1).max(axis=1) -
          pd.concat([low, close.shift()], axis=1).min(axis=1))
    out["true_range"] = tr
    out["atr_14"] = tr.rolling(14, min_periods=14).mean()
    for k in (1, 2, 3, 5):
        out[f"roc_{k}"] = close.pct_change(k)
    return out

def build_features(
    df: pd.DataFrame,
    feature_set: str = "lags",
    window: int = 5,
    horizon: int = 1,
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Construye X (features) y y (target binario: 1 si close(t+h)>close(t)) desde OHLCV.
    Devuelve: X, y, meta dict con feature_set/window/horizon.
    """
    df = _ensure_datetime_index(df)
    df = _normalize_ohlcv(df)

    if "close" not in df.columns:
        raise ValueError("Se requiere columna 'close' para construir features.")

    h = max(int(horizon), 1)
    future = df["close"].shift(-h)
    df[target_col] = (future > df["close"]).astype("Int64")

    fs = (feature_set or "lags").lower()
    if fs not in {"lags", "tech", "both"}:
        fs = "lags"

    w = max(int(window), 1)
    feats = []
    if fs in {"lags", "both"}:
        feats.append(_build_lag_features(df, w))
    if fs in {"tech", "both"}:
        feats.append(_build_tech_features(df, w))
    X = pd.concat(feats, axis=1)

    valid_len = len(df) - h
    if valid_len < 1:
        return X.iloc[0:0], pd.Series(dtype=int), {"feature_set": fs, "window": w, "horizon": h, "target_col": target_col}

    X = X.iloc[:valid_len]
    y = df[target_col].iloc[:valid_len].astype(int)

    mask_valid = ~X.isna().any(axis=1)
    X = X.loc[mask_valid]
    y = y.loc[X.index]

    meta: Dict[str, Any] = {
        "feature_set": fs,
        "window": int(w),
        "horizon": int(h),
        "target_col": target_col,
    }
    return X, y, meta
