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

import numpy as np
import pandas as pd

def build_features(df, feature_set="basic", window=5, horizon=1, fee=0.001):
    """
    Construye features y el target para entrenamiento.
    - horizon: días hacia adelante para predecir
    - fee: comisión total ida y vuelta, usada para filtrar señales débiles
    """

    # Ordenar por fecha si existe
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Calcular target (subida o bajada)
    df["target"] = (df["close"].shift(-horizon) / df["close"] - 1)

    # Filtrar movimientos insignificantes (menores que doble comisión)
    min_move = fee * 2
    df.loc[df["target"].abs() < min_move, "target"] = np.nan

    # Convertir a binario: 1 si sube, 0 si baja
    df["target"] = (df["target"] > 0).astype(float)
    df.dropna(subset=["target"], inplace=True)

    # Features: precios con lags
    feats = []
    for lag in range(1, window + 1):
        df[f"lag_close_{lag}"] = df["close"].shift(lag)
        feats.append(f"lag_close_{lag}")

    # Features: retornos porcentuales
    for lag in range(1, window + 1):
        df[f"ret_{lag}"] = df["close"].pct_change(lag)
        feats.append(f"ret_{lag}")

    df.dropna(inplace=True)

    X = df[feats].values
    y = df["target"].values
    meta = {"features": feats, "window": window, "horizon": horizon}

    return X, y, meta
