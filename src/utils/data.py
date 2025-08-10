# src/utils/data.py
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple

NUM_COLS = ["open", "high", "low", "close", "volume"]

def read_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    Lee OHLCV desde CSV con máxima tolerancia:
    - elimina timestamps duplicados
    - detecta ms/ISO y normaliza a UTC
    - convierte numéricos
    - ordena por timestamp
    - valida columnas requeridas
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV no encontrado: {path}")

    df = pd.read_csv(path, sep=",", engine="python")
    # Normaliza headers
    df.columns = [c.strip().lower() for c in df.columns]

    # Consolidar timestamp
    ts_cols = [c for c in df.columns if c.startswith("timestamp")]
    if not ts_cols:
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
            ts_cols = ["timestamp"]
        else:
            df["timestamp"] = pd.RangeIndex(len(df))
            ts_cols = ["timestamp"]
    keep = ts_cols[0]
    drop_ = [c for c in ts_cols if c != keep]
    if drop_:
        df = df.drop(columns=drop_)

    if np.issubdtype(df[keep].dtype, np.number):
        df[keep] = pd.to_datetime(df[keep], unit="ms", utc=True)
    else:
        df[keep] = pd.to_datetime(df[keep], utc=True, errors="coerce")

    df = df.rename(columns={keep: "timestamp"})

    # Validar numéricos
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("Falta columna 'close' en el CSV.")

    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])
    if df.empty:
        raise ValueError(f"Archivo vacío o mal formateado: {path}")

    return df.reset_index(drop=True)


def add_basic_features(df: pd.DataFrame, windows=(5, 10, 20, 50)) -> pd.DataFrame:
    """
    Features 100% pasadas (sin leakage).
    - returns, rolling mean/std, momentum, RSI simple
    """
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()

    for w in windows:
        out[f"ret_{w}"] = out["close"].pct_change(w)
        out[f"vol_{w}"] = out["ret_1"].rolling(w).std()
        out[f"mom_{w}"] = out["close"].diff(w)
        out[f"sma_{w}"] = out["close"].rolling(w).mean()

    # RSI (simple)
    win = 14
    delta = out["close"].diff()
    up = delta.clip(lower=0).rolling(win).mean()
    down = (-delta.clip(upper=0)).rolling(win).mean()
    rs = up / (down.replace(0, np.nan))
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out = out.dropna().reset_index(drop=True)
    return out


def train_val_test_split_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
