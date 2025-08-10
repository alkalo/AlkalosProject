import os
import json
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # pragma: no cover
    keras = None
    layers = None



def _build_dataset(df: pd.DataFrame, horizon: int, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Build feature matrix and target vector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Close`` column.
    horizon : int
        Number of periods ahead to predict.
    window : int
        Number of past returns used as features.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary targets (1 if future return > 0).
    future_ret : np.ndarray
        Actual future returns for each sample.
    idx : pd.Index
        Index corresponding to samples.
    """
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["future_return"] = df["return"].shift(-horizon)
    for i in range(1, window + 1):
        df[f"ret_{i}"] = df["return"].shift(i)

    df.dropna(inplace=True)
    features = [f"ret_{i}" for i in range(1, window + 1)]
    X = df[features].values
    y = (df["future_return"] > 0).astype(int).values
    future_ret = df["future_return"].values
    return X, y, future_ret, df.index


def _build_lstm_model(window: int) -> "keras.Model":
    model = keras.Sequential(
        [
            layers.Input(shape=(window, 1)),
            layers.LSTM(32, activation="tanh"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_evaluate(
    model_type: str,
    csv_path: str,
    horizon: int,
    window: int,
    outdir: str,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and evaluate a classifier on price data.

    Parameters
    ----------
    model_type : {"lgbm", "lstm"}
        Type of model to train.
    csv_path : str
        Path to CSV file with at least a ``Close`` column.
    horizon : int
        Horizon to predict.
    window : int
        Window size for past returns as features.
    outdir : str
        Output directory for saving artifacts.
    symbol : Optional[str]
        Asset symbol used for naming outputs. If ``None`` it is inferred from
        ``csv_path`` basename split by "_".

    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluation metrics.
    """
    if model_type not in {"lgbm", "lstm"}:
        raise ValueError("model_type must be 'lgbm' or 'lstm'")

    if symbol is None:
        base = os.path.basename(csv_path)
        symbol = base.split("_")[0]

    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X, y, future_ret, idx = _build_dataset(df, horizon, window)

    X_train, X_test, y_train, y_test, fr_train, fr_test = train_test_split(
        X, y, future_ret, test_size=0.3, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for LGBM model")
        model = LGBMClassifier()
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        model_path = os.path.join(outdir, f"{symbol}_lgbm.pkl")
        joblib.dump(model, model_path)
    else:  # lstm
        if keras is None:
            raise ImportError("tensorflow is required for LSTM model")
        X_train_seq = X_train_scaled.reshape(-1, window, 1)
        X_test_seq = X_test_scaled.reshape(-1, window, 1)
        model = _build_lstm_model(window)
        model.fit(X_train_seq, y_train, epochs=10, batch_size=32, verbose=0)
        proba = model.predict(X_test_seq).ravel()
        model_path = os.path.join(outdir, f"{symbol}_lstm.h5")
        model.save(model_path)

    y_pred = (proba > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
    }

    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    scaler_path = os.path.join(outdir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    features = [f"ret_{i}" for i in range(1, window + 1)]
    features_path = os.path.join(outdir, "features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    # Diagnostic plot
    plt.figure()
    plt.scatter(fr_test, proba, alpha=0.5)
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Probability")
    plt.title(f"{symbol} {model_type.upper()} Diagnostic")
    diag_path = os.path.join(outdir, f"{symbol}_diagnostic.png")
    plt.savefig(diag_path)
    plt.close()

    return metrics
