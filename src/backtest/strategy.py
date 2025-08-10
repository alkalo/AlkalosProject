from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd


class SignalStrategy:
    """Utility strategy used during backtesting.

    The strategy expects model artefacts stored using the layout produced by
    :func:`src.ml.train.train`::

        models/{SYMBOL}/
            model.pkl
            model.h5       # optional, presence indicates LSTM
            scaler.pkl
            features.json
            report.json
            diagnostic.png

    Parameters
    ----------
    symbol:
        Trading pair or symbol identifier.
    model_dir:
        Root directory containing the ``models`` folder.
    buy_thr, sell_thr, min_edge:
        Thresholds controlling the generated signal.
    """

    def __init__(
        self,
        symbol: str,
        model_dir: str = "models",
        *,
        buy_thr: float = 0.6,
        sell_thr: float = 0.4,
        min_edge: float = 0.02,
    ) -> None:
        self.symbol = symbol
        self.buy_thr = buy_thr
        self.sell_thr = sell_thr
        self.min_edge = min_edge

        base = Path(model_dir) / symbol
        self.model = joblib.load(base / "model.pkl")
        scaler_path = base / "scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        with open(base / "features.json", "r", encoding="utf-8") as fh:
            self.feature_names: List[str] = json.load(fh)
        # Presence of model.h5 marks an LSTM model which requires special
        # treatment of input shapes.
        self.is_lstm = (base / "model.h5").exists()

    def _predict_proba_last(self, X: np.ndarray) -> float:
        """Return probability of the positive class for the last sample."""
        proba = self.model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
        return proba[-1, 1]

    def generate_signal(self, df_window: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
        """Generate trading signal from a window of data."""
        X = df_window[self.feature_names].values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.is_lstm:
            X = X.reshape(1, X.shape[0], X.shape[1])
        proba = self._predict_proba_last(X)
        if proba >= self.buy_thr and (proba - 0.5) >= self.min_edge:
            return "BUY"
        if proba <= self.sell_thr and (0.5 - proba) >= self.min_edge:
            return "SELL"
        return "HOLD"
