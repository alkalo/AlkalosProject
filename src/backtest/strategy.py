

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Any

import joblib
import numpy as np
import pandas as pd

from src.utils.env import get_models_dir


class SignalStrategy:
    """Utility strategy used during backtesting.

    The class normally expects a ``symbol`` string and loads the
    corresponding model artefacts from ``model_dir``.  For testing purposes
    a model instance can be supplied directly as the first argument.
    """

    def __init__(
        self,

        symbol_or_model,
        model_dir: str = str(get_models_dir()),
        *,
        buy_thr: float = 0.6,
        sell_thr: float = 0.4,
        min_edge: float = 0.02,
    ) -> None:
        self.buy_thr = buy_thr
        self.sell_thr = sell_thr
        self.min_edge = min_edge


        if isinstance(symbol_or_model, str):
            symbol = symbol_or_model
            self.symbol = symbol
            base = Path(model_dir) / symbol
            self.model = joblib.load(base / "model.pkl")
            scaler_path = base / "scaler.pkl"
            self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            with open(base / "features.json", "r", encoding="utf-8") as fh:
                self.feature_names: List[str] = json.load(fh)
            # Presence of model.h5 marks an LSTM model which requires special
            # treatment of input shapes.
            self.is_lstm = (base / "model.h5").exists()
        else:

            # Direct model instance supplied (used in unit tests)
            self.symbol = ""
            self.model = symbol_or_model

            self.scaler = None
            self.feature_names = []
            self.is_lstm = False

    def _prepare_input(self, X: Any) -> np.ndarray:
        """Select the relevant columns and apply scaling/reshaping."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
            X = X.values
        else:  # assume array-like
            X = np.asarray(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.is_lstm:
            X = X.reshape(1, X.shape[0], X.shape[1])
        return X

    def _predict_proba_last(self, X: Any) -> float:
        """Return probability of the positive class for the last sample."""
        X = self._prepare_input(X)
        proba = self.model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
        return proba[-1, 1]

    # Public wrapper used in tests
    def predict_proba_last(self, X: Any) -> float:  # pragma: no cover - thin wrapper
        return self._predict_proba_last(X)


    def generate_signal(self, df_window: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
        """Generate trading signal from a window of data."""
        proba = self._predict_proba_last(df_window)
        if proba >= self.buy_thr and (proba - 0.5) >= self.min_edge:
            return "BUY"
        if proba <= self.sell_thr and (0.5 - proba) >= self.min_edge:
            return "SELL"
        return "HOLD"

