"""Backtesting strategy helpers."""
from __future__ import annotations

import os
from typing import Any, Union

import joblib
import numpy as np


class SignalStrategy:
    """Utilities for working with prediction probabilities and stored features.

    Parameters
    ----------
    model_or_dir:
        Either a model implementing ``predict_proba`` or a path to a
        directory containing precomputed ``features.pkl``.
    """

    def __init__(self, model_or_dir: Union[Any, str]) -> None:
        if isinstance(model_or_dir, str):
            feature_path = os.path.join(model_or_dir, "features.pkl")
            self.features = joblib.load(feature_path)
            self.model = None
        else:
            self.model = model_or_dir

    def predict_proba_last(self, X: Any) -> float:
        """Return the probability for the positive class of the last sample."""
        if self.model is None:
            raise ValueError("Model is not set for probability prediction")
        proba = self.model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
        return proba[-1, 1]
