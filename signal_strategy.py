"""Strategy module for generating trading signals using a trained model.

This module defines :class:`SignalStrategy` which wraps a classification
model and a scaler to compute the probability of an upward move on the
latest observation window.  Depending on the predicted probability the
strategy returns one of three signals:

* ``"BUY"``  - predicted probability is above the buy threshold.
* ``"SELL"`` - predicted probability is below the sell threshold.
* ``"HOLD"`` - otherwise.

The thresholds as well as an additional margin (``min_edge``) can be
configured.  The margin is useful to ensure that the expected edge is
sufficient to cover trading fees and slippage.

The class expects three serialized artefacts: ``model``, ``scaler`` and
``feature_names``.  They are typically produced during the training
process and saved using :func:`joblib.dump`.  ``model`` must implement a
``predict_proba`` method, while ``scaler`` should provide ``transform``.
``feature_names`` is a list describing the order of features used during
training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import joblib
import numpy as np
import pandas as pd


@dataclass
class SignalStrategy:
    """Trading strategy driven by a probabilistic classifier.

    Parameters
    ----------
    model_path:
        Path to the serialized classification model supporting
        :meth:`predict_proba`.
    scaler_path:
        Path to the serialized scaler used during model training.
    features_path:
        Path to a serialized list with feature names.
    buy_thr:
        Probability threshold to emit a ``"BUY"`` signal. Defaults to
        ``0.6`` as per project specification.
    sell_thr:
        Probability threshold to emit a ``"SELL"`` signal. Defaults to
        ``0.4``.
    min_edge:
        Additional margin applied to ``buy_thr``/``sell_thr`` to account
        for fees or slippage.
    """

    model_path: str
    scaler_path: str
    features_path: str
    buy_thr: float = 0.6
    sell_thr: float = 0.4
    min_edge: float = 0.0
    model: any = field(init=False, repr=False)
    scaler: any = field(init=False, repr=False)
    feature_names: List[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_names = joblib.load(self.features_path)

    # ------------------------------------------------------------------
    def _extract_features(self, df_window: pd.DataFrame) -> np.ndarray:
        """Build feature vector from the last row of ``df_window``.

        The method selects the columns present in ``self.feature_names``
        from the final row of ``df_window`` and returns them as a
        2-dimensional array compatible with scikit-learn estimators.

        Parameters
        ----------
        df_window:
            DataFrame containing the most recent observation window.

        Returns
        -------
        np.ndarray
            Array with shape ``(1, n_features)`` ready for scaling and
            model inference.
        """

        if df_window.empty:
            raise ValueError("df_window is empty")

        missing = [c for c in self.feature_names if c not in df_window.columns]
        if missing:
            raise ValueError(f"Missing features in df_window: {missing}")

        last_row = df_window.iloc[-1]
        features = last_row[self.feature_names].to_numpy(dtype=float)
        return features.reshape(1, -1)

    # ------------------------------------------------------------------
    def predict_proba_last(self, df_window: pd.DataFrame) -> float:
        """Predict the probability of an upward move for ``df_window``.

        This computes features on the latest row of ``df_window``, scales
        them and feeds them into the classification model.  The method
        returns the probability that the price will go up (``p_up``).

        Parameters
        ----------
        df_window:
            DataFrame holding the last ``n`` observations.

        Returns
        -------
        float
            Probability of an upward move.
        """

        features = self._extract_features(df_window)
        scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(scaled)[0, 1]
        return float(proba)

    # ------------------------------------------------------------------
    def generate_signal(self, df_window: pd.DataFrame) -> str:
        """Generate a trading signal for ``df_window``.

        The decision is based on the predicted probability and the
        thresholds specified in the constructor.  ``min_edge`` adjusts the
        thresholds further, providing an extra safety margin.

        Parameters
        ----------
        df_window:
            DataFrame holding the observation window.

        Returns
        -------
        str
            One of ``"BUY"``, ``"SELL"`` or ``"HOLD"``.
        """

        p_up = self.predict_proba_last(df_window)
        buy_level = self.buy_thr + self.min_edge
        sell_level = self.sell_thr - self.min_edge

        if p_up >= buy_level:
            return "BUY"
        if p_up <= sell_level:
            return "SELL"
        return "HOLD"
