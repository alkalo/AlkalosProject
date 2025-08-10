"""Model wrapper utilities.

This module contains small helper classes that provide a consistent interface
across different machine learning libraries.  They are intentionally minimal
and only implement the parts of the scikit-learn API that are required by the
unit tests and the training script.
"""

from __future__ import annotations

import numpy as np


class LGBMClassifierModel:
    """Thin wrapper around :class:`lightgbm.LGBMClassifier`.

    The project prefers to keep third-party dependencies lightweight.  Rather
    than exposing the LightGBM estimator directly we wrap it in a tiny class
    that mimics the parts of the scikit-learn API we rely on.  This keeps the
    training code agnostic of the underlying library while still performing a
    real gradient boosting training run.
    """

    def __init__(self, **params: dict):
        from lightgbm import LGBMClassifier

        self.model = LGBMClassifier(**params)

    def fit(self, X, y):  # pragma: no cover - simple passthrough
        self.model.fit(X, y)
        return self

    def predict(self, X):  # pragma: no cover - simple passthrough
        return self.model.predict(X)

    def predict_proba(self, X):  # pragma: no cover - simple passthrough
        return self.model.predict_proba(X)


# Usado por tests/test_predict_proba.py

class KerasLSTMClassifier:
    """Wrapper around a Keras model providing scikit-learn style predict_proba."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        """Return probabilities in scikit-learn's two-column format.

        Many Keras binary classifiers output an array with a single column
        representing the probability of the positive class. scikit-learn's
        ``predict_proba`` convention, however, is to return two columns:
        ``[P(class=0), P(class=1)]``.  This method ensures that behaviour.
        """
        p = self.model.predict(X)
        p = np.asarray(p)

        # If the underlying model already returns two columns we simply
        # forward the output.
        if p.ndim == 2 and p.shape[1] == 2:
            return p

        # For one-dimensional outputs we ensure the array is 1D
        if p.ndim == 2 and p.shape[1] == 1:
            p = p.ravel()
        elif p.ndim > 2:
            raise ValueError("Unexpected probability shape: %s" % (p.shape,))

        # At this point ``p`` should be one-dimensional. Construct the
        # negative class probability and stack the result into the expected
        # two-column format.
        return np.column_stack([1 - p, p])
