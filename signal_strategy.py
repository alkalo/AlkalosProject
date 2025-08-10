import numpy as np


class SignalStrategy:
    """Simple strategy utilities for working with prediction probabilities."""

    def __init__(self, model):
        self.model = model

    def predict_proba_last(self, X):
        """Return the probability for the positive class of the last sample.

        The method is resilient to models that return either a single column
        of positive class probabilities or the two-column scikit-learn
        convention. In all cases the probability of the positive class for
        the last sample is returned as a scalar.
        """
        proba = self.model.predict_proba(X)
        proba = np.asarray(proba)

        # Normalise single column outputs into two-column format
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])

        return proba[-1, 1]
