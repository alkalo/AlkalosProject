import numpy as np


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
