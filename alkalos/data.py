import numpy as np
import pandas as pd

def make_supervised(series: pd.Series, window: int):
    """Transform a time series into supervised learning format.

    Parameters
    ----------
    series : pandas.Series
        Input time series.
    window : int
        Number of past observations to use as features.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Features matrix X of shape (n_samples, window) and target vector y
        of shape (n_samples,).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    values = series.to_numpy()
    X = []
    y = []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)
