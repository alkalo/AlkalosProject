"""Dataset manipulation helpers.

This module contains a couple of tiny helpers used in the unit tests.  The
functions intentionally keep the implementation compact while providing a
realistic interface for working with time series data.
"""

from __future__ import annotations

import pandas as pd
from typing import Tuple


def make_lagged_features(series: pd.Series, window: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Create lagged features for a univariate series.

    Parameters
    ----------
    series:
        Input time series.
    window:
        Number of past observations to include as features.

    Returns
    -------
    X, y:
        Feature ``DataFrame`` where each column represents a lagged value and
        the corresponding target ``Series`` aligned such that ``y_t`` depends
        only on values strictly prior to ``t``.
    """

    data = {f"lag_{i}": series.shift(i) for i in range(1, window + 1)}
    X = pd.DataFrame(data)
    y = series.copy()
    df = pd.concat([X, y], axis=1).dropna()
    X = df[[f"lag_{i}" for i in range(1, window + 1)]]
    y = df[series.name]
    return X, y


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split ``X`` and ``y`` preserving temporal order.

    Unlike :func:`sklearn.model_selection.train_test_split` this helper never
    shuffles the data which is essential when working with time series where
    chronological ordering matters.
    """

    n_samples = len(X)
    split = int(n_samples * (1 - test_size))
    X_train = X.iloc[:split].copy()
    X_test = X.iloc[split:].copy()
    y_train = y.iloc[:split].copy()
    y_test = y.iloc[split:].copy()
    return X_train, X_test, y_train, y_test


