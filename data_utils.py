"""Utility functions for time series datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def time_series_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series ``DataFrame`` into train and test sets preserving order.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series ordered by date. The index should represent the chronology
        or an explicit date column must be used for sorting.
    test_size : float, optional
        Fraction of samples to allocate to the test set. Defaults to ``0.2``.

    Returns
    -------
    train_df : pandas.DataFrame
        Training subset of shape ``(n_train, n_features)``.
    test_df : pandas.DataFrame
        Test subset of shape ``(n_test, n_features)``.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    df_sorted = df.sort_index()
    n_test = int(len(df_sorted) * test_size)
    if n_test == 0:
        raise ValueError("test_size too small for the number of samples")

    train_df = df_sorted.iloc[:-n_test]
    test_df = df_sorted.iloc[-n_test:]
    return train_df, test_df


def build_dataset_from_csv(path: str | Path, horizon: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    """Create a supervised dataset from a CSV time series file.

    The CSV file is expected to have a first column with dates and a second
    column with numeric values. Features are constructed using a sliding
    ``window`` of past observations to predict ``horizon`` future steps.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the CSV file to load.
    horizon : int
        Number of future steps to forecast.
    window : int
        Size of the lag window used as features.

    Returns
    -------
    X_train : numpy.ndarray of shape ``(n_train, window)``
        Training feature matrix scaled with :class:`~sklearn.preprocessing.StandardScaler`.
    X_test : numpy.ndarray of shape ``(n_test, window)``
        Test feature matrix.
    y_train : numpy.ndarray of shape ``(n_train, horizon)``
        Training targets.
    y_test : numpy.ndarray of shape ``(n_test, horizon)``
        Test targets.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object used on ``X_train``.
    features : pandas.DataFrame of shape ``(n_samples, window + horizon)``
        Complete set of generated features and targets before splitting.
    """
    if horizon <= 0 or window <= 0:
        raise ValueError("horizon and window must be positive integers")

    path = Path(path)
    df = pd.read_csv(path, parse_dates=[0])
    df.sort_values(df.columns[0], inplace=True)
    df.set_index(df.columns[0], inplace=True)

    series = df.iloc[:, 0].astype(float)
    data = []
    for i in range(window, len(series) - horizon + 1):
        past = series.iloc[i - window:i].to_numpy()
        future = series.iloc[i:i + horizon].to_numpy()
        data.append(np.concatenate([past, future]))

    if not data:
        raise ValueError("Not enough data to build the dataset")

    columns = [f"lag_{window - i}" for i in range(window)] + [f"y_{i + 1}" for i in range(horizon)]
    index = series.index[window: len(series) - horizon + 1]
    features = pd.DataFrame(data, columns=columns, index=index)

    train_feat, test_feat = time_series_train_test_split(features, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat.iloc[:, :window])
    X_test = scaler.transform(test_feat.iloc[:, :window])
    y_train = train_feat.iloc[:, window:].to_numpy()
    y_test = test_feat.iloc[:, window:].to_numpy()

    return X_train, X_test, y_train, y_test, scaler, features
