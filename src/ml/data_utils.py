"""Dataset manipulation helpers.

This module contains a couple of tiny helpers used in the unit tests.  The
functions intentionally keep the implementation compact while providing a
realistic interface for working with time series data.
"""

from __future__ import annotations

import pandas as pd

from .feature_engineering import (
    add_simple_returns,
    add_tech_indicators,
    make_lagged_features,
)
from typing import Sequence, Tuple, Union


ArrayLike = Union[pd.DataFrame, pd.Series]


def temporal_train_test_split(
    *arrays: ArrayLike,
    test_size: float = 0.2,
) -> Tuple[ArrayLike, ...]:
    """Split arrays preserving temporal order.

    Parameters
    ----------
    *arrays:
        Any number of ``pandas`` ``Series`` or ``DataFrame`` objects sharing the
        same length.  The function will split each array at the same index and
        return the train portions followed by the test portions.
    test_size:
        Fraction of the dataset to include in the test split.

    Returns
    -------
    tuple of arrays:
        The train splits of each input array followed by their respective test
        splits, mirroring :func:`sklearn.model_selection.train_test_split` but
        without shuffling the data.
    """

    if not arrays:
        raise ValueError("At least one array is required")

    first = next((a for a in arrays if a is not None), None)
    if first is None:
        raise ValueError("At least one array must be non-None")

    n_samples = len(first)
    split = int(n_samples * (1 - test_size))

    train_parts = []
    test_parts = []
    for arr in arrays:
        if arr is None:
            train_parts.append(None)
            test_parts.append(None)
            continue
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same length")
        train_parts.append(arr.iloc[:split].copy())
        test_parts.append(arr.iloc[split:].copy())

    result = []
    for train_part, test_part in zip(train_parts, test_parts):
        result.extend([train_part, test_part])

    return tuple(result)

def build_features(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    feature_set: str = "lags",
    window: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, Sequence[str]]:
    """Generate feature matrix and target aligned consistently.

    Parameters
    ----------
    df:
        Input DataFrame containing at least a ``close`` price column and a
        target column.
    target_col:
        Name of the target column.
    feature_set:
        ``"lags"`` to use past price information or ``"tech"`` for a small
        collection of technical indicators.
    window:
        Number of lagged observations when ``feature_set`` is ``"lags"``.
    """
    df = df.copy()
    if feature_set == "tech":
        df = add_tech_indicators(df)
        feature_cols = [
            c for c in df.columns if c not in {target_col, "date", "close"}
        ]
    else:
        lag_df, _ = make_lagged_features(df["close"], window)
        feature_cols = list(lag_df.columns)
        df = pd.concat(
            [lag_df, df[[target_col] + (["date"] if "date" in df.columns else [])]],
            axis=1,
        )

    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols
