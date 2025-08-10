"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd
from typing import Tuple


def make_lagged_features(
    series: pd.Series, window: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create lagged features for a univariate series.

    Parameters
    ----------
    series:
        Input time series to lag. The current value will be dropped so that
        each row only contains information strictly from the past.
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


def add_simple_returns(df: pd.DataFrame, price_col: str = "close", *, col_name: str = "return") -> pd.DataFrame:
    """Add simple percentage returns to ``df``.

    Parameters
    ----------
    df:
        Input DataFrame containing a price column.
    price_col:
        Name of the column with close prices.
    col_name:
        Name of the generated return column.
    """
    df = df.copy()
    df[col_name] = df[price_col].pct_change()
    return df


def add_tech_indicators(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Augment ``df`` with a small set of technical indicators.

    The implementation purposefully keeps the computations lightweight; it is
    sufficient for unit tests and small examples and does not aim to be
    an exhaustive technical analysis library.
    """
    df = add_simple_returns(df, price_col)

    # Simple moving averages
    df["sma_5"] = df[price_col].rolling(5).mean()
    df["sma_10"] = df[price_col].rolling(10).mean()

    # Relative Strength Index (RSI)
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - 100 / (1 + rs)

    return df
