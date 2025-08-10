import pandas as pd

from src.ml.data_utils import make_lagged_features, temporal_train_test_split


def test_temporal_split_and_no_lookahead():
    series = pd.Series(range(10), name="price")
    X, y = make_lagged_features(series, window=3)

    # First row should contain only past information
    assert X.iloc[0].tolist() == [2, 1, 0]
    assert y.iloc[0] == 3

    # The label must be greater than any feature for this monotonic series
    assert (X.max(axis=1) < y).all()

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.3)

    # Ensure chronological split without shuffling
    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.max() < y_test.index.min()
