import pandas as pd
from alkalos.data import make_supervised

def test_make_supervised_shapes_and_no_leakage():
    series = pd.Series([1, 2, 3, 4, 5])
    X, y = make_supervised(series, window=2)

    assert X.shape == (3, 2)
    assert y.shape == (3,)

    for i in range(X.shape[0]):
        assert (X[i] == series[i:i+2].to_numpy()).all()
        assert y[i] == series[i+2]
