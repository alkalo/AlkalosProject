import numpy as np
import pandas as pd

from src.ml.train import train
from src.backtest.strategy import SignalStrategy


class DummyModel:
    def predict_proba(self, X):
        # Return constant probability favouring the positive class.
        return np.array([[0.3, 0.7] for _ in range(len(X))])


class LSTMModel(DummyModel):
    def __init__(self):
        self.last_shape = None

    def predict_proba(self, X):  # pragma: no cover - simple wrapper
        self.last_shape = X.shape
        return super().predict_proba(X)


def test_signal_strategy_loads_features_and_generates_signal(tmp_path):
    model_root = tmp_path / "models"
    train("SYMBOL", ["a", "b"], model=DummyModel(), model_dir=str(model_root))
    strat = SignalStrategy("SYMBOL", model_dir=str(model_root))
    assert strat.feature_names == ["a", "b"]
    df = pd.DataFrame({"a": [0], "b": [0]})
    assert strat.generate_signal(df) == "BUY"


def test_lstm_window_reshaping(tmp_path):
    model_root = tmp_path / "models"
    train(
        "LSTM",
        ["x"],
        model=LSTMModel(),
        model_dir=str(model_root),
        is_lstm=True,
    )
    strat = SignalStrategy("LSTM", model_dir=str(model_root))
    df = pd.DataFrame({"x": [1, 2, 3]})
    strat.generate_signal(df)
    assert strat.model.last_shape == (1, 3, 1)
