import numpy as np

from models import KerasLSTMClassifier
from signal_strategy import SignalStrategy


class DummyKerasModel:
    def predict(self, X):
        # Return a single column of probabilities
        return np.array([[0.2], [0.8], [0.6]])


def test_predict_proba_two_columns():
    clf = KerasLSTMClassifier(DummyKerasModel())
    proba = clf.predict_proba(np.zeros((3, 2)))
    assert proba.shape == (3, 2)
    assert np.allclose(proba[:, 0], 1 - proba[:, 1])


class TwoColumnModel:
    def predict_proba(self, X):
        return np.array([[0.4, 0.6], [0.3, 0.7]])


class OneColumnModel:
    def predict_proba(self, X):
        return np.array([[0.6], [0.8]])


def test_signal_strategy_two_columns():
    strat = SignalStrategy(TwoColumnModel())
    last = strat.predict_proba_last(np.zeros((2, 2)))
    assert last == 0.7


def test_signal_strategy_one_column():
    strat = SignalStrategy(OneColumnModel())
    last = strat.predict_proba_last(np.zeros((2, 2)))
    assert last == 0.8
