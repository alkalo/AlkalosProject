import numpy as np
import pandas as pd
import pytest

from src.backtest.strategy import SignalStrategy


class SeqModel:
    def __init__(self, probs):
        self.probs = list(probs)

    def predict_proba(self, X):
        p = self.probs.pop(0)
        return np.array([[1 - p, p] for _ in range(len(X))])


def test_hysteresis_prevents_flip_flop():
    model = SeqModel([0.61, 0.59, 0.41, 0.39])
    strat = SignalStrategy(model, buy_thr=0.6, sell_thr=0.4)
    df = pd.DataFrame({"f": [0]})
    assert strat.generate_signal(df) == "BUY"
    assert strat.generate_signal(df) == "HOLD"
    assert strat.generate_signal(df) == "HOLD"
    assert strat.generate_signal(df) == "SELL"


def test_buy_thr_must_exceed_sell_thr():
    with pytest.raises(ValueError):
        SignalStrategy(SeqModel([0.5]), buy_thr=0.4, sell_thr=0.4)


def test_min_edge_must_cover_costs():
    with pytest.raises(ValueError):
        SignalStrategy(SeqModel([0.5]), buy_thr=0.6, sell_thr=0.4, min_edge=0.01, costs=0.02)
