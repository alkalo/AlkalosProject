import numpy as np
import pandas as pd

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
