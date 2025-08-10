from src.ml.train import train
from src.backtest.strategy import SignalStrategy


def test_signal_strategy_loads_features(tmp_path):
    model_dir = tmp_path / "model"
    train(str(model_dir), [1, 2, 3])
    strategy = SignalStrategy(str(model_dir))
    assert strategy.features == [1, 2, 3]
