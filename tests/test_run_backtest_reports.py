import sys
import pandas as pd

from src.ml.train import train
from src.utils import env as env_mod


class DummyModel:
    def predict_proba(self, X):
        return [[0.3, 0.7]] * len(X)


def test_run_backtest_generates_reports(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    logs_dir = tmp_path / "logs"
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("REPORTS_DIR", str(reports_dir))
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    env_mod.get_models_dir.cache_clear()
    env_mod.get_reports_dir.cache_clear()
    env_mod.get_logs_dir.cache_clear()

    train("BTC", ["x", "y"], model=DummyModel(), model_dir=str(models_dir))

    df = pd.DataFrame({"close": [1, 2, 3], "x": [0, 1, 0], "y": [1, 0, 1], "extra": [5, 5, 5]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    from src.backtest import run_backtest
    import sys

    monkeypatch.setattr(sys, "argv", ["run_backtest", "--symbol", "BTC", "--csv", str(csv_path)])
    run_backtest.main()

    assert (reports_dir / "BTC_summary.json").exists()
    assert (reports_dir / "BTC_trades.csv").exists()
    assert (reports_dir / "BTC_equity.png").exists()
