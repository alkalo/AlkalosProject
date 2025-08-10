import pandas as pd

from src.live.paper_bot import RiskManager


def test_risk_manager_triggers_on_drawdown():
    rm = RiskManager(0.1)
    now = pd.Timestamp("2024-01-01")
    assert rm.check(100, now)
    assert rm.check(110, now)
    # Drawdown below 10% from high (110 -> threshold 99)
    assert not rm.check(98, now)


def test_risk_manager_resets_each_day():
    rm = RiskManager(0.05)
    day1 = pd.Timestamp("2024-01-01")
    day2 = pd.Timestamp("2024-01-02")
    assert rm.check(100, day1)
    assert rm.check(110, day1)
    # New day resets high
    assert rm.check(100, day2)
