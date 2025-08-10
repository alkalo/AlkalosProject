import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import backtest_spot


def test_accounting_with_varied_signals_and_fees():
    df = pd.DataFrame({
        "close": [10, 12, 11, 13],
        "signal": ["BUY", "SELL", "BUY", "SELL"],
    })
    summary, equity, trades = backtest_spot(df, fee=0.01, initial_cash=100)
    assert len(trades) == 4
    expected_final = 100 - 10 * 1.01 + 12 * 0.99 - 11 * 1.01 + 13 * 0.99
    assert summary["final_equity"] == pytest.approx(expected_final)


def test_sharpe_and_drawdown_on_synthetic_data():
    df = pd.DataFrame({
        "close": [100, 105, 95, 100, 110],
        "signal": ["BUY", "HOLD", "HOLD", "HOLD", "SELL"],
    })
    _, equity, _ = backtest_spot(df, initial_cash=100, fee=0.0)
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(len(returns))
    assert sharpe == pytest.approx(0.6349, rel=1e-3)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    assert drawdown.min() == pytest.approx(-0.095238, rel=1e-3)


def test_stop_loss_and_trade_limits_respected():
    df = pd.DataFrame({
        "close": [100, 90, 110, 200],
        "signal": ["BUY", "SELL", "SELL", "BUY"],
    })
    summary, _, trades = backtest_spot(df, initial_cash=100, fee=0.0)
    assert trades["type"].tolist() == ["BUY", "SELL"]
    assert summary["final_equity"] == 90
