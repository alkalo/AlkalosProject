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
    ratio1 = (12 * 0.99) / (10 * 1.01)
    ratio2 = (13 * 0.99) / (11 * 1.01)
    expected_final = 100 * ratio1 * ratio2
    assert summary["final_equity"] == pytest.approx(expected_final)
    assert summary["trades"] == 2


def test_sharpe_and_drawdown_on_synthetic_data():
    df = pd.DataFrame({
        "close": [100, 105, 95, 100, 110],
        "signal": ["BUY", "HOLD", "HOLD", "HOLD", "SELL"],
    })
    summary, equity, _ = backtest_spot(df, initial_cash=100, fee=0.0)
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = -drawdown.min()
    assert summary["sharpe"] == pytest.approx(sharpe, rel=1e-3)
    assert summary["max_drawdown"] == pytest.approx(max_dd, rel=1e-3)
    years = len(equity) / 252
    expected_cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    assert summary["cagr"] == pytest.approx(expected_cagr, rel=1e-3)


def test_stop_loss_and_trade_limits_respected():
    df = pd.DataFrame({
        "close": [100, 90, 110, 200],
        "signal": ["BUY", "SELL", "SELL", "BUY"],
    })
    summary, _, trades = backtest_spot(df, initial_cash=100, fee=0.0)
    assert trades["type"].tolist() == ["BUY", "SELL"]
    assert summary["final_equity"] == 90
    assert summary["trades"] == 1
