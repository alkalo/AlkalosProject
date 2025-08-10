import pandas as pd
import pytest

from src.backtest import engine


def test_no_signal_has_constant_equity_and_no_trades():
    df = pd.DataFrame({
        "close": [10, 11, 12],
        "signal": ["HOLD", "HOLD", "HOLD"],
    })
    summary, equity, trades = engine.backtest_spot(df, fee=0.01, initial_cash=100)
    assert trades.empty
    assert (equity == 100).all()
    assert summary["final_equity"] == 100


def test_buy_then_sell_with_fee():
    df = pd.DataFrame({
        "close": [10, 12],
        "signal": ["BUY", "SELL"],
    })
    summary, equity, trades = engine.backtest_spot(df, fee=0.01, initial_cash=1000)
    # Two trades: buy and sell
    assert len(trades) == 2
    # Expected PnL accounting for fees
    expected_final = 1000 - 10 * (1 + 0.01) + 12 * (1 - 0.01)
    assert summary["final_equity"] == pytest.approx(expected_final)
