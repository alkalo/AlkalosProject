import pandas as pd
import pytest

from src.backtest.engine import backtest_spot


def test_no_signal_has_constant_equity_and_no_trades():
    df = pd.DataFrame({
        "close": [10, 11, 12],
        "signal": ["HOLD", "HOLD", "HOLD"],
    })
    summary, equity, trades = backtest_spot(df, fee=0.01, initial_cash=100)
    assert trades.empty
    assert (equity == 100).all()
    assert summary["final_equity"] == 100
    assert summary["trades"] == 0


def test_buy_then_sell_with_fee():
    df = pd.DataFrame({
        "close": [10, 12],
        "signal": ["BUY", "SELL"],
    })
    summary, equity, trades = backtest_spot(df, fee=0.01, initial_cash=1000)
    # Two records: buy and sell
    assert len(trades) == 2
    qty = 1000 / (10 * (1 + 0.01))
    expected_final = qty * 12 * (1 - 0.01)
    assert summary["final_equity"] == pytest.approx(expected_final)
    assert summary["trades"] == 1
    assert summary["win_rate"] == pytest.approx(1.0)
    expected_ret = (12 * 0.99) / (10 * 1.01) - 1
    assert summary["avg_trade_return"] == pytest.approx(expected_ret)


def test_stop_loss_triggered():
    df = pd.DataFrame({
        "close": [10, 9],
        "signal": ["BUY", "HOLD"],
    })
    summary, equity, trades = backtest_spot(
        df, fee=0.01, initial_cash=1000, stop_loss=0.05
    )
    assert summary["trades"] == 1
    assert summary["win_rate"] == 0.0
    expected_ret = (9 * 0.99) / (10 * 1.01) - 1
    assert summary["avg_trade_return"] == pytest.approx(expected_ret)
    qty = 1000 / (10 * (1 + 0.01))
    expected_final = qty * 9 * (1 - 0.01)
    assert summary["final_equity"] == pytest.approx(expected_final)


def test_slippage_affects_final_equity():
    df = pd.DataFrame({
        "close": [100, 110],
        "signal": ["BUY", "SELL"],
    })
    summary_no_slip, _, _ = backtest_spot(
        df, fee=0.0, slippage=0.0, initial_cash=1000
    )
    summary_slip, _, _ = backtest_spot(
        df, fee=0.0, slippage=0.01, initial_cash=1000
    )
    assert summary_no_slip["final_equity"] == pytest.approx(1100.0)
    expected_with_slip = 1000 / (100 * 1.01) * 110 * (1 - 0.01)
    assert summary_slip["final_equity"] == pytest.approx(expected_with_slip)
    assert summary_slip["final_equity"] < summary_no_slip["final_equity"]
