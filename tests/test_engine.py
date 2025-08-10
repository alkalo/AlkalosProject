import pytest
from alkalos.engine import run_engine

def test_engine_no_signals_no_trades():
    prices = [10, 11, 12]
    signals = [0, 0, 0]
    trades = run_engine(prices, signals, fee=0.01)
    assert trades == []

def test_engine_buy_sell_trades_and_fees():
    prices = [10, 12]
    signals = [1, -1]
    fee = 0.01
    trades = run_engine(prices, signals, fee)

    assert len(trades) == 2
    assert trades[0].action == "buy"
    assert trades[1].action == "sell"
    assert trades[0].fee == pytest.approx(prices[0] * fee)
    assert trades[1].fee == pytest.approx(prices[1] * fee)
