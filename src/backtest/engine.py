"""Minimal backtesting engine used in unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class Trade:
    """Simple trade record."""

    timestamp: object
    type: str
    price: float
    qty: float


TradeList = List[Trade]


def backtest_spot(
    df: pd.DataFrame,
    *,
    fee: float = 0.0,
    slippage: float = 0.0,  # pragma: no cover - reserved for future use
    initial_cash: float = 1000.0,
) -> Tuple[dict, pd.Series, pd.DataFrame]:
    """Run a tiny spot backtest driven by ``signal`` column.

    Parameters
    ----------
    df:
        ``DataFrame`` containing at least ``close`` and ``signal`` columns.
    fee:
        Proportional trading fee applied on both entry and exit.
    slippage:
        Currently unused but kept for API compatibility.
    initial_cash:
        Starting cash for the backtest.
    """

    cash = initial_cash
    position = 0.0
    equity_curve = []
    trades: TradeList = []

    for ts, row in df.iterrows():
        price = float(row["close"])
        signal = row.get("signal", "HOLD")

        if signal == "BUY" and cash >= price * (1 + fee):
            qty = 1.0
            cash -= price * (1 + fee)
            position += qty
            trades.append(Trade(ts, "BUY", price, qty))
        elif signal == "SELL" and position >= 1.0:
            cash += price * (1 - fee) * 1.0
            trades.append(Trade(ts, "SELL", price, 1.0))
            position -= 1.0

        equity_curve.append(cash + position * price)

    summary = {"final_equity": equity_curve[-1] if equity_curve else initial_cash}
    equity = pd.Series(equity_curve, index=df.index, name="equity")
    trades_df = pd.DataFrame(trades)
    return summary, equity, trades_df
