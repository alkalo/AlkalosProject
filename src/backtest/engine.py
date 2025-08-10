"""Minimal backtesting engine used in unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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
    risk_per_trade: float = 1.0,
    stop_loss: float | None = None,
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
    risk_per_trade:
        Fraction of available cash to deploy on each trade.
    stop_loss:
        Optional stop loss expressed as a decimal percentage from entry price.
    """

    cash = initial_cash
    position = 0.0
    equity_curve: list[float] = []
    trades: TradeList = []
    trade_profits: list[float] = []

    entry_cost = 0.0
    stop_price = None

    for ts, row in df.iterrows():
        price = float(row["close"])
        signal = row.get("signal", "HOLD")

        # Check stop loss first
        if position > 0 and stop_loss is not None and price <= stop_price:  # type: ignore[arg-type]
            proceeds = position * price * (1 - fee)
            cash += proceeds
            profit = proceeds - entry_cost
            trade_profits.append(profit)
            trades.append(Trade(ts, "SELL", price, position))
            position = 0.0
            entry_cost = 0.0
            stop_price = None

        # Trading signals
        if signal == "BUY" and cash > 0 and position == 0:
            cash_to_use = cash * risk_per_trade
            qty = cash_to_use / (price * (1 + fee))
            if qty > 0:
                cost = qty * price * (1 + fee)
                cash -= cost
                position = qty
                entry_cost = cost
                if stop_loss is not None:
                    stop_price = price * (1 - stop_loss)
                trades.append(Trade(ts, "BUY", price, qty))
        elif signal == "SELL" and position > 0:
            proceeds = position * price * (1 - fee)
            cash += proceeds
            profit = proceeds - entry_cost
            trade_profits.append(profit)
            trades.append(Trade(ts, "SELL", price, position))
            position = 0.0
            entry_cost = 0.0
            stop_price = None

        equity_curve.append(cash + position * price)

    equity = pd.Series(equity_curve, index=df.index, name="equity")

    final_equity = equity.iloc[-1] if not equity.empty else initial_cash
    periods = len(equity)
    years = periods / 252 if periods else 0
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if years else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    returns = equity.pct_change().dropna()
    sharpe = (
        returns.mean() / returns.std() * sqrt(252)
        if not returns.empty and returns.std() != 0
        else 0.0
    )

    num_trades = len(trade_profits)
    win_rate = (
        sum(p > 0 for p in trade_profits) / num_trades if num_trades else 0.0
    )
    avg_profit = sum(trade_profits) / num_trades if num_trades else 0.0

    summary = {
        "final_equity": final_equity,
        "CAGR": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "avg_profit_per_trade": avg_profit,
    }

    trades_df = pd.DataFrame(trades)
    return summary, equity, trades_df
