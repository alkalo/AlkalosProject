"""Minimal backtesting engine used in unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
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
    position_pct: float = 1.0,
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
    position_pct:
        Fraction of available equity to deploy on each trade.
    stop_loss:
        Optional stop loss expressed as a decimal percentage from entry price.
    """

    cash = initial_cash
    position = 0.0
    equity_curve: list[float] = []
    trades: TradeList = []
    trade_returns: List[float] = []

    entry_price: float | None = None

    for ts, row in df.iterrows():
        price = float(row["close"])
        signal = row.get("signal", "HOLD")

        # Check stop loss first
        if position > 0 and stop_loss is not None and entry_price is not None:
            if price <= entry_price * (1 - stop_loss):
                cash += position * price * (1 - fee)
                trade_returns.append(
                    price * (1 - fee) / (entry_price * (1 + fee)) - 1
                )
                trades.append(Trade(ts, "SELL", price, position))
                position = 0.0
                entry_price = None

        if signal == "BUY" and cash >= price * (1 + fee) and position == 0.0:
            qty = cash * position_pct / (price * (1 + fee))
            cash -= qty * price * (1 + fee)
            position += qty
            entry_price = price
            trades.append(Trade(ts, "BUY", price, qty))
        elif signal == "SELL" and position > 0.0:
            cash += position * price * (1 - fee)
            if entry_price is None:
                entry_price = price
            trade_returns.append(
                price * (1 - fee) / (entry_price * (1 + fee)) - 1
            )
            trades.append(Trade(ts, "SELL", price, position))
            position = 0.0
            entry_price = None

        equity_curve.append(cash + position * price)

    equity = pd.Series(equity_curve, index=df.index, name="equity")

    final_equity = equity.iloc[-1] if not equity.empty else initial_cash
    pnl = final_equity - initial_cash
    return_pct = pnl / initial_cash if initial_cash else 0.0
    # Performance metrics
    returns = equity.pct_change().dropna()
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if not returns.empty and returns.std() != 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1 if not equity.empty else pd.Series()
    max_drawdown = -float(drawdown.min()) if not drawdown.empty else 0.0
    years = len(equity) / 252 if len(equity) > 1 else 0
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if years else 0.0

    num_trades = len(trade_returns)
    if trade_returns:
        win_rate = sum(r > 0 for r in trade_returns) / len(trade_returns)
        avg_trade_return = float(sum(trade_returns) / len(trade_returns))
    else:
        win_rate = 0.0
        avg_trade_return = 0.0

    summary = {
        "final_equity": final_equity,
        "pnl": pnl,
        "return_pct": return_pct,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
    }
    trades_df = pd.DataFrame(trades)
    return summary, equity, trades_df
