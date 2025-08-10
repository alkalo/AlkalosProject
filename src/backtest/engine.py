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
    slippage: float = 0.0,
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
        Proportional slippage applied to the execution price.
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
    trade_pnls: List[float] = []

    entry_price: float | None = None

    for ts, row in df.iterrows():
        price = float(row["close"])
        signal = row.get("signal", "HOLD")

        # Check stop loss first
        if position > 0 and stop_loss is not None and entry_price is not None:
            if price <= entry_price * (1 - stop_loss):
                sell_price = price * (1 - slippage)
                cash += position * sell_price * (1 - fee)
                trade_pnls.append(sell_price * (1 - fee) - entry_price * (1 + fee))
                trades.append(Trade(ts, "SELL", sell_price, position))
                position = 0.0
                entry_price = None

        if (
            signal == "BUY"
            and cash >= price * (1 + slippage) * (1 + fee)
            and position == 0.0
        ):
            buy_price = price * (1 + slippage)
            qty = cash * risk_per_trade / (buy_price * (1 + fee))
            cash -= qty * buy_price * (1 + fee)
            position += qty
            entry_price = buy_price
            trades.append(Trade(ts, "BUY", buy_price, qty))
        elif signal == "SELL" and position > 0.0:
            sell_price = price * (1 - slippage)
            cash += position * sell_price * (1 - fee)
            if entry_price is None:
                entry_price = sell_price
            trade_pnls.append(sell_price * (1 - fee) - entry_price * (1 + fee))
            trades.append(Trade(ts, "SELL", sell_price, position))
            position = 0.0
            entry_price = None

        equity_curve.append(cash + position * price)

    equity = pd.Series(equity_curve, index=df.index, name="equity")

    final_equity = equity.iloc[-1] if not equity.empty else initial_cash
    pnl = final_equity - initial_cash
    return_pct = pnl / initial_cash if initial_cash else 0.0
    max_drawdown = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    num_trades = len(trade_pnls)
    if trade_pnls:
        win_rate = sum(p > 0 for p in trade_pnls) / len(trade_pnls)
        avg_trade = float(sum(trade_pnls) / len(trade_pnls))
    else:
        win_rate = 0.0
        avg_trade = 0.0

    summary = {
        "final_equity": final_equity,
        "pnl": pnl,
        "return_pct": return_pct,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
    }
    trades_df = pd.DataFrame(trades)
    return summary, equity, trades_df
