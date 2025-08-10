"""Simple backtest module using risk utilities."""

from typing import Iterable, Tuple

from risk_utils import apply_fee, daily_kill_switch, position_sizing


def run_backtest(returns: Iterable[float], initial_equity: float = 10000,
                 fee_rate: float = 0.0, stop_distance_frac: float = 0.01,
                 risk_per_trade: float = 0.005) -> Tuple[float, list]:
    """Run a simplistic backtest over a sequence of returns.

    Parameters
    ----------
    returns: Iterable[float]
        Fractional price changes per trade (e.g. 0.01 for +1%).
    initial_equity: float
        Starting account value.
    fee_rate: float
        Fee rate applied to profit/loss.
    stop_distance_frac: float
        Stop loss distance used for position sizing.
    risk_per_trade: float
        Fraction of equity risked per trade.
    """
    equity = initial_equity
    equity_curve = [equity]

    for r in returns:
        size = position_sizing(equity, risk_per_trade, stop_distance_frac)
        pnl = size * r
        pnl_after_fee = apply_fee(pnl, fee_rate)
        equity += pnl_after_fee
        equity_curve.append(equity)
        if daily_kill_switch(equity_curve):
            break

    return equity, equity_curve

