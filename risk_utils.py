"""Risk management utility functions."""

from typing import Sequence


def apply_fee(amount: float, fee_rate: float) -> float:
    """Return *amount* after deducting a fee.

    Parameters
    ----------
    amount: float
        Original amount in USD.
    fee_rate: float
        Fee rate as a fraction (e.g. 0.001 for 0.1%).
    """
    return amount * (1 - fee_rate)


def position_sizing(equity: float, risk_per_trade: float = 0.005,
                    stop_distance_frac: float = 0.01) -> float:
    """Return target position size in USD.

    The size is computed as the capital at risk divided by the
    fractional stop distance.
    """
    if stop_distance_frac <= 0:
        raise ValueError("stop_distance_frac must be positive")
    risk_capital = equity * risk_per_trade
    return risk_capital / stop_distance_frac


def daily_kill_switch(equity_series: Sequence[float], max_dd: float = 0.02) -> bool:
    """Return True if drawdown from the peak exceeds *max_dd*.

    Parameters
    ----------
    equity_series: Sequence[float]
        Historical equity values for the current day.
    max_dd: float
        Maximum allowed drawdown as a fraction.
    """
    if not equity_series:
        return False

    peak = equity_series[0]
    for value in equity_series:
        if value > peak:
            peak = value
    drawdown = (peak - equity_series[-1]) / peak
    return drawdown >= max_dd

