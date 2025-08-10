"""Paper trading bot utilizing risk utilities."""

from typing import List

from risk_utils import apply_fee, daily_kill_switch, position_sizing


class PaperBot:
    """A minimal paper trading bot that tracks equity and trades returns."""

    def __init__(self, starting_equity: float, fee_rate: float = 0.0):
        self.equity = starting_equity
        self.fee_rate = fee_rate
        self.equity_curve: List[float] = [starting_equity]

    def trade(self, fractional_return: float, stop_distance_frac: float = 0.01,
              risk_per_trade: float = 0.005) -> float:
        """Execute a trade and update internal equity."""
        size = position_sizing(self.equity, risk_per_trade, stop_distance_frac)
        pnl = size * fractional_return
        pnl_after_fee = apply_fee(pnl, self.fee_rate)
        self.equity += pnl_after_fee
        self.equity_curve.append(self.equity)
        return self.equity

    def should_stop_trading(self, max_dd: float = 0.02) -> bool:
        """Return True if drawdown exceeds the limit."""
        return daily_kill_switch(self.equity_curve, max_dd)

