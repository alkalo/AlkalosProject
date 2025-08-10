from dataclasses import dataclass
from typing import List

@dataclass
class Trade:
    action: str
    price: float
    fee: float


def run_engine(prices: List[float], signals: List[int], fee: float = 0.0) -> List[Trade]:
    """Simple trading engine.

    Parameters
    ----------
    prices : List[float]
        List of prices corresponding to each signal.
    signals : List[int]
        Trading signals: 1 for buy, -1 for sell, 0 for hold.
    fee : float
        Fee percentage applied to each trade.

    Returns
    -------
    List[Trade]
        List of executed trades.
    """
    trades: List[Trade] = []
    for price, signal in zip(prices, signals):
        if signal == 1:
            trades.append(Trade("buy", price, price * fee))
        elif signal == -1:
            trades.append(Trade("sell", price, price * fee))
    return trades
