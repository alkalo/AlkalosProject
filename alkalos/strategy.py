from typing import Iterable, List

def threshold_strategy(indicator: Iterable[float], buy_threshold: float, sell_threshold: float) -> List[int]:
    """Generate trading signals based on thresholds.

    Parameters
    ----------
    indicator : Iterable[float]
        Sequence of indicator values.
    buy_threshold : float
        Values above this threshold trigger a buy signal (1).
    sell_threshold : float
        Values below this threshold trigger a sell signal (-1).

    Returns
    -------
    List[int]
        Generated signals for each indicator value.
    """
    signals: List[int] = []
    for value in indicator:
        if value > buy_threshold:
            signals.append(1)
        elif value < sell_threshold:
            signals.append(-1)
        else:
            signals.append(0)
    return signals
