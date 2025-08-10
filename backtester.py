import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union


def _generate_signals(df: pd.DataFrame, model_bundle, params) -> pd.Series:
    """Generate trading signals from ``model_bundle``.

    ``model_bundle`` can be a callable or an object implementing ``get_signals``
    or ``predict``. The result should be a ``pd.Series`` aligned with ``df``
    index containing 1 for long and 0 for flat positions.
    """
    if hasattr(model_bundle, "get_signals"):
        signals = model_bundle.get_signals(df, params)
    elif hasattr(model_bundle, "predict"):
        signals = model_bundle.predict(df, params)
    elif callable(model_bundle):
        signals = model_bundle(df, params)
    else:
        raise ValueError("model_bundle must be callable or implement get_signals/predict")

    if isinstance(signals, pd.DataFrame):
        if "signal" in signals.columns:
            signals = signals["signal"]
        else:
            raise ValueError("DataFrame signals must contain a 'signal' column")
    signals = pd.Series(signals).reindex(df.index).fillna(method="ffill").fillna(0)
    return signals.astype(int)


def _backtest_spot(
    df: pd.DataFrame,
    signals: pd.Series,
    fee_rate: float = 0.006,
    initial_cash: float = 1000.0,
    slippage: float = 0.0005,
) -> Tuple[Dict[str, float], List[Dict[str, float]], pd.Series]:
    """Run a simple spot backtest using provided signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with a ``close`` column and datetime index.
    signals : pd.Series
        Series with desired position: 1 for long, 0 for flat.
    fee_rate : float
        Fee rate per trade.
    initial_cash : float
        Starting capital in quote currency.
    slippage : float
        Fractional slippage to apply on trade prices.

    Returns
    -------
    metrics : dict
        Calculated performance metrics.
    trades : list
        List of executed trades.
    equity_curve : pd.Series
        Equity value for each bar.
    """
    cash = initial_cash
    asset_qty = 0.0
    position = 0

    trades: List[Dict[str, float]] = []
    equity_curve: List[float] = []

    last_buy: Dict[str, float] = {}

    for date, row in df.iterrows():
        price = row["close"]
        desired_pos = int(signals.loc[date])

        if position == 0 and desired_pos == 1:
            buy_price = price * (1 + slippage)
            qty = cash / buy_price if buy_price > 0 else 0
            fee = qty * buy_price * fee_rate
            cost = qty * buy_price + fee
            cash -= cost
            asset_qty += qty
            position = 1
            last_buy = {"price": buy_price, "qty": qty, "fee": fee}
            trades.append({"date": date, "side": "buy", "price": buy_price, "qty": qty, "fee": fee})
        elif position == 1 and desired_pos == 0:
            sell_price = price * (1 - slippage)
            qty = asset_qty
            revenue = qty * sell_price
            fee = revenue * fee_rate
            cash += revenue - fee
            asset_qty = 0
            position = 0
            trades.append({"date": date, "side": "sell", "price": sell_price, "qty": qty, "fee": fee})

        equity = cash + asset_qty * price
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=df.index)

    # Performance metrics
    total_return = equity_series.iloc[-1] / initial_cash - 1

    cumulative_max = equity_series.cummax()
    drawdowns = equity_series / cumulative_max - 1
    max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0

    # Trade statistics
    trade_returns: List[float] = []
    for i in range(1, len(trades), 2):
        buy = trades[i - 1]
        sell = trades[i]
        cost = buy["price"] * buy["qty"] + buy["fee"]
        proceeds = sell["price"] * sell["qty"] - sell["fee"]
        trade_returns.append((proceeds - cost) / cost if cost else 0.0)

    num_trades = len(trade_returns)
    win_rate = float(np.mean([tr > 0 for tr in trade_returns])) if trade_returns else 0.0
    avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0

    returns = equity_series.pct_change().dropna()
    sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if not returns.empty else 0.0

    metrics = {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "num_trades": num_trades,
        "sharpe": sharpe,
    }

    return metrics, trades, equity_series


def run_backtest_for_symbol(
    csv_path: str,
    model_bundle,
    params: Dict[str, Union[int, float, str]],
    fee_rate: float = 0.006,
    initial_cash: float = 1000.0,
    slippage: float = 0.0005,
) -> Dict[str, float]:
    """Run backtest for a CSV file using ``model_bundle`` strategy.

    Parameters
    ----------
    csv_path : str
        Path to OHLCV CSV. Must contain a ``close`` column and ``date`` column.
    model_bundle : callable or object
        Used to generate trading signals from data and ``params``.
    params : dict
        Parameters supplied to ``model_bundle``.
    fee_rate : float, optional
        Trading fee rate per operation.
    initial_cash : float, optional
        Starting capital.
    slippage : float, optional
        Fractional slippage for executions.

    Returns
    -------
    dict
        Backtest performance metrics.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    signals = _generate_signals(df, model_bundle, params)

    metrics, _trades, _equity = _backtest_spot(
        df, signals, fee_rate=fee_rate, initial_cash=initial_cash, slippage=slippage
    )

    return metrics
