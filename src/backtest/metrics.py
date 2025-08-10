# src/backtest/metrics.py
import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min()

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    # returns diarios; ajusta periods_per_year si usas 1D real
    if returns.std(ddof=0) == 0:
        return 0.0
    sr = ((returns.mean() - rf/periods_per_year) / returns.std(ddof=0)) * np.sqrt(periods_per_year)
    return float(sr)

def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    ds = downside.std(ddof=0)
    if ds == 0:
        return 0.0
    return ((returns.mean() - rf/periods_per_year) / ds) * np.sqrt(periods_per_year)

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_ret) ** (1 / years) - 1

def annual_breakdown(equity: pd.Series) -> pd.DataFrame:
    df = equity.rename("equity").to_frame()
    df["year"] = df.index.year
    by = df.groupby("year")["equity"].agg(["first", "last"])
    by["ret"] = by["last"] / by["first"] - 1.0
    return by[["ret"]]
