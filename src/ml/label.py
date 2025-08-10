# src/ml/label.py
import numpy as np
import pandas as pd

def make_labels(df: pd.DataFrame, horizon: int = 1, fee: float = 0.001, slippage: float = 0.0002, min_edge: float = 0.0) -> pd.DataFrame:
    """
    Crea objetivo binario:
    y = 1 si retorno futuro neto > min_edge
    y = -1 si retorno futuro neto < -min_edge
    y = 0 en el resto (opcionalmente se puede filtrar luego)

    net_ret = close[t+h]/close[t]-1 - fee - slippage
    """
    out = df.copy()
    future = out["close"].shift(-horizon)
    net = (future / out["close"] - 1.0) - fee - slippage
    y = np.where(net > min_edge, 1, np.where(net < -min_edge, -1, 0))
    out["y"] = y
    out = out.iloc[:-horizon]  # quitar filas del final sin target
    return out
