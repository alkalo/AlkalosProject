# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# ---------------------- Indicadores ----------------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n, min_periods=n).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    tr_n = tr.rolling(n, min_periods=n).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n, min_periods=n).sum() / tr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n, min_periods=n).sum() / tr_n

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(n, min_periods=n).mean()

# ---------------------- Parámetros ----------------------
@dataclass
class Params:
    regime_adx: int = 14           # ventana ADX
    regime_adx_thr: float = 20.0   # umbral ADX
    sma_fast: int = 20
    sma_slow: int = 200
    atr_n: int = 14
    risk_per_trade: float = 0.01   # 1% equity por trade
    sl_atr: float = 2.0            # stop loss 2*ATR
    ts_atr: float = 1.0            # trailing stop 1*ATR
    fee: float = 0.001             # 10 bps
    slippage: float = 0.0005       # 5 bps
    initial_equity: float = 10_000.0

# ---------------------- Utilidades ----------------------
def prepare_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "timestamp" in cols_lower:
        ts = cols_lower["timestamp"]
    elif "date" in cols_lower:
        ts = cols_lower["date"]
    else:
        df["timestamp"] = pd.date_range("2000-01-01", periods=len(df), freq="D")
        ts = "timestamp"

    df[ts] = pd.to_datetime(df[ts], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in cols_lower:  # pragma: no cover
            raise ValueError(f"Falta columna {c} en {csv_path}")
        df[cols_lower[c]] = pd.to_numeric(df[cols_lower[c]], errors="coerce")

    if ts != "timestamp":
        df = df.rename(columns={ts: "timestamp"})

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

# ---------------------- Estrategia ----------------------
def backtest_long_only(df: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    df["sma_fast"] = sma(df["close"], p.sma_fast)
    df["sma_slow"] = sma(df["close"], p.sma_slow)
    df["atr"] = atr(df, p.atr_n)
    df["adx"] = adx(df, p.regime_adx)
    df["regime_ok"] = (df["sma_fast"] > df["sma_slow"]) & (df["adx"] >= p.regime_adx_thr)

    equity = p.initial_equity
    pos_size = 0.0
    entry_px = np.nan
    stop_px = np.nan
    trail_px = np.nan
    last_ts = None

    equities = []
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = float(row["close"])
        this_atr = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0

        # actualizar trailing
        if pos_size > 0 and this_atr > 0:
            trail_px = max(trail_px, price - p.ts_atr * this_atr)

        # salidas
        exit_reason = None
        if pos_size > 0:
            hit_stop = (price <= stop_px) if not math.isnan(stop_px) else False
            hit_trail = (price <= trail_px) if not math.isnan(trail_px) else False
            regime_off = not bool(row["regime_ok"])
            if hit_stop:
                exit_reason = "SL"
            elif hit_trail:
                exit_reason = "TS"
            elif regime_off:
                exit_reason = "RegimeOff"

            if exit_reason:
                pnl_gross = (price - entry_px) * pos_size
                cost = (entry_px + price) * pos_size * (p.fee + p.slippage)
                pnl_net = pnl_gross - cost
                equity += pnl_net
                trades.append({
                    "entry_time": last_ts if last_ts is not None else ts,
                    "exit_time": ts,
                    "entry_price": entry_px,
                    "exit_price": price,
                    "size": pos_size,
                    "pnl": pnl_net,
                    "reason": exit_reason
                })
                pos_size = 0.0
                entry_px = np.nan
                stop_px = np.nan
                trail_px = np.nan

        # entradas
        can_enter = (pos_size == 0) and bool(row["regime_ok"]) and (this_atr > 0)
        if can_enter:
            risk_money = equity * p.risk_per_trade
            dist = p.sl_atr * this_atr
            units = max(risk_money / dist, 0.0) if dist > 0 else 0.0
            if units > 0:
                pos_size = units
                entry_px = price
                stop_px = price - p.sl_atr * this_atr
                trail_px = price - p.ts_atr * this_atr
                entry_cost = entry_px * pos_size * (p.fee + p.slippage)
                equity -= entry_cost

        equities.append({"timestamp": ts, "equity": equity})
        last_ts = ts

    eq_df = pd.DataFrame(equities)
    if not eq_df.empty:
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)

    tr_df = pd.DataFrame(trades)

    summary = {}
    if not eq_df.empty:
        e = eq_df["equity"].astype(float).values
        total_ret = e[-1] / e[0] - 1.0
        rets = pd.Series(e).pct_change().dropna()
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() != 0 else np.nan
        sortino = (rets.mean() / rets[rets < 0].std() * np.sqrt(252)) if rets[rets < 0].std() != 0 else np.nan
        dd = e / np.maximum.accumulate(e) - 1.0
        max_dd = float(dd.min()) if len(dd) else np.nan
        ts0 = eq_df["timestamp"].iloc[0]
        ts1 = eq_df["timestamp"].iloc[-1]
        days = (ts1 - ts0).days
        yrs = days / 365.25 if days > 0 else np.nan
        cagr = (e[-1] / e[0]) ** (1 / yrs) - 1 if isinstance(yrs, float) and yrs > 0 else np.nan
        summary = {
            "final_equity": float(e[-1]),
            "return_total": float(total_ret),
            "cagr": None if math.isnan(cagr) else float(cagr),
            "sharpe": None if math.isnan(sharpe) else float(sharpe),
            "sortino": None if math.isnan(sortino) else float(sortino),
            "max_drawdown": None if math.isnan(max_dd) else float(max_dd),
            "n_trades_events": int(len(tr_df)),
        }
    return eq_df, tr_df, summary
