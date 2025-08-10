# src/backtest/run_backtest.py
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.data import read_ohlcv_csv, add_basic_features
from src.ml.label import make_labels
from src.backtest.metrics import max_drawdown, sharpe, sortino, cagr, annual_breakdown
from src.utils.env import get_reports_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0002)
    p.add_argument("--buy-thr", type=float, default=0.6)
    p.add_argument("--sell-thr", type=float, default=0.4)
    p.add_argument("--min-edge", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--allocation", type=float, default=1.0)  # 100% por operación
    p.add_argument("--outdir", default=str(get_reports_dir()))
    return p.parse_args()

def simple_signal_from_features(df: pd.DataFrame, buy_thr: float, sell_thr: float) -> pd.Series:
    """
    Ejemplo: señal heurística basada en momentum/RSI:
    - score en [0,1] (normalizado) y comparamos con thresholds.
    Esto es un placeholder si no quieres cargar el modelo.
    """
    # score proxy: z de ret_5 + rsi scaled
    z = (df["ret_5"] - df["ret_5"].mean()) / (df["ret_5"].std(ddof=0) + 1e-9)
    rsi_s = (df["rsi_14"] - 50.0) / 50.0  # approx (-1..+1)
    score = 1/(1+np.exp(-(0.7*z + 0.3*rsi_s)))  # squash a (0..1)
    sig = np.where(score >= buy_thr, 1, np.where(score <= sell_thr, -1, 0))
    return pd.Series(sig, index=df.index, name="signal")

def run_backtest(args):
    df = read_ohlcv_csv(args.csv)
    df = add_basic_features(df)
    df = make_labels(
        df,
        horizon=args.horizon,
        fee=args.fee,
        slippage=args.slippage,
        min_edge=args.min_edge,
    )

    # Señales (si no hay modelo cargado)
    df["signal"] = simple_signal_from_features(df, args.buy_thr, args.sell_thr)

    # PnL: entrada al close de la barra señal, salida tras 'horizon' barras
    trades = []
    capital = args.initial_capital
    equity = []
    eq = capital
    pos = 0  # -1, 0, +1 (no gestion multi-posición para mantenerlo simple)
    for i in range(len(df) - args.horizon):
        row = df.iloc[i]
        px_entry = row["close"]
        sig = row["signal"]

        if sig != 0 and pos == 0:
            # abrir
            notional = eq * args.allocation
            qty = notional / px_entry
            # fees y slippage al entrar
            px_entry_eff = px_entry * (1 + args.slippage) if sig > 0 else px_entry * (1 - args.slippage)
            fee_entry = notional * args.fee
            pos = sig
            trades.append({
                "timestamp": row["timestamp"],
                "side": "BUY" if sig > 0 else "SELL",
                "price": float(px_entry_eff),
                "qty": float(qty),
                "fee": float(fee_entry),
            })
            eq -= fee_entry

        # cerrar tras horizon
        j = i + args.horizon
        if pos != 0 and j < len(df):
            exit_row = df.iloc[j]
            px_exit = exit_row["close"]
            px_exit_eff = px_exit * (1 - args.slippage) if pos > 0 else px_exit * (1 + args.slippage)
            notional = qty * px_exit_eff
            fee_exit = notional * args.fee
            pnl = (notional - fee_exit) - (trades[-1]["qty"] * trades[-1]["price"])
            eq += pnl
            trades.append({
                "timestamp": exit_row["timestamp"],
                "side": "SELL" if pos > 0 else "BUY",
                "price": float(px_exit_eff),
                "qty": float(qty),
                "fee": float(fee_exit),
                "pnl": float(pnl)
            })
            pos = 0

        equity.append(eq)

    eq_series = pd.Series(equity, index=df.iloc[:len(equity)]["timestamp"], name="equity")
    rets = eq_series.pct_change().fillna(0)

    metrics = {
        "final_equity": float(eq_series.iloc[-1]) if len(eq_series) else args.initial_capital,
        "return_total": float(eq_series.iloc[-1] / args.initial_capital - 1) if len(eq_series) else 0.0,
        "cagr": float(cagr(eq_series)),
        "sharpe": float(sharpe(rets)),
        "sortino": float(sortino(rets)),
        "max_drawdown": float(max_drawdown(eq_series)),
        "n_trades_events": int(len(trades)//2)
    }

    # export
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(trades).to_csv(
        os.path.join(outdir, f"{args.symbol}_trades.csv"), index=False
    )
    eq_series.to_frame().to_csv(
        os.path.join(outdir, f"{args.symbol}_equity.csv")
    )
    pd.Series(metrics).to_json(
        os.path.join(outdir, f"{args.symbol}_summary.json")
    )
    png_path = Path(outdir) / f"{args.symbol}_equity.png"
    try:  # pragma: no cover - best effort plotting
        import matplotlib.pyplot as plt

        eq_series.plot()
        plt.savefig(png_path)
        plt.close()
    except Exception:  # pragma: no cover - placeholder file
        png_path.write_bytes(b"")

    print(f"[OK] backtest listo en {outdir} para {args.symbol}")
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

def main():
    args = parse_args()
    run_backtest(args)

if __name__ == "__main__":
    main()
