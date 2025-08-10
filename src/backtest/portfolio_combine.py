# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np

def load_eq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], errors="coerce", utc=True)
    df = df.rename(columns={ts: "timestamp"})
    df = df[["timestamp", "equity"]].dropna().sort_values("timestamp").reset_index(drop=True)
    return df

def vol_weighted_combine(paths, lookback=90):
    eqs = [load_eq(p).set_index("timestamp")["equity"] for p in paths]
    df = pd.concat(eqs, axis=1).dropna()
    df.columns = [f"a{i}" for i in range(len(paths))]
    rets = df.pct_change().dropna()
    vol = rets.rolling(lookback).std().replace(0, np.nan)
    weights = (1.0 / vol)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(1.0 / len(paths))
    port_rets = (rets * weights.shift(1)).sum(axis=1).fillna(0.0)
    port_eq = (1.0 + port_rets).cumprod()
    out = port_eq.to_frame("equity").reset_index()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC,ETH")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--enhanced", action="store_true")
    ap.add_argument("--lookback", type=int, default=90)
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    paths = []
    for s in syms:
        p = os.path.join(args.reports, f"{s}_equity_enhanced.csv" if args.enhanced else f"{s}_equity.csv")
        if os.path.exists(p):
            paths.append(p)

    if len(paths) < 2:
        print("[WARN] Se necesitan ≥2 equity.csv para combinar.")
        return

    port = vol_weighted_combine(paths, lookback=args.lookback)
    out = os.path.join(args.reports, "portfolio_equity.csv")
    port.to_csv(out, index=False)
    print(f"[OK] Portfolio combinado guardado en: {out}")

if __name__ == "__main__":
    main()
