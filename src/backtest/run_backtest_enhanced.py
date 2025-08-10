# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Dict

import pandas as pd

from src.backtest.strategy_enhanced import prepare_ohlcv, Params, backtest_long_only

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_reports(symbol: str, eq_df: pd.DataFrame, tr_df: pd.DataFrame, summary: Dict, reports_dir="reports"):
    ensure_dir(reports_dir)
    eq_path = os.path.join(reports_dir, f"{symbol}_equity_enhanced.csv")
    tr_path = os.path.join(reports_dir, f"{symbol}_trades_enhanced.csv")
    sm_path = os.path.join(reports_dir, f"{symbol}_summary_enhanced.json")
    eq_df.to_csv(eq_path, index=False)
    tr_df.to_csv(tr_path, index=False)
    with open(sm_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] backtest (enhanced) listo en reports para {symbol}")
    for k, v in summary.items():
        print(f"- {k}: {v}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--sma-fast", type=int, default=20)
    ap.add_argument("--sma-slow", type=int, default=200)
    ap.add_argument("--adx-n", type=int, default=14)
    ap.add_argument("--adx-thr", type=float, default=20.0)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--sl-atr", type=float, default=2.0)
    ap.add_argument("--ts-atr", type=float, default=1.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    return ap.parse_args()

def main():
    args = parse_args()
    df = prepare_ohlcv(args.csv)
    p = Params(
        regime_adx=args.adx_n,
        regime_adx_thr=args.adx_thr,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        atr_n=args.atr_n,
        risk_per_trade=args.risk_per_trade,
        sl_atr=args.sl_atr,
        ts_atr=args.ts_atr,
        fee=args.fee,
        slippage=args.slippage,
        initial_equity=args.initial_equity
    )
    symbol = args.symbol.upper()
    eq_df, tr_df, summary = backtest_long_only(df, p)
    save_reports(symbol, eq_df, tr_df, summary, args.reports)

if __name__ == "__main__":
    main()
