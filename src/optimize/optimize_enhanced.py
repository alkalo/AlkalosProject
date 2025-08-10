# -*- coding: utf-8 -*-
import argparse, os, json, itertools, time, math
import pandas as pd, numpy as np
from datetime import datetime
from src.backtest.strategy_enhanced import prepare_ohlcv, Params, backtest_long_only

def drawdown(e):
    peak = np.maximum.accumulate(e)
    return e/peak - 1.0

def metrics(eq_df: pd.DataFrame):
    if eq_df.empty: return {}
    e = eq_df["equity"].astype(float).values
    ret = e[-1]/e[0]-1 if e[0]!=0 else np.nan
    rets = pd.Series(e).pct_change().dropna()
    sharpe = (rets.mean()/rets.std()*np.sqrt(252)) if rets.std()!=0 else np.nan
    dd = drawdown(e)
    maxdd = float(dd.min()) if len(dd) else np.nan
    # cagr
    ts0, ts1 = eq_df["timestamp"].iloc[0], eq_df["timestamp"].iloc[-1]
    yrs = max((ts1 - ts0).days/365.25, 1e-9)
    cagr = (e[-1]/e[0])**(1/yrs)-1 if e[0]>0 else np.nan
    calmar = (cagr/abs(maxdd)) if (not math.isnan(cagr) and maxdd<0) else np.nan
    return {"total_return":ret, "Sharpe":sharpe, "MaxDD":maxdd, "CAGR":cagr, "Calmar":calmar}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reports", default="reports")
    # grids (modifica libremente)
    ap.add_argument("--sma-fast", default="10,20,30")
    ap.add_argument("--sma-slow", default="100,200,300")
    ap.add_argument("--adx-thr",  default="15,20,25,30")
    ap.add_argument("--risk",     default="0.005,0.01,0.015")
    ap.add_argument("--sl-atr",   default="1.5,2.0,2.5")
    ap.add_argument("--ts-atr",   default="0.5,1.0,1.5")
    ap.add_argument("--fee",      default="0.0005,0.001")
    ap.add_argument("--slippage", default="0.0002,0.0005")
    ap.add_argument("--objective", choices=["calmar","sharpe","cagr"], default="calmar")
    args = ap.parse_args()

    def to_floats(s): return [float(x) for x in str(s).split(",") if x.strip()]
    def to_ints(s):   return [int(float(x)) for x in str(s).split(",") if x.strip()]

    df = prepare_ohlcv(args.csv)
    grid = list(itertools.product(
        to_ints(args.sma_fast), to_ints(args.sma_slow),
        to_floats(args.adx_thr), to_floats(args.risk),
        to_floats(args.sl_atr), to_floats(args.ts_atr),
        to_floats(args.fee), to_floats(args.slippage)
    ))

    rows = []
    for i,(sf,ss,adx_thr,risk,sl_a,ts_a,fee,slip) in enumerate(grid,1):
        if sf>=ss:  # evita combinaciones absurdas
            continue
        p = Params(
            sma_fast=sf, sma_slow=ss,
            regime_adx=14, regime_adx_thr=adx_thr,
            atr_n=14, risk_per_trade=risk,
            sl_atr=sl_a, ts_atr=ts_a, fee=fee, slippage=slip,
            initial_equity=10_000.0
        )
        eq,tr,sm = backtest_long_only(df, p)
        m = metrics(eq)
        rows.append({
            "sma_fast":sf,"sma_slow":ss,"adx_thr":adx_thr,
            "risk":risk,"sl_atr":sl_a,"ts_atr":ts_a,"fee":fee,"slippage":slip,
            **m
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("[WARN] sin resultados")
        return
    # ranking
    if args.objective=="calmar":
        res = res.sort_values(["Calmar","Sharpe","CAGR"], ascending=[False,False,False])
    elif args.objective=="sharpe":
        res = res.sort_values(["Sharpe","Calmar","CAGR"], ascending=[False,False,False])
    else:
        res = res.sort_values(["CAGR","Sharpe","Calmar"], ascending=[False,False,False])

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.reports, exist_ok=True)
    out_csv = os.path.join(args.reports, f"opt_enhanced_{args.symbol}_{ts}.csv")
    res.to_csv(out_csv, index=False)
    best = res.iloc[0].to_dict()
    with open(os.path.join(args.reports, f"opt_enhanced_best_{args.symbol}_{ts}.json"),"w",encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("[OK] guardado:", out_csv)
    print("[BEST]", best)

if __name__ == "__main__":
    main()
