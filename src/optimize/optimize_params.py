# src/optimize/optimize_params.py
import argparse
import itertools
import json
import os
from tempfile import TemporaryDirectory
from src.backtest.run_backtest import run_backtest, parse_args as parse_bt_args

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0002)
    p.add_argument("--min-edge", type=float, default=0.0)
    p.add_argument("--objective", choices=["calmar", "sharpe", "sortino", "return"], default="calmar")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--outdir", default="reports/optim")
    return p.parse_args()

def score(summary_path: str, objective: str) -> float:
    with open(summary_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if objective == "calmar":
        mdd = abs(m.get("max_drawdown", 1e-9))
        return (m.get("cagr", 0.0) / mdd) if mdd > 0 else 0.0
    return m.get(objective, 0.0)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # rejilla simple (si quieres Optuna, se puede añadir luego)
    buy_range = [round(x, 2) for x in (0.50, 0.55, 0.6, 0.65, 0.7, 0.75)]
    sell_range = [round(x, 2) for x in (0.25, 0.3, 0.35, 0.4, 0.45)]
    alloc_range = [0.25, 0.5, 0.75, 1.0]

    best = None
    tried = 0
    for buy_thr, sell_thr, alloc in itertools.product(buy_range, sell_range, alloc_range):
        if sell_thr >= buy_thr:
            continue
        with TemporaryDirectory() as tmp:
            bt_ns = argparse.Namespace(
                symbol=args.symbol, csv=args.csv, fee=args.fee, slippage=args.slippage,
                buy_thr=buy_thr, sell_thr=sell_thr, min_edge=args.min_edge, horizon=args.horizon,
                initial_capital=10_000.0, allocation=alloc, outdir=tmp
            )
            run_backtest(bt_ns)
            summ_path = os.path.join(tmp, args.symbol, "summary.json")
            s = score(summ_path, args.objective)
            tried += 1
            if (best is None) or (s > best["score"]):
                best = {"score": s, "buy_thr": buy_thr, "sell_thr": sell_thr, "allocation": alloc}

    out_path = os.path.join(args.outdir, f"{args.symbol}_best.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"[OK] Optimización terminada ({tried} combinaciones).")
    print("Mejor:", best)

if __name__ == "__main__":
    main()
