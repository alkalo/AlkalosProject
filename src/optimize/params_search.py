# -*- coding: utf-8 -*-
"""
Grid search/WFO para backtests de AlkalosProject sin tocar lógica interna.
Lanza `python -m src.backtest.run_backtest` con combinaciones de parámetros
y consolida métricas leyendo los outputs de /reports.

Uso básico:
python -m src.optimize.params_search --symbol BTC --csv data/BTC_USDT_1d.csv \
  --buy-thr 0.55,0.6,0.65 --sell-thr 0.35,0.4 --min-edge 0.01,0.02 \
  --fee 0.0005,0.001 --slippage 0.0002,0.0005

Walk-forward simple (particiona el CSV en N splits temporales):
  --wf-splits 3

Salida principal:
  reports/optimize/results_<symbol>_<ts>.csv (ranking de combinaciones)
  reports/optimize/best_<symbol>_<ts>.json (mejor set y métricas)

Requisitos:
- Que el backtest genere en /reports los ficheros de equity y trades. Si
  no existen equity.csv/trades.csv, intenta leer summary.json; si tampoco
  existe, se calculan métricas básicas desde la curva si está disponible.
"""

import argparse
import csv
import os
import sys
import time
import glob
import json
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

REPORTS_DIR = "reports"
OPT_DIR = os.path.join(REPORTS_DIR, "optimize")
os.makedirs(OPT_DIR, exist_ok=True)


# --------------- Utilidades de métricas -----------------
def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def _calc_metrics_from_equity(eq_df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    if eq_df is None or eq_df.empty or "equity" not in eq_df.columns:
        return out
    eq_df = eq_df.copy()
    eq_df = eq_df.sort_values(eq_df.columns[0])
    equity = eq_df["equity"].astype(float)
    rets = equity.pct_change().dropna()
    # Periodo (años)
    try:
        tscol = None
        for c in ["timestamp","date","datetime"]:
            if c in eq_df.columns:
                tscol = c; break
        if tscol is not None:
            eq_df[tscol] = pd.to_datetime(eq_df[tscol], errors="coerce")
            days = (eq_df[tscol].iloc[-1] - eq_df[tscol].iloc[0]).days
            years = days / 365.25 if days and days>0 else np.nan
        else:
            years = np.nan
    except Exception:
        years = np.nan

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1 if equity.iloc[0] != 0 else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0])**(1/years) - 1 if years and years>0 else np.nan
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() not in (0, np.nan) else np.nan
    dn = rets[rets<0].std()
    sortino = (rets.mean() / dn) * np.sqrt(252) if dn not in (0, np.nan) else np.nan
    dd = _drawdown(equity)
    max_dd = dd.min() if not dd.empty else np.nan
    calmar = (cagr / abs(max_dd)) if (cagr is not np.nan and max_dd not in (0, np.nan)) else np.nan
    out.update({
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total_ret) if total_ret==total_ret else np.nan,
        "CAGR": float(cagr) if cagr==cagr else np.nan,
        "Sharpe": float(sharpe) if sharpe==sharpe else np.nan,
        "Sortino": float(sortino) if sortino==sortino else np.nan,
        "MaxDD": float(max_dd) if max_dd==max_dd else np.nan,
        "Calmar": float(calmar) if calmar==calmar else np.nan,
    })
    return out

def _calc_trade_metrics(tr_df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    if tr_df is None or tr_df.empty: 
        return out
    tr = tr_df.copy()
    # normalizar nombres
    low = {c.lower(): c for c in tr.columns}
    pnl_col = low.get("pnl") or low.get("profit") or low.get("pl")
    if pnl_col is None:
        return out
    wins = tr[pnl_col] > 0
    gross_profit = tr.loc[wins, pnl_col].sum()
    gross_loss = -tr.loc[~wins, pnl_col].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    win_rate = wins.mean() if len(tr)>0 else np.nan
    out.update({
        "n_trades": int(len(tr)),
        "win_rate": float(win_rate) if win_rate==win_rate else np.nan,
        "profit_factor": float(profit_factor) if profit_factor==profit_factor else np.nan,
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_pnl": float(gross_profit - gross_loss),
    })
    return out


# --------------- Localizar outputs más recientes -----------------
def _latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def _collect_reports_for_symbol(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Intenta localizar equity.csv, trades.csv, summary.json más recientes para el símbolo."""
    sym = symbol.upper()
    eq_csv = _latest_file(os.path.join(REPORTS_DIR, f"*{sym}*equity*.csv"))
    if not eq_csv:
        eq_csv = _latest_file(os.path.join(REPORTS_DIR, f"{sym}_equity.csv"))
    tr_csv = _latest_file(os.path.join(REPORTS_DIR, f"*{sym}*trades*.csv"))
    if not tr_csv:
        tr_csv = _latest_file(os.path.join(REPORTS_DIR, f"{sym}_trades.csv"))
    summ_json = _latest_file(os.path.join(REPORTS_DIR, f"*{sym}*summary*.json"))
    if not summ_json:
        summ_json = _latest_file(os.path.join(REPORTS_DIR, f"{sym}_summary.json"))

    eq_df = pd.read_csv(eq_csv) if eq_csv and os.path.exists(eq_csv) else pd.DataFrame()
    tr_df = pd.read_csv(tr_csv) if tr_csv and os.path.exists(tr_csv) else pd.DataFrame()
    summ = {}
    if summ_json and os.path.exists(summ_json):
        try:
            with open(summ_json, "r", encoding="utf-8") as f:
                summ = json.load(f)
        except Exception:
            summ = {}
    return eq_df, tr_df, summ


# --------------- Backtest runner -----------------
def run_backtest_once(symbol: str, csv_path: str, params: dict, python_bin: str = sys.executable) -> bool:
    """Ejecuta una vez el backtest. Devuelve True si el proceso termina con 0."""
    cmd = [
        python_bin, "-m", "src.backtest.run_backtest",
        "--symbol", symbol,
        "--csv", csv_path,
        "--fee", str(params.get("fee", 0.001)),
        "--slippage", str(params.get("slippage", 0.0005)),
        "--buy-thr", str(params.get("buy_thr", 0.6)),
        "--sell-thr", str(params.get("sell_thr", 0.4)),
        "--min-edge", str(params.get("min_edge", 0.02)),
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("[ERR] backtest fallo:", p.stderr[:500], flush=True)
        return False
    return True


# --------------- Walk-forward simple -----------------
def _make_wf_slices(csv_path: str, splits: int) -> List[str]:
    """Crea N slices temporales (conservando orden) y devuelve rutas a CSVs temporales."""
    if splits <= 1:
        return [csv_path]
    df = pd.read_csv(csv_path)
    n = len(df)
    sizes = [n // splits] * splits
    sizes[-1] += n - sum(sizes)
    paths = []
    start = 0
    base_dir = os.path.join("data", "_wf_slices")
    os.makedirs(base_dir, exist_ok=True)
    for i, sz in enumerate(sizes, 1):
        part = df.iloc[start:start+sz].copy()
        out = os.path.join(base_dir, f"{os.path.basename(csv_path).replace('.csv','')}_wf{i}.csv")
        part.to_csv(out, index=False)
        paths.append(out)
        start += sz
    return paths


# --------------- Bucle principal de optimización -----------------
def optimize(symbol: str, csv_path: str,
             buy_thr: List[float], sell_thr: List[float], min_edge: List[float],
             fee: List[float], slippage: List[float],
             wf_splits: int, objective: str) -> pd.DataFrame:
    grid = []
    for b in buy_thr:
        for s in sell_thr:
            for m in min_edge:
                for f in fee:
                    for sl in slippage:
                        grid.append(dict(buy_thr=b, sell_thr=s, min_edge=m, fee=f, slippage=sl))
    print(f"[INFO] Combinaciones totales: {len(grid)}")

    all_rows = []
    slices = _make_wf_slices(csv_path, wf_splits)

    for i, params in enumerate(grid, 1):
        print(f"\n=== {i}/{len(grid)}: {params} ===")
        wf_metrics = []
        for j, slice_path in enumerate(slices, 1):
            print(f"[WF {j}/{len(slices)}] {slice_path}")
            ok = run_backtest_once(symbol, slice_path, params)
            if not ok:
                wf_metrics.append({"Sharpe": np.nan, "Calmar": np.nan, "MaxDD": np.nan})
                continue
            eq_df, tr_df, summ = _collect_reports_for_symbol(symbol)
            metrics = _calc_metrics_from_equity(eq_df)
            metrics.update(_calc_trade_metrics(tr_df))
            wf_metrics.append(metrics)

        # Agregar las métricas de las ventanas (media o robusta)
        def _avg(key):
            vals = [m.get(key) for m in wf_metrics if m.get(key) == m.get(key)]
            return float(np.mean(vals)) if vals else np.nan
        row = {
            "symbol": symbol,
            **params,
            "Sharpe": _avg("Sharpe"),
            "Calmar": _avg("Calmar"),
            "MaxDD": _avg("MaxDD"),
            "CAGR": _avg("CAGR"),
            "total_return": _avg("total_return"),
            "profit_factor": _avg("profit_factor"),
            "win_rate": _avg("win_rate"),
            "n_trades": _avg("n_trades"),
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # Ranking según objetivo
    if objective.lower() == "calmar":
        df = df.sort_values(["Calmar","Sharpe","CAGR"], ascending=[False, False, False])
    elif objective.lower() == "sharpe":
        df = df.sort_values(["Sharpe","Calmar","CAGR"], ascending=[False, False, False])
    else:
        df = df.sort_values(["CAGR","Sharpe","Calmar"], ascending=[False, False, False])
    return df


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--buy-thr", default="0.55,0.6,0.65")
    ap.add_argument("--sell-thr", default="0.35,0.4,0.45")
    ap.add_argument("--min-edge", default="0.01,0.02,0.03")
    ap.add_argument("--fee", default="0.0005,0.001,0.002")
    ap.add_argument("--slippage", default="0.0002,0.0005,0.001")
    ap.add_argument("--wf-splits", type=int, default=1, help="N ventanas walk-forward simple")
    ap.add_argument("--objective", default="calmar", choices=["calmar","sharpe","cagr"])
    args = ap.parse_args()

    buy_thr = parse_list_floats(args["buy_thr"] if isinstance(args, dict) else args.buy_thr)
    sell_thr = parse_list_floats(args["sell_thr"] if isinstance(args, dict) else args.sell_thr)
    min_edge = parse_list_floats(args["min_edge"] if isinstance(args, dict) else args.min_edge)
    fee = parse_list_floats(args["fee"] if isinstance(args, dict) else args.fee)
    slippage = parse_list_floats(args["slippage"] if isinstance(args, dict) else args.slippage)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    df = optimize(
        symbol=args.symbol,
        csv_path=args.csv,
        buy_thr=buy_thr,
        sell_thr=sell_thr,
        min_edge=min_edge,
        fee=fee,
        slippage=slippage,
        wf_splits=args.wf_splits,
        objective=args.objective,
    )
    out_csv = os.path.join(OPT_DIR, f"results_{args.symbol}_{ts}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Resultados guardados en: {out_csv}")

    # Mejor combo
    if not df.empty:
        best = df.iloc[0].to_dict()
        best_json = os.path.join(OPT_DIR, f"best_{args.symbol}_{ts}.json")
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"[OK] Mejor set guardado en: {best_json}")
    else:
        print("[WARN] No se pudieron calcular resultados (df vacío).")


if __name__ == "__main__":
    main()
