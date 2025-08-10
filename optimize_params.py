import itertools
import subprocess
import json
from pathlib import Path

csv_path = "data/BTC_USD_1d.csv"
symbol = "BTC"

buy_range = [0.55, 0.6, 0.65]
sell_range = [0.35, 0.4, 0.45]
edge_range = [0.005, 0.01, 0.015, 0.02]

best = None

for buy, sell, edge in itertools.product(buy_range, sell_range, edge_range):
    subprocess.run([
        "python", "-m", "src.backtest.run_backtest",
        "--symbol", symbol,
        "--csv", csv_path,
        "--buy-thr", str(buy),
        "--sell-thr", str(sell),
        "--min-edge", str(edge)
    ], check=True)

    summary_file = sorted(Path("reports").glob(f"{symbol}_backtest_summary.json"))[-1]
    with open(summary_file) as f:
        summary = json.load(f)
    eq = summary.get("final_equity", 0)
    if best is None or eq > best["equity"]:
        best = {"buy": buy, "sell": sell, "edge": edge, "equity": eq}

print("Mejores parámetros encontrados:", best)
