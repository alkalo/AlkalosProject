import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import talib
from datetime import datetime

def load_model_bundle(symbol):
    model_dir = Path("models") / symbol
    clf = joblib.load(model_dir / "model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    with open(model_dir / "features.json", "r") as f:
        fjson = json.load(f)
    return clf, scaler, fjson, model_dir

def simulate_backtest(prices, probs, args):
    equity = [1.0]
    trades = []
    position = 0
    entry_price = None

    high = prices.rolling(14).max()
    low = prices.rolling(14).min()
    close = prices
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
    adx = talib.ADX(high.values, low.values, close.values, timeperiod=14)

    for i in range(len(prices)):
        prob = float(probs[i])

        # filtro de tendencia
        if adx[i] < 20:
            position = 0
            continue

        if position == 0 and prob > args.buy_thr:
            position = 1
            entry_price = prices[i]
        elif position == 1:
            stop_loss_price = entry_price * (1 - atr[i] * 2 / entry_price)
            if prices[i] < stop_loss_price or prob < args.sell_thr:
                position = 0

        # actualizar equity
        daily_ret = (prices[i] / prices[i - 1] - 1) if i > 0 else 0
        equity.append(equity[-1] * (1 + position * daily_ret))

    return np.array(equity), trades, {"final_equity": equity[-1]}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--buy-thr", type=float, default=0.6)
    parser.add_argument("--sell-thr", type=float, default=0.4)
    parser.add_argument("--min-edge", type=float, default=0.02)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    prices = df["close"].astype(float)

    clf, scaler, fjson, model_dir = load_model_bundle(args.symbol)
    X = df[fjson["features"]]
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]

    eq, trades, summary = simulate_backtest(prices, probs, args)

    reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
    base = reports_dir / f"{args.symbol}_backtest"
    plt.plot(eq)
    plt.savefig(base.with_name(base.stem + "_equity.png"))
    with open(base.with_name(base.stem + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
