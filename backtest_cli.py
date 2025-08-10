import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_model(model_path: str, scaler_path: str):
    """Load model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def run_backtest(
    symbol: str,
    csv_path: str,
    model_path: str,
    scaler_path: str,
    features_path: str,
    fee: float,
    slippage: float,
    buy_thr: float,
    sell_thr: float,
    initial_cash: float,
):
    """Run a simple backtest using model predictions."""
    # Load data
    prices = pd.read_csv(csv_path)
    features = pd.read_csv(features_path)

    if "timestamp" in prices.columns:
        index = pd.to_datetime(prices["timestamp"])
    else:
        index = pd.RangeIndex(len(prices))
    prices.index = index
    features.index = index

    model, scaler = load_model(model_path, scaler_path)

    X = scaler.transform(features.values)
    preds = model.predict_proba(X)[:, 1]

    cash = initial_cash
    position = 0.0
    equity_curve = []
    trade_records: List[Dict[str, float]] = []

    for ts, price, pred in zip(index, prices["price" if "price" in prices.columns else prices.columns[-1]], preds):
        # update equity
        equity = cash + position * price
        equity_curve.append((ts, equity))

        if position == 0 and pred >= buy_thr:
            # buy as much as possible
            buy_price = price * (1 + slippage)
            qty = cash / (buy_price * (1 + fee))
            cash -= qty * buy_price * (1 + fee)
            position += qty
            equity = cash + position * price
            trade_records.append({
                "timestamp": ts,
                "side": "buy",
                "price": price,
                "qty": qty,
                "cash": cash,
                "equity": equity,
            })
        elif position > 0 and pred <= sell_thr:
            sell_price = price * (1 - slippage)
            cash += position * sell_price * (1 - fee)
            qty = position
            position = 0
            equity = cash
            trade_records.append({
                "timestamp": ts,
                "side": "sell",
                "price": price,
                "qty": qty,
                "cash": cash,
                "equity": equity,
            })

    # Final equity update
    final_equity = cash + position * prices.iloc[-1]["price" if "price" in prices.columns else prices.columns[-1]]

    summary = {
        "symbol": symbol,
        "final_equity": final_equity,
        "return": (final_equity - initial_cash) / initial_cash,
        "num_trades": len(trade_records),
    }

    os.makedirs("reports", exist_ok=True)

    # Save summary
    summary_path = os.path.join("reports", f"{symbol}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save trades
    trades_path = os.path.join("reports", f"{symbol}_trades.csv")
    pd.DataFrame(trade_records).to_csv(trades_path, index=False)

    # Plot equity curve with trades
    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_df["timestamp"], equity_df["equity"], label="Equity")

    # Mark trades
    for record in trade_records:
        marker = "^" if record["side"] == "buy" else "v"
        color = "green" if record["side"] == "buy" else "red"
        ax.scatter(record["timestamp"], record["equity"], marker=marker, color=color)

    ax.set_title(f"Equity Curve - {symbol}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.autofmt_xdate()
    equity_path = os.path.join("reports", f"{symbol}_equity.png")
    fig.savefig(equity_path)
    plt.close(fig)

    return summary_path, trades_path, equity_path


def parse_args():
    parser = argparse.ArgumentParser(description="Simple backtesting CLI")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    parser.add_argument("--csv", required=True, help="CSV with price data")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scaler-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--fee", type=float, default=0.006)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--buy-thr", type=float, default=0.6)
    parser.add_argument("--sell-thr", type=float, default=0.4)
    parser.add_argument("--initial-cash", type=float, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    run_backtest(
        symbol=args.symbol,
        csv_path=args.csv,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        features_path=args.features_path,
        fee=args.fee,
        slippage=args.slippage,
        buy_thr=args.buy_thr,
        sell_thr=args.sell_thr,
        initial_cash=args.initial_cash,
    )


if __name__ == "__main__":
    main()
