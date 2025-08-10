# src/backtest/run_backtest.py
import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
import inspect

import pandas as pd
import numpy as np
import joblib

from src.utils.features_io import load_features_json
from src.ml.data_utils import build_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", required=True, help="OHLCV o features")
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--buy-thr", type=float, default=0.6)
    p.add_argument("--sell-thr", type=float, default=0.4)
    p.add_argument("--min-edge", type=float, default=0.0)
    p.add_argument("--scope", choices=["test", "full"], default="test", help="Periodo a backtestear")
    return p.parse_args()


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["timestamp", "date", "Datetime", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df = df.set_index(col).sort_index()
            return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    return df.sort_index()


def load_model_bundle(symbol: str):
    model_dir = Path("models") / symbol
    clf = joblib.load(model_dir / "model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    fjson = load_features_json(model_dir / "features.json")
    # cargamos report.json para obtener n_train
    with (model_dir / "report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    n_train = int(report.get("split", {}).get("n_train", 0))
    return clf, scaler, fjson, model_dir, n_train


def ensure_features(df_raw: pd.DataFrame, fjson: dict) -> pd.DataFrame:
    cols = fjson["columns"]
    if set(cols).issubset(set(df_raw.columns)):
        return df_raw.loc[:, cols].copy()

    feature_set = fjson.get("feature_set", "lags")
    window = int(fjson.get("window", 5))
    kwargs = {"feature_set": feature_set, "window": window}

    try:
        if "horizon" in fjson and "horizon" in inspect.signature(build_features).parameters:
            kwargs["horizon"] = int(fjson.get("horizon", 1))
    except Exception:
        pass

    X, *_ = build_features(df_raw.copy(), **kwargs)
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Faltan columnas tras build_features: {missing}")
    return X.loc[:, cols].copy()


def simulate_backtest(prices: pd.Series, probs: np.ndarray, args) -> tuple[pd.Series, list, dict]:
    equity = []
    trades = []
    cash = 1.0
    position = 0
    entry = None

    for i in range(len(prices)):
        price = float(prices.iloc[i])
        prob = float(probs[i])
        dt = prices.index[i]

        edge = abs(prob - 0.5)
        if edge >= args.min_edge:
            if prob >= args.buy_thr and position <= 0:
                if position == -1 and entry is not None:
                    pnl = (entry - price) / entry
                    pnl -= (args.fee + args.slippage) * 2
                    cash *= (1 + pnl)
                    trades.append({"datetime": dt.isoformat(), "side": "BUY_to_close_short", "price": price, "equity": cash})
                position = 1
                entry = price
                trades.append({"datetime": dt.isoformat(), "side": "BUY", "price": price, "equity": cash})
            elif prob <= args.sell_thr and position >= 0:
                if position == 1 and entry is not None:
                    pnl = (price - entry) / entry
                    pnl -= (args.fee + args.slippage) * 2
                    cash *= (1 + pnl)
                    trades.append({"datetime": dt.isoformat(), "side": "SELL_to_close_long", "price": price, "equity": cash})
                position = -1
                entry = price
                trades.append({"datetime": dt.isoformat(), "side": "SELL", "price": price, "equity": cash})

        if position == 1 and entry is not None:
            mtm = (price - entry) / entry
            equity.append(cash * (1 + mtm))
        elif position == -1 and entry is not None:
            mtm = (entry - price) / entry
            equity.append(cash * (1 + mtm))
        else:
            equity.append(cash)

    if position != 0 and entry is not None:
        final_price = float(prices.iloc[-1])
        pnl = (final_price - entry) / entry if position == 1 else (entry - final_price) / entry
        pnl -= (args.fee + args.slippage) * 2
        cash *= (1 + pnl)
        trades.append({"datetime": prices.index[-1].isoformat(), "side": "FLAT", "price": final_price, "equity": cash})

    summary = {
        "start": prices.index[0].isoformat(),
        "end": prices.index[-1].isoformat(),
        "n_bars": int(len(prices)),
        "final_equity": cash,
        "return_pct": (cash - 1.0) * 100.0,
        "trades": len([t for t in trades if t["side"] in ("BUY", "SELL")]),
        "params": {
            "buy_thr": args.buy_thr, "sell_thr": args.sell_thr, "min_edge": args.min_edge,
            "fee": args.fee, "slippage": args.slippage,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return pd.Series(equity, index=prices.index, name="equity"), trades, summary


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = ensure_datetime_index(df)
    if "close" not in df.columns:
        raise ValueError("El CSV debe contener 'close'.")

    clf, scaler, fjson, _, n_train = load_model_bundle(args.symbol)
    X = ensure_features(df.copy(), fjson)
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    probs_all = clf.predict_proba(X_scaled)[:, 1]
    prices_all = df["close"].astype(float).reindex(X.index)

    mask = prices_all.notna()
    prices_all = prices_all[mask]
    probs_all = probs_all[mask.values]

    if args.scope == "test" and n_train > 0 and n_train < len(prices_all):
        prices = prices_all.iloc[n_train:]
        probs = probs_all[n_train:]
    else:
        prices = prices_all
        probs = probs_all

    eq, trades, summary = simulate_backtest(prices, probs, args)

    reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
    base = args.symbol + ("_test" if args.scope == "test" else "")

    plt.figure()
    eq.plot()
    plt.title(f"Equity Curve - {args.symbol} ({args.scope})")
    plt.xlabel("Time"); plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{base}_equity.png")

    pd.DataFrame(trades).to_csv(reports_dir / f"{base}_trades.csv", index=False)

    with (reports_dir / f"{base}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] backtest ({args.scope}) listo en /reports para {args.symbol}")


if __name__ == "__main__":
    main()
