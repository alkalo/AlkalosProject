import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

from src.utils.features_io import load_features_json
from src.ml.data_utils import build_features  # usaremos para fallback de features

import matplotlib
matplotlib.use("Agg")  # para correr sin GUI
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", required=True, help="OHLCV o features")
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--buy-thr", type=float, default=0.6)
    p.add_argument("--sell-thr", type=float, default=0.4)
    p.add_argument("--min-edge", type=float, default=0.0, help="margen mínimo prob-0.5")
    p.add_argument("--initial-train", type=int, default=300, help="mín. barras para primer entreno interno (si aplica)")
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
    return clf, scaler, fjson

def ensure_features(df_raw: pd.DataFrame, fjson: dict) -> pd.DataFrame:
    cols = fjson["columns"]
    # si ya están, ordenamos y devolvemos
    if set(cols).issubset(set(df_raw.columns)):
        return df_raw.reindex(columns=sorted(set(df_raw.columns))).loc[:, cols].copy()

    # si no están, construimos features con el mismo feature_set/window
    feature_set = fjson.get("feature_set", "lags")
    window = int(fjson.get("window", 5))
    horizon = int(fjson.get("horizon", 1))
    X, _, _ = build_features(df_raw.copy(), feature_set=feature_set, window=window, horizon=horizon)

    # asegurar orden según features.json
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Faltan columnas tras build_features: {missing}")
    X = X.loc[:, cols].copy()
    return X

def backtest(df_price: pd.DataFrame, X: pd.DataFrame, clf, scaler, args):
    # escalado
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    equity = []
    trades = []

    cash = 1.0  # equity normalizada
    position = 0  # -1, 0, 1
    entry_price = None

    # asumimos que df_price tiene 'close'
    if "close" not in df_price.columns:
        raise ValueError("El CSV debe contener columna 'close' para simular PnL.")

    prices = df_price["close"].astype(float)
    probs = getattr(clf, "predict_proba")(X_scaled)[:, 1]

    for i in range(len(prices)):
        price = float(prices.iloc[i])
        prob = float(probs[i])
        dt = prices.index[i]

        # decisión
        edge = abs(prob - 0.5)
        action = "hold"
        if edge >= args.min_edge:
            if prob >= args.buy_thr and position <= 0:
                # cerrar corto si lo hubiera
                if position == -1 and entry_price is not None:
                    # cerrar corto → pagas fee*2 y slippage*2
                    pnl = (entry_price - price) / entry_price
                    pnl -= (args.fee + args.slippage) * 2
                    cash *= (1 + pnl)
                    trades.append({"datetime": dt.isoformat(), "side": "BUY_to_close_short", "price": price, "equity": cash})
                # abrir largo
                position = 1
                entry_price = price
                trades.append({"datetime": dt.isoformat(), "side": "BUY", "price": price, "equity": cash})
                action = "long"

            elif prob <= args.sell_thr and position >= 0:
                if position == 1 and entry_price is not None:
                    # cerrar largo
                    pnl = (price - entry_price) / entry_price
                    pnl -= (args.fee + args.slippage) * 2
                    cash *= (1 + pnl)
                    trades.append({"datetime": dt.isoformat(), "side": "SELL_to_close_long", "price": price, "equity": cash})
                # abrir corto
                position = -1
                entry_price = price
                trades.append({"datetime": dt.isoformat(), "side": "SELL", "price": price, "equity": cash})
                action = "short"

        # marca de equity mark-to-market simple (sin apalancamiento)
        if position == 1 and entry_price is not None:
            mtm = (price - entry_price) / entry_price
            equity.append(cash * (1 + mtm))
        elif position == -1 and entry_price is not None:
            mtm = (entry_price - price) / entry_price
            equity.append(cash * (1 + mtm))
        else:
            equity.append(cash)

    equity_series = pd.Series(equity, index=prices.index, name="equity")
    # cerrar posición al final con fees/slippage
    if position != 0 and entry_price is not None:
        final_price = float(prices.iloc[-1])
        if position == 1:
            pnl = (final_price - entry_price) / entry_price
        else:
            pnl = (entry_price - final_price) / entry_price
        pnl -= (args.fee + args.slippage) * 2
        cash *= (1 + pnl)
        trades.append({"datetime": prices.index[-1].isoformat(), "side": "FLAT", "price": final_price, "equity": cash})

    summary = {
        "symbol": args.symbol,
        "start": prices.index[0].isoformat(),
        "end": prices.index[-1].isoformat(),
        "n_bars": int(len(prices)),
        "final_equity": cash,
        "return_pct": (cash - 1.0) * 100.0,
        "trades": len([t for t in trades if t["side"] in ("BUY", "SELL")]),
        "params": {
            "buy_thr": args.buy_thr,
            "sell_thr": args.sell_thr,
            "min_edge": args.min_edge,
            "fee": args.fee,
            "slippage": args.slippage,
        },
        "created_at": datetime.utcnow().isoformat(),
    }
    return equity_series, trades, summary

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = ensure_datetime_index(df)

    clf, scaler, fjson = load_model_bundle(args.symbol)

    # asegurar 'close' para PnL
    if "close" not in df.columns:
        raise ValueError("El CSV debe contener 'close'.")

    # construir/validar features y reordenar según features.json
    X = ensure_features(df.copy(), fjson)

    # ejecutar backtest
    eq, trades, summary = backtest(df, X, clf, scaler, args)

    # persistir
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    base = reports_dir / f"{args.symbol}"

    # equity
    (base.with_suffix("_equity.png")).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    eq.plot()
    plt.title(f"Equity Curve - {args.symbol}")
    plt.xlabel("Time"); plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(base.with_suffix("_equity.png"))

    # trades csv
    pd.DataFrame(trades).to_csv(base.with_suffix("_trades.csv"), index=False)

    # summary json
    with (base.with_suffix("_summary.json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] backtest listo en /reports para {args.symbol}")

if __name__ == "__main__":
    main()
