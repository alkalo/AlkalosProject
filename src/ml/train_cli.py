# src/ml/train_cli.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from src.utils.data import read_ohlcv_csv, add_basic_features, train_val_test_split_time, save_json
from src.ml.label import make_labels

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Ruta CSV OHLCV (p.ej. data/BTC_USDT_1d.csv)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--min-edge", type=float, default=0.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0002)
    p.add_argument("--model", choices=["logreg"], default="logreg")  # simple y estable
    p.add_argument("--outdir", default="models")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) datos
    df = read_ohlcv_csv(args.csv)
    df = add_basic_features(df)
    df = make_labels(df, horizon=args.horizon, fee=args.fee, slippage=args.slippage, min_edge=args.min_edge)

    # Filtrar "no-trade" si quieres binario estricto (1 vs -1)
    df = df[df["y"] != 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No hay ejemplos tras etiquetado; revisa min-edge/horizon.")

    # 2) split temporal
    train, val, test = train_val_test_split_time(df, 0.7, 0.15)

    features = [c for c in df.columns if c not in ("timestamp", "y")]
    Xtr, ytr = train[features].values, train["y"].values
    Xv, yv   = val[features].values,   val["y"].values
    Xte, yte = test[features].values,  test["y"].values

    # 3) scaler sin leakage
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xv_s  = scaler.transform(Xv)
    Xte_s = scaler.transform(Xte)

    # 4) modelo (logistic regression multinomial, robusta y rápida)
    if args.model == "logreg":
        clf = LogisticRegression(max_iter=1000, multi_class="auto", n_jobs=None)
    clf.fit(Xtr_s, ytr)

    # 5) evaluación
    val_rep = classification_report(yv, clf.predict(Xv_s), output_dict=True, zero_division=0)
    test_rep = classification_report(yte, clf.predict(Xte_s), output_dict=True, zero_division=0)

    # 6) guardado de artefactos
    outdir = os.path.join(args.outdir, args.symbol)
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(clf, os.path.join(outdir, "model.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))

    save_json({"features": features, "horizon": args.horizon, "fee": args.fee, "slippage": args.slippage,
               "min_edge": args.min_edge}, os.path.join(outdir, "features.json"))
    save_json({"val": val_rep, "test": test_rep}, os.path.join(outdir, "report.json"))

    print(f"[OK] Modelo entrenado y guardado en {outdir}")
    print(f"Val F1 (1): {val_rep.get('1',{}).get('f1-score',0):.3f} | Test F1 (1): {test_rep.get('1',{}).get('f1-score',0):.3f}")

if __name__ == "__main__":
    main()
