# -*- coding: utf-8 -*-
"""
Entrenador simple para AlkalosProject.
- Lee un CSV OHLCV (columns: timestamp|date, open, high, low, close, volume)
- Genera features básicas con ventana (--window)
- Target: retorno de la próxima vela > 0 (clasificación binaria)
- Modelos: logreg (sklearn), lgbm (si lightgbm está instalado)
- Guarda modelo y summary en /models y /reports

Uso:
python -m src.ml.train_cli --model lgbm --csv data/BTC_USDT_1d.csv --symbol BTC --horizon 1 --window 5
"""
import argparse
import os
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

def infer_ts_col(df):
    for c in ["timestamp","date","datetime","time"]:
        if c in df.columns:
            return c
    return None

def make_features(df: pd.DataFrame, window: int):
    out = df.copy()
    # precios como float
    for c in ["open","high","low","close","volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["close"]).reset_index(drop=True)

    # retornos
    out["ret1"] = out["close"].pct_change()
    # medias y std rolling
    out[f"ret_mean_{window}"] = out["ret1"].rolling(window).mean()
    out[f"ret_std_{window}"]  = out["ret1"].rolling(window).std()
    # momentum simple
    out[f"mom_{window}"] = out["close"] / out["close"].shift(window) - 1.0
    # volatilidad
    out[f"vol_{window}"] = (out["high"] - out["low"]) / out["close"]

    out = out.dropna().reset_index(drop=True)
    return out

def build_target(df: pd.DataFrame, horizon: int):
    # Binaria: 1 si retorno futuro > 0
    df = df.copy()
    df["fwd_ret"] = df["close"].pct_change(periods=horizon).shift(-horizon)
    df["y"] = (df["fwd_ret"] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

def split_train_test(df: pd.DataFrame, test_frac=0.2):
    n = len(df)
    ntest = max(int(n * test_frac), 1)
    ntrain = n - ntest
    train = df.iloc[:ntrain].copy()
    test  = df.iloc[ntrain:].copy()
    return train, test

def train_logreg(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    clf.fit(Xtr, ytr)
    return clf

def train_lgbm(Xtr, ytr):
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError(f"LightGBM no disponible: {e}")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    clf.fit(Xtr, ytr)
    return clf

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min-edge", type=float, default=0.0, help="(no usado aquí, reservado)")
    ap.add_argument("--fee", type=float, default=0.0, help="(no usado aquí, reservado)")
    ap.add_argument("--slippage", type=float, default=0.0, help="(no usado aquí, reservado)")
    ap.add_argument("--model", choices=["logreg","lgbm"], default="logreg")
    ap.add_argument("--window", type=int, default=5, help="ventana para features")
    ap.add_argument("--outdir", default="models")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Cargar CSV
    df = pd.read_csv(args.csv)
    tscol = infer_ts_col(df)
    if tscol is None:
        # crea columna timestamp sintética si no existe
        df["timestamp"] = pd.date_range(end=datetime.now(timezone.utc), periods=len(df), freq="D")
        tscol = "timestamp"
    else:
        df[tscol] = pd.to_datetime(df[tscol], errors="coerce")
    df = df.dropna(subset=[tscol, "close"]).reset_index(drop=True)

    # Features y target
    df = make_features(df, window=args.window)
    df = build_target(df, horizon=args.horizon)

    feature_cols = [c for c in df.columns if c not in [tscol,"y","fwd_ret"]]
    X = df[feature_cols].values
    y = df["y"].values

    # Split temporal
    train, test = split_train_test(df, test_frac=0.2)
    Xtr, ytr = train[feature_cols].values, train["y"].values
    Xte, yte = test[feature_cols].values,  test["y"].values

    # Entrenar
    if args.model == "logreg":
        model = train_logreg(Xtr, ytr)
        model_name = "logreg"
    else:
        model = train_lgbm(Xtr, ytr)
        model_name = "lgbm"

    # Métricas simples de clasificación
    from sklearn.metrics import accuracy_score, roc_auc_score
    pte = model.predict_proba(Xte)[:,1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
    yhat = (pte >= 0.5).astype(int)
    acc = float(accuracy_score(yte, yhat))
    try:
        auc = float(roc_auc_score(yte, pte))
    except Exception:
        auc = float("nan")

    # Guardar modelo
    model_path = os.path.join(args.outdir, f"{args.symbol}_{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols, "window": args.window, "horizon": args.horizon}, f)

    # Summary entrenamiento
    summary = {
        "symbol": args.symbol,
        "model": model_name,
        "window": args.window,
        "horizon": args.horizon,
        "n_samples": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "acc_test": round(acc, 4),
        "auc_test": None if np.isnan(auc) else round(auc, 4),
        "model_path": model_path
    }
    out_json = os.path.join("reports", f"train_summary_{args.symbol}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] train listo:", summary)

if __name__ == "__main__":
    main()
