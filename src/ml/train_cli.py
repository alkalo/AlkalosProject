# src/ml/train_cli.py
import argparse
from pathlib import Path
import inspect
from datetime import datetime, timezone
import json

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier

from src.ml.data_utils import build_features
from src.utils.features_io import save_features_json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Ruta CSV con OHLCV o con features")
    p.add_argument("--symbol", required=True)
    p.add_argument("--model", choices=["lgbm"], default="lgbm")
    p.add_argument("--feature_set", choices=["lags", "tech", "both"], default="lags")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2, help="porción temporal para test (0-1)")
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

def _supports_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = ensure_datetime_index(df)

    # Detecta si CSV ya trae features
    feature_like = [c for c in df.columns if any(s in c.lower() for s in ["lag", "rsi", "ema", "sma", "macd"])]
    has_features = len(feature_like) > 0

    if has_features:
        if "target" not in df.columns:
            raise ValueError("CSV con features debe incluir 'target'")
        y = df["target"].astype(int)
        X = df.drop(columns=["target"])
        meta = {"feature_set": args.feature_set, "window": args.window}
        if _supports_param(build_features, "horizon"):
            meta["horizon"] = args.horizon
    else:
        kwargs = {"feature_set": args.feature_set, "window": args.window}
        if _supports_param(build_features, "horizon"):
            kwargs["horizon"] = args.horizon
        X, y, meta = build_features(df.copy(), **kwargs)

    if len(X) == 0:
        raise ValueError("No hay filas válidas tras construir features. Revisa datos o usa un window menor.")

    # Split temporal 80/20 (sin shuffle)
    n = len(X)
    n_test = max(1, int(n * args.test_size))
    n_train = n - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # Métricas
    y_pred_tr = clf.predict(X_train_scaled); y_prob_tr = clf.predict_proba(X_train_scaled)[:, 1]
    y_pred_te = clf.predict(X_test_scaled);  y_prob_te = clf.predict_proba(X_test_scaled)[:, 1]

    def _metrics(y_true, y_pred, y_prob):
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else None
        cm = confusion_matrix(y_true, y_pred).tolist()
        acc = accuracy_score(y_true, y_pred)
        return {"roc_auc": auc, "confusion_matrix": cm, "accuracy": acc, **rep}

    metrics = {
        "train": _metrics(y_train, y_pred_tr, y_prob_tr),
        "test": _metrics(y_test, y_pred_te, y_prob_te),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Persistencia
    model_dir = Path("models") / args.symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    save_features_json(model_dir / "features.json", columns=list(X.columns), meta=meta)

    # Punto de corte (para backtest fuera de muestra)
    split_info = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_index_label": X.index[len(X_train)].isoformat() if hasattr(X.index, "isoformat") else str(X.index[len(X_train)]),
    }

    with (model_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "symbol": args.symbol,
                "model": args.model,
                "metrics": metrics,
                "features_count": int(X.shape[1]),
                "libs": {"lightgbm": "4.5.0", "sklearn": "1.6.1"},
                "meta": meta,
                "split": split_info,   # <-- NUEVO
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Guardado en: {model_dir}")
    print(f"[INFO] Train: {len(X_train)} filas | Test: {len(X_test)} filas")

if __name__ == "__main__":
    main()
