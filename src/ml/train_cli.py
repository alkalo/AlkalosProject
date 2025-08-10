import argparse
from pathlib import Path
import inspect
from datetime import datetime
import json

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier

from src.ml.data_utils import build_features
from src.utils.features_io import save_features_json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Ruta CSV con OHLCV o con features")
    p.add_argument("--symbol", required=True)
    p.add_argument("--model", choices=["lgbm"], default="lgbm")
    p.add_argument("--feature_set", choices=["lags", "tech"], default="lags")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
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

    # ¿El CSV ya trae features? Heurística simple por nombres
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

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    clf.fit(X_scaled, y)

    # Métricas (fit metrics, rápidas)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred).tolist()

    # Persistencia
    model_dir = Path("models") / args.symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    save_features_json(model_dir / "features.json", columns=list(X.columns), meta=meta)

    with (model_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.utcnow().isoformat(),
                "symbol": args.symbol,
                "model": args.model,
                "metrics": {"roc_auc": auc, **report, "confusion_matrix": cm},
                "n_samples": int(len(X)),
                "features_count": int(X.shape[1]),
                "libs": {"lightgbm": "4.5.0", "sklearn": "1.6.1"},
                "meta": meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Guardado en: {model_dir}")

if __name__ == "__main__":
    main()
