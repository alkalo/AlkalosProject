# debug_train_shapes.py
import pandas as pd
from src.ml.train import build_features
from src.ml.train import temporal_train_test_split  # si lo definiste como hotfix en train.py

CSV = "data/BTC_USD_1d_labeled.csv"
WINDOW = 30

df = pd.read_csv(CSV).sort_values("timestamp").copy()
print("rows csv:", len(df), "cols:", list(df.columns))

X, y, feats = build_features(
    df, feature_type="lags", feature_set="lags", window=WINDOW, target_col="target"
)
print("after build_features -> X:", X.shape, "y:", y.shape, "n_feats:", len(feats))

# si tienes 'date' en el CSV, úsala; si no, crea una para el split
if "date" in df.columns:
    dates = pd.to_datetime(df["date"])
else:
    dates = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

Xtr, Xte, ytr, yte, dtr, dte = temporal_train_test_split(X, y, dates, test_size=0.2)
print("split -> X_train:", Xtr.shape, "X_test:", Xte.shape)
