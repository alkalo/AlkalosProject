# inspect_csv.py
import sys, pandas as pd

path = sys.argv[1] if len(sys.argv)>1 else "data/BTC_USD_1d.csv"
df = pd.read_csv(path)
print("Archivo:", path)
print("Columnas:", list(df.columns))
print("Tipos:")
print(df.dtypes)
print("Primeras filas:")
print(df.head(3))
print("Nulos por columna:")
print(df.isna().sum())
if "timestamp" in df.columns:
    try:
        ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        print("timestamp (ms) válidos:", ts.notna().sum(), "/", len(ts))
        print("rango fechas:", ts.min(), "→", ts.max())
    except Exception as e:
        print("timestamp no convertible a ms:", e)
