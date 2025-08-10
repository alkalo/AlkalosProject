import pandas as pd

for symbol in ["BTC", "ETH"]:
    infile = f"data/{symbol}_USD_1d.csv"
    outfile = f"data/{symbol}_USD_1d_labeled.csv"

    df = pd.read_csv(infile)
    df = df.sort_values("timestamp")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date.astype(str)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    df.to_csv(outfile, index=False)
    print(f"{symbol}: {len(df)} filas guardadas en {outfile}")
