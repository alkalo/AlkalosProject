# src/data_fetch.py
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time
import contextlib

import pandas as pd

# ---------- utils ----------
def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas a: timestamp, open, high, low, close, volume (en ese orden)."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        raise ValueError("No hay columna de tiempo (timestamp/date).")

    # tipos numéricos
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    return df

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=False)

def _map_crypto_fiat_for_ccxt(symbol: str, fiat: str) -> tuple[str, str]:
    """Para cripto, muchos exchanges cotizan en USDT en vez de USD."""
    s = symbol.upper()
    q = fiat.upper()
    if s in {"BTC","ETH","SOL","ADA","XRP","DOGE","BNB","MATIC","AVAX"} and q == "USD":
        return s, "USDT"
    return s, q

# ---------- fuentes ----------
def fetch_yfinance(symbol: str, fiat: str, days: int, retries: int = 2, sleep_s: float = 1.5) -> pd.DataFrame:
    import yfinance as yf
    ticker = f"{symbol.upper()}-{fiat.upper()}"

    last_err = None
    for attempt in range(retries + 1):
        try:
            if days <= 730:
                period = f"{max(days,1)}d"
                data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
            else:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=days)
                data = yf.download(ticker, start=_iso(start)[:10], end=_iso(end)[:10], interval="1d", auto_adjust=False, progress=False)
            if data is not None and len(data) > 0:
                data = data.reset_index()
                data = data.rename(columns={"Date":"timestamp","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                return _ensure_columns(data)
        except Exception as e:
            last_err = e
        time.sleep(sleep_s)
    raise RuntimeError(f"yfinance devolvió 0 filas para {ticker}" + (f" | último error: {last_err}" if last_err else ""))

def fetch_ccxt(symbol: str, fiat: str, days: int, exchange_name: str = "binance") -> pd.DataFrame:
    import ccxt
    base, quote = _map_crypto_fiat_for_ccxt(symbol, fiat)
    market = f"{base}/{quote}"
    exch_cls = getattr(ccxt, exchange_name)
    ex = exch_cls({"enableRateLimit": True})
    timeframe = "1d"
    millis_per_day = 24 * 60 * 60 * 1000
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    limit = 1000
    while True:
        batch = ex.fetch_ohlcv(market, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        since = batch[-1][0] + millis_per_day
        time.sleep(ex.rateLimit / 1000.0)
    if not all_rows:
        raise RuntimeError(f"ccxt devolvió 0 filas para {market} en {exchange_name}")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return _ensure_columns(df)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["yf", "ccxt"], default="yf")
    p.add_argument("--symbols", default="BTC", help="Símbolo base, e.g., BTC")
    p.add_argument("--fiat", default="USD", help="Moneda de cotización, e.g., USD")
    p.add_argument("--days", type=int, default=365, help="Días hacia atrás a descargar")
    p.add_argument("--exchange", default="binance", help="Exchange para ccxt (si source=ccxt)")
    p.add_argument("--outdir", default="data", help="Directorio de salida")
    p.add_argument("--fallback", action="store_true", help="Si YF falla, intenta CCXT automáticamente")
    return p.parse_args()

def main():
    args = parse_args()
    sym = args.symbols.upper()
    fiat = args.fiat.upper()
    outdir = Path(args.outdir)
    outfile = outdir / f"{sym}_{fiat}_1d.csv"

    df = None
    if args.source == "yf":
        with contextlib.suppress(Exception):
            df = fetch_yfinance(sym, fiat, args.days)
        if (df is None or df.empty) and args.fallback:
            print("[WARN] yfinance falló. Intentando CCXT (Binance)...", file=sys.stderr)
            df = fetch_ccxt(sym, fiat, args.days, exchange_name=args.exchange)
    else:
        df = fetch_ccxt(sym, fiat, args.days, exchange_name=args.exchange)

    if df is None or df.empty or df["close"].isna().all():
        raise RuntimeError(f"Datos vacíos para {sym}-{fiat} ({'yfinance' if args.source=='yf' else 'ccxt'}).")

    _write_csv(df, outfile)
    start = df["timestamp"].iloc[0]; end = df["timestamp"].iloc[-1]
    print(f"[OK] {sym}-{fiat} 1D: {len(df)} filas  | {start} → {end}")
    print(f"[OK] CSV: {outfile.resolve()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
