# src/data_fetch/__main__.py
# -*- coding: utf-8 -*-
"""
Data fetcher para AlkalosProject.
- CCXT: descarga OHLCV paginando hasta cubrir `--days`
- Yahoo Finance: fallback principalmente para USD (BTC-USD, ETH-USD, etc.)
Guarda CSVs en data/{SYMBOL}_{FIAT}_{TIMEFRAME}.csv con columnas:
[timestamp, timestamp_ms, open, high, low, close, volume]
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

def log(msg, level="INFO"):
    print(f"[{level}] {msg}", flush=True)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def dt_to_ms(dt):
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)

# ------------------------------
# CCXT
# ------------------------------
def fetch_ccxt(exchange_name, symbols, fiat, days, timeframe, outdir, retries, sleep_s):
    try:
        import ccxt  # noqa
    except Exception as e:
        log(f"No se pudo importar ccxt: {e}", "ERROR")
        sys.exit(1)

    if not hasattr(__import__('ccxt'), exchange_name):
        log(f"Exchange '{exchange_name}' no soportado por ccxt.", "ERROR")
        sys.exit(1)

    ex = getattr(__import__('ccxt'), exchange_name)({
        "enableRateLimit": True,
    })

    log(f"Cargando markets de {exchange_name}...")
    ex.load_markets()

    # since -> ahora - days
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    since = dt_to_ms(since_dt)
    end = now_ms()

    tf_to_ms = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
    }
    if timeframe not in tf_to_ms:
        log(f"timeframe '{timeframe}' no soportado en este script.", "ERROR")
        sys.exit(1)

    ensure_dir(outdir)

    for raw_sym in symbols:
        sym = raw_sym.strip().upper()
        market_symbol = f"{sym}/{fiat.upper()}"

        if market_symbol not in ex.markets:
            log(f"{exchange_name} no tiene el par {market_symbol}. Saltando.", "WARN")
            continue

        log(f"Descargando {market_symbol} ({timeframe}) últimos {days} días...")
        header = ["timestamp", "timestamp_ms", "open", "high", "low", "close", "volume"]
        rows = []
        cursor = since
        step = tf_to_ms[timeframe] * 900  # pedir ~900 velas por página (seguro para muchos exchanges)

        for attempt in range(retries):
            try:
                # Paginación simple hasta cubrir el rango [since, now]
                while cursor < end:
                    # Algunos exchanges ignoran 'limit'; el troceo por 'since' es el patrón robusto
                    data = ex.fetch_ohlcv(market_symbol, timeframe=timeframe, since=cursor, limit=None)
                    if not data:
                        break

                    # Filtrar duplicados por timestamp ms
                    last_ms = rows[-1][1] if rows else -1
                    for ts, o, h, l, c, v in data:
                        if ts > last_ms:
                            iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                            rows.append([iso, ts, o, h, l, c, v])

                    # Avanzar cursor
                    cursor = data[-1][0] + tf_to_ms[timeframe]

                    # Respetar límites
                    time.sleep(sleep_s)

                break  # éxito → salimos de bucle de reintentos
            except Exception as e:
                wait = min(2 ** attempt, 30)
                log(f"Error en fetch_ohlcv({market_symbol}): {e}. Reintentando en {wait}s...", "WARN")
                time.sleep(wait)
        else:
            log(f"Fallo persistente descargando {market_symbol}.", "ERROR")
            continue

        if not rows:
            log(f"Sin datos recibidos para {market_symbol}.", "WARN")
            continue

        # Ordenar por timestamp_ms por si acaso
        rows.sort(key=lambda r: r[1])

        outpath = os.path.join(outdir, f"{sym}_{fiat.upper()}_{timeframe}.csv")
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        log(f"Guardado: {outpath} ({len(rows)} velas)")

# ------------------------------
# Yahoo Finance
# ------------------------------
def fetch_yf(symbols, fiat, days, timeframe, outdir):
    try:
        import yfinance as yf  # noqa
    except Exception as e:
        log(f"No se pudo importar yfinance: {e}", "ERROR")
        sys.exit(1)

    if timeframe != "1d":
        log("Aviso: yfinance en este script solo usa intervalo 1d.", "WARN")

    if fiat.upper() not in ("USD",):
        log("Aviso: yfinance crypto está pensado aquí para pares en USD (BTC-USD, etc.).", "WARN")

    ensure_dir(outdir)

    for raw_sym in symbols:
        sym = raw_sym.strip().upper()
        ticker = f"{sym}-{fiat.upper()}"
        log(f"Descargando {ticker} desde Yahoo Finance últimos {days} días...")

        try:
            df = yf.download(tickers=ticker, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
        except Exception as e:
            log(f"Error descargando {ticker}: {e}", "ERROR")
            continue

        if df is None or df.empty:
            log(f"Sin datos para {ticker}.", "WARN")
            continue

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df["timestamp"] = df.index.tz_localize(timezone.utc, nonexistent="shift_forward", ambiguous="NaT").to_pydatetime()
        df["timestamp"] = df["timestamp"].apply(lambda d: d.isoformat())
        df["timestamp_ms"] = df.index.tz_localize(timezone.utc, nonexistent="shift_forward", ambiguous="NaT").asi8 // 1_000_000

        cols = ["timestamp", "timestamp_ms", "open", "high", "low", "close", "volume"]
        df = df[cols]

        outpath = os.path.join(outdir, f"{sym}_{fiat.upper()}_{timeframe}.csv")
        df.to_csv(outpath, index=False)
        log(f"Guardado: {outpath} ({len(df)} velas)")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fetcher de datos OHLCV para AlkalosProject")
    p.add_argument("--source", choices=["ccxt", "yf"], required=True, help="Fuente de datos")
    p.add_argument("--exchange", default="binance", help="Exchange CCXT (si --source ccxt)")
    p.add_argument("--symbols", required=True, help="Símbolos separados por coma: BTC,ETH,SOL")
    p.add_argument("--fiat", default="USDT", help="Moneda cotizada: USDT (ccxt) / USD (yfinance)")
    p.add_argument("--days", type=int, default=1825, help="Días hacia atrás a descargar")
    p.add_argument("--timeframe", default="1d", help="1m,5m,15m,30m,1h,4h,1d,1w (limitado por exchange)")
    p.add_argument("--outdir", default="data", help="Directorio de salida")
    p.add_argument("--retries", type=int, default=4, help="Reintentos por símbolo (ccxt)")
    p.add_argument("--sleep", type=float, default=0.4, help="Sleep entre páginas (ccxt)")
    return p.parse_args()

def main():
    args = parse_args()
    symbols = [s for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        log("No se obtuvieron símbolos tras el split. Revisa --symbols.", "ERROR")
        sys.exit(1)

    if args.source == "ccxt":
        fetch_ccxt(
            exchange_name=args.exchange,
            symbols=symbols,
            fiat=args.fiat,
            days=args.days,
            timeframe=args.timeframe,
            outdir=args.outdir,
            retries=args.retries,
            sleep_s=args.sleep
        )
    elif args.source == "yf":
        fetch_yf(
            symbols=symbols,
            fiat=args.fiat,
            days=args.days,
            timeframe=args.timeframe,
            outdir=args.outdir
        )
    else:
        log(f"Fuente desconocida: {args.source}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
