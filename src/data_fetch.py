#!/usr/bin/env python3
"""Fetch daily cryptocurrency data from various sources.

This script reads configuration from ``settings.json`` (if present) and
fetches historical price data for the provided symbols. The resulting CSV
files are sorted by date, deduplicated, and stored in the configured data
folder. Logs are written to ``logs/data_fetch.log``.

Usage example::

    python src/data_fetch.py --source coingecko --symbols BTC,ETH --fiat USD --days max

The default output pattern is ``data/{symbol}_{fiat}_1d.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable, Dict, Iterable, List

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_settings(path: str = "settings.json") -> Dict[str, str]:
    """Load configuration from ``settings.json`` if it exists."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


SETTINGS = load_settings()
DATA_DIR = Path(SETTINGS.get("data_dir", "data"))
LOG_DIR = Path(SETTINGS.get("log_dir", "logs"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "data_fetch.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_with_retry(request_func: Callable[[], requests.Response], retries: int = 5, backoff: float = 1.0) -> requests.Response:
    """Execute ``request_func`` with exponential backoff."""
    delay = backoff
    for attempt in range(1, retries + 1):
        try:
            response = request_func()
            response.raise_for_status()
            return response
        except Exception as exc:  # pylint: disable=broad-except
            if attempt == retries:
                logger.error("Request failed after %s attempts: %s", attempt, exc)
                raise
            logger.warning("Request error: %s. Retrying in %.1fs", exc, delay)
            time.sleep(delay)
            delay *= 2


def sort_deduplicate(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Sort by date and remove duplicate dates."""
    unique: Dict[str, Dict[str, str]] = {}
    for row in rows:
        unique[row["date"]] = row
    return sorted(unique.values(), key=lambda r: r["date"])


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def coingecko_id(symbol: str) -> str:
    """Resolve symbol to CoinGecko coin id."""
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = fetch_with_retry(lambda: requests.get(url, timeout=10))
    data = response.json()
    lower = symbol.lower()
    for coin in data:
        if coin.get("symbol") == lower:
            return coin["id"]
    raise ValueError(f"Symbol {symbol!r} not found on CoinGecko")


def fetch_coingecko(symbol: str, fiat: str, days: str) -> List[Dict[str, str]]:
    coin_id = coingecko_id(symbol)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": fiat.lower(), "days": days, "interval": "daily"}

    response = fetch_with_retry(lambda: requests.get(url, params=params, timeout=10))
    prices = response.json().get("prices", [])
    rows = []
    for ts, price in prices:
        date = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000))
        rows.append({"date": date, "price": f"{price:.8f}"})
    return rows


def fetch_yahoo(symbol: str, fiat: str, days: str) -> List[Dict[str, str]]:
    y_symbol = f"{symbol}-{fiat}"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_symbol}"
    params = {"range": days, "interval": "1d"}
    response = fetch_with_retry(lambda: requests.get(url, params=params, timeout=10))
    result = response.json()["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])

    rows = []
    for ts, price in zip(timestamps, closes):
        if price is None:
            continue
        date = time.strftime("%Y-%m-%d", time.gmtime(ts))
        rows.append({"date": date, "price": f"{price:.8f}"})
    return rows


FETCHERS = {
    "coingecko": fetch_coingecko,
    "yf": fetch_yahoo,
}


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch cryptocurrency data")
    parser.add_argument("--source", choices=FETCHERS.keys(), required=True, help="Data source")
    parser.add_argument("--symbols", default="BTC,ETH", help="Comma-separated symbols")
    parser.add_argument("--fiat", default="USD", help="Fiat currency")
    parser.add_argument("--days", default="max", help="Number of days or 'max'")
    parser.add_argument(
        "--outfile",
        default="data/{symbol}_{fiat}_1d.csv",
        help="Output file pattern",
    )
    return parser.parse_args(argv)


def save_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_deduplicate(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "price"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", path)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fiat = args.fiat.upper()
    fetcher = FETCHERS[args.source]

    for symbol in symbols:
        logger.info("Fetching %s from %s", symbol, args.source)
        rows = fetcher(symbol, fiat, args.days)
        outfile = Path(args.outfile.format(symbol=symbol, fiat=fiat))
        save_csv(rows, outfile)
        time.sleep(1)  # rate limiting between symbols


if __name__ == "__main__":
    main()
