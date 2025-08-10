"""Command line utility for downloading OHLCV market data.

The script provides a small CLI that can pull daily OHLCV candles from
either the public CoinGecko API or Yahoo! Finance via :mod:`yfinance`.
Downloaded data are normalised and written to CSV files following the
pattern ``data/{symbol}_{fiat}_1d.csv``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from src.utils.env import get_data_dir, get_logs_dir
from src.utils.market_data import fetch_coingecko_ohlc, fetch_yf_ohlc


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OHLCV market data")
    parser.add_argument(
        "--source",
        choices=["coingecko", "yf"],
        required=True,
        help="Data provider to use",
    )
    parser.add_argument(
        "--symbols",
        default="BTC,ETH",
        help="Comma separated list of symbols to download",
    )
    parser.add_argument(
        "--fiat",
        default="USD",
        help="Fiat currency to quote against",
    )
    parser.add_argument(
        "--days",
        default="max",
        help="Number of days to pull (""yfinance"" period argument)",
    )
    parser.add_argument(
        "--outfile",
        default=str(get_data_dir() / "{symbol}_{fiat}_1d.csv"),
        help="Output file pattern",
    )
    return parser.parse_args()


def _ensure_dirs(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _fetch_with_retry(func: Callable[..., pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """Execute ``func`` honouring a basic rate limit and exponential backoff."""

    rate_limit = 1.0  # seconds between requests
    retries = 5
    for attempt in range(retries):
        if attempt:
            wait = 2**attempt
            logger.warning("Retrying after %.1f s", wait)
            time.sleep(wait)
        time.sleep(rate_limit)
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Fetch failed (%s)", exc)
    raise RuntimeError("Maximum retries exceeded")


def main() -> None:
    args = _parse_args()

    _ensure_dirs(str(get_logs_dir() / "data_fetch.log"))
    logging.basicConfig(
        filename=str(get_logs_dir() / "data_fetch.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fetchers: Dict[str, Callable[[str, str, str], pd.DataFrame]] = {
        "coingecko": lambda sym, fiat, days: fetch_coingecko_ohlc(sym, vs_currency=fiat, days=days),
        "yf": lambda sym, fiat, days: fetch_yf_ohlc(sym, vs_currency=fiat, days=days),
    }
    fetch = fetchers[args.source]

    for symbol in symbols:
        logger.info("Fetching %s from %s", symbol, args.source)
        try:
            df = _fetch_with_retry(fetch, symbol, args.fiat, args.days)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("Failed to fetch %s: %s", symbol, exc)
            continue

        # Ensure expected schema
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        outfile = args.outfile.format(symbol=symbol, fiat=args.fiat.upper())
        _ensure_dirs(outfile)
        try:
            df.to_csv(outfile)
        except OSError as exc:
            logger.exception("Failed to write %s: %s", outfile, exc)
        else:
            logger.info("Saved %s rows to %s", len(df), outfile)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

