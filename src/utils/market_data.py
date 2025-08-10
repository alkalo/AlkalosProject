"""Helpers for working with market data sources.

This module exposes thin wrappers around external data providers that
return OHLCV data in a consistent :class:`pandas.DataFrame` format.  Both
helpers share a small request utility that implements basic rate limiting
and exponential backoff to play nicely with public APIs.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Internal request helper


def _request_with_retry(
    url: str,
    *,
    retries: int = 5,
    backoff: float = 1.0,
    rate_limit: float = 1.0,
    session: Optional[requests.Session] = None,
):
    """Retrieve ``url`` handling rate limiting and exponential backoff.

    Parameters
    ----------
    url:
        Target URL to download.
    retries:
        Number of attempts before giving up.
    backoff:
        Base backoff delay in seconds.  The actual delay grows
        exponentially (``backoff * 2**attempt``).
    rate_limit:
        Minimum number of seconds between consecutive requests.
    session:
        Optional :class:`requests.Session` to reuse TCP connections.

    Returns
    -------
    requests.Response
        Successful HTTP response.
    """

    session = session or requests.Session()
    for attempt in range(retries):
        if _request_with_retry._last_call is not None:
            delta = time.time() - _request_with_retry._last_call
            if delta < rate_limit:
                time.sleep(rate_limit - delta)
        try:
            response = session.get(url, timeout=30)
            _request_with_retry._last_call = time.time()
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - best effort logging
            wait = backoff * (2**attempt)
            logging.warning("Request failed (%s). Retrying in %.1f s", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}")


_request_with_retry._last_call = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public fetch helpers

COINGECKO_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


def fetch_coingecko_ohlc(
    symbol: str,
    *,
    vs_currency: str = "usd",
    days: str = "max",
) -> pd.DataFrame:
    """Fetch OHLCV data from the public CoinGecko API.

    The returned :class:`~pandas.DataFrame` is indexed by UTC datetimes and
    contains the columns ``timestamp``, ``open``, ``high``, ``low``,
    ``close`` and ``volume``.
    """

    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        raise ValueError(f"Unsupported symbol {symbol}")

    url = (
        "https://api.coingecko.com/api/v3/coins/"
        f"{coin_id}/market_chart?vs_currency={vs_currency.lower()}&days={days}&interval=daily"
    )

    response = _request_with_retry(url)
    data = response.json()

    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    if not prices:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df_p = pd.DataFrame(prices, columns=["timestamp", "price"])
    df_v = pd.DataFrame(volumes, columns=["timestamp", "volume"])
    df = pd.merge(df_p, df_v, on="timestamp", how="left")

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("D")
    price_grp = df.groupby("date")["price"]
    vol_grp = df.groupby("date")["volume"]
    ohlc = price_grp.agg(open="first", high="max", low="min", close="last")
    vol = vol_grp.sum()
    out = pd.concat([ohlc, vol], axis=1)
    out.index.name = None
    out["timestamp"] = (out.index.view("int64") // 10**6)
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def fetch_yf_ohlc(
    symbol: str,
    *,
    vs_currency: str = "USD",
    days: str = "max",
) -> pd.DataFrame:
    """Fetch OHLCV data using :mod:`yfinance`.

    Parameters mirror :func:`fetch_coingecko_ohlc` and the returned DataFrame
    follows the same schema.
    """

    import yfinance as yf

    ticker = f"{symbol.upper()}-{vs_currency.upper()}"
    df = yf.download(ticker, period=days, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df["timestamp"] = (df.index.view("int64") // 10**6)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


__all__ = ["fetch_coingecko_ohlc", "fetch_yf_ohlc"]


