"""Market data utilities.

Functions to fetch OHLCV data from CoinGecko and Yahoo Finance (yfinance),
handle persistence to CSV, and provide basic symbol mappings.

All returned DataFrames use UTC based ``timestamp`` index and the standard
columns: ``open``, ``high``, ``low``, ``close`` and ``volume``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests
import yfinance as yf


def get_symbol_mapping() -> Dict[str, Dict[str, str]]:
    """Return mapping for common crypto symbols.

    The mapping contains both CoinGecko ids and Yahoo Finance tickers.
    """
    return {
        "BTC": {"coingecko": "bitcoin", "yfinance": "BTC-USD"},
        "ETH": {"coingecko": "ethereum", "yfinance": "ETH-USD"},
        # expose direct access with yfinance tickers as well
        "BTC-USD": {"coingecko": "bitcoin", "yfinance": "BTC-USD"},
        "ETH-USD": {"coingecko": "ethereum", "yfinance": "ETH-USD"},
    }


def _coingecko_market_chart(coin_id: str, fiat: str, days: str | int) -> Dict[str, Any]:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    resp = requests.get(url, params={"vs_currency": fiat, "days": days}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_coingecko_ohlc(symbol: str, fiat: str, days: str | int) -> pd.DataFrame:
    """Fetch OHLCV data from CoinGecko.

    Attempts to use the dedicated OHLC endpoint and augment with volume data.
    If that fails, falls back to synthesising OHLC from ``market_chart`` data.
    """
    mapping = get_symbol_mapping()
    coin_id = mapping.get(symbol, {}).get("coingecko", symbol.lower())

    # Try dedicated OHLC endpoint first
    try:
        ohlc_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        resp = requests.get(
            ohlc_url, params={"vs_currency": fiat, "days": days}, timeout=30
        )
        resp.raise_for_status()
        ohlc_data = resp.json()
        if ohlc_data:
            ohlc_df = pd.DataFrame(
                ohlc_data, columns=["timestamp", "open", "high", "low", "close"]
            )
            ohlc_df["timestamp"] = pd.to_datetime(
                ohlc_df["timestamp"], unit="ms", utc=True
            )
            ohlc_df.set_index("timestamp", inplace=True)
            ohlc_df = ohlc_df.resample("1D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            )

            # Fetch volume via market_chart
            market = _coingecko_market_chart(coin_id, fiat, days)
            volumes = market.get("total_volumes", [])
            vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            if not vol_df.empty:
                vol_df["timestamp"] = pd.to_datetime(
                    vol_df["timestamp"], unit="ms", utc=True
                )
                vol_df.set_index("timestamp", inplace=True)
                vol_df = vol_df.resample("1D").sum()
                ohlc_df["volume"] = vol_df["volume"]
            else:
                ohlc_df["volume"] = 0

            ohlc_df.index.name = "timestamp"
            return ohlc_df[["open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    # Fall back to market_chart data
    market = _coingecko_market_chart(coin_id, fiat, days)
    prices = pd.DataFrame(market.get("prices", []), columns=["timestamp", "price"])
    volumes = pd.DataFrame(market.get("total_volumes", []), columns=["timestamp", "volume"])

    if prices.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True)
    volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms", utc=True)

    df = pd.merge(prices, volumes, on="timestamp", how="left")
    df.set_index("timestamp", inplace=True)

    ohlc = df["price"].resample("1D").ohlc()
    volume = df["volume"].resample("1D").sum()

    result = ohlc.join(volume)
    result.index.name = "timestamp"
    result.columns = ["open", "high", "low", "close", "volume"]
    return result


def fetch_yf_ohlc(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance using ``yfinance``."""
    data = yf.download(ticker, interval=interval, progress=False)
    if data.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    data.index = pd.to_datetime(data.index, utc=True)
    data.index.name = "timestamp"
    return data


def save_df_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV ensuring timestamp index."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_df_csv(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from CSV produced by :func:`save_df_csv`."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index(pd.to_datetime(df["timestamp"], utc=True), inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    df.index.name = "timestamp"
    return df

