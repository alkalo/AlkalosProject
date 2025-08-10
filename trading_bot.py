#!/usr/bin/env python3
"""Simple Coinbase trading bot for educational purposes.

The bot loads a model, scaler and feature configuration for a given symbol
from the ``artifacts/<symbol>`` directory.  It maintains a simulated
portfolio in memory and decides to BUY/SELL/HOLD based on the probability of
an upward move returned by the ``SignalStrategy``.

Every ``interval_minutes`` it retrieves the last trade price from Coinbase,
updates a price window and runs the strategy.  Executed trades and portfolio
value are logged.  Once per hour a CSV snapshot of the state is written to
``snapshots/``.

Real orders are **not** sent by default.  If ``--send-orders`` is supplied the
client will log that an order would be sent to the Coinbase sandbox.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import statistics
import time
from collections import deque
from pathlib import Path
from typing import List
from urllib.error import URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Coinbase client
# ---------------------------------------------------------------------------


class CoinbaseClient:
    """Minimal Coinbase client used for price retrieval and sandbox orders."""

    def __init__(self, base_url: str = "https://api.exchange.coinbase.com", *, send_orders: bool = False) -> None:
        self.base_url = base_url
        self.send_orders = send_orders

    def get_ticker(self, product_id: str) -> float:
        """Return the latest trade price for ``product_id``."""
        url = f"{self.base_url}/products/{product_id}/ticker"
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "trading-bot"})
        with urlopen(req) as resp:
            data = json.load(resp)
        return float(data["price"])

    def buy(self, product_id: str, funds: float) -> None:
        if self.send_orders:
            logging.info("[SANDBOX] Sending buy order for %s with funds %.2f", product_id, funds)
        else:
            logging.info("Simulated buy for %s with funds %.2f", product_id, funds)

    def sell(self, product_id: str, size: float) -> None:
        if self.send_orders:
            logging.info("[SANDBOX] Sending sell order for %s size %.6f", product_id, size)
        else:
            logging.info("Simulated sell for %s size %.6f", product_id, size)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SignalStrategy:
    """Generate buy/sell signals based on a model probability output."""

    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def p_up(self, window: List[float]) -> float:
        features = self._build_features(window)
        X = self.scaler.transform([features])
        prob = self.model.predict_proba(X)[0][1]
        return float(prob)

    def _build_features(self, window: List[float]) -> List[float]:
        feats: List[float] = []
        for name in self.feature_names:
            if name == "price":
                feats.append(window[-1])
            elif name == "mean":
                feats.append(statistics.mean(window))
            elif name == "std":
                feats.append(statistics.stdev(window) if len(window) > 1 else 0.0)
            elif name == "return":
                first = window[0]
                feats.append(((window[-1] - first) / first) if first else 0.0)
            else:
                raise ValueError(f"Unsupported feature '{name}'")
        return feats


class DummyModel:
    """Fallback model returning 50% probability for both classes."""

    def predict_proba(self, X):
        return [[0.5, 0.5]]


class DummyScaler:
    """Identity scaler used when no scaler is provided."""

    def transform(self, X):  # pragma: no cover - trivial
        return X


def load_artifacts(symbol: str):
    """Load model, scaler and feature names for ``symbol``.

    If artifacts are missing, dummy components are returned so that the bot can
    operate with placeholder behaviour.
    """

    base = Path("artifacts") / symbol
    try:
        model = pickle.load(open(base / "model.pkl", "rb"))
        scaler = pickle.load(open(base / "scaler.pkl", "rb"))
        feature_names = json.load(open(base / "features.json"))
        logging.info("Loaded artifacts for %s from %s", symbol, base)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Could not load artifacts for %s: %s. Using dummy strategy.", symbol, exc)
        model = DummyModel()
        scaler = DummyScaler()
        feature_names = ["price", "mean", "std", "return"]
    return model, scaler, feature_names


# ---------------------------------------------------------------------------
# Trading bot
# ---------------------------------------------------------------------------


class TradingBot:
    def __init__(
        self,
        symbol: str,
        interval_minutes: float,
        window_size: int,
        coinbase_client: CoinbaseClient,
        strategy: SignalStrategy,
        *,
        min_trade: float = 10.0,
        fee: float = 0.001,
        initial_cash: float = 1000.0,
    ) -> None:
        self.symbol = symbol
        self.interval = interval_minutes * 60
        self.window = deque(maxlen=window_size)
        self.client = coinbase_client
        self.strategy = strategy
        self.min_trade = min_trade
        self.fee = fee
        self.cash = initial_cash
        self.asset_qty = 0.0
        self.last_snapshot = time.time()

    def run(self, *, max_iterations: int | None = None) -> None:
        iteration = 0
        while True:
            try:
                price = self.client.get_ticker(self.symbol)
            except URLError as exc:
                logging.error("Ticker fetch failed: %s", exc)
                time.sleep(self.interval)
                continue

            self.window.append(price)
            action = "HOLD"

            if len(self.window) == self.window.maxlen:
                p_up = self.strategy.p_up(list(self.window))
                if p_up > 0.6 and self.cash > self.min_trade:
                    self._buy(price)
                    action = "BUY"
                elif p_up < 0.4 and self.asset_qty > 0:
                    self._sell(price)
                    action = "SELL"

            equity = self.cash + self.asset_qty * price
            logging.info(
                "Price: %.2f | Action: %s | Cash: %.2f | Asset: %.6f | Equity: %.2f",
                price,
                action,
                self.cash,
                self.asset_qty,
                equity,
            )

            self._maybe_snapshot(price, equity)

            iteration += 1
            if max_iterations and iteration >= max_iterations:
                break

            time.sleep(self.interval)

    def _buy(self, price: float) -> None:
        qty = (self.cash * (1 - self.fee)) / price
        self.client.buy(self.symbol, self.cash)
        self.asset_qty += qty
        self.cash = 0.0

    def _sell(self, price: float) -> None:
        proceeds = self.asset_qty * price * (1 - self.fee)
        self.client.sell(self.symbol, self.asset_qty)
        self.cash += proceeds
        self.asset_qty = 0.0

    def _maybe_snapshot(self, price: float, equity: float) -> None:
        now = time.time()
        if now - self.last_snapshot >= 3600:
            os.makedirs("snapshots", exist_ok=True)
            filename = time.strftime("snapshots/%Y%m%d%H%M%S.csv", time.gmtime(now))
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "price", "cash", "asset_qty", "equity"])
                writer.writerow([int(now), price, self.cash, self.asset_qty, equity])
            self.last_snapshot = now
            logging.info("Snapshot saved to %s", filename)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulated Coinbase trading bot")
    parser.add_argument("--symbol", required=True, help="Product symbol, e.g. BTC-USD")
    parser.add_argument(
        "--interval-minutes", type=float, default=1.0, help="Polling interval in minutes"
    )
    parser.add_argument("--window", type=int, default=10, help="Number of prices in window")
    parser.add_argument("--initial-cash", type=float, default=1000.0)
    parser.add_argument("--min-trade", type=float, default=10.0)
    parser.add_argument("--send-orders", action="store_true", help="Send orders to sandbox")
    parser.add_argument(
        "--max-iterations", type=int, help="Stop after N iterations (useful for tests)", default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
    )

    model, scaler, feature_names = load_artifacts(args.symbol)
    strategy = SignalStrategy(model, scaler, feature_names)
    client = CoinbaseClient(send_orders=args.send_orders)
    bot = TradingBot(
        args.symbol,
        args.interval_minutes,
        args.window,
        client,
        strategy,
        min_trade=args.min_trade,
        initial_cash=args.initial_cash,
    )
    bot.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
