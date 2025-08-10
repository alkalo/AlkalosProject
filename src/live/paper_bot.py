"""Trading bot for live data using a trained ``SignalStrategy``.

The bot can operate in *paper* mode, simulating trades on a virtual account,
or in *live* mode by sending orders to a real exchange client.  The program
periodically reads the most recent data from a CSV file, generates a trading
signal and executes trades accordingly.  Portfolio snapshots are appended to
``reports/`` and actions are logged to ``logs/paper_bot.log``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.backtest.strategy import SignalStrategy
from src.utils.env import get_reports_dir
from src.utils.logging_config import setup_logging
from src.utils.notify import notify

FEE_RATE = 0.006
SLIPPAGE = 0.0005


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Build command line parser."""

    parser = argparse.ArgumentParser(description="Run trading bot")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTC/USDT")
    parser.add_argument("--csv", required=True, help="Path to CSV with market data")
    parser.add_argument(
        "--interval-minutes", type=int, default=60, help="Polling interval in minutes"
    )
    parser.add_argument(
        "--window", type=int, default=30, help="Lookback window length"
    )
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper", help="Trading mode"
    )
    parser.add_argument(
        "--exchange", default="binance", help="Exchange name for live mode"
    )
    parser.add_argument(
        "--max-allocation",
        type=float,
        default=1.0,
        help="Fraction of equity to allocate per trade",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.02,
        help="Daily drawdown fraction triggering kill switch",
    )
    return parser.parse_args()


def _ensure_dirs(path: str) -> None:
    """Create parent directories for ``path`` if necessary."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_window(path: str, window: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the last ``window`` rows of ``path`` and the latest row."""

    df = pd.read_csv(path)
    df = df.tail(window)
    return df, df.iloc[-1]


class RiskManager:
    """Track equity highs and stop when drawdown exceeds a threshold."""

    def __init__(self, max_drawdown: float):
        self.max_drawdown = max_drawdown
        self.daily_high: float | None = None
        self.last_day: pd.Timestamp | None = None

    def check(self, equity: float, now: pd.Timestamp) -> bool:
        """Return ``True`` to continue trading, ``False`` to stop."""

        if self.last_day is None or now.date() != self.last_day.date():
            self.daily_high = equity
            self.last_day = now
        assert self.daily_high is not None
        self.daily_high = max(self.daily_high, equity)
        if equity < self.daily_high * (1 - self.max_drawdown):
            logger.warning(
                "Daily drawdown exceeded %.2f%%", self.max_drawdown * 100
            )
            return False
        return True


def _init_exchange_client(exchange: str):  # pragma: no cover - thin wrapper
    """Return a ccxt client for ``exchange`` using env credentials."""

    import ccxt  # type: ignore

    cls = getattr(ccxt, exchange)
    return cls(
        {
            "apiKey": os.getenv("EXCHANGE_API_KEY"),
            "secret": os.getenv("EXCHANGE_API_SECRET"),
        }
    )


def _fetch_balances(client, symbol: str) -> Tuple[float, float]:
    """Return cash and asset quantity from the exchange."""

    base, quote = symbol.split("/")
    bal = client.fetch_balance()
    cash = float(bal[quote]["free"])
    asset_qty = float(bal[base]["free"])
    return cash, asset_qty

def quantize_order(exchange, symbol, amount, price=None):
    market = exchange.market(symbol)
    prec_amt = market["precision"].get("amount", 8)
    amount_q = float(exchange.amount_to_lots(symbol, round(amount, prec_amt)))
    if price is not None:
        prec_px = market["precision"].get("price", 8)
        price_q = float(exchange.price_to_precision(symbol, round(price, prec_px)))
    else:
        price_q = None
    return amount_q, price_q


def main() -> None:  # pragma: no cover - CLI entry point
    args = _parse_args()

    setup_logging("paper_bot")

    logger.info("Starting %s bot for %s", args.mode, args.symbol)
    try:
        costs = (FEE_RATE + SLIPPAGE) * 2
        strat = SignalStrategy(args.symbol, costs=costs)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to load strategy: %s", exc)
        notify(f"Failed to load strategy: {exc}")
        return
    risk = RiskManager(args.max_drawdown)

    if args.mode == "live":
        try:  # pragma: no cover - best effort logging
            client = _init_exchange_client(args.exchange)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to init exchange client: %s", exc)
            return
        cash = asset_qty = 0.0
    else:
        cash = 10_000.0
        asset_qty = 0.0
    equity = cash

    report_path = get_reports_dir() / f"paper_bot_{args.symbol}.csv"
    _ensure_dirs(str(report_path))
    write_header = not report_path.exists()

    while True:
        try:
            df_window, last = _read_window(args.csv, args.window)
        except FileNotFoundError:
            logger.error("Market data file not found: %s", args.csv)
            notify(f"Market data file not found: {args.csv}")
            time.sleep(args.interval_minutes * 60)
            continue
        except pd.errors.EmptyDataError:
            logger.warning("Market data file empty: %s", args.csv)
            notify(f"Market data file empty: {args.csv}")
            time.sleep(args.interval_minutes * 60)
            continue
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to read market data: %s", exc)
            notify(f"Failed to read market data: {exc}")
            time.sleep(args.interval_minutes * 60)
            continue

        try:
            X = df_window[strat.feature_names].values
            if strat.scaler is not None:
                X = strat.scaler.transform(X)
            if strat.is_lstm:
                X = X.reshape(1, X.shape[0], X.shape[1])
            p_up = float(strat.predict_proba_last(X))
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Prediction failed: %s", exc)
            notify(f"Prediction failed: {exc}")
            time.sleep(args.interval_minutes * 60)
            continue
        if p_up >= strat.buy_thr and (p_up - 0.5) >= strat.min_edge:
            signal = "BUY"
        elif p_up <= strat.sell_thr and (0.5 - p_up) >= strat.min_edge:
            signal = "SELL"
        else:
            signal = "HOLD"

        price = float(last["close"])
        if signal == "BUY" and cash > 0:
            trade_price = price * (1 + SLIPPAGE)
            qty = (cash * (1 - FEE_RATE)) / trade_price
            asset_qty += qty
            cash = 0.0
            notify(f"BUY {qty:.6f} {args.symbol} at {trade_price:.2f}")
        elif signal == "SELL" and asset_qty > 0:
            trade_price = price * (1 - SLIPPAGE)
            proceeds = asset_qty * trade_price * (1 - FEE_RATE)
            notify(f"SELL {asset_qty:.6f} {args.symbol} at {trade_price:.2f}")
            cash += proceeds
            asset_qty = 0.0


        if args.mode == "live":
            try:
                cash, asset_qty = _fetch_balances(client, args.symbol)
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.exception("Failed to fetch balances: %s", exc)
                time.sleep(args.interval_minutes * 60)
                continue


        equity = cash + asset_qty * price

        if signal == "BUY" and cash > 0:
            trade_value = min(cash, equity * args.max_allocation)
            if trade_value > 0:
                if args.mode == "paper":
                    trade_price = price * (1 + SLIPPAGE)
                    qty = (trade_value * (1 - FEE_RATE)) / trade_price
                    asset_qty += qty
                    cash -= trade_value
                else:
                    qty = trade_value / price
                    try:  # pragma: no cover - best effort logging
                        client.create_market_buy_order(args.symbol, qty)
                    except Exception as exc:  # pragma: no cover
                        logger.exception("Live buy failed: %s", exc)
                    cash, asset_qty = _fetch_balances(client, args.symbol)
                equity = cash + asset_qty * price
        elif signal == "SELL" and asset_qty > 0:
            qty_to_sell = asset_qty * args.max_allocation
            if qty_to_sell > 0:
                if args.mode == "paper":
                    trade_price = price * (1 - SLIPPAGE)
                    proceeds = qty_to_sell * trade_price * (1 - FEE_RATE)
                    cash += proceeds
                    asset_qty -= qty_to_sell
                else:
                    try:  # pragma: no cover - best effort logging
                        client.create_market_sell_order(args.symbol, qty_to_sell)
                    except Exception as exc:  # pragma: no cover
                        logger.exception("Live sell failed: %s", exc)
                    cash, asset_qty = _fetch_balances(client, args.symbol)
                equity = cash + asset_qty * price

        logger.info(
            "price=%.2f p_up=%.4f signal=%s cash=%.2f qty=%.6f equity=%.2f",
            price,
            p_up,
            signal,
            cash,
            asset_qty,
            equity,
        )

        snapshot = pd.DataFrame(
            [
                {
                    "timestamp": last.get("timestamp")
                    or pd.Timestamp.utcnow().isoformat(),
                    "price": price,
                    "p_up": p_up,
                    "signal": signal,
                    "cash": cash,
                    "asset_qty": asset_qty,
                    "equity": equity,
                }
            ]
        )
        try:
            snapshot.to_csv(report_path, mode="a", header=write_header, index=False)
        except OSError as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to write snapshot: %s", exc)
            notify(f"Failed to write snapshot: {exc}")
        else:
            write_header = False

        now = pd.Timestamp.utcnow()
        if last_day is None or now.date() != last_day:
            daily_high = equity
            last_day = now.date()
        daily_high = max(daily_high, equity)
        if equity < daily_high * (1 - 0.02):
            msg = "Daily drawdown exceeded 2%, pausing 24h"
            logger.warning(msg)
            notify(msg)
            time.sleep(24 * 60 * 60)
            daily_high = equity
            last_day = pd.Timestamp.utcnow().date()

        if not risk.check(equity, now):
            break

        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
