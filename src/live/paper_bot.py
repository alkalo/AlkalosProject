"""Paper trading bot for live data using a trained ``SignalStrategy``.

The bot periodically reads the most recent data from a CSV file, generates a
trading signal and simulates trades on a paper account.  Portfolio snapshots
are appended to ``reports/`` and actions are logged to
``logs/paper_bot.log``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.backtest.strategy import SignalStrategy

FEE_RATE = 0.006
SLIPPAGE = 0.0005


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Build command line parser."""

    parser = argparse.ArgumentParser(description="Run paper trading bot")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument(
        "--interval-minutes", type=int, default=60, help="Polling interval in minutes"
    )
    parser.add_argument(
        "--window", type=int, default=30, help="Lookback window length"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV with market data")
    return parser.parse_args()


def _ensure_dirs(path: str) -> None:
    """Create parent directories for ``path`` if necessary."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_window(path: str, window: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the last ``window`` rows of ``path`` and the latest row."""

    df = pd.read_csv(path)
    df = df.tail(window)
    return df, df.iloc[-1]


def main() -> None:  # pragma: no cover - CLI entry point
    args = _parse_args()

    _ensure_dirs("logs/paper_bot.log")
    logging.basicConfig(
        filename="logs/paper_bot.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Starting paper bot for %s", args.symbol)
    try:
        strat = SignalStrategy(args.symbol)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to load strategy: %s", exc)
        return
    cash = 10_000.0
    asset_qty = 0.0
    equity = cash

    report_path = Path("reports") / f"paper_bot_{args.symbol}.csv"
    _ensure_dirs(str(report_path))
    write_header = not report_path.exists()

    daily_high = equity
    last_day = None

    while True:
        try:
            df_window, last = _read_window(args.csv, args.window)
        except FileNotFoundError:
            logger.error("Market data file not found: %s", args.csv)
            time.sleep(args.interval_minutes * 60)
            continue
        except pd.errors.EmptyDataError:
            logger.warning("Market data file empty: %s", args.csv)
            time.sleep(args.interval_minutes * 60)
            continue
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to read market data: %s", exc)
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
        elif signal == "SELL" and asset_qty > 0:
            trade_price = price * (1 - SLIPPAGE)
            proceeds = asset_qty * trade_price * (1 - FEE_RATE)
            cash += proceeds
            asset_qty = 0.0

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
        else:
            write_header = False

        now = pd.Timestamp.utcnow()
        if last_day is None or now.date() != last_day:
            daily_high = equity
            last_day = now.date()
        daily_high = max(daily_high, equity)
        if equity < daily_high * (1 - 0.02):
            logger.warning("Daily drawdown exceeded 2%%, pausing 24h")
            time.sleep(24 * 60 * 60)
            daily_high = equity
            last_day = pd.Timestamp.utcnow().date()

        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
