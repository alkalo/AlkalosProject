import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .strategy import SignalStrategy
from . import engine


logger = logging.getLogger(__name__)


def _ensure_dirs(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a spot backtest using a signal strategy")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True, help="Trading symbol")
    parser.add_argument("--csv", required=True, help="Path to CSV file with price data")
    parser.add_argument("--fee", type=float, default=0.006, help="Trading fee proportion")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage proportion")
    parser.add_argument("--buy-thr", dest="buy_thr", type=float, default=0.6, help="Buy probability threshold")
    parser.add_argument("--sell-thr", dest="sell_thr", type=float, default=0.4, help="Sell probability threshold")
    parser.add_argument("--min-edge", dest="min_edge", type=float, default=0.02, help="Minimum edge over 0.5 to trigger a trade")
    parser.add_argument("--initial-cash", dest="initial_cash", type=float, default=1000.0, help="Initial cash for the backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _ensure_dirs("logs/run_backtest.log")
    logging.basicConfig(
        filename="logs/run_backtest.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Starting backtest for %s", args.symbol)
    try:
        df = pd.read_csv(args.csv, parse_dates=True)
    except FileNotFoundError:
        logger.error("CSV file not found: %s", args.csv)
        return
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to read %s: %s", args.csv, exc)
        return

    try:
        strategy = SignalStrategy(
            args.symbol,
            model_dir="models",
            buy_thr=args.buy_thr,
            sell_thr=args.sell_thr,
            min_edge=args.min_edge,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to initialise strategy: %s", exc)
        return

    logger.info("Generating signals")
    signals = []
    for i in range(len(df)):
        window = df.iloc[: i + 1]
        try:
            signals.append(strategy.generate_signal(window))
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Signal generation failed: %s", exc)
            return
    df["signal"] = signals

    try:
        summary, equity, trades = engine.backtest_spot(
            df,
            fee=args.fee,
            slippage=args.slippage,
            initial_cash=args.initial_cash,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Backtest failed: %s", exc)
        return

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    summary_path = reports_dir / f"{args.symbol}_summary.json"
    equity_path = reports_dir / f"{args.symbol}_equity.png"
    trades_path = reports_dir / f"{args.symbol}_trades.csv"

    logger.info("Writing reports to %s", reports_dir)
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        trades.to_csv(trades_path, index=False)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        if hasattr(equity, "plot"):
            equity.plot(ax=plt.gca())
        else:
            plt.plot(equity)
        plt.title(f"{args.symbol} Equity Curve")
        plt.tight_layout()
        plt.savefig(equity_path)
        plt.close()
    except OSError as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to write report files: %s", exc)
        return

    logger.info("Backtest completed for %s", args.symbol)


if __name__ == "__main__":
    main()
