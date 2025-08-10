import argparse
import json
import logging

import pandas as pd

from .strategy import SignalStrategy
from src.backtest.engine import backtest_spot
from src.utils.env import get_models_dir, get_reports_dir
from src.utils.logging_config import setup_logging


logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--window-size",
        dest="window_size",
        type=int,
        default=0,
        help="Window size for signal generation (0 uses full history)",
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="Path to JSON config file specifying thresholds and window size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override CLI options with values from configuration file if provided
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("Failed to load config file %s: %s", args.config, exc)
            return
        args.buy_thr = cfg.get("buy_thr", args.buy_thr)
        args.sell_thr = cfg.get("sell_thr", args.sell_thr)
        args.min_edge = cfg.get("min_edge", args.min_edge)
        args.window_size = cfg.get("window", cfg.get("window_size", args.window_size))

    setup_logging("run_backtest")

    logger.info("Starting backtest for %s", args.symbol)
    try:
        df = pd.read_csv(args.csv, parse_dates=True)
    except FileNotFoundError:
        logger.error("CSV file not found: %s", args.csv)
        return
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to read %s: %s", args.csv, exc)
        return

    model_base = get_models_dir() / args.symbol
    try:
        with open(model_base / "features.json", "r", encoding="utf-8") as fh:
            feature_names = json.load(fh)
    except FileNotFoundError:
        logger.error("features.json not found for %s", args.symbol)
        return
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to load features.json: %s", exc)
        return

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        logger.error("CSV missing required features: %s", ", ".join(missing))
        return

    feature_df = df[feature_names]

    try:
        costs = (args.fee + args.slippage) * 2
        strategy = SignalStrategy(
            args.symbol,
            model_dir=str(get_models_dir()),
            buy_thr=args.buy_thr,
            sell_thr=args.sell_thr,
            min_edge=args.min_edge,
            costs=costs,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to initialise strategy: %s", exc)
        return

    logger.info("Generating signals")
    signals = []
    window_size = args.window_size
    for i in range(len(feature_df)):
        if window_size and window_size > 0:
            start = max(0, i + 1 - window_size)
            window = feature_df.iloc[start : i + 1]
        else:
            window = feature_df.iloc[: i + 1]
        try:
            signals.append(strategy.generate_signal(window))
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Signal generation failed: %s", exc)
            return
    df["signal"] = signals

    try:
        summary, equity, trades = backtest_spot(
            df,
            fee=args.fee,
            slippage=args.slippage,
            initial_cash=args.initial_cash,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Backtest failed: %s", exc)
        return

    reports_dir = get_reports_dir()
    reports_dir.mkdir(exist_ok=True)

    summary_path = reports_dir / f"{args.symbol}_summary.json"
    equity_path = reports_dir / f"{args.symbol}_equity.png"
    trades_path = reports_dir / f"{args.symbol}_trades.csv"

    logger.info("Writing reports to %s", reports_dir)
    try:
        # Ensure all summary values are JSON serialisable
        serialisable_summary = {
            k: (float(v) if hasattr(v, "__float__") else v) for k, v in summary.items()
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable_summary, fh, indent=2)
        trades.to_csv(trades_path, index=False)

        import matplotlib
        matplotlib.use("Agg")
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
