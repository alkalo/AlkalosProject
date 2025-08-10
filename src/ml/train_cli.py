"""Command-line interface for model training.

This small script exposes :func:`src.ml.train.train_evaluate` through a
user-friendly CLI.  Defaults mirror the parameters used throughout the
repository and are primarily aimed at being a simple example rather than a
production ready tool.
"""

from __future__ import annotations

import argparse

from src.utils.env import get_data_dir, get_models_dir
from .train import train_evaluate


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by :func:`main`."""

    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument(
        "--model",
        choices=["lgbm", "lstm"],
        default="lgbm",
        help="Model type to train",
    )
    parser.add_argument(
        "--csv",
        default=str(get_data_dir() / "BTC_USD_1d.csv"),
        help="Path to CSV file containing the dataset",
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Trading symbol used for storing artefacts",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in days",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Window size for features",
    )
    parser.add_argument(
        "--features",
        choices=["returns", "indicators"],
        default="returns",
        help="Feature set to use",
    )
    parser.add_argument(
        "--outdir",
        default=str(get_models_dir()),
        help="Directory where artefacts will be stored",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    train_evaluate(
        csv_path=parsed.csv,
        symbol=parsed.symbol,
        model_type=parsed.model,
        horizon=parsed.horizon,
        window=parsed.window,
        feature_set=parsed.features,
        outdir=parsed.outdir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

