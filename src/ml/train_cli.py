import argparse
from .train import train_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train financial models")
    parser.add_argument("--model", choices=["lgbm", "lstm"], required=True)
    parser.add_argument("--csv", required=True, help="Path to CSV data file")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--symbol", required=False, help="Asset symbol")
    args = parser.parse_args()

    metrics = train_evaluate(
        model_type=args.model,
        csv_path=args.csv,
        horizon=args.horizon,
        window=args.window,
        outdir=args.outdir,
        symbol=args.symbol,
    )
    print(metrics)


if __name__ == "__main__":
    main()
