# AlkalosProject

This repository contains a simple Coinbase trading bot used for simulation.

## Usage

1. Place model artifacts in `artifacts/<symbol>/`:
   - `model.pkl`
   - `scaler.pkl`
   - `features.json` (list of feature names such as `["price", "mean", "std", "return"]`)

2. Run the bot:

```bash
python trading_bot.py --symbol BTC-USD --interval-minutes 1
```

### Optional arguments

- `--send-orders` : attempt to send orders to the Coinbase sandbox (otherwise trades are simulated only).
- `--window` : number of price points in the feature window (default 10).
- `--max-iterations` : stop after N iterations, useful for tests.

The bot logs actions and equity to `bot.log` and writes a CSV snapshot of the
portfolio every hour in the `snapshots/` directory.
