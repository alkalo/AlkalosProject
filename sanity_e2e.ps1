# sanity_e2e.ps1
$ErrorActionPreference = "Stop"
Write-Host "Setting up environment..." -ForegroundColor Cyan

# Asegura que Python vea el paquete 'src'
$env:PYTHONPATH = "$PWD"

Write-Host "Fetching BTC data (365 days)..." -ForegroundColor Cyan
python -m src.data_fetch --source yf --symbols BTC --fiat USD --days 365

Write-Host "Training model for BTC..." -ForegroundColor Cyan
python -m src.ml.train_cli --csv data/BTC_USD_1d.csv --symbol BTC --model lgbm --feature_set lags --window 5 --horizon 1

Write-Host "Running BTC backtest..." -ForegroundColor Cyan
python -m src.backtest.run_backtest --symbol BTC --csv data/BTC_USD_1d.csv --fee 0.001 --slippage 0.0005 --buy-thr 0.6 --sell-thr 0.4 --min-edge 0.02

Write-Host "Done. Check outputs in /models/BTC and /reports" -ForegroundColor Green
