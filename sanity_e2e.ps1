# PowerShell script to run a full BTC workflow for sanity checking.

# Step 1: environment setup
$venv = ".venv\\Scripts\\Activate.ps1"
Write-Host "Setting up environment..." -ForegroundColor Cyan
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}
& $venv
pip install -r requirements.txt
# Additional dependencies for training and plotting
pip install scikit-learn matplotlib

# Step 2: fetch BTC data (365 days)
Write-Host "Fetching BTC data (365 days)..." -ForegroundColor Cyan
python src\data_fetch.py --source yf --symbols BTC --days 365

# Step 3: train model for BTC
Write-Host "Training model for BTC..." -ForegroundColor Cyan
python src\ml\train_cli.py --model lgbm --csv data/BTC_USD_1d.csv --symbol BTC --window 30 --horizon 1

# Step 4: backtest BTC
Write-Host "Running BTC backtest..." -ForegroundColor Cyan
python src\backtest\run_backtest.py --symbol BTC --csv data/BTC_USD_1d.csv --fee 0.006

# Step 5: short paper trading demo
Write-Host "Launching paper bot demo..." -ForegroundColor Cyan
$bot = Start-Process -FilePath "python" -ArgumentList "src\\live\\paper_bot.py --symbol BTC --csv data/BTC_USD_1d.csv --interval-minutes 1" -PassThru
Start-Sleep -Seconds 5
Stop-Process -Id $bot.Id

# Step 6: print summary path and total return
$summaryPath = Join-Path (Resolve-Path ".") "reports/BTC_summary.json"
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath | ConvertFrom-Json
    $initial = 1000.0
    $totalReturn = $summary.final_equity - $initial
    Write-Host $summaryPath
    Write-Host $totalReturn
} else {
    Write-Host "Summary file not found: $summaryPath" -ForegroundColor Red
    exit 1
}

