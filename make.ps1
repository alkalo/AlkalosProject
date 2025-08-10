param(
    [string]$task
)

if (-not $task) {
    Write-Host "Uso: .\make.ps1 <setup|fetch|train|bt|paper|test>" -ForegroundColor Yellow
    exit 1
}

$venv = ".venv\Scripts\Activate.ps1"

switch ($task.ToLower()) {
    "setup" {
        Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
        if (-Not (Test-Path ".venv")) {
            python -m venv .venv
        } else {
            Write-Host "Entorno virtual ya existe" -ForegroundColor Yellow
        }
        Write-Host "Instalando dependencias..." -ForegroundColor Cyan
        & $venv
        pip install -r requirements.txt
    }
    "fetch" {
        Write-Host "Descargando BTC y ETH (3650 días) via yfinance..." -ForegroundColor Cyan
        & $venv
        python src\data_fetch.py --source yf --symbols BTC,ETH --days 3650
    }
    "train" {
        & $venv
        Write-Host "Entrenando LGBM para BTC..." -ForegroundColor Cyan
        python src\ml\train_cli.py --model lgbm --csv data/BTC_USD_1d.csv --symbol BTC --window 30 --horizon 1
        Write-Host "Entrenando LGBM para ETH..." -ForegroundColor Cyan
        python src\ml\train_cli.py --model lgbm --csv data/ETH_USD_1d.csv --symbol ETH --window 30 --horizon 1
    }
    "bt" {
        & $venv
        Write-Host "Backtest BTC con fee=0.006..." -ForegroundColor Cyan
        python src\backtest\run_backtest.py --symbol BTC --csv data/BTC_USD_1d.csv --fee 0.006
        Write-Host "Backtest ETH con fee=0.006..." -ForegroundColor Cyan
        python src\backtest\run_backtest.py --symbol ETH --csv data/ETH_USD_1d.csv --fee 0.006
    }
    "paper" {
        Write-Host "Lanzando paper bot para BTC (intervalo 60 min)..." -ForegroundColor Cyan
        & $venv
        python src\live\paper_bot.py --symbol BTC --csv data/BTC_USD_1d.csv --interval-minutes 60
    }
    "test" {
        Write-Host "Ejecutando tests..." -ForegroundColor Cyan
        & $venv
        pytest -q
    }
    Default {
        Write-Host "Tarea desconocida '$task'." -ForegroundColor Red
        exit 1
    }
}
