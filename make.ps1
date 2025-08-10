param(
    [string]$task
)

if (-not $task) {
    Write-Host "Uso: .\\make.ps1 <setup|fetch|train|bt|paper|test>" -ForegroundColor Yellow
    exit 1
}

$venv = ".venv\\Scripts\\Activate.ps1"
$python = "python"
$symbols = @("BTC", "ETH")

switch ($task.ToLower()) {
    "setup" {
        Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
        if (-Not (Test-Path ".venv")) {
            & $python -m venv .venv
        } else {
            Write-Host "Entorno virtual ya existe" -ForegroundColor Yellow
        }
        Write-Host "Instalando dependencias..." -ForegroundColor Cyan
        & $venv
        pip install -r requirements.txt
    }
    "fetch" {
        & $venv
        Write-Host "Descargando $($symbols -join ', ') (3650 días) via yfinance..." -ForegroundColor Cyan
        & $python src\data_fetch.py --source yf --symbols $($symbols -join ',') --days 3650
    }
    "train" {
        & $venv
        foreach ($sym in $symbols) {
            Write-Host "Entrenando LGBM para $sym..." -ForegroundColor Cyan
            & $python src\ml\train_cli.py --model lgbm --csv data/${sym}_USD_1d.csv --symbol $sym --window 30 --horizon 1
        }
    }
    "bt" {
        & $venv
        foreach ($sym in $symbols) {
            Write-Host "Backtest $sym con fee=0.006..." -ForegroundColor Cyan
            & $python src\backtest\run_backtest.py --symbol $sym --csv data/${sym}_USD_1d.csv --fee 0.006
        }
    }
    "paper" {
        & $venv
        Write-Host "Lanzando paper bot para BTC (intervalo 60 min)..." -ForegroundColor Cyan
        & $python src\live\paper_bot.py --symbol BTC --csv data/BTC_USD_1d.csv --interval-minutes 60
    }
    "test" {
        & $venv
        Write-Host "Ejecutando tests..." -ForegroundColor Cyan
        pytest -q
    }
    Default {
        Write-Host "Tarea desconocida '$task'." -ForegroundColor Red
        exit 1
    }
}

