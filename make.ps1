param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup","fetch","train","bt","paper","test")]
    [string]$Task
)

function Invoke-Step {
    param(
        [string]$Command,
        [string]$ErrorMessage
    )
    Write-Host "==> $Command"
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            throw "$ErrorMessage (exit code: $LASTEXITCODE)"
        }
    } catch {
        Write-Error $_
        exit 1
    }
}

switch ($Task) {
    "setup" {
        Invoke-Step "python -m venv .venv" "Virtual environment creation failed"
        $venv = Resolve-Path ".venv"
        Write-Host "Virtual environment: $venv"
        Invoke-Step ".\\.venv\\Scripts\\python -m pip install -r requirements.txt" "Dependency installation failed"
    }
    "fetch" {
        $out = Resolve-Path "data/raw" -ErrorAction SilentlyContinue
        if (-not $out) {
            $out = (New-Item -ItemType Directory -Path "data/raw").FullName
        }
        Invoke-Step "python scripts/fetch.py --symbols BTC ETH --provider coingecko --days max --outdir $out" "Fetch failed"
        Write-Host "Data stored in: $out"
    }
    "train" {
        $models = Resolve-Path "models" -ErrorAction SilentlyContinue
        if (-not $models) {
            $models = (New-Item -ItemType Directory -Path "models").FullName
        }
        Invoke-Step "python scripts/train_lgbm.py --symbols BTC ETH --window 30 --horizon 1 --outdir $models" "Training failed"
        Write-Host "Models stored in: $models"
    }
    "bt" {
        $bt = Resolve-Path "backtests" -ErrorAction SilentlyContinue
        if (-not $bt) {
            $bt = (New-Item -ItemType Directory -Path "backtests").FullName
        }
        Invoke-Step "python scripts/backtest.py --symbols BTC ETH --outdir $bt" "Backtest failed"
        Write-Host "Backtest results: $bt"
    }
    "paper" {
        Invoke-Step "python bots/paper_bot.py --interval 60" "Paper bot failed"
    }
    "test" {
        Invoke-Step "pytest" "Tests failed"
    }
}
