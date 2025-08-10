param(
  [string]$Symbol = "BTC",
  [string]$Csv = "data\BTC_USDT_1d.csv",
  [string]$BuyThr = "0.55,0.6,0.65",
  [string]$SellThr = "0.35,0.4,0.45",
  [string]$MinEdge = "0.01,0.02,0.03",
  [string]$Fee = "0.0005,0.001,0.002",
  [string]$Slippage = "0.0002,0.0005,0.001",
  [int]$WFSplits = 1,
  [ValidateSet("calmar","sharpe","cagr")]
  [string]$Objective = "calmar"
)

# 1) Resuelve el Python del venv; si no existe, usa 'python' del sistema
$VenvPython = Join-Path $PSScriptRoot ".\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = "python" }

# 2) Construye argumentos como array (PowerShell pasará cada elemento como arg)
$ArgsList = @(
  "-m", "src.optimize.params_search",
  "--symbol",   $Symbol,
  "--csv",      $Csv,
  "--buy-thr",  $BuyThr,
  "--sell-thr", $SellThr,
  "--min-edge", $MinEdge,
  "--fee",      $Fee,
  "--slippage", $Slippage,
  "--wf-splits", $WFSplits,
  "--objective", $Objective
)

Write-Host "RUNNING: $VenvPython $($ArgsList -join ' ')" -ForegroundColor Cyan

# 3) Ejecuta y devuelve el exit code correcto
& $VenvPython @ArgsList
exit $LASTEXITCODE
