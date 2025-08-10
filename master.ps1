<# =====================================================================
  master.ps1 – AlkalosProject (Windows / PowerShell + Python)

  Flujo:
    1) Prepara/activa .venv con install_env.ps1
    2) Fetch por símbolo con validación CSV
       - YF → CCXT(fiat) → CCXT(USDT) y normaliza a *_USD_1d.csv
       - -ForceFetch borra y re-descarga
       - -ForceCCXT fuerza CCXT (evita YF)
    3) Entrena modelo (LightGBM por defecto)
    4) Backtest DEFAULT
    5) Optimize → lee best_params.json → Backtest OPTIMIZED (saltable)
    6) Comparativa DEFAULT vs OPTIMIZED vs PORTFOLIO (tabla consola + CSV)
    7) Bot de trading (paper/live) opcional

  Uso típico:
    .\master.ps1 -Symbols "BTC,ETH" -Fiat "USD" -Days 1825 -Model "lgbm" -Fee 0.001 -Slippage 0.0005

  Opciones:
    -ForceFetch   → re-descarga datos aunque existan
    -ForceCCXT    → usa CCXT directamente (recomendado si YF falla)
    -SkipOptimize → salta optimización y backtest optimized
    -BotMode      → paper | live | none (paper por defecto)
===================================================================== #>

[CmdletBinding()]
param(
  [string]$Symbols = "BTC,ETH",
  [string]$Fiat = "USD",
  [int]$Days = 1825,
  [string]$Source = "yf",            # yf | ccxt  (se ignora si -ForceCCXT)
  [string]$Exchange = "binance",     # para ccxt
  [string]$Model = "lgbm",
  [int]$Horizon = 1,
  [int]$Window = 40,
  [double]$Fee = 0.001,
  [double]$Slippage = 0.0005,
  [double]$BuyThr = 0.6,
  [double]$SellThr = 0.4,
  [double]$MinEdge = 0.02,
  [ValidateSet("paper","live","none")]
  [string]$BotMode = "paper",
  [switch]$ForceFetch,
  [switch]$ForceCCXT,
  [switch]$SkipOptimize
)

$ErrorActionPreference = "Stop"

# ---------- Logging ----------
function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Ok($m){   Write-Host "[OK]  $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Err($m){  Write-Host "[ERR] $m" -ForegroundColor Red }

# ---------- Helpers ----------
function CsvPath([string]$sym,[string]$fiat){ return "data\$($sym)_$($fiat)_1d.csv" }
function ReportBase([string]$sym){ return "reports\$($sym)" }
function S([double]$n){ return $n.ToString([System.Globalization.CultureInfo]::InvariantCulture) }  # decimales con punto

# Ejecuta Python, captura salida, y muestra traceback si falla
function Run-Py([string[]]$argv, [string]$step){
  $output = & python @argv 2>&1
  $exit = $LASTEXITCODE
  if ($exit -ne 0) {
    Err "$step falló (exit=$exit)"
    if ($output) { $output | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray } }
    throw "$step_failed"
  }
}

# Validación básica del CSV OHLCV
function Test-ValidCsv([string]$path){
  if (-not (Test-Path $path)) { return $false }
  try {
    $lines = Get-Content -Path $path -TotalCount 50
    if ($lines.Count -lt 12) {
      $total = (Get-Content -Path $path | Measure-Object -Line).Lines
      if ($total -lt 12) { return $false }
    }
    $hdr = $lines[0].ToLower()
    foreach($k in @("timestamp","open","high","low","close")){
      if ($hdr -notmatch $k){ return $false }
    }
    return $true
  } catch { return $false }
}

# Descarga robusta: YF → CCXT(fiat) → CCXT(USDT) + normalización a *_USD_1d.csv si aplica
function Fetch-Symbol([string]$sym, [string]$fiat, [int]$days, [string]$exchange, [bool]$forceCCXT){
  $csv = CsvPath $sym $fiat

  if ($ForceFetch -and (Test-Path $csv)) {
    Warn "Forzando re-descarga: eliminando $csv…"
    Remove-Item $csv -Force
  }

  if (Test-Path $csv -and (Test-ValidCsv $csv)) {
    Info "Datos ya existen y son válidos: $csv"
    return
  } elseif (Test-Path $csv) {
    Warn "CSV inválido detectado → eliminando $csv y reintentando…"
    Remove-Item $csv -Force
  }

  function _Try-CCXT([string]$targetFiat){
    Info "Descargando con CCXT ($exchange): $sym/$targetFiat ($days días)…"
    Run-Py @("-m","src.data_fetch","--source","ccxt","--exchange",$exchange,"--symbols",$sym,"--fiat",$targetFiat,"--days",$days.ToString()) "data_fetch(CCXT,$sym,$targetFiat)"
    $csvTry = CsvPath $sym $targetFiat
    if (-not (Test-ValidCsv $csvTry)) { throw "ccxt_invalid_csv_${sym}_$targetFiat" }
    return $csvTry
  }

  # Ruta CCXT directa si se fuerza
  if ($forceCCXT -or $Source -eq "ccxt") {
    try {
      $null = _Try-CCXT $fiat
    } catch {
      if ($fiat -eq "USD") {
        Warn "CCXT $sym/$fiat falló. Probando $sym/USDT y normalizando a *_USD_1d.csv…"
        $csvUsdt = _Try-CCXT "USDT"
        Copy-Item $csvUsdt $csv -Force
      } else {
        Err "Fallo CCXT para $sym/$fiat."
        throw "fetch_failed_${sym}_${fiat}"
      }
    }
    if (-not (Test-ValidCsv $csv)) {
      if ($fiat -eq "USD") {
        $csvUsdt = CsvPath $sym "USDT"
        if (Test-ValidCsv $csvUsdt) { Copy-Item $csvUsdt $csv -Force }
      }
    }
    if (-not (Test-ValidCsv $csv)) { throw "fetch_failed_${sym}_${fiat}" }
    Ok "Descarga CCXT OK para $sym ($fiat)"
    return
  }

  # YF primero; si falla, CCXT (y si USD falla, USDT -> normaliza)
  Info "Descargando con Yahoo Finance: $sym-$fiat ($days días)…"
  try {
    Run-Py @("-m","src.data_fetch","--source","yf","--symbols",$sym,"--fiat",$fiat,"--days",$days.ToString()) "data_fetch(YF,$sym)"
    if (-not (Test-ValidCsv $csv)) { throw "yf_invalid_csv" }
    Ok "Descarga YF OK para $sym"
    return
  } catch {
    Warn "YF falló o CSV inválido para $sym. Intentando CCXT ($exchange)…"
    try {
      $null = _Try-CCXT $fiat
    } catch {
      if ($fiat -eq "USD") {
        Warn "CCXT $sym/$fiat falló. Probando $sym/USDT y normalizando a *_USD_1d.csv…"
        $csvUsdt = _Try-CCXT "USDT"
        Copy-Item $csvUsdt $csv -Force
      } else {
        Err "Fallo CCXT para $sym/$fiat."
        throw "fetch_failed_${sym}_${fiat}"
      }
    }
    if (-not (Test-ValidCsv $csv)) {
      if ($fiat -eq "USD") {
        $csvUsdt = CsvPath $sym "USDT"
        if (Test-ValidCsv $csvUsdt) { Copy-Item $csvUsdt $csv -Force }
      }
    }
    if (-not (Test-ValidCsv $csv)) { throw "fetch_failed_${sym}_${fiat}" }
    Ok "Descarga CCXT OK para $sym ($fiat)"
    return
  }
}

# ---------- 1) Entorno ----------
Info "Preparando entorno…"
& "$PSScriptRoot\install_env.ps1"
if ($LASTEXITCODE -ne 0) { Err "install_env.ps1 falló"; exit 1 }
. "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# ---------- 2) Fetch ----------
$symbolsArr = $Symbols.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
foreach ($sym in $symbolsArr) {
  try {
    Fetch-Symbol -sym $sym -fiat $Fiat -days $Days -exchange $Exchange -forceCCXT:$ForceCCXT.IsPresent
  } catch {
    Err "Fetch falló para $sym. Continuo con el resto…"
  }
}

# ---------- 3) Train ----------
foreach ($sym in $symbolsArr) {
  $csv = CsvPath $sym $Fiat
  if (-not (Test-ValidCsv $csv)) { Warn "Saltando train: CSV inválido $csv"; continue }
  Info "Entrenando $Model para $sym…"
  try {
    Run-Py @(
      "-m","src.ml.train_cli",
      "--model",$Model,"--csv",$csv,"--symbol",$sym,
      "--horizon",$Horizon.ToString(),"--window",$Window.ToString()
    ) "train($sym)"
    Ok "Train OK $sym"
  } catch {
    Err "Train falló para $sym. Continuo…"
  }
}

# ---------- 4) Backtest DEFAULT ----------
foreach ($sym in $symbolsArr) {
  $csv = CsvPath $sym $Fiat
  if (-not (Test-ValidCsv $csv)) { continue }
  Info "Backtest DEFAULT $sym…"
  try {
    Run-Py @(
      "-m","src.backtest.run_backtest",
      "--symbol",$sym,"--csv",$csv,
      "--fee",(S $Fee),"--slippage",(S $Slippage),
      "--buy-thr",(S $BuyThr),"--sell-thr",(S $SellThr),"--min-edge",(S $MinEdge)
    ) "backtest_default($sym)"
    Ok "Backtest DEFAULT OK $sym"
  } catch {
    Err "Backtest DEFAULT falló para $sym."
  }
}

# ---------- 5) Optimize + Backtest OPTIMIZED ----------
if (-not $SkipOptimize) {
  foreach ($sym in $symbolsArr) {
    $csv = CsvPath $sym $Fiat
    if (-not (Test-ValidCsv $csv)) { continue }

    Info "Optimizando parámetros para $sym…"
    try {
      Run-Py @(
        "-m","src.optimize.optimize_enhanced",
        "--csv",$csv,"--symbol",$sym,
        "--fee",(S $Fee),"--slippage",(S $Slippage)
      ) "optimize($sym)"
    } catch {
      Warn "Optimize falló para $sym. Sigo sin optimizado."
      continue
    }

    $bestJson = Join-Path (ReportBase $sym) "${sym}_best_params.json"
    if (-not (Test-Path $bestJson)) {
      Warn "No se encontró $bestJson para $sym. Sigo con DEFAULT."
      continue
    }

    try {
      $best = Get-Content $bestJson | ConvertFrom-Json
      if ($null -eq $best.buy_thr -or $null -eq $best.sell_thr -or $null -eq $best.min_edge) {
        Warn "best_params incompleto para $sym."
        continue
      }
      $optBuy  = [double]$best.buy_thr
      $optSell = [double]$best.sell_thr
      $optEdge = [double]$best.min_edge

      Info "Backtest OPTIMIZED $sym (buy=$optBuy, sell=$optSell, edge=$optEdge)…"
      Run-Py @(
        "-m","src.backtest.run_backtest",
        "--symbol",$sym,"--csv",$csv,
        "--fee",(S $Fee),"--slippage",(S $Slippage),
        "--buy-thr",(S $optBuy),"--sell-thr",(S $optSell),"--min-edge",(S $optEdge),
        "--tag","optimized"
      ) "backtest_optimized($sym)"
      Ok "Backtest OPTIMIZED OK $sym"
    } catch {
      Err "Backtest OPTIMIZED falló para $sym."
    }
  }
} else {
  Warn "SkipOptimize activo → no se realizará optimización ni backtest optimized."
}

# ---------- 6) Comparativa y salida en consola ----------
function Load-Metrics([string]$sym, [string]$tag="default"){
  $summary = "reports\${sym}_summary.json"
  if ($tag -eq "optimized") { $summary = "reports\${sym}_summary_optimized.json" }
  if (-not (Test-Path $summary)) { return $null }
  try { return Get-Content $summary | ConvertFrom-Json } catch { return $null }
}

$rows = @()
foreach ($sym in $symbolsArr) {
  $d = Load-Metrics $sym "default"
  $o = Load-Metrics $sym "optimized"
  if ($d) {
    $rows += [pscustomobject]@{
      Symbol=$sym; Variant="default"
      CAGR=$d.cagr; Sharpe=$d.sharpe; Sortino=$d.sortino
      MaxDD=$d.max_drawdown; WinRate=$d.win_rate; Trades=$d.trades
      FinalEquity=$d.final_equity
    }
  }
  if ($o) {
    $rows += [pscustomobject]@{
      Symbol=$sym; Variant="optimized"
      CAGR=$o.cagr; Sharpe=$o.sharpe; Sortino=$o.sortino
      MaxDD=$o.max_drawdown; WinRate=$o.win_rate; Trades=$o.trades
      FinalEquity=$o.final_equity
    }
  }
}

$portfolioCsv = "reports\portfolio_equity.csv"
if (Test-Path $portfolioCsv) {
  try {
    $port = Import-Csv $portfolioCsv
    $last = $port | Select-Object -Last 1
    $rows += [pscustomobject]@{
      Symbol="PORTFOLIO"; Variant="portfolio"
      CAGR=$null; Sharpe=$null; Sortino=$null
      MaxDD=$null; WinRate=$null; Trades=$null
      FinalEquity=$last.Equity
    }
  } catch {
    Warn "No se pudo leer $portfolioCsv para comparativa."
  }
}

$compCsv = "reports\comparison_default_optimized_portfolio.csv"
$rows | Export-Csv -NoTypeInformation -Path $compCsv -Encoding UTF8

Write-Host ""
if (Test-Path $compCsv) {
  Ok "Resumen final (DEFAULT vs OPTIMIZED vs PORTFOLIO):"
  Import-Csv $compCsv | Sort-Object Symbol, Variant | Format-Table -AutoSize
  Ok "Comparativa guardada en $compCsv"
} else {
  Warn "No se generó comparativa."
}

# ---------- 7) Trading bot (paper/live) ----------
if ($BotMode -ne "none") {
  foreach ($sym in $symbolsArr) {
    $csv = CsvPath $sym $Fiat
    if (-not (Test-ValidCsv $csv)) { Warn "Saltando bot: CSV inválido $csv"; continue }
    $pair = "$sym/$Fiat"
    Info "Lanzando bot $BotMode para $pair..."
    $args = @("-m","src.live.paper_bot","--symbol",$pair,"--csv",$csv,"--window",$Window.ToString(),"--mode",$BotMode)
    try {
      Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory (Get-Location).Path -NoNewWindow
      Ok "Bot iniciado para $pair"
    } catch {
      Err "No se pudo iniciar bot para $pair"
    }
  }
} else {
  Warn "Bot deshabilitado (BotMode=none)"
}

exit 0
