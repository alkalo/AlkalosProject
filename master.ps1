param(
  [string]$Symbols   = "BTC,ETH",
  [string]$Exchange  = "binance",
  [string]$Fiat      = "USDT",
  [int]$Days         = 1825,
  [string]$Timeframe = "1d",
  [string]$Model     = "logreg",   # tu train_cli solo acepta 'logreg'
  [switch]$Optimize  = $false,
  [ValidateSet("calmar","sharpe","cagr")]
  [string]$Objective = "calmar"
)

function Info($msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Err($msg){ Write-Host "[ERROR] $msg" -ForegroundColor Red }

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

# Python del venv
$VENV_PY = Join-Path $ROOT ".\.venv\Scripts\python.exe"
$PY = "python"
if (Test-Path $VENV_PY) { $PY = $VENV_PY }

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   AlkalosProject – Master Runner" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# 1) Entorno (usa tu install_env.ps1)
if (Test-Path ".\install_env.ps1") {
  Info "Instalando/actualizando entorno (install_env.ps1)..."
  powershell -NoProfile -ExecutionPolicy Bypass -File .\install_env.ps1
  if ($LASTEXITCODE -ne 0) { Err "install_env.ps1 falló"; exit 1 }
} else {
  Warn "install_env.ps1 no encontrado. Creo venv e instalo dependencias mínimas..."
  if (-not (Test-Path $VENV_PY)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Err "Creando venv falló"; exit 1 }
  }
  $PY = $VENV_PY
  & $PY -m pip install --upgrade pip
  if (Test-Path ".\requirements.txt") {
    & $PY -m pip install -r .\requirements.txt
  } else {
    & $PY -m pip install pandas numpy matplotlib ccxt yfinance
  }
  if ($LASTEXITCODE -ne 0) { Err "Instalación de dependencias falló"; exit 1 }
}
if (Test-Path $VENV_PY) { $PY = $VENV_PY }

# 2) Detectar si src.data_fetch soporta --timeframe
$timeframeSupported = $false
try {
  $helpText = & $PY -m src.data_fetch --help 2>&1
  if ($helpText -match "--timeframe") { $timeframeSupported = $true }
} catch { $timeframeSupported = $false }

# 3) Fetch (CCXT)
$symbols_list = $Symbols.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne "" }
Info "Descargando datos con CCXT ($Exchange / $Fiat / $Days días$([string]::Format(' / {0}', $Timeframe)))..."
foreach ($sym in $symbols_list) {
  $args = @("-m","src.data_fetch","--source","ccxt","--exchange",$Exchange,"--symbols",$sym,"--fiat",$Fiat,"--days",$Days)
  if ($timeframeSupported) { $args += @("--timeframe",$Timeframe) }
  Write-Host ("  • {0}" -f $sym) -ForegroundColor DarkCyan
  & $PY @args
  if ($LASTEXITCODE -ne 0) { Warn ("Fetch falló para {0}" -f $sym) }
}

# 4) Entrenamiento
foreach ($sym in $symbols_list) {
  # Detecta CSV: con timeframe o sin él
  $csv1 = "data\${sym}_${Fiat}_${Timeframe}.csv"
  $csv2 = "data\${sym}_${Fiat}.csv"
  $csv = $null
  if (Test-Path $csv1) { $csv = $csv1 } elseif (Test-Path $csv2) { $csv = $csv2 }

  if (-not $csv) { Warn ( "CSV no encontrado para {0} (probé {1} y {2})" -f $sym, $csv1, $csv2 ); continue }

  Info ("Entrenando {0} con {1} ..." -f $sym, $Model)
  & $PY -m src.ml.train_cli --model $Model --csv $csv --symbol $sym --horizon 1 --window 5
  if ($LASTEXITCODE -ne 0) { Warn ("Entrenamiento falló para {0}" -f $sym) }
}

# 5) Backtest (enhanced)
foreach ($sym in $symbols_list) {
  $csv1 = "data\${sym}_${Fiat}_${Timeframe}.csv"
  $csv2 = "data\${sym}_${Fiat}.csv"
  $csv = $null
  if (Test-Path $csv1) { $csv = $csv1 } elseif (Test-Path $csv2) { $csv = $csv2 }
  if (-not $csv) { continue }
  Info ("Backtest (enhanced) de {0} ..." -f $sym)
  & $PY -m src.backtest.run_backtest_enhanced --symbol $sym --csv $csv `
        --sma-fast 20 --sma-slow 200 --adx-n 14 --adx-thr 20 `
        --atr-n 14 --risk-per-trade 0.01 --sl-atr 2.0 --ts-atr 1.0 `
        --fee 0.001 --slippage 0.0005 --initial-equity 10000
  if ($LASTEXITCODE -ne 0) { Warn ("Backtest (enhanced) falló para {0}" -f $sym) }
}


# 6) Optimización enhanced (si existe el optimizador y hay CSV)
$first = $symbols_list[0]
$firstCsv1 = "data\${first}_${Fiat}_${Timeframe}.csv"
$firstCsv2 = "data\${first}_${Fiat}.csv"
$firstCsv  = $null
if (Test-Path $firstCsv1) { $firstCsv = $firstCsv1 } elseif (Test-Path $firstCsv2) { $firstCsv = $firstCsv2 }

if ( (Test-Path ".\src\optimize\optimize_enhanced.py") -and ($firstCsv) ) {
  Info ("Optimizando estrategia enhanced sobre {0}..." -f $first)
  & $PY -m src.optimize.optimize_enhanced --symbol $first --csv $firstCsv `
      --sma-fast "10,20,30" --sma-slow "150,200,300" --adx-thr "20,25,30" `
      --risk "0.005,0.01,0.015" --sl-atr "1.5,2.0,2.5" --ts-atr "0.5,1.0,1.5" `
      --fee "0.0005,0.001" --slippage "0.0002,0.0005" --objective calmar
  if ($LASTEXITCODE -ne 0) { Warn "Optimización enhanced falló (sigo)"; }
} else {
  Warn "optimize_enhanced no disponible o CSV ausente; salto optimización."
}



# 6.5) Cartera BTC+ETH por paridad de riesgo (si hay ≥2)
if ($symbols_list.Count -ge 2) {
  Info "Combinando cartera por paridad de riesgo (equity_enhanced)..."
  & $PY -m src.backtest.portfolio_combine --symbols ($Symbols) --reports "reports" --enhanced --lookback 90
}


# 7) Resumen → reports/MASTER_SUMMARY_*.md
$reports = Join-Path $ROOT "reports"
if (!(Test-Path $reports)) { New-Item -ItemType Directory -Path $reports | Out-Null }
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$summaryPath = Join-Path $reports ("MASTER_SUMMARY_" + $ts + ".md")

# Ruta del script Python temporal
$tempPy = Join-Path ([System.IO.Path]::GetTempPath()) ("alkalos_summary_" + ([System.Guid]::NewGuid().ToString("N")) + ".py")

# Código Python que calcula métricas (incluye PORTFOLIO si existe)
$pyCode = @'
import os, glob, math
import pandas as pd
import numpy as np

REPORTS = "reports"

def latest(pattern):
    files = glob.glob(pattern)
    if not files: return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def drawdown(e):
    peak = np.maximum.accumulate(e)
    return e/peak - 1.0

def metrics_from_eq(eq_path):
    if not eq_path or not os.path.exists(eq_path): return {}
    df = pd.read_csv(eq_path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], errors="coerce", utc=True)
    df = df.dropna(subset=[ts,"equity"]).sort_values(ts)
    e = df["equity"].astype(float).values
    ret = e[-1]/e[0]-1 if e[0]!=0 else float("nan")
    rets = pd.Series(e).pct_change().dropna()
    sharpe = (rets.mean()/rets.std()*np.sqrt(252)) if rets.std()!=0 else float("nan")
    dd = drawdown(e); maxdd = float(dd.min()) if len(dd) else float("nan")
    yrs = max((df[ts].iloc[-1]-df[ts].iloc[0]).days/365.25, 1e-9)
    cagr = (e[-1]/e[0])**(1/yrs)-1 if e[0]>0 else float("nan")
    return {
        "total_return(%)": None if math.isnan(ret) else round(100*ret,2),
        "CAGR(%)": None if math.isnan(cagr) else round(100*cagr,2),
        "Sharpe": None if math.isnan(sharpe) else round(float(sharpe),3),
        "MaxDD(%)": None if math.isnan(maxdd) else round(100*maxdd,2)
    }

def symbol_metrics(sym):
    eq = latest(os.path.join(REPORTS, f"{sym}_equity_enhanced.csv")) or latest(os.path.join(REPORTS, f"{sym}_equity.csv"))
    out = {"symbol": sym, **metrics_from_eq(eq)}
    # trades
    tr = latest(os.path.join(REPORTS, f"{sym}_trades_enhanced.csv")) or latest(os.path.join(REPORTS, f"{sym}_trades.csv"))
    if tr and os.path.exists(tr):
        t = pd.read_csv(tr)
        cols = {c.lower(): c for c in t.columns}
        pnl = cols.get("pnl") or cols.get("profit") or cols.get("pl")
        if pnl is not None:
            wins = t[pnl] > 0
            gp = t.loc[wins, pnl].sum(); gl = -t.loc[~wins, pnl].sum()
            pf = gp/gl if gl>0 else float("nan")
            wr = wins.mean() if len(t)>0 else float("nan")
            out.update({
                "ProfitFactor": None if math.isnan(pf) else round(float(pf),3),
                "WinRate(%)": None if math.isnan(wr) else round(100*float(wr),2),
                "Trades": int(len(t))
            })
    # verdict
    s = out.get("Sharpe") or 0.0
    dd = out.get("MaxDD(%)"); dd = dd if dd is not None else -100.0
    verdict = "🔴 NO listo (Sharpe<0.8 o DD<-20%)"
    if s>=1.2 and dd>-20: verdict = "🟢 Listo para paper serio"
    elif s>=0.8 and dd>-25: verdict = "🟠 Mejorable; validar con filtros/ATR"
    out["Verdict"] = verdict
    return out

symbols = os.environ.get("ALKALOS_SYMBOLS","BTC,ETH").split(",")
rows = [symbol_metrics(s.strip().upper()) for s in symbols if s.strip()]

md = ["# Master Summary", ""]
for r in rows:
    md.append(f"## {r['symbol']}")
    for k in ["total_return(%)","CAGR(%)","Sharpe","MaxDD(%)","ProfitFactor","WinRate(%)","Trades","Verdict"]:
        if k in r: md.append(f"- **{k}**: {r[k]}")
    md.append("")

# portfolio (si existe)
port = os.path.join(REPORTS, "portfolio_equity.csv")
pm = metrics_from_eq(port)
if pm:
    md.append("## PORTFOLIO (BTC+ETH vol-target 90d)")
    for k,v in pm.items():
        md.append(f"- **{k}**: {v}")
    s = pm.get("Sharpe") or 0.0
    dd = pm.get("MaxDD(%)"); dd = dd if dd is not None else -100.0
    verdict = "🔴 NO listo (Sharpe<0.8 o DD<-20%)"
    if s>=1.2 and dd>-20: verdict = "🟢 Listo para paper serio"
    elif s>=0.8 and dd>-25: verdict = "🟠 Mejorable; validar con filtros/ATR"
    md.append(f"- **Verdict**: {verdict}")
    md.append("")

out_path = os.environ.get("ALKALOS_SUMMARY_OUT","MASTER_SUMMARY.md")
open(out_path,"w",encoding="utf-8").write("\n".join(md))
print("\n".join(md))
'@

Set-Content -Path $tempPy -Value $pyCode -Encoding UTF8

$env:ALKALOS_SYMBOLS     = $Symbols
$env:ALKALOS_SUMMARY_OUT = $summaryPath

& $PY $tempPy
Remove-Item $tempPy -ErrorAction SilentlyContinue

Info ("Resumen guardado en: {0}" -f $summaryPath)
Write-Host ""
Write-Host "================  NOTAS  ================" -ForegroundColor Green
Write-Host "• 🟢 Listo para paper serio  => Sharpe ≥ 1.2 y MaxDD > -20%" -ForegroundColor Green
Write-Host "• 🟠 Mejorable                => Sharpe ≥ 0.8 y MaxDD > -25%" -ForegroundColor Yellow
Write-Host "• 🔴 NO listo                 => umbrales peores" -ForegroundColor Red
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
