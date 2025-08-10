<# =====================================================================
  master.ps1 – AlkalosProject (One-File E2E Runner)

  USO DIARIO:
    .\master.ps1

  MODOS:
    -Mode backtest  (por defecto) → fetch/entrena/backtests/comparativa
    -Mode paper                  → simula órdenes (no reales)
    -Mode live                   → envía órdenes reales (API keys)

  FLAGS:
    -ForceFetch                  → borra y re-descarga datos
    -ForceCCXT                   → evita YF y usa CCXT directamente
    -SkipOptimize                → salta optimización y backtest optimized

  ENV (para live/paper):
    EXCHANGE=binance
    API_KEY=xxxx
    API_SECRET=yyyy
    API_PASSWORD=zzz  (si tu exchange lo pide)
===================================================================== #>

[CmdletBinding()]
param(
  [string]$Symbols = "BTC,ETH",
  [string]$Fiat = "USD",
  [int]$Days = 1825,

  [string]$Model = "lgbm",
  [int]$Horizon = 1,
  [int]$Window = 40,

  [double]$Fee = 0.001,
  [double]$Slippage = 0.0005,
  [double]$BuyThr = 0.6,
  [double]$SellThr = 0.4,
  [double]$MinEdge = 0.02,

  [string]$Source = "yf",
  [string]$Exchange = "binance",

  [ValidateSet("backtest","paper","live")]
  [string]$Mode = "backtest",

  [double]$BaseCapital = 10000,
  [double]$RiskPct = 0.01,
  [double]$MaxDailyLossPct = 0.05,
  [double]$MaxPosPct = 0.25,
  [double]$MinNotional = 20,

  [switch]$ForceFetch,
  [switch]$ForceCCXT,
  [switch]$SkipOptimize
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "data","reports","logs" | Out-Null

# ---------- Logging ----------
function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Ok($m){   Write-Host "[OK]  $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Err($m){  Write-Host "[ERR] $m" -ForegroundColor Red }

# ---------- Helpers ----------
function CsvPath([string]$sym,[string]$fiat){ return ("data\{0}_{1}_1d.csv" -f $sym,$fiat) }
function ReportBase([string]$sym){ return ("reports\{0}" -f $sym) }
function S([double]$n){ return $n.ToString([System.Globalization.CultureInfo]::InvariantCulture) }

# Ejecuta Python y muestra traceback si falla
function Run-Py([string[]]$argv, [string]$step){
  $output = & python @argv 2>&1
  $exit = $LASTEXITCODE
  if ($exit -ne 0) {
    Err ("{0} falló (exit={1})" -f $step,$exit)
    if ($output) { $output | ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor DarkGray } }
    throw ("{0}_failed" -f $step)
  }
}

# Debug/Validación CSV
function Debug-Csv([string]$path){
  if (-not (Test-Path $path)) { Warn ("CSV no existe: {0}" -f $path); return }
  Write-Host "----- DEBUG CSV: $path (primeras 5 líneas) -----" -ForegroundColor DarkGray
  try { (Get-Content -Path $path -TotalCount 5) | ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor DarkGray } } catch {}
  Write-Host "-------------------------------------------------" -ForegroundColor DarkGray
}

function Test-ValidCsv([string]$path){
  if (-not (Test-Path $path)) { return $false }
  try {
    $lines = Get-Content -Path $path -TotalCount 50
    if (-not $lines -or $lines.Count -lt 3) { return $false }

    $hdr = $lines[0].ToLower().Replace('"','')
    $delim = ','
    if ($hdr -match ';') { $delim = ';' }
    $cols  = $hdr.Split($delim) | ForEach-Object { $_.Trim() }

    $hasTime = $false
    if ($cols -contains 'timestamp') { $hasTime = $true }
    if ($cols -contains 'date') { $hasTime = $true }

    $hasO = $cols -contains 'open'
    $hasH = $cols -contains 'high'
    $hasL = $cols -contains 'low'
    $hasC = $cols -contains 'close'

    if (-not $hasTime) { return $false }
    if (-not $hasO) { return $false }
    if (-not $hasH) { return $false }
    if (-not $hasL) { return $false }
    if (-not $hasC) { return $false }

    $total = (Get-Content -Path $path | Measure-Object -Line).Lines
    if ($total -lt 12) { return $false }  # ≥10 velas + cabecera

    return $true
  } catch { return $false }
}

# Detecta el mejor CSV existente (USD/USDT, con o sin _labeled)
function Find-ExistingCsv([string]$sym,[string]$fiat){
  $candidates = @(
    ("data\{0}_{1}_1d.csv"         -f $sym,$fiat),
    ("data\{0}_{1}_1d_labeled.csv"  -f $sym,$fiat),
    ("data\{0}_USDT_1d.csv"         -f $sym),
    ("data\{0}_USDT_1d_labeled.csv" -f $sym)
  )
  foreach($p in $candidates){
    if (Test-Path $p) {
      if (Test-ValidCsv $p) { return $p }
    }
  }
  return $null
}

# ---------- Entorno ----------
Info "Preparando entorno…"
& "$PSScriptRoot\install_env.ps1"
if ($LASTEXITCODE -ne 0) { Err "install_env.ps1 falló"; exit 1 }
. "$PSScriptRoot\.venv\Scripts\Activate.ps1"
Ok  "Entorno listo."

# ---------- Descarga DIRECTA con CCXT (embebida; bypass src.data_fetch) ----------
function Fetch-CCXT-Direct([string]$exchange, [string]$sym, [string]$fiat, [int]$days) {
  $py = @"
import sys, time, csv
from datetime import datetime, timedelta, timezone
import ccxt

exchange_id = "${exchange}"
sym = "${sym}"
fiat = "${fiat}"
days = int("${days}")
market = f"{sym}/{fiat}"

ex = getattr(ccxt, exchange_id)()
ex.load_markets()

since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
limit = 1000
all_candles = []
cursor = since

while True:
    try:
        data = ex.fetch_ohlcv(market, timeframe="1d", since=cursor, limit=limit)
    except Exception as e:
        print(f"[ERR] CCXT fetch_ohlcv: {e}", file=sys.stderr)
        break
    if not data:
        break
    all_candles.extend(data)
    cursor = data[-1][0] + 1
    if len(data) < limit:
        break
    time.sleep(0.2)

out = f"data/{sym}_{fiat}_1d.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp","open","high","low","close","volume"])
    for ts, o, h, l, c, v in all_candles:
        w.writerow([ts, o, h, l, c, v])

print(f"[OK] DIRECT CCXT {market} -> {out} | rows={len(all_candles)}")
"@
  $tmp = Join-Path $env:TEMP "fetch_direct_ccxt.py"
  Set-Content -Path $tmp -Value $py -Encoding UTF8
  Run-Py @($tmp) "fetch_direct_ccxt"
}

# ---------- Fetch con todas las redes de seguridad ----------
function Fetch-Symbol([string]$sym, [string]$fiat, [int]$days, [string]$exchange, [bool]$forceCCXT){
  $csvUSD  = CsvPath $sym $fiat
  $csvUSDT = CsvPath $sym "USDT"

  # (0) Si ForceFetch, limpia
  if ($ForceFetch) {
    if (Test-Path $csvUSD)  { Remove-Item $csvUSD  -Force }
    if (Test-Path $csvUSDT) { Remove-Item $csvUSDT -Force }
  }

  # (1) Si ya existe algún CSV válido (USD/USDT, labeled o no), úsalo y normaliza a USD
  $existing = Find-ExistingCsv $sym $fiat
  if ($existing) {
    Warn ("Usando CSV existente para {0}: {1}" -f $sym, $existing)
    try { Copy-Item $existing $csvUSD -Force } catch {}
    if (-not (Test-ValidCsv $csvUSD)) {
      Warn "Normalizado USD no pasó validación, mantengo el existente para pipeline."
    }
    Ok ("Fetch omitido (archivo ya presente) para {0}" -f $sym)
    return
  }

  # (2) Intento normal (según flags): YF -> CCXT o CCXT directo
  $ok = $false
  try {
    if ($forceCCXT -or $Source -eq "ccxt") {
      Info ("CCXT normal {0}/{1}…" -f $sym,$fiat)
      try {
        Run-Py @("-m","src.data_fetch","--source","ccxt","--exchange",$exchange,"--symbols",$sym,"--fiat",$fiat,"--days",$days.ToString()) "data_fetch(CCXT,$sym,$fiat)"
        if (-not (Test-ValidCsv $csvUSD)) { throw "csv_usd_invalid" }
        $ok = $true
      } catch {
        Info ("CCXT normal {0}/USDT…" -f $sym)
        Run-Py @("-m","src.data_fetch","--source","ccxt","--exchange",$exchange,"--symbols",$sym,"--fiat","USDT","--days",$days.ToString()) "data_fetch(CCXT,$sym,USDT)"
        if (Test-ValidCsv $csvUSDT) { Copy-Item $csvUSDT $csvUSD -Force; $ok = $true }
      }
    } else {
      Info ("YF {0}-{1}…" -f $sym,$fiat)
      try {
        Run-Py @("-m","src.data_fetch","--source","yf","--symbols",$sym,"--fiat",$fiat,"--days",$days.ToString()) "data_fetch(YF,$sym)"
        if (-not (Test-ValidCsv $csvUSD)) { throw "yf_invalid" }
        $ok = $true
      } catch {
        Info ("Fallback CCXT normal {0}/{1}…" -f $sym,$fiat)
        try {
          Run-Py @("-m","src.data_fetch","--source","ccxt","--exchange",$exchange,"--symbols",$sym,"--fiat",$fiat,"--days",$days.ToString()) "data_fetch(CCXT,$sym,$fiat)"
          if (-not (Test-ValidCsv $csvUSD)) { throw "ccxt_usd_invalid" }
          $ok = $true
        } catch {
          Info ("Fallback CCXT normal {0}/USDT…" -f $sym)
          Run-Py @("-m","src.data_fetch","--source","ccxt","--exchange",$exchange,"--symbols",$sym,"--fiat","USDT","--days",$days.ToString()) "data_fetch(CCXT,$sym,USDT)"
          if (Test-ValidCsv $csvUSDT) { Copy-Item $csvUSDT $csvUSD -Force; $ok = $true }
        }
      }
    }
  } catch {
    # ignoramos; probaremos plan B
  }

  # (3) Si no quedó OK, plan B: DIRECT CCXT USDT (embebido)
  if (-not $ok) {
    Warn ("Fetch normal falló para {0} → usando DIRECT CCXT (USDT) embebido…" -f $sym)
    try {
      Fetch-CCXT-Direct -exchange $exchange -sym $sym -fiat "USDT" -days $days
      if (Test-ValidCsv $csvUSDT) {
        try { Copy-Item $csvUSDT $csvUSD -Force } catch {}
        $ok = $true
      }
    } catch {
      Err ("DIRECT CCXT también falló para {0}." -f $sym)
    }
  }

  # (4) Verificación final (sin -and; ifs anidados)
  if ($ok) {
    if (Test-ValidCsv $csvUSD) {
      Ok ("Descarga OK para {0} (normalizado a USD)" -f $sym)
      return
    } elseif (Test-ValidCsv $csvUSDT) {
      Warn ("USD no válido; seguiré con USDT para {0}." -f $sym)
      return
    } else {
      Err ("Fetch definitivamente falló para {0} (ok sin CSV válido)." -f $sym)
      Warn "DEBUG post-fetch USD:";  Debug-Csv $csvUSD
      Warn "DEBUG post-fetch USDT:"; Debug-Csv $csvUSDT
      throw ("fetch_failed_{0}" -f $sym)
    }
  } else {
    Err ("Fetch definitivamente falló para {0} (ok=false)." -f $sym)
    Warn "DEBUG post-fetch USD:";  Debug-Csv $csvUSD
    Warn "DEBUG post-fetch USDT:"; Debug-Csv $csvUSDT
    throw ("fetch_failed_{0}" -f $sym)
  }
}

# ========== 1) FETCH ==========
$symbolsArr = $Symbols.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
foreach ($sym in $symbolsArr) {
  try {
    Fetch-Symbol -sym $sym -fiat $Fiat -days $Days -exchange $Exchange -forceCCXT:$ForceCCXT.IsPresent
  } catch {
    Err ("Fetch falló para {0}. Continuo con el resto…" -f $sym)
  }
}

# Helper para elegir el mejor CSV (USD → USD_labeled → USDT → USDT_labeled)
function Select-Csv-For([string]$sym,[string]$fiat){
  $c = Find-ExistingCsv $sym $fiat
  if ($c) { return $c }
  return $null
}

# ========== 2) TRAIN ==========
foreach ($sym in $symbolsArr) {
  $csv = Select-Csv-For $sym $Fiat
  if (-not $csv) { Warn ("Saltando train: no hay CSV válido (USD/USDT, labeled o no) para {0}" -f $sym); continue }
  Info ("Entrenando {0} para {1} con {2}…" -f $Model,$sym,$csv)
  try {
    Run-Py @("-m","src.ml.train_cli","--model",$Model,"--csv",$csv,"--symbol",$sym,"--horizon",$Horizon.ToString(),"--window",$Window.ToString()) "train($sym)"
    Ok ("Train OK {0}" -f $sym)
  } catch {
    Err ("Train falló para {0}. Continuo…" -f $sym)
  }
}

# ========== 3) BACKTEST DEFAULT ==========
foreach ($sym in $symbolsArr) {
  $csv = Select-Csv-For $sym $Fiat
  if (-not $csv) { continue }
  Info ("Backtest DEFAULT {0} (csv={1})…" -f $sym,$csv)
  try {
    Run-Py @("-m","src.backtest.run_backtest","--symbol",$sym,"--csv",$csv,"--fee",(S $Fee),"--slippage",(S $Slippage),"--buy-thr",(S $BuyThr),"--sell-thr",(S $SellThr),"--min-edge",(S $MinEdge)) "backtest_default($sym)"
    Ok ("Backtest DEFAULT OK {0}" -f $sym)
  } catch {
    Err ("Backtest DEFAULT falló para {0}." -f $sym)
  }
}

# ========== 4) OPTIMIZE + BACKTEST OPTIMIZED ==========
if (-not $SkipOptimize) {
  foreach ($sym in $symbolsArr) {
    $csv = Select-Csv-For $sym $Fiat
    if (-not $csv) { continue }
    Info ("Optimizando parámetros para {0}…" -f $sym)
    try {
      Run-Py @("-m","src.optimize.optimize_enhanced","--csv",$csv,"--symbol",$sym,"--fee",(S $Fee),"--slippage",(S $Slippage)) "optimize($sym)"
    } catch {
      Warn ("Optimize falló para {0}. Sigo con DEFAULT." -f $sym)
      continue
    }

    $bestJson = Join-Path (ReportBase $sym) ("{0}_best_params.json" -f $sym)
    if (-not (Test-Path $bestJson)) { Warn ("No se encontró {0} para {1}. Sigo con DEFAULT." -f $bestJson,$sym); continue }

    try {
      $best = Get-Content $bestJson | ConvertFrom-Json
      if ($null -eq $best.buy_thr -or $null -eq $best.sell_thr -or $null -eq $best.min_edge) { Warn ("best_params incompleto para {0}." -f $sym); continue }
      $optBuy  = [double]$best.buy_thr
      $optSell = [double]$best.sell_thr
      $optEdge = [double]$best.min_edge

      Info ("Backtest OPTIMIZED {0} (buy={1}, sell={2}, edge={3})…" -f $sym,$optBuy,$optSell,$optEdge)
      Run-Py @("-m","src.backtest.run_backtest","--symbol",$sym,"--csv",$csv,"--fee",(S $Fee),"--slippage",(S $Slippage),"--buy-thr",(S $optBuy),"--sell-thr",(S $optSell),"--min-edge",(S $optEdge),"--tag","optimized") "backtest_optimized($sym)"
      Ok ("Backtest OPTIMIZED OK {0}" -f $sym)
    } catch {
      Err ("Backtest OPTIMIZED falló para {0}." -f $sym)
    }
  }
} else {
  Warn "SkipOptimize activo → no se realizará optimización ni backtest optimized."
}

# ========== 5) COMPARATIVA ==========
function Load-Metrics([string]$sym, [string]$tag="default"){
  $summary = ("reports\{0}_summary.json" -f $sym)
  if ($tag -eq "optimized") { $summary = ("reports\{0}_summary_optimized.json" -f $sym) }
  if (-not (Test-Path $summary)) { return $null }
  try { return Get-Content $summary | ConvertFrom-Json } catch { return $null }
}

$rows = @()
foreach ($sym in $symbolsArr) {
  $d = Load-Metrics $sym "default"
  $o = Load-Metrics $sym "optimized"
  if ($d) { $rows += [pscustomobject]@{Symbol=$sym; Variant="default";   CAGR=$d.cagr; Sharpe=$d.sharpe; Sortino=$d.sortino; MaxDD=$d.max_drawdown; WinRate=$d.win_rate; Trades=$d.trades; FinalEquity=$d.final_equity} }
  if ($o) { $rows += [pscustomobject]@{Symbol=$sym; Variant="optimized"; CAGR=$o.cagr; Sharpe=$o.sharpe; Sortino=$o.sortino; MaxDD=$o.max_drawdown; WinRate=$o.win_rate; Trades=$o.trades; FinalEquity=$o.final_equity} }
}

$portfolioCsv = "reports\portfolio_equity.csv"
if (Test-Path $portfolioCsv) {
  try {
    $port = Import-Csv $portfolioCsv
    $last = $port | Select-Object -Last 1
    $rows += [pscustomobject]@{Symbol="PORTFOLIO"; Variant="portfolio"; CAGR=$null; Sharpe=$null; Sortino=$null; MaxDD=$null; WinRate=$null; Trades=$null; FinalEquity=$last.Equity}
  } catch {
    Warn "No se pudo leer reports\portfolio_equity.csv para comparativa."
  }
}

$compCsv = "reports\comparison_default_optimized_portfolio.csv"
$rows | Export-Csv -NoTypeInformation -Path $compCsv -Encoding UTF8
Write-Host ""
if (Test-Path $compCsv) {
  Ok "Resumen final (DEFAULT vs OPTIMIZED vs PORTFOLIO):"
  Import-Csv $compCsv | Sort-Object Symbol, Variant | Format-Table -AutoSize
  Ok ("Comparativa guardada en {0}" -f $compCsv)
} else {
  Warn "No se generó comparativa."
}

# ========== 6) Señales + Paper/Live trading ==========
if ($Mode -ne "backtest") {
  $signalsJson = "reports\signals_today.json"
  $hasCustomSignals = Test-Path ".\src\signals\generate_cli.py"
  if ($hasCustomSignals) {
    Info "Generando señales con src.signals.generate_cli…"
    try {
      Run-Py @("-m","src.signals.generate_cli","--symbols",$Symbols,"--fiat",$Fiat,"--out",$signalsJson) "signals_generate_cli"
    } catch {
      Err "signals_generate_cli falló. Intentaré fallback SMA."
      $hasCustomSignals = $false
    }
  }
  if (-not $hasCustomSignals) {
    Info "Generando señales fallback SMA(50/200)…"
    $py = @"
import os, json, pandas as pd
from pathlib import Path

symbols = "${Symbols}".split(",")
fiat = "${Fiat}"
out = Path(r"$signalsJson")
out.parent.mkdir(parents=True, exist_ok=True)

def find_existing(sym):
    cands = [
        f"data/{sym}_{fiat}_1d.csv",
        f"data/{sym}_{fiat}_1d_labeled.csv",
        f"data/{sym}_USDT_1d.csv",
        f"data/{sym}_USDT_1d_labeled.csv",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

signals = []
for sym in [s.strip() for s in symbols if s.strip()]:
    path = find_existing(sym)
    if not path:
        continue
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns and "date" in df.columns:
        df.rename(columns={"date":"timestamp"}, inplace=True)
    for k in ["open","high","low","close"]:
        if k not in df.columns:
            df[k] = None
    df = df.dropna(subset=["close"])
    if len(df) < 210:
        continue
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    last = df.iloc[-1]; prev = df.iloc[-2]
    signal = "hold"
    if prev["sma50"] <= prev["sma200"] and last["sma50"] > last["sma200"]:
        signal = "buy"
    elif prev["sma50"] >= prev["sma200"] and last["sma50"] < last["sma200"]:
        signal = "sell"
    signals.append({"symbol": sym, "fiat": "${Fiat}", "signal": signal, "price": float(last["close"])})
out.write_text(json.dumps({"signals": signals}, indent=2))
print(f"[OK] señales en {out}")
"@
    $tmp = Join-Path $env:TEMP "signals_fallback.py"
    Set-Content -Path $tmp -Value $py -Encoding UTF8
    Run-Py @($tmp) "signals_fallback"
  }

  if (Test-Path $signalsJson) {
    Info ("Procesando {0} con {1}…" -f $Mode,$signalsJson)
    $pyTrade = @"
import os, json, time
from pathlib import Path
import ccxt

MODE = "${Mode}"
EXCHANGE = os.getenv("EXCHANGE", "${Exchange}")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
API_PASSWORD = os.getenv("API_PASSWORD", "")
BASE = float("${BaseCapital}")
RISK_PCT = float("${RiskPct}")
MAX_DAILY_LOSS = float("${MaxDailyLossPct}")
MAX_POS_PCT = float("${MaxPosPct}")
MIN_NOTIONAL = float("${MinNotional}")

sig = json.loads(Path(r"$signalsJson").read_text())
signals = sig.get("signals", [])

def build_exchange():
    klass = getattr(ccxt, EXCHANGE)
    params = {}
    if API_PASSWORD: params["password"] = API_PASSWORD
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "options": params})
    ex.set_sandbox_mode(MODE=="paper")
    return ex

def clamp_size(value, price):
    units = max(0.0, value / max(1e-9, price))
    if value < MIN_NOTIONAL: return 0.0
    return units

def decide_order(ex, sym, signal, price):
    market = f"{sym}/USDT"
    notional = min(BASE*RISK_PCT, BASE*MAX_POS_PCT)
    size = clamp_size(notional, price or 1.0)
    if size <= 0:
        return None
    side = "buy" if signal=="buy" else ("sell" if signal=="sell" else None)
    if side is None: return None
    ex.load_markets()
    amount = ex.amount_to_precision(market, size)
    return {"symbol": market, "type":"market", "side": side, "amount": amount}

def main():
    if MODE not in ("paper","live"):
        print("[SKIP] MODE is not paper/live"); return
    ex = build_exchange()
    sent = []
    for s in signals:
        sym, sig, price = s["symbol"], s["signal"], float(s.get("price") or 0)
        if sig not in ("buy","sell"): continue
        try:
            o = decide_order(ex, sym, sig, price)
            if not o:
                continue
            print(f"[{MODE.upper()}] {sym}: {sig} -> {o['amount']} @ market")
            if MODE=="live":
                res = ex.create_order(o["symbol"], o["type"], o["side"], o["amount"])
                print(f"[LIVE OK] orderId={res.get('id')}")
            sent.append({"symbol": sym, "side": o["side"], "amount": o["amount"]})
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERR] {sym} {sig}: {e}")
    Path("reports/orders_last.json").write_text(json.dumps({"mode":MODE, "sent":sent}, indent=2))
    print(f"[OK] órdenes registradas en reports/orders_last.json")

if __name__ == "__main__":
    main()
"@
    $tmpTrade = Join-Path $env:TEMP "trade_exec.py"
    Set-Content -Path $tmpTrade -Value $pyTrade -Encoding UTF8
    Run-Py @($tmpTrade) "trade_exec"
  } else {
    Warn "No hay señales en reports\signals_today.json — no se ejecutan órdenes."
  }
}

Ok "Pipeline completo."
exit 0
