<# 
  install_env.ps1
  - Crea/activa .venv de forma robusta en Windows
  - Soluciona el caso "python no se reconoce" usando 'py' y fallback a winget
  - Actualiza pip/setuptools/wheel y instala requirements.txt
#>

[CmdletBinding()]
param(
  [string]$PyVersion = "3.11",      # versión preferida si hay que instalar
  [string]$VenvPath = ".venv",
  [switch]$ForceRecreate
)

function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[ERR]  $msg" -ForegroundColor Red }

function Ensure-Winget {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Err "winget no está disponible. Instálalo desde Microsoft Store o habilítalo en tu Windows."
    throw "winget_missing"
  }
}

function Ensure-Python {
  # 1) Intenta 'py' (Python Launcher para Windows) → evita el alias de la Store
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    Write-Info "Usando 'py' para gestionar Python."
    return "py"
  }

  # 2) Intenta 'python' real (no alias)
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    Write-Info "Encontrado 'python' en PATH: $($python.Source)"
    return "python"
  }

  # 3) Instalar con winget si no hay nada
  Write-Warn "Python no encontrado. Intentando instalar Python $PyVersion con winget…"
  Ensure-Winget
  # Microsoft.Python.3.11 suele ser el id estable; si usas otra versión, ajusta el paquete
  $pkgId = "Python.Python.$($PyVersion.Replace('.',''))"
  # fallback común: "Python.Python.3.11" o "Python.Python.3.12"
  $ids = @("Python.Python.$($PyVersion.Replace('.',''))","Python.Python.3.12","Python.Python.3.11")
  $installed = $false
  foreach ($id in $ids) {
    try {
      winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements
      $installed = $true
      break
    } catch {
      Write-Warn "Fallo instalando $id. Probando siguiente…"
    }
  }
  if (-not $installed) {
    Write-Err "No se pudo instalar Python automáticamente. Instálalo manualmente (python.org) y reintenta."
    throw "python_install_failed"
  }

  # Reintenta detección
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) { return "py" }
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) { return "python" }

  Write-Err "Python sigue sin detectarse tras la instalación."
  throw "python_not_detected"
}

function New-Or-Reset-Venv($pyCmd, $venv) {
  if ($ForceRecreate -and (Test-Path $venv)) {
    Write-Warn "Eliminando venv existente por --ForceRecreate…"
    Remove-Item -Recurse -Force $venv
  }
  if (-not (Test-Path $venv)) {
    Write-Info "Creando entorno virtual en $venv…"
    if ($pyCmd -eq "py") {
      & py -$PyVersion -m venv $venv 2>$null
      if ($LASTEXITCODE -ne 0) { & py -m venv $venv }
    } else {
      & $pyCmd -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) {
      Write-Err "Fallo creando el entorno virtual."
      throw "venv_create_failed"
    }
  } else {
    Write-Info "Usando entorno virtual existente: $venv"
  }
}

function Activate-Venv($venv) {
  $activate = Join-Path $venv "Scripts\Activate.ps1"
  if (-not (Test-Path $activate)) {
    Write-Err "No se encontró $activate"
    throw "venv_missing_activate"
  }
  Write-Info "Activando venv…"
  . $activate
  $null = & python --version
  if ($LASTEXITCODE -ne 0) {
    Write-Warn "'python' podría apuntar al alias de la Store. Probando 'py' dentro del venv…"
    $env:Path = (Join-Path (Resolve-Path $venv) "Scripts") + ";" + $env:Path
  }
  Write-Ok "Venv activo. Python: $(python --version 2>$null)"
}

function Upgrade-Pip-And-Install {
  Write-Info "Actualizando pip, setuptools, wheel…"
  python -m pip install --upgrade pip setuptools wheel
  if (Test-Path "wheelhouse") {
    Write-Info "Instalando ruedas locales (si existen)…"
    python -m pip install --no-index --find-links=wheelhouse -r requirements.txt
  } else {
    Write-Info "Instalando requirements.txt…"
    python -m pip install -r requirements.txt
  }
  Write-Ok "Dependencias instaladas."
}

try {
  $pyCmd = Ensure-Python
  New-Or-Reset-Venv -pyCmd $pyCmd -venv $VenvPath
  Activate-Venv -venv $VenvPath
  Upgrade-Pip-And-Install
  Write-Ok "Entorno listo."
} catch {
  Write-Err "install_env.ps1 falló: $($_.Exception.Message)"
  exit 1
}
