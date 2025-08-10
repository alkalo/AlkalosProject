# 📈 AlkalosProject
Sistema de **backtesting** y **trading algorítmico** para criptomonedas con pipeline de extremo a extremo: **fetch → train → backtest → (paper/live)**.  
Diseñado para Windows + PowerShell con Python ≥ 3.10.

---

## 🔰 TL;DR (primeros 5 minutos)
```powershell
# 1) Clonar
git clone https://github.com/alkalo/AlkalosProject.git
cd AlkalosProject

# 2) Permitir scripts (solo 1ª vez)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3) Instalar entorno
.\install_env.ps1

# 4) Prueba E2E rápida (descarga -> entrena -> backtest -> report)
.\sanity_e2e.ps1
```
> Al terminar, encontrarás resultados en **/reports**.

---

## 🧰 Requisitos
- **Windows 10/11**
- **Python 3.10+** (64‑bit)
- **PowerShell** (abre como usuario normal; si un script pide permisos, ver sección de permisos)
- Conexión a internet (descarga de datos vía **CCXT** o **Yahoo Finance**)

> Opcional: WSL o macOS/Linux son viables, pero este README está optimizado para PowerShell en Windows.

---

## 🗂️ Estructura del proyecto
```
AlkalosProject/
├─ configs/        # Configs de modelos/exchanges/trading
├─ data/           # CSVs descargados (OHLCV)
├─ logs/           # Logs de ejecución
├─ models/         # Modelos entrenados
├─ reports/        # Resultados del backtest
├─ src/            # Código fuente
├─ tests/          # Tests unitarios
├─ install_env.ps1 # Crea venv + pip install -r requirements
├─ sanity_e2e.ps1  # Pipeline mínimo de extremo a extremo
├─ make.ps1        # Atajos (si aplica)
└─ README.md
```

---

## 🧭 Paso a paso (desde cero)

### 1) Clonar el repositorio
```powershell
git clone https://github.com/alkalo/AlkalosProject.git
cd AlkalosProject
```

### 2) Permitir ejecución de scripts (si es la primera vez)
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
> Si la política de ejecución te bloquea, **abre PowerShell** y repite el comando.  
> No uses “Unrestricted”; `RemoteSigned` es suficiente para scripts locales.

### 3) Instalar el entorno
```powershell
.\install_env.ps1
```
Este script normalmente:
- Crea/actualiza el **entorno virtual** (por ejemplo `.venv/`)
- Instala dependencias de `requirements.txt` (incluye `ccxt`, `pandas`, `numpy`, `yfinance` si aplica, etc.)

> Si algo falla en esta etapa, ver **🩺 Troubleshooting** al final.

### 4) Verificación rápida E2E
```powershell
.\sanity_e2e.ps1
```
Qué hace:
- Descarga un dataset pequeño de ejemplo
- Entrena un modelo rápido
- Ejecuta **backtest** y deja un informe en **/reports**

Si esto termina OK, tu entorno está bien configurado ✅.

---

## 📥 Descarga de datos históricos (OHLCV)

### Opción A – **CCXT** (recomendada para cripto)
```powershell
# Un símbolo
python -m src.data_fetch --source ccxt --exchange binance --symbols BTC --fiat USDT --days 1825 --timeframe 1d

# Varios símbolos (requiere el script actualizado con split por coma)
python -m src.data_fetch --source ccxt --exchange binance --symbols BTC,ETH,SOL --fiat USDT --days 1825 --timeframe 1d
```
Salida esperada en `/data`:
```
data/BTC_USDT_1d.csv
data/ETH_USDT_1d.csv
data/SOL_USDT_1d.csv
```

**Parámetros útiles**
- `--exchange`: por ejemplo `binance`, `kucoin`, etc. (debe existir en CCXT)
- `--symbols`: separado por comas (`BTC,ETH`)
- `--fiat`: moneda cotizada (típicamente `USDT` en cripto)
- `--days`: días hacia atrás (ej. 1825 = 5 años)
- `--timeframe`: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w` (según soporte del exchange)

> Si ves `binance does not have market symbol BTC,ETH/USDT`, significa que el script no separó bien los símbolos.  
> Usa **uno por comando** o **actualiza** `src/data_fetch` a la versión que soporta split (ver notas más abajo).

### Opción B – **Yahoo Finance** (acciones/ETFs y algunos pares USD)
```powershell
python -m src.data_fetch --source yf --symbols BTC,ETH --fiat USD --days 1825 --timeframe 1d
```
> Nota: para cripto a veces Yahoo limita o cambia símbolos; **CCXT** es más fiable.

---

## 🧠 Entrenamiento de modelos

Ejemplo con LightGBM (`lgbm`):
```powershell
python -m src.ml.train_cli --model lgbm --csv data\BTC_USDT_1d.csv --symbol BTC --horizon 1 --window 5
```
**Parámetros típicos**
- `--model`: `lgbm`, `rf`, etc. (según lo que soporte tu código)
- `--csv`: ruta al CSV OHLCV (ver sección de descarga)
- `--symbol`: símbolo base (ej. `BTC`)
- `--horizon`: horizonte de predicción en velas (ej. 1 = próxima vela)
- `--window`: ventana para features (ej. 5)

Salida esperada:
- Artefactos de modelo en `/models` (según implementación)
- Logs en `/logs` (si aplica)

---

## 🧪 Backtesting

Ejemplo:
```powershell
# En una línea
python -m src.backtest.run_backtest --symbol BTC --csv data\BTC_USDT_1d.csv --fee 0.001 --slippage 0.0005 --buy-thr 0.6 --sell-thr 0.4 --min-edge 0.02

# O con continuaciones de línea en PowerShell
python -m src.backtest.run_backtest `
  --symbol BTC `
  --csv data\BTC_USDT_1d.csv `
  --fee 0.001 `
  --slippage 0.0005 `
  --buy-thr 0.6 `
  --sell-thr 0.4 `
  --min-edge 0.02
```
Al finalizar, verás resultados en **/reports**. Suelen incluir métricas, curva de equity y/o CSV de operaciones (dependiendo de tu implementación).

---

## 📈 Paper / Live Trading (opcional)
- Scripts típicos: `src/paper/run_paper`, `src/live/run_live`
- Requieren **API keys** y parámetros en `/configs/` (exchange, símbolo, tamaño de orden, límites de riesgo, etc.)
- Confirma bien en modo **paper** antes de pasar a **live**.

> Si aún no tienes estos módulos configurados, puedes limitarte a **fetch → train → backtest**.

---

## ⚡ Atajos con `make.ps1` (si aplica)
Algunos repos incluyen atajos. Ejemplos típicos:
```powershell
# Entrenar + backtest de BTC con configuración por defecto
.\make.ps1 train_backtest BTC

# Solo backtest con CSV concreto
.\make.ps1 backtest data\BTC_USDT_1d.csv
```
> Revisa el contenido de `make.ps1` para ver los comandos disponibles en tu versión.

---

## 🔄 Actualizar dependencias
Si cambiaste `requirements.txt` o el entorno:
```powershell
.\install_env.ps1
```
> Esto reinstala/actualiza la venv y las dependencias necesarias.

---

## 🧪 Flujos útiles (recetas)

### 1) Preparar dataset para varios símbolos
```powershell
# CCXT (recomendado)
python -m src.data_fetch --source ccxt --exchange binance --symbols BTC,ETH,SOL --fiat USDT --days 1825 --timeframe 1d
```

### 2) Entrenar y backtest de un símbolo
```powershell
python -m src.ml.train_cli --model lgbm --csv data\BTC_USDT_1d.csv --symbol BTC --horizon 1 --window 5

python -m src.backtest.run_backtest --symbol BTC --csv data\BTC_USDT_1d.csv `
  --fee 0.001 --slippage 0.0005 --buy-thr 0.6 --sell-thr 0.4 --min-edge 0.02
```

### 3) Repetir para otro símbolo
```powershell
python -m src.ml.train_cli --model lgbm --csv data\ETH_USDT_1d.csv --symbol ETH --horizon 1 --window 5

python -m src.backtest.run_backtest --symbol ETH --csv data\ETH_USDT_1d.csv `
  --fee 0.001 --slippage 0.0005 --buy-thr 0.6 --sell-thr 0.4 --min-edge 0.02
```

---

## 🩺 Troubleshooting (errores comunes)

### 1) `YFTzMissingError ('BTC-USD')` (Yahoo Finance)
- A veces Yahoo limita cripto o cambia símbolos.
- Solución rápida: usa **CCXT** para cripto y deja YF para acciones/ETFs o pares USD que confirmen datos.

### 2) `binance does not have market symbol BTC,ETH/USDT` (CCXT)
- Pasaste varios símbolos sin separarlos correctamente en el script.
- **Opciones**:
  - Llamar **un símbolo por vez**: `--symbols BTC` (y luego `--symbols ETH`)
  - Actualizar `src/data_fetch` para **split** por comas (sección “Notas de implementación”).

### 3) CSV vacío o columnas como `object`
- Revisa que `--days` cubra un rango realista y que el exchange tenga histórico suficiente.
- Asegúrate de que el `--timeframe` está soportado por el exchange.
- Verifica que tu zona horaria/locale no esté “forzando” parseos raros (mantén ISO‑8601).

### 4) Problemas de permisos en PowerShell
- Ejecuta: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Si sigue fallando, **abre PowerShell como Administrador** y repite.

### 5) No se crea la venv / fallos de pip
- Confirma `python --version` ≥ 3.10 (64‑bit)
- Vuelve a correr: `.\install_env.ps1`
- Si sigue, borra `.venv/` y reinstala.

---

## 🛠️ Notas de implementación (para desarrolladores)
- El módulo `src.data_fetch` soporta:
  - **CCXT** con paginación y múltiples símbolos (si aplicaste el patch que separa `--symbols` por comas y construye pares `SYMBOL/FIAT`).
  - **Yahoo Finance** como alternativa (principalmente `*_USD_1d.csv`).
- Convención de salida: `data/{SYMBOL}_{FIAT}_{TIMEFRAME}.csv` con columnas:
  `timestamp, timestamp_ms, open, high, low, close, volume`
- Si necesitas mapear archivos `*_USD_*` a `*_USDT_*` (o viceversa), crea un pequeño helper para copiar/renombrar y mantener compatibilidad con tus scripts de backtest.

---

## 🧾 Licencia
MIT License.

---

## 🤝 Contribuciones
¡PRs y sugerencias bienvenidas! Mantén los módulos pequeños, con logs claros y tests básicos en `tests/` donde sea posible.
