# AlkalosProject

Este proyecto demuestra un flujo básico de descarga de datos, entrenamiento de
un modelo y generación de señales para backtesting y paper trading.

## Requisitos y setup (Windows)

1. Instalar [Python 3.10+](https://www.python.org/downloads/windows/).
2. Clonar el repositorio y crear un entorno virtual:
   ```powershell
   git clone <url>
   cd AlkalosProject
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Flujo de trabajo

1. **Fetch** – Descargar datos OHLCV:
   ```powershell
   python -m src.data_fetch --source yf --symbols BTC --fiat USD --days 365
   ```
2. **Train** – Entrenar y guardar artefactos:
   ```powershell
   python -m src.ml.train_cli --csv data/BTC_USD_1d.csv --symbol BTC --model lgbm
   ```
3. **Backtest** – Ejecutar el backtest:
   ```powershell
   python -m src.backtest.run_backtest --symbol BTC --csv data/BTC_USD_1d.csv
   ```
4. **Paper/Live** – Lanzar el bot de trading:
   ```powershell
   # Paper (modo por defecto)
   python -m src.live.paper_bot --symbol BTC/USDT --csv data/BTC_USDT_1d.csv

   # Live (requiere credenciales y un exchange soportado por ccxt)
   python -m src.live.paper_bot --symbol BTC/USDT --csv data/BTC_USDT_1d.csv --mode live --exchange binance
   ```

## Artefactos y reportes

- **Datos**: `data/{symbol}_{fiat}_1d.csv`
- **Modelos**: `models/<symbol>/` (`model.pkl`, `scaler.pkl`, `features.json`,
  `report.json`, `diagnostic.png`)
- **Backtests**: `reports/{symbol}_summary.json`,
  `reports/{symbol}_equity.png`, `reports/{symbol}_trades.csv`
- **Bot de trading**: `reports/paper_bot_{symbol}.csv`
- **Logs**: `logs/*.log` (rotados automáticamente)

## Ajustar fees y umbrales

- **Backtest**: parámetros `--fee`, `--slippage`, `--buy-thr`, `--sell-thr` y
  `--min-edge` de `src/backtest/run_backtest.py`.
- **Bot de trading**: constantes `FEE_RATE` y `SLIPPAGE`, valores `buy_thr`,
  `sell_thr` y `min_edge` al instanciar `SignalStrategy`, además de las
  opciones `--mode`, `--exchange`, `--max-allocation` y `--max-drawdown`.

## Módulos experimentales

Las utilidades `src/ml/feature_engineering.py` y `src/ml/data_utils.py`
se mantienen para exploraciones fuera del flujo principal de entrenamiento y
pueden cambiar sin previo aviso.

## Descargo de responsabilidad

Este código se proporciona solo con fines educativos. No constituye asesoría
financiera ni una recomendación de inversión. Use los scripts bajo su propia
responsabilidad.

