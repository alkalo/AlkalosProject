# AlkalosProject

This repository contains utilities for running quantitative trading
strategies.  The main component is ``SignalStrategy`` which loads a
pre-trained classification model and emits BUY/SELL/HOLD signals based on
the predicted probability of the asset moving up.

## SignalStrategy

```python
from signal_strategy import SignalStrategy

strategy = SignalStrategy(
    model_path="model.pkl",
    scaler_path="scaler.pkl",
    features_path="features.pkl",
)

signal = strategy.generate_signal(df_window)
```

Signals are produced using the following default rules:

* **BUY**  – probability >= 0.6
* **SELL** – probability <= 0.4
* **HOLD** – otherwise

Thresholds and an additional ``min_edge`` margin can be customized when
instantiating the strategy.
