# AlkalosProject

This project demonstrates training a simple model and saving its features using `joblib`.

The training script saves two files in the specified model directory:

- `model.pkl` – the trained model.
- `features.pkl` – the features used to train the model.

`SignalStrategy` loads `features.pkl` with `joblib.load`.

