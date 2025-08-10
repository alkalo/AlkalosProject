import joblib
import lightgbm as lgb

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class LGBMClassifierModel:
    """LightGBM classifier wrapper."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, X, y):
        """Fit the underlying LGBM classifier."""
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)

    def predict_proba(self, X):
        """Predict probabilities for samples in X."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        return self.model.predict_proba(X)

    def save(self, path: str):
        """Persist model to disk."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)


class KerasLSTMClassifier:
    """Binary LSTM classifier using Keras."""

    def __init__(self, window: int, n_features: int):
        self.window = window
        self.n_features = n_features
        self.model = self._build_model()
        self.best_model_path = "best_lstm_model.h5"

    def _build_model(self):
        model = Sequential([
            LSTM(50, input_shape=(self.window, self.n_features)),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2):
        """Train the LSTM classifier."""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(self.best_model_path, save_best_only=True, monitor="val_loss"),
        ]
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )
        # Load the best model saved by ModelCheckpoint
        self.model = load_model(self.best_model_path)

    def predict_proba(self, X):
        """Predict probabilities for samples in X."""
        return self.model.predict(X)

    def save(self, path: str):
        """Save model to an H5 file."""
        self.model.save(path)

    def load(self, path: str):
        """Load model from an H5 file."""
        self.model = load_model(path)
