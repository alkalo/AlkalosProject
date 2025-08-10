"""SignalStrategy module for loading features."""
from __future__ import annotations

import os
import joblib


class SignalStrategy:
    """Simple strategy that loads precomputed features."""

    def __init__(self, model_dir: str) -> None:
        feature_path = os.path.join(model_dir, "features.pkl")
        self.features = joblib.load(feature_path)
