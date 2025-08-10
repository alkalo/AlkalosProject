"""Training utilities for AlkalosProject."""
from __future__ import annotations

import os
from typing import Any
import joblib


def train(model_dir: str, features: Any) -> None:
    """Train a model and persist the features.

    Parameters
    ----------
    model_dir:
        Directory where the model and features will be stored.
    features:
        The feature data used for training.
    """
    os.makedirs(model_dir, exist_ok=True)
    # Placeholder model; in a real project this would be a trained estimator.
    model = {"weights": [0.1, 0.2, 0.3]}
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    # Persist features with joblib for fast loading.
    joblib.dump(features, os.path.join(model_dir, "features.pkl"))


if __name__ == "__main__":
    train("model", [1, 2, 3])
