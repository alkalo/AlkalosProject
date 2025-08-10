"""Training utilities for AlkalosProject.

This module provides a very small stub ``train`` function that persists a
trained model and its accompanying artefacts following the convention used by
``SignalStrategy`` during backtesting.  The function is intentionally minimal
but mirrors the expected directory layout so that unit tests can exercise the
loading logic of the strategy component.
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Sequence, Any
import importlib

import joblib
import logging

from src.utils.env import get_logs_dir, get_models_dir


logger = logging.getLogger(__name__)


def _ensure_dirs(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


_ensure_dirs(str(get_logs_dir() / "train.log"))
logging.basicConfig(
    filename=str(get_logs_dir() / "train.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class _IdentityScaler:
    """Fallback scaler used for tests.

    The scaler simply returns the input unchanged which is sufficient for the
    lightweight unit tests in this kata-style repository.
    """

    def transform(self, X):  # pragma: no cover - trivial behaviour
        return X


def train(
    symbol: str,
    feature_names: Sequence[str],
    model: Any | None = None,
    *,
    model_dir: str = str(get_models_dir()),
    scaler: Any | None = None,
    is_lstm: bool = False,
    report: dict | None = None,
    diagnostic: bytes | None = None,
) -> None:
    """Persist a model and its artefacts following the project contract.

    Parameters
    ----------
    symbol:
        Trading symbol for which the model was trained.
    feature_names:
        Ordered list of feature names used during training.
    model:
        Trained estimator to serialise with :mod:`joblib`.  When ``None`` a
        simple placeholder object is stored.
    model_dir:
        Root directory under which the symbol folder will be created.
    scaler:
        Optional pre-processing scaler.  When omitted an identity scaler is
        stored so that downstream code can unconditionally apply
        ``scaler.transform``.
    is_lstm:
        If ``True`` an empty ``model.h5`` file is created to mimic the
        presence of an LSTM model.
    report:
        Arbitrary metadata to be written to ``report.json``.
    diagnostic:
        Optional binary content for ``diagnostic.png``.
    """

    artefact_dir = Path(model_dir) / symbol
    artefact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving artefacts for %s", symbol)

    if model is None:
        model = {"weights": [0.1, 0.2, 0.3]}
    try:
        joblib.dump(model, artefact_dir / "model.pkl")
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to save model.pkl: %s", exc)
        raise

    if is_lstm:
        # The actual LSTM weights would live in this file.  For test purposes we
        # merely create an empty placeholder so that the strategy can detect the
        # model type.
        try:
            (artefact_dir / "model.h5").write_bytes(b"")
        except OSError as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to write model.h5: %s", exc)
            raise

    if scaler is None:
        scaler = _IdentityScaler()
    try:
        joblib.dump(scaler, artefact_dir / "scaler.pkl")
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to save scaler.pkl: %s", exc)
        raise

    try:
        with open(artefact_dir / "features.json", "w", encoding="utf-8") as fh:
            json.dump(list(feature_names), fh)
        with open(artefact_dir / "report.json", "w", encoding="utf-8") as fh:
            json.dump(report or {}, fh)
        with open(artefact_dir / "diagnostic.png", "wb") as fh:
            fh.write(diagnostic or b"")
    except OSError as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to write artefact files: %s", exc)
        raise

    logger.info("Artefacts stored in %s", artefact_dir)


def train_evaluate(
    *,
    csv_path: str,
    symbol: str,
    model_type: str,
    horizon: int,
    window: int,
    outdir: str = str(get_models_dir()),
) -> None:
    """Train a trivial model and persist artefacts.

    The implementation is intentionally lightweight; it trains a simple model on
    the provided CSV file and stores artefacts using :func:`train`.  The CSV is
    expected to contain a ``target`` column and optionally a ``date`` column
    used solely for reporting purposes.
    """

    import pandas as pd  # Imported lazily to keep test environment minimal
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    logger.info("Training %s model for %s", model_type, symbol)
    model_type_lower = model_type.lower()
    if model_type_lower == "lstm":
        try:
            importlib.import_module("tensorflow")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TensorFlow no está instalado; instala tensorflow-cpu"
            ) from exc

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:  # pragma: no cover - best effort logging
        logger.error("CSV file not found: %s", csv_path)
        raise
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to read %s: %s", csv_path, exc)
        raise

    if "target" not in df.columns:
        logger.error("CSV must contain a 'target' column")
        raise ValueError("CSV must contain a 'target' column")

    dates = pd.to_datetime(df["date"]) if "date" in df.columns else None
    X = df.drop(columns=[col for col in ["target", "date"] if col in df.columns])
    y = df["target"]

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X,
        y,
        dates,
        test_size=0.2,
        shuffle=False,
    )

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")

    report = {
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "num_features": X.shape[1],
        "window": window,
        "horizon": horizon,
    }
    if dates is not None:
        report.update(
            {
                "train_start": dates_train.min().strftime("%Y-%m-%d"),
                "train_end": dates_train.max().strftime("%Y-%m-%d"),
                "test_start": dates_test.min().strftime("%Y-%m-%d"),
                "test_end": dates_test.max().strftime("%Y-%m-%d"),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(y_test.values, label="y_true")
    axes[0].plot(proba, label="p_up")
    axes[0].legend()
    if len(set(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[1].plot(fpr, tpr)
        axes[1].plot([0, 1], [0, 1], "--")
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    diagnostic = buf.getvalue()

    try:
        train(
            symbol,
            list(X.columns),
            model=model,
            model_dir=outdir,
            is_lstm=model_type_lower == "lstm",
            report=report,
            diagnostic=diagnostic,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.exception("Failed to persist artefacts: %s", exc)
        raise

    logger.info("Training complete for %s", symbol)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    train("EXAMPLE", ["feat1", "feat2"])  # type: ignore[arg-type]
