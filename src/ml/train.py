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
from pathlib import Path
from typing import Iterable, Sequence, Any

import joblib


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
    model_dir: str = "models",
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

    if model is None:
        model = {"weights": [0.1, 0.2, 0.3]}
    joblib.dump(model, artefact_dir / "model.pkl")

    if is_lstm:
        # The actual LSTM weights would live in this file.  For test purposes we
        # merely create an empty placeholder so that the strategy can detect the
        # model type.
        (artefact_dir / "model.h5").write_bytes(b"")

    if scaler is None:
        scaler = _IdentityScaler()
    joblib.dump(scaler, artefact_dir / "scaler.pkl")

    with open(artefact_dir / "features.json", "w", encoding="utf-8") as fh:
        json.dump(list(feature_names), fh)

    with open(artefact_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report or {}, fh)

    with open(artefact_dir / "diagnostic.png", "wb") as fh:
        fh.write(diagnostic or b"")


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    train("EXAMPLE", ["feat1", "feat2"])  # type: ignore[arg-type]
