"""Стадия evaluate.

Считает метрики обученной модели на отложенном test-наборе
и проверяет порог качества evaluate.min_roc_auc.
Завершается с кодом 1, если ROC-AUC ниже порога — так CI не пустит
плохую модель дальше.

Запуск: python -m src.evaluate
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import TARGET, feature_columns, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


def compute_metrics(params: dict[str, Any], pipe: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    thresh = params["evaluate"]["threshold"]

    y_true = df[TARGET]

    y_proba = pipe.predict_proba(df[feature_columns(params)])[:, 1]
    y_pred = (y_proba >= thresh).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def main() -> None:
    params = load_params()

    model_path = os.path.join(resolve(params["train"]["save_dir"]), "model.joblib")
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}. Run `make train` first")
        sys.exit(1)
    pipe = joblib.load(model_path)
    log.info(f"Model loaded from: {model_path}")

    test = pd.read_csv(os.path.join(resolve(params["data"]["processed_dir"]), "test.csv"))
    log.info("Test data loaded")

    metrics = compute_metrics(params, pipe, test)
    log.info(f"Test metrics: {metrics}")

    reports_dir = resolve(params["reports"]["dir"])
    os.makedirs(reports_dir, exist_ok=True)
    reports_path = os.path.join(reports_dir, "eval_metrics.json")
    with open(reports_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to: {reports_path}")

    min_roc_auc = params["evaluate"]["min_roc_auc"]
    if metrics["roc_auc"] < min_roc_auc:
        log.error(f"ROC-AUC {metrics['roc_auc']:.4f} is lower than specified threshold {min_roc_auc:.4f}")
        sys.exit(1)
    log.info(f"Quality gate passed: ROC-AUC {metrics['roc_auc']:.4f} >= {min_roc_auc:.4f}")


if __name__ == "__main__":
    main()
