"""Стадия train.

TODO (занятие 1): перенести сюда логику из notebooks/baseline_notebook.py,
исправив всё, что вы в ней нашли.

Обязательно:
  * никаких абсолютных путей — только src.config.resolve();
  * никаких магических чисел — только params.yaml;
  * зафиксированный seed;
  * модель сохраняется в models/model.joblib вместе с препроцессором;
  * метрики пишутся в reports/train_metrics.json.

Запуск: python -m src.train
"""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import TARGET, feature_columns, get_git_sha, load_params, resolve
from src.features import build_preprocessor
from src.logging_setup import setup_logging

log = setup_logging()


def load_dataset(params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    d = params["data"]
    processed_dir = resolve(d["processed_dir"])

    return {
        "train": pd.read_csv(os.path.join(processed_dir, "train.csv")),
        "val": pd.read_csv(os.path.join(processed_dir, "val.csv")),
    }


def build_model(params: dict[str, Any]):
    seed = params["seed"]
    t = params["train"]
    name = t["model"]
    cfg = t.get(name, {})

    match name:
        case "logreg":
            return LogisticRegression(random_state=seed, **cfg)
        case "random_forest":
            return RandomForestClassifier(random_state=seed, **cfg)
        case "gradient_boosting":
            return GradientBoostingClassifier(random_state=seed, **cfg)
        case _:
            return NotImplementedError(f"Uknown model name: {name}")


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(params)),
            ("model", build_model(params)),
        ]
    )


def train_model(params: dict[str, Any], pipe: Pipeline, train: pd.DataFrame) -> None:
    pipe.fit(train[feature_columns(params)], train[TARGET])


def validate_model(params: dict[str, Any], pipe: Pipeline, val: pd.DataFrame) -> dict[Any]:
    thresh = params["evaluate"]["threshold"]
    min_roc_auc = params["evaluate"]["min_roc_auc"]

    y_true = val[TARGET]

    y_proba = pipe.predict_proba(val[feature_columns(params)])[:, 1]
    y_pred = (y_proba >= thresh).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }

    if metrics["roc_auc"] < min_roc_auc:
        log.error(f"ROC-AUC {metrics['roc_auc']: .4f} is lower than specified threshold {min_roc_auc: .4f}")
        sys.exit(1)

    return metrics


def main() -> None:
    params = load_params()

    data = load_dataset(params=params)
    log.info("Data loaded")

    pipe = build_pipeline(params=params)
    log.info("Pipeline built")

    log.info("Starting training")
    train_model(params, pipe, data["train"])
    log.info("Model trained")

    metrics = validate_model(params, pipe, data["val"])
    log.info(f"Validation metrics: {metrics}")

    model_dir = resolve(params["train"]["save_dir"])
    model_path = os.path.join(model_dir, "model.joblib")
    meta_path = os.path.join(model_dir, "model_meta.json")
    os.makedirs(model_dir, exist_ok=True)

    meta = {
        "name": params["train"]["model"],
        "git_sha": get_git_sha(),
        "python_version": platform.python_version(),
        "hyperparams": params["train"].get(params["train"]["model"], {}),
        "features": feature_columns(params),
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Model meta saved to: {meta_path}")

    joblib.dump(pipe, model_path)
    log.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
