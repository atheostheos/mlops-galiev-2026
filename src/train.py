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

import argparse
import json
import os
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import TARGET, data_md5, feature_columns, git_sha, load_params, resolve
from src.evaluate import compute_metrics
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
            raise NotImplementedError(f"Unknown model name: {name}")


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(params)),
            ("model", build_model(params)),
        ]
    )


def train_model(params: dict[str, Any], pipe: Pipeline, train: pd.DataFrame) -> None:
    pipe.fit(train[feature_columns(params)], train[TARGET])


def form_model_meta(params: dict[str, Any], metrics: dict[Any]) -> dict[str, Any]:
    return {
        "name": params["train"]["model"],
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "hyperparams": params["train"].get(params["train"]["model"], {}),
        "features": feature_columns(params),
        "metrics": metrics,
        "dvc_hash": data_md5(params["data"]["raw_path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="обучить без логирования в MLflow")
    args = parser.parse_args()

    params = load_params()
    np.random.seed(params["seed"])

    data = load_dataset(params=params)
    log.info("Data loaded")

    pipe = build_pipeline(params=params)
    log.info("Pipeline built")

    log.info("Starting training")
    train_model(params, pipe, data["train"])
    log.info("Model trained")

    metrics, y_proba, _ = compute_metrics(params, pipe, data["val"])
    log.info(f"Validation metrics: {metrics}")

    reports_path = os.path.join(resolve(params["reports"]["dir"]), "train_metrics.json")
    with open(reports_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    log.info(f"Metrics saved to: {reports_path}")

    model_dir = resolve(params["train"]["save_dir"])
    model_path = os.path.join(model_dir, "model.joblib")
    meta_path = os.path.join(model_dir, "model_meta.json")
    os.makedirs(model_dir, exist_ok=True)

    meta = form_model_meta(params, metrics)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    log.info(f"Model meta saved to: {meta_path}")

    joblib.dump(pipe, model_path)
    log.info(f"Model saved to: {model_path}")

    if params["mlflow"]["enabled"] and not args.no_mlflow:
        from src.tracking import log_to_mlflow

        y_true = data["val"][TARGET]
        log_to_mlflow(
            params,
            pipe,
            metrics,
            y_proba,
            y_true,
            input_example=data["train"][feature_columns(params)].head(5),
        )
        log.info("Logged to MLflow")


if __name__ == "__main__":
    main()
